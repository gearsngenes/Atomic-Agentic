from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Optional

from ..core.Invokable import AtomicInvokable
from ..models.parameters import ParamSpec
from ..utils.parameters import _validate_parameter_order, to_paramspec_list
from ..core.constants import IDENTIFIER_PATTERN
from ..models.results.workflows import ParallelFlowResult, WorkflowResult
from .base import Workflow
from .basic import BasicFlow

logger = logging.getLogger(__name__)

__all__ = ["ParallelFlow"]


class ParallelFlow(Workflow):
    """Execute a fixed set of branches concurrently and project one outer result.

    Input contract
    --------------
    Inputs are broadcast: the same outer input mapping is forwarded to every
    branch unchanged. ``parameters`` should ideally be provided explicitly; if
    omitted, this class mirrors the first branch's parameters verbatim.

    Output contract
    ----------------
    All branches always execute. The final outer result is projected from the
    branches selected by ``output_indices``/``output_range`` (default: all
    branches, in order), shaped according to ``output_type``:

    - ``SCALAR``: exactly one output index; result is that branch's payload.
    - ``LIST`` / ``TUPLE`` / ``SET``: result is the selected payloads in a
      ``list`` / ``tuple`` / ``set``.
    - ``DICT``: requires ``output_names`` of matching length; result is
      ``dict(zip(output_names, payloads))``.

    Mutability
    ----------
    Branch topology and output projection are both fixed at construction.
    """

    SCALAR = "scalar"
    LIST = "list"
    TUPLE = "tuple"
    SET = "set"
    DICT = "dict"

    _OUTPUT_TYPES = {SCALAR, LIST, TUPLE, SET, DICT}

    def __init__(
        self,
        name: str,
        description: str,
        branches: list[Workflow | AtomicInvokable],
        *,
        parameters: type | list[str] | tuple[str, ...] | set[str] | list[ParamSpec] | None = None,
        output_type: str = LIST,
        output_indices: list[int] | None = None,
        output_range: tuple[int, int] | None = None,
        output_names: list[str] | None = None,
        filter_extraneous_inputs: Optional[bool] = None,
    ) -> None:
        if not isinstance(branches, list):
            raise TypeError(
                f"branches must be a non-empty list[Workflow | AtomicInvokable], got {type(branches)!r}"
            )
        if not branches:
            raise ValueError("branches must not be empty")

        normalized_branches = tuple(self._normalize_branch(branch) for branch in branches)

        if parameters is None:
            declared_parameters = list(normalized_branches[0].parameters)
        else:
            declared_parameters = to_paramspec_list(parameters)
            _validate_parameter_order(declared_parameters)

        resolved_filter = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else True
        )

        resolved_output_type, resolved_indices, resolved_names, resolved_return_type = self._configure_output(
            branch_count=len(normalized_branches),
            branches=normalized_branches,
            output_type=output_type,
            output_indices=output_indices,
            output_range=output_range,
            output_names=output_names,
        )

        super().__init__(
            name=name,
            description=description,
            parameters=list(declared_parameters),
            return_type=resolved_return_type,
            filter_extraneous_inputs=resolved_filter,
        )

        self._branches: tuple[Workflow, ...] = normalized_branches
        self._output_type: str = resolved_output_type
        self._output_indices: tuple[int, ...] = resolved_indices
        self._output_names: Optional[tuple[str, ...]] = resolved_names

    # ------------------------------------------------------------------ #
    # Read-only topology / contract properties
    # ------------------------------------------------------------------ #
    @property
    def branches(self) -> tuple[Workflow, ...]:
        """Return the fixed normalized branch tuple."""
        return self._branches

    @property
    def output_type(self) -> str:
        """Return the fixed output projection type."""
        return self._output_type

    @property
    def output_indices(self) -> list[int]:
        """Return the fixed ordered output branch indices."""
        return list(self._output_indices)

    @property
    def output_names(self) -> Optional[list[str]]:
        """Return the fixed dict output keys, if ``output_type='dict'``."""
        return list(self._output_names) if self._output_names is not None else None

    # ------------------------------------------------------------------ #
    # Internal normalization / configuration helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_branch(branch: Workflow | AtomicInvokable) -> Workflow:
        """Normalize one configured branch into a workflow-shaped node."""
        if isinstance(branch, Workflow):
            return branch
        if isinstance(branch, AtomicInvokable):
            return BasicFlow(component=branch)
        raise TypeError(
            "ParallelFlow branches must be Workflow or AtomicInvokable, "
            f"got {type(branch)!r}"
        )

    @staticmethod
    def _resolve_output_indices(
        *,
        branch_count: int,
        output_indices: list[int] | None,
        output_range: tuple[int, int] | None,
    ) -> list[int]:
        """Normalize output filtering into one ordered list of branch indices."""
        if output_indices is not None and output_range is not None:
            raise ValueError("Pass either output_indices or output_range, not both")

        if output_indices is None and output_range is None:
            return list(range(branch_count))

        if output_indices is not None:
            if not isinstance(output_indices, list):
                raise TypeError(
                    f"output_indices must be a list[int], got {type(output_indices)!r}"
                )

            resolved: list[int] = []
            for raw_index in output_indices:
                if not isinstance(raw_index, int):
                    raise TypeError(
                        f"output_indices items must be int, got {type(raw_index)!r}"
                    )

                resolved_index = raw_index if raw_index >= 0 else branch_count + raw_index
                if resolved_index < 0 or resolved_index >= branch_count:
                    raise IndexError(
                        f"output index {raw_index} out of range for {branch_count} configured branch(es)"
                    )
                resolved.append(resolved_index)

            if len(resolved) != len(set(resolved)):
                raise ValueError("output_indices must not contain duplicates")

            return resolved

        if (
            not isinstance(output_range, tuple)
            or len(output_range) != 2
            or not all(isinstance(item, int) for item in output_range)
        ):
            raise TypeError("output_range must be a 2-int tuple")

        start, end = output_range
        resolved = list(range(branch_count))[start:end]

        if len(resolved) != len(set(resolved)):
            raise ValueError("resolved output indices must not contain duplicates")

        return resolved

    @classmethod
    def _configure_output(
        cls,
        *,
        branch_count: int,
        branches: tuple[Workflow, ...],
        output_type: str,
        output_indices: list[int] | None,
        output_range: tuple[int, int] | None,
        output_names: list[str] | None,
    ) -> tuple[str, tuple[int, ...], Optional[tuple[str, ...]], str]:
        """Resolve and validate the fixed output projection for one construction."""
        normalized_output_type = str(output_type).strip().lower()
        if normalized_output_type not in cls._OUTPUT_TYPES:
            raise ValueError(
                f"output_type must be one of {sorted(cls._OUTPUT_TYPES)!r}, got {output_type!r}"
            )

        resolved_indices = cls._resolve_output_indices(
            branch_count=branch_count,
            output_indices=output_indices,
            output_range=output_range,
        )

        if normalized_output_type == cls.SCALAR:
            if len(resolved_indices) != 1:
                raise ValueError("output_type='scalar' requires exactly one output index")
            if output_names is not None:
                raise ValueError("output_names must be None when output_type='scalar'")
            resolved_return_type = branches[resolved_indices[0]].return_type
            return normalized_output_type, tuple(resolved_indices), None, resolved_return_type

        if normalized_output_type == cls.DICT:
            if output_names is None:
                raise ValueError("output_names are required when output_type='dict'")
            if not isinstance(output_names, list):
                raise TypeError(
                    f"output_names must be a list[str], got {type(output_names)!r}"
                )
            if len(output_names) != len(resolved_indices):
                raise ValueError("len(output_names) must equal the number of projected outputs")

            cleaned_names: list[str] = []
            for index, raw_name in enumerate(output_names):
                if not isinstance(raw_name, str) or not raw_name.strip():
                    raise ValueError(f"output_names[{index}] must be a non-empty string")
                cleaned = raw_name.strip()
                if not IDENTIFIER_PATTERN.match(cleaned):
                    raise ValueError(
                        f"output_names[{index}]={cleaned!r} is not a valid parameter-style name"
                    )
                cleaned_names.append(cleaned)

            if len(cleaned_names) != len(set(cleaned_names)):
                raise ValueError("output_names must be unique")

            return normalized_output_type, tuple(resolved_indices), tuple(cleaned_names), "dict"

        # LIST / TUPLE / SET
        if output_names is not None:
            raise ValueError(
                f"output_names must be None when output_type={normalized_output_type!r}"
            )
        return normalized_output_type, tuple(resolved_indices), None, normalized_output_type

    def _project_result(self, branch_results: list[WorkflowResult]) -> Any:
        """Project executed branch results into the configured outer result."""
        payloads = [branch_results[index].result for index in self._output_indices]

        if self._output_type == self.SCALAR:
            return payloads[0]
        if self._output_type == self.LIST:
            return list(payloads)
        if self._output_type == self.TUPLE:
            return tuple(payloads)
        if self._output_type == self.SET:
            return set(payloads)

        assert self._output_names is not None
        return dict(zip(self._output_names, payloads))

    # ------------------------------------------------------------------ #
    # Run-oriented retrieval
    # ------------------------------------------------------------------ #
    def get_branch_results(self, run_id: str) -> Optional[list[Any]]:
        """Return all child branch result objects for one parallel run.

        Returns
        -------
        Optional[list[Any]]
            - ``None`` if the parent parallel run is not found
            - otherwise, one entry per configured branch, in order
            - each entry is the branch's result object (an ``AtomicResult``-family
              instance, ``child_checkpoint.result``), or ``None`` if that
              branch's run id no longer resolves to a retained child checkpoint
        """
        checkpoint = self.get_checkpoint(run_id)
        if checkpoint is None:
            return None

        results: list[Any] = []
        for branch, branch_run_id in zip(self._branches, checkpoint.result.branch_runs):
            child_checkpoint = branch.get_checkpoint(branch_run_id)
            results.append(child_checkpoint.result if child_checkpoint else None)

        return results

    def get_branch_result(self, run_id: str, branch_index: int) -> Optional[Any]:
        """Return one child branch's result object for one parallel run.

        Raises
        ------
        TypeError
            If ``branch_index`` is not an int.
        IndexError
            If ``branch_index`` is out of range for the configured branches.
            There is no negative-index wraparound.
        """
        if not isinstance(branch_index, int):
            raise TypeError(f"branch_index must be an int, got {type(branch_index)!r}")

        if not (0 <= branch_index < len(self._branches)):
            raise IndexError(
                f"branch_index {branch_index} out of range for {len(self._branches)} configured branch(es)"
            )

        results = self.get_branch_results(run_id)
        if results is None:
            return None

        return results[branch_index]

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #
    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> ParallelFlowResult:
        """Construct a ParallelFlowResult envelope for this workflow's invocation."""
        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=ParallelFlowResult,
            **result_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Synchronously execute all configured branches concurrently."""
        branch_inputs = [dict(inputs) for _ in self._branches]

        def run_one(item: tuple[int, Workflow, dict[str, Any]]) -> WorkflowResult:
            index, branch, branch_input = item
            try:
                return branch.invoke(branch_input)
            except Exception as exc:
                raise RuntimeError(
                    f"{self.full_name}: branch {index} ({branch.full_name}) failed during invoke"
                ) from exc

        items = [
            (index, branch, branch_inputs[index])
            for index, branch in enumerate(self._branches)
        ]

        with ThreadPoolExecutor(max_workers=len(items)) as executor:
            branch_results = list(executor.map(run_one, items))

        return self._project_result(branch_results), {
            "branch_runs": tuple(r.run_id for r in branch_results),
            "output_indices": self._output_indices,
        }

    async def _async_run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Asynchronously execute all configured branches concurrently."""
        branch_inputs = [dict(inputs) for _ in self._branches]

        async def run_one(index: int, branch: Workflow, branch_input: dict[str, Any]) -> WorkflowResult:
            try:
                return await branch.async_invoke(branch_input)
            except Exception as exc:
                raise RuntimeError(
                    f"{self.full_name}: async branch {index} ({branch.full_name}) failed during invoke"
                ) from exc

        branch_results = list(
            await asyncio.gather(
                *[
                    run_one(index, branch, branch_inputs[index])
                    for index, branch in enumerate(self._branches)
                ]
            )
        )

        return self._project_result(branch_results), {
            "branch_runs": tuple(r.run_id for r in branch_results),
            "output_indices": self._output_indices,
        }

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the fixed branch topology plus output projection."""
        data = super().to_dict()
        data.update(
            {
                "branches": [branch.to_dict() for branch in self._branches],
                "branch_count": len(self._branches),
                "output_type": self._output_type,
                "output_indices": list(self._output_indices),
                "output_names": list(self._output_names) if self._output_names is not None else None,
            }
        )
        return data
