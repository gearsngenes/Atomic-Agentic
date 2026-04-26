from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from .StructuredInvokable import StructuredInvokable
from ..core.Exceptions import ValidationError
from ..core.Parameters import ParamSpec, is_valid_parameter_order, to_paramspec_list
from ..core.sentinels import NO_VAL
from .base import FlowResultDict, Workflow
from .basic import BasicFlow
from .metadata import ChildRunRecord, OutputTopology, ParallelFlowRunMetadata

logger = logging.getLogger(__name__)

__all__ = ["ParallelFlow"]

_VALID_OUTPUT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ParallelFlow(Workflow[ParallelFlowRunMetadata]):
    """Execute a fixed set of branches concurrently and project one outer result.

    Input contract
    --------------
    ``ParallelFlow`` accepts one fixed outer input contract determined at
    construction time by ``input_shape`` + ``parameters``.

    - ``input_shape="broadcast"``
        The same outer input mapping is forwarded to every branch.
        ``parameters`` should ideally be provided explicitly. As a convenience
        fallback, if ``parameters`` is omitted, this class mirrors the first
        branch's parameters verbatim. That shortcut is intentionally risky:
        the outer workflow may then filter away keys needed by later branches.

    - ``input_shape="nested"``
        ``parameters`` are required explicitly, must normalize to a fixed
        ``list[ParamSpec]`` with length exactly equal to ``len(branches)``,
        and may not contain ``VAR_POSITIONAL`` or ``VAR_KEYWORD`` entries.
        Parameter ``i`` supplies the nested input payload for branch ``i`` and
        each such payload must itself be a mapping at runtime.

    Output contract
    ---------------
    Branches always execute, but the final outer result is projected through
    the read-only output configuration:

    - ``output_indices``:
        Ordered branch indices whose results are forwarded.
    - ``output_shape="nested"``
        Requires ``output_names`` of matching length. Result shape is:
        ``{output_names[i]: branch_result[output_indices[i]]}``
    - ``output_shape="flattened"``
        Selected branch dicts are flattened in ``output_indices`` order using
        ``duplicate_key_policy``.

    Metadata
    --------
    Per-run metadata contains:

    - ``branch_records``:
        ``tuple[ChildRunRecord, ...]`` containing one typed child execution
        record per executed branch.
    - ``output_topology``:
        ``OutputTopology`` describing the resolved outward projection for
        this run.
    - ``output_count``:
        Number of branch results forwarded into the final outer result.

    Mutability
    ----------
    - Branch topology is fixed after construction.
    - Input contract is fixed after construction.
    - Output projection remains mutable only through :meth:`configure_output`.
    - ``duplicate_key_policy`` is independently mutable.
    """

    # Input/output shape options:
    BROADCAST = "broadcast"
    NESTED = OutputTopology.NESTED
    FLATTENED = OutputTopology.FLATTENED

    # Compatibility alias for older code paths; prefer ``NESTED``.
    ENVELOPED = NESTED

    # Flattened result duplicate key policies:
    RAISE = "raise"
    SKIP = "skip"
    UPDATE = "update"

    def __init__(
        self,
        name: str,
        description: str,
        branches: list[Workflow | StructuredInvokable],
        *,
        input_shape: str = BROADCAST,
        parameters: type | list[str] | tuple[str, ...] | set[str] | list[ParamSpec] | None = None,
        output_shape: str = NESTED,
        output_indices: list[int] | None = None,
        output_range: tuple[int, int] | None = None,
        output_names: list[str] | None = None,
        duplicate_key_policy: str = RAISE,
        filter_extraneous_inputs: Optional[bool] = None,
    ) -> None:
        if not isinstance(branches, list):
            raise TypeError(
                f"branches must be a non-empty list[Workflow | StructuredInvokable], got {type(branches)!r}"
            )
        if not branches:
            raise ValueError("branches must not be empty")

        normalized_branches = tuple(self._normalize_branch(branch) for branch in branches)
        self._branches: tuple[Workflow, ...] = normalized_branches

        normalized_input_shape = str(input_shape).strip().lower()
        if normalized_input_shape not in {self.BROADCAST, self.NESTED}:
            raise ValueError("input_shape must be either 'broadcast' or 'nested'")

        used_parameter_fallback = False
        if parameters is None:
            if normalized_input_shape != self.BROADCAST:
                raise ValueError(
                    "parameters are required when input_shape='nested'"
                )
            declared_parameters = list(normalized_branches[0].parameters)
            used_parameter_fallback = True
        else:
            declared_parameters = to_paramspec_list(parameters)
            is_valid_parameter_order(declared_parameters)

        if normalized_input_shape == self.NESTED:
            if len(declared_parameters) != len(normalized_branches):
                raise ValueError(
                    "nested input_shape requires len(parameters) == len(branches)"
                )
            for spec in declared_parameters:
                if spec.kind in {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}:
                    raise ValueError(
                        "nested input_shape does not permit VAR_POSITIONAL or VAR_KEYWORD parameters"
                    )

        resolved_filter = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else normalized_branches[0].filter_extraneous_inputs
        )

        super().__init__(
            name=name,
            description=description,
            parameters=list(declared_parameters),
            filter_extraneous_inputs=resolved_filter,
        )

        self._input_shape: str = normalized_input_shape
        self._used_parameter_fallback: bool = used_parameter_fallback

        self._output_indices: tuple[int, ...] = tuple()
        self._output_shape: str = self.NESTED
        self._output_names: tuple[str, ...] | None = None

        self.duplicate_key_policy = duplicate_key_policy

        self.configure_output(
            output_indices=output_indices,
            output_range=output_range,
            output_shape=output_shape,
            output_names=output_names,
            duplicate_key_policy=duplicate_key_policy,
        )

    # ------------------------------------------------------------------ #
    # Read-only topology / contract properties
    # ------------------------------------------------------------------ #
    @property
    def branches(self) -> tuple[Workflow, ...]:
        """Return the fixed normalized branch tuple."""
        return self._branches

    @property
    def input_shape(self) -> str:
        """Return the fixed outer input-shape contract."""
        return self._input_shape

    @property
    def output_indices(self) -> list[int]:
        """Return the resolved ordered output branch indices."""
        return list(self._output_indices)

    @property
    def output_shape(self) -> str:
        """Return the current output projection shape."""
        return self._output_shape

    @property
    def output_names(self) -> Optional[list[str]]:
        """Return the current nested output keys, if any."""
        return list(self._output_names) if self._output_names is not None else None

    # ------------------------------------------------------------------ #
    # Independently mutable flattening policy
    # ------------------------------------------------------------------ #
    @property
    def duplicate_key_policy(self) -> str:
        """Flattened duplicate-key behavior."""
        return self._duplicate_key_policy

    @duplicate_key_policy.setter
    def duplicate_key_policy(self, value: str) -> None:
        normalized = str(value).strip().lower()
        if normalized not in {self.RAISE, self.SKIP, self.UPDATE}:
            raise ValueError(
                "duplicate_key_policy must be one of: 'raise', 'skip', 'update'"
            )
        self._duplicate_key_policy = normalized

    # ------------------------------------------------------------------ #
    # Output reconfiguration
    # ------------------------------------------------------------------ #
    def configure_output(
        self,
        *,
        output_indices: list[int] | None = None,
        output_range: tuple[int, int] | None = None,
        output_shape: str,
        output_names: list[str] | None = None,
        duplicate_key_policy: Optional[str] = None,
    ) -> None:
        """Atomically update the output projection contract.

        Notes
        -----
        - Pass either ``output_indices`` or ``output_range``, not both.
        - If neither is supplied, all configured branches are projected in their
        natural order.
        - ``output_shape='nested'`` requires valid ``output_names`` whose
        length matches the resolved output count.
        - ``output_shape='flattened'`` requires ``output_names is None``.
        - ``duplicate_key_policy`` is optional for flattened output; when omitted,
        the current configured policy is reused.
        """
        resolved_indices = self._resolve_output_indices(
            output_indices=output_indices,
            output_range=output_range,
        )

        normalized_output_shape = str(output_shape).strip().lower()
        if normalized_output_shape not in {self.NESTED, self.FLATTENED}:
            raise ValueError("output_shape must be either 'nested' or 'flattened'")

        normalized_names: tuple[str, ...] | None = None

        if normalized_output_shape == self.NESTED:
            if output_names is None:
                raise ValueError(
                    "output_names are required when output_shape='nested'"
                )
            if not isinstance(output_names, list):
                raise TypeError(
                    f"output_names must be a list[str], got {type(output_names)!r}"
                )
            if len(output_names) != len(resolved_indices):
                raise ValueError(
                    "len(output_names) must equal the number of projected outputs"
                )

            cleaned_names: list[str] = []
            for index, name in enumerate(output_names):
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        f"output_names[{index}] must be a non-empty string"
                    )
                cleaned = name.strip()
                if not _VALID_OUTPUT_NAME.match(cleaned):
                    raise ValueError(
                        f"output_names[{index}]={cleaned!r} is not a valid parameter-style name"
                    )
                cleaned_names.append(cleaned)

            if len(cleaned_names) != len(set(cleaned_names)):
                raise ValueError("output_names must be unique")

            normalized_names = tuple(cleaned_names)

        else:
            if output_names is not None:
                raise ValueError(
                    "output_names must be None when output_shape='flattened'"
                )

            if duplicate_key_policy is not None:
                self.duplicate_key_policy = duplicate_key_policy

            if self._duplicate_key_policy not in {self.RAISE, self.SKIP, self.UPDATE}:
                raise ValueError(
                    "duplicate_key_policy must be one of: 'raise', 'skip', 'update'"
                )

        self._output_indices = tuple(resolved_indices)
        self._output_shape = normalized_output_shape
        self._output_names = normalized_names

    # ------------------------------------------------------------------ #
    # Internal normalization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_branch(branch: Workflow | StructuredInvokable) -> Workflow:
        """Normalize one configured branch into a workflow-shaped node."""
        if isinstance(branch, Workflow):
            return branch
        if isinstance(branch, StructuredInvokable):
            return BasicFlow(component=branch)
        raise TypeError(
            "ParallelFlow branches must be Workflow or StructuredInvokable, "
            f"got {type(branch)!r}"
        )

    def _resolve_output_indices(
        self,
        *,
        output_indices: list[int] | None,
        output_range: tuple[int, int] | None,
    ) -> list[int]:
        """Normalize output filtering into one ordered list of branch indices."""
        if output_indices is not None and output_range is not None:
            raise ValueError("Pass either output_indices or output_range, not both")

        branch_count = len(self._branches)

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

    def _build_branch_inputs(
        self,
        inputs: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Build one concrete input mapping per branch for the current run."""
        if self._input_shape == self.BROADCAST:
            shared_inputs = dict(inputs)
            return [dict(shared_inputs) for _ in self._branches]

        branch_inputs: list[dict[str, Any]] = []
        declared_parameters = self.parameters

        for branch_index, spec in enumerate(declared_parameters):
            if spec.name in inputs:
                payload = inputs[spec.name]
            elif spec.default is not NO_VAL:
                payload = spec.default
            else:
                raise ValidationError(
                    f"{self.full_name}: missing nested payload for branch parameter {spec.name!r}"
                )

            if not isinstance(payload, Mapping):
                raise ValidationError(
                    f"{self.full_name}: nested payload for branch {branch_index} "
                    f"({self._branches[branch_index].full_name}) must be a mapping, "
                    f"got {type(payload)!r}"
                )

            branch_inputs.append(dict(payload))

        return branch_inputs

    def _project_result(
        self,
        branch_results: list[FlowResultDict],
    ) -> dict[str, Any]:
        """Project canonical branch results into the configured outer result shape."""
        if self._output_shape == self.NESTED:
            assert self._output_names is not None
            return {
                self._output_names[position]: dict(branch_results[branch_index])
                for position, branch_index in enumerate(self._output_indices)
            }

        flattened: dict[str, Any] = {}
        for branch_index in self._output_indices:
            branch_result = branch_results[branch_index]
            for key, value in branch_result.items():
                if key in flattened:
                    if self._duplicate_key_policy == self.RAISE:
                        raise ValidationError(
                            f"{self.full_name}: duplicate flattened output key {key!r}"
                        )
                    if self._duplicate_key_policy == self.SKIP:
                        continue
                flattened[key] = value

        return flattened

    def _build_output_topology(self) -> OutputTopology:
        """Build the typed outward projection descriptor for one run."""
        if self._output_shape == self.NESTED:
            return OutputTopology(
                topology=OutputTopology.NESTED,
                indices=tuple(self._output_indices),
                names=tuple(self._output_names) if self._output_names is not None else None,
                duplicate_key_policy=None,
            )

        return OutputTopology(
            topology=OutputTopology.FLATTENED,
            indices=tuple(self._output_indices),
            names=None,
            duplicate_key_policy=self._duplicate_key_policy,
        )

    # ------------------------------------------------------------------ #
    # Retrieval helpers
    # ------------------------------------------------------------------ #
    def get_branch_records(
        self,
        run_id: str,
    ) -> Optional[tuple[ChildRunRecord, ...]]:
        """Return stored typed branch execution records for one parent run."""
        checkpoint = self.get_checkpoint(run_id)
        if checkpoint is None:
            return None

        return tuple(checkpoint.metadata.branch_records)

    def get_branch_results(
        self,
        run_id: str,
    ) -> Optional[list[dict[str, Any] | None]]:
        """Return child branch results for one parent run."""
        branch_records = self.get_branch_records(run_id)
        if branch_records is None:
            return None

        branch_results: list[dict[str, Any] | None] = []

        for record_index, record in enumerate(branch_records):
            branch_index = record.slot
            if branch_index >= len(self._branches):
                raise ValidationError(
                    f"{self.full_name}: branch_records[{record_index}] references branch index "
                    f"{branch_index}, but only {len(self._branches)} configured branches exist"
                )

            branch = self._branches[branch_index]
            recorded_instance_id = record.instance_id

            if branch.instance_id != recorded_instance_id:
                raise ValidationError(
                    f"{self.full_name}: branch_records[{record_index}] instance_id mismatch for "
                    f"branch {branch_index}: recorded {recorded_instance_id!r}, current {branch.instance_id!r}"
                )

            child_checkpoint = branch.get_checkpoint(record.run_id)
            if child_checkpoint is None:
                branch_results.append(None)
            else:
                branch_results.append(dict(child_checkpoint.result))

        return branch_results

    def get_branch_result(
        self,
        run_id: str,
        branch_index: int,
    ) -> Optional[dict[str, Any]]:
        """Return one child branch result for one parent run."""
        if not isinstance(branch_index, int):
            raise TypeError(
                f"branch_index must be an int, got {type(branch_index)!r}"
            )

        branch_count = len(self._branches)
        resolved_index = branch_index if branch_index >= 0 else branch_count + branch_index
        if resolved_index < 0 or resolved_index >= branch_count:
            raise IndexError(
                f"branch_index {branch_index} out of range for {branch_count} configured branch(es)"
            )

        branch_results = self.get_branch_results(run_id)
        if branch_results is None:
            return None

        return branch_results[resolved_index]

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[ParallelFlowRunMetadata, Mapping[str, Any]]:
        """Synchronously execute all configured branches concurrently."""
        branch_inputs = self._build_branch_inputs(inputs)

        def run_one(item: tuple[int, Workflow, dict[str, Any]]) -> FlowResultDict:
            index, branch, branch_input = item
            try:
                result = branch.invoke(branch_input)
            except Exception as exc:
                raise RuntimeError(
                    f"{self.full_name}: branch {index} ({branch.full_name}) failed during invoke"
                ) from exc

            if not isinstance(result, FlowResultDict):
                raise ValidationError(
                    f"{self.full_name}: branch {index} ({branch.full_name}) returned "
                    f"{type(result)!r}, expected FlowResultDict"
                )

            return result

        items = [
            (index, branch, branch_inputs[index])
            for index, branch in enumerate(self._branches)
        ]

        with ThreadPoolExecutor(max_workers=len(items)) as executor:
            branch_results = list(executor.map(run_one, items))

        branch_records = tuple(
            ChildRunRecord(
                slot=index,
                instance_id=branch.instance_id,
                full_name=branch.full_name,
                run_id=result.run_id,
            )
            for index, (branch, result) in enumerate(zip(self._branches, branch_results))
        )

        metadata = ParallelFlowRunMetadata(
            branch_records=branch_records,
            output_topology=self._build_output_topology(),
            output_count=len(self._output_indices),
        )
        return metadata, self._project_result(branch_results)

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[ParallelFlowRunMetadata, Mapping[str, Any]]:
        """Asynchronously execute all configured branches concurrently."""
        branch_inputs = self._build_branch_inputs(inputs)

        async def run_one(
            index: int,
            branch: Workflow,
            branch_input: dict[str, Any],
        ) -> FlowResultDict:
            try:
                result = await branch.async_invoke(branch_input)
            except Exception as exc:
                raise RuntimeError(
                    f"{self.full_name}: async branch {index} ({branch.full_name}) failed during invoke"
                ) from exc

            if not isinstance(result, FlowResultDict):
                raise ValidationError(
                    f"{self.full_name}: async branch {index} ({branch.full_name}) returned "
                    f"{type(result)!r}, expected FlowResultDict"
                )

            return result

        branch_results = list(
            await asyncio.gather(
                *[
                    run_one(index, branch, branch_inputs[index])
                    for index, branch in enumerate(self._branches)
                ]
            )
        )

        branch_records = tuple(
            ChildRunRecord(
                slot=index,
                instance_id=branch.instance_id,
                full_name=branch.full_name,
                run_id=result.run_id,
            )
            for index, (branch, result) in enumerate(zip(self._branches, branch_results))
        )

        metadata = ParallelFlowRunMetadata(
            branch_records=branch_records,
            output_topology=self._build_output_topology(),
            output_count=len(self._output_indices),
        )
        return metadata, self._project_result(branch_results)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the fixed branch topology plus current projection policy."""
        data = super().to_dict()
        data.update(
            {
                "branches": [branch.to_dict() for branch in self._branches],
                "branch_count": len(self._branches),
                "input_shape": self._input_shape,
                "parameters_fallback_used": self._used_parameter_fallback,
                "output_shape": self._output_shape,
                "output_indices": list(self._output_indices),
                "output_names": list(self._output_names) if self._output_names is not None else None,
                "duplicate_key_policy": self._duplicate_key_policy,
            }
        )
        return data