from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional

from ..core.Exceptions import ValidationError
from ..core.Invokable import AtomicInvokable
from ..models.results.workflows import SequentialFlowResult, WorkflowResult
from .base import Workflow
from .basic import BasicFlow

logger = logging.getLogger(__name__)

__all__ = ["SequentialFlow"]


class SequentialFlow(Workflow):
    """Execute a fixed ordered sequence of workflow-shaped steps.

    Step normalization
    ------------------
    - Existing ``Workflow`` instances are kept as-is.
    - Non-workflow ``AtomicInvokable`` instances are wrapped once in ``BasicFlow``.

    Construction contract
    ---------------------
    - ``steps`` must be a non-empty ``list[Workflow | AtomicInvokable]``.
    - The topology is fixed at construction.
    - The configured step instances are fixed at construction.
    - No post-construction step mutation API is provided.
    - ``return_index`` selects which executed step result becomes the outer
      workflow result. It is resolved once at construction and is fixed
      thereafter:
        - ``None`` (default) resolves to the last step.
        - An explicit ``int`` must satisfy ``0 <= return_index < len(steps)``;
          out-of-range or negative values raise ``IndexError``. There is no
          negative-index wraparound.

    Runtime contract
    ----------------
    - Inputs are passed to the first step.
    - Each step's result is invoked with the previous step's result payload.
    - All configured steps execute on every run.
    - Every step except the last must produce a mapping-shaped result
      (``result.result``), since it is passed as the next step's inputs.
    - The step selected by ``return_index`` determines the final result
      payload returned to the workflow base.

    Result
    ------
    Per-run result fields (see ``SequentialFlowResult``):

    - ``step_run_ids``:
        ``tuple[str, ...]`` containing one child run id per executed step, in
        step order.
    - ``return_index``:
        The fixed step index whose result became the outer result payload.

    Retrieval helpers
    -----------------
    - ``get_step_results(run_id)`` resolves each step's result object (an
      ``AtomicResult``-family instance) for a parent sequential run, returning
      ``list[Any | None]``, or ``None`` if the parent run is not found.
    - ``get_step_result(run_id, step_index)`` is a convenience wrapper over
      ``get_step_results(run_id)``.

    Notes
    -----
    This class enforces *fixed sequence topology*, but this is still only
    shallow graph immutability. Nested step objects may retain their own broader
    AA mutability elsewhere.
    """

    def __init__(
        self,
        name: str,
        description: str,
        steps: list[Workflow | AtomicInvokable],
        *,
        return_index: Optional[int] = None,
        filter_extraneous_inputs: Optional[bool] = None,
    ) -> None:
        if not isinstance(steps, list):
            raise TypeError(
                f"steps must be a non-empty list[Workflow | AtomicInvokable], got {type(steps)!r}"
            )
        if not steps:
            raise ValueError("steps must not be empty")

        normalized_steps = tuple(self._normalize_step(step) for step in steps)

        if return_index is None:
            resolved_return_index = len(normalized_steps) - 1
        elif isinstance(return_index, int):
            if not (0 <= return_index < len(normalized_steps)):
                raise IndexError(
                    f"return_index {return_index} out of range for "
                    f"{len(normalized_steps)} configured step(s)"
                )
            resolved_return_index = return_index
        else:
            raise TypeError(
                f"return_index must be an int or None, got {type(return_index)!r}"
            )

        resolved_filter = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else True
        )

        super().__init__(
            name=name,
            description=description,
            parameters=normalized_steps[0].parameters,
            return_type=normalized_steps[resolved_return_index].return_type,
            filter_extraneous_inputs=resolved_filter,
        )

        self._steps: tuple[Workflow, ...] = normalized_steps
        self._return_index: int = resolved_return_index

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def steps(self) -> tuple[Workflow, ...]:
        """Return the fixed normalized step tuple."""
        return self._steps

    @property
    def return_index(self) -> int:
        """Fixed step index whose result becomes the outer flow result."""
        return self._return_index

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_step(step: Workflow | AtomicInvokable) -> Workflow:
        """Normalize one configured step into a workflow-shaped step."""
        if isinstance(step, Workflow):
            return step
        if isinstance(step, AtomicInvokable):
            return BasicFlow(component=step)
        raise TypeError(
            "SequentialFlow steps must be Workflow or AtomicInvokable, "
            f"got {type(step)!r}"
        )

    def _resolve_step_index(self, index: int) -> int:
        """Resolve a configured step index using normal Python negative indexing."""
        if not isinstance(index, int):
            raise TypeError(f"step index must be an int, got {type(index)!r}")

        length = len(self._steps)
        resolved = index if index >= 0 else length + index
        if resolved < 0 or resolved >= length:
            raise IndexError(
                f"step index {index} out of range for {length} configured step(s)"
            )
        return resolved

    # ------------------------------------------------------------------ #
    # Run-oriented retrieval
    # ------------------------------------------------------------------ #
    def get_step_results(self, run_id: str) -> Optional[list[WorkflowResult]]:
        """Return all child step result objects for one sequential run.

        Returns
        -------
        Optional[list[WorkflowResult]]
            - ``None`` if the parent sequential run is not found
            - otherwise, one entry per configured step, in order
            - each entry is the step's result object (an ``AtomicResult``-family
              instance, ``child_checkpoint.result``), or ``None`` if that
              step's run id no longer resolves to a retained child checkpoint
        """
        checkpoint = self.get_checkpoint(run_id)
        if checkpoint is None:
            return None

        results: list[Any] = []
        for step, step_run_id in zip(self._steps, checkpoint.result.step_runs):
            child_checkpoint = step.get_checkpoint(step_run_id)
            results.append(child_checkpoint.result if child_checkpoint else None)

        return results

    def get_step_result(self, run_id: str, step_index: int) -> Optional[WorkflowResult]:
        """Return one child step's result object for one sequential run.

        This method uses :meth:`get_step_results` as its source of truth.

        Returns
        -------
        Optional[Any]
            - ``None`` if the parent sequential run is not found
            - the resolved child step's result object (an ``AtomicResult``-family
              instance)
            - ``None`` if the child checkpoint for that recorded step run id is
              no longer available

        Raises
        ------
        TypeError
            If ``step_index`` is not an int.
        IndexError
            If ``step_index`` is out of range for the configured steps.
        """
        resolved_index = self._resolve_step_index(step_index)

        results = self.get_step_results(run_id)
        if results is None:
            return None

        return results[resolved_index]

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #
    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> SequentialFlowResult:
        """Construct a SequentialFlowResult envelope for this workflow's invocation."""
        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=SequentialFlowResult,
            **result_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Synchronously execute all configured steps and return the selected step result."""
        running_inputs: Mapping[str, Any] = inputs
        step_results = []

        for index, step in enumerate(self._steps):
            logger.info("%s: invoking step %d (%s)", self.full_name, index, step.full_name)
            result = step.invoke(running_inputs)
            step_results.append(result)

            if index < len(self._steps) - 1:
                if not isinstance(result.result, Mapping):
                    raise ValidationError(
                        f"{self.full_name}: step {index} ({step.full_name}) returned "
                        f"non-mapping result ({type(result.result)!r}); only the "
                        f"final step's result may be a non-mapping type"
                    )
                running_inputs = result.result

        return step_results[self._return_index].result, {
            "step_runs": tuple(r.run_id for r in step_results),
            "return_index": self._return_index,
        }

    async def _async_run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Asynchronously execute all configured steps and return the selected step result."""
        running_inputs: Mapping[str, Any] = inputs
        step_results = []

        for index, step in enumerate(self._steps):
            logger.info(
                "[Async %s]: invoking step %d (%s)",
                self.full_name,
                index,
                step.full_name,
            )
            result = await step.async_invoke(running_inputs)
            step_results.append(result)

            if index < len(self._steps) - 1:
                if not isinstance(result.result, Mapping):
                    raise ValidationError(
                        f"{self.full_name}: async step {index} ({step.full_name}) returned "
                        f"non-mapping result ({type(result.result)!r}); only the "
                        f"final step's result may be a non-mapping type"
                    )
                running_inputs = result.result

        return step_results[self._return_index].result, {
            "step_runs": tuple(r.run_id for r in step_results),
            "return_index": self._return_index,
        }

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the fixed sequence and its configured selection policy."""
        data = super().to_dict()
        data.update(
            {
                "steps": [step.to_dict() for step in self._steps],
                "step_count": len(self._steps),
                "return_index": self._return_index,
            }
        )
        return data
