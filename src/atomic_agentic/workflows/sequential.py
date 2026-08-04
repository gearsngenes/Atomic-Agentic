from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional

from ..exceptions import ValidationError
from ..core.Invokable import AtomicInvokable
from ..models.results.workflows import SequentialFlowResult
from .base import Workflow

logger = logging.getLogger(__name__)

__all__ = ["SequentialFlow"]


class SequentialFlow(Workflow):
    """Execute a fixed ordered sequence of workflow-shaped steps.

    Construction contract
    ---------------------
    - ``steps`` must be a non-empty ``list[AtomicInvokable]`` — any
      invokable, ``Workflow`` or not, stored exactly as configured with no
      wrapping.
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

    Notes
    -----
    This class enforces *fixed sequence topology*, but this is still only
    shallow graph immutability. Nested step objects may retain their own broader
    AA mutability elsewhere.
    """

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        steps: list[AtomicInvokable],
        *,
        return_index: Optional[int] = None,
    ) -> None:
        if not isinstance(steps, list):
            raise TypeError(
                f"steps must be a non-empty list[AtomicInvokable], got {type(steps)!r}"
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

        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            parameters=normalized_steps[0].parameters,
            return_type=normalized_steps[resolved_return_index].return_type,
        )

        self._steps: tuple[AtomicInvokable, ...] = normalized_steps
        self._return_index: int = resolved_return_index

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def steps(self) -> tuple[AtomicInvokable, ...]:
        """Return the fixed normalized step tuple."""
        return self._steps

    @property
    def return_index(self) -> int:
        """Fixed step index whose result becomes the outer flow result."""
        return self._return_index

    def _extra_description(self) -> str:
        """Chain into the return-index step's own extra description verbatim.

        Mirrors how `return_type` is sourced at construction — the step
        selected by `return_index` is what produces this flow's result
        payload each run.
        """
        return self._steps[self._return_index]._extra_description()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_step(step: AtomicInvokable) -> AtomicInvokable:
        """Validate one configured step is an AtomicInvokable."""
        if not isinstance(step, AtomicInvokable):
            raise TypeError(
                f"SequentialFlow steps must be AtomicInvokable, got {type(step)!r}"
            )
        return step

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
