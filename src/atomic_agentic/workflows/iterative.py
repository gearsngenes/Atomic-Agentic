from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional

from ..core.constants import NO_VAL
from ..core.Exceptions import ValidationError
from ..core.Invokable import AtomicInvokable
from ..results.workflows import IterativeFlowResult, WorkflowResult
from .base import Workflow
from .basic import BasicFlow
from .sequential import SequentialFlow

logger = logging.getLogger(__name__)

__all__ = ["IterativeFlow"]


def _always_false() -> bool:
    """Fallback iterative judge implementation."""
    return False


def create_fallback_judge_tool():
    """Create a shared fallback judge tool that always returns False."""
    from ..tools import Tool
    return Tool(
        function=_always_false,
        name="always_false_judge",
        namespace="workflow",
        description="Fallback iterative judge that always returns False.",
    )


class IterativeFlow(Workflow):
    """Repeat a fixed sequential body up to ``max_iterations``.

    Overview
    --------
    ``IterativeFlow`` is a composite workflow that repeatedly invokes a
    normalized sequential body. After each completed body run, a normalized
    judge inspects one selected body-step result; the loop exits early once
    the judge's unwrapped result matches a configured approval value.

    Construction contract
    ----------------------
    - ``body_steps`` must be a non-empty ``list[Workflow | AtomicInvokable]``.
    - The body is always normalized inline into a private ``SequentialFlow``.
    - The normalized loop body topology is fixed at construction.
    - The normalized loop body is exposed read-only via :attr:`loop_body`.
    - ``judge`` accepts any ``AtomicInvokable`` or ``None``:
        - If ``None``, the shared fallback judge (always returns ``False``)
          is installed, and ``approval_value`` must be left as ``NO_VAL``
          (raises ``ValueError`` if a value is provided).
        - Otherwise, the judge is normalized ``_normalize_branch``-style:
          ``Workflow`` instances are kept as-is, other ``AtomicInvokable``
          instances are wrapped once in ``BasicFlow``. In this case
          ``approval_value`` must be provided (raises ``ValueError`` if it
          is left as ``NO_VAL``).
    - ``approval_value`` is fixed at construction (read-only thereafter).
    - ``return_index``, ``handoff_index``, ``evaluate_index`` each default to
      the last body step when ``None``. Explicit values support normal
      negative indexing and are resolved once at construction; out-of-range
      values raise ``IndexError``. All three are fixed thereafter (read-only).

    Judge contract
    --------------
    At runtime, the normalized judge is invoked with the mapping-shaped
    result of the configured ``evaluate_index`` body step. The loop exits
    early once ``judge_result.result == approval_value``. Any value may be
    used for ``approval_value`` as long as it supports ``==`` against the
    judge's unwrapped result.

    Loop semantics
    --------------
    - Iteration 0 receives the outer workflow inputs.
    - The loop body executes *all* configured body steps on every iteration.
    - The loop body's ``return_index`` determines this run's ``result``
      payload (taken from the most recently completed iteration).
    - ``evaluate_index`` determines which body step result is passed to the
      judge.
    - ``handoff_index`` determines which body step result becomes the next
      iteration's inputs. The handoff result is fetched and validated as
      mapping-shaped on every iteration, including the last, regardless of
      whether it is used.
    - The loop stops early once the judge's result matches ``approval_value``.
      Otherwise it continues until ``max_iterations`` is exhausted.

    Mutability notes
    ----------------
    - The loop body topology is fixed after construction.
    - The judge and ``approval_value`` are fixed after construction.
    - ``return_index``, ``handoff_index``, ``evaluate_index`` are fixed after
      construction.
    - Only ``max_iterations`` remains mutable post-construction.

    Result
    ------
    See ``IterativeFlowResult`` for per-run fields (``iteration_runs``,
    ``judge_runs``, ``return_step_index``, ``handoff_step_index``,
    ``evaluate_step_index``, ``max_iterations``).

    Retrieval helpers
    ------------------
    - ``get_iteration_results(run_id)`` resolves, for each completed
      iteration, the ``(body_result, judge_result)`` pair of result objects.
    """

    def __init__(
        self,
        name: str,
        description: str,
        body_steps: list[Workflow | AtomicInvokable],
        judge: AtomicInvokable | None = None,
        max_iterations: int = 1,
        *,
        return_index: Optional[int] = None,
        handoff_index: Optional[int] = None,
        evaluate_index: Optional[int] = None,
        approval_value: Any = NO_VAL,
        filter_extraneous_inputs: Optional[bool] = None,
    ) -> None:
        """Initialize the iterative workflow.

        Parameters
        ----------
        name:
            Workflow name.
        description:
            Human-readable workflow description.
        body_steps:
            Non-empty list of workflow-shaped or structured steps that will be
            normalized into the private sequential loop body.
        judge:
            Optional judge invokable. Passing ``None`` installs the shared
            fallback always-false judge, causing the loop to continue until
            ``max_iterations`` is exhausted (``approval_value`` must be left
            as ``NO_VAL`` in that case).
        max_iterations:
            Maximum number of body iterations to execute per run. Must be > 0.
        return_index:
            Loop-body return selection, resolved once at construction.
            Defaults to the last body step.
        handoff_index:
            Body-step selection whose result becomes the next iteration's
            inputs, resolved once at construction. Defaults to the last body
            step.
        evaluate_index:
            Body-step selection whose result is passed to the judge, resolved
            once at construction. Defaults to the last body step.
        approval_value:
            Value compared (via ``==``) against the judge's unwrapped result
            to determine early exit. Required (must not be ``NO_VAL``) when
            an explicit ``judge`` is given; must be left as ``NO_VAL`` when
            ``judge`` is ``None``.
        filter_extraneous_inputs:
            Optional outer workflow input-filter flag. When omitted, inherits
            from the normalized loop body.
        """
        if not isinstance(body_steps, list):
            raise TypeError(
                f"body_steps must be a list[Workflow | AtomicInvokable], got {type(body_steps)!r}"
            )
        if not body_steps:
            raise ValueError("body_steps must not be empty")

        for index, step in enumerate(body_steps):
            if not isinstance(step, AtomicInvokable):
                raise TypeError(
                    "body_steps items must be Workflow or AtomicInvokable, "
                    f"got {type(step)!r} at index {index}"
                )

        num_steps = len(body_steps)
        resolved_return_index = self._resolve_body_step_index(return_index, num_steps)
        resolved_handoff_index = self._resolve_body_step_index(handoff_index, num_steps)
        resolved_evaluate_index = self._resolve_body_step_index(evaluate_index, num_steps)

        self._loop_body = SequentialFlow(
            name=f"{name}_loop_body",
            description=f"Normalized body for iterative workflow {name}",
            steps=body_steps,
            return_index=resolved_return_index,
        )
        self._handoff_step_index = resolved_handoff_index
        self._evaluate_step_index = resolved_evaluate_index

        if not isinstance(max_iterations, int):
            raise TypeError(f"max_iterations must be an int, got {type(max_iterations)!r}")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        self._max_iterations = max_iterations

        if judge is None:
            if approval_value is not NO_VAL:
                raise ValueError(
                    f"{name}: approval_value must be NO_VAL when judge is None; "
                    f"got {approval_value!r}"
                )
            resolved_judge: AtomicInvokable = create_fallback_judge_tool()
        else:
            if approval_value is NO_VAL:
                raise ValueError(
                    f"{name}: approval_value must be provided when an explicit judge is given"
                )
            resolved_judge = judge

        self._judge = self._normalize_judge(resolved_judge)
        self._approval_value = approval_value

        resolved_filter = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else self._loop_body.filter_extraneous_inputs
        )

        super().__init__(
            name=name,
            description=description,
            parameters=self._loop_body.parameters,
            return_type=self._loop_body.return_type,
            filter_extraneous_inputs=resolved_filter,
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def loop_body(self) -> SequentialFlow:
        """The fixed normalized sequential loop body."""
        return self._loop_body

    @property
    def judge(self) -> Workflow:
        """The fixed normalized judge."""
        return self._judge

    @property
    def approval_value(self) -> Any:
        """Fixed value compared against the judge's unwrapped result for early exit."""
        return self._approval_value

    @property
    def return_index(self) -> int:
        """Fixed resolved loop-body return step index."""
        return self._loop_body.return_index

    @property
    def handoff_index(self) -> int:
        """Fixed resolved body-step index whose result becomes next-iteration input."""
        return self._handoff_step_index

    @property
    def evaluate_index(self) -> int:
        """Fixed resolved body-step index whose result is passed to the judge."""
        return self._evaluate_step_index

    @property
    def max_iterations(self) -> int:
        """Maximum number of body iterations per run."""
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        """Validate and set the iteration bound."""
        if not isinstance(value, int):
            raise TypeError(f"max_iterations must be an int, got {type(value)!r}")
        if value <= 0:
            raise ValueError("max_iterations must be > 0")
        self._max_iterations = value

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_body_step_index(index: Optional[int], num_steps: int) -> int:
        """Resolve a configured body-step index, defaulting ``None`` to the last step."""
        if index is None:
            return num_steps - 1
        if not isinstance(index, int):
            raise TypeError(f"step index must be an int or None, got {type(index)!r}")
        resolved = index if index >= 0 else num_steps + index
        if not (0 <= resolved < num_steps):
            raise IndexError(
                f"step index {index} out of range for {num_steps} configured step(s)"
            )
        return resolved

    @staticmethod
    def _normalize_judge(judge: AtomicInvokable) -> Workflow:
        """Normalize the configured judge into a workflow-shaped node."""
        if isinstance(judge, Workflow):
            return judge
        if isinstance(judge, AtomicInvokable):
            return BasicFlow(component=judge)
        raise TypeError(
            f"IterativeFlow judge must be Workflow or AtomicInvokable, got {type(judge)!r}"
        )

    # ------------------------------------------------------------------ #
    # Run-oriented retrieval
    # ------------------------------------------------------------------ #
    def get_iteration_results(
        self, run_id: str
    ) -> Optional[list[tuple[Optional[WorkflowResult], Optional[WorkflowResult]]]]:
        """Return ``(body_result, judge_result)`` pairs for one iterative run.

        Returns
        -------
        Optional[list[tuple[Optional[WorkflowResult], Optional[WorkflowResult]]]]
            - ``None`` if the parent iterative run is not found
            - otherwise, one ``(body_result, judge_result)`` tuple per
              completed iteration, in iteration order
            - either element of a tuple is ``None`` if the corresponding
              child run id no longer resolves to a retained checkpoint
        """
        checkpoint = self.get_checkpoint(run_id)
        if checkpoint is None:
            return None

        pairs: list[tuple[Optional[WorkflowResult], Optional[WorkflowResult]]] = []
        for body_run_id, judge_run_id in zip(
            checkpoint.result.iteration_runs, checkpoint.result.judge_runs
        ):
            body_checkpoint = self._loop_body.get_checkpoint(body_run_id)
            judge_checkpoint = self._judge.get_checkpoint(judge_run_id)
            pairs.append((
                body_checkpoint.result if body_checkpoint else None,
                judge_checkpoint.result if judge_checkpoint else None,
            ))

        return pairs

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #
    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> IterativeFlowResult:
        """Construct an IterativeFlowResult envelope for this workflow's invocation."""
        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=IterativeFlowResult,
            **result_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Synchronously execute the iterative loop."""
        current_inputs: Mapping[str, Any] = inputs
        final_payload: Any = None
        iteration_runs: list[str] = []
        judge_runs: list[str] = []

        for iteration in range(self._max_iterations):
            logger.info("%s: iteration %d", self.full_name, iteration)
            body_result = self._loop_body.invoke(current_inputs)
            iteration_runs.append(body_result.run_id)
            final_payload = body_result.result

            evaluate_payload = self._loop_body.get_step_result(
                body_result.run_id, self._evaluate_step_index
            ).result
            if not isinstance(evaluate_payload, Mapping):
                raise ValidationError(
                    f"{self.full_name}: evaluate step {self._evaluate_step_index} result "
                    f"must be mapping-shaped, got {type(evaluate_payload)!r}"
                )

            logger.info("%s: invoking judge for iteration %d", self.full_name, iteration)
            judge_result = self._judge.invoke(evaluate_payload)
            judge_runs.append(judge_result.run_id)

            handoff_payload = self._loop_body.get_step_result(
                body_result.run_id, self._handoff_step_index
            ).result
            if not isinstance(handoff_payload, Mapping):
                raise ValidationError(
                    f"{self.full_name}: handoff step {self._handoff_step_index} result "
                    f"must be mapping-shaped, got {type(handoff_payload)!r}"
                )

            if judge_result.result == self._approval_value:
                break

            current_inputs = handoff_payload

        return final_payload, {
            "iteration_runs": tuple(iteration_runs),
            "judge_runs": tuple(judge_runs),
            "return_step_index": self._loop_body.return_index,
            "handoff_step_index": self._handoff_step_index,
            "evaluate_step_index": self._evaluate_step_index,
            "max_iterations": self._max_iterations,
        }

    async def _async_run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Asynchronously execute the iterative loop."""
        current_inputs: Mapping[str, Any] = inputs
        final_payload: Any = None
        iteration_runs: list[str] = []
        judge_runs: list[str] = []

        for iteration in range(self._max_iterations):
            logger.info("[Async %s]: iteration %d", self.full_name, iteration)
            body_result = await self._loop_body.async_invoke(current_inputs)
            iteration_runs.append(body_result.run_id)
            final_payload = body_result.result

            evaluate_payload = self._loop_body.get_step_result(
                body_result.run_id, self._evaluate_step_index
            ).result
            if not isinstance(evaluate_payload, Mapping):
                raise ValidationError(
                    f"{self.full_name}: evaluate step {self._evaluate_step_index} result "
                    f"must be mapping-shaped, got {type(evaluate_payload)!r}"
                )

            logger.info("[Async %s]: invoking judge for iteration %d", self.full_name, iteration)
            judge_result = await self._judge.async_invoke(evaluate_payload)
            judge_runs.append(judge_result.run_id)

            handoff_payload = self._loop_body.get_step_result(
                body_result.run_id, self._handoff_step_index
            ).result
            if not isinstance(handoff_payload, Mapping):
                raise ValidationError(
                    f"{self.full_name}: handoff step {self._handoff_step_index} result "
                    f"must be mapping-shaped, got {type(handoff_payload)!r}"
                )

            if judge_result.result == self._approval_value:
                break

            current_inputs = handoff_payload

        return final_payload, {
            "iteration_runs": tuple(iteration_runs),
            "judge_runs": tuple(judge_runs),
            "return_step_index": self._loop_body.return_index,
            "handoff_step_index": self._handoff_step_index,
            "evaluate_step_index": self._evaluate_step_index,
            "max_iterations": self._max_iterations,
        }

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the iterative workflow and its fixed selection policy."""
        data = super().to_dict()
        data.update(
            {
                "loop_body": self.loop_body.to_dict(),
                "judge": self._judge.to_dict(),
                "max_iterations": self._max_iterations,
                "return_index": self.return_index,
                "handoff_index": self._handoff_step_index,
                "evaluate_index": self._evaluate_step_index,
                "approval_value": self._approval_value,
            }
        )
        return data
