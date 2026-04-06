from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from .StructuredInvokable import StructuredInvokable
from .sequential import SequentialFlow
from ..core.Exceptions import ValidationError
from ..core.Invokable import AtomicInvokable
from ..tools import Tool
from ..core.sentinels import NO_VAL
from .base import FlowResultDict, Workflow
from .basic import BasicFlow

logger = logging.getLogger(__name__)


def _always_false() -> bool:
    """Fallback iterative judge implementation."""
    return False


_fallback_judge_tool = Tool(
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
    judge inspects one selected body-step result and decides whether the loop
    should stop early.

    Construction contract
    ---------------------
    - ``body_steps`` must be a non-empty ``list[Workflow | StructuredInvokable]``.
    - The body is always normalized inline into a private ``SequentialFlow``.
    - The normalized loop body topology is fixed at construction.
    - The normalized loop body is exposed read-only via :attr:`loop_body`.
    - ``judge`` accepts any ``AtomicInvokable`` or ``None`` at construction.
    - Internally, the judge is always a normalized ``BasicFlow`` whose wrapped
      component is a fresh ``StructuredInvokable`` exposing a single output key
      named ``"judge_decision"``.
    - Passing ``None`` installs a shared fallback always-false structured judge.
    - ``return_index`` is proxied onto the loop body and controls which body
      step result becomes the body workflow's outer result.
    - ``handoff_index`` controls which body step result becomes the next
      iteration's input.
    - ``evaluate_index`` controls which body step result is passed to the judge.

    Judge contract
    --------------
    At runtime, the normalized judge path must yield a mapping-shaped result
    containing the single key ``"judge_decision"`` whose value must be a
    ``bool``. If it does not, invocation raises ``ValidationError``.

    Loop semantics
    --------------
    - Iteration 0 receives the outer workflow inputs.
    - The loop body executes *all* configured body steps on every iteration.
    - The loop body's ``return_index`` determines the outer body result.
    - ``handoff_index`` determines which body step result becomes the next
      iteration's inputs.
    - ``evaluate_index`` determines which body step result is passed to the judge.
    - The loop stops early only when the judge decision is ``True``.
    - Otherwise it continues until ``max_iterations`` is exhausted.

    Mutability notes
    ----------------
    - The loop body topology is fixed after construction.
    - The judge is fixed after construction.
    - This class does not re-expose any loop-body step mutation API.
    - Only selection policy is mutable post-construction:
      ``return_index``, ``handoff_index``, ``evaluate_index``, and ``max_iterations``.

    Metadata
    --------
    Per-run metadata contains:

    - ``iterations_completed``:
      Number of completed body iterations.
    - ``max_iterations``:
      Maximum number of iterations permitted for this run.
    - ``stop_reason``:
      String reason for loop termination. Currently one of:
        - ``"judge_approved"``
        - ``"max_iterations_exhausted"``
    - ``return_step_index``:
      Resolved absolute body-step index selected for the body return value.
    - ``handoff_step_index``:
      Resolved absolute body-step index selected for next-iteration inputs.
    - ``evaluate_step_index``:
      Resolved absolute body-step index selected for judge evaluation.
    - ``judge_component_instance_id``:
      Instance id of the normalized judge component active for this run.
    - ``iteration_records``:
      ``list[dict[str, Any]]`` where each record contains:
        - ``body_run_id``: loop-body workflow run id for that iteration
        - ``judge_run_id``: judge workflow run id for that iteration
        - ``judge_decision``: boolean decision for that iteration
    """

    def __init__(
        self,
        name: str,
        description: str,
        body_steps: list[Workflow | StructuredInvokable],
        judge: AtomicInvokable | None = None,
        max_iterations: int = 1,
        *,
        return_index: int = -1,
        handoff_index: int = -1,
        evaluate_index: int = -1,
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
            ``max_iterations`` is exhausted unless another failure occurs.
        max_iterations:
            Maximum number of body iterations to execute per run. Must be > 0.
        return_index:
            Loop-body return selection. This is proxied onto ``loop_body``.
        handoff_index:
            Body-step selection whose result becomes the next iteration inputs.
        evaluate_index:
            Body-step selection whose result is passed to the judge.
        filter_extraneous_inputs:
            Optional outer workflow input-filter flag. When omitted, inherits
            from the normalized loop body.
        """
        if not isinstance(body_steps, list):
            raise TypeError(
                f"body_steps must be a list[Workflow | StructuredInvokable], got {type(body_steps)!r}"
            )
        if not body_steps:
            raise ValueError("body_steps must not be empty")

        for index, step in enumerate(body_steps):
            if not isinstance(step, (Workflow, StructuredInvokable)):
                raise TypeError(
                    "body_steps items must be Workflow or StructuredInvokable, "
                    f"got {type(step)!r} at index {index}"
                )

        self._loop_body = SequentialFlow(
            name=f"{name}_loop_body",
            description=f"Normalized body for iterative workflow {name}",
            steps=body_steps,
            return_index=return_index,
        )

        self.handoff_index = handoff_index
        self.evaluate_index = evaluate_index
        self.max_iterations = max_iterations

        resolved_filter = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else self._loop_body.filter_extraneous_inputs
        )

        super().__init__(
            name=name,
            description=description,
            parameters=self._loop_body.parameters,
            filter_extraneous_inputs=resolved_filter,
        )
        resolved_judge = judge if judge is not None else _fallback_judge_tool

        if not isinstance(resolved_judge, AtomicInvokable):
            raise TypeError(
                f"judge must be an AtomicInvokable or None, got {type(resolved_judge)!r}"
            )

        structured_judge = StructuredInvokable(
            component=resolved_judge,
            output_schema=["judge_decision"],
            map_single_fields=True,
            map_extras=True,
            ignore_unhandled=True,
            absent_value_mode=StructuredInvokable.RAISE,
            none_is_absent=True,
            coerce_to_collection=True,
        )

        self._judge = BasicFlow(
            component=structured_judge,
            name=f"{name}_judge",
            description=f"Normalized judge for iterative workflow {name}",
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def loop_body(self) -> SequentialFlow:
        """The fixed normalized sequential loop body."""
        return self._loop_body

    @property
    def judge(self) -> BasicFlow:
        """The fixed normalized judge wrapper."""
        return self._judge

    @property
    def return_index(self) -> int:
        """Proxy onto the loop body's configured return index."""
        return self._loop_body.return_index

    @return_index.setter
    def return_index(self, value: int) -> None:
        """Update the loop body's configured return index."""
        self._loop_body.return_index = value

    @property
    def handoff_index(self) -> int:
        """Configured body-step index whose result becomes next-iteration input."""
        return self._handoff_index

    @handoff_index.setter
    def handoff_index(self, value: int) -> None:
        """Validate and store the configured handoff step index."""
        if not isinstance(value, int):
            raise TypeError(f"handoff_index must be an int, got {type(value)!r}")
        self._loop_body._resolve_step_index(value)
        self._handoff_index = value

    @property
    def evaluate_index(self) -> int:
        """Configured body-step index whose result is passed to the judge."""
        return self._evaluate_index

    @evaluate_index.setter
    def evaluate_index(self, value: int) -> None:
        """Validate and store the configured evaluation step index."""
        if not isinstance(value, int):
            raise TypeError(f"evaluate_index must be an int, got {type(value)!r}")
        self._loop_body._resolve_step_index(value)
        self._evaluate_index = value

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
    # Helpers
    # ------------------------------------------------------------------ #
    def _extract_judge_decision(
        self,
        judge_result: FlowResultDict,
    ) -> bool:
        """Extract and validate the boolean decision from a judge result."""
        decision = judge_result.get("judge_decision", NO_VAL)
        if decision is NO_VAL:
            raise ValidationError(
                f"{self.full_name}: judge result did not contain 'judge_decision'"
            )
        if not isinstance(decision, bool):
            raise ValidationError(
                f"{self.full_name}: judge_decision must be bool, got {type(decision)!r}"
            )

        return decision

    def _require_body_step_result(
        self,
        body_run_id: str,
        step_index: int,
        *,
        purpose: str,
    ) -> dict[str, Any]:
        """Strictly resolve one body step result for live orchestration use.

        Unlike the public retrieval helpers, this raises instead of returning
        ``None`` when the referenced child checkpoint cannot be resolved.
        """
        result = self._loop_body.get_step_result(body_run_id, step_index)
        if result is None:
            raise ValidationError(
                f"{self.full_name}: could not resolve {purpose} body step result "
                f"for body run_id {body_run_id!r} and step index {step_index}"
            )

        if not isinstance(result, Mapping):
            raise ValidationError(
                f"{self.full_name}: {purpose} body step result must be mapping-shaped, "
                f"got {type(result)!r}"
            )

        return dict(result)

    def _build_run_metadata(
        self,
        *,
        iterations_completed: int,
        stop_reason: str,
        return_step: int,
        handoff_step: int,
        evaluate_step: int,
        iteration_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the final outer checkpoint metadata for one iterative run."""
        return {
            "iterations_completed": iterations_completed,
            "max_iterations": self._max_iterations,
            "stop_reason": stop_reason,
            "return_step_index": return_step,
            "handoff_step_index": handoff_step,
            "evaluate_step_index": evaluate_step,
            "judge_component_instance_id": self._judge.component.instance_id,
            "iteration_records": iteration_records,
        }

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Synchronously execute the iterative loop."""
        resolved_return_step = self._loop_body._resolve_step_index(self.return_index)
        resolved_handoff_step = self._loop_body._resolve_step_index(self.handoff_index)
        resolved_evaluate_step = self._loop_body._resolve_step_index(self.evaluate_index)

        current_inputs: Mapping[str, Any] = inputs
        final_result: Mapping[str, Any] = dict(inputs)
        iteration_records: list[dict[str, Any]] = []
        iterations_completed = 0
        stop_reason = "max_iterations_exhausted"

        for iteration in range(self.max_iterations):
            logger.info("%s: iteration %d", self.full_name, iteration)
            body_result = self._loop_body.invoke(current_inputs)

            if not isinstance(body_result, FlowResultDict):
                raise ValidationError(
                    f"{self.full_name}: body returned {type(body_result)!r}, expected FlowResultDict"
                )

            body_run_id = body_result.run_id

            handoff_result = self._require_body_step_result(
                body_run_id,
                self.handoff_index,
                purpose="handoff",
            )
            evaluate_result = self._require_body_step_result(
                body_run_id,
                self.evaluate_index,
                purpose="evaluate",
            )

            logger.info("%s: invoking judge for iteration %d", self.full_name, iteration)
            judge_result = self._judge.invoke(evaluate_result)

            if not isinstance(judge_result, FlowResultDict):
                raise ValidationError(
                    f"{self.full_name}: judge returned {type(judge_result)!r}, expected FlowResultDict"
                )

            judge_decision = self._extract_judge_decision(judge_result)

            iteration_records.append(
                {
                    "iteration": iteration,
                    "body_run_id": body_run_id,
                    "judge_run_id": judge_result.run_id,
                    "judge_decision": judge_decision,
                }
            )
            iterations_completed += 1

            final_result = body_result
            current_inputs = handoff_result

            if judge_decision is True:
                stop_reason = "judge_approved"
                break

        metadata = self._build_run_metadata(
            iterations_completed=iterations_completed,
            stop_reason=stop_reason,
            return_step=resolved_return_step,
            handoff_step=resolved_handoff_step,
            evaluate_step=resolved_evaluate_step,
            iteration_records=iteration_records,
        )
        return metadata, final_result

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Asynchronously execute the iterative loop."""
        configured_return_index = self.return_index
        resolved_return_step = self._loop_body._resolve_step_index(configured_return_index)

        configured_handoff_index = self._handoff_index
        resolved_handoff_step = self._loop_body._resolve_step_index(configured_handoff_index)

        configured_evaluate_index = self._evaluate_index
        resolved_evaluate_step = self._loop_body._resolve_step_index(configured_evaluate_index)

        current_inputs: Mapping[str, Any] = inputs
        final_result: Mapping[str, Any] = dict(inputs)
        iteration_records: list[dict[str, Any]] = []
        iterations_completed = 0
        stop_reason = "max_iterations_exhausted"

        for iteration in range(self._max_iterations):
            logger.info(
                "[Async %s]: invoking body iteration %d",
                self.full_name,
                iteration,
            )
            body_result = await self._loop_body.async_invoke(current_inputs)

            if not isinstance(body_result, FlowResultDict):
                raise ValidationError(
                    f"{self.full_name}: async body returned {type(body_result)!r}, expected FlowResultDict"
                )

            body_run_id = body_result.run_id

            handoff_result = self._require_body_step_result(
                body_run_id,
                configured_handoff_index,
                purpose="handoff",
            )
            evaluate_result = self._require_body_step_result(
                body_run_id,
                configured_evaluate_index,
                purpose="evaluate",
            )

            logger.info(
                "[Async %s]: invoking judge for iteration %d",
                self.full_name,
                iteration,
            )
            judge_result = await self._judge.async_invoke(evaluate_result)

            if not isinstance(judge_result, FlowResultDict):
                raise ValidationError(
                    f"{self.full_name}: async judge returned {type(judge_result)!r}, expected FlowResultDict"
                )

            judge_decision = self._extract_judge_decision(judge_result)

            iteration_records.append(
                {
                    "iteration": iteration,
                    "body_run_id": body_run_id,
                    "judge_run_id": judge_result.run_id,
                    "judge_decision": judge_decision,
                }
            )

            iterations_completed += 1

            final_result = body_result
            current_inputs = handoff_result

            if judge_decision is True:
                stop_reason = "judge_approved"
                break

        metadata = self._build_run_metadata(
            iterations_completed=iterations_completed,
            stop_reason=stop_reason,
            return_step=resolved_return_step,
            handoff_step=resolved_handoff_step,
            evaluate_step=resolved_evaluate_step,
            iteration_records=iteration_records,
        )
        return metadata, final_result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the iterative workflow and its selection policies."""
        data = super().to_dict()
        data.update(
            {
                "loop_body": self.loop_body.to_dict(),
                "judge": self._judge.to_dict(),
                "max_iterations": self._max_iterations,
                "return_index": self.return_index,
                "handoff_index": self._handoff_index,
                "evaluate_index": self._evaluate_index,
            }
        )
        return data
