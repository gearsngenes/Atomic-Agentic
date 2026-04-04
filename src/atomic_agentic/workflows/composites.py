from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from .StructuredInvokable import StructuredInvokable
from ..core.Exceptions import ValidationError
from ..core.Invokable import AtomicInvokable
from ..core.sentinels import NO_VAL
from ..tools import Tool
from .base import FlowResultDict, Workflow
from .basic import BasicFlow

logger = logging.getLogger(__name__)

__all__ = ["SequentialFlow", "IterativeFlow"]


def _always_false() -> bool:
    """Fallback iterative judge implementation."""
    return False


_fallback_judge_tool = Tool(
    function=_always_false,
    name="always_false_judge",
    namespace="workflow",
    description="Fallback iterative judge that always returns False.",
)

_fallback_judge_component = StructuredInvokable(
    component=_fallback_judge_tool,
    name="fallback_judge_component",
    description="Fallback iterative judge component that always returns False.",
    output_schema=[],
)

class SequentialFlow(Workflow):
    """Execute a fixed ordered sequence of workflow-shaped steps.

    Step normalization
    ------------------
    - Existing ``Workflow`` instances are kept as-is.
    - ``StructuredInvokable`` instances are wrapped once in ``BasicFlow``.

    Construction contract
    ---------------------
    - ``steps`` must be a non-empty ``list[Workflow | StructuredInvokable]``.
    - The topology is fixed at construction.
    - The configured step instances are fixed at construction.
    - No post-construction step mutation API is provided.
    - ``return_index`` selects which executed step result becomes the outer
      workflow result. This is selection-only mutability; it does not alter
      topology or execution order.

    Runtime contract
    ----------------
    - Inputs are passed to the first step.
    - Each step's mapping result is passed directly to the next step.
    - All configured steps execute on every run.
    - The step selected by ``return_index`` determines the final step result
      returned to the workflow base, which then wraps it in the outer
      ``FlowResultDict`` and records the sequential checkpoint.

    Metadata
    --------
    Per-run metadata contains:

    - ``step_records``: list[dict[str, Any]]
        One record per executed step, each containing:
        - ``step``: zero-based step index for that run
        - ``instance_id``: executed step instance id
        - ``full_name``: executed step full name
        - ``run_id``: that step's workflow run id
    - ``return_index``:
        The configured return index for that run.
    - ``resolved_return_step``:
        The resolved absolute step index selected for return.
    - ``returned_step_run_id``:
        The child step run id whose result became the outer sequential result.

    Retrieval helpers
    -----------------
    - ``get_step_records(run_id)`` returns the stored step records for a parent
      sequential run, or ``None`` if the parent run is not found.
    - ``get_step_results(run_id)`` resolves those records back into child step
      checkpoint results and returns ``list[result | None]``. ``None`` is used
      when the child run id no longer resolves to a retained child checkpoint.
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
        steps: list[Workflow | StructuredInvokable],
        *,
        return_index: int = -1,
        filter_extraneous_inputs: Optional[bool] = None,
    ) -> None:
        if not isinstance(steps, list):
            raise TypeError(
                f"steps must be a non-empty list[Workflow | StructuredInvokable], got {type(steps)!r}"
            )
        if not steps:
            raise ValueError("steps must not be empty")

        normalized_steps = tuple(self._normalize_step(step) for step in steps)

        resolved_filter = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else normalized_steps[0].filter_extraneous_inputs
        )

        super().__init__(
            name=name,
            description=description,
            parameters=normalized_steps[0].parameters,
            filter_extraneous_inputs=resolved_filter,
        )

        self._steps: tuple[Workflow, ...] = normalized_steps
        self.return_index = return_index

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def steps(self) -> tuple[Workflow, ...]:
        """Return the fixed normalized step tuple."""
        return self._steps

    @property
    def return_index(self) -> int:
        """Configured step index whose result becomes the outer flow result."""
        return self._return_index

    @return_index.setter
    def return_index(self, value: int) -> None:
        """Validate and store the configured return step index."""
        if not isinstance(value, int):
            raise TypeError(f"return_index must be an int, got {type(value)!r}")
        self._resolve_step_index(value)
        self._return_index = value

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_step(step: Workflow | StructuredInvokable) -> Workflow:
        """Normalize one configured step into a workflow-shaped step."""
        if isinstance(step, Workflow):
            return step
        if isinstance(step, StructuredInvokable):
            return BasicFlow(component=step)
        raise TypeError(
            "SequentialFlow steps must be Workflow or StructuredInvokable, "
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
    def get_step_records(self, run_id: str) -> Optional[list[dict[str, Any]]]:
        """Return the stored step records for one sequential run.

        Returns ``None`` when the parent sequential checkpoint is not found.

        The returned list is a shallow-copied snapshot of the stored metadata
        records. Each record is validated to contain the current required fields.
        """
        checkpoint = self.get_checkpoint(run_id)
        if checkpoint is None:
            return None

        raw_records = checkpoint.metadata.get("step_records")
        if not isinstance(raw_records, list):
            raise ValidationError(
                f"{self.full_name}: checkpoint metadata missing valid 'step_records' list "
                f"for run_id {run_id!r}"
            )

        validated_records: list[dict[str, Any]] = []

        for record_index, record in enumerate(raw_records):
            if not isinstance(record, Mapping):
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}] must be a mapping, "
                    f"got {type(record)!r}"
                )

            step_index = record.get("step")
            instance_id = record.get("instance_id")
            full_name = record.get("full_name")
            child_run_id = record.get("run_id")

            if not isinstance(step_index, int) or step_index < 0:
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}]['step'] must be an int >= 0, "
                    f"got {step_index!r}"
                )
            if not isinstance(instance_id, str) or not instance_id.strip():
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}]['instance_id'] must be a "
                    f"non-empty string, got {instance_id!r}"
                )
            if not isinstance(full_name, str) or not full_name.strip():
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}]['full_name'] must be a "
                    f"non-empty string, got {full_name!r}"
                )
            if not isinstance(child_run_id, str) or not child_run_id.strip():
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}]['run_id'] must be a "
                    f"non-empty string, got {child_run_id!r}"
                )

            validated_records.append(dict(record))

        return validated_records

    def get_step_results(
        self,
        run_id: str,
    ) -> Optional[list[dict[str, Any] | None]]:
        """Return child step results for one sequential run.

        This method uses :meth:`get_step_records` as the source of truth.

        Returns
        -------
        Optional[list[dict[str, Any] | None]]
            - ``None`` if the parent sequential run is not found
            - otherwise, one entry per stored step record
            - each entry is the child step checkpoint result dict, or ``None`` if
              that child run id no longer resolves to a retained child checkpoint
        """
        step_records = self.get_step_records(run_id)
        if step_records is None:
            return None

        step_results: list[dict[str, Any] | None] = []

        for record_index, record in enumerate(step_records):
            step_index = record["step"]
            if step_index >= len(self._steps):
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}] references step index "
                    f"{step_index}, but only {len(self._steps)} configured steps exist"
                )

            step: Workflow = self._steps[step_index]

            recorded_instance_id = record["instance_id"]
            if step.instance_id != recorded_instance_id:
                raise ValidationError(
                    f"{self.full_name}: step_records[{record_index}] instance_id mismatch for "
                    f"step {step_index}: recorded {recorded_instance_id!r}, current {step.instance_id!r}"
                )

            child_checkpoint = step.get_checkpoint(record["run_id"])
            if child_checkpoint is None:
                step_results.append(None)
            else:
                step_results.append(dict(child_checkpoint.result))

        return step_results

    def get_step_result(
        self,
        run_id: str,
        step_index: int,
    ) -> Optional[dict[str, Any]]:
        """Return one child step result for one sequential run.

        This method uses :meth:`get_step_results` as its source of truth.

        Returns
        -------
        Optional[dict[str, Any]]
            - ``None`` if the parent sequential run is not found
            - the resolved child result dict for the requested step
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

        step_results = self.get_step_results(run_id)
        if step_results is None:
            return None

        return step_results[resolved_index]

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Synchronously execute all configured steps and return the selected step result."""
        if not self._steps:
            raise ValidationError(
                f"{self.full_name}: cannot execute an empty SequentialFlow"
            )

        running_result: Mapping[str, Any] = inputs
        step_records: list[dict[str, Any]] = []
        step_results: list[FlowResultDict] = []

        for index, step in enumerate(self._steps):
            logger.info("%s: invoking step %d (%s)", self.full_name, index, step.full_name)
            result = step.invoke(running_result)

            if not isinstance(result, FlowResultDict):
                raise ValidationError(
                    f"{self.full_name}: step {index} ({step.full_name}) returned "
                    f"{type(result)!r}, expected FlowResultDict"
                )

            step_records.append(
                {
                    "step": index,
                    "instance_id": step.instance_id,
                    "full_name": step.full_name,
                    "run_id": result.run_id,
                }
            )
            step_results.append(result)
            running_result = result

        resolved_return_step = self._resolve_step_index(self._return_index)
        returned_result = step_results[resolved_return_step]

        metadata = {
            "step_records": step_records,
            "return_index": self._return_index,
            "resolved_return_step": resolved_return_step,
            "returned_step_run_id": step_records[resolved_return_step]["run_id"],
        }
        return metadata, returned_result

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Asynchronously execute all configured steps and return the selected step result."""
        if not self._steps:
            raise ValidationError(
                f"{self.full_name}: cannot execute an empty SequentialFlow"
            )

        running_result: Mapping[str, Any] = inputs
        step_records: list[dict[str, Any]] = []
        step_results: list[FlowResultDict] = []

        for index, step in enumerate(self._steps):
            logger.info(
                "[Async %s]: invoking step %d (%s)",
                self.full_name,
                index,
                step.full_name,
            )
            result = await step.async_invoke(running_result)

            if not isinstance(result, FlowResultDict):
                raise ValidationError(
                    f"{self.full_name}: async step {index} ({step.full_name}) returned "
                    f"{type(result)!r}, expected FlowResultDict"
                )

            step_records.append(
                {
                    "step": index,
                    "instance_id": step.instance_id,
                    "full_name": step.full_name,
                    "run_id": result.run_id,
                }
            )
            step_results.append(result)
            running_result = result

        resolved_return_step = self._resolve_step_index(self._return_index)
        returned_result = step_results[resolved_return_step]

        metadata = {
            "step_records": step_records,
            "return_index": self._return_index,
            "resolved_return_step": resolved_return_step,
            "returned_step_run_id": step_records[resolved_return_step]["run_id"],
        }
        return metadata, returned_result

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
    - The normalized loop body is exposed read-only via :attr:`loop_body` and
      :attr:`body_steps`.
    - The judge setter accepts any ``AtomicInvokable`` or ``None``.
    - Internally, the judge wrapper is always present as a normalized ``BasicFlow``.
    - The judge wrapper itself stays stable across the object's lifecycle; only
      its wrapped component may change.
    - Passing ``None`` installs a shared fallback always-false structured judge.
    - ``return_index`` is proxied onto the loop body and controls which body
      step result becomes the body workflow's outer result.
    - ``handoff_index`` controls which body step result becomes the next
      iteration's input.
    - ``evaluate_index`` controls which body step result is passed to the judge.

    Judge contract
    --------------
    The judge is intentionally accepted as a broad ``AtomicInvokable``. At
    runtime, however, the *normalized* judge path must yield a raw boolean
    decision discoverable through the judge checkpoint metadata under
    ``"run_raw_result"``. If it does not, invocation raises ``ValidationError``.

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
    - This class does not re-expose any loop-body step mutation API.
    - Only selection policy is mutable post-construction:
      ``return_index``, ``handoff_index``, ``evaluate_index``, and ``judge``.

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
    - ``resolved_return_step``:
      Resolved absolute body-step index selected for the body return value.
    - ``resolved_handoff_step``:
      Resolved absolute body-step index selected for next-iteration inputs.
    - ``resolved_evaluate_step``:
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
            fallback always-false judge. The runtime contract still requires
            that the normalized judge path ultimately yields a raw boolean
            decision.
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

        self._judge = BasicFlow(
            component=_fallback_judge_component,
            name=f"{name}_judge",
            description=f"Normalized judge for iterative workflow {name}",
        )
        self.judge = judge

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def loop_body(self) -> SequentialFlow:
        """The fixed normalized sequential loop body."""
        return self._loop_body

    @property
    def judge(self) -> BasicFlow:
        """The always-installed normalized judge wrapper."""
        return self._judge

    @judge.setter
    def judge(self, candidate: AtomicInvokable | None) -> None:
        """Set the judge or reset it to the shared fallback always-false judge.

        Notes
        -----
        This setter intentionally accepts any ``AtomicInvokable`` or ``None``.
        It does not attempt to prove at assignment time that a non-fallback
        candidate can satisfy the runtime boolean-decision contract; that
        validation still occurs during invocation when the judge result is
        interpreted.
        """
        normalized_component = self._normalize_judge_component(candidate)
        self._judge.component = normalized_component

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
    def _normalize_judge_component(
        self,
        candidate: AtomicInvokable | None,
    ) -> StructuredInvokable:
        """Normalize a user-supplied judge or fallback into a structured component."""
        if candidate is None:
            return _fallback_judge_component

        if not isinstance(candidate, AtomicInvokable):
            raise TypeError(
                f"judge must be an AtomicInvokable or None, got {type(candidate)!r}"
            )

        if isinstance(candidate, StructuredInvokable):
            return candidate

        return StructuredInvokable(component=candidate, output_schema=[])

    def _extract_judge_decision(
        self,
        judge_flow: Workflow,
        judge_result: FlowResultDict,
    ) -> bool:
        """Extract and validate the raw boolean decision for one judge run.

        The decision is read from the normalized judge checkpoint metadata under
        ``"run_raw_result"``. ``NO_VAL`` is used as the fallback sentinel so the
        implementation can distinguish a missing key from a present-but-invalid
        non-boolean value.
        """
        checkpoint = judge_flow.get_checkpoint(judge_result.run_id)
        if checkpoint is None:
            raise ValidationError(
                f"{self.full_name}: judge checkpoint not found for run_id {judge_result.run_id!r}"
            )

        raw_decision = checkpoint.metadata.get("run_raw_result", NO_VAL)
        if raw_decision is NO_VAL:
            raise ValidationError(
                f"{self.full_name}: judge metadata did not contain 'run_raw_result'"
            )
        if not isinstance(raw_decision, bool):
            raise ValidationError(
                f"{self.full_name}: judge raw result must be bool, got {type(raw_decision)!r}"
            )

        return raw_decision

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
        resolved_return_step: int,
        resolved_handoff_step: int,
        resolved_evaluate_step: int,
        iteration_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the final outer checkpoint metadata for one iterative run."""
        return {
            "iterations_completed": iterations_completed,
            "max_iterations": self._max_iterations,
            "stop_reason": stop_reason,
            "resolved_return_step": resolved_return_step,
            "resolved_handoff_step": resolved_handoff_step,
            "resolved_evaluate_step": resolved_evaluate_step,
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

            judge_decision = self._extract_judge_decision(
                self._judge,
                judge_result,
            )

            iteration_records.append({
                "iteration": iteration,
                "body_run_id": body_run_id,
                "judge_run_id": judge_result.run_id,
                "judge_decision": judge_decision,
            })
            iterations_completed += 1

            final_result = body_result
            current_inputs = handoff_result

            if judge_decision is True:
                stop_reason = "judge_approved"
                break

        metadata = self._build_run_metadata(
            iterations_completed=iterations_completed,
            stop_reason=stop_reason,
            resolved_return_step=resolved_return_step,
            resolved_handoff_step=resolved_handoff_step,
            resolved_evaluate_step=resolved_evaluate_step,
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

            judge_decision = self._extract_judge_decision(
                self._judge,
                judge_result,
            )

            iteration_records.append({
                "iteration": iteration,
                "body_run_id": body_run_id,
                "judge_run_id": judge_result.run_id,
                "judge_decision": judge_decision,
            })

            iterations_completed += 1

            final_result = body_result
            current_inputs = handoff_result

            if judge_decision is True:
                stop_reason = "judge_approved"
                break

        metadata = self._build_run_metadata(
            iterations_completed=iterations_completed,
            stop_reason=stop_reason,
            resolved_return_step=resolved_return_step,
            resolved_handoff_step=resolved_handoff_step,
            resolved_evaluate_step=resolved_evaluate_step,
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
