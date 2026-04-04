from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from .StructuredInvokable import StructuredInvokable
from ..core.Exceptions import ValidationError
from ..core.Invokable import AtomicInvokable
from ..core.sentinels import NO_VAL
from .base import FlowResultDict, Workflow
from .basic import BasicFlow

logger = logging.getLogger(__name__)

__all__ = ["SequentialFlow", "IterativeFlow"]


class SequentialFlow(Workflow):
    """Execute an ordered sequence of workflow-shaped steps.

    Step normalization
    ------------------
    - Existing ``Workflow`` instances are kept as-is.
    - ``StructuredInvokable`` instances are wrapped once in ``BasicFlow``.

    Runtime contract
    ----------------
    - Inputs are passed to the first step.
    - Each step's mapping result is passed directly to the next step.
    - The final step result is returned unchanged to the workflow base, which
      then wraps it in the outer ``FlowResultDict`` and records the sequential
      checkpoint.

    Metadata
    --------
    Per-run metadata contains:

    - ``step_records``: list[dict[str, Any]]
        One record per executed step, each containing:
        - ``step``: zero-based step index for that run
        - ``instance_id``: executed step instance id
        - ``full_name``: executed step full name
        - ``run_id``: that step's workflow run id

    Notes
    -----
    - No packaging or handoff-schema rewiring is performed here.
    - No removed-step archive is maintained; displaced steps are returned
      directly by mutation methods, and sequential checkpoints preserve
      historical ``step_records`` for interpretation later.
    """

    def __init__(
        self,
        name: str,
        description: str,
        steps: Optional[list[Workflow | StructuredInvokable]] = None,
        *,
        filter_extraneous_inputs: Optional[bool] = None,
    ) -> None:
        normalized_steps = [
            self._normalize_step(step) for step in (steps or [])
        ]

        self._steps: list[Workflow] = normalized_steps

        resolved_filter = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else (
                normalized_steps[0].filter_extraneous_inputs
                if normalized_steps
                else True
            )
        )

        super().__init__(
            name=name,
            description=description,
            parameters=normalized_steps[0].parameters if normalized_steps else [],
            filter_extraneous_inputs=resolved_filter,
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def steps(self) -> list[Workflow]:
        """Return a shallow copy of the active steps."""
        return list(self._steps)

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

    def _refresh_parameters(self) -> None:
        """Refresh the sequential input contract from the first active step."""
        self._parameters = self._steps[0].parameters if self._steps else []

    def _resolve_existing_index(self, index: int) -> int:
        """Resolve a list-style index for an existing step."""
        if not isinstance(index, int):
            raise TypeError(f"step index must be an int, got {type(index)!r}")

        length = len(self._steps)
        resolved = index if index >= 0 else length + index
        if resolved < 0 or resolved >= length:
            raise IndexError(
                f"step index {index} out of range for {length} configured steps"
            )
        return resolved

    def _find_index_by_full_name(self, full_name: str) -> int:
        """Find exactly one active step by full name."""
        if not isinstance(full_name, str) or not full_name.strip():
            raise ValueError("full_name must be a non-empty string")

        matches = [
            index
            for index, step in enumerate(self._steps)
            if step.full_name == full_name
        ]

        if not matches:
            raise ValueError(f"no step found with full_name {full_name!r}")
        if len(matches) > 1:
            raise ValueError(
                f"multiple steps found with full_name {full_name!r}; remove by index instead"
            )
        return matches[0]

    # ------------------------------------------------------------------ #
    # Mutation API
    # ------------------------------------------------------------------ #
    def add_step(self, step: Workflow | StructuredInvokable) -> None:
        """Append one normalized step to the end of the sequence."""
        self._steps.append(self._normalize_step(step))
        self._refresh_parameters()

    def insert_step(self, index: int, step: Workflow | StructuredInvokable) -> None:
        """Insert one normalized step using standard list.insert semantics."""
        if not isinstance(index, int):
            raise TypeError(f"step index must be an int, got {type(index)!r}")
        self._steps.insert(index, self._normalize_step(step))
        self._refresh_parameters()

    def pop_step(self, index: int = -1) -> Workflow:
        """Remove and return the step at ``index``."""
        resolved = self._resolve_existing_index(index)
        removed = self._steps.pop(resolved)
        self._refresh_parameters()
        return removed

    def remove_step(self, target: int | str) -> Workflow:
        """Remove and return a step by index or full name."""
        if isinstance(target, int):
            resolved = self._resolve_existing_index(target)
        elif isinstance(target, str):
            resolved = self._find_index_by_full_name(target)
        else:
            raise TypeError(
                "remove_step target must be an int index or str full_name, "
                f"got {type(target)!r}"
            )

        removed = self._steps.pop(resolved)
        self._refresh_parameters()
        return removed

    def replace_step(
        self,
        index: int,
        step: Workflow | StructuredInvokable,
    ) -> Workflow:
        """Replace one step by index and return the displaced step."""
        resolved = self._resolve_existing_index(index)
        replacement = self._normalize_step(step)
        displaced = self._steps[resolved]
        self._steps[resolved] = replacement
        self._refresh_parameters()
        return displaced

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Synchronously execute each configured step in order."""
        if not self._steps:
            raise ValidationError(
                f"{self.full_name}: cannot execute an empty SequentialFlow"
            )

        running_result: Mapping[str, Any] = inputs
        step_records: list[dict[str, Any]] = []

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
            running_result = result

        return {"step_records": step_records}, running_result

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Asynchronously execute each configured step in order."""
        if not self._steps:
            raise ValidationError(
                f"{self.full_name}: cannot execute an empty SequentialFlow"
            )

        running_result: Mapping[str, Any] = inputs
        step_records: list[dict[str, Any]] = []

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
            running_result = result

        return {"step_records": step_records}, running_result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the sequence and its active step snapshots."""
        data = super().to_dict()
        data.update(
            {
                "steps": [step.to_dict() for step in self._steps],
                "step_count": len(self._steps),
            }
        )
        return data

class IterativeFlow(Workflow):
    """Repeat a fixed sequential body up to ``max_iterations``.

    Overview
    --------
    ``IterativeFlow`` is a composite workflow that repeatedly invokes a
    normalized sequential body. After each completed body run, an optional
    judge may inspect that iteration's body result and decide whether the
    loop should stop early.

    Construction contract
    ---------------------
    - ``body_steps`` must be a non-empty ``list[Workflow | StructuredInvokable]``.
    - The body is always normalized inline into a private ``SequentialFlow``.
    - The normalized loop body is exposed read-only via :attr:`loop_body`.
    - The judge is optional and may be any ``AtomicInvokable``.
      Internally, non-``StructuredInvokable`` judges are wrapped in a
      ``StructuredInvokable(output_schema=[])`` and then in a ``BasicFlow``.

    Judge contract
    --------------
    The judge is intentionally accepted as a broad ``AtomicInvokable``. At
    runtime, however, the *normalized* judge path must yield a raw boolean
    decision discoverable through the judge checkpoint metadata under
    ``"run_raw_result"``. If it does not, invocation raises ``ValidationError``.

    Loop semantics
    --------------
    - Iteration 0 receives the outer workflow inputs.
    - Each completed body result becomes the next iteration's inputs.
    - If a judge is present, it is invoked after each body run.
    - The loop stops early only when the judge decision is ``True``.
    - Otherwise it continues until ``max_iterations`` is exhausted.

    Metadata
    --------
    Per-run metadata contains:

    - ``body_instance_id``:
      The normalized sequential body instance id.
    - ``body_full_name``:
      The normalized sequential body full name.
    - ``iterations_completed``:
      Number of completed body iterations.
    - ``max_iterations``:
      Maximum number of iterations permitted for this run.
    - ``stopped_early``:
      Whether a judge approved early termination.
    - ``iteration_records``:
      ``list[dict[str, Any]]`` where each record contains:
        - ``iteration``: zero-based iteration index
        - ``body_run_id``: body workflow run id for that iteration
        - ``body_result``: body workflow result mapping snapshot
        - ``judge_instance_id``: judge instance id, or ``None`` when absent
        - ``judge_run_id``: judge workflow run id, or ``None`` when absent
        - ``judge_decision``: boolean decision, or ``None`` when no judge ran
    """

    def __init__(
        self,
        name: str,
        description: str,
        body_steps: list[Workflow | StructuredInvokable],
        judge: AtomicInvokable | None = None,
        max_iterations: int = 1,
        *,
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
            Optional judge invokable. The runtime contract requires that the
            normalized judge path ultimately yields a raw boolean decision.
        max_iterations:
            Maximum number of body iterations to execute per run. Must be > 0.
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
        )

        self._judge: Optional[BasicFlow] = None
        self.judge = judge
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

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def loop_body(self) -> SequentialFlow:
        """The normalized sequential loop body."""
        return self._loop_body

    @property
    def judge(self) -> Optional[BasicFlow]:
        """The optional normalized judge wrapper."""
        return self._judge

    @judge.setter
    def judge(self, candidate: AtomicInvokable | None) -> None:
        """Set or clear the optional judge.

        Notes
        -----
        This setter intentionally accepts any ``AtomicInvokable``. It does not
        attempt to prove at assignment time that the candidate can satisfy the
        runtime boolean-decision contract; that validation occurs during
        invocation when the judge result is interpreted.
        """
        if candidate is None:
            self._judge = None
            return

        if not isinstance(candidate, AtomicInvokable):
            raise TypeError(
                f"judge must be an AtomicInvokable or None, got {type(candidate)!r}"
            )

        resolved_component = (
            candidate
            if isinstance(candidate, StructuredInvokable)
            else StructuredInvokable(component=candidate, output_schema=[])
        )

        if self._judge is not None:
            self._judge.component = resolved_component
        else:
            self._judge = BasicFlow(component=resolved_component)

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

    # ------------------------------------------------------------------ #
    # Body augmentation passthrough
    # ------------------------------------------------------------------ #
    @property
    def body_steps(self) -> list[Workflow]:
        """Return the current normalized body step list."""
        return self._loop_body.steps

    def add_step(self, step: Workflow | StructuredInvokable) -> None:
        """Append one normalized step to the loop body."""
        self._loop_body.add_step(step)
        self._parameters = self._loop_body.parameters

    def insert_step(self, index: int, step: Workflow | StructuredInvokable) -> None:
        """Insert one normalized step into the loop body."""
        self._loop_body.insert_step(index, step)
        self._parameters = self._loop_body.parameters

    def pop_step(self, index: int = -1) -> Workflow:
        """Remove and return the loop body step at ``index``."""
        removed = self._loop_body.pop_step(index)
        self._parameters = self._loop_body.parameters
        return removed

    def remove_step(self, target: int | str) -> Workflow:
        """Remove and return a loop body step by index or full name."""
        removed = self._loop_body.remove_step(target)
        self._parameters = self._loop_body.parameters
        return removed

    def replace_step(
        self,
        index: int,
        step: Workflow | StructuredInvokable,
    ) -> Workflow:
        """Replace one loop body step and return the displaced step."""
        displaced = self._loop_body.replace_step(index, step)
        self._parameters = self._loop_body.parameters
        return displaced

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Synchronously execute the iterative loop."""
        current_inputs: Mapping[str, Any] = inputs
        final_result: Mapping[str, Any] = dict(inputs)
        iteration_records: list[dict[str, Any]] = []
        iterations_completed = 0
        stopped_early = False

        for iteration in range(self.max_iterations):
            logger.info("%s: iteration %d", self.full_name, iteration)
            body_result = self._loop_body.invoke(current_inputs)

            if not isinstance(body_result, FlowResultDict):
                raise ValidationError(
                    f"{self.full_name}: body returned {type(body_result)!r}, expected FlowResultDict"
                )

            judge_run_id: str | None = None
            judge_decision: bool | None = None

            final_result = body_result
            current_inputs = body_result

            if self._judge is not None:
                logger.info(
                    "%s: invoking judge for iteration %d",
                    self.full_name,
                    iteration,
                )
                judge_result = self._judge.invoke(body_result)

                if not isinstance(judge_result, FlowResultDict):
                    raise ValidationError(
                        f"{self.full_name}: judge returned {type(judge_result)!r}, expected FlowResultDict"
                    )

                judge_run_id = judge_result.run_id
                judge_decision = self._extract_judge_decision(
                    self._judge,
                    judge_result,
                )

            iteration_records.append(
                {
                    "iteration": iteration,
                    "body_run_id": body_result.run_id,
                    "body_result": dict(body_result),
                    "judge_instance_id": (
                        self._judge.instance_id if self._judge is not None else None
                    ),
                    "judge_run_id": judge_run_id,
                    "judge_decision": judge_decision,
                }
            )
            iterations_completed += 1

            if judge_decision is True:
                stopped_early = True
                break

        metadata = {
            "body_instance_id": self._loop_body.instance_id,
            "body_full_name": self._loop_body.full_name,
            "iterations_completed": iterations_completed,
            "max_iterations": self._max_iterations,
            "stopped_early": stopped_early,
            "iteration_records": iteration_records,
        }
        return metadata, final_result

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Asynchronously execute the iterative loop."""
        current_inputs: Mapping[str, Any] = inputs
        final_result: Mapping[str, Any] = dict(inputs)
        iteration_records: list[dict[str, Any]] = []
        iterations_completed = 0
        stopped_early = False

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

            judge_run_id: str | None = None
            judge_decision: bool | None = None

            final_result = body_result
            current_inputs = body_result

            if self._judge is not None:
                logger.info(
                    "[Async %s]: invoking judge for iteration %d",
                    self.full_name,
                    iteration,
                )
                judge_result = await self._judge.async_invoke(body_result)

                if not isinstance(judge_result, FlowResultDict):
                    raise ValidationError(
                        f"{self.full_name}: async judge returned {type(judge_result)!r}, expected FlowResultDict"
                    )

                judge_run_id = judge_result.run_id
                judge_decision = self._extract_judge_decision(
                    self._judge,
                    judge_result,
                )

            iteration_records.append(
                {
                    "iteration": iteration,
                    "body_run_id": body_result.run_id,
                    "body_result": dict(body_result),
                    "judge_instance_id": (
                        self._judge.instance_id if self._judge is not None else None
                    ),
                    "judge_run_id": judge_run_id,
                    "judge_decision": judge_decision,
                }
            )
            iterations_completed += 1

            if judge_decision is True:
                stopped_early = True
                break

        metadata = {
            "body_instance_id": self._loop_body.instance_id,
            "body_full_name": self._loop_body.full_name,
            "iterations_completed": iterations_completed,
            "max_iterations": self._max_iterations,
            "stopped_early": stopped_early,
            "iteration_records": iteration_records,
        }
        return metadata, final_result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the iterative workflow and its normalized children."""
        data = super().to_dict()
        data.update(
            {
                "loop_body": self.loop_body.to_dict(),
                "judge": self._judge.to_dict() if self._judge is not None else None,
                "max_iterations": self._max_iterations,
            }
        )
        return data
