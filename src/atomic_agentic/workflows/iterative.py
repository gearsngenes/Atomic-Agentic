from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any

from ..constants.core import NO_VAL
from ..exceptions import ValidationError
from ..core.Invokable import AtomicInvokable
from ..models.workflows.checkers import CheckerSpec
from ..models.results.workflows import IterativeFlowResult
from .base import Workflow

logger = logging.getLogger(__name__)

__all__ = ["IterativeFlow"]


class IterativeFlow(Workflow):
    """Repeat a body of steps up to ``max_iterations``, with checker-gated
    early exit.

    Construction contract
    ----------------------
    - ``body_steps`` must be a non-empty ``list[AtomicInvokable]``.
      ``IterativeFlow`` owns step-by-step execution directly -- there is no
      nested ``SequentialFlow`` loop body.
    - ``checkers`` is an optional bulk list of ``CheckerSpec`` for
      construction-time convenience; each is fed through :meth:`add_checker`
      (the single validated entry point -- no duplicated validation logic).
      Checkers are the one fixed-topology exception on this class:
      :meth:`add_checker`/:meth:`remove_checker` remain callable
      post-construction, since checkers never affect declared
      ``parameters``/``return_type`` (see ``01-overview.md`` SS4's "Fixed
      topology" carve-out).
    - ``result_setting_indices`` is a ``list[int]``, defaulting to ``[]``.
      Each element is resolved via ``_resolve_body_step_index`` (negative
      wraparound + range check). Fixed at construction -- it feeds
      ``return_type``.
    - ``handoff_index`` defaults to the last body step (same resolution
      helper). Fixed at construction.
    - ``fallback_value`` defaults to ``NO_VAL``. If ``result_setting_indices``
      resolves to empty *and* ``fallback_value`` is ``NO_VAL``, construction
      raises immediately -- that combination is guaranteed to fail on the
      very first invocation.
    - ``include_trace`` defaults to ``True``, forwarded to the base
      ``Workflow``.

    Runtime contract
    -----------------
    Every iteration executes ``body_steps`` in order. At each step:
    result-setting positions update a running "current answer"; the
    handoff position's result is captured (used only if the loop
    continues); a configured checker's judge is invoked on that step's
    result (which must be mapping-shaped) and a match stops the loop
    immediately, mid-body. Absent an early exit, all steps run every
    iteration and the loop continues until ``max_iterations``. The final
    "current answer" is the result; if none was ever set, ``fallback_value``
    is used, or the run raises if no fallback was configured either.

    Mutability notes
    ----------------
    - ``body_steps``, ``result_setting_indices``, ``handoff_index`` are
      fixed after construction.
    - ``checkers`` and ``max_iterations`` remain mutable post-construction.

    Result
    ------
    See ``IterativeFlowResult`` for per-run fields (``exited_early``,
    ``iterations_completed``, ``triggering_step``, ``result_setting_indices``,
    ``handoff_index``, ``max_iterations``).
    """

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        body_steps: list[AtomicInvokable],
        max_iterations: int = 1,
        *,
        checkers: list[CheckerSpec] | None = None,
        result_setting_indices: list[int] | None = None,
        handoff_index: int | None = None,
        fallback_value: Any = NO_VAL,
        include_trace: bool = True,
    ) -> None:
        """Initialize the iterative workflow.

        Parameters
        ----------
        name:
            Workflow name.
        description:
            Human-readable workflow description.
        body_steps:
            Non-empty list of steps executed, in order, every iteration.
        max_iterations:
            Maximum number of body iterations to execute per run. Must be > 0.
        checkers:
            Optional bulk-construction list of ``CheckerSpec``. Each is
            registered via :meth:`add_checker`.
        result_setting_indices:
            Body-step positions whose results update the running "current
            answer". Defaults to ``[]``.
        handoff_index:
            Body-step position whose result feeds the next iteration's
            inputs. Defaults to the last body step.
        fallback_value:
            Value returned when no result-setting step is ever reached.
            Required (must not be ``NO_VAL``) when ``result_setting_indices``
            resolves to empty.
        include_trace:
            Whether results populate ``trace``. Forwarded to the base
            ``Workflow``.
        """
        if not isinstance(body_steps, list):
            raise TypeError(
                f"body_steps must be a list[AtomicInvokable], got {type(body_steps)!r}"
            )
        if not body_steps:
            raise ValueError("body_steps must not be empty")

        for index, step in enumerate(body_steps):
            if not isinstance(step, AtomicInvokable):
                raise TypeError(
                    "body_steps items must be AtomicInvokable, "
                    f"got {type(step)!r} at index {index}"
                )
        self._body_steps: tuple[AtomicInvokable, ...] = tuple(body_steps)

        if not isinstance(max_iterations, int):
            raise TypeError(f"max_iterations must be an int, got {type(max_iterations)!r}")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be > 0")
        self._max_iterations = max_iterations

        # Checkers are validated and stored entirely through add_checker --
        # bulk construction just replays the same single validated path.
        self._checkers: dict[int, CheckerSpec] = {}
        for spec in checkers or []:
            self.add_checker(spec.index, spec.judge, spec.approval_value)

        num_steps = len(self._body_steps)
        resolved_result_setting_indices = []
        for index in result_setting_indices or []:
            if not isinstance(index, int):
                raise TypeError(
                    f"result_setting_indices items must be int, got {type(index)!r}"
                )
            resolved_result_setting_indices.append(
                self._resolve_body_step_index(index, num_steps)
            )
        self._result_setting_indices: tuple[int, ...] = tuple(resolved_result_setting_indices)

        self._handoff_index: int = self._resolve_body_step_index(handoff_index, num_steps)

        if not self._result_setting_indices and fallback_value is NO_VAL:
            raise ValueError(
                f"{name}: result_setting_indices is empty and fallback_value is NO_VAL; "
                "this configuration would always fail at runtime -- provide a "
                "fallback_value or at least one result-setting index"
            )
        self._fallback_value = fallback_value

        parameters = self._body_steps[0].parameters

        if self._result_setting_indices:
            selected_types = [
                self._body_steps[i].return_type for i in self._result_setting_indices
            ]
            unique_types = list(dict.fromkeys(selected_types))
            resolved_return_type = (
                unique_types[0] if len(unique_types) == 1 else " | ".join(unique_types)
            )
        else:
            resolved_return_type = type(self._fallback_value).__name__

        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            parameters=parameters,
            return_type=resolved_return_type,
            include_trace=include_trace,
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def body_steps(self) -> tuple[AtomicInvokable, ...]:
        """Fixed configured body-step tuple."""
        return self._body_steps

    @property
    def checkers(self) -> MappingProxyType[int, CheckerSpec]:
        """Read-only live view of currently configured checkers, keyed by index."""
        return MappingProxyType(self._checkers)

    @property
    def result_setting_indices(self) -> tuple[int, ...]:
        """Fixed body-step positions that update the running current answer."""
        return self._result_setting_indices

    @property
    def handoff_index(self) -> int:
        """Fixed body-step position whose result feeds the next iteration."""
        return self._handoff_index

    @property
    def fallback_value(self) -> Any:
        """Fixed value returned when no result-setting step is ever reached."""
        return self._fallback_value

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

    def _extra_description(self) -> str:
        """Chain the result-setting steps' own extra descriptions, then the
        fixed iteration-bound note (always present)."""
        parts = []
        for i, step in enumerate(self._body_steps):
            if i in self._result_setting_indices:
                extra = step._extra_description()
                if extra:
                    parts.append(extra)
        parts.append(f"Runs up to {self._max_iterations} iteration(s).")
        return "\n".join(parts)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_body_step_index(index: int | None, num_steps: int) -> int:
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

    # ------------------------------------------------------------------ #
    # Checker mutation API (post-construction, unlike everything else)
    # ------------------------------------------------------------------ #
    def add_checker(self, index: int, judge: AtomicInvokable, approval_value: Any) -> None:
        """Register a new checker at ``index``.

        Raises ``TypeError`` if ``index``/``judge`` are the wrong type,
        ``IndexError`` if ``index`` is out of range (no negative-index
        wraparound -- the index you add is the index you get back), and
        ``ValueError`` if ``index`` already has a checker.
        """
        if not isinstance(index, int):
            raise TypeError(f"checker index must be an int, got {type(index)!r}")
        if not (0 <= index < len(self._body_steps)):
            raise IndexError(
                f"checker index {index} out of range for "
                f"{len(self._body_steps)} configured body step(s)"
            )
        if not isinstance(judge, AtomicInvokable):
            raise TypeError(f"checker judge must be AtomicInvokable, got {type(judge)!r}")
        if index in self._checkers:
            raise ValueError(f"a checker is already registered at index {index}")
        self._checkers[index] = CheckerSpec(index=index, judge=judge, approval_value=approval_value)

    def remove_checker(self, index: int) -> None:
        """Remove the checker registered at ``index``.

        Raises ``ValueError`` if no checker is registered there.
        """
        if index not in self._checkers:
            raise ValueError(f"no checker is registered at index {index}")
        del self._checkers[index]

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
        """Construct an IterativeFlowResult envelope for this workflow's invocation.

        No fixed-topology constant is injected here -- everything (including
        the mutable checkers/max_iterations, snapshotted at call time)
        already arrives via ``result_kwargs`` from ``_run``/``_async_run``.
        """
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
        current_value: Any = self._fallback_value
        running_inputs: Mapping[str, Any] = inputs
        exited_early = False
        triggering_step: int | None = None
        trace_entries = []

        for iteration in range(self._max_iterations):
            logger.info("%s: iteration %d", self.full_name, iteration)
            handoff_candidate: Any = NO_VAL

            for step_index, step in enumerate(self._body_steps):
                result = step.invoke(running_inputs)
                trace_entries.append(result)

                if step_index in self._result_setting_indices:
                    current_value = result.result

                if step_index == self._handoff_index:
                    handoff_candidate = result.result

                checker = self._checkers.get(step_index)
                if checker is not None:
                    if not isinstance(result.result, Mapping):
                        raise ValidationError(
                            f"{self.full_name}: checker at step {step_index} requires a "
                            f"mapping-shaped result, got {type(result.result)!r}"
                        )
                    logger.info(
                        "%s: invoking checker at step %d", self.full_name, step_index
                    )
                    judge_result = checker.judge.invoke(result.result)
                    trace_entries.append(judge_result)
                    if judge_result.result == checker.approval_value:
                        exited_early = True
                        triggering_step = step_index
                        break

                if step_index < len(self._body_steps) - 1:
                    if not isinstance(result.result, Mapping):
                        raise ValidationError(
                            f"{self.full_name}: step {step_index} ({step.full_name}) must "
                            f"produce a mapping-shaped result to chain to the next step, "
                            f"got {type(result.result)!r}"
                        )
                    running_inputs = result.result

            if exited_early:
                break

            if iteration < self._max_iterations - 1:
                if not isinstance(handoff_candidate, Mapping):
                    raise ValidationError(
                        f"{self.full_name}: handoff step {self._handoff_index} requires a "
                        f"mapping-shaped result for the next iteration, "
                        f"got {type(handoff_candidate)!r}"
                    )
                running_inputs = handoff_candidate

        if current_value is NO_VAL:
            raise ValidationError(
                f"{self.full_name}: no result-setting step was ever reached and no "
                "fallback_value was configured"
            )

        return current_value, {
            "trace": tuple(trace_entries) if self.include_trace else None,
            "exited_early": exited_early,
            "iterations_completed": iteration + 1,
            "triggering_step": triggering_step,
            "result_setting_indices": self._result_setting_indices,
            "handoff_index": self._handoff_index,
            "max_iterations": self._max_iterations,
        }

    async def _async_run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Asynchronously execute the iterative loop."""
        current_value: Any = self._fallback_value
        running_inputs: Mapping[str, Any] = inputs
        exited_early = False
        triggering_step: int | None = None
        trace_entries = []

        for iteration in range(self._max_iterations):
            logger.info("[Async %s]: iteration %d", self.full_name, iteration)
            handoff_candidate: Any = NO_VAL

            for step_index, step in enumerate(self._body_steps):
                result = await step.async_invoke(running_inputs)
                trace_entries.append(result)

                if step_index in self._result_setting_indices:
                    current_value = result.result

                if step_index == self._handoff_index:
                    handoff_candidate = result.result

                checker = self._checkers.get(step_index)
                if checker is not None:
                    if not isinstance(result.result, Mapping):
                        raise ValidationError(
                            f"{self.full_name}: checker at step {step_index} requires a "
                            f"mapping-shaped result, got {type(result.result)!r}"
                        )
                    logger.info(
                        "[Async %s]: invoking checker at step %d", self.full_name, step_index
                    )
                    judge_result = await checker.judge.async_invoke(result.result)
                    trace_entries.append(judge_result)
                    if judge_result.result == checker.approval_value:
                        exited_early = True
                        triggering_step = step_index
                        break

                if step_index < len(self._body_steps) - 1:
                    if not isinstance(result.result, Mapping):
                        raise ValidationError(
                            f"{self.full_name}: async step {step_index} ({step.full_name}) "
                            f"must produce a mapping-shaped result to chain to the next "
                            f"step, got {type(result.result)!r}"
                        )
                    running_inputs = result.result

            if exited_early:
                break

            if iteration < self._max_iterations - 1:
                if not isinstance(handoff_candidate, Mapping):
                    raise ValidationError(
                        f"{self.full_name}: handoff step {self._handoff_index} requires a "
                        f"mapping-shaped result for the next iteration, "
                        f"got {type(handoff_candidate)!r}"
                    )
                running_inputs = handoff_candidate

        if current_value is NO_VAL:
            raise ValidationError(
                f"{self.full_name}: no result-setting step was ever reached and no "
                "fallback_value was configured"
            )

        return current_value, {
            "trace": tuple(trace_entries) if self.include_trace else None,
            "exited_early": exited_early,
            "iterations_completed": iteration + 1,
            "triggering_step": triggering_step,
            "result_setting_indices": self._result_setting_indices,
            "handoff_index": self._handoff_index,
            "max_iterations": self._max_iterations,
        }

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the iterative workflow and its fixed/current configuration."""
        data = super().to_dict()
        data.update(
            {
                "body_steps": [step.to_dict() for step in self._body_steps],
                "step_count": len(self._body_steps),
                "checkers": [
                    {
                        "index": c.index,
                        "judge": c.judge.to_dict(),
                        "approval_value": c.approval_value,
                    }
                    for c in sorted(self._checkers.values(), key=lambda c: c.index)
                ],
                "result_setting_indices": list(self._result_setting_indices),
                "handoff_index": self._handoff_index,
                "fallback_value": self._fallback_value,
                "max_iterations": self._max_iterations,
            }
        )
        return data
