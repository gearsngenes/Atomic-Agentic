from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional

from ..constants.core import NO_VAL
from ..exceptions import ValidationError
from ..core.Invokable import AtomicInvokable
from ..models.results.workflows import IterativeFlowResult
from .base import Workflow
from .sequential import SequentialFlow

logger = logging.getLogger(__name__)

__all__ = ["IterativeFlow"]


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
    - ``body_steps`` must be a non-empty ``list[AtomicInvokable]``.
    - The body is always normalized inline into a private ``SequentialFlow``.
    - The normalized loop body topology is fixed at construction.
    - The normalized loop body is exposed read-only via :attr:`loop_body`.
    - ``judge`` accepts any ``AtomicInvokable`` or ``None``:
        - If ``None``, no judge is invoked at all; the loop always runs
          every configured iteration to ``max_iterations``.
          ``approval_value`` must be left as ``NO_VAL`` (raises
          ``ValueError`` if a value is provided).
        - Otherwise, the judge is validated as an ``AtomicInvokable`` and
          stored as configured, no wrapping. ``approval_value`` must be
          provided (raises ``ValueError`` if it is left as ``NO_VAL``).
    - ``approval_value`` is fixed at construction (read-only thereafter).
    - ``return_index``, ``handoff_index``, ``evaluate_index`` each default to
      the last body step when ``None``. Explicit values support normal
      negative indexing and are resolved once at construction; out-of-range
      values raise ``IndexError``. All three are fixed thereafter (read-only).

    Judge contract
    --------------
    At runtime, when a judge is configured, it is invoked with the
    mapping-shaped result of the configured ``evaluate_index`` body step.
    The loop exits early once ``judge_result.result == approval_value``. Any
    value may be used for ``approval_value`` as long as it supports ``==``
    against the judge's unwrapped result. When no judge is configured, this
    step is skipped entirely every iteration and the loop always runs to
    ``max_iterations``.

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
    """

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        body_steps: list[AtomicInvokable],
        judge: AtomicInvokable | None = None,
        max_iterations: int = 1,
        *,
        return_index: Optional[int] = None,
        handoff_index: Optional[int] = None,
        evaluate_index: Optional[int] = None,
        approval_value: Any = NO_VAL,
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
            Optional judge invokable. Passing ``None`` means no judge is
            invoked at all; the loop always runs every configured iteration
            to ``max_iterations`` (``approval_value`` must be left as
            ``NO_VAL`` in that case).
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

        num_steps = len(body_steps)
        resolved_return_index = self._resolve_body_step_index(return_index, num_steps)
        resolved_handoff_index = self._resolve_body_step_index(handoff_index, num_steps)
        resolved_evaluate_index = self._resolve_body_step_index(evaluate_index, num_steps)

        self._loop_body = SequentialFlow(
            name=f"{name}_loop_body",
            namespace=namespace,
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
            self._judge: AtomicInvokable | None = None
        else:
            if approval_value is NO_VAL:
                raise ValueError(
                    f"{name}: approval_value must be provided when an explicit judge is given"
                )
            self._judge = self._normalize_judge(judge)

        self._approval_value = approval_value

        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            parameters=self._loop_body.parameters,
            return_type=self._loop_body.return_type,
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def loop_body(self) -> SequentialFlow:
        """The fixed normalized sequential loop body."""
        return self._loop_body

    @property
    def judge(self) -> AtomicInvokable | None:
        """The fixed configured judge, or None if no judge was provided."""
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

    def _extra_description(self) -> str:
        """Chain the loop body's extra description with a fixed iteration-bound note.

        The loop body is a `SequentialFlow`, whose own `_extra_description()`
        already resolves to its return-index step - the step that actually
        produces this flow's result payload each run - so this is a true 1:1
        passthrough. The iteration-bound note is always present.
        """
        loop_extra = self._loop_body._extra_description()
        parts = [loop_extra] if loop_extra else []
        parts.append(f"Runs up to {self._max_iterations} iteration(s).")
        return "\n".join(parts)

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
    def _normalize_judge(judge: AtomicInvokable) -> AtomicInvokable:
        """Validate the configured judge is an AtomicInvokable."""
        if not isinstance(judge, AtomicInvokable):
            raise TypeError(
                f"IterativeFlow judge must be AtomicInvokable, got {type(judge)!r}"
            )
        return judge

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

            handoff_payload = self._loop_body.get_step_result(
                body_result.run_id, self._handoff_step_index
            ).result
            if not isinstance(handoff_payload, Mapping):
                raise ValidationError(
                    f"{self.full_name}: handoff step {self._handoff_step_index} result "
                    f"must be mapping-shaped, got {type(handoff_payload)!r}"
                )

            approved = False
            if self._judge is not None:
                logger.info("%s: invoking judge for iteration %d", self.full_name, iteration)
                judge_result = self._judge.invoke(evaluate_payload)
                judge_runs.append(judge_result.run_id)
                approved = judge_result.result == self._approval_value

            if approved:
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

            handoff_payload = self._loop_body.get_step_result(
                body_result.run_id, self._handoff_step_index
            ).result
            if not isinstance(handoff_payload, Mapping):
                raise ValidationError(
                    f"{self.full_name}: handoff step {self._handoff_step_index} result "
                    f"must be mapping-shaped, got {type(handoff_payload)!r}"
                )

            approved = False
            if self._judge is not None:
                logger.info("[Async %s]: invoking judge for iteration %d", self.full_name, iteration)
                judge_result = await self._judge.async_invoke(evaluate_payload)
                judge_runs.append(judge_result.run_id)
                approved = judge_result.result == self._approval_value

            if approved:
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
                "judge": self._judge.to_dict() if self._judge is not None else None,
                "max_iterations": self._max_iterations,
                "return_index": self.return_index,
                "handoff_index": self._handoff_step_index,
                "evaluate_index": self._evaluate_step_index,
                "approval_value": self._approval_value,
            }
        )
        return data
