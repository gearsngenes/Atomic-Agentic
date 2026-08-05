from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

from .atomic import AtomicResult

__all__ = [
    "WorkflowResult",
    "SequentialFlowResult",
    "RoutingFlowResult",
    "IterativeFlowResult",
    "ParallelFlowResult",
]


# ------------------------------------------------------------------ #
# WorkflowResult hierarchy
# ------------------------------------------------------------------ #

@dataclass(frozen=True, slots=True)
class WorkflowResult(AtomicResult):
    """Base successful Workflow invocation result.

    Fields
    ------
    trace:
        Full ordered tuple of this invocation's child ``AtomicResult``
        objects, populated by the executing subclass's own ``_run``/
        ``_async_run`` when ``include_trace`` is enabled. ``None`` when
        trace collection is disabled, or when the executing subclass has
        not yet been updated to populate it.
    """

    trace: tuple[AtomicResult, ...] | None = field(default=None, kw_only=True)


@dataclass(frozen=True, slots=True)
class SequentialFlowResult(WorkflowResult):
    """Result for a SequentialFlow run.

    Fields
    ------
    return_index:
        The fixed step index whose result became this result's ``result``
        payload. When ``trace`` is populated (inherited from
        ``WorkflowResult``), ``trace[return_index]`` is that step's full
        ``AtomicResult``, including its own ``run_id``.
    """

    return_index: int


@dataclass(frozen=True, slots=True)
class RoutingFlowResult(WorkflowResult):
    """Result for a RoutingFlow run.

    Fields
    ------
    selected_branch:
        The router's validated selector -- the branch that was actually
        invoked. An ``int`` index into ``RoutingFlow.branches`` for
        list-configured topology, or the dict key for dict-configured
        topology. Always populated, independent of ``include_trace``.
    trace (inherited):
        Exactly two entries when ``include_trace`` is enabled: ``trace[0]``
        is the router's own ``AtomicResult``, ``trace[1]`` is the selected
        branch's own ``AtomicResult``. ``None`` when tracing is disabled.
    """

    selected_branch: Hashable


@dataclass(frozen=True, slots=True)
class IterativeFlowResult(WorkflowResult):
    """Result for an IterativeFlow run.

    Fields
    ------
    iteration_runs:
        Tuple of loop-body run ids, one per completed iteration, in
        iteration order. ``iteration_runs[i]`` can be used with
        ``loop_body.get_checkpoint(iteration_runs[i])`` to retrieve that
        iteration's body result.
    judge_runs:
        Tuple of judge run ids, one per completed iteration, in iteration
        order, parallel to ``iteration_runs``. ``judge_runs[i]`` can be used
        with ``judge.get_checkpoint(judge_runs[i])`` to retrieve that
        iteration's judge result (and, via a retrieval helper, the judge's
        decision for that iteration).
    return_step_index:
        Fixed loop-body step index whose result became this result's
        ``result`` payload.
    handoff_step_index:
        Fixed loop-body step index whose result became the next iteration's
        inputs.
    evaluate_step_index:
        Fixed loop-body step index whose result was passed to the judge.
    max_iterations:
        Iteration bound configured for this run.
    """

    iteration_runs: tuple[str, ...]
    judge_runs: tuple[str, ...]
    return_step_index: int
    handoff_step_index: int
    evaluate_step_index: int
    max_iterations: int


@dataclass(frozen=True, slots=True)
class ParallelFlowResult(WorkflowResult):
    """Result for a ParallelFlow run.

    Fields
    ------
    result_mode:
        The fixed output projection mode (``SCALAR``/``LIST``/``TUPLE``/
        ``SET``/``DICT``), duplicated from ``ParallelFlow.result_mode``.
    selected_indices:
        The fixed branch positions selected for projection, in projection
        order, duplicated from ``ParallelFlow.selected_indices``. Always
        ``tuple[int, ...]`` regardless of mode — may be empty (no branch
        selected). ``trace[i] for i in selected_indices`` is how to pull
        the actual per-branch results that fed ``result``.
    result_keys:
        Single source of truth for the projection's labels, duplicated
        from ``ParallelFlow.result_keys``. For ``DICT`` mode, the
        validated ``result_keys`` constructor value (``tuple[str, ...]``).
        For every other mode, exactly ``selected_indices``
        (``tuple[int, ...]``) — may be empty (no branch selected), never
        ``None``.
    """

    result_mode: str
    selected_indices: tuple[int, ...]
    result_keys: tuple[int, ...] | tuple[str, ...]
