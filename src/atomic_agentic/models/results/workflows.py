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
    step_runs:
        Tuple of child run ids, one per executed step, in step order.
        ``step_runs[i]`` corresponds to ``SequentialFlow.steps[i]`` and can
        be used with ``steps[i].get_checkpoint(step_runs[i])`` to retrieve
        that step's own checkpoint.
    return_index:
        The fixed step index whose result became this result's ``result``
        payload (i.e. ``result == steps[return_index]``'s checkpoint result).
    """

    step_runs: tuple[str, ...]
    return_index: int


@dataclass(frozen=True, slots=True)
class RoutingFlowResult(WorkflowResult):
    """Result for a RoutingFlow run.

    Fields
    ------
    selected_key:
        The validated selector returned by the router that determined which
        branch ran. For list-configured branches, an ``int`` index into
        ``RoutingFlow.branches``. For dict-configured branches, the dict key
        used to look up the executed branch.
    chosen_branch_run:
        Run id of the selected branch's invocation. Used with
        ``branches[selected_key].get_checkpoint(chosen_branch_run)`` to
        retrieve that branch's own checkpoint.
    router_run_id:
        Run id of the router's invocation. Use with
        ``router.get_checkpoint(router_run_id)`` to retrieve the router's
        own checkpoint.
    """

    selected_key: Hashable
    chosen_branch_run: str
    router_run_id: str


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
    branch_runs:
        Tuple of child run ids, one per executed branch, in branch order.
        ``branch_runs[i]`` corresponds to ``ParallelFlow.branches[i]`` and can
        be used with ``branches[i].get_checkpoint(branch_runs[i])`` to
        retrieve that branch's own checkpoint.
    output_indices:
        Tuple of branch indices, in projection order, whose payloads were
        combined into ``result``. ``output_indices[k]`` indexes into
        ``branch_runs``/``ParallelFlow.branches``. For ``output_type``
        ``"list"``/``"tuple"``, this is the order of ``result``'s elements
        (``result[k]`` came from branch ``output_indices[k]``).
    """

    branch_runs: tuple[str, ...]
    output_indices: tuple[int, ...]
