from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

from .atomic import AtomicResult

__all__ = [
    "IterationRecord",
    "WorkflowResult",
    "BasicFlowResult",
    "SequentialFlowResult",
    "RoutingFlowResult",
    "IterativeWorkflowResult",
    "ParallelFlowResult",
]


# ------------------------------------------------------------------ #
# Value types (used as fields on WorkflowResult subclasses)
# ------------------------------------------------------------------ #

@dataclass(frozen=True, slots=True)
class IterationRecord:
    """Record of one completed iterative workflow iteration."""

    iteration: int
    body_run_id: str
    judge_run_id: str
    judge_decision: bool


# ------------------------------------------------------------------ #
# WorkflowResult hierarchy
# ------------------------------------------------------------------ #

@dataclass(frozen=True, slots=True)
class WorkflowResult(AtomicResult):
    """Base successful Workflow invocation result."""


@dataclass(frozen=True, slots=True)
class BasicFlowResult(WorkflowResult):
    """Result for a BasicFlow run.

    ``result`` holds the unwrapped child payload (``child_result.result``),
    not the child's AtomicResult envelope.

    Fields
    ------
    child_id:
        ``child_result.invoker_id`` — instance identifier of the wrapped
        component that executed.
    child_type:
        ``type(component).__name__`` — the wrapped component's class name.
    child_run_id:
        ``child_result.run_id`` — run identifier of the wrapped component's
        invocation, usable to correlate against the component's own history
        (e.g. ``Agent._history`` or a child ``Workflow``'s checkpoints), when
        the component retains one.
    """

    child_id: str
    child_type: str
    child_run_id: str


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
class IterativeWorkflowResult(WorkflowResult):
    """Result for an IterativeFlow run."""

    iterations_completed: int
    max_iterations: int
    judge_approved_early: bool
    return_step_index: int
    handoff_step_index: int
    evaluate_step_index: int
    iteration_records: tuple[IterationRecord, ...]


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
