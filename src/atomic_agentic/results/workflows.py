from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from .atomic import AtomicResult

__all__ = [
    "ChildRunRecord",
    "OutputTopology",
    "IterationRecord",
    "WorkflowResult",
    "BasicFlowResult",
    "SequentialWorkflowResult",
    "RoutingWorkflowResult",
    "IterativeWorkflowResult",
    "ParallelWorkflowResult",
]


# ------------------------------------------------------------------ #
# Value types (used as fields on WorkflowResult subclasses)
# ------------------------------------------------------------------ #

@dataclass(frozen=True, slots=True)
class ChildRunRecord:
    """Record of one executed child workflow-shaped node.

    Fields
    ------
    slot:
        Zero-based position in the owning workflow's configured child topology.
    instance_id:
        Stable instance identifier of the child node that executed.
    full_name:
        Human-readable runtime identity of the child node.
    run_id:
        Run identifier emitted by the child node for this execution.
    """

    slot: int
    instance_id: str
    full_name: str
    run_id: str


@dataclass(frozen=True, slots=True)
class OutputTopology:
    """Resolved output projection description for a parallel workflow run.

    Fields
    ------
    topology:
        Effective outward arrangement mode. Expected current values are
        typically ``"nested"`` or ``"flattened"``.
    indices:
        Ordered resolved child indices included in the outward projection.
    names:
        Output names used for nested projection, or ``None`` for flattened
        projection.
    duplicate_key_policy:
        Duplicate-key behavior used for flattened projection, or ``None`` when
        not applicable.
    """

    NESTED: ClassVar[str] = "nested"
    FLATTENED: ClassVar[str] = "flattened"

    topology: str
    indices: tuple[int, ...]
    names: tuple[str, ...] | None = None
    duplicate_key_policy: str | None = None


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
class SequentialWorkflowResult(WorkflowResult):
    """Result for a SequentialFlow run."""

    step_records: tuple[ChildRunRecord, ...]
    return_child_run_id: str


@dataclass(frozen=True, slots=True)
class RoutingWorkflowResult(WorkflowResult):
    """Result for a RoutingFlow run."""

    chosen_index: int
    chosen_branch_record: ChildRunRecord
    router_run_id: str
    router_instance_id: str


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
class ParallelWorkflowResult(WorkflowResult):
    """Result for a ParallelFlow run."""

    branch_records: tuple[ChildRunRecord, ...]
    output_topology: OutputTopology
    output_count: int
