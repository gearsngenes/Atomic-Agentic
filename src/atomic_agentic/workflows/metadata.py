from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar, ClassVar
from ..core.constants import NO_VAL

__all__ = [
    "WorkflowRunMetadata",
    "ChildRunRecord",
    "OutputTopology",
    "IterationRecord",
    "BasicFlowRunMetadata",
    "SequentialFlowRunMetadata",
    "RoutingFlowRunMetadata",
    "IterativeFlowRunMetadata",
    "ParallelFlowRunMetadata",
    "WorkflowCheckpoint",
]


@dataclass(frozen=True, slots=True)
class WorkflowRunMetadata:
    """Marker base class for typed workflow checkpoint metadata.

    Concrete workflow metadata records may define their own discriminator fields
    when useful, but the base workflow runtime only requires metadata objects to
    be instances of this base class.
    """


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


@dataclass(frozen=True, slots=True)
class BasicFlowRunMetadata(WorkflowRunMetadata):
    """Typed metadata for a BasicFlow wrapper run.

    BasicFlow records only the delegated child's identity and, when that child is
    itself a Workflow, the child workflow run id that can be used to inspect the
    child's own checkpoint history.

    For non-workflow AtomicInvokable children, ``child_run_id`` is ``NO_VAL``.
    A later AtomicResult integration can make run ids universal across all
    AtomicInvokable children.
    """

    child_is_workflow: bool
    child_id: str
    child_run_id: str | Any = NO_VAL


@dataclass(frozen=True, slots=True)
class SequentialFlowRunMetadata(WorkflowRunMetadata):
    """Typed metadata for a sequential workflow run."""

    step_records: tuple[ChildRunRecord, ...]
    return_child_index: int
    return_child_run_id: str
    kind: str = field(default="sequential", init=False)


@dataclass(frozen=True, slots=True)
class RoutingFlowRunMetadata(WorkflowRunMetadata):
    """Typed metadata for a routing workflow run."""

    router_run_id: str
    router_instance_id: str
    chosen_index: int
    chosen_branch_record: ChildRunRecord
    kind: str = field(default="routing", init=False)


@dataclass(frozen=True, slots=True)
class IterativeFlowRunMetadata(WorkflowRunMetadata):
    """Typed metadata for an iterative workflow run."""

    iterations_completed: int
    max_iterations: int
    judge_approved_early: bool
    return_step_index: int
    handoff_step_index: int
    evaluate_step_index: int
    iteration_records: tuple[IterationRecord, ...]
    kind: str = field(default="iterative", init=False)


@dataclass(frozen=True, slots=True)
class ParallelFlowRunMetadata(WorkflowRunMetadata):
    """Typed metadata for a parallel workflow run."""
    branch_records: tuple[ChildRunRecord, ...]
    output_topology: OutputTopology
    output_count: int
    kind: str = field(default="parallel", init=False)


M = TypeVar("M", bound=WorkflowRunMetadata)


@dataclass(frozen=True, slots=True)
class WorkflowCheckpoint(Generic[M]):
    """A single typed workflow invocation record."""

    run_id: str
    started_at: datetime
    ended_at: datetime
    elapsed_s: float
    inputs: dict[str, Any]
    result: dict[str, Any]
    metadata: M