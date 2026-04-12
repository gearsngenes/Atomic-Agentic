from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar, ClassVar
from ..core.sentinels import NO_VAL

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
    """Base class for typed workflow checkpoint metadata.

    Every concrete workflow metadata record should inherit from this class and
    provide a stable ``kind`` discriminator.
    """

    kind: str


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
    """Typed metadata for a basic wrapper workflow run.

    Interpretation rules
    --------------------
    - If ``child_is_workflow`` is ``True``:
        - ``child_run_id`` should contain the child workflow run id
        - ``child_raw_result`` should be ``NO_VAL``
        - ``has_child_raw_result`` should be ``False``
    - If ``child_is_workflow`` is ``False``:
        - ``child_run_id`` should be ``NO_VAL``
        - ``child_raw_result`` should contain the wrapped structured child's raw result
        - ``has_child_raw_result`` should be ``True``

    ``child_raw_result_type`` should describe the meaningful child-level result
    payload, not merely the outer wrapper carrier.
    """

    child_is_workflow: bool
    child_id: str
    child_full_name: str
    child_run_id: str | Any = NO_VAL
    child_raw_result: Any = NO_VAL
    has_child_raw_result: bool = False
    child_raw_result_type: str = "Any"
    kind: str = field(default="basic", init=False)


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