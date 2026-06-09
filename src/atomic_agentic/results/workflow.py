from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .atomic import AtomicResult
from ..workflows.metadata import ChildRunRecord, IterationRecord, OutputTopology

__all__ = [
    "WorkflowResult",
    "BasicWorkflowResult",
    "SequentialWorkflowResult",
    "RoutingWorkflowResult",
    "IterativeWorkflowResult",
    "ParallelWorkflowResult",
]


@dataclass(frozen=True, slots=True)
class WorkflowResult(AtomicResult):
    """Base successful Workflow invocation result."""


@dataclass(frozen=True, slots=True)
class BasicWorkflowResult(WorkflowResult):
    """Result for a BasicFlow run.

    ``result`` holds the unwrapped child payload (``child_result.result``),
    not the child's AtomicResult envelope.
    """

    child_id: str
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
