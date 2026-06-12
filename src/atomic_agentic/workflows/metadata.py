from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..results.workflows import WorkflowResult

__all__ = [
    "IterationRecord",
    "IterativeFlowRunMetadata",
    "WorkflowCheckpoint",
]


@dataclass(frozen=True, slots=True)
class IterationRecord:
    """Record of one completed iterative workflow iteration."""

    iteration: int
    body_run_id: str
    judge_run_id: str
    judge_decision: bool


@dataclass(frozen=True, slots=True)
class IterativeFlowRunMetadata:
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
class WorkflowCheckpoint:
    """A single workflow invocation record."""

    inputs: dict[str, Any]
    result: WorkflowResult
