from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..results.workflows import WorkflowResult

__all__ = ["WorkflowCheckpoint"]


@dataclass(frozen=True, slots=True)
class WorkflowCheckpoint:
    """A single workflow invocation record."""

    inputs: dict[str, Any]
    result: WorkflowResult
