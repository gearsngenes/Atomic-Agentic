from __future__ import annotations

from dataclasses import dataclass

from .atomic import AtomicResult

__all__ = ["ToolResult"]


@dataclass(frozen=True, slots=True)
class ToolResult(AtomicResult):
    """
    Successful Tool invocation result.

    ``ToolResult.result`` is the caller-facing payload produced by the Tool.
    This subclass intentionally adds no Tool-specific fields yet.
    """
