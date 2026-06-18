from __future__ import annotations

__all__ = ["create_fallback_judge_tool"]


def _always_false() -> bool:
    """Fallback iterative judge implementation."""
    return False


def create_fallback_judge_tool():
    """Create a shared fallback judge tool that always returns False."""
    from ..tools import Tool
    return Tool(
        function=_always_false,
        name="always_false_judge",
        namespace="workflow",
        description="Fallback iterative judge that always returns False.",
    )
