"""Conversion layer above ``tools/``, ``mcp/`` and ``a2a/``: turns callables,
invokables and remote hubs into ``Tool`` / ``AtomicInvokable`` objects."""
from .invokification import to_invokables, to_invokable
from .toolification import to_tools, to_tool, tool

__all__ = ["to_tool", "to_tools", "tool", "to_invokable", "to_invokables"]
