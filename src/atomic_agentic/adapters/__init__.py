"""Conversion layer above ``tools/``, ``mcp/`` and ``a2a/``: turns callables,
invokables and remote hubs into ``Tool`` / ``AtomicInvokable`` objects."""
from .invokification import batch_to_invokable, to_invokable
from .toolification import batch_toolify, toolify

__all__ = ["toolify", "batch_toolify", "to_invokable", "batch_to_invokable"]
