"""Conversion layer above ``tools/``, ``mcp/``, ``a2a/`` and ``llm/``: turns
callables, invokables and remote hubs into ``Tool`` / ``AtomicInvokable``
objects, and SDK clients into ``LLMEngine`` objects."""
from .llmification import to_llm
from .toolification import to_tools, to_tool

__all__ = ["to_tool", "to_tools", "to_llm"]
