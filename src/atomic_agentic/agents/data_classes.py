from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

__all__ = ["AgentTurn", "ToolAgentTurn"]


@dataclass(slots=True)
class AgentTurn:
    """
    Canonical memory record for one completed Agent invocation.

    A turn stores the important lifecycle artifacts needed to reconstruct
    future LLM-facing context without storing provider-facing message dicts
    as the canonical memory format.
    """
    prompt: str
    raw_response: Any
    final_response: Any

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow dictionary representation of this turn."""
        return asdict(self)

@dataclass(slots=True)
class ToolAgentTurn(AgentTurn):
    """
    Canonical memory record for one completed ToolAgent invocation.

    In addition to the base AgentTurn lifecycle artifacts, a ToolAgentTurn
    stores the half-open span of persisted blackboard entries produced by
    the invocation. The ToolAgent renders that span into future LLM-facing
    context when building messages.
    """
    blackboard_start: int | None = None
    blackboard_end: int | None = None
