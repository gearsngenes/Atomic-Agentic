from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..results.agents import AgentResult

__all__ = [
    "AgentRecord",
    "ToolAgentRecord",
]


@dataclass(frozen=True, slots=True)
class AgentRecord:
    """
    Canonical memory record for one completed Agent invocation.

    A record stores the lifecycle artifacts needed to reconstruct future
    LLM-facing context. It is related to, but distinct from, AgentResult:
    AgentResult is the public successful-invocation envelope carrying LLM
    accounting; AgentRecord is the memory/rendering record. The completed
    record points to its AgentResult via ``final_result``.

    Fields
    ------
    user_prompt:
        Prompt string produced by ``pre_invoke`` for this invocation.

    generated_response:
        Raw post-engine response material for this invocation, prior to
        ``post_invoke`` processing.

    final_result:
        The completed ``AgentResult`` for this invocation. ``None`` during
        the draft phase (between ``_invoke`` return and ``make_result``
        completion); an ``AgentResult`` instance on all stored records.
    """

    user_prompt: str
    generated_response: Any
    final_result: AgentResult | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.user_prompt, str):
            raise TypeError(
                f"AgentRecord.user_prompt must be a str, got {type(self.user_prompt).__name__}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "user_prompt": self.user_prompt,
            "generated_response": self.generated_response,
            "final_result": self.final_result.to_dict() if self.final_result is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ToolAgentRecord(AgentRecord):
    """
    Canonical memory record for one completed ToolAgent invocation.

    In addition to the base AgentRecord lifecycle artifacts, a ToolAgentRecord
    stores the half-open span of persisted blackboard entries produced by
    the invocation. The ToolAgent renders that span into future LLM-facing
    context when building messages.
    """

    blackboard_start: int | None = None
    blackboard_end: int | None = None
