from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..results.agents import AgentResult
from ..results.llm import LLMResult

__all__ = [
    "LLMRecord",
    "AgentRecord",
    "ToolAgentRecord",
]


@dataclass(frozen=True, slots=True)
class LLMRecord:
    """
    Canonical memory record for one completed LLM generation made during an
    Agent invocation.

    An Agent invocation may involve one or more LLM generations (e.g. a
    ToolAgent's planning loop). Each generation that contributes to an
    invocation is preserved here in full — not distilled — so that future
    rendering, debugging, and accounting needs are not constrained by what an
    earlier pass chose to keep.

    Fields
    ------
    user_prompt:
        User-facing prompt text that triggered this specific LLM generation.
        May differ from the AgentRecord's overall ``user_prompt`` when an
        invocation makes multiple LLM calls with distinct prompts.

    llm_result:
        The complete LLMResult produced by this generation, including its
        token usage, model identity, timing, and run identity.
    """

    user_prompt: str
    llm_result: LLMResult

    def __post_init__(self) -> None:
        if not isinstance(self.user_prompt, str):
            raise TypeError(
                f"LLMRecord.user_prompt must be a str, got {type(self.user_prompt).__name__}."
            )
        if not isinstance(self.llm_result, LLMResult):
            raise TypeError(
                "LLMRecord.llm_result must be an LLMResult instance, "
                f"got {type(self.llm_result).__name__}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "user_prompt": self.user_prompt,
            "llm_result": self.llm_result.to_dict(),
        }


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

    llm_records:
        Complete record of every LLM generation that contributed to this
        invocation. Empty tuple during the draft phase; populated in the
        completion ``replace`` step after ``make_result`` runs.
    """

    user_prompt: str
    generated_response: Any
    final_result: AgentResult | None = None
    llm_records: tuple[LLMRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.user_prompt, str):
            raise TypeError(
                f"AgentRecord.user_prompt must be a str, got {type(self.user_prompt).__name__}."
            )

        if not isinstance(self.llm_records, (tuple, list)) or isinstance(self.llm_records, (str, bytes)):
            raise TypeError(
                "AgentRecord.llm_records must be a sequence of LLMRecord instances, "
                f"got {type(self.llm_records).__name__}."
            )
        normalized = tuple(self.llm_records)
        for index, record in enumerate(normalized):
            if not isinstance(record, LLMRecord):
                raise TypeError(
                    "AgentRecord.llm_records must contain only LLMRecord instances; "
                    f"item {index} is {type(record).__name__}."
                )
        object.__setattr__(self, "llm_records", normalized)

    def to_dict(self) -> Dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "user_prompt": self.user_prompt,
            "generated_response": self.generated_response,
            "final_result": self.final_result.to_dict() if self.final_result is not None else None,
            "llm_records": [r.to_dict() for r in self.llm_records],
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
