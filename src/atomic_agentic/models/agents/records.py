from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict

from ..results import LLMResult

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

    def to_dict(self) -> Dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "user_prompt": self.user_prompt,
            "llm_result": self.llm_result.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AgentRecord:
    """
    Canonical memory record for one completed Agent invocation.

    A record stores the important lifecycle artifacts needed to reconstruct
    future LLM-facing context without storing provider-facing message dicts
    as the canonical memory format. It is related to, but distinct from,
    AgentResult: AgentResult is the public successful-invocation envelope,
    while AgentRecord is the memory/rendering record. Neither wraps the
    other; ``run_id`` links a record to the AgentResult of the same
    invocation.

    Fields
    ------
    user_prompt:
        Prompt string produced by ``pre_invoke`` for this invocation.

    generated_response:
        Raw post-engine response material for this invocation, prior to
        ``post_invoke`` processing.

    final_response:
        Final Agent output for this invocation, after ``post_invoke``
        processing has completed.

    llm_records:
        Non-empty record of every LLM generation that contributed to this
        invocation. Stored as an immutable tuple after construction.

    run_id:
        Identifier linking this record to the AgentResult produced by the
        same invocation.
    """
    user_prompt: str
    generated_response: Any
    final_response: Any
    llm_records: tuple[LLMRecord, ...]
    run_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.user_prompt, str):
            raise TypeError(
                f"AgentRecord.user_prompt must be a str, got {type(self.user_prompt).__name__}."
            )

        normalized_llm_records = self._normalize_llm_records(self.llm_records)
        normalized_run_id = self._normalize_run_id(self.run_id)

        object.__setattr__(self, "llm_records", normalized_llm_records)
        object.__setattr__(self, "run_id", normalized_run_id)

    @staticmethod
    def _normalize_llm_records(value: Any) -> tuple[LLMRecord, ...]:
        """Validate and normalize the invocation's LLM generation records."""
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError(
                "AgentRecord.llm_records must be a non-empty sequence of "
                f"LLMRecord instances, got {type(value).__name__}."
            )

        normalized = tuple(value)
        if not normalized:
            raise ValueError("AgentRecord.llm_records must not be empty.")

        for index, record in enumerate(normalized):
            if not isinstance(record, LLMRecord):
                raise TypeError(
                    "AgentRecord.llm_records must contain only LLMRecord instances; "
                    f"item {index} is {type(record).__name__}."
                )

        return normalized

    @staticmethod
    def _normalize_run_id(value: Any) -> str:
        """Validate and normalize the linked AgentResult run identifier."""
        if not isinstance(value, str):
            raise TypeError(
                f"AgentRecord.run_id must be a str, got {type(value).__name__}."
            )

        normalized = value.strip()
        if not normalized:
            raise ValueError("AgentRecord.run_id must be a non-empty string.")

        return normalized

    def to_dict(self) -> Dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "user_prompt": self.user_prompt,
            "generated_response": self.generated_response,
            "final_response": self.final_response,
            "llm_records": [record.to_dict() for record in self.llm_records],
            "run_id": self.run_id,
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
