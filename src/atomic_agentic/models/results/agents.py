from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .atomic import AtomicResult
from .llm import LLMModelData, LLMResult

__all__ = [
    "LLMRecord",
    "ToolUsageRecord",
    "AgentResult",
    "ToolAgentResult",
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
class ToolUsageRecord:
    """
    Aggregate usage record for one tool across one ToolAgent invocation.

    Fields
    ------
    tool_name:
        Full registered tool name (e.g. ``"Tool.math.add"``).
    call_count:
        Number of non-return executions of this tool during the invocation.
        Always >= 1 for any entry present in a ToolAgentResult.
    """

    tool_name: str
    call_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise TypeError(
                "ToolUsageRecord.tool_name must be a non-empty str, "
                f"got {self.tool_name!r}."
            )
        if isinstance(self.call_count, bool) or not isinstance(self.call_count, int) or self.call_count < 1:
            raise ValueError(
                "ToolUsageRecord.call_count must be a positive int (>= 1), "
                f"got {self.call_count!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "tool_name": self.tool_name,
            "call_count": self.call_count,
        }


@dataclass(frozen=True, slots=True)
class AgentResult(AtomicResult):
    """
    Successful Agent invocation result.

    ``AgentResult.result`` is the final caller-facing payload produced by the
    Agent after post-invoke processing has completed.

    Fields
    ------
    llm_records:
        Complete record of every LLM generation that contributed to this
        invocation. Non-empty; stored as an immutable tuple after construction.
        Carries full token usage, model identity, and timing per generation.

    llm_model_data:
        Model identity associated with the LLM activity that produced this
        Agent result. Sourced from the last LLMRecord's engine model_data.
    """

    llm_records: tuple[LLMRecord, ...]
    llm_model_data: LLMModelData

    def __post_init__(self) -> None:
        normalized_llm_records = self._normalize_llm_records(self.llm_records)

        if not isinstance(self.llm_model_data, LLMModelData):
            raise TypeError(
                "AgentResult.llm_model_data must be an LLMModelData instance, "
                f"got {type(self.llm_model_data).__name__}."
            )

        object.__setattr__(self, "llm_records", normalized_llm_records)
        AtomicResult.__post_init__(self)

    @staticmethod
    def _normalize_llm_records(value: Any) -> tuple[LLMRecord, ...]:
        """Validate and normalize the invocation's LLM generation records."""
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError(
                "AgentResult.llm_records must be a non-empty sequence of "
                f"LLMRecord instances, got {type(value).__name__}."
            )

        normalized = tuple(value)
        if not normalized:
            raise ValueError("AgentResult.llm_records must not be empty.")

        for index, record in enumerate(normalized):
            if not isinstance(record, LLMRecord):
                raise TypeError(
                    "AgentResult.llm_records must contain only LLMRecord instances; "
                    f"item {index} is {type(record).__name__}."
                )

        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        data = AtomicResult.to_dict(self)
        data.update(
            {
                "llm_records": [r.to_dict() for r in self.llm_records],
                "llm_model_data": self.llm_model_data.to_dict(),
            }
        )
        return data


@dataclass(frozen=True, slots=True)
class ToolAgentResult(AgentResult):
    """
    Successful ToolAgent invocation result.

    Extends ``AgentResult`` with per-tool call-count accounting derived
    from the invocation's execution loop.

    Fields
    ------
    tool_usage:
        Ordered tuple of per-tool usage records, ordered by first-call order
        within the invocation. May be empty if no non-return tools executed.
    """

    tool_usage: tuple[ToolUsageRecord, ...]

    def __post_init__(self) -> None:
        normalized_tool_usage = self._normalize_tool_usage(self.tool_usage)
        object.__setattr__(self, "tool_usage", normalized_tool_usage)
        AgentResult.__post_init__(self)

    @staticmethod
    def _normalize_tool_usage(value: Any) -> tuple[ToolUsageRecord, ...]:
        """Validate and normalize the invocation's tool usage records."""
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError(
                "ToolAgentResult.tool_usage must be a sequence of ToolUsageRecord "
                f"instances, got {type(value).__name__}."
            )

        normalized = tuple(value)

        for index, record in enumerate(normalized):
            if not isinstance(record, ToolUsageRecord):
                raise TypeError(
                    "ToolAgentResult.tool_usage must contain only ToolUsageRecord "
                    f"instances; item {index} is {type(record).__name__}."
                )

        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        data = AgentResult.to_dict(self)
        data["tool_usage"] = [r.to_dict() for r in self.tool_usage]
        return data
