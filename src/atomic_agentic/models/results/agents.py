from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .atomic import AtomicResult
from .llm import LLMModelData, TokenUsage

__all__ = [
    "ToolUsageRecord",
    "AgentResult",
    "ToolAgentResult",
]


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
    llm_token_usage:
        Per-call token usage for every LLM generation that contributed to this
        invocation, ordered by call order. Entries where the provider did not
        report usage are omitted. Empty tuple is valid.

    llm_model_data:
        Model identity associated with the LLM activity that produced this
        Agent result. Sourced from the last LLMRecord's engine model_data.
    """

    llm_token_usage: tuple[TokenUsage, ...]
    llm_model_data: LLMModelData

    def __post_init__(self) -> None:
        normalized = self._normalize_llm_token_usage(self.llm_token_usage)
        object.__setattr__(self, "llm_token_usage", normalized)

        if not isinstance(self.llm_model_data, LLMModelData):
            raise TypeError(
                "AgentResult.llm_model_data must be an LLMModelData instance, "
                f"got {type(self.llm_model_data).__name__}."
            )

        AtomicResult.__post_init__(self)

    @staticmethod
    def _normalize_llm_token_usage(value: Any) -> tuple[TokenUsage, ...]:
        """Validate and normalize the invocation's per-call token usage records."""
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise TypeError(
                "AgentResult.llm_token_usage must be a sequence of TokenUsage "
                f"instances, got {type(value).__name__}."
            )
        normalized = tuple(value)
        for index, entry in enumerate(normalized):
            if not isinstance(entry, TokenUsage):
                raise TypeError(
                    "AgentResult.llm_token_usage must contain only TokenUsage instances; "
                    f"item {index} is {type(entry).__name__}."
                )
        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        data = AtomicResult.to_dict(self)
        data.update(
            {
                "llm_token_usage": [u.to_dict() for u in self.llm_token_usage],
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
