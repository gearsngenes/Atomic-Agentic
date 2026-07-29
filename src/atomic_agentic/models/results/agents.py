from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .atomic import AtomicResult
from .llm import LLMModelData, TokenUsage

__all__ = [
    "ToolUsageRecord",
    "AgentResult",
    "ToolAgentResult",
    "ThinkingAgentResult",
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

    thoughts_start, thoughts_end:
        Mirrors ``AgentRecord``'s own fields -- carried through by
        ``Agent2.build_result_from_record``. ``kw_only=True`` so these two
        optional trailing fields don't force every subclass's own
        non-default fields (e.g. ``ToolAgentResult.tool_usage``) to also
        gain a default -- keyword-only fields are excluded from
        dataclasses' positional default-ordering check entirely.
    """

    llm_token_usage: tuple[TokenUsage, ...]
    llm_model_data: LLMModelData
    thoughts_start: int | None = field(default=None, kw_only=True)
    thoughts_end: int | None = field(default=None, kw_only=True)

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
                "thoughts_start": self.thoughts_start,
                "thoughts_end": self.thoughts_end,
            }
        )
        return data


@dataclass(frozen=True, slots=True)
class ToolAgentResult(AgentResult):
    """
    Successful ToolAgent invocation result.

    Extends ``AgentResult`` with per-tool call-count accounting and optional
    partial-failure records when the agent ran with ``fail_fast=False``.

    Fields
    ------
    tool_usage:
        Ordered tuple of per-tool usage records, ordered by first-call order
        within the invocation. May be empty if no non-return tools executed.

    exception_records:
        Tuple of ``(global_blackboard_index, exception)`` pairs for every
        tool slot that failed during the run. Empty when ``fail_fast=True``
        (failures raise immediately) or when all tools succeeded.
        The index is the absolute position of the failed slot in the agent's
        persistent blackboard after ``update_blackboard`` has run.
    """

    tool_usage: tuple[ToolUsageRecord, ...]
    exception_records: tuple[tuple[int, Exception], ...] = ()

    def __post_init__(self) -> None:
        normalized_tool_usage = self._normalize_tool_usage(self.tool_usage)
        object.__setattr__(self, "tool_usage", normalized_tool_usage)
        normalized_exc = self._normalize_exception_records(self.exception_records)
        object.__setattr__(self, "exception_records", normalized_exc)
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

    @staticmethod
    def _normalize_exception_records(
        value: Any,
    ) -> tuple[tuple[int, Exception], ...]:
        """Validate and normalize the invocation's per-slot failure records."""
        if not isinstance(value, (tuple, list)):
            raise TypeError(
                "ToolAgentResult.exception_records must be a sequence of "
                f"(int, Exception) tuples, got {type(value).__name__}."
            )
        normalized = tuple(value)
        for i, item in enumerate(normalized):
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], int)
                or not isinstance(item[1], Exception)
            ):
                raise TypeError(
                    "ToolAgentResult.exception_records items must be "
                    f"(int, Exception) tuples; item {i} is invalid."
                )
        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        data = AgentResult.to_dict(self)
        data["tool_usage"] = [r.to_dict() for r in self.tool_usage]
        data["exception_records"] = [
            {"blackboard_index": idx, "error": str(e)}
            for idx, e in self.exception_records
        ]
        return data


@dataclass(frozen=True, slots=True)
class ThinkingAgentResult(AgentResult):
    """
    Successful ThinkingAgent invocation result.

    Extends ``AgentResult`` with the half-open index span into the agent's
    persisted thoughts list produced by this invocation. Indices only, not
    the thought content itself -- mirrors ``ThinkingAgentRecord``'s own
    ``thoughts_start``/``thoughts_end`` exactly (no ``__post_init__``
    override needed here, matching that precedent: plain ``int | None``
    fields, no cross-field validation). A caller needing the actual
    ``AgentThought`` content reads
    ``agent._thoughts[thoughts_start:thoughts_end]`` directly.

    Fields
    ------
    thoughts_start:
        Start index (inclusive) of this invocation's thoughts in the
        agent's persisted ``self._thoughts`` list.
    thoughts_end:
        End index (exclusive) of this invocation's thoughts in the agent's
        persisted ``self._thoughts`` list.
    """

    thoughts_start: int | None = None
    thoughts_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        data = AgentResult.to_dict(self)
        data["thoughts_start"] = self.thoughts_start
        data["thoughts_end"] = self.thoughts_end
        return data
