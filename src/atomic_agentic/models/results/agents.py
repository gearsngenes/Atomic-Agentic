from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .atomic import AtomicResult
from .llm import LLMModelData, TokenUsage

__all__ = [
    "ToolUsageRecord",
    "ToolUsageReport",
    "AgentResult",
    "ToolAgentResult",
    "ScriptActAgentResult",
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
class ToolUsageReport:
    """
    Per-invocation tool-usage report, shared identically by every concrete
    ToolAgent (ScriptActAgent/PlanActAgent/ReActAgent).

    Fields
    ------
    by_tool:
        One entry per distinct real tool identity actually dispatched,
        ordered by first-call order, ``call_count >= 1`` each.

    total_dispatched:
        Sum of every ``by_tool`` entry's ``call_count`` -- every dispatched
        call this run, successful or failed, across every tool identity.

    total_failed:
        Count of dispatched calls that actually raised this run (the subset
        of ``total_dispatched`` that failed). ``total_dispatched -
        total_failed`` is the successful-dispatch count.
    """

    by_tool: tuple[ToolUsageRecord, ...]
    total_dispatched: int
    total_failed: int

    def __post_init__(self) -> None:
        if not isinstance(self.by_tool, Sequence) or isinstance(self.by_tool, (str, bytes, bytearray)):
            raise TypeError(
                "ToolUsageReport.by_tool must be a sequence of ToolUsageRecord "
                f"instances, got {type(self.by_tool).__name__}."
            )

        normalized = tuple(self.by_tool)

        for index, record in enumerate(normalized):
            if not isinstance(record, ToolUsageRecord):
                raise TypeError(
                    "ToolUsageReport.by_tool must contain only ToolUsageRecord "
                    f"instances; item {index} is {type(record).__name__}."
                )

        object.__setattr__(self, "by_tool", normalized)

        if isinstance(self.total_dispatched, bool) or not isinstance(self.total_dispatched, int) or self.total_dispatched < 0:
            raise ValueError(
                "ToolUsageReport.total_dispatched must be a non-negative int, "
                f"got {self.total_dispatched!r}."
            )
        if isinstance(self.total_failed, bool) or not isinstance(self.total_failed, int) or self.total_failed < 0:
            raise ValueError(
                "ToolUsageReport.total_failed must be a non-negative int, "
                f"got {self.total_failed!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "by_tool": [r.to_dict() for r in self.by_tool],
            "total_dispatched": self.total_dispatched,
            "total_failed": self.total_failed,
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
    Successful ToolAgent invocation result. Shared directly by
    ``PlanActAgent``/``ReActAgent``, and the base class of
    ``ScriptActAgentResult`` too.

    Extends ``AgentResult`` with per-tool call-count accounting and a
    lightweight failure summary when the agent ran with ``fail_fast=False``.

    Fields
    ------
    usage_report:
        ``ToolUsageReport`` summarizing per-tool call counts plus
        ``total_dispatched``/``total_failed`` for the invocation. Re-derived
        (by the caller, e.g. ``PlanActAgent.build_result_from_record``) from
        ``record.usage_report()`` rather than a blackboard span.

    failed_call_count:
        Count of calls whose dispatch actually raised this run -- a cheap
        "did anything fail, how much" summary only; the rich per-failure
        detail (identifier, tool, args, the actual exception) lives on
        ``record.failed_statements`` instead. ``0`` when ``fail_fast=True``
        (failures raise immediately) or when nothing failed.

    regenerations_used:
        Threaded from ``record.regenerations_used`` verbatim.
    """

    usage_report: ToolUsageReport
    failed_call_count: int = 0
    regenerations_used: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.usage_report, ToolUsageReport):
            raise TypeError(
                "ToolAgentResult.usage_report must be a ToolUsageReport instance, "
                f"got {type(self.usage_report).__name__}."
            )
        AgentResult.__post_init__(self)

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        data = AgentResult.to_dict(self)
        data["usage_report"] = self.usage_report.to_dict()
        data["failed_call_count"] = self.failed_call_count
        data["regenerations_used"] = self.regenerations_used
        return data


@dataclass(frozen=True, slots=True)
class ScriptActAgentResult(ToolAgentResult):
    """
    Successful ScriptActAgent invocation result, a ``ToolAgentResult``
    subclass. ``usage_report``/``failed_call_count``/``regenerations_used``
    are all real, inherited, required fields -- ``agents/scriptact.py``'s
    ``build_result_from_record`` populates ``usage_report``/
    ``failed_call_count`` via ``record.usage_report()``/
    ``len(record.failed_statements)``, mirroring ``PlanActAgent``'s/
    ``ReActAgent``'s own pattern. Only ``repair_rounds_used`` remains
    genuinely ``ScriptActAgent``-specific.

    Fields
    ------
    repair_rounds_used:
        Threaded from ``record.repair_rounds_used`` verbatim -- framework-
        granted repair rounds actually consumed after a resolution or
        execution failure (see ``ScriptActAgent``'s ``replanning_limit``).
        ``0`` means the plan finished without ever needing one.
    """

    repair_rounds_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation --
        extends the inherited ``ToolAgentResult.to_dict()`` (already
        includes ``usage_report``/``failed_call_count``/``regenerations_used``)
        with just ``repair_rounds_used``."""
        # Explicit two-argument super() -- @dataclass(slots=True) rebuilds
        # the class object to add __slots__, which invalidates the
        # zero-arg super()'s implicit __class__ closure cell (the same
        # slotted-dataclass-subclass gotcha records.py's own __post_init__
        # methods already document and work around).
        data = super(ScriptActAgentResult, self).to_dict()
        data["repair_rounds_used"] = self.repair_rounds_used
        return data


@dataclass(frozen=True, slots=True)
class ThinkingAgentResult(AgentResult):
    """
    Successful thinking-capable agent invocation result (currently only
    ``ThinkingAgent``).

    Extends ``AgentResult`` with a derived count of completed thinking
    rounds -- mirrors how ``llm_token_usage`` is already a derived
    reduction of ``record.llm_records``, not a duplicate of it. A caller
    needing the actual thought content reads it off the record instead
    (``agent.get_conversation(turns=1)[0].thoughts`` for the invocation
    just made; scan ``get_conversation(conversation_id, turns=None)`` and
    match ``r.final_result.run_id`` for an older one).

    No ``__post_init__`` override needed -- plain ``int`` field, no
    cross-field validation, matching this class's existing precedent (the
    value is always internally computed as ``len(record.thoughts)``, never
    caller-supplied at a real boundary).

    Fields
    ------
    thinking_rounds_used:
        Count of thinking rounds completed during this invocation.
    """

    thinking_rounds_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        data = AgentResult.to_dict(self)
        data["thinking_rounds_used"] = self.thinking_rounds_used
        return data
