from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from atomic_agentic.models.results.agents import (
    AgentResult,
    ThinkingAgentResult,
    ToolAgentResult,
    ToolUsageRecord,
    ToolUsageReport,
)
from atomic_agentic.models.results.llm import LLMModelData, LLMResult, TokenUsage


# ── helpers ───────────────────────────────────────────────────────────────────

def make_token_usage(*, input_tokens: int = 10, generated_tokens: int = 5) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        total_tokens=input_tokens + generated_tokens,
        response_tokens=generated_tokens,
    )


def make_model_data(*, provider: str = "openai") -> LLMModelData:
    return LLMModelData(provider=provider)


def make_agent_result(*, value: Any = "output") -> AgentResult:
    started_at = datetime.now(timezone.utc)
    return AgentResult(
        result=value,
        invoker_id="agent-1",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=1),
        llm_token_usage=(make_token_usage(),),
        llm_model_data=make_model_data(),
    )


# ── TestToolUsageRecord ───────────────────────────────────────────────────────

class TestToolUsageRecord:
    def test_valid_record_stores_name_and_count(self) -> None:
        rec = ToolUsageRecord(tool_name="Tool.math.add", call_count=3)
        assert rec.tool_name == "Tool.math.add"
        assert rec.call_count == 3

    def test_to_dict(self) -> None:
        rec = ToolUsageRecord(tool_name="Tool.math.add", call_count=2)
        assert rec.to_dict() == {"tool_name": "Tool.math.add", "call_count": 2}

    def test_rejects_empty_tool_name(self) -> None:
        with pytest.raises(TypeError, match="tool_name"):
            ToolUsageRecord(tool_name="", call_count=1)

    def test_rejects_non_string_tool_name(self) -> None:
        with pytest.raises(TypeError, match="tool_name"):
            ToolUsageRecord(tool_name=123, call_count=1)  # type: ignore[arg-type]

    def test_rejects_zero_call_count(self) -> None:
        with pytest.raises(ValueError, match="call_count"):
            ToolUsageRecord(tool_name="Tool.x", call_count=0)

    def test_rejects_negative_call_count(self) -> None:
        with pytest.raises(ValueError, match="call_count"):
            ToolUsageRecord(tool_name="Tool.x", call_count=-1)

    def test_rejects_bool_call_count(self) -> None:
        with pytest.raises(ValueError, match="call_count"):
            ToolUsageRecord(tool_name="Tool.x", call_count=True)  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        rec = ToolUsageRecord(tool_name="Tool.x", call_count=1)
        with pytest.raises(FrozenInstanceError):
            rec.call_count = 2  # type: ignore[misc]


# ── TestAgentResult ───────────────────────────────────────────────────────────

class TestAgentResult:
    def test_valid_result_exposes_all_fields(self) -> None:
        token_usage = make_token_usage()
        model_data = make_model_data()
        started_at = datetime.now(timezone.utc)
        result = AgentResult(
            result="done",
            invoker_id="agent-1",
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=1),
            llm_token_usage=(token_usage,),
            llm_model_data=model_data,
        )
        assert result.result == "done"
        assert result.llm_token_usage == (token_usage,)
        assert result.llm_model_data is model_data

    def test_normalizes_llm_token_usage_list_to_tuple(self) -> None:
        started_at = datetime.now(timezone.utc)
        result = AgentResult(
            result="out",
            invoker_id="agent-1",
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=1),
            llm_token_usage=[make_token_usage()],
            llm_model_data=make_model_data(),
        )
        assert isinstance(result.llm_token_usage, tuple)

    def test_to_dict_includes_llm_token_usage_and_model_data(self) -> None:
        result = make_agent_result()
        d = result.to_dict()
        assert "llm_token_usage" in d
        assert isinstance(d["llm_token_usage"], list)
        assert "llm_model_data" in d
        assert "llm_records" not in d

    def test_empty_llm_token_usage_is_valid(self) -> None:
        started_at = datetime.now(timezone.utc)
        result = AgentResult(
            result="out",
            invoker_id="agent-1",
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=1),
            llm_token_usage=(),
            llm_model_data=make_model_data(),
        )
        assert result.llm_token_usage == ()

    def test_rejects_non_sequence_llm_token_usage(self) -> None:
        started_at = datetime.now(timezone.utc)
        with pytest.raises(TypeError, match="llm_token_usage"):
            AgentResult(
                result="out",
                invoker_id="agent-1",
                started_at=started_at,
                ended_at=started_at + timedelta(seconds=1),
                llm_token_usage=42,  # type: ignore[arg-type]
                llm_model_data=make_model_data(),
            )

    def test_rejects_non_token_usage_item(self) -> None:
        started_at = datetime.now(timezone.utc)
        with pytest.raises(TypeError, match="llm_token_usage"):
            AgentResult(
                result="out",
                invoker_id="agent-1",
                started_at=started_at,
                ended_at=started_at + timedelta(seconds=1),
                llm_token_usage=("not a token usage",),  # type: ignore[arg-type]
                llm_model_data=make_model_data(),
            )

    def test_rejects_non_llm_model_data(self) -> None:
        started_at = datetime.now(timezone.utc)
        with pytest.raises(TypeError, match="llm_model_data"):
            AgentResult(
                result="out",
                invoker_id="agent-1",
                started_at=started_at,
                ended_at=started_at + timedelta(seconds=1),
                llm_token_usage=(make_token_usage(),),
                llm_model_data="not model data",  # type: ignore[arg-type]
            )

    def test_run_id_auto_generated(self) -> None:
        result = make_agent_result()
        assert isinstance(result.run_id, str)
        assert result.run_id  # non-empty

    def test_is_frozen(self) -> None:
        result = make_agent_result()
        with pytest.raises(FrozenInstanceError):
            result.llm_model_data = make_model_data()  # type: ignore[misc]


# ── TestToolAgentResult ───────────────────────────────────────────────────────

class TestToolAgentResult:
    def _make_result(
        self,
        *,
        usage_report: ToolUsageReport | None = None,
        failed_call_count: int = 0,
        regenerations_used: int = 0,
    ) -> ToolAgentResult:
        started_at = datetime.now(timezone.utc)
        return ToolAgentResult(
            result="done",
            invoker_id="agent-1",
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=1),
            llm_token_usage=(make_token_usage(),),
            llm_model_data=make_model_data(),
            usage_report=usage_report
            if usage_report is not None
            else ToolUsageReport(by_tool=(), total_dispatched=0, total_failed=0),
            failed_call_count=failed_call_count,
            regenerations_used=regenerations_used,
        )

    def test_is_agent_result(self) -> None:
        result = self._make_result()
        assert isinstance(result, AgentResult)

    def test_empty_usage_report_accepted(self) -> None:
        result = self._make_result()
        assert result.usage_report.by_tool == ()
        assert result.usage_report.total_dispatched == 0

    def test_usage_report_stored_verbatim(self) -> None:
        rec = ToolUsageRecord(tool_name="Tool.x", call_count=2)
        report = ToolUsageReport(by_tool=(rec,), total_dispatched=2, total_failed=0)
        result = self._make_result(usage_report=report)
        assert result.usage_report is report

    def test_to_dict_includes_usage_report(self) -> None:
        rec = ToolUsageRecord(tool_name="Tool.x", call_count=1)
        report = ToolUsageReport(by_tool=(rec,), total_dispatched=1, total_failed=0)
        result = self._make_result(usage_report=report, failed_call_count=0, regenerations_used=2)
        d = result.to_dict()
        assert d["usage_report"] == {
            "by_tool": [{"tool_name": "Tool.x", "call_count": 1}],
            "total_dispatched": 1,
            "total_failed": 0,
        }
        assert "llm_token_usage" in d
        assert d["failed_call_count"] == 0
        assert d["regenerations_used"] == 2

    def test_rejects_non_usage_report(self) -> None:
        started_at = datetime.now(timezone.utc)
        with pytest.raises(TypeError, match="ToolUsageReport"):
            ToolAgentResult(
                result="done",
                invoker_id="agent-1",
                started_at=started_at,
                ended_at=started_at + timedelta(seconds=1),
                llm_token_usage=(make_token_usage(),),
                llm_model_data=make_model_data(),
                usage_report="not a report",  # type: ignore[arg-type]
            )

    def test_is_frozen(self) -> None:
        result = self._make_result()
        with pytest.raises(FrozenInstanceError):
            result.usage_report = ToolUsageReport(by_tool=(), total_dispatched=0, total_failed=0)  # type: ignore[misc]

    def test_failed_call_count_defaults_to_zero(self) -> None:
        result = self._make_result()
        assert result.failed_call_count == 0

    def test_failed_call_count_stored(self) -> None:
        result = self._make_result(failed_call_count=3)
        assert result.failed_call_count == 3

    def test_regenerations_used_defaults_to_zero(self) -> None:
        result = self._make_result()
        assert result.regenerations_used == 0

    def test_regenerations_used_stored(self) -> None:
        result = self._make_result(regenerations_used=4)
        assert result.regenerations_used == 4


# ── TestThinkingAgentResult ───────────────────────────────────────────────────

class TestThinkingAgentResult:
    def _make_result(self, *, thinking_rounds_used=0) -> ThinkingAgentResult:
        started_at = datetime.now(timezone.utc)
        return ThinkingAgentResult(
            result="done",
            invoker_id="agent-1",
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=1),
            llm_token_usage=(make_token_usage(),),
            llm_model_data=make_model_data(),
            thinking_rounds_used=thinking_rounds_used,
        )

    def test_is_agent_result(self) -> None:
        result = self._make_result()
        assert isinstance(result, AgentResult)

    def test_thinking_rounds_used_defaults_to_zero(self) -> None:
        result = self._make_result()
        assert result.thinking_rounds_used == 0

    def test_thinking_rounds_used_stores_value(self) -> None:
        result = self._make_result(thinking_rounds_used=5)
        assert result.thinking_rounds_used == 5

    def test_to_dict_includes_thinking_rounds_used(self) -> None:
        result = self._make_result(thinking_rounds_used=5)
        d = result.to_dict()
        assert d["thinking_rounds_used"] == 5
        assert "llm_token_usage" in d
        assert "thoughts_start" not in d
        assert "thoughts_end" not in d
        assert "thoughts" not in d

    def test_is_frozen(self) -> None:
        result = self._make_result()
        with pytest.raises(FrozenInstanceError):
            result.thinking_rounds_used = 1  # type: ignore[misc]
