from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from atomic_agentic.models.results.agents import (
    AgentResult,
    ToolAgentResult,
    ToolUsageRecord,
)
from atomic_agentic.models.results.llm import LLMModelData, LLMResult, TokenUsage


# ── helpers ───────────────────────────────────────────────────────────────────

def make_token_usage(*, input_tokens: int = 10, generated_tokens: int = 5) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        total_tokens=input_tokens + generated_tokens,
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
    def _make_result(self, tool_usage=()) -> ToolAgentResult:
        started_at = datetime.now(timezone.utc)
        return ToolAgentResult(
            result="done",
            invoker_id="agent-1",
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=1),
            llm_token_usage=(make_token_usage(),),
            llm_model_data=make_model_data(),
            tool_usage=tool_usage,
        )

    def test_is_agent_result(self) -> None:
        result = self._make_result()
        assert isinstance(result, AgentResult)

    def test_empty_tool_usage_accepted(self) -> None:
        result = self._make_result(tool_usage=())
        assert result.tool_usage == ()

    def test_tool_usage_stored_as_tuple(self) -> None:
        rec = ToolUsageRecord(tool_name="Tool.x", call_count=2)
        result = self._make_result(tool_usage=[rec])
        assert isinstance(result.tool_usage, tuple)
        assert result.tool_usage == (rec,)

    def test_to_dict_includes_tool_usage(self) -> None:
        rec = ToolUsageRecord(tool_name="Tool.x", call_count=1)
        result = self._make_result(tool_usage=(rec,))
        d = result.to_dict()
        assert d["tool_usage"] == [{"tool_name": "Tool.x", "call_count": 1}]
        assert "llm_token_usage" in d

    def test_rejects_non_tool_usage_record_items(self) -> None:
        started_at = datetime.now(timezone.utc)
        with pytest.raises(TypeError, match="ToolUsageRecord"):
            ToolAgentResult(
                result="done",
                invoker_id="agent-1",
                started_at=started_at,
                ended_at=started_at + timedelta(seconds=1),
                llm_token_usage=(make_token_usage(),),
                llm_model_data=make_model_data(),
                tool_usage=("not a record",),  # type: ignore[arg-type]
            )

    def test_is_frozen(self) -> None:
        result = self._make_result()
        with pytest.raises(FrozenInstanceError):
            result.tool_usage = ()  # type: ignore[misc]
