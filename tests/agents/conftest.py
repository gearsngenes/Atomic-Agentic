from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
import json

from atomic_agentic.agents.planact import PlanActAgent
from atomic_agentic.agents.react import ReActAgent
from atomic_agentic.models.agents.records import LLMRecord
from atomic_agentic.models.results import LLMModelData, LLMResult, TokenUsage, ToolResult
from ..fake_engines import FakeLLMEngine


def add(x: int, y: int) -> int:
    return x + y


def multiply(x: int, y: int) -> int:
    return x * y


def join_text(prefix: str, value: Any) -> str:
    return f"{prefix}:{value}"


def fail_tool() -> str:
    raise RuntimeError("intentional failure")


def package_tool_result(result: Any, label: str) -> dict[str, Any]:
    return {"label": label, "result": result}


def register_math_tools(agent: PlanActAgent | ReActAgent) -> dict[str, str]:
    """Register add/multiply/join_text/fail_tool and return their effective
    (bare, unaliased) ids -- each toolified under agent.name as namespace, so
    the effective id (no alias given) is just the function's own __name__."""
    for tool in (add, multiply, join_text, fail_tool):
        agent.register_tool(tool)
    return {"add": "add", "multiply": "multiply", "join_text": "join_text", "fail_tool": "fail_tool"}


def make_planact_agent(
    responses: list[str],
    *,
    context_enabled: bool = False,
    tool_calls_limit: int | None = None,
    regeneration_limit: int = 5,
    fail_fast: bool = True,
    post_invoke: Any = None,
    post_result_key: str | None = None,
) -> PlanActAgent:
    agent = PlanActAgent(
        name="tests",
        namespace="tests",
        description="PlanAct agent under test.",
        llm_engine=FakeLLMEngine(responses),
        context_enabled=context_enabled,
        tool_calls_limit=tool_calls_limit,
        regeneration_limit=regeneration_limit,
        fail_fast=fail_fast,
        post_invoke=post_invoke,
        post_result_key=post_result_key,
    )
    register_math_tools(agent)
    return agent


def make_react_agent(
    responses: list[str],
    *,
    context_enabled: bool = False,
    tool_calls_limit: int | None = 3,
    regeneration_limit: int = 5,
    fail_fast: bool = False,
    post_invoke: Any = None,
    post_result_key: str | None = None,
) -> ReActAgent:
    agent = ReActAgent(
        name="tests",
        namespace="tests",
        description="ReAct agent under test.",
        llm_engine=FakeLLMEngine(responses),
        context_enabled=context_enabled,
        tool_calls_limit=tool_calls_limit,
        regeneration_limit=regeneration_limit,
        fail_fast=fail_fast,
        post_invoke=post_invoke,
        post_result_key=post_result_key,
    )
    register_math_tools(agent)
    return agent


def planact_plan_json(
    *,
    plan: list[dict[str, Any]],
    return_value: Any = None,
    summary: str = "Running the plan.",
) -> str:
    """Build one PlanActAgent-schema generation: {summary, plan, return}.
    Each `plan` item is {"call": str, "arguments": [{"name", "value"}, ...],
    "result_name": str | None} -- the current sigil-grammar wire shape
    (PLANACT_OUTPUT_SCHEMA), not the retired step/tool/args/await/duration
    shape."""
    return json.dumps({"summary": summary, "plan": plan, "return": return_value})


def react_step_json(
    *,
    call: str,
    arguments: list[dict[str, Any]],
    result_name: str | None = None,
    summary: str = "Running the next tool call needed for the current test task.",
) -> str:
    """Build one ReActAgent-schema generation: {summary, call, arguments,
    result_name} -- the current sigil-grammar wire shape (REACT_OUTPUT_SCHEMA),
    not the retired step/tool/args/await/duration shape."""
    return json.dumps(
        {"summary": summary, "call": call, "arguments": arguments, "result_name": result_name}
    )


def arg(name: str | None, value: Any) -> dict[str, Any]:
    """One `arguments[]` entry in the current wire schema."""
    return {"name": name, "value": value}


def make_llm_result(*, text: str = "generated text", invoker_id: str = "engine-1") -> LLMResult:
    started_at = datetime.now(timezone.utc)
    return LLMResult(
        result=text,
        invoker_id=invoker_id,
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=1),
        token_usage=TokenUsage(
            input_tokens=10, generated_tokens=5, total_tokens=15, response_tokens=5
        ),
        model_data=LLMModelData(provider="test"),
    )


def make_llm_record(*, text: str = "generated text") -> LLMRecord:
    return LLMRecord(
        messages=({"role": "user", "content": "generate a response"},),
        llm_result=make_llm_result(text=text),
    )


def make_tool_result(value: Any, *, invoker_id: str = "tool-1") -> ToolResult:
    started_at = datetime.now(timezone.utc)
    return ToolResult(
        result=value,
        invoker_id=invoker_id,
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=1),
    )
