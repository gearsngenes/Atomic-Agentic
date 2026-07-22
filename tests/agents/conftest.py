from __future__ import annotations

import pytest

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
import asyncio
import json

import pytest

from atomic_agentic.agents.toolagent import ToolAgent, extract_dependencies, return_tool
from atomic_agentic.agents.planact import PlanActAgent
from atomic_agentic.agents.react import ReActAgent
from atomic_agentic.models.agents.runstates import ToolAgentRunState
from atomic_agentic.models.agents.records import AgentRecord, ToolAgentRecord
from atomic_agentic.models.agents.records import LLMRecord
from atomic_agentic.models.results.agents import ToolAgentResult
from atomic_agentic.models.agents.blackboard_models import BlackboardSlot, ConstantSpec
from atomic_agentic.exceptions import (
    AgentError,
    ToolAgentError,
    ToolInvocationError,
    ToolRegistrationError,
)
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.llm import LLMEngine
from atomic_agentic.models.results import LLMModelData, LLMResult, TokenUsage, ToolResult
from atomic_agentic.tools import Tool
from atomic_agentic.core.Invokable import AtomicInvokable


ROLE_TEMPLATE = "Tools:\n{TOOLS}\nLimit: {TOOL_CALLS_LIMIT}\nConstants:\n{CONSTANTS}"


class EchoLLMEngine(LLMEngine):
    """Minimal deterministic LLMEngine used only to satisfy Agent construction."""

    def __init__(self, *, response: str = "{}", **kwargs: Any) -> None:
        super().__init__(
            name="echo_llm_engine",
            description="Echo LLM engine for ToolAgent tests.",
            **kwargs,
        )
        self.response = response
        self.calls: list[list[dict[str, str]]] = []

    def _build_provider_payload(
        self,
        messages: list[dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append([dict(message) for message in messages])
        return {"messages": messages}

    def _call_provider(self, payload: Any) -> str:
        return self.response

    def _extract_text(self, response: Any) -> str:
        return str(response)

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        return TokenUsage(
            input_tokens=10, generated_tokens=5, total_tokens=15, response_tokens=5
        )

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        return False

    def _get_model_data(self) -> LLMModelData:
        return LLMModelData(provider="echo")

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        return {"path": path}

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        return None

class ScriptedLLMEngine(LLMEngine):
    """Deterministic LLMEngine that returns one scripted text response per call."""

    def __init__(self, responses: list[str], **kwargs: Any) -> None:
        super().__init__(
            name="scripted_llm_engine",
            description="Scripted LLM engine for ToolAgent subclass tests.",
            **kwargs,
        )
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def _build_provider_payload(
        self,
        messages: list[dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        copied_messages = [dict(message) for message in messages]
        self.calls.append(copied_messages)
        return {"messages": copied_messages}

    def _call_provider(self, payload: Any) -> str:
        if not self.responses:
            raise RuntimeError("No scripted LLM responses remain.")
        return self.responses.pop(0)

    def _extract_text(self, response: Any) -> str:
        return str(response)

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        return TokenUsage(
            input_tokens=10, generated_tokens=5, total_tokens=15, response_tokens=5
        )

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        return False

    def _get_model_data(self) -> LLMModelData:
        return LLMModelData(provider="scripted")

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        return {"path": path}

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        return None


class BadRepr:
    def __repr__(self) -> str:
        raise RuntimeError("repr failed")

    def __str__(self) -> str:
        return "fallback string value that is long"


def make_planact_agent(
    responses: list[str],
    *,
    context_enabled: bool = False,
    tool_calls_limit: int | None = None,
    peek_at_cache: bool = False,
    response_preview_limit: int | None = None,
    blackboard_preview_limit: int | None = None,
    post_invoke: Any = None,
    post_result_key: str | None = None,
    fail_fast: bool = True,
    generation_retries: int = 0,
) -> PlanActAgent:
    agent = PlanActAgent(
        name="tests",
        namespace="tests",
        description="PlanAct agent under test.",
        llm_engine=ScriptedLLMEngine(responses),
        context_enabled=context_enabled,
        tool_calls_limit=tool_calls_limit,
        peek_at_cache=peek_at_cache,
        response_preview_limit=response_preview_limit,
        blackboard_preview_limit=blackboard_preview_limit,
        post_invoke=post_invoke,
        post_result_key=post_result_key,
        fail_fast=fail_fast,
        generation_retries=generation_retries,
    )
    register_math_tools(agent)  # type: ignore[arg-type]
    return agent


def make_react_agent(
    responses: list[str],
    *,
    context_enabled: bool = False,
    tool_calls_limit: int = 3,
    peek_at_cache: bool = False,
    response_preview_limit: int | None = None,
    blackboard_preview_limit: int | None = None,
    post_invoke: Any = None,
    post_result_key: str | None = None,
    fail_fast: bool = True,
    generation_retries: int = 0,
) -> ReActAgent:
    agent = ReActAgent(
        name="tests",
        namespace="tests",
        description="ReAct agent under test.",
        llm_engine=ScriptedLLMEngine(responses),
        context_enabled=context_enabled,
        tool_calls_limit=tool_calls_limit,
        peek_at_cache=peek_at_cache,
        response_preview_limit=response_preview_limit,
        blackboard_preview_limit=blackboard_preview_limit,
        post_invoke=post_invoke,
        post_result_key=post_result_key,
        fail_fast=fail_fast,
        generation_retries=generation_retries,
    )
    register_math_tools(agent)  # type: ignore[arg-type]
    return agent


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


def react_step_json(
    *,
    tool: str,
    args: Any,
    step: int | None = 0,
    duration: int = 0,
    description: str = "Run the next tool call needed for the current test task.",
    **extra: Any,
) -> str:
    payload: dict[str, Any] = {}
    if step is not None:
        payload["step"] = step
    payload.update(
        {
            "tool": tool,
            "args": args,
            "duration": duration,
            "description": description,
        }
    )
    payload.update(extra)
    return json.dumps(payload)


@dataclass(slots=True)
class ScriptedRunState(ToolAgentRunState):
    batches: list[list[dict[str, Any]]] = field(default_factory=list)
    batch_index: int = 0
    next_step_index: int = 0


class ScriptedToolAgent(ToolAgent):
    """Deterministic ToolAgent subclass for testing the base ToolAgent loop."""

    def __init__(
        self,
        *,
        script: list[list[dict[str, Any]]] | None = None,
        context_enabled: bool = False,
        fail_fast: bool = True,
        generation_retries: int = 0,
        tool_calls_limit: int | None = None,
        peek_at_cache: bool = False,
        response_preview_limit: int | None = None,
        blackboard_preview_limit: int | None = None,
        post_invoke: Any = None,
        post_result_key: str | None = None,
    ) -> None:
        super().__init__(
            name="tests",
            namespace="tests",
            description="Scripted ToolAgent for unit tests.",
            llm_engine=EchoLLMEngine(),
            context_enabled=context_enabled,
            fail_fast=fail_fast,
            generation_retries=generation_retries,
            tool_calls_limit=tool_calls_limit,
            peek_at_cache=peek_at_cache,
            response_preview_limit=response_preview_limit,
            blackboard_preview_limit=blackboard_preview_limit,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
        )
        self._system_prompts["tool_instructions"] = PromptConfig(
            template=ROLE_TEMPLATE,
            description="Scripted test agent tool instructions.",
        )
        self.script = script or []

    def set_script(self, script: list[list[dict[str, Any]]]) -> None:
        self.script = script

    def _initialize_run_state(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> ScriptedRunState:
        limit_text = "unlimited" if self._tool_calls_limit is None else str(self._tool_calls_limit)
        render_ctx = {
            ToolAgent.TOOLS_FIELD: self.actions_context(),
            ToolAgent.LIMIT_FIELD: limit_text,
            ToolAgent.CONSTANTS_FIELD: self.constants_context(),
        }
        system = self._system_prompts["tool_instructions"].render(render_ctx)
        messages = self.build_messages(system, turns, prompt)

        total_steps = sum(len(batch) for batch in self.script)
        running_blackboard = [BlackboardSlot(step=index) for index in range(total_steps)]

        engine_result = self.llm_engine.invoke({"messages": messages})
        llm_record = LLMRecord(messages=[messages[-1]], llm_result=engine_result)

        return ScriptedRunState(
            inputs=inputs,
            messages=[dict(message) for message in messages],
            cache_blackboard=[slot.copy() for slot in self._blackboard],
            running_blackboard=running_blackboard,
            executed_steps=set(),
            prepared_steps=[],
            tool_calls_used=0,
            is_done=False,
            return_value=NO_VAL,
            llm_records=[llm_record],
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
            batches=[[dict(call) for call in batch] for batch in self.script],
            batch_index=0,
            next_step_index=0,
        )

    def _prepare_next_batch(self, state: ScriptedRunState) -> ScriptedRunState:
        if state.batch_index >= len(state.batches):
            raise ToolAgentError("No scripted batches remain.")

        batch = state.batches[state.batch_index]
        if not batch:
            raise ToolAgentError(f"ScriptedToolAgent: encountered empty batch at index {state.batch_index}.")

        prepared_steps: list[int] = []

        for call in batch:
            step = state.next_step_index
            if step >= len(state.running_blackboard):
                raise ToolAgentError("Scripted batch exceeded running blackboard size.")

            tool_name = call["tool"]
            args = call.get("args", {})

            self.get_tool(tool_name)

            slot = state.running_blackboard[step]
            slot.tool = tool_name
            slot.args = args
            slot.resolved_args = self._resolve_placeholders(args, state=state)
            slot.result = NO_VAL
            slot.error = NO_VAL
            slot.step_dependencies = tuple(
                sorted(extract_dependencies(obj=args, placeholder_pattern=ToolAgent.STEP_REF_PATTERN))
            )
            slot.await_step = NO_VAL
            slot.status = "prepared"

            prepared_steps.append(step)
            state.next_step_index += 1

        state.prepared_steps = prepared_steps
        state.batch_index += 1
        return state


class BadInitializeToolAgent(ScriptedToolAgent):
    def _initialize_run_state(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> Any:
        return {"bad": "state"}


class PendingPreparedToolAgent(ScriptedToolAgent):
    def _initialize_run_state(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> ScriptedRunState:
        state = super()._initialize_run_state(
            turns=turns,
            prompt=prompt,
            inputs=inputs,
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
        )
        state.prepared_steps = [0]
        return state


def make_agent(
    *,
    context_enabled: bool = False,
    fail_fast: bool = True,
    tool_calls_limit: int | None = None,
    peek_at_cache: bool = False,
    response_preview_limit: int | None = None,
    blackboard_preview_limit: int | None = None,
    post_invoke: Any = None,
    post_result_key: str | None = None,
) -> ScriptedToolAgent:
    return ScriptedToolAgent(
        context_enabled=context_enabled,
        fail_fast=fail_fast,
        tool_calls_limit=tool_calls_limit,
        peek_at_cache=peek_at_cache,
        response_preview_limit=response_preview_limit,
        blackboard_preview_limit=blackboard_preview_limit,
        post_invoke=post_invoke,
        post_result_key=post_result_key,
    )


def register_math_tools(agent: ScriptedToolAgent) -> dict[str, str]:
    return {
        "add": agent.register(add),
        "multiply": agent.register(multiply),
        "join_text": agent.register(join_text),
        "fail_tool": agent.register(fail_tool),
    }


def prepared_slot(step: int, tool: str, args: Mapping[str, Any]) -> BlackboardSlot:
    slot = BlackboardSlot(step=step)
    slot.tool = tool
    slot.args = dict(args)
    slot.resolved_args = dict(args)
    slot.status = "prepared"
    return slot


def executed_slot(step: int, result: Any, *, tool: str = "Tool.tests.add") -> BlackboardSlot:
    slot = BlackboardSlot(step=step)
    slot.tool = tool
    slot.args = {}
    slot.resolved_args = {}
    slot.result = make_tool_result(result)
    slot.status = "executed"
    return slot


def make_state(
    *,
    running: list[BlackboardSlot] | None = None,
    cache: list[BlackboardSlot] | None = None,
    prepared_steps: list[int] | None = None,
    tool_calls_used: int = 0,
    inputs: dict[str, Any] | None = None,
) -> ScriptedRunState:
    return ScriptedRunState(
        inputs=inputs or {},
        messages=[{"role": "user", "content": "run"}],
        cache_blackboard=cache or [],
        running_blackboard=running or [],
        executed_steps=set(),
        prepared_steps=prepared_steps or [],
        tool_calls_used=tool_calls_used,
        is_done=False,
        return_value=NO_VAL,
        batches=[],
        batch_index=0,
        next_step_index=0,
    )

