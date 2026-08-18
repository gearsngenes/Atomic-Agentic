from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping
import json

import pytest

from atomic_agentic.agents.toolagent import ToolAgent, extract_dependencies, return_tool
from atomic_agentic.agents.planact import PlanActAgent
from atomic_agentic.agents.react import ReActAgent
from atomic_agentic.models.agents.tasks import ToolAgentTask
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
from atomic_agentic.models.results import LLMModelData, LLMResult, TokenUsage, ToolResult
from atomic_agentic.tools import Tool
from atomic_agentic.core.Invokable import AtomicInvokable
from ..fake_engines import FakeLLMEngine


ROLE_TEMPLATE = "Tools:\n{TOOLS}\nLimit: {TOOL_CALLS_LIMIT}\nConstants:\n{CONSTANTS}"


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
        llm_engine=FakeLLMEngine(responses),
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
        llm_engine=FakeLLMEngine(responses),
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
class ScriptedTask(ToolAgentTask):
    """Test-only ToolAgentTask subclass carrying the scripted fixture's
    batch cursor state, mirroring how PlanActTask/ReActTask each carry
    their own domain-specific fields on top of ToolAgentTask. No
    cache_blackboard field -- that field does not exist on ToolAgentTask;
    cache state lives on the owning agent's self._blackboard, never on the
    task."""
    batches: list[list[dict[str, Any]]] = field(default_factory=list)
    batch_index: int = 0
    next_step_index: int = 0


class ScriptedToolAgent(ToolAgent):
    """Deterministic ToolAgent subclass for testing the base ToolAgent
    think/prepare/act loop.

    Implements every ToolAgent-abstract hook (_initialize_task, think/
    async_think, prepare/async_prepare); never overrides act/async_act --
    those are concrete and final on ToolAgent itself.

    think/async_think are no-ops: this fixture's whole script is known
    upfront (mirrors ToolAgent.think's own documented "may no-op once
    there's nothing further to decide" case for an already-compiled plan),
    so there is no per-round decision to make -- the old fixture's engine
    call in _initialize_task served no behavioral purpose once think() is
    the hook responsible for real generation, and is dropped entirely.
    """

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
            llm_engine=FakeLLMEngine(response_fn=lambda messages: "{}"),
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

    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> ScriptedTask:
        valid_cache_indices, failed_cache_indices = self._compute_cache_index_sets(turns)
        total_steps = sum(len(batch) for batch in self.script)
        running_blackboard = [BlackboardSlot(step=index) for index in range(total_steps)]

        return ScriptedTask(
            turns=turns,
            inputs=inputs,
            user_prompt=prompt,
            system_prompt_name="tool_instructions",
            running_blackboard=running_blackboard,
            executed_steps=set(),
            prepared_steps=[],
            tool_calls_used=0,
            complete=False,
            generated_response=NO_VAL,
            llm_records=[],
            valid_cache_indices=valid_cache_indices,
            failed_cache_indices=failed_cache_indices,
            batches=[[dict(call) for call in batch] for batch in self.script],
            batch_index=0,
            next_step_index=0,
        )

    def think(self, task: ScriptedTask) -> ScriptedTask:
        # The scripted batches are fixed upfront and never depend on this
        # call's result -- but every real ToolAgent family guarantees at
        # least one LLMRecord by the time build_result_from_record runs
        # (inherited unchanged here), so a nominal one-time engine call is
        # made on first entry only, mirroring PlanActAgent's own
        # generate-once-then-no-op shape.
        if not task.llm_records:
            messages = [{"role": "user", "content": task.user_prompt}]
            engine_result = self._llm_engine.invoke({"messages": messages})
            task.llm_records.append(LLMRecord(messages=tuple(messages), llm_result=engine_result))
        return task

    async def async_think(self, task: ScriptedTask) -> ScriptedTask:
        if not task.llm_records:
            messages = [{"role": "user", "content": task.user_prompt}]
            engine_result = await self._llm_engine.async_invoke({"messages": messages})
            task.llm_records.append(LLMRecord(messages=tuple(messages), llm_result=engine_result))
        return task

    def _render_task_messages(self, task: ScriptedTask) -> list[dict[str, str]]:
        """Minimal implementation satisfying the abstract contract; not
        exercised on the scripted execution path since this fixture makes
        no real per-round LLM call -- present for abstract-class
        instantiation and for any test that calls render_task directly."""
        if not task.task_messages:
            task.task_messages = [{"role": "user", "content": task.user_prompt}]
        return task.task_messages

    def prepare(self, task: ScriptedTask) -> ScriptedTask:
        if task.batch_index >= len(task.batches):
            raise ToolAgentError("No scripted batches remain.")

        batch = task.batches[task.batch_index]
        if not batch:
            raise ToolAgentError(f"ScriptedToolAgent: encountered empty batch at index {task.batch_index}.")

        prepared_steps: list[int] = []

        for call in batch:
            step = task.next_step_index
            if step >= len(task.running_blackboard):
                raise ToolAgentError("Scripted batch exceeded running blackboard size.")

            tool_name = call["tool"]
            args = call.get("args", {})

            self.get_tool(tool_name)

            slot = task.running_blackboard[step]
            slot.tool = tool_name
            slot.args = args
            slot.resolved_args = self._resolve_placeholders(args, task=task)
            slot.result = NO_VAL
            slot.error = NO_VAL
            slot.step_dependencies = tuple(
                sorted(extract_dependencies(obj=args, placeholder_pattern=ToolAgent.STEP_REF_PATTERN))
            )
            slot.await_step = NO_VAL
            slot.status = "prepared"

            prepared_steps.append(step)
            task.next_step_index += 1

        task.prepared_steps = prepared_steps
        task.batch_index += 1
        return task

    async def async_prepare(self, task: ScriptedTask) -> ScriptedTask:
        # No real I/O in this fixture -- direct passthrough, not a thread
        # offload. ToolAgent supplies no default; each family decides.
        return self.prepare(task)


class BadInitializeToolAgent(ScriptedToolAgent):
    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> Any:
        return {"bad": "state"}


class SkipFirstBatchToolAgent(ScriptedToolAgent):
    """Batch index 0 always cascade-skips to empty (simulating every step
    in that round having failed a dependency check) without executing or
    raising; batch index 1+ delegates to the real scripted behavior. Tests
    that ToolAgent.act tolerates a prepare() call that legitimately
    produces an empty task.prepared_steps."""

    def prepare(self, task: ScriptedTask) -> ScriptedTask:
        if task.batch_index == 0:
            task.batch_index += 1
            return task
        task.batch_index -= 1
        return super().prepare(task)


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


def make_task(
    *,
    running: list[BlackboardSlot] | None = None,
    prepared_steps: list[int] | None = None,
    tool_calls_used: int = 0,
    inputs: dict[str, Any] | None = None,
) -> ScriptedTask:
    """No cache/cache_blackboard parameter -- tests needing agent-level
    cache state must set agent._blackboard directly before exercising
    cache-placeholder resolution, not pass it through the task."""
    return ScriptedTask(
        turns=[],
        inputs=inputs or {},
        user_prompt="run",
        system_prompt_name="tool_instructions",
        running_blackboard=running or [],
        executed_steps=set(),
        prepared_steps=prepared_steps or [],
        tool_calls_used=tool_calls_used,
        complete=False,
        generated_response=NO_VAL,
        batches=[],
        batch_index=0,
        next_step_index=0,
    )
