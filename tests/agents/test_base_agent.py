from __future__ import annotations

import dataclasses
import json
from typing import Annotated, Any, Mapping, Optional
import warnings

import pytest
import asyncio

from atomic_agentic.tools import Tool
from atomic_agentic.agents.base import Agent
from atomic_agentic.exceptions import AgentError, AgentInvocationError, ToolInvocationError
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.models.agents.records import AgentRecord, LLMRecord
from atomic_agentic.models.agents.conversations import Conversation
from atomic_agentic.models.agents.context import ContextVariable
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.models.agents.tasks import AgentTask
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.utils.parameters import to_paramspec_list
from ..fake_engines import FakeLLMEngine, echo_latest_user


ROLE_PROMPT = "You are a deterministic test writer."

_DUMMY_RESERVED_PARAM: ParamSpec = ParamSpec(
    name="dummy_reserved", index=0, kind=ParamSpec.KEYWORD_ONLY,
    type=("None", "str"), default=None,
    description="Test-only reserved parameter standing in for a real "
                 "framework reservation.",
)
"""Base Agent reserves nothing by default (post run-id-param-removal) --
this stands in for whatever a real reservation would be, so the generic
reserved-parameter graft/collision/ordering machinery below still has
something concrete to exercise."""


# ─────────────────────────────────────────────────────────────────────────────
# Minimal concrete Agent fixture (for abstract-API tests)
# ─────────────────────────────────────────────────────────────────────────────
class _MinimalAgent(Agent):
    """Trivial concrete Agent used to test abstract Agent construction and API.

    ``act`` performs a single engine call and completes the task; tests
    that never call invoke() don't need this to run. No ``create_task``
    override -- Agent's concrete base implementation (bare AgentTask,
    ``system_prompt_name=None``) is sufficient. ``think``/``prepare`` stay
    inherited no-ops. Declares one reserved parameter
    (``_DUMMY_RESERVED_PARAM``) rather than inheriting base ``Agent``'s own
    empty default, so tests exercising the generic reserved-parameter
    machinery have a concrete reservation to work against.
    """

    @classmethod
    def get_reserved_parameters(cls) -> list[ParamSpec]:
        return [_DUMMY_RESERVED_PARAM]

    def _render_task_messages(self, task: AgentTask) -> list[dict[str, str]]:
        if not task.task_messages:
            task.task_messages = [{"role": "user", "content": task.user_prompt}]
        return task.task_messages

    def act(self, task: AgentTask) -> AgentTask:
        engine_result = self._llm_engine.invoke({
            "messages": [
                {"role": "system", "content": "minimal"},
                {"role": "user", "content": task.user_prompt},
            ]
        })
        llm_record = LLMRecord(
            messages=({"role": "user", "content": task.user_prompt},),
            llm_result=engine_result,
        )
        task.llm_records.append(llm_record)
        task.generated_response = engine_result.result
        task.complete = True
        return task

    async def async_act(self, task: AgentTask) -> AgentTask:
        engine_result = await self._llm_engine.async_invoke({
            "messages": [
                {"role": "system", "content": "minimal"},
                {"role": "user", "content": task.user_prompt},
            ]
        })
        llm_record = LLMRecord(
            messages=({"role": "user", "content": task.user_prompt},),
            llm_result=engine_result,
        )
        task.llm_records.append(llm_record)
        task.generated_response = engine_result.result
        task.complete = True
        return task


class _EchoAgent(Agent):
    """Concrete Agent fixture with a fixed system prompt and a real
    single-LLM-call ``act`` path. Used in place of ``BasicAgent`` for
    invoke-lifecycle tests in this file for fixture independence.

    ``act``/``async_act`` build the message list directly from
    ``self._echo_system_prompt`` -- a deliberate bypass of the
    ``render_task``/``_render_system_message`` pipeline (unchanged from the
    pre-1a design), so ``_render_task_messages`` below is never actually
    exercised by the main lifecycle; it exists only to satisfy the abstract
    contract.
    """

    def __init__(self, *, system_prompt: str = ROLE_PROMPT, **kwargs: Any) -> None:
        self._echo_system_prompt = system_prompt
        super().__init__(**kwargs)
        self._system_prompts["system"] = PromptConfig(
            template=system_prompt, description="Echo agent system prompt."
        )

    def _render_task_messages(self, task: AgentTask) -> list[dict[str, str]]:
        if not task.task_messages:
            task.task_messages = [{"role": "user", "content": task.user_prompt}]
        return task.task_messages

    def act(self, task: AgentTask) -> AgentTask:
        messages = [
            {"role": "system", "content": self._echo_system_prompt},
            *task.historic_messages,
            {"role": "user", "content": task.user_prompt},
        ]
        engine_result = self._llm_engine.invoke({"messages": messages})
        text = engine_result.result
        if not isinstance(text, str):
            raise AgentInvocationError(
                f"LLM engine returned non-string result (type={type(text).__name__})."
            )
        llm_record = LLMRecord(messages=(messages[-1],), llm_result=engine_result)
        task.llm_records.append(llm_record)
        task.generated_response = text
        task.complete = True
        return task

    async def async_act(self, task: AgentTask) -> AgentTask:
        messages = [
            {"role": "system", "content": self._echo_system_prompt},
            *task.historic_messages,
            {"role": "user", "content": task.user_prompt},
        ]
        engine_result = await self._llm_engine.async_invoke({"messages": messages})
        text = engine_result.result
        if not isinstance(text, str):
            raise AgentInvocationError(
                f"LLM engine returned non-string result (type={type(text).__name__})."
            )
        llm_record = LLMRecord(messages=(messages[-1],), llm_result=engine_result)
        task.llm_records.append(llm_record)
        task.generated_response = text
        task.complete = True
        return task


class _ExtraReservedAgent(_MinimalAgent):
    """Concrete Agent subclass declaring one additional reserved parameter.

    Exercises the ``get_reserved_parameters`` override contract: a subclass
    adds its own reserved name on top of ``_MinimalAgent``'s own
    ``[_DUMMY_RESERVED_PARAM]`` list.
    """

    EXTRA_RESERVED = ParamSpec(
        name="trace_id", index=0, kind=ParamSpec.KEYWORD_ONLY,
        type=("None", "str"), default=None,
        description="Subclass-reserved trace identifier.",
    )

    @classmethod
    def get_reserved_parameters(cls) -> list[ParamSpec]:
        return super().get_reserved_parameters() + [cls.EXTRA_RESERVED]


class _LifecycleOrderAgent(Agent):
    """Stub instrumenting think/prepare/act separately to verify the base
    loop's exact per-round call order and repeat-until-complete behavior --
    something a single old-style ``_progress`` override could never
    express, since it fused all three phases into one call."""

    def __init__(self, *, rounds: int, **kwargs: Any) -> None:
        self._rounds_remaining = rounds
        self.call_log: list[str] = []
        super().__init__(**kwargs)

    def _render_task_messages(self, task: AgentTask) -> list[dict[str, str]]:
        if not task.task_messages:
            task.task_messages = [{"role": "user", "content": task.user_prompt}]
        return task.task_messages

    def think(self, task: AgentTask) -> AgentTask:
        self.call_log.append("think")
        return task

    def prepare(self, task: AgentTask) -> AgentTask:
        self.call_log.append("prepare")
        return task

    def act(self, task: AgentTask) -> AgentTask:
        self.call_log.append("act")
        self._rounds_remaining -= 1
        if self._rounds_remaining <= 0:
            engine_result = self._llm_engine.invoke({
                "messages": [{"role": "user", "content": task.user_prompt}]
            })
            task.llm_records.append(LLMRecord(
                messages=({"role": "user", "content": task.user_prompt},),
                llm_result=engine_result,
            ))
            task.generated_response = "done"
            task.complete = True
        return task

    async def async_act(self, task: AgentTask) -> AgentTask:
        return self.act(task)


# ─────────────────────────────────────────────────────────────────────────────
# Test engine and helper functions
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(topic: str, tone: str = "neutral") -> str:
    return f"Write about {topic} in a {tone} tone."


def package_response(result: str) -> dict[str, Any]:
    return {
        "final": result,
        "length": len(result),
        "was_postprocessed": True,
    }


def bad_pre_invoke(topic: str) -> Any:
    return {"topic": topic}


def defaulted_post_invoke(result: str = "default") -> str:
    return result.upper()


def bad_post_two_required_args(result: str, suffix: str) -> str:
    return f"{result}{suffix}"


def pre_with_two_fields(subject: str, style: str = "plain") -> str:
    return f"{subject}:{style}"


def pre_returns_int(prompt: str) -> int:
    return 123


def post_zero_args() -> str:
    return "bad"


def post_one_required_plus_default(result: str, suffix: str = "!") -> str:
    return f"{result}{suffix}"


def post_with_suffix(result: str, suffix: str) -> str:
    return f"{result}{suffix}"


def post_with_custom_key(raw: str, suffix: str = "!") -> str:
    return f"{raw}{suffix}"


def post_with_overlap(result: str, tone: str = "post-default") -> dict[str, str]:
    return {"result": result, "tone": tone}


def post_with_type_mismatch(result: str, tone: int) -> str:
    return f"{result}|tone={tone}"


def post_with_args(*items: Any) -> tuple[Any, ...]:
    return items


def post_with_kwargs(**items: Any) -> dict[str, Any]:
    return dict(items)


def post_with_post_only_args(result: str, *extras: Any) -> str:
    return f"{result}|extras={extras!r}"


class _NonStringEngineEnvelope:
    """Stub mimicking AtomicResult `.result` access without LLMResult's str constraint."""

    def __init__(self, result: Any) -> None:
        self.result = result


class NonStringAsyncLLMEngine(FakeLLMEngine):
    async def async_invoke(self, inputs: Mapping[str, Any]) -> Any:
        return _NonStringEngineEnvelope(123)


def make_agent(
    *,
    engine: FakeLLMEngine | None = None,
    record_enabled: bool = False,
    records_window: int | None = None,
    pre_invoke: Any = build_prompt,
    post_invoke: Any = package_response,
    bind_response_to: Any = "result",
    system_prompt: str = ROLE_PROMPT,
    agent_cls: type[_EchoAgent] = _EchoAgent,
) -> _EchoAgent:
    # Default binding "result" matches the first parameter of the default
    # post (package_response) and most helper posts below; tests whose post
    # names its response parameter differently pass bind_response_to.
    return agent_cls(
        name="writer_agent",
        namespace="tests",
        description="Deterministic writer test agent.",
        llm_engine=engine or FakeLLMEngine(response_fn=echo_latest_user()),
        system_prompt=system_prompt,
        record_enabled=record_enabled,
        records_window=records_window,
        pre_invoke=pre_invoke,
        post_invoke=post_invoke,
        bind_response_to=bind_response_to,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test classes
# ─────────────────────────────────────────────────────────────────────────────
class TestBaseLoopOrder:
    """New 1f coverage: think/prepare/act are called in strict order, every
    round, unconditionally, until task.complete -- the shape 1a introduced
    to replace the old single ``_progress`` hook. A single fused hook could
    never express "was think called before prepare, before act" as a
    separate assertion; instrumenting all three separately is the point."""

    def test_think_prepare_act_called_in_order_each_round_until_complete(self) -> None:
        agent = _LifecycleOrderAgent(
            rounds=3,
            name="a", namespace="tests", description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
        )

        agent.invoke({"prompt": "hi"})

        assert agent.call_log == ["think", "prepare", "act"] * 3

    def test_async_think_prepare_act_called_in_order_each_round_until_complete(self) -> None:
        agent = _LifecycleOrderAgent(
            rounds=2,
            name="a", namespace="tests", description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
        )

        asyncio.run(agent.async_invoke({"prompt": "hi"}))

        assert agent.call_log == ["think", "prepare", "act"] * 2


class TestAgentPipeline:
    def test_pre_invoke_shapes_prompt_and_post_invoke_packages_result(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(engine=engine)

        result = agent.invoke({"topic": "pytest", "tone": "strict"})

        expected_prompt = "Write about pytest in a strict tone."
        expected_raw = f"ECHO: {expected_prompt}"

        assert result.result == {
            "final": expected_raw,
            "length": len(expected_raw),
            "was_postprocessed": True,
        }
        assert engine.calls == [
            [
                {"role": "system", "content": ROLE_PROMPT},
                {"role": "user", "content": expected_prompt},
            ]
        ]

    def test_engine_receives_system_and_current_user_message(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(engine=engine)

        agent.invoke({"topic": "agents", "tone": "concise"})

        assert engine.calls[0] == [
            {"role": "system", "content": ROLE_PROMPT},
            {"role": "user", "content": "Write about agents in a concise tone."},
        ]

    def test_default_no_pre_and_no_post_path(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = _EchoAgent(
            name="identity_agent",
            namespace="tests",
            description="Identity agent.",
            llm_engine=engine,
            system_prompt=ROLE_PROMPT,
            record_enabled=False,
        )

        result = agent.invoke({"prompt": "Hello from identity."})

        assert result.result == "ECHO: Hello from identity."
        assert engine.calls[0] == [
            {"role": "system", "content": ROLE_PROMPT},
            {"role": "user", "content": "Hello from identity."},
        ]

    def test_agent_schema_defaults_to_pre_invoke_parameters(self) -> None:
        agent = make_agent()

        assert [(param.name, param.kind, param.default) for param in agent.parameters] == [
            ("topic", "POSITIONAL_OR_KEYWORD", NO_VAL),
            ("tone", "POSITIONAL_OR_KEYWORD", "neutral"),
        ]
        assert agent.return_type == "dict[str, Any]"


class TestAgentSchemaComposition:
    def test_post_only_non_result_param_preserves_its_declared_kind(self) -> None:
        # suffix is POSITIONAL_OR_KEYWORD in post_with_suffix's own signature --
        # no forced KEYWORD_ONLY coercion (dropped this pass). Since suffix is
        # required (no default) it sorts ahead of the defaulted "tone" within
        # the POSITIONAL_OR_KEYWORD category (insert_by_category's required-
        # before-defaulted ordering rule).
        agent = make_agent(post_invoke=post_with_suffix)

        assert [(param.name, param.kind, param.default) for param in agent.parameters] == [
            ("topic", "POSITIONAL_OR_KEYWORD", NO_VAL),
            ("suffix", "POSITIONAL_OR_KEYWORD", NO_VAL),
            ("tone", "POSITIONAL_OR_KEYWORD", "neutral"),
        ]

    def test_auto_grafted_param_preserves_post_default(self) -> None:
        agent = make_agent(post_invoke=post_one_required_plus_default)

        suffix_param = next(param for param in agent.parameters if param.name == "suffix")

        # Declared kind preserved (POSITIONAL_OR_KEYWORD, not forced KEYWORD_ONLY).
        assert suffix_param.kind == "POSITIONAL_OR_KEYWORD"
        assert suffix_param.default == "!"

    def test_overlapping_param_pre_wins_in_schema(self) -> None:
        agent = make_agent(post_invoke=post_with_overlap)

        tone_param = next(param for param in agent.parameters if param.name == "tone")

        assert tone_param.kind == "POSITIONAL_OR_KEYWORD"
        assert tone_param.default == "neutral"

    def test_overlapping_param_pre_value_reaches_post_invoke(self) -> None:
        agent = make_agent(post_invoke=post_with_overlap)

        result = agent.invoke({"topic": "pytest"})

        assert result.result == {
            "result": "ECHO: Write about pytest in a neutral tone.",
            "tone": "neutral",
        }

    def test_variadic_post_only_param_is_grafted_preserving_its_kind(self) -> None:
        # Dropped this pass: post-only VAR_POSITIONAL/VAR_KEYWORD params are no
        # longer excluded from grafting -- insert_by_category places them in
        # their own valid ordering slot (after POSITIONAL_OR_KEYWORD, before
        # KEYWORD_ONLY) like any other kind. Built on _MinimalAgent directly
        # (not make_agent()'s _EchoAgent, which has no reserved params) so a
        # real KEYWORD_ONLY reserved param is present to demonstrate "before
        # KEYWORD_ONLY" against.
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            pre_invoke=build_prompt,
            post_invoke=post_with_post_only_args,
            bind_response_to="result",
        )

        names = [p.name for p in agent.parameters]
        assert names == ["topic", "tone", "extras", "dummy_reserved"]
        extras_param = next(p for p in agent.parameters if p.name == "extras")
        assert extras_param.kind == "VAR_POSITIONAL"

    def test_overlapping_param_with_type_mismatch_raises_at_construction(self) -> None:
        # "tone" overlaps pre_invoke's own "tone: str" param but is declared
        # "int" here -- an incompatible collision, caught by the N-way
        # peer-report traversal (empty witness_types for "tone").
        with pytest.raises(AgentError, match="no compatible reconciliation"):
            make_agent(post_invoke=post_with_type_mismatch)

    def test_truly_post_only_param_with_any_type_routes_correctly(self) -> None:
        # A post-only param that does NOT overlap any pre_invoke name is free
        # to declare any type -- only overlapping names are cross-checked.
        agent = make_agent(post_invoke=post_with_suffix)

        result = agent.invoke({"topic": "pytest", "tone": "strict", "suffix": "!"})

        assert result.result == "ECHO: Write about pytest in a strict tone.!"


class TestAgentPostInvokeRouting:
    def test_auto_grafted_required_param_reached_by_caller(self) -> None:
        agent = make_agent(post_invoke=post_with_suffix)

        result = agent.invoke({"topic": "pytest", "tone": "strict", "suffix": "!"})

        assert result.result == "ECHO: Write about pytest in a strict tone.!"

    def test_missing_required_auto_grafted_param_fails_at_invoke_time(self) -> None:
        agent = make_agent(post_invoke=post_with_suffix)

        with pytest.raises(TypeError, match="missing required"):
            agent.invoke({"topic": "pytest", "tone": "strict"})

    def test_custom_bind_response_to_routes_raw_result(self) -> None:
        agent = make_agent(
            post_invoke=post_with_custom_key,
            bind_response_to="raw",
        )

        result = agent.invoke({"topic": "pytest", "tone": "strict"})

        assert agent._bind_response_to == "raw"
        assert result.result == "ECHO: Write about pytest in a strict tone.!"

    def test_bind_response_to_args_parameter_is_allowed(self) -> None:
        agent = make_agent(
            post_invoke=post_with_args,
            bind_response_to="items",
        )

        assert agent._bind_response_to == "items"

    def test_bind_response_to_matching_pre_invoke_name_is_allowed(self) -> None:
        # A destination that pre_invoke also declares stays in the public
        # signature (pre wins); the response still overlays the caller value.
        def post_with_topic_key(topic: str) -> str:
            return topic

        agent = make_agent(post_invoke=post_with_topic_key, bind_response_to="topic")
        assert agent._bind_response_to == "topic"


class TestAgentContext:
    def test_record_disabled_does_not_store_records(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(engine=engine, record_enabled=False)

        first = agent.invoke({"topic": "pytest", "tone": "strict"})
        second = agent.invoke({"topic": "agents", "tone": "concise"})

        assert first.result["final"] == "ECHO: Write about pytest in a strict tone."
        assert second.result["final"] == "ECHO: Write about agents in a concise tone."

        # record_enabled=False means nothing is stored in any conversation,
        # regardless of records_window.
        assert agent.get_conversation().records == []

        assert len(engine.calls) == 2
        assert [message["role"] for message in engine.calls[0]] == ["system", "user"]
        assert [message["role"] for message in engine.calls[1]] == ["system", "user"]
        assert engine.calls[0][-1]["content"] == "Write about pytest in a strict tone."
        assert engine.calls[1][-1]["content"] == "Write about agents in a concise tone."

    def test_record_enabled_stores_history(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(engine=engine, record_enabled=True)

        first = agent.invoke({"topic": "pytest", "tone": "strict"})
        second = agent.invoke({"topic": "agents", "tone": "concise"})

        assert first.result["final"] == "ECHO: Write about pytest in a strict tone."
        assert second.result["final"] == "ECHO: Write about agents in a concise tone."

        rendered = agent.render_turn(agent.get_conversation().records[0])
        assert [m["role"] for m in rendered] == ["user", "assistant"]
        assert rendered[0]["content"] == "Write about pytest in a strict tone."
        assert rendered[1]["content"] == "ECHO: Write about pytest in a strict tone."

    def test_render_turn_stringifies_non_str_generated_response_as_json(self) -> None:
        # Regression guard: a structured (response_schema-produced)
        # generated_response can round-trip back into history as valid JSON
        # when the conversation's render_mode is "j" (the default "s" gives
        # Python str syntax) -- see utils.rendering.stringify_value.
        agent = make_agent(engine=FakeLLMEngine([]), record_enabled=True)
        value = {"summary": "ok", "confident": True, "score": None}
        record = AgentRecord(user_prompt="hello", generated_response=value)

        as_json = agent.render_turn(record, render_mode="j")
        as_str = agent.render_turn(record)

        assert json.loads(as_json[1]["content"]) == value
        assert as_str[1]["content"] == str(value)

    def test_record_enabled_resends_prior_history_on_second_call(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(engine=engine, record_enabled=True)

        agent.invoke({"topic": "pytest", "tone": "strict"})
        agent.invoke({"topic": "agents", "tone": "concise"})

        second_call_messages = engine.calls[1]

        assert [message["role"] for message in second_call_messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert second_call_messages[0]["content"] == ROLE_PROMPT
        assert second_call_messages[1]["content"] == "Write about pytest in a strict tone."
        assert second_call_messages[2]["content"] == "ECHO: Write about pytest in a strict tone."
        assert second_call_messages[3]["content"] == "Write about agents in a concise tone."

    def test_records_window_none_sends_all_prior_turns(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(
            engine=engine,
            record_enabled=True,
            records_window=None,
        )

        agent.invoke({"topic": "first topic", "tone": "plain"})
        agent.invoke({"topic": "second topic", "tone": "plain"})
        agent.invoke({"topic": "third topic", "tone": "plain"})

        third_call_messages = engine.calls[2]
        joined_contents = "\n".join(message["content"] for message in third_call_messages)

        assert [message["role"] for message in third_call_messages] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert "first topic" in joined_contents
        assert "second topic" in joined_contents
        assert "third topic" in joined_contents

    def test_records_window_one_sends_only_last_prior_turn(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(
            engine=engine,
            record_enabled=True,
            records_window=1,
        )

        agent.invoke({"topic": "first topic", "tone": "plain"})
        agent.invoke({"topic": "second topic", "tone": "plain"})
        agent.invoke({"topic": "third topic", "tone": "plain"})

        third_call_messages = engine.calls[2]
        joined_contents = "\n".join(message["content"] for message in third_call_messages)

        assert [message["role"] for message in third_call_messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert "first topic" not in joined_contents
        assert "second topic" in joined_contents
        assert "third topic" in joined_contents

    def test_records_window_zero_sends_no_prior_turns_but_still_stores_when_record_enabled(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(
            engine=engine,
            record_enabled=True,
            records_window=0,
        )

        agent.invoke({"topic": "first topic", "tone": "plain"})
        agent.invoke({"topic": "second topic", "tone": "plain"})

        second_call_messages = engine.calls[1]

        assert [message["role"] for message in second_call_messages] == [
            "system",
            "user",
        ]
        assert second_call_messages[-1]["content"] == "Write about second topic in a plain tone."

        # records_window == 0 only gates reads now -- record_enabled=True
        # (the default here) still persists both turns.
        assert len(agent.get_conversation().records) == 2

    def test_record_disabled_reads_full_history_without_storing(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(engine=engine, record_enabled=True)  # seed one turn
        agent.invoke({"topic": "first topic", "tone": "plain"})

        agent.record_enabled = False
        agent.invoke({"topic": "second topic", "tone": "plain"})

        second_call_messages = engine.calls[1]
        joined_contents = "\n".join(m["content"] for m in second_call_messages)
        assert "first topic" in joined_contents  # read still happens
        assert len(agent.get_conversation().records) == 1  # second turn not persisted

    def test_clear_memory_removes_stored_history(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(engine=engine, record_enabled=True)

        agent.invoke({"topic": "pytest", "tone": "strict"})

        assert agent.get_conversation().records

        agent.clear_memory()

        assert agent.get_conversation().records == []


class TestConversationManagement:
    """Named-conversation branching: create/import/remove/switch. Invoking in a
    conversation always appends to it in place -- there is no implicit fork,
    even when the record it continues from already has children elsewhere."""

    def test_fresh_agent_has_single_default_active_conversation(self) -> None:
        agent = make_agent()

        assert agent.conversation_names == ["default"]
        assert agent.active_conversation == "default"
        assert agent.get_conversation().records == []

    def test_create_conversation_returns_conversation_on_fresh_name(self) -> None:
        agent = make_agent()

        created = agent.create_conversation("scratch")

        assert isinstance(created, Conversation)
        assert created.records == []
        assert "scratch" in agent.conversation_names
        assert agent.get_conversation("scratch").records == []
        # create_conversation never auto-activates.
        assert agent.active_conversation == "default"

    def test_create_conversation_raises_on_existing_name_no_mutation(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.create_conversation("scratch")
        agent.switch_conversation("scratch")
        agent.invoke({"topic": "pytest", "tone": "strict"})

        with pytest.raises(ValueError, match="conversation 'scratch' already exists"):
            agent.create_conversation("scratch")
        # The existing conversation's history survives the rejected collision.
        assert len(agent.get_conversation("scratch").records) == 1

    @pytest.mark.parametrize("bad_name", ["bad name", "1bad", ""])
    def test_create_conversation_rejects_invalid_name_shape(self, bad_name: str) -> None:
        agent = make_agent()

        with pytest.raises(ValueError):
            agent.create_conversation(bad_name)

    def test_switch_conversation_switches_active(self) -> None:
        agent = make_agent()
        agent.create_conversation("scratch")

        agent.switch_conversation("scratch")

        assert agent.active_conversation == "scratch"

    def test_switch_conversation_unknown_key_raises(self) -> None:
        agent = make_agent()

        with pytest.raises(AgentInvocationError):
            agent.switch_conversation("nope")

    def test_get_conversation_unknown_key_raises(self) -> None:
        agent = make_agent()

        with pytest.raises(AgentInvocationError):
            agent.get_conversation("nope").records

    def test_get_conversation_returns_tagged_snapshot(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke({"topic": "first", "tone": "strict"})
        agent.invoke({"topic": "second", "tone": "strict"})

        snap = agent.get_conversation()

        assert isinstance(snap, Conversation)
        assert snap.agent_type is type(agent)
        assert [r.inputs["topic"] for r in snap.records[-1:]] == ["second"]

    def test_get_conversation_returns_the_stored_object(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke({"topic": "pytest", "tone": "strict"})

        stored = agent.get_conversation()
        stored.remember("x", 1)

        assert agent.get_conversation() is stored
        assert agent.get_conversation("default") is stored
        assert len(stored.records) == 1
        assert "x" in agent.get_conversation().context

    def test_create_conversation_returns_the_stored_object_with_limits(self) -> None:
        agent = make_agent()

        conv = agent.create_conversation(
            "scratch",
            response_preview_limit=5,
            prompt_preview_limit=0,
            context_preview_limit=9,
        )

        assert agent.get_conversation("scratch") is conv
        assert conv.agent_type is _EchoAgent
        assert (
            conv.response_preview_limit,
            conv.prompt_preview_limit,
            conv.context_preview_limit,
        ) == (5, 0, 9)
        assert agent.active_conversation == "default"

    def test_create_conversation_defaults_limits_to_none(self) -> None:
        conv = make_agent().create_conversation("scratch")

        assert conv.response_preview_limit is None
        assert conv.prompt_preview_limit is None
        assert conv.context_preview_limit is None

    @pytest.mark.parametrize("bad", [-1, True, 1.5, "5"])
    def test_create_conversation_rejects_bad_limits_and_stores_nothing(
        self, bad: object
    ) -> None:
        agent = make_agent()

        with pytest.raises(ValueError, match="response_preview_limit"):
            agent.create_conversation("scratch", response_preview_limit=bad)  # type: ignore[arg-type]
        assert agent.conversation_names == ["default"]

    def test_create_conversation_bad_shape_message(self) -> None:
        with pytest.raises(ValueError, match="invalid conversation name"):
            make_agent().create_conversation("bad name")

    def test_trailing_digits_names_are_valid(self) -> None:
        agent = make_agent()

        agent.create_conversation("branch_5")

        assert "branch_5" in agent.conversation_names

    def test_name_is_required(self) -> None:
        agent = make_agent()

        with pytest.raises(TypeError):
            agent.create_conversation()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            agent.import_conversation(agent.get_conversation())  # type: ignore[call-arg]

    def test_limits_are_edited_directly_on_the_conversation(self) -> None:
        agent = make_agent()
        conv = agent.get_conversation()

        conv.response_preview_limit = 500
        conv.prompt_preview_limit = 0
        assert agent.get_conversation().response_preview_limit == 500
        assert agent.get_conversation().prompt_preview_limit == 0
        assert agent.get_conversation().context_preview_limit is None

        with pytest.raises(ValueError, match="response_preview_limit"):
            conv.response_preview_limit = -1
        assert conv.response_preview_limit == 500

    def test_remove_conversation_unknown_key_raises(self) -> None:
        agent = make_agent()

        with pytest.raises(AgentInvocationError):
            agent.remove_conversation("nope")

    def test_remove_conversation_refuses_active_conversation(self) -> None:
        agent = make_agent()
        agent.create_conversation("scratch")
        agent.switch_conversation("scratch")

        with pytest.raises(AgentInvocationError):
            agent.remove_conversation("scratch")
        assert "scratch" in agent.conversation_names

    def test_remove_conversation_returns_the_popped_non_active_conversation(self) -> None:
        agent = make_agent(record_enabled=True)
        scratch = agent.create_conversation("scratch")
        agent.switch_conversation("scratch")
        agent.invoke({"topic": "pytest", "tone": "strict"})
        agent.switch_conversation("default")

        removed = agent.remove_conversation("scratch")

        assert removed is scratch
        assert len(removed.records) == 1
        assert "scratch" not in agent.conversation_names

    def test_remove_default_rebuilds_a_fresh_default(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke({"topic": "pytest", "tone": "strict"})
        old = agent.get_conversation()
        old.response_preview_limit = 5

        removed = agent.remove_conversation("default")

        assert removed is old
        assert len(removed.records) == 1
        fresh = agent.get_conversation("default")
        assert fresh is not old
        assert fresh.records == []
        assert fresh.response_preview_limit is None
        assert fresh.agent_type is _EchoAgent
        assert agent.active_conversation == "default"

    def test_remove_default_while_another_is_active_keeps_active_key(self) -> None:
        agent = make_agent()
        agent.create_conversation("scratch")
        agent.switch_conversation("scratch")

        agent.remove_conversation("default")

        assert agent.active_conversation == "scratch"
        assert agent.conversation_names == ["default", "scratch"]


class _OtherEchoAgent(_EchoAgent):
    pass


class TestConversationContext:
    """Context is changed through the conversation object itself."""

    def test_mutators_through_get_conversation(self) -> None:
        agent = make_agent()
        agent.create_conversation("scratch")
        scratch = agent.get_conversation("scratch")

        scratch.remember("x", 1, "one")
        scratch.describe("x", "uno")
        scratch.remember_many([ContextVariable(name="y", value=2)])

        assert scratch.context["x"].value == 1 and scratch.context["x"].description == "uno"
        assert scratch.context["y"].value == 2
        assert len(agent.get_conversation("default").context) == 0

        scratch.forget("x")
        scratch.forget_many(["y"])
        assert len(scratch.context) == 0

    def test_agent_has_no_context_wrappers(self) -> None:
        agent = make_agent()

        for attr in ("remember", "describe", "forget", "remember_many", "forget_many"):
            assert not hasattr(agent, attr)

    def test_remove_default_and_clear_memory_reset_context_and_keep_tag(self) -> None:
        agent = make_agent()
        agent.get_conversation().remember("x", 1)

        agent.remove_conversation("default")
        assert len(agent.get_conversation().context) == 0
        assert agent.get_conversation().agent_type is _EchoAgent

        agent.get_conversation().remember("y", 1)
        agent.clear_memory()
        assert len(agent.get_conversation().context) == 0
        assert agent.get_conversation().agent_type is _EchoAgent

    def test_every_created_conversation_is_tagged(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke({"topic": "pytest", "tone": "strict"})
        agent.create_conversation("scratch")
        agent.import_conversation(agent.get_conversation().slice(0, 1), name="sliced")

        for name in ("default", "scratch", "sliced"):
            assert agent.get_conversation(name).agent_type is _EchoAgent


class TestImportConversation:
    def _source(self) -> tuple[_EchoAgent, list[str]]:
        agent = make_agent(record_enabled=True)
        ids = [
            agent.invoke({"topic": t, "tone": "strict"}).run_id
            for t in ("a", "b", "c")
        ]
        agent.get_conversation().remember("k", [1], "kay")
        return agent, ids

    def test_registers_the_same_object_and_returns_none(self) -> None:
        source, _ = self._source()
        target = make_agent()

        result = target.import_conversation(source.get_conversation(), name="imp")

        assert result is None
        assert target.get_conversation("imp") is source.get_conversation()
        assert target.active_conversation == "default"
        assert "imp" in target.conversation_names

    def test_same_object_may_sit_under_different_keys_in_different_agents(self) -> None:
        source, _ = self._source()
        target = make_agent()

        target.import_conversation(source.get_conversation(), name="other_key")

        assert source.conversation_names == ["default"]
        assert target.get_conversation("other_key") is source.get_conversation("default")

    def test_slice_import_is_independent_and_carries_latest_context(self) -> None:
        source, ids = self._source()
        target = make_agent()

        target.import_conversation(source.get_conversation().slice(1, 3), name="imp")
        imported = target.get_conversation("imp")

        assert [r.run_id for r in imported.records] == ids[1:]
        assert imported.context["k"] is source.get_conversation().context["k"]
        imported.remember("only_target", 1)
        source.get_conversation().forget("k")
        assert "only_target" not in source.get_conversation().context
        assert "k" in imported.context

    def test_slice_import_carries_limits_and_render_mode(self) -> None:
        source, _ = self._source()
        conv = source.get_conversation()
        conv.response_preview_limit = 4
        conv.prompt_preview_limit = 0
        conv.context_preview_limit = 8
        conv.render_mode = "j"
        target = make_agent()

        target.import_conversation(conv.slice(), name="imp")
        imported = target.get_conversation("imp")

        assert (
            imported.response_preview_limit,
            imported.prompt_preview_limit,
            imported.context_preview_limit,
        ) == (4, 0, 8)
        assert imported.render_mode == "j"

    def test_name_rules(self) -> None:
        source, _ = self._source()
        conv = source.get_conversation()
        target = make_agent()
        target.create_conversation("taken")

        with pytest.raises(ValueError, match="invalid conversation name 'bad name'"):
            target.import_conversation(conv, name="bad name")
        with pytest.raises(ValueError, match="conversation 'taken' already exists"):
            target.import_conversation(conv, name="taken")
        assert target.conversation_names == ["default", "taken"]

    def test_wrong_agent_type_raises_and_registers_nothing(self) -> None:
        source, _ = self._source()
        other = make_agent(agent_cls=_OtherEchoAgent)

        with pytest.raises(TypeError, match="_EchoAgent"):
            other.import_conversation(source.get_conversation(), name="imp")
        assert other.conversation_names == ["default"]

    def test_non_conversation_raises(self) -> None:
        with pytest.raises(TypeError):
            make_agent().import_conversation(make_agent(), name="imp")  # type: ignore[arg-type]

    def test_same_object_twice_in_one_agent_raises(self) -> None:
        source, _ = self._source()
        target = make_agent()
        target.import_conversation(source.get_conversation(), name="imp")

        with pytest.raises(ValueError, match="already registered here"):
            target.import_conversation(source.get_conversation(), name="again")
        with pytest.raises(ValueError, match="already registered here"):
            source.import_conversation(source.get_conversation(), name="alias")
        assert target.conversation_names == ["default", "imp"]

    def test_continuing_a_slice_appends_in_place_and_keeps_its_context(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke({"topic": "a", "tone": "strict"})
        agent.get_conversation().remember("k", 1)
        agent.invoke({"topic": "b", "tone": "strict"})
        agent.import_conversation(agent.get_conversation().slice(0, 1), name="back")
        agent.switch_conversation("back")

        agent.invoke({"topic": "c", "tone": "strict"})
        agent.invoke({"topic": "d", "tone": "strict"})

        assert set(agent.conversation_names) == {"default", "back"}
        assert len(agent.get_conversation("back").records) == 3
        assert len(agent.get_conversation("default").records) == 2
        assert agent.get_conversation("back").context["k"].value == 1

    def test_slice_of_whole_conversation_continues_after_original_advances(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke(_INPUTS)
        agent.import_conversation(agent.get_conversation().slice(), name="whole")
        agent.invoke(_INPUTS)  # the original advances past the shared tail

        agent.switch_conversation("whole")
        agent.invoke(_INPUTS)

        assert len(agent.get_conversation("whole").records) == 2
        assert len(agent.get_conversation("default").records) == 2


class TestSharedConversation:
    def _pair(self) -> tuple[_EchoAgent, _EchoAgent]:
        a = make_agent(record_enabled=True)
        b = make_agent(record_enabled=True)
        b.import_conversation(a.get_conversation(), name="default_alias")
        b.switch_conversation("default_alias")
        return a, b

    def test_agents_see_each_others_records_in_commit_order(self) -> None:
        a, b = self._pair()

        ra = a.invoke({"topic": "a", "tone": "strict"})
        rb = b.invoke({"topic": "b", "tone": "strict"})
        rc = a.invoke({"topic": "c", "tone": "strict"})

        shared = a.get_conversation()
        assert b.get_conversation() is shared
        assert [r.run_id for r in shared.records] == [ra.run_id, rb.run_id, rc.run_id]
        assert shared.records[1].prev is shared.records[0]
        assert shared.records[2].prev is shared.records[1]

    def test_agent_id_differs_per_writer(self) -> None:
        a, b = self._pair()

        a.invoke(_INPUTS)
        b.invoke(_INPUTS)

        first, second = a.get_conversation().records
        assert first.agent_id == f"{a.full_name};{a.instance_id}"
        assert second.agent_id == f"{b.full_name};{b.instance_id}"
        assert first.agent_id != second.agent_id

    def test_context_is_shared(self) -> None:
        a, b = self._pair()

        a.get_conversation().remember("k", 1)

        assert b.get_conversation().context["k"].value == 1

    def test_context_write_back_keeps_variables_in_the_shared_conversation(self) -> None:
        a, b = self._pair()
        a.get_conversation().remember("k", 1)

        b.invoke(_INPUTS)

        assert a.get_conversation().context["k"].value == 1
        assert len(a.get_conversation().records) == 1

    def test_removing_on_one_agent_leaves_the_other_intact(self) -> None:
        a, b = self._pair()
        a.invoke(_INPUTS)
        b.switch_conversation("default")

        removed = b.remove_conversation("default_alias")

        assert removed is a.get_conversation()
        assert len(a.get_conversation().records) == 1
        a.invoke(_INPUTS)
        assert len(a.get_conversation().records) == 2

    def test_clear_memory_on_one_agent_leaves_the_other_intact(self) -> None:
        a, b = self._pair()
        a.invoke(_INPUTS)

        b.clear_memory()

        assert len(a.get_conversation().records) == 1


class TestAgentId:
    def test_format_and_serialization(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke(_INPUTS)

        record = agent.get_conversation().records[0]

        assert record.agent_id == f"{agent.full_name};{agent.instance_id}"
        assert record.to_dict()["agent_id"] == record.agent_id

    def test_distinct_instances_get_distinct_ids(self) -> None:
        a, b = make_agent(record_enabled=True), make_agent(record_enabled=True)
        a.invoke(_INPUTS)
        b.invoke(_INPUTS)

        assert a.get_conversation().records[0].agent_id != b.get_conversation().records[0].agent_id


class TestGetRecord:
    def _agent(self) -> tuple[_EchoAgent, list[str]]:
        agent = make_agent(record_enabled=True)
        ids = [agent.invoke({"topic": t, "tone": "strict"}).run_id for t in ("a", "b")]
        agent.create_conversation("other")
        agent.switch_conversation("other")
        ids.append(agent.invoke({"topic": "c", "tone": "strict"}).run_id)
        return agent, ids

    def test_run_id_alone_scans_every_conversation(self) -> None:
        agent, ids = self._agent()

        assert agent.get_record(run_id=ids[0]).run_id == ids[0]
        assert agent.get_record(run_id=ids[2]).run_id == ids[2]

    def test_run_id_with_conversation_is_scoped(self) -> None:
        agent, ids = self._agent()

        assert agent.get_record(run_id=ids[2], conversation="other").run_id == ids[2]
        with pytest.raises(ValueError, match="no record"):
            agent.get_record(run_id=ids[0], conversation="other")

    def test_conversation_and_index(self) -> None:
        agent, ids = self._agent()

        assert agent.get_record(conversation="default", index=1).run_id == ids[1]
        assert agent.get_record(conversation="default", index=-1).run_id == ids[1]

    def test_index_alone_uses_the_active_conversation(self) -> None:
        agent, ids = self._agent()

        assert agent.get_record(index=0).run_id == ids[2]

    def test_unknown_run_id_raises_value_error(self) -> None:
        agent, _ = self._agent()

        with pytest.raises(ValueError, match="no record"):
            agent.get_record(run_id="nope")

    def test_unknown_conversation_raises(self) -> None:
        agent, ids = self._agent()

        with pytest.raises(AgentInvocationError):
            agent.get_record(conversation="nope", index=0)
        with pytest.raises(AgentInvocationError):
            agent.get_record(run_id=ids[0], conversation="nope")

    def test_out_of_range_index_raises_index_error(self) -> None:
        agent, _ = self._agent()

        with pytest.raises(IndexError):
            agent.get_record(conversation="default", index=9)

    def test_invalid_combinations_raise_value_error(self) -> None:
        agent, ids = self._agent()

        with pytest.raises(ValueError):
            agent.get_record()
        with pytest.raises(ValueError):
            agent.get_record(conversation="default")
        with pytest.raises(ValueError):
            agent.get_record(run_id=ids[0], index=0)
        with pytest.raises(ValueError):
            agent.get_record(run_id=ids[0], conversation="default", index=0)


class TestAgentValidation:
    def test_invoke_rejects_non_mapping_inputs(self) -> None:
        agent = make_agent()

        with pytest.raises(TypeError):
            agent.invoke(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_pre_invoke_returning_non_string_raises_at_invoke_time(self) -> None:
        agent = make_agent(pre_invoke=bad_pre_invoke)

        with pytest.raises(AgentInvocationError, match="pre_invoke returned a non-string result"):
            agent.invoke({"topic": "pytest"})

    def test_post_invoke_with_one_defaulted_parameter_is_allowed(self) -> None:
        agent = make_agent(post_invoke=defaulted_post_invoke)

        result = agent.invoke({"topic": "pytest", "tone": "strict"})

        assert result.result == "ECHO: WRITE ABOUT PYTEST IN A STRICT TONE."

    def test_post_invoke_two_required_args_auto_grafts_second(self) -> None:
        agent = make_agent(post_invoke=bad_post_two_required_args)

        result = agent.invoke({"topic": "pytest", "tone": "strict", "suffix": "!"})

        assert result.result == "ECHO: Write about pytest in a strict tone.!"

    def test_record_enabled_setter_rejects_non_bool(self) -> None:
        agent = make_agent()

        with pytest.raises(ValueError, match="record_enabled"):
            agent.record_enabled = "yes"  # type: ignore[assignment]

    def test_records_window_setter_rejects_negative_values(self) -> None:
        agent = make_agent()

        with pytest.raises(ValueError, match="records_window"):
            agent.records_window = -1

    def test_constructor_rejects_negative_records_window(self) -> None:
        with pytest.raises(AgentError, match="records_window"):
            make_agent(records_window=-1)

    def test_llm_engine_setter_rejects_non_llm_engine(self) -> None:
        agent = make_agent()

        with pytest.raises(TypeError, match="llm_engine"):
            agent.llm_engine = object()  # type: ignore[assignment]


class TestAgentNamespace:
    def test_namespace_is_required(self) -> None:
        with pytest.raises(TypeError):
            _MinimalAgent(
                name="a",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            )

    def test_agent_namespace_explicit(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="my_team",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
        )
        assert agent.namespace == "my_team"

    def test_agent_namespace_in_to_dict(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="my_team",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
        )
        assert agent.to_dict()["namespace"] == "my_team"


class TestAgentSerialization:
    def test_to_dict_includes_agent_configuration(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(
            engine=engine,
            record_enabled=True,
            records_window=1,
        )

        agent.invoke({"topic": "pytest", "tone": "strict"})

        data = agent.to_dict()

        assert data["type"] == "_EchoAgent"
        assert data["name"] == "writer_agent"
        assert data["description"] == "Deterministic writer test agent."
        assert data["record_enabled"] is True
        assert data["records_window"] == 1
        assert data["conversations"][agent.active_conversation] == [
            turn.to_dict() for turn in agent.get_conversation().records
        ]
        assert "system_prompts" in data
        assert data["pre_invoke"]["name"] == "pre_invoke"
        assert data["post_invoke"]["name"] == "post_invoke"
        assert data["bind_response_to"] == agent._bind_response_to == "result"
        assert "post_result_key" not in data
        assert data["llm"]["type"] == "FakeLLMEngine"
        assert "secret" not in str(data)

    def test_to_dict_includes_system_prompts_key(self) -> None:
        agent = make_agent()

        data = agent.to_dict()

        assert "system_prompts" in data
        assert "system" in data["system_prompts"]

    def test_to_dict_excludes_passthrough_inputs(self) -> None:
        agent = make_agent()

        data = agent.to_dict()

        assert "passthrough_inputs" not in data


class TestAgentAsyncInvoke:
    def test_async_invoke_shapes_prompt_and_post_invoke_packages_result(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(engine=engine)

        result = asyncio.run(
            agent.async_invoke({"topic": "pytest", "tone": "strict"})
        )

        expected_prompt = "Write about pytest in a strict tone."
        expected_raw = f"ECHO: {expected_prompt}"

        assert result.result == {
            "final": expected_raw,
            "length": len(expected_raw),
            "was_postprocessed": True,
        }
        assert engine.calls == [
            [
                {"role": "system", "content": ROLE_PROMPT},
                {"role": "user", "content": expected_prompt},
            ]
        ]

    def test_async_record_enabled_stores_history(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(engine=engine, record_enabled=True)

        first = asyncio.run(
            agent.async_invoke({"topic": "pytest", "tone": "strict"})
        )
        second = asyncio.run(
            agent.async_invoke({"topic": "agents", "tone": "concise"})
        )

        assert first.result["final"] == "ECHO: Write about pytest in a strict tone."
        assert second.result["final"] == "ECHO: Write about agents in a concise tone."

        rendered = agent.render_turn(agent.get_conversation().records[0])
        assert [m["role"] for m in rendered] == ["user", "assistant"]
        assert rendered[0]["content"] == "Write about pytest in a strict tone."
        assert rendered[1]["content"] == "ECHO: Write about pytest in a strict tone."

    def test_async_record_enabled_resends_prior_history_on_second_call(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_agent(engine=engine, record_enabled=True)

        asyncio.run(agent.async_invoke({"topic": "pytest", "tone": "strict"}))
        asyncio.run(agent.async_invoke({"topic": "agents", "tone": "concise"}))

        second_call_messages = engine.calls[1]

        assert [message["role"] for message in second_call_messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert second_call_messages[0]["content"] == ROLE_PROMPT
        assert second_call_messages[1]["content"] == "Write about pytest in a strict tone."
        assert second_call_messages[2]["content"] == "ECHO: Write about pytest in a strict tone."
        assert second_call_messages[3]["content"] == "Write about agents in a concise tone."

    def test_async_pre_invoke_returning_non_string_raises_at_invoke_time(self) -> None:
        agent = make_agent(pre_invoke=bad_pre_invoke)

        with pytest.raises(AgentInvocationError, match="pre_invoke returned a non-string result"):
            asyncio.run(agent.async_invoke({"topic": "pytest"}))

    def test_async_engine_non_string_response_raises_agent_invocation_error(self) -> None:
        agent = make_agent(engine=NonStringAsyncLLMEngine())

        with pytest.raises(AgentInvocationError, match="engine returned non-string"):
            asyncio.run(agent.async_invoke({"topic": "pytest", "tone": "strict"}))

    def test_async_invoke_routes_auto_grafted_param_to_post_invoke(self) -> None:
        agent = make_agent(post_invoke=post_with_suffix)

        result = asyncio.run(
            agent.async_invoke({"topic": "pytest", "tone": "strict", "suffix": "!"})
        )

        assert result.result == "ECHO: Write about pytest in a strict tone.!"


class TestAgentFrozenRenderingProperties:
    def test_response_preview_limit_is_not_an_agent_surface(self) -> None:
        agent = make_agent()

        assert not hasattr(agent, "response_preview_limit")
        assert "response_preview_limit" not in agent.to_dict()
        with pytest.raises(TypeError, match="response_preview_limit"):
            _EchoAgent(
                name="writer_agent",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                system_prompt=ROLE_PROMPT,
                response_preview_limit=5,  # type: ignore[call-arg]
            )

    def test_attach_api_removed(self) -> None:
        agent = make_agent()

        with pytest.raises(AttributeError):
            agent.attach("some/path.txt")  # type: ignore[attr-defined]

        with pytest.raises(AttributeError):
            agent.detach("some/path.txt")  # type: ignore[attr-defined]

        with pytest.raises(AttributeError):
            agent.clear_attachments()  # type: ignore[attr-defined]

        with pytest.raises(AttributeError):
            _ = agent.attachments  # type: ignore[attr-defined]


class TestAgentMutableRuntimeProperties:
    def test_pre_and_post_invoke_are_read_only_lifecycle_references(self) -> None:
        agent = make_agent()

        with pytest.raises(AttributeError):
            agent.pre_invoke = build_prompt  # type: ignore[misc]

        with pytest.raises(AttributeError):
            agent.post_invoke = package_response  # type: ignore[misc]

    def test_constructor_with_custom_pre_invoke_builds_parameters(self) -> None:
        agent = make_agent(pre_invoke=pre_with_two_fields)

        assert [param.name for param in agent.parameters] == ["subject", "style"]
        assert agent.invoke({"subject": "pytest", "style": "direct"}).result == {
            "final": "ECHO: pytest:direct",
            "length": len("ECHO: pytest:direct"),
            "was_postprocessed": True,
        }


class TestGetReservedParameters:
    """Coverage for Agent.get_reserved_parameters() -- default and override."""

    def test_base_agent_default_is_empty(self) -> None:
        assert Agent.get_reserved_parameters.__func__(Agent) == []

    def test_minimal_agent_default_is_dummy_reserved_only(self) -> None:
        assert _MinimalAgent.get_reserved_parameters() == [_DUMMY_RESERVED_PARAM]

    def test_subclass_override_extends_base_reserved_list(self) -> None:
        reserved = _ExtraReservedAgent.get_reserved_parameters()
        assert reserved == [_DUMMY_RESERVED_PARAM, _ExtraReservedAgent.EXTRA_RESERVED]

    def test_subclass_reserved_param_is_grafted_into_schema(self) -> None:
        agent = _ExtraReservedAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
        )
        names = [p.name for p in agent.parameters]
        assert "dummy_reserved" in names
        assert "trace_id" in names


class TestExtraParametersNormalization:
    """Coverage for extra_parameters normalization via to_paramspec_list."""

    def test_extra_parameters_none_produces_no_extra_schema_entries(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
        )
        # "prompt" comes from the built-in PROMPT_PARAM; no extra_parameters
        # means nothing else is grafted beyond the reserved dummy_reserved.
        assert [p.name for p in agent.parameters] == ["prompt", "dummy_reserved"]

    def test_extra_parameters_str_list_preserves_declared_kind(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            extra_parameters=["persona", "style"],
        )
        names = [p.name for p in agent.parameters]
        assert "persona" in names
        assert "style" in names
        # to_paramspec_list default for a bare string list is POSITIONAL_OR_KEYWORD,
        # not force-coerced to KEYWORD_ONLY (unlike the retired Graft-D behavior).
        persona_param = next(p for p in agent.parameters if p.name == "persona")
        assert persona_param.kind == "POSITIONAL_OR_KEYWORD"

    def test_extra_parameters_paramspec_list_preserves_declared_kind(self) -> None:
        specs = [
            ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="English"),
        ]
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            extra_parameters=specs,
        )
        lang_param = next(p for p in agent.parameters if p.name == "lang")
        assert lang_param.kind == "KEYWORD_ONLY"
        assert lang_param.default == "English"

    def test_extra_parameters_rejects_variadic_paramspec(self) -> None:
        specs = [
            ParamSpec(name="items", index=0, kind=ParamSpec.VAR_POSITIONAL, type="Any", default=NO_VAL),
        ]
        with pytest.raises(AgentError, match="variadic"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                extra_parameters=specs,
            )

    def test_extra_parameters_values_reach_invoke_via_inputs(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            extra_parameters=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="English"),
            ],
        )
        # _MinimalAgent ignores extra inputs entirely, but invoke() must
        # accept the extra top-level key without complaint (no more "context"
        # dict wrapper -- extra_parameters values are flat top-level keys now).
        result = agent.invoke({"prompt": "hi", "lang": "French"})
        assert result is not None


class TestReservedNameReconciliation:
    """Per-source reserved-name reconciliation: the new two-outcome model.

    ``_strip_reserved_overlaps`` (replacing the old three-tier
    ``_reconcile_reserved_names``) only has two outcomes: a declaration
    ``semantically_identical`` to the reserved spec is silently stripped (no
    warning at all -- a real behavior change from the old three-tier
    identical/compatible/incompatible scheme), and anything else raises
    ``AgentError`` immediately. There is deliberately no "compatible but not
    identical" warning tier on this axis anymore.
    """

    def test_identical_reserved_param_in_extra_parameters_is_silently_stripped(self) -> None:
        # _DUMMY_RESERVED_PARAM itself, declared verbatim -- guaranteed
        # field-for-field identical (name/type/kind/default/description all
        # match). Silent strip, no warning at all (dropped this pass).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            agent = _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                extra_parameters=[_DUMMY_RESERVED_PARAM],
            )
        assert caught == []
        # Only one dummy_reserved entry survives in the final schema (the framework's).
        assert [p.name for p in agent.parameters].count("dummy_reserved") == 1

    def test_pre_invoke_compatible_but_not_identical_dummy_reserved_now_raises(self) -> None:
        # Under the old three-tier scheme this warned ("compatible but not
        # identical"); under the new two-outcome model, anything short of
        # exact semantic identity raises -- there is no middle tier anymore.
        def pre_with_compatible_dummy_reserved(prompt: str, *, dummy_reserved: Any = "unset") -> str:
            return prompt

        with pytest.raises(AgentError, match="not identical"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                pre_invoke=pre_with_compatible_dummy_reserved,
            )

    def test_pre_invoke_incompatible_dummy_reserved_raises(self) -> None:
        def pre_with_bad_dummy_reserved(prompt: str, dummy_reserved: int) -> str:
            return prompt

        with pytest.raises(AgentError, match="not identical"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                pre_invoke=pre_with_bad_dummy_reserved,
            )

    def test_post_invoke_incompatible_dummy_reserved_raises_independently_of_clean_pre(self) -> None:
        # A clean pre_invoke (no reserved-name overlap at all) must not mask
        # a non-identical reserved-name collision detected independently in
        # post_invoke -- each of the three sources (pre/post/extra) is
        # checked entirely on its own, never against each other.
        def post_with_bad_dummy_reserved(result: str, dummy_reserved: int) -> str:
            return result

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(AgentError, match="not identical"):
                _MinimalAgent(
                    name="a",
                    namespace="tests",
                    description="d",
                    llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                    pre_invoke=build_prompt,
                    post_invoke=post_with_bad_dummy_reserved,
                    bind_response_to="result",
                )

    def test_extra_parameters_incompatible_dummy_reserved_raises(self) -> None:
        bad = ParamSpec(name="dummy_reserved", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int", default=NO_VAL)
        with pytest.raises(AgentError, match="not identical"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                extra_parameters=[bad],
            )

    def test_variadic_param_named_like_reserved_is_rejected_by_the_kwargs_ban(self) -> None:
        # A VAR_KEYWORD catch-all happens to be named "dummy_reserved". The
        # reserved-overlap check exempts variadics, but the **kwargs ban on
        # pre_invoke now rejects it at construction (AgentError) before any
        # reserved handling or schema grafting runs.
        def pre_with_dummy_reserved_kwargs(prompt: str, **dummy_reserved: Any) -> str:
            return prompt

        with pytest.raises(AgentError, match=r"\*\*kwargs"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                pre_invoke=pre_with_dummy_reserved_kwargs,
            )

    def test_multiple_compatible_but_not_identical_overlaps_emit_exactly_one_grouped_warning(self) -> None:
        # Two independently compatible-but-not-identical peer overlaps
        # (topic, tone -- both shared between pre_invoke and post_invoke)
        # must batch into exactly ONE UserWarning event for the whole
        # construction call, not one per name.
        def post_with_two_overlaps(
            result: str, topic: str = "default-topic", tone: str = "default-tone"
        ) -> str:
            return result

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_agent(post_invoke=post_with_two_overlaps)

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "topic" in str(user_warnings[0].message)
        assert "tone" in str(user_warnings[0].message)

    def test_same_named_variadic_with_conflicting_kind_across_sources_raises(self) -> None:
        # A same-named variadic declared under different kinds by two peer
        # sources (pre_invoke's *extra vs post_invoke's **extra) now fails
        # earlier, at the **kwargs ban on post_invoke.
        def pre_with_var_positional(*extra: Any) -> str:
            return str(extra)

        def post_with_var_keyword(result: str, **extra: Any) -> str:
            return result

        with pytest.raises(AgentError, match=r"\*\*kwargs"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                pre_invoke=pre_with_var_positional,
                post_invoke=post_with_var_keyword,
                bind_response_to="result",
            )


class TestBindResponseToSharedNames:
    """A destination that another source (pre_invoke / extra_parameters) also
    declares, and a destination that is a reserved name."""

    def test_bound_name_declared_by_pre_must_be_type_compatible(self) -> None:
        def pre(prompt: str, x: int = 7) -> str:
            return prompt

        def post(x: str = "d") -> str:
            return x

        with pytest.raises(AgentError, match="no compatible reconciliation"):
            make_agent(pre_invoke=pre, post_invoke=post, bind_response_to="x")

    def test_explicit_binding_to_non_first_parameter_accepted(self) -> None:
        agent = make_agent(post_invoke=post_with_custom_key, bind_response_to="suffix")
        assert agent._bind_response_to == "suffix"

    def test_binding_to_a_name_no_post_parameter_declares_is_accepted(self) -> None:
        agent = make_agent(post_invoke=post_with_custom_key, bind_response_to="nope")
        assert agent._bind_response_to == "nope"

    def test_collision_with_pre_invoke_name_is_allowed(self) -> None:
        agent = make_agent(
            post_invoke=lambda topic: topic,
            bind_response_to="topic",
        )
        assert agent._bind_response_to == "topic"

    def test_collision_with_extra_parameters_name_is_allowed(self) -> None:
        def post_with_flag(result: str, flag: str = "x") -> str:
            return result

        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            post_invoke=post_with_flag,
            bind_response_to="flag",
            extra_parameters=["flag"],
        )
        assert agent._bind_response_to == "flag"

    def test_binding_to_a_reserved_parameter_raises(self) -> None:
        # Built on _MinimalAgent (not make_agent()'s _EchoAgent, which has no
        # reserved params) so there's a real reserved name to collide with.
        def post_with_dummy_reserved_param(result: str, dummy_reserved: str = "x") -> str:
            return result

        with pytest.raises(AgentError, match="framework-reserved"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                post_invoke=post_with_dummy_reserved_param,
                bind_response_to="dummy_reserved",
            )


class TestPreVsPostReconciliation:
    """Named for the overlap it exercises (pre_invoke vs. post_invoke); the
    underlying mechanism is now the single unified N-way peer report over
    all three sources, not a distinct pairwise fold stage -- these tests
    stay valid because a 2-source-only overlap is a degenerate case of the
    N-way report."""

    def test_incompatible_overlap_raises(self) -> None:
        def post_with_int_topic(result: str, topic: int) -> str:
            return result

        with pytest.raises(AgentError, match="no compatible reconciliation"):
            make_agent(post_invoke=post_with_int_topic)

    def test_compatible_but_not_identical_overlap_warns(self) -> None:
        def post_with_defaulted_topic(result: str, topic: str = "default-topic") -> str:
            return result

        with pytest.warns(UserWarning, match="not identical"):
            make_agent(post_invoke=post_with_defaulted_topic)

    def test_identical_overlap_is_silent(self) -> None:
        def pre_identical(topic: str) -> str:
            return topic

        def post_identical(result: str, topic: str) -> str:
            return result

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            make_agent(pre_invoke=pre_identical, post_invoke=post_identical)
        assert not any("topic" in str(w.message) for w in caught)

    def test_post_only_remainder_keeps_original_declared_kind(self) -> None:
        # A post-only VAR_POSITIONAL remainder now grafts preserving its own
        # kind (dropped: the old unconditional exclusion of variadic-kind
        # post-only params).
        def post_with_var_positional(result: str, *extras: Any) -> str:
            return result

        agent = make_agent(post_invoke=post_with_var_positional)
        extras_param = next(p for p in agent.parameters if p.name == "extras")
        assert extras_param.kind == "VAR_POSITIONAL"

    def test_post_only_remainder_lands_via_insert_by_category(self) -> None:
        agent = make_agent(post_invoke=post_with_suffix)
        names = [p.name for p in agent.parameters]
        # topic (pre, required) and suffix (post-only, required) both sort
        # ahead of tone (pre, defaulted) within the POSITIONAL_OR_KEYWORD
        # category -- insert_by_category enforces required-before-defaulted
        # regardless of which source declared the parameter.
        assert names == ["topic", "suffix", "tone"]


class TestCombinedVsExtraReconciliation:
    """Named for the overlap it exercises (pre_invoke/post_invoke's combined
    schema vs. extra_parameters); the underlying mechanism is now the same
    single unified N-way peer report described in TestPreVsPostReconciliation,
    not a distinct second fold stage."""

    def test_incompatible_overlap_raises(self) -> None:
        with pytest.raises(AgentError, match="no compatible reconciliation"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                pre_invoke=build_prompt,
                extra_parameters=[
                    ParamSpec(name="topic", index=0, kind=ParamSpec.KEYWORD_ONLY, type="int", default=NO_VAL),
                ],
            )

    def test_compatible_but_not_identical_overlap_warns_and_combined_wins(self) -> None:
        with pytest.warns(UserWarning, match="not identical"):
            agent = _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                pre_invoke=build_prompt,
                extra_parameters=[
                    ParamSpec(name="topic", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="fallback"),
                ],
            )
        topic_param = next(p for p in agent.parameters if p.name == "topic")
        # combined (pre_invoke's) declaration wins: required, no default.
        assert topic_param.default is NO_VAL
        assert topic_param.kind == "POSITIONAL_OR_KEYWORD"

    def test_identical_overlap_is_silent(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                pre_invoke=build_prompt,
                extra_parameters=[
                    ParamSpec(name="topic", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="str", default=NO_VAL),
                ],
            )
        assert not any("topic" in str(w.message) for w in caught)

    def test_extra_remainder_grafted_via_insert_by_category(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            pre_invoke=pre_with_two_fields,
            extra_parameters=[
                ParamSpec(name="flag", index=0, kind=ParamSpec.KEYWORD_ONLY, type="bool", default=False),
            ],
        )
        names = [p.name for p in agent.parameters]
        assert names == ["subject", "style", "flag", "dummy_reserved"]


def pre_with_dict_shared(shared: dict[str, int]) -> str:
    return str(shared)


def post_with_bare_dict_shared(result: str, shared: dict) -> str:
    return result


class TestNWayWitnessOrderIndependence:
    """The N-way type-witness search (utils.parameters.n_way_type_witness)
    replaces Agent's old two-step accumulator fold (pre-vs-post, then
    combined-vs-extra), which always kept pre_invoke's own declaration
    unmodified as the fold "winner" and discarded post_invoke's full type
    declaration for any overlapping name -- meaning the old fold's final
    verdict for a name shared by all three sources reduced to a bare
    pre_invoke-vs-extra_parameters pairwise check, silently blind to
    whatever bridging information only post_invoke held.

    NOTE ON THE BRAINSTORM DOC'S OWN WORKED EXAMPLE: the brainstorm doc's
    illustrative "concrete break" example (pre=(int,), post=(int,str),
    extra=(str,)) is NOT reproduced verbatim here. Verified directly against
    both the shipped n_way_type_witness algorithm and a hand-simulation of
    the old fold: that specific example is a genuine "triangle" conflict
    (see the spec's own Helly-counterexample discussion) -- pre requires
    strictly int, extra requires strictly str, and no token can satisfy both
    regardless of what post additionally accepts. It legitimately raises
    under BOTH the old fold and the new witness search (same final verdict,
    for different, more principled reasons under the new one). Reproduced
    below instead: a corrected, hand-verified example using generic/
    parameterized types, where the old fold provably raises (a real false
    collision, order-dependent-masking bug) and the new witness search
    provably finds a genuine bridging witness and succeeds.
    """

    def test_generic_bridging_witness_construction_succeeds_where_the_old_fold_would_raise(self) -> None:
        # pre declares "shared: dict[str, int]", post declares "shared: dict"
        # (bare origin -- a generic supertype of both siblings), extra
        # declares "shared: dict[str, str]" via an explicit ParamSpec.
        # pre and extra do NOT share a directly compatible token
        # (dict[str, int] vs dict[str, str] disagree on the value-type
        # argument) -- a pairwise pre-vs-extra check, which is what the old
        # fold reduces to once pre "wins" against post (discarding post's
        # bare "dict" declaration), would raise. The N-way witness search
        # instead finds "dict" -- contributed uniquely by post's own bare
        # declaration -- structurally compatible with all three tuples at
        # once, and construction succeeds.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            agent = _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                pre_invoke=pre_with_dict_shared,
                post_invoke=post_with_bare_dict_shared,
                bind_response_to="result",
                extra_parameters=[
                    ParamSpec(
                        name="shared", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                        type="dict[str, str]", default=NO_VAL,
                    ),
                ],
            )

        shared_param = next(p for p in agent.parameters if p.name == "shared")
        assert shared_param.type == ("dict",)
        # All three observations differ (dict[str, int] vs dict vs
        # dict[str, str]) -- compatible-not-identical, so exactly one
        # grouped warning is expected.
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "shared" in str(user_warnings[0].message)

    def test_pairwise_compatible_triangle_with_no_universal_witness_still_raises(self) -> None:
        # The brainstorm doc's own literal numeric example (pre=(int,),
        # post=(int,str), extra=(str,)) -- verified (see class docstring) to
        # be a genuine "triangle" conflict, not a false collision: pre and
        # extra are flatly, directly incompatible (int vs str, neither
        # "Any"), and no token in the pool can satisfy both regardless of
        # what post additionally accepts. The witness-set standard correctly
        # raises here -- unlike a weaker pairwise-all-pairs check, which
        # would wrongly call this compatible (pre-post share int, post-extra
        # share str) despite there being no principled merged type.
        def pre_int_shared(shared: int) -> str:
            return str(shared)

        def post_int_or_str_shared(result: str, shared: int | str) -> str:
            return result

        with pytest.raises(AgentError, match="no compatible reconciliation"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                pre_invoke=pre_int_shared,
                post_invoke=post_int_or_str_shared,
                bind_response_to="result",
                extra_parameters=[
                    ParamSpec(name="shared", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="str", default=NO_VAL),
                ],
            )


class TestFinalReservedGraftOrdering:
    def test_reserved_graft_valid_ordering_with_positional_only_pre(self) -> None:
        # Built on _MinimalAgent (not make_agent()'s _EchoAgent, which has no
        # reserved params) so the reserved graft's ordering claim has a real
        # KEYWORD_ONLY reserved param to land after the POSITIONAL_ONLY pre
        # param.
        def pre_positional_only(subject: str, /) -> str:
            return subject

        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            pre_invoke=pre_positional_only,
        )
        names = [p.name for p in agent.parameters]
        assert names == ["subject", "dummy_reserved"]
        assert agent.parameters[0].kind == "POSITIONAL_ONLY"
        assert agent.parameters[-1].kind == "KEYWORD_ONLY"

    def test_reserved_graft_after_extra_keyword_only_params(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            extra_parameters=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="English"),
            ],
        )
        names = [p.name for p in agent.parameters]
        # "prompt" (built-in PROMPT_PARAM, POSITIONAL_OR_KEYWORD) sorts ahead of
        # "lang" (extra, KEYWORD_ONLY); combined-vs-extra
        # reconciliation appends extra's remainder after pre's, and the final
        # reserved graft appends dummy_reserved last.
        assert names == ["prompt", "lang", "dummy_reserved"]


class TestUpdatePrompt:
    def test_system_prompts_property_returns_shallow_copy(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
        )
        copy = agent.system_prompts
        copy["mutated"] = PromptConfig(template="x", description="d")

        assert "mutated" not in agent.system_prompts


class TestAgentRecordsNeverStoredWhenRecordDisabled:
    def test_record_disabled_records_never_stored(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=engine,
            record_enabled=False,
        )

        agent.invoke({"prompt": "hello"})
        agent.invoke({"prompt": "world"})

        assert agent.get_conversation().records == []

    def test_records_window_set_to_zero_at_runtime_suppresses_reads(self) -> None:
        # record_enabled no longer has any bearing on reads post-decoupling
        # -- records_window is the only runtime knob that can suppress them,
        # which this test now exercises instead (same intent: toggling a
        # knob mid-stream stops prior turns from being read back).
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=engine,
            record_enabled=True,
        )
        agent.invoke({"prompt": "first"})

        agent.records_window = 0
        agent.invoke({"prompt": "second"})

        # Second call's messages should have no prior-turn content
        second_call = engine.calls[1]
        assert len(second_call) == 2  # just system + user from _MinimalAgent


class TestAgentDescriptionOverrideRemoval:
    def test_agent_description_follows_base_getter(self) -> None:
        # Built on _MinimalAgent (not make_agent()'s _EchoAgent, which has
        # no reserved params) so the description-bullet-augmentation
        # mechanism has a real described reserved param to demonstrate.
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="Deterministic writer test agent.",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
        )

        assert agent.description.startswith("Deterministic writer test agent.")
        assert "- dummy_reserved:" in agent.description

    def test_agent_description_contains_dummy_reserved_description_text(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="Deterministic writer test agent.",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
        )

        assert "Test-only reserved parameter" in agent.description

    def test_agent_to_dict_description_is_raw(self) -> None:
        agent = make_agent()

        assert agent.to_dict()["description"] == "Deterministic writer test agent."


class TestInvocationLifecycle:
    """Reserved-name exclusion from pre/post slices, full inputs threading
    to create_task, and AgentRecord.inputs widening."""

    def test_invoke_receives_full_unsliced_inputs_in_invoke_arg(self) -> None:
        captured: dict[str, Any] = {}

        class CapturingAgent(_MinimalAgent):
            def create_task(self, inputs):  # type: ignore[override]
                captured.update(inputs)
                return super().create_task(inputs)

        agent = CapturingAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            extra_parameters=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="English"),
            ],
        )
        agent.invoke({"prompt": "hi", "lang": "French"})

        assert captured == {"prompt": "hi", "lang": "French", "dummy_reserved": None}

    def test_reserved_name_excluded_from_pre_invoke_slice_even_if_declared_identically(self) -> None:
        """A source declaring a reserved param byte-for-byte identical to the
        framework's own spec survives construction (silently stripped, no
        warning -- under the old three-tier scheme this was merely
        "compatible," which is no longer a survivable outcome on its own;
        exact identity is required now). But surviving construction must not
        mean the framework's real reserved value gets slotted into that
        source's own declaration at invoke time -- the exclusion applies at
        every invoke() regardless. This is invoke-time routing behavior,
        distinct from what TestReservedNameReconciliation covers
        (construction-time strip/raise only)."""
        seen: list[Any] = []

        def pre_with_identical_dummy_reserved(
            prompt: str, *, dummy_reserved: Annotated[Optional[str], _DUMMY_RESERVED_PARAM.description] = None
        ) -> str:
            seen.append(dummy_reserved)
            return prompt

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            agent = _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                pre_invoke=pre_with_identical_dummy_reserved,
                record_enabled=True,
            )
        assert caught == []

        agent.invoke({"prompt": "hi"})
        agent.invoke({"prompt": "again", "dummy_reserved": "some-value"})

        # Even on the second call, where the caller explicitly supplies a
        # real, non-default reserved value, pre_invoke's own declaration
        # never receives it -- both calls see only pre_invoke's own default
        # (None), confirming the exclusion holds at invoke time and isn't
        # just an artifact of the value happening to be None on a single,
        # default-only call.
        assert seen == [None, None]

    def test_committed_agent_record_inputs_equals_full_filtered_inputs(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            extra_parameters=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="English"),
            ],
        )
        agent.invoke({"prompt": "hi", "lang": "French"})

        assert agent.get_conversation().records[0].inputs == {"prompt": "hi", "lang": "French", "dummy_reserved": None}

    def test_async_committed_agent_record_inputs_equals_full_filtered_inputs(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            extra_parameters=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="English"),
            ],
        )
        asyncio.run(agent.async_invoke({"prompt": "hi", "lang": "French"}))

        assert agent.get_conversation().records[0].inputs == {"prompt": "hi", "lang": "French", "dummy_reserved": None}

    def test_inputs_isolated_across_invocations(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            extra_parameters=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="English"),
            ],
        )
        agent.invoke({"prompt": "hi"})
        agent.invoke({"prompt": "yo"})
        history = agent.get_conversation().records
        assert history[0].inputs is not history[1].inputs


# ─────────────────────────────────────────────────────────────────────────────
# Public lifecycle steps (create_task / summarize / pre_process / post_process
# / record)
# ─────────────────────────────────────────────────────────────────────────────
_INPUTS = {"topic": "pytest", "tone": "strict"}


def _run_until_post(agent: Agent, inputs: Mapping[str, Any]) -> tuple[AgentTask, Any]:
    """Hand-assembled lifecycle up to (not including) ``record``."""
    task = agent.create_task(agent.filter_inputs(inputs))
    task = agent.summarize(task)
    task = agent.pre_process(task)
    while not task.complete:
        task = agent.think(task)
        task = agent.prepare(task)
        task = agent.act(task)
    return task, agent.post_process(task)


class TestCreateTask:
    def test_binds_conversation_and_mints_identity(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.create_conversation("side")
        agent.switch_conversation("side")

        task = agent.create_task(agent.filter_inputs(_INPUTS))

        assert task.conversation == "side"
        assert task.run_id and task.started_at is not None and task.ended_at is None
        assert task.context == {}
        assert task.user_prompt == ""

    def test_run_ids_are_distinct_per_task(self) -> None:
        agent = make_agent()
        inputs = agent.filter_inputs(_INPUTS)
        assert agent.create_task(inputs).run_id != agent.create_task(inputs).run_id

    @pytest.mark.parametrize(
        ("window", "expected_topics"),
        [(None, ["a", "b", "c"]), (2, ["b", "c"]), (0, [])],
    )
    def test_turns_follow_records_window(self, window, expected_topics) -> None:
        agent = make_agent(record_enabled=True)
        for topic in ("a", "b", "c"):
            agent.invoke({"topic": topic, "tone": "plain"})
        agent.records_window = window

        task = agent.create_task(agent.filter_inputs(_INPUTS))

        assert [t.inputs["topic"] for t in task.turns] == expected_topics

    def test_turns_are_a_pinned_copy(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke({"topic": "a", "tone": "plain"})
        task = agent.create_task(agent.filter_inputs(_INPUTS))

        agent.invoke({"topic": "b", "tone": "plain"})

        assert len(task.turns) == 1

    def test_task_inputs_is_the_full_filtered_mapping(self) -> None:
        agent = make_agent()
        inputs = agent.filter_inputs(_INPUTS)
        assert agent.create_task(inputs).inputs is inputs

    def test_views_are_slices_not_a_partition(self) -> None:
        def pre(subject: str, shared: str = "p") -> str:
            return f"{subject}:{shared}"

        def post(result: str, shared: str = "q", only_post: str = "z") -> str:
            return f"{result}{shared}{only_post}"

        agent = _MinimalAgent(
            name="a", namespace="tests", description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            pre_invoke=pre, post_invoke=post, bind_response_to="result",
            extra_parameters=[ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="en")],
        )
        inputs = agent.filter_inputs(
            {"subject": "s", "shared": "x", "only_post": "y", "lang": "fr"}
        )

        task = agent.create_task(inputs)

        assert task.pre_inputs == {"subject": "s", "shared": "x"}
        assert task.post_inputs == {"shared": "x", "only_post": "y"}
        # reserved + extra_parameters names only; the reserved name is in no
        # other view.
        assert task.agentic_inputs == {"lang": "fr", "dummy_reserved": None}
        assert "dummy_reserved" not in task.pre_inputs
        assert "dummy_reserved" not in task.post_inputs


class TestPreviewLimitRendering:
    """History is limited by the bound conversation's limits, read at build time."""

    def _agent(self, **limits: int | None) -> _EchoAgent:
        agent = make_agent(record_enabled=True)
        agent.invoke({"topic": "pytest", "tone": "strict"})
        conv = agent.get_conversation()
        for name, value in limits.items():
            setattr(conv, name, value)
        return agent

    def _history(self, agent: _EchoAgent) -> list[dict[str, str]]:
        task = agent.summarize(agent.create_task(agent.filter_inputs(_INPUTS)))
        return task.historic_messages

    def test_no_limits_truncates_nothing(self) -> None:
        history = self._history(self._agent())

        assert history[0]["content"] == "Write about pytest in a strict tone."
        assert history[1]["content"] == "ECHO: Write about pytest in a strict tone."

    def test_response_and_prompt_limits_use_the_marker(self) -> None:
        history = self._history(
            self._agent(response_preview_limit=4, prompt_preview_limit=5)
        )

        assert history[0]["content"] == "Write...[truncated: 5 of 36 chars shown]"
        assert history[1]["content"] == "ECHO...[truncated: 4 of 42 chars shown]"

    def test_zero_hides_each_side_independently(self) -> None:
        history = self._history(self._agent(response_preview_limit=0))

        assert history[0]["content"] == "Write about pytest in a strict tone."
        assert history[1]["content"] == "value hidden"

    def test_limit_edit_between_summaries_is_picked_up(self) -> None:
        agent = self._agent()
        assert self._history(agent)[1]["content"].startswith("ECHO: Write")

        agent.get_conversation().response_preview_limit = 2

        assert self._history(agent)[1]["content"].startswith("EC...[truncated")

    def test_limits_come_from_the_task_conversation_not_the_active_one(self) -> None:
        agent = self._agent(response_preview_limit=3)
        agent.create_conversation("other")
        task = agent.create_task(agent.filter_inputs(_INPUTS))
        agent.switch_conversation("other")
        agent.get_conversation("other").response_preview_limit = 0

        task = agent.summarize(task)

        assert task.historic_messages[1]["content"].startswith("ECH...[truncated")

    def test_direct_render_turn_with_no_kwargs_truncates_nothing(self) -> None:
        agent = self._agent(response_preview_limit=1, prompt_preview_limit=1)

        rendered = agent.render_turn(agent.get_conversation().records[0])

        assert rendered[0]["content"] == "Write about pytest in a strict tone."
        assert rendered[1]["content"] == "ECHO: Write about pytest in a strict tone."

    def test_render_turn_accepts_explicit_limits(self) -> None:
        agent = self._agent()

        rendered = agent.render_turn(
            agent.get_conversation().records[0],
            response_preview_limit=4,
            prompt_preview_limit=0,
        )

        assert rendered[0]["content"] == "value hidden"
        assert rendered[1]["content"] == "ECHO...[truncated: 4 of 42 chars shown]"


class TestHistoryRenderMode:
    def test_default_is_str_and_str_responses_stay_plain(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke({"topic": "pytest", "tone": "strict"})

        assert agent.get_conversation().render_mode == "s"
        task = agent.summarize(agent.create_task(agent.filter_inputs(_INPUTS)))
        assert task.historic_messages[1]["content"] == "ECHO: Write about pytest in a strict tone."

    def test_conversation_mode_drives_history_not_the_prompt(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke({"topic": "pytest", "tone": "strict"})
        agent.get_conversation().render_mode = "j"

        task = agent.summarize(agent.create_task(agent.filter_inputs(_INPUTS)))

        assert task.historic_messages[0]["content"] == "Write about pytest in a strict tone."
        assert task.historic_messages[1]["content"] == '"ECHO: Write about pytest in a strict tone."'

    def test_create_import_and_slice_carry_the_mode(self) -> None:
        agent = make_agent(record_enabled=True)
        created = agent.create_conversation("scratch", render_mode="p")
        assert created.render_mode == "p"

        sliced = created.slice()
        assert sliced.render_mode == "p"
        target = make_agent()
        target.import_conversation(created, name="imp")
        assert target.get_conversation("imp").render_mode == "p"

    def test_factory_reset_restores_the_default(self) -> None:
        agent = make_agent()
        agent.get_conversation().render_mode = "j"

        agent.clear_memory()

        assert agent.get_conversation().render_mode == "s"

    def test_invalid_mode_rejected(self) -> None:
        agent = make_agent()

        with pytest.raises(ValueError, match="render_mode"):
            agent.create_conversation("scratch", render_mode="nope")
        with pytest.raises(ValueError, match="render_mode"):
            agent.get_conversation().render_mode = None  # type: ignore[assignment]
        assert agent.get_conversation().render_mode == "s"


class TestFactoryReset:
    def _with_limits(self) -> _EchoAgent:
        agent = make_agent(record_enabled=True)
        conv = agent.get_conversation()
        conv.response_preview_limit = 5
        conv.prompt_preview_limit = 6
        conv.context_preview_limit = 7
        return agent

    def test_clear_memory_resets_default_limits_to_none(self) -> None:
        agent = self._with_limits()
        agent.create_conversation("scratch", response_preview_limit=1)

        agent.clear_memory()

        conv = agent.get_conversation()
        assert conv.response_preview_limit is None
        assert conv.prompt_preview_limit is None
        assert conv.context_preview_limit is None
        assert agent.conversation_names == ["default"]

    def test_remove_default_resets_limits_and_returns_the_old_one(self) -> None:
        agent = self._with_limits()

        old = agent.remove_conversation("default")

        assert old.response_preview_limit == 5
        fresh = agent.get_conversation("default")
        assert fresh is not old
        assert fresh.response_preview_limit is None
        assert fresh.prompt_preview_limit is None
        assert fresh.context_preview_limit is None


class TestRememberRenderOptions:
    def test_remember_passes_render_mode_and_preview_limit_through(self) -> None:
        agent = make_agent()

        agent.get_conversation().remember("k", {"a": 1}, render_mode="j", preview_limit=4)

        stored = agent.get_conversation().context["k"]
        assert stored.render_mode == "j"
        assert stored.preview_limit == 4

    def test_remember_is_add_only_and_update_keeps_options(self) -> None:
        agent = make_agent()
        agent.get_conversation().remember("k", 1, render_mode="r", preview_limit=2)

        with pytest.raises(ValueError, match="already exists"):
            agent.get_conversation().remember("k", 2)
        agent.get_conversation().update("k", 2)

        stored = agent.get_conversation().context["k"]
        assert stored.value == 2
        assert stored.render_mode == "r"
        assert stored.preview_limit == 2

    def test_remember_targets_a_named_conversation(self) -> None:
        agent = make_agent()
        agent.create_conversation("scratch")

        agent.get_conversation("scratch").remember("k", 1, render_mode="a")

        assert agent.get_conversation("scratch").context["k"].render_mode == "a"
        assert "k" not in agent.get_conversation().context

    def test_remember_rejects_bad_options(self) -> None:
        agent = make_agent()

        with pytest.raises(ValueError, match="render_mode"):
            agent.get_conversation().remember("k", 1, render_mode="nope")
        with pytest.raises(ValueError, match="preview_limit"):
            agent.get_conversation().remember("k", 1, preview_limit=-1)
        assert "k" not in agent.get_conversation().context


class TestSummarizeAndPreProcess:
    def test_summarize_renders_window_relative_history(self) -> None:
        agent = make_agent(record_enabled=True, records_window=1)
        agent.invoke({"topic": "a", "tone": "plain"})
        agent.invoke({"topic": "b", "tone": "plain"})

        task = agent.summarize(agent.create_task(agent.filter_inputs(_INPUTS)))

        assert [m["role"] for m in task.historic_messages] == ["user", "assistant"]
        assert "b" in task.historic_messages[0]["content"]

    def test_summarize_with_no_turns_leaves_history_empty(self) -> None:
        agent = make_agent()
        task = agent.summarize(agent.create_task(agent.filter_inputs(_INPUTS)))
        assert task.historic_messages == []

    def test_async_summarize_matches_sync(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke({"topic": "a", "tone": "plain"})
        task = agent.create_task(agent.filter_inputs(_INPUTS))
        expected = agent.summarize(agent.create_task(agent.filter_inputs(_INPUTS))).historic_messages

        task = asyncio.run(agent.async_summarize(task))

        assert task.historic_messages == expected

    def test_pre_process_sets_and_overwrites_user_prompt(self) -> None:
        agent = make_agent()
        task = agent.create_task(agent.filter_inputs(_INPUTS))
        task.user_prompt = "stale"

        task = agent.pre_process(task)

        assert task.user_prompt == "Write about pytest in a strict tone."

    def test_pre_process_non_string_raises(self) -> None:
        def pre_any(prompt: str) -> Any:
            return 123

        agent = make_agent(pre_invoke=pre_any)
        task = agent.create_task(agent.filter_inputs({"prompt": "x"}))
        with pytest.raises(AgentInvocationError, match="non-string"):
            agent.pre_process(task)

    def test_async_pre_process_matches_sync(self) -> None:
        agent = make_agent()
        task = agent.create_task(agent.filter_inputs(_INPUTS))
        task = asyncio.run(agent.async_pre_process(task))
        assert task.user_prompt == "Write about pytest in a strict tone."


class TestPostProcessAndRecord:
    def test_post_process_builds_result_from_the_task(self) -> None:
        agent = make_agent()
        task, result = _run_until_post(agent, _INPUTS)

        assert result.run_id == task.run_id
        assert result.started_at == task.started_at
        assert result.ended_at == task.ended_at and task.ended_at is not None
        assert result.result["final"] == "ECHO: Write about pytest in a strict tone."

    def test_post_process_overlays_the_response_onto_post_inputs(self) -> None:
        agent = make_agent(post_invoke=post_with_suffix)
        task = agent.pre_process(agent.create_task(agent.filter_inputs({**_INPUTS, "suffix": "!"})))
        while not task.complete:
            task = agent.act(task)
        before = dict(task.post_inputs)

        agent.post_process(task)

        assert task.post_inputs == {**before, "result": task.generated_response}

    def test_async_post_process_matches_sync(self) -> None:
        agent = make_agent()
        task = agent.pre_process(agent.create_task(agent.filter_inputs(_INPUTS)))
        while not task.complete:
            task = agent.act(task)
        sync = agent.post_process(task)

        asynchronous = asyncio.run(agent.async_post_process(task))

        assert asynchronous.result == sync.result
        assert asynchronous.run_id == sync.run_id

    def test_record_disabled_stores_nothing(self) -> None:
        agent = make_agent(record_enabled=False)
        task, _ = _run_until_post(agent, _INPUTS)
        agent.record(task)
        assert agent.get_conversation().records == []

    def test_record_appends_with_task_identity_and_prev(self) -> None:
        agent = make_agent(record_enabled=True)
        first_task, _ = _run_until_post(agent, _INPUTS)
        agent.record(first_task)
        second_task, _ = _run_until_post(agent, _INPUTS)
        agent.record(second_task)

        first, second = agent.get_conversation().records
        assert first.run_id == first_task.run_id and first.prev is None
        assert second.run_id == second_task.run_id and second.prev is first
        assert first.children == [second]

    def test_racing_calls_both_append_in_commit_order_without_forking(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke(_INPUTS)
        slow_task, _ = _run_until_post(agent, _INPUTS)  # read the 1-record tail
        agent.invoke(_INPUTS)  # a racing call commits first

        agent.record(slow_task)

        records = agent.get_conversation().records
        assert list(agent._conversations) == ["default"]
        assert [r.run_id for r in records][-1] == slow_task.run_id
        assert len(records) == 3
        assert records[0].children == [records[1], records[2]]

    def test_record_at_window_zero_appends_a_root(self) -> None:
        agent = make_agent(record_enabled=True, records_window=0)
        agent.invoke(_INPUTS)
        slow_task, _ = _run_until_post(agent, _INPUTS)
        agent.invoke(_INPUTS)

        agent.record(slow_task)

        records = agent.get_conversation().records
        assert len(records) == 3 and list(agent._conversations) == ["default"]
        assert all(r.prev is None for r in records)

    def test_hand_assembled_lifecycle_matches_invoke(self) -> None:
        manual = make_agent(record_enabled=True)
        task, result = _run_until_post(manual, _INPUTS)
        manual.record(task)

        automatic = make_agent(record_enabled=True)
        auto_result = automatic.invoke(_INPUTS)

        assert result.result == auto_result.result
        m, a = manual.get_conversation().records[0], automatic.get_conversation().records[0]
        assert (m.user_prompt, m.generated_response, m.inputs) == (a.user_prompt, a.generated_response, a.inputs)



class TestContextLifecycle:
    @staticmethod
    def _bump(task: AgentTask, key: str, value: Any) -> None:
        task.context[key] = dataclasses.replace(task.context[key], value=value)

    def test_seed_is_a_distinct_equal_copy(self) -> None:
        agent = make_agent()
        agent.get_conversation().remember("k", 1)
        conv = agent.get_conversation()

        task = agent.create_task(agent.filter_inputs(_INPUTS))

        assert task.context == dict(conv.context)
        assert task.context["k"] is conv.context["k"]
        task.context["x"] = ContextVariable(name="x", value=1)
        assert "x" not in agent.get_conversation().context

    def test_seed_is_a_snapshot_at_create_task(self) -> None:
        agent = make_agent()
        task = agent.create_task(agent.filter_inputs(_INPUTS))
        agent.get_conversation().remember("k", 1)
        assert "k" not in task.context

    def test_unmodified_call_leaves_context_equal(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.get_conversation().remember("k", 1, "d")
        before = dict(agent.get_conversation().context)

        agent.invoke(_INPUTS)

        assert dict(agent.get_conversation().context) == before

    def test_changed_value_is_written_back(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.get_conversation().remember("k", 1)
        task, _ = _run_until_post(agent, _INPUTS)
        self._bump(task, "k", 2)

        agent.record(task)

        assert agent.get_conversation().context["k"].value == 2

    def test_readonly_destination_is_skipped_without_failing(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.get_conversation().remember("k", 1, readonly=True)
        agent.get_conversation().remember("m", 1)
        task, _ = _run_until_post(agent, _INPUTS)
        self._bump(task, "k", 2)
        self._bump(task, "m", 2)

        agent.record(task)

        ctx = agent.get_conversation().context
        assert ctx["k"].value == 1 and ctx["m"].value == 2
        assert len(agent.get_conversation().records) == 1

    def test_key_forgotten_mid_call_is_not_resurrected(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.get_conversation().remember("k", 1)
        task, _ = _run_until_post(agent, _INPUTS)
        agent.get_conversation().forget("k")

        agent.record(task)

        assert "k" not in agent.get_conversation().context

    def test_scratch_key_is_dropped(self) -> None:
        agent = make_agent(record_enabled=True)
        task, _ = _run_until_post(agent, _INPUTS)
        task.context["new"] = ContextVariable(name="new", value=1)

        agent.record(task)

        assert "new" not in agent.get_conversation().context

    def test_write_back_targets_the_bound_conversation(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.get_conversation().remember("k", 1)
        task, _ = _run_until_post(agent, _INPUTS)
        agent.create_conversation("side")
        agent.switch_conversation("side")  # active changes mid-call
        self._bump(task, "k", 9)

        agent.record(task)

        assert agent.get_conversation("default").context["k"].value == 9
        assert len(agent.get_conversation("default").records) == 1
        assert agent.get_conversation("side").records == []

    def test_record_disabled_writes_nothing(self) -> None:
        agent = make_agent(record_enabled=False)
        agent.get_conversation().remember("k", 1)
        task, _ = _run_until_post(agent, _INPUTS)
        self._bump(task, "k", 2)

        agent.record(task)

        assert agent.get_conversation().context["k"].value == 1

    def test_hand_assembled_equals_invoke(self) -> None:
        manual = make_agent(record_enabled=True)
        manual.get_conversation().remember("k", 1, "d")
        task, _ = _run_until_post(manual, _INPUTS)
        manual.record(task)
        automatic = make_agent(record_enabled=True)
        automatic.get_conversation().remember("k", 1, "d")
        automatic.invoke(_INPUTS)

        assert dict(manual.get_conversation().context) == dict(
            automatic.get_conversation().context
        )

    def test_write_back_is_value_only(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.get_conversation().remember("k", 1, "keep")
        task, _ = _run_until_post(agent, _INPUTS)
        task.context["k"] = dataclasses.replace(
            task.context["k"], value=2, description="x", readonly=True
        )

        agent.record(task)

        stored = agent.get_conversation().context["k"]
        assert stored.value == 2
        assert stored.description == "keep"
        assert stored.readonly is False


# ─────────────────────────────────────────────────────────────────────────────
# bind_response_to (Pass 10a): construction + post-stage behavior
# ─────────────────────────────────────────────────────────────────────────────
class _ValueAgent(Agent):
    """Agent whose generated response is a fixed arbitrary Python value
    (mapping / list / int / bytes ...), so binding tests are not limited to
    the str responses ``_EchoAgent`` requires. It still makes one real
    engine call so ``build_result_from_task`` has an LLM record to read."""

    def __init__(self, *, response_value: Any, **kwargs: Any) -> None:
        self._response_value = response_value
        super().__init__(**kwargs)

    def _render_task_messages(self, task: AgentTask) -> list[dict[str, str]]:
        if not task.task_messages:
            task.task_messages = [{"role": "user", "content": task.user_prompt}]
        return task.task_messages

    def _finish(self, task: AgentTask, engine_result: Any) -> AgentTask:
        task.llm_records.append(LLMRecord(
            messages=({"role": "user", "content": task.user_prompt},),
            llm_result=engine_result,
        ))
        task.generated_response = self._response_value
        task.complete = True
        return task

    def act(self, task: AgentTask) -> AgentTask:
        engine_result = self._llm_engine.invoke(
            {"messages": [{"role": "user", "content": task.user_prompt}]}
        )
        return self._finish(task, engine_result)

    async def async_act(self, task: AgentTask) -> AgentTask:
        engine_result = await self._llm_engine.async_invoke(
            {"messages": [{"role": "user", "content": task.user_prompt}]}
        )
        return self._finish(task, engine_result)


class _ReservedEchoAgent(_EchoAgent):
    """_EchoAgent that also reserves ``dummy_reserved`` (for reserved-name
    destination tests through ``make_agent``)."""

    @classmethod
    def get_reserved_parameters(cls) -> list[ParamSpec]:
        return [_DUMMY_RESERVED_PARAM]


def make_value_agent(response_value: Any, **kwargs: Any) -> _ValueAgent:
    kwargs.setdefault("record_enabled", False)
    return _ValueAgent(
        name="value_agent",
        namespace="tests",
        description="Fixed-response test agent.",
        llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
        response_value=response_value,
        **kwargs,
    )


def _names(agent: Agent) -> list[str]:
    return [p.name for p in agent.parameters]


class TestBindResponseToConstruction:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("a", "a"),
            (["a", "b"], {"a": "a", "b": "b"}),
            (("a", "b"), {"a": "a", "b": "b"}),
            ({"b", "a"}, {"a": "a", "b": "b"}),
            (frozenset({"b", "a"}), {"a": "a", "b": "b"}),
            ({0: "first", "k": "second"}, {0: "first", "k": "second"}),
        ],
    )
    def test_forms_normalize(self, value: Any, expected: Any) -> None:
        agent = make_agent(bind_response_to=value)

        assert agent._bind_response_to == expected

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("  a  ", "a"),
            ([" a ", "b	"], {"a": "a", "b": "b"}),
            ({0: " x "}, {0: "x"}),
        ],
    )
    def test_surrounding_whitespace_is_stripped(self, value: Any, expected: Any) -> None:
        assert make_agent(bind_response_to=value)._bind_response_to == expected

    def test_duplicates_are_detected_after_stripping(self) -> None:
        with pytest.raises(AgentError, match="duplicate"):
            make_agent(bind_response_to=["a", " a "])

    @pytest.mark.parametrize(
        "make",
        [
            lambda: iter(["b", "a"]),
            lambda: (n for n in ["a", "b"]),
            lambda: {"b": 1, "a": 2}.keys(),
        ],
        ids=["iterator", "generator", "dict_keys_view"],
    )
    def test_any_non_bytes_iterable_of_names_is_a_collection(self, make: Any) -> None:
        agent = make_agent(bind_response_to=make())

        assert sorted(agent._bind_response_to) == ["a", "b"]
        assert agent._bind_response_to == {n: n for n in agent._bind_response_to}

    def test_bytes_is_rejected(self) -> None:
        with pytest.raises(AgentError, match="bind_response_to"):
            make_agent(bind_response_to=b"abc")

    def test_set_form_is_stored_in_sorted_order(self) -> None:
        agent = make_agent(bind_response_to={"c", "a", "b"})

        assert list(agent._bind_response_to) == ["a", "b", "c"]

    def test_none_is_none(self) -> None:
        assert make_agent(bind_response_to=None)._bind_response_to is None

    @pytest.mark.parametrize("value", [[], (), set(), {}])
    def test_empty_collection_or_dict_is_none(self, value: Any) -> None:
        assert make_agent(bind_response_to=value)._bind_response_to is None

    @pytest.mark.parametrize("value", [5, b"ab", [["a"]], 1.5, object()])
    def test_bad_type_raises(self, value: Any) -> None:
        with pytest.raises(AgentError, match="bind_response_to"):
            make_agent(bind_response_to=value)

    @pytest.mark.parametrize(
        "value",
        ["my key", "", "   ", "1x", "café", [1], ["ok", 2], {"k": 1}, {"k": "not valid"}, {"k": ""}],
    )
    def test_non_identifier_or_non_str_destination_raises(self, value: Any) -> None:
        with pytest.raises(AgentError, match="bind_response_to"):
            make_agent(bind_response_to=value)

    @pytest.mark.parametrize(
        "value", ["task_result_x", ["ok", "task_result_y"], {"k": "task_result_z"}]
    )
    def test_task_result_prefix_raises(self, value: Any) -> None:
        with pytest.raises(AgentError, match="task_result_"):
            make_agent(bind_response_to=value)

    @pytest.mark.parametrize("value", [["a", "a"], ("a", "a"), {0: "a", 1: "a"}])
    def test_duplicate_destinations_raise(self, value: Any) -> None:
        with pytest.raises(AgentError, match="duplicate"):
            make_agent(bind_response_to=value)

    @pytest.mark.parametrize(
        "value", ["dummy_reserved", ["dummy_reserved"], {"k": "dummy_reserved"}]
    )
    def test_reserved_name_destination_raises(self, value: Any) -> None:
        with pytest.raises(AgentError, match="framework-reserved"):
            make_agent(agent_cls=_ReservedEchoAgent, bind_response_to=value)

    def test_python_keyword_destination_is_accepted(self) -> None:
        agent = make_agent(bind_response_to="class")

        assert agent._bind_response_to == "class"

    def test_kwargs_in_pre_invoke_raises(self) -> None:
        def pre_kwargs(prompt: str, **rest: Any) -> str:
            return prompt

        with pytest.raises(AgentError, match=r"pre_invoke.*\*\*kwargs"):
            make_agent(pre_invoke=pre_kwargs)

    def test_kwargs_in_post_invoke_raises(self) -> None:
        with pytest.raises(AgentError, match=r"post_invoke.*\*\*kwargs"):
            make_agent(post_invoke=post_with_kwargs, bind_response_to="items")

    def test_args_in_pre_and_post_invoke_accepted(self) -> None:
        def pre_args(*parts: Any) -> str:
            return "".join(str(p) for p in parts)

        agent = make_agent(
            pre_invoke=pre_args, post_invoke=post_with_args, bind_response_to="items"
        )

        assert agent._bind_response_to == "items"

    def test_post_result_key_is_removed(self) -> None:
        assert not hasattr(make_agent(), "post_result_key")
        with pytest.raises(TypeError):
            _EchoAgent(
                name="a", namespace="tests", description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                post_result_key="result",  # removed argument
            )


class TestPostStageBinding:
    def test_str_binds_whole_response_and_leaves_public_signature(self) -> None:
        def post(resp: Any, suffix: str = "!") -> str:
            return f"{resp}{suffix}"

        agent = make_value_agent("hi", post_invoke=post, bind_response_to="resp")

        assert set(_names(agent)) == {"prompt", "suffix"}
        assert agent.invoke({"prompt": "x"}).result == "hi!"

    def test_list_binds_keys_of_a_mapping_response(self) -> None:
        def post(a: Any, b: Any) -> dict[str, Any]:
            return {"a": a, "b": b}

        agent = make_value_agent(
            {"a": 1, "b": 2, "c": 3}, post_invoke=post, bind_response_to=["a", "b"]
        )

        assert set(_names(agent)) == {"prompt"}
        assert agent.invoke({"prompt": "x"}).result == {"a": 1, "b": 2}

    def test_dict_selectors_index_keys_and_list_positions(self) -> None:
        def post(first: Any, second: Any) -> list[Any]:
            return [first, second]

        mapping_agent = make_value_agent(
            {"k": "v", 7: "seven"},
            post_invoke=post,
            bind_response_to={"k": "first", 7: "second"},
        )
        list_agent = make_value_agent(
            ["x", "y", "z"], post_invoke=post, bind_response_to={0: "first", 2: "second"}
        )

        assert mapping_agent.invoke({"prompt": "p"}).result == ["v", "seven"]
        assert list_agent.invoke({"prompt": "p"}).result == ["x", "z"]

    def test_absent_key_is_skipped_and_required_parameter_fails_naturally(self) -> None:
        def post(a: Any, b: Any) -> Any:
            return (a, b)

        agent = make_value_agent({"a": 1}, post_invoke=post, bind_response_to=["a", "b"])

        with pytest.raises(TypeError, match="(?i)missing"):
            agent.invoke({"prompt": "x"})

    def test_absent_key_leaves_defaulted_parameter_at_its_default(self) -> None:
        def post(a: Any, b: Any = "dflt") -> Any:
            return (a, b)

        agent = make_value_agent({"a": 1}, post_invoke=post, bind_response_to=["a", "b"])

        assert agent.invoke({"prompt": "x"}).result == (1, "dflt")

    def test_absent_index_is_skipped(self) -> None:
        def post(a: Any, b: Any = "dflt") -> Any:
            return (a, b)

        agent = make_value_agent(["only"], post_invoke=post, bind_response_to={0: "a", 5: "b"})

        assert agent.invoke({"prompt": "x"}).result == ("only", "dflt")

    def test_present_none_value_is_bound(self) -> None:
        def post(a: Any = "dflt") -> Any:
            return a

        agent = make_value_agent({"a": None}, post_invoke=post, bind_response_to=["a"])

        assert agent.invoke({"prompt": "x"}).result is None

    def test_shared_name_stays_in_signature_caller_value_stands_when_key_absent(self) -> None:
        def post(topic: str) -> str:
            return topic

        agent = make_value_agent(
            {"other": 1}, pre_invoke=build_prompt, post_invoke=post, bind_response_to=["topic"]
        )

        assert "topic" in _names(agent)
        assert agent.invoke({"topic": "caller"}).result == "caller"

    def test_shared_name_is_overwritten_by_response_when_key_present(self) -> None:
        def post(topic: str) -> str:
            return topic

        agent = make_value_agent(
            {"topic": "from-response"},
            pre_invoke=build_prompt,
            post_invoke=post,
            bind_response_to=["topic"],
        )

        assert agent.invoke({"topic": "caller"}).result == "from-response"

    def test_no_binding_required_post_parameter_is_caller_driven_and_fails_when_missing(self) -> None:
        def post(resp: Any) -> Any:
            return resp

        agent = make_value_agent("response", post_invoke=post)

        # Migration case: the response is no longer passed implicitly.
        assert "resp" in _names(agent)
        with pytest.raises(TypeError, match="(?i)missing"):
            agent.invoke({"prompt": "x"})
        assert agent.invoke({"prompt": "x", "resp": "from-caller"}).result == "from-caller"

    def test_no_binding_defaulted_parameter_runs_without_the_response(self) -> None:
        def post(resp: str = "dflt") -> str:
            return resp

        agent = make_value_agent("response", post_invoke=post)

        assert agent.invoke({"prompt": "x"}).result == "dflt"

    def test_bound_name_matching_no_post_parameter_is_inert(self) -> None:
        def post(resp: str = "dflt") -> str:
            return resp

        agent = make_value_agent("response", post_invoke=post, bind_response_to="ghost")

        assert agent.invoke({"prompt": "x"}).result == "dflt"
        assert "ghost" not in _names(agent)

    def test_post_process_overlays_bound_values_onto_post_inputs(self) -> None:
        def post(resp: str, tag: str = "t") -> str:
            return f"{resp}|{tag}"

        agent = make_value_agent(
            {"a": 1, "b": 2}, post_invoke=post, bind_response_to={"a": "resp", "b": "ghost"}
        )

        task = agent.create_task(agent.filter_inputs({"prompt": "x", "tag": "T"}))
        assert task.post_inputs == {"tag": "T"}
        task = agent.pre_process(task)
        while not task.complete:
            task = agent.act(agent.prepare(agent.think(task)))
        agent.post_process(task)

        # The sliced piece lands on the task's post inputs; a bound name that
        # matches no post parameter is not added; caller inputs are kept.
        assert task.post_inputs == {"tag": "T", "resp": 1}

    def test_no_post_stage_leaves_post_inputs_untouched(self) -> None:
        agent = make_value_agent("response", bind_response_to="resp")

        task, _ = _run_until_post(agent, {"prompt": "x"})

        assert task.post_inputs == {}

    @pytest.mark.parametrize("binding", [None, "resp"])
    def test_zero_parameter_post_runs_and_leaves_signature_alone(self, binding: Any) -> None:
        agent = make_value_agent("response", post_invoke=post_zero_args, bind_response_to=binding)

        assert _names(agent) == ["prompt"]
        assert agent.invoke({"prompt": "x"}).result == "bad"

    def test_post_invoke_none_returns_generated_response_as_is(self) -> None:
        payload = {"a": 1, "b": [2, 3]}
        agent = make_value_agent(payload, post_invoke=None)

        result = agent.invoke({"prompt": "x"})

        assert result.result == payload
        assert agent.return_type == "any"
        assert agent.post_invoke is None
        assert agent.to_dict()["post_invoke"] is None

    def test_post_invoke_none_async(self) -> None:
        agent = make_value_agent({"a": 1}, post_invoke=None)

        result = asyncio.run(agent.async_invoke({"prompt": "x"}))

        assert result.result == {"a": 1}

    def test_post_invoke_none_with_binding_still_returns_response(self) -> None:
        agent = make_value_agent({"a": 1}, post_invoke=None, bind_response_to=["a"])

        assert agent.invoke({"prompt": "x"}).result == {"a": 1}

    def test_default_agent_has_no_post_stage(self) -> None:
        agent = _EchoAgent(
            name="a", namespace="tests", description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
        )

        assert agent.post_invoke is None
        assert agent._bind_response_to is None
        assert _names(agent) == ["prompt"]

    def test_args_parameter_bound_with_list_response_expands_positionally(self) -> None:
        agent = make_value_agent(
            ["a", "b"], post_invoke=post_with_args, bind_response_to="items"
        )

        assert agent.invoke({"prompt": "x"}).result == ("a", "b")

    def test_args_parameter_with_non_list_response_raises_and_turn_is_not_recorded(self) -> None:
        agent = make_value_agent(
            5, post_invoke=post_with_args, bind_response_to="items", record_enabled=True
        )

        with pytest.raises(TypeError, match="list or tuple"):
            agent.invoke({"prompt": "x"})

        assert agent.get_conversation().records == []

    @pytest.mark.parametrize("response", ["text", b"bytes"])
    def test_str_or_bytes_response_with_selectors_raises(self, response: Any) -> None:
        agent = make_value_agent(
            response, post_invoke=post_zero_args, bind_response_to={0: "first"}
        )

        with pytest.raises(AgentInvocationError, match="str/bytes"):
            agent.invoke({"prompt": "x"})

    def test_str_response_with_str_binding_is_fine(self) -> None:
        def post(resp: str) -> str:
            return resp.upper()

        agent = make_value_agent("text", post_invoke=post, bind_response_to="resp")

        assert agent.invoke({"prompt": "x"}).result == "TEXT"

    @pytest.mark.parametrize("binding", [{"k": "v"}, ["k"]])
    def test_non_subscriptable_response_with_selectors_raises_type_error(
        self, binding: Any
    ) -> None:
        agent = make_value_agent(5, post_invoke=post_zero_args, bind_response_to=binding)

        with pytest.raises(TypeError):
            agent.invoke({"prompt": "x"})

    def test_bindings_resolve_before_the_post_runs(self) -> None:
        calls: list[str] = []

        def post(v: Any = None) -> Any:
            calls.append("ran")
            return v

        agent = make_value_agent(5, post_invoke=post, bind_response_to={"k": "v"})

        with pytest.raises(TypeError):
            agent.invoke({"prompt": "x"})

        assert calls == []

    def test_failing_post_does_not_record_the_turn(self) -> None:
        def post(resp: Any) -> Any:
            raise RuntimeError("boom")

        agent = make_value_agent(
            "r", post_invoke=post, bind_response_to="resp", record_enabled=True
        )

        with pytest.raises(ToolInvocationError, match="boom"):
            agent.invoke({"prompt": "x"})

        assert agent.get_conversation().records == []

    def test_to_dict_reports_bind_response_to_not_post_result_key(self) -> None:
        agent = make_value_agent(
            {"a": 1}, post_invoke=post_zero_args, bind_response_to={"a": "x"}
        )

        data = agent.to_dict()

        assert data["bind_response_to"] == {"a": "x"}
        assert "post_result_key" not in data

    def test_sync_and_async_bind_identically(self) -> None:
        def post(a: Any, b: Any = "dflt") -> Any:
            return (a, b)

        sync_agent = make_value_agent({"a": 1}, post_invoke=post, bind_response_to=["a", "b"])
        async_agent = make_value_agent({"a": 1}, post_invoke=post, bind_response_to=["a", "b"])

        sync_result = sync_agent.invoke({"prompt": "x"}).result
        async_result = asyncio.run(async_agent.async_invoke({"prompt": "x"})).result

        assert sync_result == async_result == (1, "dflt")

    def test_async_selector_error_matches_sync(self) -> None:
        agent = make_value_agent("text", post_invoke=post_zero_args, bind_response_to={0: "f"})

        with pytest.raises(AgentInvocationError, match="str/bytes"):
            asyncio.run(agent.async_invoke({"prompt": "x"}))


class TestBuildMessagesRemoved:
    def test_agent_has_no_build_messages(self) -> None:
        assert not hasattr(Agent, "build_messages")


class TestOptionalPreInvoke:
    def test_default_agent_has_no_pre_stage(self) -> None:
        agent = make_value_agent("r")

        assert agent.pre_invoke is None
        assert agent.to_dict()["pre_invoke"] is None

    def test_default_signature_is_the_built_in_prompt_parameter(self) -> None:
        agent = make_value_agent("r")
        prompt = agent.parameters[0]

        assert _names(agent) == ["prompt"]
        assert prompt.kind == ParamSpec.POSITIONAL_OR_KEYWORD
        assert prompt.type == ("str",)
        assert prompt.default is NO_VAL

    def test_prompt_passes_through_verbatim_sync_and_async(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = _EchoAgent(
            name="a", namespace="tests", description="d",
            llm_engine=engine, system_prompt=ROLE_PROMPT,
        )

        agent.invoke({"prompt": "hello"})
        asyncio.run(agent.async_invoke({"prompt": "hello"}))
        agent("hello")

        assert [c[-1] for c in engine.calls] == [{"role": "user", "content": "hello"}] * 3

    def test_non_str_prompt_raises_agent_invocation_error(self) -> None:
        agent = make_value_agent("r")

        with pytest.raises(AgentInvocationError, match="the prompt is not a string"):
            agent.invoke({"prompt": 5})
        with pytest.raises(AgentInvocationError, match="the prompt is not a string"):
            asyncio.run(agent.async_invoke({"prompt": 5}))

    def test_post_parameter_named_prompt_shares_the_built_in_parameter(self) -> None:
        def post(prompt: str, resp: Any) -> str:
            return f"{prompt}->{resp}"

        agent = make_value_agent("r", post_invoke=post, bind_response_to="resp")

        assert _names(agent).count("prompt") == 1
        assert agent.invoke({"prompt": "q"}).result == "q->r"

    def test_incompatible_post_prompt_type_raises_at_construction(self) -> None:
        def post(prompt: int, resp: Any) -> Any:
            return resp

        with pytest.raises(AgentError, match="(?i)compatible"):
            make_value_agent("r", post_invoke=post, bind_response_to="resp")

    def test_custom_pre_invoke_with_its_own_prompt_parameter_still_works(self) -> None:
        def pre(prompt: str, tone: str = "calm") -> str:
            return f"{prompt} ({tone})"

        agent = make_value_agent("r", pre_invoke=pre)

        assert agent.pre_invoke is not None
        assert set(_names(agent)) == {"prompt", "tone"}
        assert agent.invoke({"prompt": "go"}).result == "r"


class TestContextSink:
    @staticmethod
    def _post(resp: Any) -> Any:
        return resp

    def _run(self, agent: Agent) -> tuple[AgentTask, Any]:
        return _run_until_post(agent, {"prompt": "p"})

    def test_writable_variable_value_is_replaced_keeping_description(self) -> None:
        agent = make_value_agent("new", bind_response_to="notes")
        agent.get_conversation().remember("notes", "old", "my notes")

        task, _ = self._run(agent)

        assert task.context["notes"].value == "new"
        assert task.context["notes"].description == "my notes"
        assert task.context["notes"].readonly is False

    def test_readonly_variable_is_skipped_but_post_still_receives_piece(self) -> None:
        agent = make_value_agent("new", post_invoke=self._post, bind_response_to="resp")
        agent.get_conversation().remember("resp", "old", readonly=True)

        task, result = self._run(agent)

        assert task.context["resp"].value == "old"
        assert result.result == "new"

    def test_absent_name_is_ignored_and_nothing_is_created(self) -> None:
        agent = make_value_agent("new", bind_response_to="notes", record_enabled=True)

        task, _ = self._run(agent)
        agent.record(task)

        assert "notes" not in task.context
        assert "notes" not in agent.get_conversation().context

    def test_no_binding_writes_nothing(self) -> None:
        agent = make_value_agent("new", bind_response_to=None)
        agent.get_conversation().remember("notes", "old")

        task, _ = self._run(agent)

        assert task.context["notes"].value == "old"

    def test_collection_and_dict_bindings_and_skipped_selector(self) -> None:
        agent = make_value_agent(
            {"a": 1, "z": None}, bind_response_to={"a": "x", "z": "y", "missing": "w"}
        )
        for name in ("x", "y", "w"):
            agent.get_conversation().remember(name, "old")

        task, _ = self._run(agent)

        assert task.context["x"].value == 1
        assert task.context["y"].value is None
        assert task.context["w"].value == "old"

    def test_no_post_stage_still_writes(self) -> None:
        agent = make_value_agent("new", post_invoke=None, bind_response_to="notes")
        agent.get_conversation().remember("notes", "old")

        task, result = self._run(agent)

        assert result.result == "new"
        assert task.context["notes"].value == "new"

    def test_failing_post_writes_nothing_and_records_nothing(self) -> None:
        def post(resp: Any) -> Any:
            raise RuntimeError("boom")

        agent = make_value_agent(
            "new", post_invoke=post, bind_response_to="resp", record_enabled=True
        )
        agent.get_conversation().remember("resp", "old")

        with pytest.raises(Exception):
            agent.invoke({"prompt": "p"})

        assert agent.get_conversation().context["resp"].value == "old"
        assert agent.get_conversation().records == []

    def test_remembered_variable_persists_through_record(self) -> None:
        agent = make_value_agent("new", bind_response_to="notes", record_enabled=True)
        agent.get_conversation().remember("notes", "old", "d")

        agent.invoke({"prompt": "p"})

        stored = agent.get_conversation().context["notes"]
        assert (stored.value, stored.description) == ("new", "d")

    def test_record_disabled_persists_nothing(self) -> None:
        agent = make_value_agent("new", bind_response_to="notes", record_enabled=False)
        agent.get_conversation().remember("notes", "old")

        agent.invoke({"prompt": "p"})

        assert agent.get_conversation().context["notes"].value == "old"

    def test_bind_response_returns_resolved_values_only(self) -> None:
        agent = make_value_agent({"a": 1}, bind_response_to=["a", "b"])
        task = agent.create_task(agent.filter_inputs({"prompt": "p"}))
        task.generated_response = {"a": 1}

        assert agent._bind_response(task) == {"a": 1}
        assert make_value_agent("r")._bind_response(task) == {}

    def test_async_parity(self) -> None:
        agent = make_value_agent("new", bind_response_to="notes", record_enabled=True)
        agent.get_conversation().remember("notes", "old")

        asyncio.run(agent.async_invoke({"prompt": "p"}))

        assert agent.get_conversation().context["notes"].value == "new"
