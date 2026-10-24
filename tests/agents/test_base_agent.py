from __future__ import annotations

import dataclasses
import json
from typing import Annotated, Any, Mapping, Optional
import warnings

import pytest
import asyncio

from atomic_agentic.tools import Tool
from atomic_agentic.agents.base import Agent
from atomic_agentic.exceptions import AgentError, AgentInvocationError, SchemaError
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

    ``act``/``async_act`` call ``build_messages`` directly with
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
        messages = self.build_messages(self._echo_system_prompt, task.turns, task.user_prompt)
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
        messages = self.build_messages(self._echo_system_prompt, task.turns, task.user_prompt)
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
    post_result_key: str | None = None,
    system_prompt: str = ROLE_PROMPT,
    response_preview_limit: int | None = None,
    agent_cls: type[_EchoAgent] = _EchoAgent,
) -> _EchoAgent:
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
        post_result_key=post_result_key,
        response_preview_limit=response_preview_limit,
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

    def test_default_identity_pre_and_post_invoke_path(self) -> None:
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

    def test_custom_post_result_key_routes_raw_result(self) -> None:
        agent = make_agent(
            post_invoke=post_with_custom_key,
            post_result_key="raw",
        )

        result = agent.invoke({"topic": "pytest", "tone": "strict"})

        assert agent.post_result_key == "raw"
        assert result.result == "ECHO: Write about pytest in a strict tone.!"

    def test_post_result_key_defaults_to_first_post_parameter(self) -> None:
        agent = make_agent(post_invoke=post_with_custom_key)

        assert agent.post_result_key == "raw"

    def test_empty_post_result_key_raises(self) -> None:
        with pytest.raises(AgentError, match="post_result_key"):
            make_agent(post_invoke=post_with_custom_key, post_result_key="  ")

    def test_unknown_post_result_key_raises(self) -> None:
        with pytest.raises(AgentError, match="post_result_key"):
            make_agent(post_invoke=post_with_custom_key, post_result_key="missing")

    def test_variadic_post_result_key_is_allowed_for_args(self) -> None:
        agent = make_agent(
            post_invoke=post_with_args,
            post_result_key="items",
        )

        assert agent.post_result_key == "items"

    def test_variadic_post_result_key_is_allowed_for_kwargs(self) -> None:
        agent = make_agent(
            post_invoke=post_with_kwargs,
            post_result_key="items",
        )

        assert agent.post_result_key == "items"

    def test_post_result_key_naming_an_undeclared_name_raises(self) -> None:
        # "not_a_param" is never a legal post_result_key here: it's simply
        # not among post_invoke's own declared parameter names at all
        # (resolved against post_tool.parameters directly, unstripped).
        with pytest.raises(AgentError, match="post_result_key"):
            make_agent(post_result_key="not_a_param")

    def test_post_result_key_matching_pre_invoke_name_no_longer_raises(self) -> None:
        # Spec-mandated behavior change: the old pairwise "post_result_key
        # collides with a pre_invoke/extra_parameters name" check is dropped
        # in the __init__ rebuild -- post_only_raw excludes the result key
        # BEFORE peer reconciliation ever runs, so there is no longer a
        # schema-level name for it to collide with. pre_invoke's "topic" and
        # the internal result-routing slot stay independently sliced at
        # invoke time (pre_inputs/post_inputs), so no runtime ambiguity
        # results either.
        def post_with_topic_key(topic: str) -> str:
            return topic

        agent = make_agent(post_invoke=post_with_topic_key, post_result_key="topic")
        assert agent.post_result_key == "topic"


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
        # generated_response must round-trip back into history as valid
        # JSON, not Python repr syntax (single-quoted keys, True/None) --
        # see utils.agents.stringify_result.
        agent = make_agent(engine=FakeLLMEngine([]), record_enabled=True)
        record = AgentRecord(
            user_prompt="hello",
            generated_response={"summary": "ok", "confident": True, "score": None},
        )

        rendered = agent.render_turn(record)

        assert rendered[1]["content"] == '{"summary": "ok", "confident": true, "score": null}'
        assert json.loads(rendered[1]["content"]) == {"summary": "ok", "confident": True, "score": None}

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
    """Named-conversation branching: create/clone/delete/switch. Invoking in a
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
        assert created.name == "scratch"
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

        with pytest.raises(AgentInvocationError):
            agent.create_conversation("scratch")
        # The existing conversation's history survives the rejected collision.
        assert len(agent.get_conversation("scratch").records) == 1

    @pytest.mark.parametrize("bad_name", ["bad name", "1bad", "name_5", ""])
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
        assert snap.name == "default"
        assert snap.agent_type is type(agent)
        assert [r.inputs["topic"] for r in snap.records[-1:]] == ["second"]

    def test_get_conversation_snapshot_is_independent_of_storage(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke({"topic": "pytest", "tone": "strict"})

        snap = agent.get_conversation()
        snap.records.append("not a real record")  # type: ignore[arg-type]
        snap.remember("x", 1)

        fresh = agent.get_conversation()
        assert len(fresh.records) == 1
        assert "x" not in fresh.context

    def test_delete_conversation_unknown_key_raises(self) -> None:
        agent = make_agent()

        with pytest.raises(AgentInvocationError):
            agent.delete_conversation("nope")

    def test_delete_conversation_refuses_active_conversation(self) -> None:
        agent = make_agent()
        agent.create_conversation("scratch")
        agent.switch_conversation("scratch")

        with pytest.raises(AgentInvocationError):
            agent.delete_conversation("scratch")

    def test_delete_conversation_removes_non_active_conversation(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.create_conversation("scratch")
        agent.switch_conversation("scratch")
        agent.invoke({"topic": "pytest", "tone": "strict"})
        agent.switch_conversation("default")

        removed = agent.delete_conversation("scratch")

        assert isinstance(removed, Conversation)
        assert len(removed.records) == 1
        assert "scratch" not in agent.conversation_names

    def test_delete_conversation_default_resets_in_place_never_removed(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke({"topic": "pytest", "tone": "strict"})

        removed = agent.delete_conversation("default")

        assert isinstance(removed, Conversation)
        assert len(removed.records) == 1
        assert "default" in agent.conversation_names
        assert agent.get_conversation("default").records == []

    def test_clone_conversation_unknown_source_raises(self) -> None:
        agent = make_agent()

        with pytest.raises(AgentInvocationError):
            agent.clone_conversation("nope", end_id="whatever")

    def test_clone_conversation_unknown_run_id_raises(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke({"topic": "pytest", "tone": "strict"})

        with pytest.raises(ValueError):
            agent.clone_conversation("default", end_id="nonexistent")

    def test_clone_conversation_never_changes_active_conversation(self) -> None:
        agent = make_agent(record_enabled=True)
        first = agent.invoke({"topic": "pytest", "tone": "strict"})

        agent.clone_conversation("default", end_id=first.run_id)

        assert agent.active_conversation == "default"

    def test_clone_conversation_explicit_name_collision_raises(self) -> None:
        agent = make_agent(record_enabled=True)
        first = agent.invoke({"topic": "pytest", "tone": "strict"})
        agent.create_conversation("taken")

        with pytest.raises(ValueError):
            agent.clone_conversation("default", end_id=first.run_id, name="taken")

    def test_clone_conversation_result_is_independent_copy(self) -> None:
        agent = make_agent(record_enabled=True)
        first = agent.invoke({"topic": "pytest", "tone": "strict"})

        cloned = agent.clone_conversation("default", end_id=first.run_id)
        agent.switch_conversation(cloned.name)
        agent.invoke({"topic": "forked-only", "tone": "strict"})

        # The clone's own new turn must never appear back in "default".
        assert len(agent.get_conversation("default").records) == 1
        assert len(agent.get_conversation(cloned.name).records) == 2

    def test_clone_conversation_auto_name_survives_clear_memory(self) -> None:
        # self._branch_counter is deliberately not reset by clear_memory(),
        # so an auto-generated name must never be reissued across a clear
        # even though the old name becomes technically free again.
        agent = make_agent(record_enabled=True)
        first = agent.invoke({"topic": "pytest", "tone": "strict"})
        first_clone_name = agent.clone_conversation("default", end_id=first.run_id).name

        agent.clear_memory()

        second = agent.invoke({"topic": "pytest", "tone": "strict"})
        second_clone_name = agent.clone_conversation("default", end_id=second.run_id).name

        assert second_clone_name != first_clone_name


class _OtherEchoAgent(_EchoAgent):
    pass


class TestConversationContextWrappers:
    def test_wrappers_target_active_conversation_by_default(self) -> None:
        agent = make_agent()
        agent.create_conversation("scratch")
        agent.switch_conversation("scratch")

        agent.remember("x", 1, "one")
        agent.describe("x", "uno")
        agent.remember_many([ContextVariable(name="y", value=2)])

        scratch = agent.get_conversation("scratch").context
        assert scratch["x"].value == 1 and scratch["x"].description == "uno"
        assert scratch["y"].value == 2
        assert len(agent.get_conversation("default").context) == 0

    def test_wrappers_target_named_conversation(self) -> None:
        agent = make_agent()
        agent.create_conversation("scratch")

        agent.remember("x", 1, conversation="scratch")
        agent.remember_many([ContextVariable(name="y", value=2)], conversation="scratch")
        agent.forget("x", conversation="scratch")
        agent.forget_many(["y"], conversation="scratch")

        assert len(agent.get_conversation("scratch").context) == 0

    @pytest.mark.parametrize(
        "call",
        [
            lambda a: a.remember("x", 1, conversation="nope"),
            lambda a: a.describe("x", "d", conversation="nope"),
            lambda a: a.forget("x", conversation="nope"),
            lambda a: a.remember_many([], conversation="nope"),
            lambda a: a.forget_many([], conversation="nope"),
        ],
    )
    def test_unknown_conversation_raises(self, call: Any) -> None:
        with pytest.raises(AgentInvocationError):
            call(make_agent())

    def test_conversation_errors_surface_unchanged(self) -> None:
        agent = make_agent()
        agent.remember("x", 1, readonly=True)

        with pytest.raises(ValueError, match="readonly"):
            agent.remember("x", 2)
        with pytest.raises(KeyError):
            agent.forget("nope")
        with pytest.raises(KeyError):
            agent.forget_many(["x", "nope"])
        assert agent.get_conversation().context["x"].value == 1

    def test_resets_empty_context_and_keep_tag(self) -> None:
        agent = make_agent()
        agent.remember("x", 1)

        agent.delete_conversation("default")
        assert len(agent.get_conversation().context) == 0
        assert agent.get_conversation().agent_type is _EchoAgent

        agent.remember("y", 1)
        agent.clear_memory()
        assert len(agent.get_conversation().context) == 0
        assert agent.get_conversation().agent_type is _EchoAgent

    def test_every_created_conversation_is_tagged(self) -> None:
        agent = make_agent(record_enabled=True)
        first = agent.invoke({"topic": "pytest", "tone": "strict"})
        agent.create_conversation("scratch")
        cloned = agent.clone_conversation("default", end_id=first.run_id)

        for name in ("default", "scratch", cloned.name):
            assert agent.get_conversation(name).agent_type is _EchoAgent


class TestImportConversation:
    def _source(self) -> tuple[_EchoAgent, list[str]]:
        agent = make_agent(record_enabled=True)
        ids = [
            agent.invoke({"topic": t, "tone": "strict"}).run_id
            for t in ("a", "b", "c")
        ]
        agent.remember("k", [1], "kay")
        return agent, ids

    def test_whole_conversation_is_new_tagged_and_shares_records(self) -> None:
        source, _ = self._source()
        target = make_agent()

        imported = target.import_conversation(source.get_conversation(), name="imp")

        assert target.get_conversation("imp").agent_type is _EchoAgent
        assert imported.records == source.get_conversation().records
        assert all(a is b for a, b in zip(imported.records, source.get_conversation().records))
        assert target.active_conversation == "default"

    def test_index_and_run_id_ranges(self) -> None:
        source, ids = self._source()
        snap = source.get_conversation()
        target = make_agent()

        by_index = target.import_conversation(snap, start=1, end=-1, name="i")
        by_id = target.import_conversation(snap, start_id=ids[1], name="j")

        assert [r.run_id for r in by_index.records] == ids[1:]
        assert [r.run_id for r in by_id.records] == ids[1:]

    def test_range_errors(self) -> None:
        source, ids = self._source()
        snap = source.get_conversation()
        target = make_agent()

        with pytest.raises(ValueError, match="mixed"):
            target.import_conversation(snap, start=0, end_id=ids[0])
        with pytest.raises(IndexError):
            target.import_conversation(snap, start=5)
        with pytest.raises(ValueError):
            target.import_conversation(snap, start=2, end=0)
        with pytest.raises(ValueError, match="run_id"):
            target.import_conversation(snap, end_id="nope")
        assert target.conversation_names == ["default"]

    def test_context_is_copied_into_independent_container(self) -> None:
        source, _ = self._source()
        target = make_agent()
        imported = target.import_conversation(source.get_conversation(), name="imp")

        assert imported.context["k"] is source.get_conversation().context["k"]
        target.remember("only_target", 1, conversation="imp")
        source.forget("k")

        assert "only_target" not in source.get_conversation().context
        assert "k" in target.get_conversation("imp").context

    def test_name_rules(self) -> None:
        source, _ = self._source()
        snap = source.get_conversation()
        target = make_agent()
        target.create_conversation("taken")

        with pytest.raises(ValueError):
            target.import_conversation(snap, name="bad name")
        with pytest.raises(ValueError):
            target.import_conversation(snap, name="taken")
        auto = target.import_conversation(snap)
        assert auto.name == "default_0"

    def test_wrong_agent_type_raises_and_creates_nothing(self) -> None:
        source, _ = self._source()
        other = make_agent(agent_cls=_OtherEchoAgent)

        with pytest.raises(TypeError, match="_EchoAgent"):
            other.import_conversation(source.get_conversation())
        assert other.conversation_names == ["default"]

    def test_non_conversation_raises(self) -> None:
        with pytest.raises(TypeError):
            make_agent().import_conversation(make_agent())  # type: ignore[arg-type]

    def test_clone_carries_context(self) -> None:
        source, ids = self._source()

        cloned = source.clone_conversation("default", end_id=ids[0], name="cl")

        assert cloned.context["k"].value == [1]
        assert len(cloned.records) == 1

    def test_continuing_a_clone_appends_in_place_and_keeps_its_context(self) -> None:
        agent = make_agent(record_enabled=True)
        first = agent.invoke({"topic": "a", "tone": "strict"})
        agent.remember("k", 1)
        agent.invoke({"topic": "b", "tone": "strict"})
        agent.clone_conversation("default", end_id=first.run_id, name="back")
        agent.switch_conversation("back")

        agent.invoke({"topic": "c", "tone": "strict"})
        agent.invoke({"topic": "d", "tone": "strict"})

        assert set(agent.conversation_names) == {"default", "back"}
        assert len(agent.get_conversation("back").records) == 3
        assert len(agent.get_conversation("default").records) == 2
        assert agent.get_conversation("back").context["k"].value == 1

    def test_clone_of_whole_conversation_continues_after_original_advances(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.invoke(_INPUTS)
        agent.clone_conversation("default", name="whole")
        agent.invoke(_INPUTS)  # the original advances past the shared tail

        agent.switch_conversation("whole")
        agent.invoke(_INPUTS)

        assert set(agent.conversation_names) == {"default", "whole"}
        assert len(agent.get_conversation("whole").records) == 2


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
        assert data["post_result_key"] == agent.post_result_key
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
    def test_response_preview_limit_is_frozen(self) -> None:
        agent = make_agent(response_preview_limit=50)

        with pytest.raises(AttributeError):
            agent.response_preview_limit = 100  # type: ignore[misc]

    def test_response_preview_limit_construction_rejects_zero(self) -> None:
        with pytest.raises(AgentError, match="response_preview_limit"):
            make_agent(response_preview_limit=0)

    def test_response_preview_limit_construction_rejects_negative(self) -> None:
        with pytest.raises(AgentError, match="response_preview_limit"):
            make_agent(response_preview_limit=-1)

    def test_response_preview_limit_construction_rejects_non_int(self) -> None:
        with pytest.raises(AgentError, match="response_preview_limit"):
            make_agent(response_preview_limit="100")  # type: ignore[arg-type]

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
        # "prompt" comes from the default identity_pre_tool; no extra_parameters
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

    def test_variadic_param_named_like_reserved_is_exempt_from_the_reserved_check(self) -> None:
        # A VAR_KEYWORD catch-all happens to be named "dummy_reserved" --
        # exempt from the reserved-overlap check entirely: never checked,
        # never stripped, regardless of the name collision (a catch-all's
        # bucket name is a naming coincidence, not the same declared slot).
        # The final signature still can't legally contain two params both
        # literally named "dummy_reserved" (this VAR_KEYWORD bucket and the
        # framework's own KEYWORD_ONLY graft), so construction still fails
        # overall -- but via the pre-existing structural duplicate-name
        # invariant every AtomicInvokable already enforces (SchemaError from
        # insert_by_category's final _validate_parameter_order call), never
        # via the reserved-overlap policy's own AgentError. That distinction
        # is the point of the exemption.
        def pre_with_dummy_reserved_kwargs(prompt: str, **dummy_reserved: Any) -> str:
            return prompt

        with pytest.raises(SchemaError, match="Duplicate"):
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
        # sources (pre_invoke's *extra vs post_invoke's **extra) is caught by
        # n_way_kind_compatible inside the peer report -- not by the old,
        # now-removed variadic_compatible primitive. variadic_compatible's
        # only remaining real job -- differently-NAMED independent variadics
        # -- is now caught structurally by insert_by_category's own
        # duplicate/order validation instead (see the spec's "Removed"
        # section); this is the same-name case, which the peer report
        # itself must catch.
        def pre_with_var_positional(*extra: Any) -> str:
            return str(extra)

        def post_with_var_keyword(result: str, **extra: Any) -> str:
            return result

        with pytest.raises(AgentError, match="no compatible reconciliation"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
                pre_invoke=pre_with_var_positional,
                post_invoke=post_with_var_keyword,
            )


class TestPostResultKeyResolution:
    def test_default_resolves_to_first_reserved_popped_post_name(self) -> None:
        agent = make_agent(post_invoke=post_with_custom_key)
        assert agent.post_result_key == "raw"

    def test_explicit_override_accepted(self) -> None:
        agent = make_agent(post_invoke=post_with_custom_key, post_result_key="suffix")
        assert agent.post_result_key == "suffix"

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(AgentError, match="post_result_key"):
            make_agent(post_invoke=post_with_custom_key, post_result_key="nope")

    def test_collision_with_pre_invoke_name_no_longer_raises(self) -> None:
        # See TestAgentPostInvokeRouting.
        # test_post_result_key_matching_pre_invoke_name_no_longer_raises for
        # the full rationale -- this pairwise collision check was dropped in
        # the __init__ rebuild.
        agent = make_agent(
            post_invoke=lambda topic: topic,
            post_result_key="topic",
        )
        assert agent.post_result_key == "topic"

    def test_collision_with_extra_parameters_name_no_longer_raises(self) -> None:
        def post_with_flag(result: str, flag: str = "x") -> str:
            return result

        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            post_invoke=post_with_flag,
            post_result_key="flag",
            extra_parameters=["flag"],
        )
        assert agent.post_result_key == "flag"

    def test_explicit_post_result_key_naming_a_reserved_parameter_raises(self) -> None:
        # post_result_key is resolved against post_tool.parameters directly,
        # unstripped of reserved-name overlaps (step 5) -- so an explicit
        # post_result_key naming a real post_invoke parameter that also
        # happens to be a reserved name is caught by the dedicated new guard
        # (step 6), with a message identifying the actual problem (not the
        # old generic "not a declared parameter" message this would
        # otherwise surface as, since "dummy_reserved" IS a real declared
        # parameter of post_with_dummy_reserved_param below). Built on
        # _MinimalAgent (not make_agent()'s _EchoAgent, which has no
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
                post_result_key="dummy_reserved",
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
        # "prompt" (default identity_pre_tool, KEYWORD_ONLY) and "lang" (extra,
        # KEYWORD_ONLY) both sort at the same category tier; combined-vs-extra
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
            pre_invoke=pre, post_invoke=post,
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

    def test_post_process_does_not_mutate_post_inputs(self) -> None:
        agent = make_agent(post_invoke=post_with_suffix)
        task = agent.pre_process(agent.create_task(agent.filter_inputs({**_INPUTS, "suffix": "!"})))
        while not task.complete:
            task = agent.act(task)
        before = dict(task.post_inputs)

        agent.post_process(task)

        assert task.post_inputs == before

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
        agent.remember("k", 1)
        conv = agent.get_conversation()

        task = agent.create_task(agent.filter_inputs(_INPUTS))

        assert task.context == dict(conv.context)
        assert task.context["k"] is conv.context["k"]
        task.context["x"] = ContextVariable(name="x", value=1)
        assert "x" not in agent.get_conversation().context

    def test_seed_is_a_snapshot_at_create_task(self) -> None:
        agent = make_agent()
        task = agent.create_task(agent.filter_inputs(_INPUTS))
        agent.remember("k", 1)
        assert "k" not in task.context

    def test_unmodified_call_leaves_context_equal(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.remember("k", 1, "d")
        before = dict(agent.get_conversation().context)

        agent.invoke(_INPUTS)

        assert dict(agent.get_conversation().context) == before

    def test_changed_value_is_written_back(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.remember("k", 1)
        task, _ = _run_until_post(agent, _INPUTS)
        self._bump(task, "k", 2)

        agent.record(task)

        assert agent.get_conversation().context["k"].value == 2

    def test_readonly_destination_is_skipped_without_failing(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.remember("k", 1, readonly=True)
        agent.remember("m", 1)
        task, _ = _run_until_post(agent, _INPUTS)
        self._bump(task, "k", 2)
        self._bump(task, "m", 2)

        agent.record(task)

        ctx = agent.get_conversation().context
        assert ctx["k"].value == 1 and ctx["m"].value == 2
        assert len(agent.get_conversation().records) == 1

    def test_key_forgotten_mid_call_is_not_resurrected(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.remember("k", 1)
        task, _ = _run_until_post(agent, _INPUTS)
        agent.forget("k")

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
        agent.remember("k", 1)
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
        agent.remember("k", 1)
        task, _ = _run_until_post(agent, _INPUTS)
        self._bump(task, "k", 2)

        agent.record(task)

        assert agent.get_conversation().context["k"].value == 1

    def test_hand_assembled_equals_invoke(self) -> None:
        manual = make_agent(record_enabled=True)
        manual.remember("k", 1, "d")
        task, _ = _run_until_post(manual, _INPUTS)
        manual.record(task)
        automatic = make_agent(record_enabled=True)
        automatic.remember("k", 1, "d")
        automatic.invoke(_INPUTS)

        assert dict(manual.get_conversation().context) == dict(
            automatic.get_conversation().context
        )

    def test_writer_marked_readonly_persists_readonly(self) -> None:
        agent = make_agent(record_enabled=True)
        agent.remember("k", 1)
        task, _ = _run_until_post(agent, _INPUTS)
        task.context["k"] = dataclasses.replace(task.context["k"], readonly=True)

        agent.record(task)

        assert agent.get_conversation().context["k"].readonly is True
