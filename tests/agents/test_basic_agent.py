from __future__ import annotations

from typing import Any, Mapping

import pytest
import asyncio

from atomic_agentic.agents.basic import BasicAgent
from atomic_agentic.constants.agents import (
    CONTEXT_END,
    CONTEXT_START,
    TASK_PROMPT_END,
    TASK_PROMPT_START,
)
from atomic_agentic.exceptions import AgentError
from atomic_agentic.models.agents.conversations import Conversation
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.models.parameters import ParamSpec
from ..fake_engines import FakeLLMEngine, echo_latest_user


def make_basic_agent(
    *,
    engine: FakeLLMEngine | None = None,
    role_prompt: str | PromptConfig | None = None,
    record_enabled: bool = False,
    **kwargs: Any,
) -> BasicAgent:
    return BasicAgent(
        name="basic_agent",
        namespace="tests",
        description="BasicAgent under test.",
        llm_engine=engine or FakeLLMEngine(response_fn=echo_latest_user()),
        role_prompt=role_prompt,
        record_enabled=record_enabled,
        **kwargs,
    )


def wrap(prompt: str) -> str:
    """The user message BasicAgent sends for ``prompt`` with empty context."""
    return f"{TASK_PROMPT_START}\n{prompt}\n{TASK_PROMPT_END}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestBasicAgentRolePrompt:
    def test_none_role_prompt_uses_default(self) -> None:
        agent = make_basic_agent(role_prompt=None)
        assert agent.role_prompt == BasicAgent.DEFAULT_ROLE_PROMPT

    def test_empty_string_role_prompt_uses_default(self) -> None:
        agent = make_basic_agent(role_prompt="   ")
        assert agent.role_prompt == BasicAgent.DEFAULT_ROLE_PROMPT

    def test_string_role_prompt_stored_stripped(self) -> None:
        agent = make_basic_agent(role_prompt="  You are a poet.  ")
        assert agent.role_prompt == "You are a poet."

    def test_prompt_config_role_prompt_stored_as_is(self) -> None:
        config = PromptConfig(template="Act like {role}.", description="Custom role")
        agent = make_basic_agent(role_prompt=config)
        assert agent.role_prompt == "Act like {role}."

    def test_invalid_role_prompt_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="role_prompt"):
            make_basic_agent(role_prompt=123)  # type: ignore[arg-type]

    def test_role_prompt_has_no_setter(self) -> None:
        agent = make_basic_agent(role_prompt="You are a poet.")
        with pytest.raises(AttributeError):
            agent.role_prompt = "You are a chef."  # type: ignore[misc]


class TestBasicAgentRemovedAPISurfaces:
    """Locks down the pass-4 removals: no mutation API remains anywhere."""

    def test_no_update_prompt_method(self) -> None:
        agent = make_basic_agent()
        assert not hasattr(agent, "update_prompt")

    def test_no_set_context_properties_method(self) -> None:
        agent = make_basic_agent()
        assert not hasattr(agent, "set_context_properties")

    def test_no_set_extra_context_properties_method(self) -> None:
        agent = make_basic_agent()
        assert not hasattr(agent, "set_extra_context_properties")

    def test_constructor_rejects_extra_context_properties_kwarg(self) -> None:
        with pytest.raises(TypeError):
            make_basic_agent(extra_context_properties=[])  # type: ignore[call-arg]


class TestBasicAgentSchemaComposition:
    """Role-prompt placeholders are now a flat extra_parameters source."""

    def test_role_prompt_placeholder_becomes_top_level_param(self) -> None:
        config = PromptConfig(template="You are a {persona} assistant.", description="d")
        agent = make_basic_agent(role_prompt=config)

        names = [p.name for p in agent.parameters]
        assert names == ["prompt", "persona"]

    def test_static_role_prompt_produces_no_extra_params(self) -> None:
        agent = make_basic_agent(role_prompt="You are a generic assistant.")

        names = [p.name for p in agent.parameters]
        assert names == ["prompt"]

    def test_placeholder_default_from_field_specs_is_preserved(self) -> None:
        config = PromptConfig(
            template="Speak as {persona}.",
            description="d",
            field_specs={"persona": {"default": "a helper"}},
        )
        agent = make_basic_agent(role_prompt=config)

        persona_param = next(p for p in agent.parameters if p.name == "persona")
        assert persona_param.default == "a helper"

    def test_placeholder_name_colliding_with_pre_invoke_param_type_raises(self) -> None:
        def custom_pre(*, x: str) -> str:
            return x

        config = PromptConfig(
            template="Value: {x}.",
            description="d",
            field_specs={"x": {"type": "int"}},
        )
        with pytest.raises(AgentError):
            make_basic_agent(role_prompt=config, pre_invoke=custom_pre)


class _MergingAgent(BasicAgent):
    """Test-only subclass whose override adds one prompt-derived parameter."""

    def __init__(self, *, extra: list[ParamSpec], **kwargs: Any) -> None:
        self._extra = extra  # read by the hook, so set before super().__init__
        super().__init__(**kwargs)

    def _merge_prompt_parameters(self, role_params: list[ParamSpec]) -> list[ParamSpec]:
        return role_params + self._extra


class TestBasicAgentMergePromptParameters:
    def test_default_is_identity(self) -> None:
        agent = make_basic_agent()
        params = [ParamSpec(name="x", index=0, kind=ParamSpec.KEYWORD_ONLY, type=("str",))]

        assert agent._merge_prompt_parameters(params) is params

    def test_default_leaves_the_signature_to_the_role_prompt(self) -> None:
        config = PromptConfig(template="You are {persona}.", description="d")

        agent = make_basic_agent(role_prompt=config)

        assert [p.name for p in agent.parameters] == ["prompt", "persona"]

    def test_override_adds_parameters_to_the_signature(self) -> None:
        tone = ParamSpec(name="tone", index=0, kind=ParamSpec.KEYWORD_ONLY, type=("str",))
        config = PromptConfig(template="You are {persona}.", description="d")

        agent = _MergingAgent(
            extra=[tone],
            name="merging",
            namespace="tests",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            role_prompt=config,
        )

        assert [p.name for p in agent.parameters] == ["prompt", "persona", "tone"]


class TestBasicAgentInvoke:
    def test_invoke_renders_system_prompt_and_passes_to_engine(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_basic_agent(
            engine=engine,
            role_prompt="You are a poet.",
        )

        agent.invoke({"prompt": "Write a haiku."})

        assert engine.calls[0][0] == {"role": "system", "content": "You are a poet."}

    def test_invoke_renders_placeholder_in_system_prompt(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        config = PromptConfig(template="You are a {persona}.", description="d")
        agent = make_basic_agent(engine=engine, role_prompt=config)

        agent.invoke({"prompt": "Speak.", "persona": "pirate"})

        assert engine.calls[0][0]["content"] == "You are a pirate."

    def test_invoke_uses_placeholder_default_when_omitted(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        config = PromptConfig(
            template="Speak as {persona}.",
            description="d",
            field_specs={"persona": {"default": "a helper"}},
        )
        agent = make_basic_agent(engine=engine, role_prompt=config)

        agent.invoke({"prompt": "Speak."})

        assert engine.calls[0][0]["content"] == "Speak as a helper."

    def test_invoke_missing_required_placeholder_raises(self) -> None:
        config = PromptConfig(template="You are a {persona}.", description="d")
        agent = make_basic_agent(role_prompt=config)

        with pytest.raises(ValueError, match="persona"):
            agent.invoke({"prompt": "Speak."})

    def test_invoke_result_is_echo_of_prompt(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_basic_agent(engine=engine)

        result = agent.invoke({"prompt": "Hello."})

        assert result.result == f"ECHO: {wrap('Hello.')}"

    def test_invoke_appends_record_with_correct_prompt(self) -> None:
        agent = make_basic_agent(record_enabled=True)

        agent.invoke({"prompt": "What is AI?"})

        assert len(agent.get_conversation().records) == 1
        assert agent.get_conversation().records[0].user_prompt == "What is AI?"

    def test_invoke_record_llm_record_has_system_prompt_name_role(self) -> None:
        agent = make_basic_agent(record_enabled=True)

        agent.invoke({"prompt": "test"})

        record = agent.get_conversation().records[0]
        assert len(record.llm_records) == 1
        assert record.llm_records[0].system_prompt_name == "role"

    def test_invoke_record_messages_contains_last_user_message(self) -> None:
        agent = make_basic_agent(record_enabled=True)

        agent.invoke({"prompt": "hello"})

        llm_record = agent.get_conversation().records[0].llm_records[0]
        assert llm_record.messages == ({"role": "user", "content": wrap("hello")},)

    def test_invoke_never_stores_record_when_context_disabled(self) -> None:
        agent = make_basic_agent(record_enabled=False)

        agent.invoke({"prompt": "test 1"})
        agent.invoke({"prompt": "test 2"})

        assert agent.get_conversation().records == []


class TestBasicAgentAsyncInvoke:
    def test_async_invoke_result_matches_sync(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_basic_agent(engine=engine, role_prompt="You are helpful.")

        result = asyncio.run(agent.async_invoke({"prompt": "Hi."}))

        assert result.result == f"ECHO: {wrap('Hi.')}"

    def test_async_invoke_system_prompt_passed_to_engine(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_basic_agent(engine=engine, role_prompt="Async role.")

        asyncio.run(agent.async_invoke({"prompt": "Test."}))

        assert engine.calls[0][0]["content"] == "Async role."

    def test_async_invoke_renders_placeholder_in_system_prompt(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        config = PromptConfig(template="You are a {persona}.", description="d")
        agent = make_basic_agent(engine=engine, role_prompt=config)

        asyncio.run(agent.async_invoke({"prompt": "Speak.", "persona": "pirate"}))

        assert engine.calls[0][0]["content"] == "You are a pirate."

    def test_async_invoke_llm_record_has_system_prompt_name_role(self) -> None:
        agent = make_basic_agent(record_enabled=True)

        asyncio.run(agent.async_invoke({"prompt": "async test"}))

        assert agent.get_conversation().records[0].llm_records[0].system_prompt_name == "role"


class TestBasicAgentResponseSchema:
    def test_defaults_to_none(self) -> None:
        agent = make_basic_agent()
        assert agent.response_schema is None

    def test_stored_as_given(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        agent = make_basic_agent(response_schema=schema)
        assert agent.response_schema == schema

    def test_has_no_setter(self) -> None:
        agent = make_basic_agent()
        with pytest.raises(AttributeError):
            agent.response_schema = {"type": "object"}  # type: ignore[misc]

    def test_rejects_non_mapping_non_none(self) -> None:
        with pytest.raises(AgentError, match="response_schema"):
            make_basic_agent(response_schema="not a dict")  # type: ignore[arg-type]

    def test_act_forwards_schema_into_output_structure(self) -> None:
        engine = FakeLLMEngine(response_fn=lambda msgs: "plain text")
        schema = {"type": "object"}
        agent = make_basic_agent(engine=engine, response_schema=schema)

        agent.invoke({"prompt": "hi"})

        assert engine.payloads[-1]["output_structure"] == schema

    def test_act_omits_schema_when_unset(self) -> None:
        engine = FakeLLMEngine(response_fn=lambda msgs: "plain text")
        agent = make_basic_agent(engine=engine)

        agent.invoke({"prompt": "hi"})

        assert engine.payloads[-1]["output_structure"] is None

    def test_act_accepts_dict_result_when_schema_set(self) -> None:
        engine = FakeLLMEngine(responses=[{"answer": 42}])
        agent = make_basic_agent(engine=engine, response_schema={"type": "object"})

        result = agent.invoke({"prompt": "hi"})

        assert result.result == {"answer": 42}

    def test_act_accepts_list_result_when_schema_set(self) -> None:
        engine = FakeLLMEngine(responses=[["a", "b"]])
        agent = make_basic_agent(engine=engine, response_schema={"type": "array"})

        result = agent.invoke({"prompt": "hi"})

        assert result.result == ["a", "b"]

    def test_act_accepts_int_result_when_schema_set(self) -> None:
        engine = FakeLLMEngine(responses=[7])
        agent = make_basic_agent(engine=engine, response_schema={"type": "integer"})

        result = agent.invoke({"prompt": "hi"})

        assert result.result == 7

    def test_act_accepts_float_result_when_schema_set(self) -> None:
        engine = FakeLLMEngine(responses=[3.14])
        agent = make_basic_agent(engine=engine, response_schema={"type": "number"})

        result = agent.invoke({"prompt": "hi"})

        assert result.result == 3.14

    def test_act_accepts_bool_result_when_schema_set(self) -> None:
        engine = FakeLLMEngine(responses=[True])
        agent = make_basic_agent(engine=engine, response_schema={"type": "boolean"})

        result = agent.invoke({"prompt": "hi"})

        assert result.result is True

    def test_act_accepts_none_result_when_schema_set(self) -> None:
        engine = FakeLLMEngine(responses=[None])
        agent = make_basic_agent(engine=engine, response_schema={"type": "null"})

        result = agent.invoke({"prompt": "hi"})

        assert result.result is None

    def test_async_act_forwards_schema_and_accepts_structured_result(self) -> None:
        engine = FakeLLMEngine(responses=[{"answer": 42}])
        agent = make_basic_agent(engine=engine, response_schema={"type": "object"})

        result = asyncio.run(agent.async_invoke({"prompt": "hi"}))

        assert engine.payloads[-1]["output_structure"] == {"type": "object"}
        assert result.result == {"answer": 42}


class TestBasicAgentSerialization:
    def test_to_dict_includes_role_prompt_convenience_key(self) -> None:
        agent = make_basic_agent(role_prompt="Custom.")

        data = agent.to_dict()

        assert data["role_prompt"] == "Custom."

    def test_to_dict_type_is_basic_agent(self) -> None:
        agent = make_basic_agent()

        assert agent.to_dict()["type"] == "BasicAgent"

    def test_to_dict_has_system_prompts_with_role_key(self) -> None:
        agent = make_basic_agent(role_prompt="My role.")

        data = agent.to_dict()

        assert "system_prompts" in data
        assert "role" in data["system_prompts"]
        assert data["system_prompts"]["role"]["template"] == "My role."

    def test_to_dict_omits_extra_context_properties_key(self) -> None:
        agent = make_basic_agent()
        data = agent.to_dict()
        assert "extra_context_properties" not in data


def _user_message(engine: FakeLLMEngine, call: int = -1) -> str:
    return engine.calls[call][-1]["content"]


def _echo_agent(**kwargs: Any) -> tuple[FakeLLMEngine, BasicAgent]:
    engine = FakeLLMEngine(response_fn=echo_latest_user())
    return engine, make_basic_agent(engine=engine, **kwargs)


class TestBasicAgentTaskBlock:
    def test_empty_context_sends_wrapped_prompt_only(self) -> None:
        engine, agent = _echo_agent()

        agent.invoke({"prompt": "hello"})

        assert _user_message(engine) == wrap("hello")
        assert CONTEXT_START not in _user_message(engine)

    def test_messages_are_system_then_single_user_message(self) -> None:
        engine, agent = _echo_agent(role_prompt="Be brief.")

        agent.invoke({"prompt": "hello"})

        assert [m["role"] for m in engine.calls[0]] == ["system", "user"]

    def test_context_block_precedes_wrapped_prompt(self) -> None:
        engine, agent = _echo_agent()
        agent.get_conversation().remember(
            "draft", "Hello there", "the latest draft", readonly=True
        )

        agent.invoke({"prompt": "hello"})

        msg = _user_message(engine)
        assert msg.startswith(CONTEXT_START)
        assert "name: draft" in msg
        assert "type: str" in msg
        assert "readonly: yes" in msg
        assert "description: the latest draft" in msg
        assert "Hello there" in msg
        assert msg.endswith(TASK_PROMPT_END)
        assert msg.index(CONTEXT_END) < msg.index(TASK_PROMPT_START)

    def test_entry_ordering(self) -> None:
        engine, agent = _echo_agent()
        conv = agent.get_conversation()
        conv.remember("w_none1", "a")
        conv.remember("w_desc1", "a", "d")
        conv.remember("r_none1", "a", readonly=True)
        conv.remember("r_desc1", "a", "d", readonly=True)
        conv.remember("w_none2", "a")
        conv.remember("w_desc2", "a", "d")

        agent.invoke({"prompt": "hello"})

        msg = _user_message(engine)
        order = ["r_desc1", "r_none1", "w_desc1", "w_desc2", "w_none1", "w_none2"]
        positions = [msg.index(f"name: {name}\n") for name in order]
        assert positions == sorted(positions)

    def test_hidden_value_row(self) -> None:
        engine, agent = _echo_agent()
        conv = agent.get_conversation()
        conv.context_preview_limit = 0
        conv.remember("secret", "topsecret", "a hidden one")

        agent.invoke({"prompt": "hello"})

        msg = _user_message(engine)
        assert CONTEXT_START in msg
        assert "name: secret" in msg
        assert "value hidden" in msg
        assert "value:" not in msg
        assert "topsecret" not in msg

    def test_truncated_value(self) -> None:
        engine, agent = _echo_agent()
        conv = agent.get_conversation()
        conv.context_preview_limit = 5
        conv.remember("text", "abcdefghij")

        agent.invoke({"prompt": "hello"})

        assert "abcde...[truncated: 5 of 10 chars shown]" in _user_message(engine)

    @pytest.mark.parametrize(
        ("variable_limit", "conversation_limit", "shown"),
        [(3, 10, 3), (10, 3, 3), (None, 4, 4), (4, None, 4), (None, None, None)],
    )
    def test_effective_limit_is_the_smaller(
        self, variable_limit: int | None, conversation_limit: int | None, shown: int | None
    ) -> None:
        engine, agent = _echo_agent()
        conv = agent.get_conversation()
        conv.context_preview_limit = conversation_limit
        conv.remember("text", "abcdefghij", preview_limit=variable_limit)

        agent.invoke({"prompt": "hello"})

        msg = _user_message(engine)
        if shown is None:
            assert "abcdefghij" in msg
            assert "truncated" not in msg
        else:
            assert f"{'abcdefghij'[:shown]}...[truncated: {shown} of 10 chars shown]" in msg

    def test_forged_markers_are_elided(self) -> None:
        engine, agent = _echo_agent()
        conv = agent.get_conversation()
        conv.remember(
            "text", f"before {CONTEXT_END} after", f"describes {CONTEXT_START} here"
        )

        agent.invoke({"prompt": f"{TASK_PROMPT_START} and {TASK_PROMPT_END}"})

        msg = _user_message(engine)
        assert msg.count("[marker elided]") == 4
        for marker in (CONTEXT_START, CONTEXT_END, TASK_PROMPT_START, TASK_PROMPT_END):
            assert msg.count(marker) == 1

    def test_prompt_is_never_truncated(self) -> None:
        engine, agent = _echo_agent()
        conv = agent.get_conversation()
        conv.prompt_preview_limit = 1
        conv.context_preview_limit = 1
        conv.remember("text", "abc")
        long_prompt = "x" * 500

        agent.invoke({"prompt": long_prompt})

        assert long_prompt in _user_message(engine)

    def test_stored_prompt_is_raw_and_record_matches_sent_message(self) -> None:
        engine, agent = _echo_agent(record_enabled=True)
        agent.get_conversation().remember("text", "abc")

        agent.invoke({"prompt": "hello"})

        record = agent.get_conversation().records[0]
        assert record.user_prompt == "hello"
        assert record.llm_records[0].messages[-1]["content"] == _user_message(engine)

    def test_history_stays_raw_current_prompt_wrapped(self) -> None:
        engine, agent = _echo_agent(record_enabled=True)

        agent.invoke({"prompt": "hello"})
        agent.invoke({"prompt": "again"})

        second = engine.calls[1]
        assert second[1] == {"role": "user", "content": "hello"}
        assert second[-1]["content"] == wrap("again")

    def test_rendering_is_deterministic(self) -> None:
        engines = []
        for _ in range(2):
            engine, agent = _echo_agent()
            conv = agent.get_conversation()
            conv.remember("a", {"k": [1, 2]}, "first", render_mode="j")
            conv.remember("b", "text", readonly=True)
            agent.invoke({"prompt": "hello"})
            engines.append(engine)

        assert _user_message(engines[0]) == _user_message(engines[1])

    def test_bound_response_appears_next_call(self) -> None:
        engine = FakeLLMEngine(responses=["first reply", "second reply"])
        agent = make_basic_agent(
            engine=engine, record_enabled=True, bind_response_to="draft"
        )
        agent.get_conversation().remember("draft", "seed")

        agent.invoke({"prompt": "one"})
        agent.invoke({"prompt": "two"})

        assert "seed" in _user_message(engine, 0)
        second = _user_message(engine, 1)
        assert "name: draft" in second
        assert "first reply" in second

    def test_context_limit_edit_is_seen_next_render(self) -> None:
        engine, agent = _echo_agent()
        conv = agent.get_conversation()
        conv.remember("text", "abc")

        agent.invoke({"prompt": "one"})
        conv.context_preview_limit = 0
        agent.invoke({"prompt": "two"})

        assert "value hidden" not in _user_message(engine, 0)
        assert "value hidden" in _user_message(engine, 1)

    def test_async_invoke_sends_the_same_block(self) -> None:
        sync_engine, sync_agent = _echo_agent()
        async_engine, async_agent = _echo_agent()
        for agent in (sync_agent, async_agent):
            agent.get_conversation().remember("text", "abc", "d", readonly=True)

        sync_agent.invoke({"prompt": "hello"})
        asyncio.run(async_agent.async_invoke({"prompt": "hello"}))

        assert _user_message(async_engine) == _user_message(sync_engine)


class TestBasicAgentDefaultRenderMode:
    SCHEMA = {"type": "object"}

    def test_default_conversation_mode(self) -> None:
        assert make_basic_agent().get_conversation().render_mode == "s"
        schema_agent = make_basic_agent(response_schema=self.SCHEMA)
        assert schema_agent.get_conversation().render_mode == "j"

    def test_json_mode_survives_clear_memory(self) -> None:
        agent = make_basic_agent(response_schema=self.SCHEMA)

        agent.clear_memory()

        assert agent.get_conversation().render_mode == "j"

    def test_json_mode_survives_remove_default(self) -> None:
        agent = make_basic_agent(response_schema=self.SCHEMA)

        agent.remove_conversation("default")

        assert agent.get_conversation("default").render_mode == "j"

    def test_create_conversation_uses_the_agent_default(self) -> None:
        schema_agent = make_basic_agent(response_schema=self.SCHEMA)
        plain_agent = make_basic_agent()

        assert schema_agent.create_conversation("x").render_mode == "j"
        assert plain_agent.create_conversation("x").render_mode == "s"

    def test_explicit_render_mode_wins(self) -> None:
        schema_agent = make_basic_agent(response_schema=self.SCHEMA)
        plain_agent = make_basic_agent()

        assert schema_agent.create_conversation("x", render_mode="s").render_mode == "s"
        assert plain_agent.create_conversation("x", render_mode="p").render_mode == "p"

    def test_render_mode_none_is_accepted(self) -> None:
        agent = make_basic_agent(response_schema=self.SCHEMA)

        assert agent.create_conversation("x", render_mode=None).render_mode == "j"

    def test_bad_limit_raises_and_registers_nothing(self) -> None:
        agent = make_basic_agent()

        with pytest.raises(ValueError):
            agent.create_conversation("x", response_preview_limit=-1)

        assert "x" not in agent.conversation_names

    def test_import_conversation_keeps_its_own_mode(self) -> None:
        agent = make_basic_agent(response_schema=self.SCHEMA)
        other = Conversation(render_mode="r")

        agent.import_conversation(other, name="imported")

        assert agent.get_conversation("imported") is other
        assert other.render_mode == "r"

    def test_structured_reply_replays_as_json_in_history(self) -> None:
        engine = FakeLLMEngine(responses=[{"ok": True}, {"ok": False}])
        agent = make_basic_agent(
            engine=engine, record_enabled=True, response_schema=self.SCHEMA
        )

        agent.invoke({"prompt": "one"})
        agent.invoke({"prompt": "two"})

        replayed = engine.calls[1][2]
        assert replayed["role"] == "assistant"
        assert '"ok": true' in replayed["content"]
        assert "True" not in replayed["content"]
