from __future__ import annotations

from typing import Any

import pytest
import asyncio

from atomic_agentic.agents.basic import BasicAgent
from atomic_agentic.constants.agents import DEFAULT_INSTRUCTIONS
from atomic_agentic.exceptions import AgentError, ToolInvocationError
from atomic_agentic.models.agents.conversations import Conversation
from atomic_agentic.core.atomic_invokable import AtomicInvokable
from atomic_agentic.core.prompt_template import PromptTemplate
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.tools import Tool
from ..fake_engines import FakeLLMEngine, echo_latest_user

CONTEXT_START = "==== RUNTIME CONTEXT [START] ===="
CONTEXT_END = "==== RUNTIME CONTEXT [END] ===="
TASK_PROMPT_START = "==== USER TASK PROMPT [START] ===="
TASK_PROMPT_END = "==== USER TASK PROMPT [END] ===="


def make_basic_agent(
    *,
    engine: FakeLLMEngine | None = None,
    instructions: Any = DEFAULT_INSTRUCTIONS,
    record_enabled: bool = False,
    **kwargs: Any,
) -> BasicAgent:
    return BasicAgent(
        name="basic_agent",
        description="BasicAgent under test.",
        llm_engine=engine or FakeLLMEngine(response_fn=echo_latest_user()),
        instructions=instructions,
        record_enabled=record_enabled,
        **kwargs,
    )


def wrap(prompt: str) -> str:
    """The user message BasicAgent sends for ``prompt`` with empty context."""
    return f"{TASK_PROMPT_START}\n{prompt}\n{TASK_PROMPT_END}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestBasicAgentInstructions:
    def test_omitted_instructions_use_the_default_constant(self) -> None:
        agent = make_basic_agent()
        assert agent.instructions.template == DEFAULT_INSTRUCTIONS

    def test_none_instructions_raise_type_error(self) -> None:
        with pytest.raises(TypeError, match="instructions"):
            make_basic_agent(instructions=None)

    def test_blank_instructions_raise_value_error(self) -> None:
        with pytest.raises(ValueError):
            make_basic_agent(instructions="   ")

    def test_string_instructions_are_kept_verbatim(self) -> None:
        agent = make_basic_agent(instructions="  You are a poet.  ")
        assert agent.instructions.template == "  You are a poet.  "

    def test_prompt_template_instructions_used_as_is(self) -> None:
        config = PromptTemplate(template="Act like {role}.", name="t", description="Custom role")
        agent = make_basic_agent(instructions=config)
        assert agent.instructions is config
        assert agent.instructions.template == "Act like {role}."

    def test_invalid_instructions_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="instructions"):
            make_basic_agent(instructions=123)

    def test_instructions_have_no_setter(self) -> None:
        agent = make_basic_agent(instructions="You are a poet.")
        with pytest.raises(AttributeError):
            agent.instructions = "You are a chef."  # type: ignore[misc]

    def test_role_prompt_is_gone(self) -> None:
        with pytest.raises(TypeError):
            make_basic_agent(role_prompt="x")  # type: ignore[call-arg]
        assert not hasattr(make_basic_agent(), "role_prompt")
        assert not hasattr(BasicAgent, "DEFAULT_ROLE_PROMPT")


def _persona_slot(persona: str) -> str:
    return f"You are a {persona}."


class TestBasicAgentSlotTypes:
    def test_callable_slot_is_wrapped_and_named(self) -> None:
        agent = make_basic_agent(instructions=_persona_slot)

        assert isinstance(agent.instructions, Tool)
        assert agent.instructions.name == "basic_agent_instructions"
        assert [p.name for p in agent.parameters] == ["prompt", "persona"]

    def test_string_slot_is_a_prompt_template_with_the_coerced_name(self) -> None:
        agent = make_basic_agent(instructions="You are a {persona}.")

        assert isinstance(agent.instructions, PromptTemplate)
        assert agent.instructions.name == "basic_agent_instructions"

    def test_tool_slot_is_used_as_is(self) -> None:
        tool = Tool(_persona_slot, name="my_slot", description="d")
        agent = make_basic_agent(instructions=tool)

        assert agent.instructions is tool
        assert agent.instructions.name == "my_slot"

    @pytest.mark.parametrize("slot", ["string", "callable", "tool", "template"])
    def test_every_slot_type_renders_the_system_message(self, slot: str) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        instructions = {
            "string": "You are a {persona}.",
            "callable": _persona_slot,
            "tool": Tool(_persona_slot, name="slot", description="d"),
            "template": PromptTemplate(template="You are a {persona}.", name="t", description="d"),
        }[slot]
        agent = make_basic_agent(engine=engine, instructions=instructions)

        agent.invoke({"prompt": "hi", "persona": "pirate"})

        assert engine.calls[0][0] == {"role": "system", "content": "You are a pirate."}

    def test_agent_slot_renders_its_reply_as_the_system_message(self) -> None:
        inner = make_basic_agent(instructions="inner")
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_basic_agent(engine=engine, instructions=inner)

        agent.invoke({"prompt": "hi"})

        assert engine.calls[0][0]["content"] == f"ECHO: {wrap('hi')}"

    def test_async_agent_slot_uses_async_invoke_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inner = make_basic_agent(instructions="inner")
        calls = {"async": 0}
        real_async = inner.async_invoke

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("sync invoke must not be called on an async path")

        async def counted(*args: Any, **kwargs: Any) -> Any:
            calls["async"] += 1
            return await real_async(*args, **kwargs)

        monkeypatch.setattr(inner, "invoke", boom)
        monkeypatch.setattr(inner, "async_invoke", counted)
        agent = make_basic_agent(instructions=inner)

        result = asyncio.run(agent.async_invoke({"prompt": "hi"}))

        assert calls["async"] == 1
        assert result.result.startswith("ECHO: ")


class TestBasicAgentSlotRendering:
    def test_slot_runs_exactly_once_per_invoke(self) -> None:
        calls: list[str] = []

        def slot(persona: str) -> str:
            calls.append(persona)
            return f"You are a {persona}."

        agent = make_basic_agent(instructions=slot)

        agent.invoke({"prompt": "a", "persona": "x"})
        agent.invoke({"prompt": "b", "persona": "y"})

        assert calls == ["x", "y"]

    def test_non_str_slot_result_raises_type_error_naming_slot_and_type(self) -> None:
        def slot() -> int:
            return 5

        agent = make_basic_agent(instructions=slot)

        with pytest.raises(TypeError, match=r"basic_agent_instructions.*int"):
            agent.invoke({"prompt": "hi"})
        with pytest.raises(TypeError, match=r"basic_agent_instructions.*int"):
            asyncio.run(agent.async_invoke({"prompt": "hi"}))

    def test_slot_errors_are_not_rewrapped_by_the_agent(self) -> None:
        def slot() -> str:
            raise RuntimeError("slot blew up")

        agent = make_basic_agent(instructions=slot)

        # A toolified callable surfaces as the Tool's own ToolInvocationError.
        with pytest.raises(ToolInvocationError, match="slot blew up"):
            agent.invoke({"prompt": "hi"})

    def test_system_message_is_unframed_task_instructions(self) -> None:
        agent = make_basic_agent(instructions="You are a {persona}.")
        task = agent.create_task(agent.filter_inputs({"prompt": "hi", "persona": "pirate"}))

        assert task.instructions == ""
        agent.pre_process(task)

        assert task.instructions == "You are a pirate."
        assert agent._render_system_message(task) == [
            {"role": "system", "content": "You are a pirate."}
        ]

    def test_async_pre_process_fills_the_same_field(self) -> None:
        agent = make_basic_agent(instructions="You are a {persona}.")
        task = agent.create_task(agent.filter_inputs({"prompt": "hi", "persona": "pirate"}))

        asyncio.run(agent.async_pre_process(task))

        assert task.instructions == "You are a pirate."

    def test_slot_receives_only_its_own_names(self) -> None:
        seen: dict[str, Any] = {}

        def slot(persona: str) -> str:
            seen["persona"] = persona
            return persona

        def pre(prompt: str, topic: str = "t") -> str:
            return prompt

        agent = make_basic_agent(instructions=slot, pre_invoke=pre)
        task = agent.create_task(
            agent.filter_inputs({"prompt": "hi", "persona": "x", "topic": "y"})
        )

        assert task.agentic_inputs == {"persona": "x"}
        agent.pre_process(task)
        assert seen == {"persona": "x"}

    def test_slot_declaring_runtime_context_receives_exactly_the_runtime_dict(self) -> None:
        seen: dict[str, Any] = {}

        def slot(persona: str, **runtime_context: Any) -> str:
            seen["runtime"] = dict(runtime_context)
            return persona

        agent = make_basic_agent(instructions=slot, use_runtime_context=True)

        agent.invoke({"prompt": "hi", "persona": "x", "runtime_context": {"a": 1}})

        assert seen["runtime"] == {"a": 1}

    def test_slot_placeholder_overlapping_pre_invoke_warns_and_pre_wins(self) -> None:
        def pre(prompt: str, persona: str = "pre") -> str:
            return prompt

        with pytest.warns(UserWarning, match="persona"):
            agent = make_basic_agent(instructions="You are a {persona}.", pre_invoke=pre)

        assert next(p for p in agent.parameters if p.name == "persona").default == "pre"


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
    """Instructions placeholders join the signature as a parameter source."""

    def test_instructions_placeholder_becomes_top_level_param(self) -> None:
        config = PromptTemplate(template="You are a {persona} assistant.", name="t", description="d")
        agent = make_basic_agent(instructions=config)

        names = [p.name for p in agent.parameters]
        assert names == ["prompt", "persona"]

    def test_static_instructions_produce_no_extra_params(self) -> None:
        agent = make_basic_agent(instructions="You are a generic assistant.")

        names = [p.name for p in agent.parameters]
        assert names == ["prompt"]

    def test_placeholder_default_from_explicit_parameter_is_preserved(self) -> None:
        config = PromptTemplate(
            template="Speak as {persona}.",
            name="t",
            description="d",
            parameters=[
                ParamSpec(
                    name="persona",
                    index=0,
                    kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                    type="Any",
                    default="a helper",
                )
            ],
        )
        agent = make_basic_agent(instructions=config)

        persona_param = next(p for p in agent.parameters if p.name == "persona")
        assert persona_param.default == "a helper"

    def test_role_placeholder_is_positional_or_keyword(self) -> None:
        config = PromptTemplate(template="You are a {persona}.", name="t", description="d")
        agent = make_basic_agent(instructions=config)

        persona_param = next(p for p in agent.parameters if p.name == "persona")
        assert persona_param.kind == ParamSpec.POSITIONAL_OR_KEYWORD

    def test_pre_invoke_keyword_only_declaration_wins_reconciliation(self) -> None:
        def custom_pre(*, persona: str) -> str:
            return persona

        config = PromptTemplate(template="You are a {persona}.", name="t", description="d")
        agent = make_basic_agent(instructions=config, pre_invoke=custom_pre)

        persona_param = next(p for p in agent.parameters if p.name == "persona")
        assert persona_param.kind == ParamSpec.KEYWORD_ONLY

    def test_placeholder_can_be_passed_by_keyword_through_call(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        config = PromptTemplate(template="You are a {persona}.", name="t", description="d")
        agent = make_basic_agent(engine=engine, instructions=config)

        agent("hi", persona="x")

        assert engine.calls[0][0]["content"] == "You are a x."

    def test_placeholder_name_colliding_with_pre_invoke_param_type_raises(self) -> None:
        def custom_pre(*, x: str) -> str:
            return x

        config = PromptTemplate(
            template="Value: {x}.",
            name="t",
            description="d",
            parameters=[
                ParamSpec(name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int")
            ],
        )
        with pytest.raises(AgentError):
            make_basic_agent(instructions=config, pre_invoke=custom_pre)


class _SourcingAgent(BasicAgent):
    """Test-only subclass whose override adds one more parameter source."""

    def __init__(self, *, extra: AtomicInvokable, **kwargs: Any) -> None:
        self._extra = extra  # read by the hook, so set before super().__init__
        super().__init__(**kwargs)

    def _extra_parameter_sources(self) -> list[AtomicInvokable]:
        return [*super()._extra_parameter_sources(), self._extra]


class TestBasicAgentParameterSources:
    def test_sources_are_the_instructions_slot(self) -> None:
        agent = make_basic_agent()

        assert agent._extra_parameter_sources() == [agent.instructions]

    def test_default_leaves_the_signature_to_the_instructions(self) -> None:
        config = PromptTemplate(template="You are {persona}.", name="t", description="d")

        agent = make_basic_agent(instructions=config)

        assert [p.name for p in agent.parameters] == ["prompt", "persona"]

    def test_override_adds_parameters_after_the_instructions_slot(self) -> None:
        tone = PromptTemplate(template="{tone}", name="tone_src", description="d")
        config = PromptTemplate(template="You are {persona}.", name="t", description="d")

        agent = _SourcingAgent(
            extra=tone,
            name="sourcing",
            description="d",
            llm_engine=FakeLLMEngine(response_fn=echo_latest_user()),
            instructions=config,
        )

        assert [p.name for p in agent.parameters] == ["prompt", "persona", "tone"]


class TestBasicAgentInvoke:
    def test_invoke_renders_system_prompt_and_passes_to_engine(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_basic_agent(
            engine=engine,
            instructions="You are a poet.",
        )

        agent.invoke({"prompt": "Write a haiku."})

        assert engine.calls[0][0] == {"role": "system", "content": "You are a poet."}

    def test_invoke_renders_placeholder_in_system_prompt(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        config = PromptTemplate(template="You are a {persona}.", name="t", description="d")
        agent = make_basic_agent(engine=engine, instructions=config)

        agent.invoke({"prompt": "Speak.", "persona": "pirate"})

        assert engine.calls[0][0]["content"] == "You are a pirate."

    def test_invoke_uses_placeholder_default_when_omitted(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        config = PromptTemplate(
            template="Speak as {persona}.",
            name="t",
            description="d",
            parameters=[
                ParamSpec(
                    name="persona",
                    index=0,
                    kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                    type="Any",
                    default="a helper",
                )
            ],
        )
        agent = make_basic_agent(engine=engine, instructions=config)

        agent.invoke({"prompt": "Speak."})

        assert engine.calls[0][0]["content"] == "Speak as a helper."

    def test_invoke_missing_required_placeholder_raises(self) -> None:
        config = PromptTemplate(template="You are a {persona}.", name="t", description="d")
        agent = make_basic_agent(instructions=config)

        with pytest.raises(KeyError, match="persona"):
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

    def test_invoke_record_has_one_llm_record(self) -> None:
        agent = make_basic_agent(record_enabled=True)

        agent.invoke({"prompt": "test"})

        record = agent.get_conversation().records[0]
        assert len(record.llm_records) == 1

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
        agent = make_basic_agent(engine=engine, instructions="You are helpful.")

        result = asyncio.run(agent.async_invoke({"prompt": "Hi."}))

        assert result.result == f"ECHO: {wrap('Hi.')}"

    def test_async_invoke_system_prompt_passed_to_engine(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_basic_agent(engine=engine, instructions="Async role.")

        asyncio.run(agent.async_invoke({"prompt": "Test."}))

        assert engine.calls[0][0]["content"] == "Async role."

    def test_async_invoke_renders_placeholder_in_system_prompt(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        config = PromptTemplate(template="You are a {persona}.", name="t", description="d")
        agent = make_basic_agent(engine=engine, instructions=config)

        asyncio.run(agent.async_invoke({"prompt": "Speak.", "persona": "pirate"}))

        assert engine.calls[0][0]["content"] == "You are a pirate."

    def test_async_invoke_has_one_llm_record(self) -> None:
        agent = make_basic_agent(record_enabled=True)

        asyncio.run(agent.async_invoke({"prompt": "async test"}))

        assert len(agent.get_conversation().records[0].llm_records) == 1


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
    def test_to_dict_nests_the_instructions_slot_dict(self) -> None:
        agent = make_basic_agent(instructions="Custom.")

        data = agent.to_dict()

        assert data["instructions"] == agent.instructions.to_dict()
        assert data["instructions"]["template"] == "Custom."
        assert "role_prompt" not in data

    def test_to_dict_type_is_basic_agent(self) -> None:
        agent = make_basic_agent()

        assert agent.to_dict()["type"] == "BasicAgent"

    def test_to_dict_reports_instructions_not_a_prompt_registry(self) -> None:
        agent = make_basic_agent(instructions="My role.")

        data = agent.to_dict()

        assert data["instructions"]["template"] == "My role."
        assert "system_prompts" not in data

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
        engine, agent = _echo_agent(instructions="Be brief.")

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
            engine=engine, record_enabled=True, response_to_context="draft"
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

    def test_hand_built_conversation_keeps_its_own_default(self) -> None:
        schema_agent = make_basic_agent(response_schema=self.SCHEMA)
        built = Conversation()

        schema_agent.import_conversation(built, name="x")

        assert schema_agent.get_conversation("x") is built
        assert built.render_mode == "s"
        assert schema_agent.get_conversation().render_mode == "j"

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
