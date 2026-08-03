from __future__ import annotations

from typing import Any, Mapping

import pytest
import asyncio

from atomic_agentic.agents.basic import BasicAgent
from atomic_agentic.exceptions import AgentError
from atomic_agentic.models.agents.prompts import PromptConfig
from fake_engines import FakeLLMEngine, echo_latest_user


def make_basic_agent(
    *,
    engine: FakeLLMEngine | None = None,
    role_prompt: str | PromptConfig | None = None,
    context_enabled: bool = False,
    **kwargs: Any,
) -> BasicAgent:
    return BasicAgent(
        name="basic_agent",
        namespace="tests",
        description="BasicAgent under test.",
        llm_engine=engine or FakeLLMEngine(response_fn=echo_latest_user()),
        role_prompt=role_prompt,
        context_enabled=context_enabled,
        **kwargs,
    )


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
        assert names == ["prompt", "persona", "run_id"]

    def test_static_role_prompt_produces_no_extra_params(self) -> None:
        agent = make_basic_agent(role_prompt="You are a generic assistant.")

        names = [p.name for p in agent.parameters]
        assert names == ["prompt", "run_id"]

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

        assert result.result == "ECHO: Hello."

    def test_invoke_appends_record_with_correct_prompt(self) -> None:
        agent = make_basic_agent()

        agent.invoke({"prompt": "What is AI?"})

        assert len(agent.records) == 1
        assert agent.records[0].user_prompt == "What is AI?"

    def test_invoke_record_llm_record_has_system_prompt_name_role(self) -> None:
        agent = make_basic_agent()

        agent.invoke({"prompt": "test"})

        record = agent.records[0]
        assert len(record.llm_records) == 1
        assert record.llm_records[0].system_prompt_name == "role"

    def test_invoke_record_messages_contains_last_user_message(self) -> None:
        agent = make_basic_agent()

        agent.invoke({"prompt": "hello"})

        llm_record = agent.records[0].llm_records[0]
        assert llm_record.messages == ({"role": "user", "content": "hello"},)

    def test_invoke_always_stores_record_regardless_of_context_enabled(self) -> None:
        agent = make_basic_agent(context_enabled=False)

        agent.invoke({"prompt": "test 1"})
        agent.invoke({"prompt": "test 2"})

        assert len(agent.records) == 2


class TestBasicAgentAsyncInvoke:
    def test_async_invoke_result_matches_sync(self) -> None:
        engine = FakeLLMEngine(response_fn=echo_latest_user())
        agent = make_basic_agent(engine=engine, role_prompt="You are helpful.")

        result = asyncio.run(agent.async_invoke({"prompt": "Hi."}))

        assert result.result == "ECHO: Hi."

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
        agent = make_basic_agent()

        asyncio.run(agent.async_invoke({"prompt": "async test"}))

        assert agent.records[0].llm_records[0].system_prompt_name == "role"


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
