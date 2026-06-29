from __future__ import annotations

from typing import Any, Mapping

import pytest
import asyncio

from atomic_agentic.agents.basic import BasicAgent
from atomic_agentic.exceptions import AgentError, AgentInvocationError
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.engines.LLMEngines import LLMEngine
from atomic_agentic.models.agents.records import AgentRecord, LLMRecord
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.models.results import LLMModelData, TokenUsage


# ─────────────────────────────────────────────────────────────────────────────
# Test engine
# ─────────────────────────────────────────────────────────────────────────────
class EchoLLMEngine(LLMEngine):
    """Deterministic echo engine that records all message batches received."""

    def __init__(self, *, prefix: str = "ECHO", **kwargs: Any) -> None:
        super().__init__(
            name="echo_engine",
            description="Echo engine for BasicAgent tests.",
            **kwargs,
        )
        self.prefix = prefix
        self.calls: list[list[dict[str, str]]] = []

    def _build_provider_payload(
        self,
        messages: list[dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        self.calls.append([dict(m) for m in messages])
        latest_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return {"latest_user": latest_user}

    def _call_provider(self, payload: Any) -> Any:
        return payload

    def _extract_text(self, response: Any) -> str:
        return f"{self.prefix}: {response['latest_user']}"

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        return TokenUsage(input_tokens=1, generated_tokens=1, total_tokens=2)

    def _get_model_data(self) -> LLMModelData:
        return LLMModelData(provider="echo")

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        return {"path": path}

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        return None


def make_basic_agent(
    *,
    engine: EchoLLMEngine | None = None,
    role_prompt: str | PromptConfig | None = None,
    context_enabled: bool = False,
    **kwargs: Any,
) -> BasicAgent:
    return BasicAgent(
        name="basic_agent",
        namespace="tests",
        description="BasicAgent under test.",
        llm_engine=engine or EchoLLMEngine(),
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

    def test_role_prompt_property_has_no_setter(self) -> None:
        agent = make_basic_agent()
        with pytest.raises(AttributeError):
            agent.role_prompt = "new prompt"  # type: ignore[misc]

    def test_update_prompt_role_key_raises_agent_error(self) -> None:
        agent = make_basic_agent()
        config = PromptConfig(template="Other.", description="d")
        with pytest.raises(AgentError, match="immutable"):
            agent.update_prompt("role", config)

    def test_update_prompt_non_role_key_succeeds(self) -> None:
        agent = make_basic_agent()
        config = PromptConfig(template="Supplemental.", description="extra")
        agent.update_prompt("extra", config)
        assert "extra" in agent.system_prompts


class TestBasicAgentContextKeys:
    def test_role_prompt_placeholders_appear_in_agent_schema(self) -> None:
        config = PromptConfig(template="You are a {persona} assistant.", description="d")
        agent = make_basic_agent(role_prompt=config)

        names = [p.name for p in agent.parameters]
        assert "persona" in names

    def test_role_prompt_placeholder_is_keyword_only_in_schema(self) -> None:
        config = PromptConfig(template="You are a {persona} assistant.", description="d")
        agent = make_basic_agent(role_prompt=config)

        persona_param = next(p for p in agent.parameters if p.name == "persona")
        assert persona_param.kind == "KEYWORD_ONLY"

    def test_static_role_prompt_produces_no_context_keys(self) -> None:
        agent = make_basic_agent(role_prompt="You are a generic assistant.")

        # Only expect: prompt, continue_from (no context keys)
        names = [p.name for p in agent.parameters]
        assert names == ["prompt", "continue_from"]

    def test_placeholder_default_injected_from_prompt_config(self) -> None:
        config = PromptConfig(
            template="Speak as {persona}.",
            description="d",
            defaults={"persona": "a helper"},
        )
        agent = make_basic_agent(role_prompt=config)

        persona_param = next(p for p in agent.parameters if p.name == "persona")
        assert persona_param.default == "a helper"


class TestBasicAgentInvoke:
    def test_invoke_renders_system_prompt_and_passes_to_engine(self) -> None:
        engine = EchoLLMEngine()
        agent = make_basic_agent(
            engine=engine,
            role_prompt="You are a poet.",
        )

        agent.invoke({"prompt": "Write a haiku."})

        assert engine.calls[0][0] == {"role": "system", "content": "You are a poet."}

    def test_invoke_renders_placeholder_in_system_prompt(self) -> None:
        engine = EchoLLMEngine()
        config = PromptConfig(template="You are a {persona}.", description="d")
        agent = make_basic_agent(engine=engine, role_prompt=config)

        agent.invoke({"prompt": "Speak.", "persona": "pirate"})

        assert engine.calls[0][0]["content"] == "You are a pirate."

    def test_invoke_result_is_echo_of_prompt(self) -> None:
        engine = EchoLLMEngine()
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
        engine = EchoLLMEngine()
        agent = make_basic_agent(engine=engine, role_prompt="You are helpful.")

        result = asyncio.run(agent.async_invoke({"prompt": "Hi."}))

        assert result.result == "ECHO: Hi."

    def test_async_invoke_system_prompt_passed_to_engine(self) -> None:
        engine = EchoLLMEngine()
        agent = make_basic_agent(engine=engine, role_prompt="Async role.")

        asyncio.run(agent.async_invoke({"prompt": "Test."}))

        assert engine.calls[0][0]["content"] == "Async role."

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
