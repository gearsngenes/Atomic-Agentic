from __future__ import annotations

from typing import Any, Mapping

import pytest
import asyncio

from atomic_agentic.tools import Tool
from atomic_agentic.agents.base import Agent
from atomic_agentic.core.Exceptions import AgentError, AgentInvocationError
from atomic_agentic.core.sentinels import NO_VAL
from atomic_agentic.engines.LLMEngines import LLMEngine
from atomic_agentic.agents.data_classes import AgentTurn


ROLE_PROMPT = "You are a deterministic test writer."


class StatefulEchoLLMEngine(LLMEngine):
    """Concrete deterministic LLMEngine for Agent tests.

    The engine records every normalized message batch it receives and returns
    the latest user message with a stable prefix. This makes context behavior
    observable without mocking provider SDKs.
    """

    def __init__(self, *, prefix: str = "ECHO", **kwargs: Any) -> None:
        super().__init__(
            name="stateful_echo_engine",
            description="Stateful echo engine.",
            **kwargs,
        )
        self.prefix = prefix
        self.calls: list[list[dict[str, str]]] = []

    def _build_provider_payload(
        self,
        messages: list[dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        copied_messages = [dict(message) for message in messages]
        self.calls.append(copied_messages)

        latest_user = next(
            (
                message["content"]
                for message in reversed(copied_messages)
                if message["role"] == "user"
            ),
            "",
        )

        return {
            "latest_user": latest_user,
            "message_count": len(copied_messages),
        }

    def _call_provider(self, payload: Any) -> Any:
        return payload

    def _extract_text(self, response: Any) -> str:
        return f"{self.prefix}: {response['latest_user']}"

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        return {"path": path}

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        return None


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


class NonStringAsyncLLMEngine(StatefulEchoLLMEngine):
    async def async_invoke(self, inputs: Mapping[str, Any]) -> Any:
        return 123

def make_agent(
    *,
    engine: StatefulEchoLLMEngine | None = None,
    context_enabled: bool = False,
    history_window: int | None = None,
    pre_invoke: Any = build_prompt,
    post_invoke: Any = package_response,
    role_prompt: str = ROLE_PROMPT,
) -> Agent:
    return Agent(
        name="writer_agent",
        description="Deterministic writer test agent.",
        llm_engine=engine or StatefulEchoLLMEngine(),
        role_prompt=role_prompt,
        context_enabled=context_enabled,
        history_window=history_window,
        pre_invoke=pre_invoke,
        post_invoke=post_invoke,
    )


class TestAgentPipeline:
    def test_pre_invoke_shapes_prompt_and_post_invoke_packages_result(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine)

        result = agent.invoke({"topic": "pytest", "tone": "strict"})

        expected_prompt = "Write about pytest in a strict tone."
        expected_raw = f"ECHO: {expected_prompt}"

        assert result == {
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
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine)

        agent.invoke({"topic": "agents", "tone": "concise"})

        assert engine.calls[0] == [
            {"role": "system", "content": ROLE_PROMPT},
            {"role": "user", "content": "Write about agents in a concise tone."},
        ]

    def test_default_identity_pre_and_post_invoke_path(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = Agent(
            name="identity_agent",
            description="Identity agent.",
            llm_engine=engine,
            role_prompt=ROLE_PROMPT,
            context_enabled=False,
        )

        result = agent.invoke({"prompt": "Hello from identity."})

        assert result == "ECHO: Hello from identity."
        assert engine.calls[0] == [
            {"role": "system", "content": ROLE_PROMPT},
            {"role": "user", "content": "Hello from identity."},
        ]

    def test_agent_schema_is_derived_from_pre_and_post_invoke(self) -> None:
        agent = make_agent()

        assert [(param.name, param.kind, param.default) for param in agent.parameters] == [
            ("topic", "POSITIONAL_OR_KEYWORD", NO_VAL),
            ("tone", "POSITIONAL_OR_KEYWORD", "neutral"),
        ]
        assert agent.return_type == "dict[str, Any]"


class TestAgentContext:
    def test_context_disabled_does_not_store_or_resend_history(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, context_enabled=False)

        first = agent.invoke({"topic": "pytest", "tone": "strict"})
        second = agent.invoke({"topic": "agents", "tone": "concise"})

        assert first["final"] == "ECHO: Write about pytest in a strict tone."
        assert second["final"] == "ECHO: Write about agents in a concise tone."
        assert agent.history == []

        assert len(engine.calls) == 2
        assert [message["role"] for message in engine.calls[0]] == ["system", "user"]
        assert [message["role"] for message in engine.calls[1]] == ["system", "user"]
        assert engine.calls[0][-1]["content"] == "Write about pytest in a strict tone."
        assert engine.calls[1][-1]["content"] == "Write about agents in a concise tone."

    def test_context_enabled_stores_history(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, context_enabled=True)

        first = agent.invoke({"topic": "pytest", "tone": "strict"})
        second = agent.invoke({"topic": "agents", "tone": "concise"})

        assert first["final"] == "ECHO: Write about pytest in a strict tone."
        assert second["final"] == "ECHO: Write about agents in a concise tone."

        assert [message["role"] for message in agent.history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert agent.history[0]["content"] == "Write about pytest in a strict tone."
        assert agent.history[1]["content"] == "ECHO: Write about pytest in a strict tone."
        assert agent.history[2]["content"] == "Write about agents in a concise tone."
        assert agent.history[3]["content"] == "ECHO: Write about agents in a concise tone."

    def test_context_enabled_resends_prior_history_on_second_call(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, context_enabled=True)

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

    def test_history_window_none_sends_all_prior_turns(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(
            engine=engine,
            context_enabled=True,
            history_window=None,
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

    def test_history_window_one_sends_only_last_prior_turn(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(
            engine=engine,
            context_enabled=True,
            history_window=1,
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

    def test_history_window_zero_sends_no_prior_turns_but_still_stores_history(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(
            engine=engine,
            context_enabled=True,
            history_window=0,
        )

        agent.invoke({"topic": "first topic", "tone": "plain"})
        agent.invoke({"topic": "second topic", "tone": "plain"})

        second_call_messages = engine.calls[1]

        assert [message["role"] for message in second_call_messages] == [
            "system",
            "user",
        ]
        assert second_call_messages[-1]["content"] == "Write about second topic in a plain tone."

        assert [message["role"] for message in agent.history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert agent.history[0]["content"] == "Write about first topic in a plain tone."
        assert agent.history[2]["content"] == "Write about second topic in a plain tone."

    def test_clear_memory_removes_stored_history(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, context_enabled=True)

        agent.invoke({"topic": "pytest", "tone": "strict"})

        assert agent.history

        agent.clear_memory()

        assert agent.history == []


class TestAgentValidation:
    def test_invoke_rejects_non_mapping_inputs(self) -> None:
        agent = make_agent()

        with pytest.raises(TypeError):
            agent.invoke(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_pre_invoke_returning_non_string_raises_at_invoke_time(self) -> None:
        agent = make_agent(pre_invoke=bad_pre_invoke)

        with pytest.raises(AgentInvocationError, match="pre_invoke returned non-string"):
            agent.invoke({"topic": "pytest"})

    def test_post_invoke_with_one_defaulted_parameter_is_allowed(self) -> None:
        agent = make_agent(post_invoke=defaulted_post_invoke)

        result = agent.invoke({"topic": "pytest", "tone": "strict"})

        assert result == "ECHO: WRITE ABOUT PYTEST IN A STRICT TONE."

    def test_post_invoke_with_two_required_arguments_raises(self) -> None:
        with pytest.raises(AgentError, match="exactly 1 required argument"):
            make_agent(post_invoke=bad_post_two_required_args)

    def test_context_enabled_setter_rejects_non_bool(self) -> None:
        agent = make_agent()

        with pytest.raises(ValueError, match="context_enabled"):
            agent.context_enabled = "yes"  # type: ignore[assignment]

    def test_history_window_setter_rejects_negative_values(self) -> None:
        agent = make_agent()

        with pytest.raises(ValueError, match="history_window"):
            agent.history_window = -1

    def test_constructor_rejects_negative_history_window(self) -> None:
        with pytest.raises(AgentError, match="history_window"):
            make_agent(history_window=-1)

    def test_llm_engine_setter_rejects_non_llm_engine(self) -> None:
        agent = make_agent()

        with pytest.raises(TypeError, match="llm_engine"):
            agent.llm_engine = object()  # type: ignore[assignment]


class TestAgentSerialization:
    def test_to_dict_includes_agent_configuration(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(
            engine=engine,
            context_enabled=True,
            history_window=1,
        )

        agent.invoke({"topic": "pytest", "tone": "strict"})

        data = agent.to_dict()

        assert data["type"] == "Agent"
        assert data["name"] == "writer_agent"
        assert data["description"] == "Deterministic writer test agent."
        assert data["role_prompt"] == ROLE_PROMPT
        assert data["context_enabled"] is True
        assert data["history_window"] == 1
        assert data["history"] == agent.history

        assert data["pre_invoke"]["name"] == "pre_invoke"
        assert data["post_invoke"]["name"] == "post_invoke"
        assert data["llm"]["type"] == "StatefulEchoLLMEngine"
        assert "secret" not in str(data)

    def test_history_property_returns_copy(self) -> None:
        agent = make_agent(context_enabled=True)

        agent.invoke({"topic": "pytest", "tone": "strict"})

        history_snapshot = agent.history
        history_snapshot.append({"role": "user", "content": "mutated"})

        assert len(history_snapshot) == len(agent.history) + 1
        assert all(message["content"] != "mutated" for message in agent.history)

class TestAgentAsyncInvoke:
    def test_async_invoke_shapes_prompt_and_post_invoke_packages_result(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine)

        result = asyncio.run(
            agent.async_invoke({"topic": "pytest", "tone": "strict"})
        )

        expected_prompt = "Write about pytest in a strict tone."
        expected_raw = f"ECHO: {expected_prompt}"

        assert result == {
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

    def test_async_context_enabled_stores_history(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, context_enabled=True)

        first = asyncio.run(
            agent.async_invoke({"topic": "pytest", "tone": "strict"})
        )
        second = asyncio.run(
            agent.async_invoke({"topic": "agents", "tone": "concise"})
        )

        assert first["final"] == "ECHO: Write about pytest in a strict tone."
        assert second["final"] == "ECHO: Write about agents in a concise tone."
        assert [message["role"] for message in agent.history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert agent.history[0]["content"] == "Write about pytest in a strict tone."
        assert agent.history[1]["content"] == "ECHO: Write about pytest in a strict tone."
        assert agent.history[2]["content"] == "Write about agents in a concise tone."
        assert agent.history[3]["content"] == "ECHO: Write about agents in a concise tone."

    def test_async_context_enabled_resends_prior_history_on_second_call(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, context_enabled=True)

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

        with pytest.raises(AgentInvocationError, match="pre_invoke returned non-string"):
            asyncio.run(agent.async_invoke({"topic": "pytest"}))

    def test_async_engine_non_string_response_raises_agent_invocation_error(self) -> None:
        agent = make_agent(engine=NonStringAsyncLLMEngine())

        with pytest.raises(AgentInvocationError, match="engine returned non-string"):
            asyncio.run(agent.async_invoke({"topic": "pytest", "tone": "strict"}))


class TestAgentAttachmentDelegation:
    def test_attach_delegates_to_engine_and_exposes_attachments(self, tmp_path: Any) -> None:
        path = tmp_path / "sample.txt"
        path.write_text("hello", encoding="utf-8")

        agent = make_agent()

        metadata = agent.attach(str(path))

        assert metadata["path"] == str(path)
        assert str(path) in agent.attachments
        assert agent.attachments[str(path)]["path"] == str(path)

    def test_detach_delegates_to_engine(self, tmp_path: Any) -> None:
        path = tmp_path / "sample.txt"
        path.write_text("hello", encoding="utf-8")

        agent = make_agent()
        agent.attach(str(path))

        assert agent.detach(str(path)) is True
        assert agent.detach(str(path)) is False
        assert str(path) not in agent.attachments

    def test_clear_attachments_delegates_to_engine(self, tmp_path: Any) -> None:
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("first", encoding="utf-8")
        second.write_text("second", encoding="utf-8")

        agent = make_agent()
        agent.attach(str(first))
        agent.attach(str(second))

        assert agent.attachments

        agent.clear_attachments()

        assert agent.attachments == {}


class TestAgentPropertySetters:
    def test_role_prompt_setter_rejects_non_string(self) -> None:
        agent = make_agent()

        with pytest.raises(TypeError, match="role_prompt"):
            agent.role_prompt = 123  # type: ignore[assignment]

    def test_role_prompt_setter_strips_whitespace(self) -> None:
        agent = make_agent()

        agent.role_prompt = "  New role prompt.  "

        assert agent.role_prompt == "New role prompt."

    def test_pre_invoke_setter_rebuilds_parameters(self) -> None:
        agent = make_agent()

        agent.pre_invoke = pre_with_two_fields

        assert [param.name for param in agent.parameters] == ["subject", "style"]
        assert agent.invoke({"subject": "pytest", "style": "direct"}) == {
            "final": "ECHO: pytest:direct",
            "length": len("ECHO: pytest:direct"),
            "was_postprocessed": True,
        }

    def test_pre_invoke_setter_rejects_non_string_return_type(self) -> None:
        agent = make_agent()

        with pytest.raises(AgentError, match="pre_invoke"):
            agent.pre_invoke = pre_returns_int

    def test_post_invoke_setter_rebuilds_return_type(self) -> None:
        agent = make_agent()

        agent.post_invoke = defaulted_post_invoke

        assert agent.return_type == "str"
        assert agent.invoke({"topic": "pytest", "tone": "strict"}) == (
            "ECHO: WRITE ABOUT PYTEST IN A STRICT TONE."
        )

    def test_post_invoke_setter_rejects_zero_parameter_callable(self) -> None:
        agent = make_agent()

        with pytest.raises(AgentError, match="post_invoke"):
            agent.post_invoke = post_zero_args

    def test_post_invoke_setter_accepts_one_required_plus_defaults(self) -> None:
        agent = make_agent()

        agent.post_invoke = post_one_required_plus_default

        assert agent.invoke({"topic": "pytest", "tone": "strict"}) == (
            "ECHO: Write about pytest in a strict tone.!"
        )

    def test_pre_invoke_setter_accepts_tool_instance(self) -> None:
        agent = make_agent()
        tool = Tool(
            function=pre_with_two_fields,
            name="custom_pre",
            namespace="tests",
            description="Custom preprocessor.",
        )

        agent.pre_invoke = tool

        assert agent.pre_invoke is tool
        assert [param.name for param in agent.parameters] == ["subject", "style"]

class UnexpectedMetadataAgent(Agent):
    def _invoke(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[Any, Mapping[str, Any]]:
        return "raw metadata response", {"unexpected": True}


class NonMappingMetadataAgent(Agent):
    def _invoke(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[Any, Any]:
        return "raw metadata response", []


class AsyncUnexpectedMetadataAgent(Agent):
    async def _ainvoke(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[Any, Mapping[str, Any]]:
        return "raw async metadata response", {"unexpected": True}


class AsyncNonMappingMetadataAgent(Agent):
    async def _ainvoke(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[Any, Any]:
        return "raw async metadata response", []


class TestAgentTurnHistory:
    def test_context_enabled_stores_agent_turn_objects(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, context_enabled=True)

        first = agent.invoke({"topic": "pytest", "tone": "strict"})
        second = agent.invoke({"topic": "agents", "tone": "concise"})

        assert first["final"] == "ECHO: Write about pytest in a strict tone."
        assert second["final"] == "ECHO: Write about agents in a concise tone."

        turns = agent.turn_history

        assert len(turns) == 2
        assert all(isinstance(turn, AgentTurn) for turn in turns)

        assert turns[0].prompt == "Write about pytest in a strict tone."
        assert turns[0].raw_response == "ECHO: Write about pytest in a strict tone."
        assert turns[0].final_response == first

        assert turns[1].prompt == "Write about agents in a concise tone."
        assert turns[1].raw_response == "ECHO: Write about agents in a concise tone."
        assert turns[1].final_response == second

    def test_turn_history_returns_shallow_copy(self) -> None:
        agent = make_agent(context_enabled=True)
        agent.invoke({"topic": "pytest", "tone": "strict"})

        snapshot = agent.turn_history
        snapshot.append(
            AgentTurn(
                prompt="mutated",
                raw_response="mutated",
                final_response="mutated",
            )
        )

        assert len(snapshot) == len(agent.turn_history) + 1
        assert all(turn.prompt != "mutated" for turn in agent.turn_history)

    def test_response_preview_limit_only_affects_rendered_history(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = Agent(
            name="preview_agent",
            description="Preview test agent.",
            llm_engine=engine,
            role_prompt=ROLE_PROMPT,
            context_enabled=True,
            pre_invoke=build_prompt,
            post_invoke=package_response,
            response_preview_limit=10,
        )

        result = agent.invoke({"topic": "abcdefghijklmnopqrstuvwxyz", "tone": "plain"})

        expected_prompt = "Write about abcdefghijklmnopqrstuvwxyz in a plain tone."
        expected_raw = f"ECHO: {expected_prompt}"

        turn = agent.turn_history[0]

        assert turn.prompt == expected_prompt
        assert turn.raw_response == expected_raw
        assert turn.final_response == result

        with pytest.warns(DeprecationWarning):
            rendered_history = agent.history

        assert rendered_history[1]["content"] == expected_raw[:10] + "..."

    def test_history_property_warns_and_returns_rendered_messages(self) -> None:
        agent = make_agent(context_enabled=True)
        agent.invoke({"topic": "pytest", "tone": "strict"})

        with pytest.warns(DeprecationWarning):
            rendered_history = agent.history

        assert [message["role"] for message in rendered_history] == ["user", "assistant"]
        assert rendered_history[0]["content"] == "Write about pytest in a strict tone."
        assert rendered_history[1]["content"] == "ECHO: Write about pytest in a strict tone."


class TestAgentTurnRendering:
    def test_rendered_history_uses_raw_response_by_default(self) -> None:
        agent = Agent(
            name="raw_render_agent",
            description="Raw render test agent.",
            llm_engine=StatefulEchoLLMEngine(),
            role_prompt=ROLE_PROMPT,
            context_enabled=True,
            pre_invoke=build_prompt,
            post_invoke=package_response,
        )

        agent.invoke({"topic": "pytest", "tone": "strict"})

        with pytest.warns(DeprecationWarning):
            rendered_history = agent.history

        assert rendered_history[1]["content"] == "ECHO: Write about pytest in a strict tone."

    def test_rendered_history_uses_final_response_when_configured(self) -> None:
        agent = Agent(
            name="final_render_agent",
            description="Final render test agent.",
            llm_engine=StatefulEchoLLMEngine(),
            role_prompt=ROLE_PROMPT,
            context_enabled=True,
            pre_invoke=build_prompt,
            post_invoke=package_response,
            assistant_response_source="final",
        )

        agent.invoke({"topic": "pytest", "tone": "strict"})

        with pytest.warns(DeprecationWarning):
            rendered_history = agent.history

        assert "was_postprocessed" in rendered_history[1]["content"]
        assert "ECHO: Write about pytest in a strict tone." in rendered_history[1]["content"]


class TestAgentTurnMetadataContract:
    def test_base_make_turn_rejects_unexpected_metadata(self) -> None:
        agent = UnexpectedMetadataAgent(
            name="unexpected_metadata_agent",
            description="Unexpected metadata test agent.",
            llm_engine=StatefulEchoLLMEngine(),
            context_enabled=True,
        )

        with pytest.raises(AgentInvocationError, match="unexpected metadata"):
            agent.invoke({"prompt": "run"})

    def test_invoke_rejects_non_mapping_metadata(self) -> None:
        agent = NonMappingMetadataAgent(
            name="non_mapping_metadata_agent",
            description="Non-mapping metadata test agent.",
            llm_engine=StatefulEchoLLMEngine(),
            context_enabled=True,
        )

        with pytest.raises(AgentInvocationError, match="_invoke returned non-mapping metadata"):
            agent.invoke({"prompt": "run"})

    def test_async_base_make_turn_rejects_unexpected_metadata(self) -> None:
        agent = AsyncUnexpectedMetadataAgent(
            name="async_unexpected_metadata_agent",
            description="Async unexpected metadata test agent.",
            llm_engine=StatefulEchoLLMEngine(),
            context_enabled=True,
        )

        with pytest.raises(AgentInvocationError, match="unexpected metadata"):
            asyncio.run(agent.async_invoke({"prompt": "run"}))

    def test_async_invoke_rejects_non_mapping_metadata(self) -> None:
        agent = AsyncNonMappingMetadataAgent(
            name="async_non_mapping_metadata_agent",
            description="Async non-mapping metadata test agent.",
            llm_engine=StatefulEchoLLMEngine(),
            context_enabled=True,
        )

        with pytest.raises(AgentInvocationError, match="_ainvoke returned non-mapping metadata"):
            asyncio.run(agent.async_invoke({"prompt": "run"}))
