from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import pytest
import asyncio

from atomic_agentic.tools import Tool
from atomic_agentic.agents.base import Agent
from atomic_agentic.exceptions import AgentError, AgentInvocationError, ToolInvocationError
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.engines.LLMEngines import LLMEngine
from atomic_agentic.models.agents.records import AgentRecord
from atomic_agentic.models.results import LLMModelData, LLMResult, TokenUsage


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

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        return TokenUsage(input_tokens=1, generated_tokens=1, total_tokens=2)

    def _get_model_data(self) -> LLMModelData:
        return LLMModelData(provider="stateful-echo")

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
    """Stub mimicking the AtomicResult `.result` access without LLMResult's str constraint."""

    def __init__(self, result: Any) -> None:
        self.result = result


class NonStringAsyncLLMEngine(StatefulEchoLLMEngine):
    async def async_invoke(self, inputs: Mapping[str, Any]) -> Any:
        return _NonStringEngineEnvelope(123)

def make_agent(
    *,
    engine: StatefulEchoLLMEngine | None = None,
    context_enabled: bool = False,
    history_window: int | None = None,
    pre_invoke: Any = build_prompt,
    post_invoke: Any = package_response,
    post_result_key: str | None = None,
    passthrough_inputs: list[str] | None = None,
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
        post_result_key=post_result_key,
        passthrough_inputs=passthrough_inputs,
    )


class TestAgentPipeline:
    def test_pre_invoke_shapes_prompt_and_post_invoke_packages_result(self) -> None:
        engine = StatefulEchoLLMEngine()
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

        assert result.result == "ECHO: Hello from identity."
        assert engine.calls[0] == [
            {"role": "system", "content": ROLE_PROMPT},
            {"role": "user", "content": "Hello from identity."},
        ]

    def test_agent_schema_defaults_to_pre_invoke_parameters_without_passthroughs(self) -> None:
        agent = make_agent()

        assert [(param.name, param.kind, param.default) for param in agent.parameters] == [
            ("topic", "POSITIONAL_OR_KEYWORD", NO_VAL),
            ("tone", "POSITIONAL_OR_KEYWORD", "neutral"),
            ("continue_from", "KEYWORD_ONLY", None),
        ]
        assert agent.return_type == "dict[str, Any]"


class TestAgentSchemaComposition:
    def test_post_only_passthrough_is_added_as_keyword_only_parameter(self) -> None:
        agent = make_agent(
            post_invoke=post_with_suffix,
            passthrough_inputs=["suffix"],
        )

        assert [(param.name, param.kind, param.default) for param in agent.parameters] == [
            ("topic", "POSITIONAL_OR_KEYWORD", NO_VAL),
            ("tone", "POSITIONAL_OR_KEYWORD", "neutral"),
            ("suffix", "KEYWORD_ONLY", NO_VAL),
            ("continue_from", "KEYWORD_ONLY", None),
        ]

    def test_post_only_passthrough_preserves_post_default(self) -> None:
        agent = make_agent(
            post_invoke=post_with_custom_key,
            post_result_key="raw",
            passthrough_inputs=["suffix"],
        )

        suffix_param = next(param for param in agent.parameters if param.name == "suffix")

        assert suffix_param.kind == "KEYWORD_ONLY"
        assert suffix_param.default == "!"

    def test_overlapping_passthrough_preserves_pre_invoke_default(self) -> None:
        agent = make_agent(
            post_invoke=post_with_overlap,
            passthrough_inputs=["tone"],
        )

        tone_param = next(param for param in agent.parameters if param.name == "tone")
        result = agent.invoke({"topic": "pytest"})

        assert tone_param.default == "neutral"
        assert result.result == {
            "result": "ECHO: Write about pytest in a neutral tone.",
            "tone": "neutral",
        }

    def test_overlapping_passthrough_type_mismatch_does_not_raise(self) -> None:
        agent = make_agent(
            post_invoke=post_with_type_mismatch,
            passthrough_inputs=["tone"],
        )

        assert agent.invoke({"topic": "pytest", "tone": "strict"}).result == (
            "ECHO: Write about pytest in a strict tone.|tone=strict"
        )


class TestAgentPostInvokeRouting:
    def test_post_invoke_required_passthrough_is_allowed_when_configured(self) -> None:
        agent = make_agent(
            post_invoke=bad_post_two_required_args,
            passthrough_inputs=["suffix"],
        )

        result = agent.invoke({"topic": "pytest", "tone": "strict", "suffix": "!"})

        assert result.result == "ECHO: Write about pytest in a strict tone.!"

    def test_missing_required_passthrough_fails_at_post_invoke_time(self) -> None:
        agent = make_agent(
            post_invoke=post_with_suffix,
            passthrough_inputs=["suffix"],
        )

        with pytest.raises(
            AgentInvocationError,
            match="post_invoke Tool failed:.*missing required",
        ):
            agent.invoke({"topic": "pytest", "tone": "strict"})

    def test_custom_post_result_key_routes_raw_result(self) -> None:
        agent = make_agent(
            post_invoke=post_with_custom_key,
            post_result_key="raw",
            passthrough_inputs=["suffix"],
        )

        result = agent.invoke({"topic": "pytest", "tone": "strict", "suffix": "?"})

        assert agent.post_result_key == "raw"
        assert result.result == "ECHO: Write about pytest in a strict tone.?"

    def test_post_result_key_defaults_to_first_post_parameter(self) -> None:
        agent = make_agent(post_invoke=post_with_custom_key)

        assert agent.post_result_key == "raw"

    def test_empty_post_result_key_raises(self) -> None:
        with pytest.raises(AgentError, match="post_result_key"):
            make_agent(post_invoke=post_with_custom_key, post_result_key="  ")

    def test_unknown_post_result_key_raises(self) -> None:
        with pytest.raises(AgentError, match="post_result_key"):
            make_agent(post_invoke=post_with_custom_key, post_result_key="missing")

    def test_post_result_key_cannot_also_be_passthrough(self) -> None:
        with pytest.raises(AgentError, match="post_result_key"):
            make_agent(
                post_invoke=post_with_custom_key,
                post_result_key="raw",
                passthrough_inputs=["raw"],
            )

    def test_passthrough_inputs_none_defaults_to_empty_copy(self) -> None:
        agent = make_agent()
        passthroughs = agent.passthrough_inputs
        passthroughs.append("mutated")

        assert agent.passthrough_inputs == []

    def test_passthrough_inputs_must_be_list(self) -> None:
        with pytest.raises(AgentError, match="passthrough_inputs"):
            make_agent(
                post_invoke=post_with_suffix,
                passthrough_inputs=("suffix",),  # type: ignore[arg-type]
            )

    def test_passthrough_inputs_rejects_empty_name(self) -> None:
        with pytest.raises(AgentError, match="passthrough_inputs"):
            make_agent(post_invoke=post_with_suffix, passthrough_inputs=[" "])

    def test_passthrough_inputs_strips_names(self) -> None:
        agent = make_agent(
            post_invoke=post_with_suffix,
            passthrough_inputs=[" suffix "],
        )

        assert agent.passthrough_inputs == ["suffix"]

    def test_passthrough_inputs_rejects_duplicates_after_stripping(self) -> None:
        with pytest.raises(AgentError, match="duplicate"):
            make_agent(
                post_invoke=post_with_suffix,
                passthrough_inputs=["suffix", " suffix "],
            )

    def test_passthrough_input_must_name_post_parameter(self) -> None:
        with pytest.raises(AgentError, match="passthrough_inputs"):
            make_agent(
                post_invoke=post_with_suffix,
                passthrough_inputs=["missing"],
            )

    def test_post_only_variadic_passthrough_raises(self) -> None:
        with pytest.raises(AgentError, match="Post-only passthrough"):
            make_agent(
                post_invoke=post_with_post_only_args,
                passthrough_inputs=["extras"],
            )

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


class TestAgentContext:
    def test_context_disabled_does_not_store_or_resend_history(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, context_enabled=False)

        first = agent.invoke({"topic": "pytest", "tone": "strict"})
        second = agent.invoke({"topic": "agents", "tone": "concise"})

        assert first.result["final"] == "ECHO: Write about pytest in a strict tone."
        assert second.result["final"] == "ECHO: Write about agents in a concise tone."
        assert agent.records == []

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

        assert first.result["final"] == "ECHO: Write about pytest in a strict tone."
        assert second.result["final"] == "ECHO: Write about agents in a concise tone."

        rendered = agent.render_turn(agent.records[0])
        assert [m["role"] for m in rendered] == ["user", "assistant"]
        assert rendered[0]["content"] == "Write about pytest in a strict tone."
        assert rendered[1]["content"] == "ECHO: Write about pytest in a strict tone."

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

        assert len(agent.records) == 2
        assert agent.records[0].user_prompt == "Write about first topic in a plain tone."
        assert agent.records[1].user_prompt == "Write about second topic in a plain tone."

    def test_clear_memory_removes_stored_history(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, context_enabled=True)

        agent.invoke({"topic": "pytest", "tone": "strict"})

        assert agent.records

        agent.clear_memory()

        assert agent.records == []


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

        assert result.result == "ECHO: WRITE ABOUT PYTEST IN A STRICT TONE."

    def test_post_invoke_required_passthrough_not_configured_raises(self) -> None:
        with pytest.raises(AgentError, match="required parameter"):
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
        assert data["records"] == [turn.to_dict() for turn in agent.records]

        assert data["pre_invoke"]["name"] == "pre_invoke"
        assert data["post_invoke"]["name"] == "post_invoke"
        assert data["post_result_key"] == agent.post_result_key
        assert data["passthrough_inputs"] == agent.passthrough_inputs
        assert data["llm"]["type"] == "StatefulEchoLLMEngine"
        assert "secret" not in str(data)

    def test_to_dict_includes_custom_post_routing_configuration(self) -> None:
        agent = make_agent(
            post_invoke=post_with_custom_key,
            post_result_key="raw",
            passthrough_inputs=["suffix"],
        )

        data = agent.to_dict()

        assert data["post_result_key"] == "raw"
        assert data["passthrough_inputs"] == ["suffix"]

class TestAgentAsyncInvoke:
    def test_async_invoke_shapes_prompt_and_post_invoke_packages_result(self) -> None:
        engine = StatefulEchoLLMEngine()
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

    def test_async_context_enabled_stores_history(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, context_enabled=True)

        first = asyncio.run(
            agent.async_invoke({"topic": "pytest", "tone": "strict"})
        )
        second = asyncio.run(
            agent.async_invoke({"topic": "agents", "tone": "concise"})
        )

        assert first.result["final"] == "ECHO: Write about pytest in a strict tone."
        assert second.result["final"] == "ECHO: Write about agents in a concise tone."

        rendered = agent.render_turn(agent.records[0])
        assert [m["role"] for m in rendered] == ["user", "assistant"]
        assert rendered[0]["content"] == "Write about pytest in a strict tone."
        assert rendered[1]["content"] == "ECHO: Write about pytest in a strict tone."

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

    def test_async_invoke_passes_passthrough_inputs_to_post_invoke(self) -> None:
        agent = make_agent(
            post_invoke=post_with_suffix,
            passthrough_inputs=["suffix"],
        )

        result = asyncio.run(
            agent.async_invoke({"topic": "pytest", "tone": "strict", "suffix": "!"})
        )

        assert result.result == "ECHO: Write about pytest in a strict tone.!"


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


class TestAgentMutableRuntimeProperties:
    def test_role_prompt_setter_rejects_non_string(self) -> None:
        agent = make_agent()

        with pytest.raises(TypeError, match="role_prompt"):
            agent.role_prompt = 123  # type: ignore[assignment]

    def test_role_prompt_setter_strips_whitespace(self) -> None:
        agent = make_agent()

        agent.role_prompt = "  New role prompt.  "

        assert agent.role_prompt == "New role prompt."

    def test_pre_and_post_invoke_are_read_only_lifecycle_references(self) -> None:
        agent = make_agent()

        with pytest.raises(AttributeError):
            agent.pre_invoke = build_prompt  # type: ignore[misc]

        with pytest.raises(AttributeError):
            agent.post_invoke = package_response  # type: ignore[misc]

    def test_constructor_with_custom_pre_invoke_builds_parameters(self) -> None:
        agent = make_agent(pre_invoke=pre_with_two_fields)

        assert [param.name for param in agent.parameters] == ["subject", "style", "continue_from"]
        assert agent.invoke({"subject": "pytest", "style": "direct"}).result == {
            "final": "ECHO: pytest:direct",
            "length": len("ECHO: pytest:direct"),
            "was_postprocessed": True,
        }

    def test_constructor_rejects_pre_invoke_non_string_return_type(self) -> None:
        with pytest.raises(AgentError, match="pre_invoke"):
            make_agent(pre_invoke=pre_returns_int)

    def test_constructor_with_custom_post_invoke_sets_return_type(self) -> None:
        agent = make_agent(post_invoke=defaulted_post_invoke)

        assert agent.return_type == "str"
        assert agent.invoke({"topic": "pytest", "tone": "strict"}).result == (
            "ECHO: WRITE ABOUT PYTEST IN A STRICT TONE."
        )

    def test_constructor_rejects_zero_parameter_post_invoke(self) -> None:
        with pytest.raises(AgentError, match="post_invoke"):
            make_agent(post_invoke=post_zero_args)

    def test_constructor_accepts_post_invoke_one_required_plus_defaults(self) -> None:
        agent = make_agent(post_invoke=post_one_required_plus_default)

        assert agent.invoke({"topic": "pytest", "tone": "strict"}).result == (
            "ECHO: Write about pytest in a strict tone.!"
        )

    def test_constructor_wraps_pre_invoke_tool_instance_with_lifecycle_identity(self) -> None:
        tool = Tool(
            function=pre_with_two_fields,
            name="custom_pre",
            namespace="tests",
            description="Custom preprocessor.",
        )
        agent = make_agent(pre_invoke=tool)

        assert agent.pre_invoke is not tool
        assert isinstance(agent.pre_invoke, Tool)
        assert agent.pre_invoke.function is tool
        assert agent.pre_invoke.wraps_invokable is True

        assert agent.pre_invoke.name == "pre_invoke"
        assert agent.pre_invoke.namespace == "writer_agent"
        assert agent.pre_invoke.description == (
            "The tool that preprocesses inputs into a string for Agent writer_agent"
        )

        assert tool.name == "custom_pre"
        assert tool.namespace == "tests"
        assert tool.description == "Custom preprocessor."

        assert [param.name for param in agent.parameters] == ["subject", "style", "continue_from"]
        assert agent.invoke({"subject": "pytest", "style": "direct"}).result == {
            "final": "ECHO: pytest:direct",
            "length": len("ECHO: pytest:direct"),
            "was_postprocessed": True,
        }

class StringDraftAgent(Agent):
    def _invoke(
        self,
        turns: list[AgentRecord],
        prompt: str,
    ) -> Any:
        return "raw response, not a draft AgentRecord", {}


class MappingDraftAgent(Agent):
    def _invoke(
        self,
        turns: list[AgentRecord],
        prompt: str,
    ) -> Any:
        return {"generated_response": "raw response"}, {}


class AsyncStringDraftAgent(Agent):
    async def _ainvoke(
        self,
        turns: list[AgentRecord],
        prompt: str,
    ) -> Any:
        return "raw async response, not a draft AgentRecord", {}


class AsyncMappingDraftAgent(Agent):
    async def _ainvoke(
        self,
        turns: list[AgentRecord],
        prompt: str,
    ) -> Any:
        return {"generated_response": "raw async response"}, {}


def make_mutation_record(*, user_prompt: str = "mutated") -> AgentRecord:
    return AgentRecord(
        user_prompt=user_prompt,
        generated_response="mutated",
    )


class TestAgentRecordHistory:
    def test_context_enabled_stores_agent_record_objects(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, context_enabled=True)

        first = agent.invoke({"topic": "pytest", "tone": "strict"})
        second = agent.invoke({"topic": "agents", "tone": "concise"})

        assert first.result["final"] == "ECHO: Write about pytest in a strict tone."
        assert second.result["final"] == "ECHO: Write about agents in a concise tone."

        turns = agent.records

        assert len(turns) == 2
        assert all(isinstance(turn, AgentRecord) for turn in turns)

        assert turns[0].user_prompt == "Write about pytest in a strict tone."
        assert turns[0].generated_response == "ECHO: Write about pytest in a strict tone."
        assert turns[0].final_result.result == first.result

        assert turns[1].user_prompt == "Write about agents in a concise tone."
        assert turns[1].generated_response == "ECHO: Write about agents in a concise tone."
        assert turns[1].final_result.result == second.result

    def test_turn_history_returns_shallow_copy(self) -> None:
        agent = make_agent(context_enabled=True)
        agent.invoke({"topic": "pytest", "tone": "strict"})

        snapshot = agent.records
        snapshot.append(make_mutation_record())

        assert len(snapshot) == len(agent.records) + 1
        assert all(turn.user_prompt != "mutated" for turn in agent.records)

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

        turn = agent.records[0]

        assert turn.user_prompt == expected_prompt
        assert turn.generated_response == expected_raw
        assert turn.final_result.result == result.result

        rendered = agent.render_turn(agent.records[0])
        assert rendered[1]["content"] == expected_raw[:10] + "..."


class TestAgentRecordRendering:
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

        rendered = agent.render_turn(agent.records[0])
        assert rendered[1]["content"] == "ECHO: Write about pytest in a strict tone."

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

        rendered = agent.render_turn(agent.records[0])
        assert "was_postprocessed" in rendered[1]["content"]
        assert "ECHO: Write about pytest in a strict tone." in rendered[1]["content"]


class TestAgentDraftRecordContract:
    def test_invoke_rejects_string_draft(self) -> None:
        agent = StringDraftAgent(
            name="string_draft_agent",
            description="String draft test agent.",
            llm_engine=StatefulEchoLLMEngine(),
            context_enabled=True,
        )

        with pytest.raises(
            AgentInvocationError,
            match=r"_invoke returned non-AgentRecord draft \(type=<class 'str'>\)",
        ):
            agent.invoke({"prompt": "run"})

    def test_invoke_rejects_mapping_draft(self) -> None:
        agent = MappingDraftAgent(
            name="mapping_draft_agent",
            description="Mapping draft test agent.",
            llm_engine=StatefulEchoLLMEngine(),
            context_enabled=True,
        )

        with pytest.raises(
            AgentInvocationError,
            match=r"_invoke returned non-AgentRecord draft \(type=<class 'dict'>\)",
        ):
            agent.invoke({"prompt": "run"})

    def test_async_invoke_rejects_string_draft(self) -> None:
        agent = AsyncStringDraftAgent(
            name="async_string_draft_agent",
            description="Async string draft test agent.",
            llm_engine=StatefulEchoLLMEngine(),
            context_enabled=True,
        )

        with pytest.raises(
            AgentInvocationError,
            match=r"_ainvoke returned non-AgentRecord draft \(type=<class 'str'>\)",
        ):
            asyncio.run(agent.async_invoke({"prompt": "run"}))

    def test_async_invoke_rejects_mapping_draft(self) -> None:
        agent = AsyncMappingDraftAgent(
            name="async_mapping_draft_agent",
            description="Async mapping draft test agent.",
            llm_engine=StatefulEchoLLMEngine(),
            context_enabled=True,
        )

        with pytest.raises(
            AgentInvocationError,
            match=r"_ainvoke returned non-AgentRecord draft \(type=<class 'dict'>\)",
        ):
            asyncio.run(agent.async_invoke({"prompt": "run"}))


# ---------------------------------------------------------------------------
# Module-level helper for history tests
# ---------------------------------------------------------------------------

def invoke_and_get_record(agent: Agent, prompt: str) -> AgentRecord:
    """Invoke the agent with a single-prompt input and return the committed record."""
    agent.invoke({"prompt": prompt})
    return agent.records[-1]


def pre_with_varkw(**kwargs: Any) -> str:
    """Pre-invoke that accepts any keyword arguments; used for varkw schema tests."""
    return str(kwargs)


# ---------------------------------------------------------------------------
# TestAgentContinueFromSchema
# ---------------------------------------------------------------------------

class TestAgentContinueFromSchema:
    """Tests for the reserved ``continue_from`` parameter in the Agent schema."""

    def test_continue_from_present_in_parameters(self) -> None:
        agent = make_agent()
        assert any(p.name == "continue_from" for p in agent.parameters)

    def test_continue_from_is_keyword_only(self) -> None:
        agent = make_agent()
        param = next(p for p in agent.parameters if p.name == "continue_from")
        assert param.kind == "KEYWORD_ONLY"

    def test_continue_from_default_is_none(self) -> None:
        agent = make_agent()
        param = next(p for p in agent.parameters if p.name == "continue_from")
        assert param.default is None

    def test_continue_from_type_annotation(self) -> None:
        agent = make_agent()
        param = next(p for p in agent.parameters if p.name == "continue_from")
        assert param.type == "str | None"

    def test_continue_from_with_custom_pre_invoke(self) -> None:
        agent = make_agent(pre_invoke=build_prompt)
        assert any(p.name == "continue_from" for p in agent.parameters)

    def test_continue_from_with_varkw_pre_invoke(self) -> None:
        # continue_from must be inserted before **kwargs, not appended after.
        agent = make_agent(pre_invoke=pre_with_varkw, post_invoke=None)
        names = [p.name for p in agent.parameters]
        assert "continue_from" in names
        cf_idx = names.index("continue_from")
        # Find **kwargs position
        varkw_idx = next(
            (i for i, p in enumerate(agent.parameters) if p.kind == "VAR_KEYWORD"),
            None,
        )
        assert varkw_idx is not None
        assert cf_idx < varkw_idx

    def test_continue_from_conflict_raises_agent_error(self) -> None:
        def conflicting_pre(continue_from: str) -> str:
            return continue_from

        with pytest.raises(AgentError, match="reserved"):
            make_agent(pre_invoke=conflicting_pre)

    def test_continue_from_not_in_pre_inputs(self) -> None:
        received_pre_inputs: list[dict[str, Any]] = []

        def capturing_pre(prompt: str) -> str:
            received_pre_inputs.append({"prompt": prompt})
            return prompt

        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, pre_invoke=capturing_pre, post_invoke=None, context_enabled=True)
        agent.invoke({"prompt": "hello", "continue_from": "new"})

        assert received_pre_inputs
        assert "continue_from" not in received_pre_inputs[-1]

    def test_continue_from_not_in_post_inputs(self) -> None:
        received_post_inputs: list[dict[str, Any]] = []

        def capturing_post(result: str) -> str:
            received_post_inputs.append({"result": result})
            return result

        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, pre_invoke=None, post_invoke=capturing_post, context_enabled=True)
        agent.invoke({"prompt": "hello", "continue_from": "new"})

        assert received_post_inputs
        assert "continue_from" not in received_post_inputs[-1]

    def test_continue_from_survives_filter_extraneous_inputs_true(self) -> None:
        # With filter_extraneous_inputs=True, continue_from is a declared
        # parameter and must not be stripped as extraneous.
        engine = StatefulEchoLLMEngine()
        agent = Agent(
            name="strict_agent",
            description="Strict filter agent.",
            llm_engine=engine,
            filter_extraneous_inputs=True,
            context_enabled=True,
        )
        # Should not raise; continue_from="new" means no prior turns used.
        agent.invoke({"prompt": "hello", "continue_from": "new"})
        assert len(agent.records) == 1
        assert agent.records[0].prev is None


# ---------------------------------------------------------------------------
# TestAgentGetConversation
# ---------------------------------------------------------------------------

class TestAgentGetConversation:
    """Tests for ``Agent.get_conversation``."""

    def _make_history_agent(self, history_window: int | None = None) -> Agent:
        return make_agent(pre_invoke=None, post_invoke=None, context_enabled=True, history_window=history_window)

    def test_get_conversation_turns_zero_raises_value_error(self) -> None:
        agent = self._make_history_agent()
        with pytest.raises(ValueError, match="turns"):
            agent.get_conversation(turns=0)

    def test_get_conversation_empty_history_returns_empty_list(self) -> None:
        agent = self._make_history_agent()
        assert agent.get_conversation() == []

    def test_get_conversation_no_run_id_returns_latest_chain(self) -> None:
        agent = self._make_history_agent()
        invoke_and_get_record(agent, "first")
        invoke_and_get_record(agent, "second")
        invoke_and_get_record(agent, "third")

        chain = agent.get_conversation()
        assert len(chain) == 3
        assert chain[-1] is agent.records[-1]
        # oldest-first
        assert chain[0] is agent.records[0]

    def test_get_conversation_turns_limit(self) -> None:
        agent = self._make_history_agent()
        for i in range(5):
            invoke_and_get_record(agent, f"turn {i}")

        chain = agent.get_conversation(turns=3)
        assert len(chain) == 3
        # Should be the 3 most recent, oldest-first
        assert chain[-1] is agent.records[-1]
        assert chain[0] is agent.records[2]

    def test_get_conversation_turns_none_returns_full_chain(self) -> None:
        agent = self._make_history_agent()
        for i in range(4):
            invoke_and_get_record(agent, f"turn {i}")

        chain = agent.get_conversation(turns=None)
        assert len(chain) == 4

    def test_get_conversation_oldest_first_ordering(self) -> None:
        agent = self._make_history_agent()
        invoke_and_get_record(agent, "alpha")
        invoke_and_get_record(agent, "beta")
        invoke_and_get_record(agent, "gamma")

        chain = agent.get_conversation()
        assert chain[0].final_result is agent.records[0].final_result

    def test_get_conversation_run_id_finds_correct_record(self) -> None:
        agent = self._make_history_agent()
        invoke_and_get_record(agent, "first")
        invoke_and_get_record(agent, "second")
        invoke_and_get_record(agent, "third")

        middle_run_id = agent.records[1].final_result.run_id
        chain = agent.get_conversation(run_id=middle_run_id)
        # Walking from second: second.prev = first, first.prev = None
        assert len(chain) == 2
        assert chain[0] is agent.records[0]
        assert chain[1] is agent.records[1]

    def test_get_conversation_unresolvable_run_id_raises(self) -> None:
        agent = self._make_history_agent()
        invoke_and_get_record(agent, "only")

        with pytest.raises(AgentInvocationError, match="run_id"):
            agent.get_conversation(run_id="not-a-real-id")

    def test_get_conversation_single_record(self) -> None:
        agent = self._make_history_agent()
        invoke_and_get_record(agent, "only")

        chain = agent.get_conversation()
        assert len(chain) == 1
        assert chain[0] is agent.records[0]

    def test_get_conversation_branched_history(self) -> None:
        agent = self._make_history_agent()
        # root → child A (normal) → root B (fresh start)
        invoke_and_get_record(agent, "root")
        root_run_id = agent.records[0].final_result.run_id

        invoke_and_get_record(agent, "child A")
        child_a_run_id = agent.records[1].final_result.run_id

        agent.invoke({"prompt": "root B", "continue_from": "new"})
        root_b_run_id = agent.records[2].final_result.run_id

        chain_a = agent.get_conversation(run_id=child_a_run_id)
        assert len(chain_a) == 2
        assert chain_a[0].final_result.run_id == root_run_id
        assert chain_a[1].final_result.run_id == child_a_run_id

        chain_b = agent.get_conversation(run_id=root_b_run_id)
        assert len(chain_b) == 1
        assert chain_b[0].final_result.run_id == root_b_run_id

    def test_get_conversation_chain_shorter_than_turns_limit(self) -> None:
        agent = self._make_history_agent()
        invoke_and_get_record(agent, "first")
        invoke_and_get_record(agent, "second")

        chain = agent.get_conversation(turns=10)
        assert len(chain) == 2


# ---------------------------------------------------------------------------
# TestAgentContinueFromInvoke
# ---------------------------------------------------------------------------

class TestAgentContinueFromInvoke:
    """Tests for ``continue_from`` behavior in ``invoke`` and ``async_invoke``."""

    def _make_ctx_agent(
        self,
        engine: StatefulEchoLLMEngine | None = None,
        history_window: int | None = None,
    ) -> Agent:
        return make_agent(
            engine=engine or StatefulEchoLLMEngine(),
            pre_invoke=None,
            post_invoke=None,
            context_enabled=True,
            history_window=history_window,
        )

    def test_continue_from_none_uses_history_window(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = self._make_ctx_agent(engine=engine, history_window=2)

        for i in range(4):
            agent.invoke({"prompt": f"turn {i}"})

        # 5th call with continue_from=None (default): should see exactly 2 prior turns
        agent.invoke({"prompt": "fifth", "continue_from": None})

        last_call = engine.calls[-1]
        # system + 2 prior user/assistant pairs + current user = 6 messages
        assert len(last_call) == 6
        roles = [m["role"] for m in last_call]
        assert roles == ["system", "user", "assistant", "user", "assistant", "user"]

    def test_continue_from_new_produces_empty_turns(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = self._make_ctx_agent(engine=engine, history_window=None)

        agent.invoke({"prompt": "first"})
        agent.invoke({"prompt": "second"})
        agent.invoke({"prompt": "third", "continue_from": "new"})

        last_call = engine.calls[-1]
        # Only system + current user message
        assert len(last_call) == 2
        assert last_call[0]["role"] == "system"
        assert last_call[1]["role"] == "user"
        assert last_call[1]["content"] == "third"

    def test_continue_from_new_record_has_prev_none(self) -> None:
        agent = self._make_ctx_agent()

        agent.invoke({"prompt": "first"})
        agent.invoke({"prompt": "second", "continue_from": "new"})

        assert agent.records[-1].prev is None

    def test_continue_from_new_history_still_appended(self) -> None:
        agent = self._make_ctx_agent()

        agent.invoke({"prompt": "first"})
        before = len(agent.records)
        agent.invoke({"prompt": "second", "continue_from": "new"})

        assert len(agent.records) == before + 1

    def test_continue_from_new_parallel_branches(self) -> None:
        agent = self._make_ctx_agent()

        agent.invoke({"prompt": "root"})
        agent.invoke({"prompt": "branch A", "continue_from": "new"})
        agent.invoke({"prompt": "branch B", "continue_from": "new"})

        assert len(agent.records) == 3
        assert agent.records[1].prev is None
        assert agent.records[2].prev is None

    def test_continue_from_valid_run_id_selects_branch(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = self._make_ctx_agent(engine=engine, history_window=None)

        agent.invoke({"prompt": "root A"})
        root_a_run_id = agent.records[0].final_result.run_id

        agent.invoke({"prompt": "root B", "continue_from": "new"})

        agent.invoke({"prompt": "child of A", "continue_from": root_a_run_id})

        # The last call should include root A's user message but not root B's
        last_call = engine.calls[-1]
        contents = [m["content"] for m in last_call]
        assert any("root A" in c for c in contents)
        assert not any("root B" in c for c in contents)

    def test_continue_from_invalid_run_id_raises_at_invoke(self) -> None:
        agent = self._make_ctx_agent()

        agent.invoke({"prompt": "first"})

        with pytest.raises(AgentInvocationError):
            agent.invoke({"prompt": "second", "continue_from": "bad-id"})

    def test_prev_stamped_when_turns_nonempty(self) -> None:
        agent = self._make_ctx_agent()

        agent.invoke({"prompt": "first"})
        agent.invoke({"prompt": "second"})

        first_record = agent.records[0]
        second_record = agent.records[1]
        assert second_record.prev is first_record

    def test_prev_none_when_context_disabled(self) -> None:
        # context_enabled=False: no history is stored at all
        agent = make_agent(pre_invoke=None, post_invoke=None, context_enabled=False)
        agent.invoke({"prompt": "only"})
        assert len(agent.records) == 0

    def test_prev_none_when_history_window_zero(self) -> None:
        agent = self._make_ctx_agent(history_window=0)

        agent.invoke({"prompt": "first"})
        agent.invoke({"prompt": "second"})

        assert agent.records[1].prev is None

    def test_prev_none_on_first_invocation(self) -> None:
        agent = self._make_ctx_agent()
        agent.invoke({"prompt": "only"})
        assert agent.records[0].prev is None

    def test_async_invoke_continue_from_new_parity(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = self._make_ctx_agent(engine=engine, history_window=None)

        asyncio.run(agent.async_invoke({"prompt": "first"}))
        asyncio.run(agent.async_invoke({"prompt": "second"}))
        asyncio.run(agent.async_invoke({"prompt": "third", "continue_from": "new"}))

        last_call = engine.calls[-1]
        assert len(last_call) == 2
        assert last_call[0]["role"] == "system"
        assert last_call[1]["content"] == "third"
        assert agent.records[-1].prev is None

    def test_async_invoke_branching_parity(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = self._make_ctx_agent(engine=engine, history_window=None)

        asyncio.run(agent.async_invoke({"prompt": "root A"}))
        root_a_run_id = agent.records[0].final_result.run_id

        asyncio.run(agent.async_invoke({"prompt": "root B", "continue_from": "new"}))

        asyncio.run(
            agent.async_invoke({"prompt": "child of A", "continue_from": root_a_run_id})
        )

        last_call = engine.calls[-1]
        contents = [m["content"] for m in last_call]
        assert any("root A" in c for c in contents)
        assert not any("root B" in c for c in contents)
