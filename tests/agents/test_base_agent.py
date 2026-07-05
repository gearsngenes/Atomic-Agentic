from __future__ import annotations

from typing import Any, Mapping
import warnings

import pytest
import asyncio

from atomic_agentic.tools import Tool
from atomic_agentic.agents.base import Agent
from atomic_agentic.agents.basic import BasicAgent
from atomic_agentic.exceptions import AgentError, AgentInvocationError
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.constants.agents import RUN_ID_PARAM, CONTEXT_PARAM
from atomic_agentic.engines.LLMEngines import LLMEngine
from atomic_agentic.models.agents.records import AgentRecord, LLMRecord
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results import LLMModelData, TokenUsage


ROLE_PROMPT = "You are a deterministic test writer."


# ─────────────────────────────────────────────────────────────────────────────
# Minimal concrete Agent fixture (for abstract-API tests)
# ─────────────────────────────────────────────────────────────────────────────
class _MinimalAgent(Agent):
    """Trivial concrete Agent used to test abstract Agent construction and API.

    ``_invoke`` performs a single engine call and produces the required
    tuple; tests that never call invoke() don't need this to run.
    """

    def _invoke(
        self,
        turns: list[AgentRecord],
        prompt: str,
        context: dict,
    ) -> tuple[AgentRecord, dict]:
        engine_result = self._llm_engine.invoke({
            "messages": [
                {"role": "system", "content": "minimal"},
                {"role": "user", "content": prompt},
            ]
        })
        llm_record = LLMRecord(
            messages=({"role": "user", "content": prompt},),
            llm_result=engine_result,
        )
        draft = AgentRecord(
            user_prompt=PromptConfig(template=prompt, description=""),
            generated_response=engine_result.result,
        )
        return draft, {
            "llm_records": (llm_record,),
            "llm_model_data": engine_result.model_data,
        }

    async def _ainvoke(
        self,
        turns: list[AgentRecord],
        prompt: str,
        context: dict,
    ) -> tuple[AgentRecord, dict]:
        engine_result = await self._llm_engine.async_invoke({
            "messages": [
                {"role": "system", "content": "minimal"},
                {"role": "user", "content": prompt},
            ]
        })
        llm_record = LLMRecord(
            messages=({"role": "user", "content": prompt},),
            llm_result=engine_result,
        )
        draft = AgentRecord(
            user_prompt=PromptConfig(template=prompt, description=""),
            generated_response=engine_result.result,
        )
        return draft, {
            "llm_records": (llm_record,),
            "llm_model_data": engine_result.model_data,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Test engine and helper functions
# ─────────────────────────────────────────────────────────────────────────────
class StatefulEchoLLMEngine(LLMEngine):
    """Concrete deterministic LLMEngine for Agent tests.

    Records every normalized message batch it receives and returns
    the latest user message with a stable prefix.
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
    """Stub mimicking AtomicResult `.result` access without LLMResult's str constraint."""

    def __init__(self, result: Any) -> None:
        self.result = result


class NonStringAsyncLLMEngine(StatefulEchoLLMEngine):
    async def async_invoke(self, inputs: Mapping[str, Any]) -> Any:
        return _NonStringEngineEnvelope(123)


def make_agent(
    *,
    engine: StatefulEchoLLMEngine | None = None,
    context_enabled: bool = False,
    records_window: int | None = None,
    pre_invoke: Any = build_prompt,
    post_invoke: Any = package_response,
    post_result_key: str | None = None,
    role_prompt: str = ROLE_PROMPT,
    response_preview_limit: int | None = None,
    assistant_response_source: str = "raw",
) -> BasicAgent:
    return BasicAgent(
        name="writer_agent",
        namespace="tests",
        description="Deterministic writer test agent.",
        llm_engine=engine or StatefulEchoLLMEngine(),
        role_prompt=role_prompt,
        context_enabled=context_enabled,
        records_window=records_window,
        pre_invoke=pre_invoke,
        post_invoke=post_invoke,
        post_result_key=post_result_key,
        response_preview_limit=response_preview_limit,
        assistant_response_source=assistant_response_source,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test classes
# ─────────────────────────────────────────────────────────────────────────────
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
        agent = BasicAgent(
            name="identity_agent",
            namespace="tests",
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

    def test_agent_schema_defaults_to_pre_invoke_parameters(self) -> None:
        agent = make_agent()

        assert [(param.name, param.kind, param.default) for param in agent.parameters] == [
            ("topic", "POSITIONAL_OR_KEYWORD", NO_VAL),
            ("tone", "POSITIONAL_OR_KEYWORD", "neutral"),
            ("run_id", "KEYWORD_ONLY", None),
        ]
        assert agent.return_type == "dict[str, Any]"


class TestAgentSchemaComposition:
    def test_post_only_non_result_param_is_auto_grafted_as_keyword_only(self) -> None:
        agent = make_agent(post_invoke=post_with_suffix)

        assert [(param.name, param.kind, param.default) for param in agent.parameters] == [
            ("topic", "POSITIONAL_OR_KEYWORD", NO_VAL),
            ("tone", "POSITIONAL_OR_KEYWORD", "neutral"),
            ("suffix", "KEYWORD_ONLY", NO_VAL),
            ("run_id", "KEYWORD_ONLY", None),
        ]

    def test_auto_grafted_param_preserves_post_default(self) -> None:
        agent = make_agent(post_invoke=post_one_required_plus_default)

        suffix_param = next(param for param in agent.parameters if param.name == "suffix")

        assert suffix_param.kind == "KEYWORD_ONLY"
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

    def test_variadic_post_param_is_not_grafted_into_schema(self) -> None:
        agent = make_agent(post_invoke=post_with_post_only_args)

        names = [p.name for p in agent.parameters]

        assert "extras" not in names

    def test_post_only_param_with_type_mismatch_still_routes_correctly(self) -> None:
        agent = make_agent(post_invoke=post_with_type_mismatch)

        result = agent.invoke({"topic": "pytest", "tone": "strict"})

        assert result.result == "ECHO: Write about pytest in a strict tone.|tone=strict"


class TestAgentPostInvokeRouting:
    def test_auto_grafted_required_param_reached_by_caller(self) -> None:
        agent = make_agent(post_invoke=post_with_suffix)

        result = agent.invoke({"topic": "pytest", "tone": "strict", "suffix": "!"})

        assert result.result == "ECHO: Write about pytest in a strict tone.!"

    def test_missing_required_auto_grafted_param_fails_at_invoke_time(self) -> None:
        agent = make_agent(post_invoke=post_with_suffix)

        with pytest.raises(
            AgentInvocationError,
            match="post_invoke Tool failed:.*missing required",
        ):
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

    def test_post_result_key_context_raises(self) -> None:
        def post_with_context_key(context: dict) -> dict:
            return context

        with pytest.raises(AgentError, match="framework-reserved"):
            make_agent(post_invoke=post_with_context_key, post_result_key="context")

    def test_post_result_key_run_id_raises(self) -> None:
        def post_with_run_id_key(run_id: str, result: str) -> str:
            return result

        with pytest.raises(AgentError, match="framework-reserved"):
            make_agent(post_invoke=post_with_run_id_key, post_result_key="run_id")

    def test_auto_resolved_result_key_reserved_raises(self) -> None:
        def post_context_first(context: dict) -> dict:
            return context

        with pytest.raises(AgentError, match="framework-reserved"):
            make_agent(post_invoke=post_context_first)


class TestAgentContext:
    def test_context_disabled_does_not_resend_history_but_still_stores_records(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, context_enabled=False)

        first = agent.invoke({"topic": "pytest", "tone": "strict"})
        second = agent.invoke({"topic": "agents", "tone": "concise"})

        assert first.result["final"] == "ECHO: Write about pytest in a strict tone."
        assert second.result["final"] == "ECHO: Write about agents in a concise tone."

        # Records are always appended regardless of context_enabled.
        assert len(agent.records) == 2

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

    def test_records_window_none_sends_all_prior_turns(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(
            engine=engine,
            context_enabled=True,
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
        engine = StatefulEchoLLMEngine()
        agent = make_agent(
            engine=engine,
            context_enabled=True,
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

    def test_records_window_zero_sends_no_prior_turns_but_still_stores_history(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(
            engine=engine,
            context_enabled=True,
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

        assert len(agent.records) == 2
        assert agent.records[0].user_prompt.template == "Write about first topic in a plain tone."
        assert agent.records[1].user_prompt.template == "Write about second topic in a plain tone."

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

        with pytest.raises(AgentInvocationError, match="pre_invoke returned non-string/non-PromptConfig"):
            agent.invoke({"topic": "pytest"})

    def test_post_invoke_with_one_defaulted_parameter_is_allowed(self) -> None:
        agent = make_agent(post_invoke=defaulted_post_invoke)

        result = agent.invoke({"topic": "pytest", "tone": "strict"})

        assert result.result == "ECHO: WRITE ABOUT PYTEST IN A STRICT TONE."

    def test_post_invoke_two_required_args_auto_grafts_second(self) -> None:
        agent = make_agent(post_invoke=bad_post_two_required_args)

        result = agent.invoke({"topic": "pytest", "tone": "strict", "suffix": "!"})

        assert result.result == "ECHO: Write about pytest in a strict tone.!"

    def test_context_enabled_setter_rejects_non_bool(self) -> None:
        agent = make_agent()

        with pytest.raises(ValueError, match="context_enabled"):
            agent.context_enabled = "yes"  # type: ignore[assignment]

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
                llm_engine=StatefulEchoLLMEngine(),
            )

    def test_agent_namespace_explicit(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="my_team",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
        )
        assert agent.namespace == "my_team"

    def test_agent_namespace_in_to_dict(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="my_team",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
        )
        assert agent.to_dict()["namespace"] == "my_team"


class TestAgentSerialization:
    def test_to_dict_includes_agent_configuration(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(
            engine=engine,
            context_enabled=True,
            records_window=1,
        )

        agent.invoke({"topic": "pytest", "tone": "strict"})

        data = agent.to_dict()

        assert data["type"] == "BasicAgent"
        assert data["name"] == "writer_agent"
        assert data["description"] == "Deterministic writer test agent."
        assert data["role_prompt"] == ROLE_PROMPT
        assert data["context_enabled"] is True
        assert data["records_window"] == 1
        assert data["records"] == [turn.to_dict() for turn in agent.records]
        assert "system_prompts" in data
        assert data["pre_invoke"]["name"] == "pre_invoke"
        assert data["post_invoke"]["name"] == "post_invoke"
        assert data["post_result_key"] == agent.post_result_key
        assert data["llm"]["type"] == "StatefulEchoLLMEngine"
        assert "secret" not in str(data)

    def test_to_dict_includes_system_prompts_key(self) -> None:
        agent = make_agent()

        data = agent.to_dict()

        assert "system_prompts" in data
        assert "role" in data["system_prompts"]

    def test_to_dict_excludes_passthrough_inputs(self) -> None:
        agent = make_agent()

        data = agent.to_dict()

        assert "passthrough_inputs" not in data


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

        with pytest.raises(AgentInvocationError, match="pre_invoke returned non-string/non-PromptConfig"):
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

    def test_assistant_response_source_is_frozen(self) -> None:
        agent = make_agent()

        with pytest.raises(AttributeError):
            agent.assistant_response_source = "final"  # type: ignore[misc]

    def test_response_preview_limit_construction_rejects_zero(self) -> None:
        with pytest.raises(AgentError, match="response_preview_limit"):
            make_agent(response_preview_limit=0)

    def test_response_preview_limit_construction_rejects_negative(self) -> None:
        with pytest.raises(AgentError, match="response_preview_limit"):
            make_agent(response_preview_limit=-1)

    def test_response_preview_limit_construction_rejects_non_int(self) -> None:
        with pytest.raises(AgentError, match="response_preview_limit"):
            make_agent(response_preview_limit="100")  # type: ignore[arg-type]

    def test_assistant_response_source_construction_rejects_bad_value(self) -> None:
        with pytest.raises(AgentError, match="assistant_response_source"):
            make_agent(assistant_response_source="both")  # type: ignore[arg-type]

    def test_assistant_response_source_construction_rejects_non_string(self) -> None:
        with pytest.raises(AgentError, match="assistant_response_source"):
            make_agent(assistant_response_source=1)  # type: ignore[arg-type]

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

        assert [param.name for param in agent.parameters] == ["subject", "style", "run_id"]
        assert agent.invoke({"subject": "pytest", "style": "direct"}).result == {
            "final": "ECHO: pytest:direct",
            "length": len("ECHO: pytest:direct"),
            "was_postprocessed": True,
        }


class TestAgentContextProperties:
    """Tests for context_properties, Graft D schema, and collision checks."""

    def test_normalize_context_properties_none_returns_empty_list(self) -> None:
        result = Agent._normalize_context_properties(None)
        assert result == []

    def test_normalize_context_properties_str_list_produces_keyword_only_paramspecs(self) -> None:
        result = Agent._normalize_context_properties(["alpha", "beta"])

        assert len(result) == 2
        assert result[0].name == "alpha"
        assert result[0].kind == ParamSpec.KEYWORD_ONLY
        assert result[0].default is NO_VAL
        assert result[1].name == "beta"

    def test_normalize_context_properties_paramspec_list_coerced_to_keyword_only(self) -> None:
        specs = [
            ParamSpec(name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="str", default=NO_VAL),
        ]
        result = Agent._normalize_context_properties(specs)

        assert result[0].kind == ParamSpec.KEYWORD_ONLY

    def test_normalize_context_properties_rejects_variadic_paramspec(self) -> None:
        specs = [
            ParamSpec(name="items", index=0, kind=ParamSpec.VAR_POSITIONAL, type="Any", default=NO_VAL),
        ]
        with pytest.raises(AgentError, match="variadic"):
            Agent._normalize_context_properties(specs)

    def test_normalize_context_properties_rejects_duplicate_names(self) -> None:
        with pytest.raises(AgentError, match="duplicate"):
            Agent._normalize_context_properties(["x", "x"])

    def test_normalize_context_properties_rejects_empty_string(self) -> None:
        with pytest.raises(AgentError, match="non-empty"):
            Agent._normalize_context_properties([""])

    def test_normalize_context_properties_accepts_reserved_framework_names(self) -> None:
        # "context" and "run_id" are reserved agent *parameters*, but context_properties
        # are keys inside the context dict — a separate namespace, so no conflict.
        result = Agent._normalize_context_properties(["context", "run_id"])
        assert [p.name for p in result] == ["context", "run_id"]

    def test_context_properties_grafts_single_context_dict_param(self) -> None:
        agent = _MinimalAgent(
            name="ctx_agent",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            context_properties=["persona", "style"],
        )

        names = [p.name for p in agent.parameters]
        kinds = {p.name: p.kind for p in agent.parameters}

        assert "context" in names
        assert kinds["context"] == "KEYWORD_ONLY"
        assert "persona" not in names
        assert "style" not in names

    def test_context_dict_param_appears_before_run_id(self) -> None:
        agent = _MinimalAgent(
            name="ctx_agent",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            context_properties=["persona"],
        )
        names = [p.name for p in agent.parameters]
        assert names.index("context") < names.index("run_id")

    def test_no_context_properties_does_not_graft_context_param(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
        )
        names = [p.name for p in agent.parameters]
        assert "context" not in names

    def test_required_context_property_missing_raises_agent_invocation_error(self) -> None:
        agent = _MinimalAgent(
            name="ctx_agent",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            context_properties=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default=NO_VAL),
            ],
        )
        with pytest.raises(AgentInvocationError, match="lang"):
            agent.invoke({"prompt": "hello"})

    def test_required_context_property_provided_does_not_raise(self) -> None:
        agent = _MinimalAgent(
            name="ctx_agent",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            context_properties=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default=NO_VAL),
            ],
        )
        result = agent.invoke({"prompt": "hello", "context": {"lang": "English"}})
        assert result is not None

    def test_context_stored_on_agent_record_after_invoke(self) -> None:
        agent = _MinimalAgent(
            name="ctx_agent",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            context_properties=["lang"],
        )
        agent.invoke({"prompt": "hello", "context": {"lang": "English"}})
        assert agent.records[0].context == {"lang": "English"}

    def test_warn_reserved_name_collisions_run_id_semantic_match_warns(self) -> None:
        matching = ParamSpec(
            name="run_id", index=0, kind=ParamSpec.KEYWORD_ONLY,
            type="str | None", default=None,
            description=RUN_ID_PARAM.description,
        )
        with pytest.warns(UserWarning, match="redundant"):
            Agent._warn_reserved_name_collisions(
                pre_params=[matching],
                post_params=[],
            )

    def test_warn_reserved_name_collisions_run_id_semantic_mismatch_raises(self) -> None:
        clashing = ParamSpec(
            name="run_id", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD,
            type="str", default=NO_VAL,
        )
        with pytest.raises(AgentError, match="conflicts"):
            Agent._warn_reserved_name_collisions(
                pre_params=[clashing],
                post_params=[],
            )

    def test_warn_reserved_name_collisions_context_no_description_raises(self) -> None:
        # context: dict with no description does not share the CONTEXT_PARAM description
        # root → description mismatch → raises, not warns.
        ctx_no_desc = ParamSpec(
            name="context", index=0, kind=ParamSpec.KEYWORD_ONLY,
            type="dict", default={},
        )
        with pytest.raises(AgentError, match="description mismatch"):
            Agent._warn_reserved_name_collisions(pre_params=[ctx_no_desc], post_params=[])

    def test_warn_reserved_name_collisions_context_matching_description_warns(self) -> None:
        ctx_desc_prefix = CONTEXT_PARAM.description.split("{")[0]
        matching = ParamSpec(
            name="context", index=0, kind=ParamSpec.KEYWORD_ONLY,
            type="dict", default={},
            description=ctx_desc_prefix + " Required: 'lang' (str)",
        )
        with pytest.warns(UserWarning, match="redundant"):
            Agent._warn_reserved_name_collisions(pre_params=[matching], post_params=[])

    def test_warn_reserved_name_collisions_context_incompatible_type_raises(self) -> None:
        bad_type = ParamSpec(
            name="context", index=0, kind=ParamSpec.KEYWORD_ONLY,
            type="str", default=NO_VAL,
        )
        with pytest.raises(AgentError, match="type mismatch"):
            Agent._warn_reserved_name_collisions(pre_params=[bad_type], post_params=[])

    def test_validate_pre_post_overlap_shapes_mismatch_raises(self) -> None:
        pre = [
            ParamSpec(name="items", index=0, kind=ParamSpec.VAR_POSITIONAL, type="Any", default=NO_VAL),
        ]
        post = [
            ParamSpec(name="items", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="Any", default=NO_VAL),
        ]
        with pytest.raises(AgentError, match="items"):
            Agent._validate_pre_post_overlap_shapes(pre, post)

    def test_update_prompt_registers_new_key(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
        )
        config = PromptConfig(template="New prompt.", description="custom")
        agent.update_prompt("custom", config)

        assert "custom" in agent.system_prompts
        assert agent.system_prompts["custom"].template == "New prompt."

    def test_update_prompt_rejects_empty_key(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
        )
        with pytest.raises(AgentError, match="key"):
            agent.update_prompt("  ", PromptConfig(template="x", description="d"))

    def test_update_prompt_rejects_non_prompt_config(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
        )
        with pytest.raises(AgentError, match="PromptConfig"):
            agent.update_prompt("key", "not a config")  # type: ignore[arg-type]

    def test_system_prompts_property_returns_shallow_copy(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
        )
        copy = agent.system_prompts
        copy["mutated"] = PromptConfig(template="x", description="d")

        assert "mutated" not in agent.system_prompts

    def test_context_disabled_records_always_appended(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=engine,
            context_enabled=False,
        )

        agent.invoke({"prompt": "hello"})
        agent.invoke({"prompt": "world"})

        assert len(agent.records) == 2

    def test_context_disabled_turns_always_empty(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=engine,
            context_enabled=True,
        )
        agent.invoke({"prompt": "first"})

        agent.context_enabled = False
        agent.invoke({"prompt": "second"})

        # Second call's messages should have no prior-turn content
        second_call = engine.calls[1]
        assert len(second_call) == 2  # just system + user from _MinimalAgent


class TestAgentCollisionCheckerCompleteness:
    """Tests for _warn_reserved_name_collisions evaluating every (param, source) pair."""

    def test_pre_compatible_post_incompatible_context_raises(self) -> None:
        # pre has context with matching description (compatible → warns); post has context: str
        # (type mismatch → raises). Verifies post is evaluated independently even after pre warned.
        ctx_desc_prefix = CONTEXT_PARAM.description.split("{")[0]
        pre_ctx = ParamSpec(
            name="context", index=0, kind=ParamSpec.KEYWORD_ONLY,
            type="dict", default={},
            description=ctx_desc_prefix,
        )
        post_ctx_bad = ParamSpec(
            name="context", index=0, kind=ParamSpec.KEYWORD_ONLY,
            type="str", default=NO_VAL,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with pytest.raises(AgentError, match="type mismatch"):
                Agent._warn_reserved_name_collisions(
                    pre_params=[pre_ctx],
                    post_params=[post_ctx_bad],
                )
        # pre_invoke should have produced the redundant warning before post raised
        assert any("redundant" in str(warning.message) for warning in w)

    def test_both_compatible_context_warns_twice(self) -> None:
        ctx_desc_prefix = CONTEXT_PARAM.description.split("{")[0]
        pre_ctx = ParamSpec(
            name="context", index=0, kind=ParamSpec.KEYWORD_ONLY,
            type="dict", default={},
            description=ctx_desc_prefix,
        )
        post_ctx = ParamSpec(
            name="context", index=0, kind=ParamSpec.KEYWORD_ONLY,
            type="dict", default={},
            description=ctx_desc_prefix + " extra detail",
        )
        with pytest.warns(UserWarning, match="redundant") as w:
            Agent._warn_reserved_name_collisions(
                pre_params=[pre_ctx],
                post_params=[post_ctx],
            )
        assert len(w) == 2


class TestAgentContextIsolation:
    """Tests that context is always a fresh dict per invocation (not shared mutable default)."""

    def test_omitted_context_does_not_share_state_across_invocations(self) -> None:
        agent = _MinimalAgent(
            name="ctx_agent",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            context_properties=["lang"],
        )
        # First call omits context (required property — must be provided):
        # so use an optional property instead
        agent2 = _MinimalAgent(
            name="ctx_agent2",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            context_properties=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY,
                          type="str", default="English"),
            ],
        )
        agent2.invoke({"prompt": "hello"})
        agent2.invoke({"prompt": "world"})
        assert agent2.records[0].context is not agent2.records[1].context


class TestAgentRenderTurnGuards:
    """Tests for render_turn defensive checks."""

    def test_render_turn_final_source_on_draft_record_raises(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(
            engine=engine,
            assistant_response_source="final",
        )
        draft = AgentRecord(
            user_prompt=PromptConfig(template="hello", description=""),
            generated_response="raw text",
            final_result=None,
        )
        with pytest.raises(AgentInvocationError, match="final_result is None"):
            agent.render_turn(draft)

    def test_render_turn_replays_context_into_template(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(engine=engine, context_enabled=True)

        from atomic_agentic.models.agents.prompts import PromptConfig as PC
        from dataclasses import replace as dc_replace
        from atomic_agentic.models.results.agents import AgentResult
        from datetime import datetime, timezone, timedelta
        from atomic_agentic.models.results import LLMModelData, TokenUsage

        # Craft a completed record with a template that has a placeholder
        cfg = PC(template="Task for {user}: do something.", description="")
        result = AgentResult(
            result="done",
            invoker_id="test-agent",
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc) + timedelta(seconds=1),
            llm_token_usage=(TokenUsage(input_tokens=1, generated_tokens=1, total_tokens=2),),
            llm_model_data=LLMModelData(provider="test"),
        )
        from atomic_agentic.models.agents.records import LLMRecord
        from atomic_agentic.models.results import LLMResult
        llm_result = LLMResult(
            result="done",
            invoker_id="engine-1",
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc) + timedelta(seconds=1),
            token_usage=TokenUsage(input_tokens=1, generated_tokens=1, total_tokens=2),
            model_data=LLMModelData(provider="test"),
        )
        record = AgentRecord(
            user_prompt=cfg,
            generated_response="raw",
            context={"user": "Alice"},
            final_result=result,
            llm_records=(LLMRecord(
                messages=({"role": "user", "content": "Task for Alice: do something."},),
                llm_result=llm_result,
            ),),
        )
        rendered = agent.render_turn(record)
        assert rendered[0]["content"] == "Task for Alice: do something."


class TestAgentAsyncContextProperties:
    """Tests for context_properties through async_invoke."""

    def test_async_required_context_property_missing_raises(self) -> None:
        agent = _MinimalAgent(
            name="ctx_agent",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            context_properties=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY,
                          type="str", default=NO_VAL),
            ],
        )
        with pytest.raises(AgentInvocationError, match="lang"):
            asyncio.run(agent.async_invoke({"prompt": "hello"}))

    def test_async_context_stored_on_record(self) -> None:
        agent = _MinimalAgent(
            name="ctx_agent",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            context_properties=["lang"],
        )
        asyncio.run(agent.async_invoke({"prompt": "hello", "context": {"lang": "English"}}))
        assert agent.records[0].context == {"lang": "English"}


class TestAgentDescriptionOverrideRemoval:
    def test_agent_description_follows_base_getter(self) -> None:
        agent = make_agent()

        assert agent.description.startswith("Deterministic writer test agent.")
        assert "- run_id:" in agent.description

    def test_agent_description_contains_run_id_description_text(self) -> None:
        agent = make_agent()

        assert "Optional UUID hexstring" in agent.description

    def test_agent_to_dict_description_is_raw(self) -> None:
        agent = make_agent()

        assert agent.to_dict()["description"] == "Deterministic writer test agent."
