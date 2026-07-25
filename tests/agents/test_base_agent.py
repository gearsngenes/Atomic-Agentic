from __future__ import annotations

from typing import Any, Mapping
import warnings

import pytest
import asyncio

from atomic_agentic.tools import Tool
from atomic_agentic.agents.base import Agent
from atomic_agentic.exceptions import AgentError, AgentInvocationError
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.constants.agents import RUN_ID_PARAM
from atomic_agentic.llm import LLMEngine
from atomic_agentic.models.agents.records import AgentRecord, LLMRecord
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.models.agents.tasks import AgentTask
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results import LLMModelData, TokenUsage
from atomic_agentic.utils.parameters import to_paramspec_list


ROLE_PROMPT = "You are a deterministic test writer."


# ─────────────────────────────────────────────────────────────────────────────
# Minimal concrete Agent fixture (for abstract-API tests)
# ─────────────────────────────────────────────────────────────────────────────
class _MinimalAgent(Agent):
    """Trivial concrete Agent used to test abstract Agent construction and API.

    ``_progress`` performs a single engine call and completes the task;
    tests that never call invoke() don't need this to run. No
    ``_initialize_task`` override -- Agent's concrete base implementation
    (bare AgentTask, ``system_prompt_name=None``) is sufficient.
    """

    def render_task(
        self,
        task: AgentTask,
        *,
        additional_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Minimal implementation satisfying the abstract contract -- not
        exercised by ``_progress``/``_async_progress``, which build their
        own inline message list."""
        system = self._render_system_message(task, {})
        historic = self._render_historic_messages(task)
        if not task.task_messages:
            task.task_messages = [{"role": "user", "content": task.user_prompt}]
        task.task_messages.extend(additional_messages or [])
        return system + historic + task.task_messages

    def _progress(self, task: AgentTask) -> AgentTask:
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

    async def _async_progress(self, task: AgentTask) -> AgentTask:
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
    single-LLM-call ``_progress`` path. Used in place of ``BasicAgent`` for
    invoke-lifecycle tests in this file for fixture independence.
    """

    def __init__(self, *, system_prompt: str = ROLE_PROMPT, **kwargs: Any) -> None:
        self._echo_system_prompt = system_prompt
        super().__init__(**kwargs)
        self._system_prompts["system"] = PromptConfig(
            template=system_prompt, description="Echo agent system prompt."
        )

    def render_task(
        self,
        task: AgentTask,
        *,
        additional_messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Minimal implementation satisfying the abstract contract -- not
        exercised by ``_progress``/``_async_progress``, which call
        ``build_messages`` directly with ``self._echo_system_prompt``."""
        system = self._render_system_message(task, {})
        historic = self._render_historic_messages(task)
        if not task.task_messages:
            task.task_messages = [{"role": "user", "content": task.user_prompt}]
        task.task_messages.extend(additional_messages or [])
        return system + historic + task.task_messages

    def _progress(self, task: AgentTask) -> AgentTask:
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

    async def _async_progress(self, task: AgentTask) -> AgentTask:
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
    adds its own reserved name on top of the base ``[RUN_ID_PARAM]`` list.
    """

    EXTRA_RESERVED = ParamSpec(
        name="trace_id", index=0, kind=ParamSpec.KEYWORD_ONLY,
        type="str | None", default=None,
        description="Subclass-reserved trace identifier.",
    )

    @classmethod
    def get_reserved_parameters(cls) -> list[ParamSpec]:
        return super().get_reserved_parameters() + [cls.EXTRA_RESERVED]


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
        return TokenUsage(
            input_tokens=1, generated_tokens=1, total_tokens=2, response_tokens=1
        )

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        return False

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
    system_prompt: str = ROLE_PROMPT,
    response_preview_limit: int | None = None,
    assistant_response_source: str = "raw",
) -> _EchoAgent:
    return _EchoAgent(
        name="writer_agent",
        namespace="tests",
        description="Deterministic writer test agent.",
        llm_engine=engine or StatefulEchoLLMEngine(),
        system_prompt=system_prompt,
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
        agent = _EchoAgent(
            name="identity_agent",
            namespace="tests",
            description="Identity agent.",
            llm_engine=engine,
            system_prompt=ROLE_PROMPT,
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
            ("run_id", "KEYWORD_ONLY", None),
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
        # KEYWORD_ONLY) like any other kind.
        agent = make_agent(post_invoke=post_with_post_only_args)

        names = [p.name for p in agent.parameters]
        assert names == ["topic", "tone", "extras", "run_id"]
        extras_param = next(p for p in agent.parameters if p.name == "extras")
        assert extras_param.kind == "VAR_POSITIONAL"

    def test_overlapping_param_with_type_mismatch_raises_at_construction(self) -> None:
        # "tone" overlaps pre_invoke's own "tone: str" param but is declared
        # "int" here -- an incompatible collision, caught at construction by
        # parameter_collisions (was previously undetected: pre/post overlap
        # shapes were only checked for variadic-kind mismatches).
        with pytest.raises(AgentError, match="collision"):
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

    def test_post_result_key_run_id_raises(self) -> None:
        # "run_id" is never a legal post_result_key: it's not among
        # post_invoke's own declared (reserved-popped) names.
        with pytest.raises(AgentError, match="post_result_key"):
            make_agent(post_result_key="run_id")

    def test_post_result_key_colliding_with_pre_invoke_name_raises(self) -> None:
        def post_with_topic_key(topic: str) -> str:
            return topic

        with pytest.raises(AgentError, match="post_result_key"):
            make_agent(post_invoke=post_with_topic_key, post_result_key="topic")


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

        assert data["type"] == "_EchoAgent"
        assert data["name"] == "writer_agent"
        assert data["description"] == "Deterministic writer test agent."
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
        assert "system" in data["system_prompts"]

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


class TestGetReservedParameters:
    """Coverage for Agent.get_reserved_parameters() -- default and override."""

    def test_base_agent_default_is_run_id_only(self) -> None:
        assert Agent.get_reserved_parameters.__func__(Agent) == [RUN_ID_PARAM]

    def test_minimal_agent_default_is_run_id_only(self) -> None:
        assert _MinimalAgent.get_reserved_parameters() == [RUN_ID_PARAM]

    def test_subclass_override_extends_base_reserved_list(self) -> None:
        reserved = _ExtraReservedAgent.get_reserved_parameters()
        assert reserved == [RUN_ID_PARAM, _ExtraReservedAgent.EXTRA_RESERVED]

    def test_subclass_reserved_param_is_grafted_into_schema(self) -> None:
        agent = _ExtraReservedAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
        )
        names = [p.name for p in agent.parameters]
        assert "run_id" in names
        assert "trace_id" in names


class TestExtraParametersNormalization:
    """Coverage for extra_parameters normalization via to_paramspec_list."""

    def test_extra_parameters_none_produces_no_extra_schema_entries(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
        )
        # "prompt" comes from the default identity_pre_tool; no extra_parameters
        # means nothing else is grafted beyond the reserved run_id.
        assert [p.name for p in agent.parameters] == ["prompt", "run_id"]

    def test_extra_parameters_str_list_preserves_declared_kind(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
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
            llm_engine=StatefulEchoLLMEngine(),
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
                llm_engine=StatefulEchoLLMEngine(),
                extra_parameters=specs,
            )

    def test_extra_parameters_values_reach_invoke_via_inputs(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            extra_parameters=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="English"),
            ],
        )
        # _MinimalAgent._invoke ignores inputs entirely, but invoke() must
        # accept the extra top-level key without complaint (no more "context"
        # dict wrapper -- extra_parameters values are flat top-level keys now).
        result = agent.invoke({"prompt": "hi", "lang": "French"})
        assert result is not None


class TestReservedNameReconciliation:
    """Per-source reserved-name reconciliation: identical / compatible / incompatible."""

    def test_identical_reserved_param_in_extra_parameters_warns_and_is_popped(self) -> None:
        # RUN_ID_PARAM itself, declared verbatim -- guaranteed field-for-field
        # identical (name/type/kind/default/description all match).
        with pytest.warns(UserWarning, match="redundant"):
            agent = _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=StatefulEchoLLMEngine(),
                extra_parameters=[RUN_ID_PARAM],
            )
        # Only one run_id entry survives in the final schema (the framework's).
        assert [p.name for p in agent.parameters].count("run_id") == 1

    def test_pre_invoke_compatible_run_id_warns_distinct_message(self) -> None:
        # type="Any" is compatible with RUN_ID_PARAM's "str | None" (either
        # side being "Any" satisfies semantically_compatible); default/
        # description differ, so it's compatible but not identical.
        def pre_with_compatible_run_id(prompt: str, *, run_id: Any = "unset") -> str:
            return prompt

        with pytest.warns(UserWarning, match="not identical"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=StatefulEchoLLMEngine(),
                pre_invoke=pre_with_compatible_run_id,
            )

    def test_pre_invoke_incompatible_run_id_raises(self) -> None:
        def pre_with_bad_run_id(prompt: str, run_id: int) -> str:
            return prompt

        with pytest.raises(AgentError, match="conflicts"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=StatefulEchoLLMEngine(),
                pre_invoke=pre_with_bad_run_id,
            )

    def test_post_invoke_incompatible_run_id_raises_independently_of_clean_pre(self) -> None:
        # A clean, warning-free pre_invoke must not mask an incompatible
        # reserved-name collision detected independently in post_invoke --
        # each of the three sources (pre/post/extra) is reconciled on its own.
        def post_with_bad_run_id(result: str, run_id: int) -> str:
            return result

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(AgentError, match="conflicts"):
                _MinimalAgent(
                    name="a",
                    namespace="tests",
                    description="d",
                    llm_engine=StatefulEchoLLMEngine(),
                    pre_invoke=build_prompt,
                    post_invoke=post_with_bad_run_id,
                )

    def test_extra_parameters_incompatible_run_id_raises(self) -> None:
        bad = ParamSpec(name="run_id", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int", default=NO_VAL)
        with pytest.raises(AgentError, match="conflicts"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=StatefulEchoLLMEngine(),
                extra_parameters=[bad],
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

    def test_collision_with_pre_invoke_name_raises(self) -> None:
        with pytest.raises(AgentError, match="post_result_key"):
            make_agent(
                post_invoke=lambda topic: topic,
                post_result_key="topic",
            )

    def test_collision_with_extra_parameters_name_raises(self) -> None:
        def post_with_flag(result: str, flag: str = "x") -> str:
            return result

        with pytest.raises(AgentError, match="post_result_key"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=StatefulEchoLLMEngine(),
                post_invoke=post_with_flag,
                post_result_key="flag",
                extra_parameters=["flag"],
            )


class TestPreVsPostReconciliation:
    def test_incompatible_overlap_raises(self) -> None:
        def post_with_int_topic(result: str, topic: int) -> str:
            return result

        with pytest.raises(AgentError, match="collision"):
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
        assert names == ["topic", "suffix", "tone", "run_id"]


class TestCombinedVsExtraReconciliation:
    def test_incompatible_overlap_raises(self) -> None:
        with pytest.raises(AgentError, match="collision"):
            _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=StatefulEchoLLMEngine(),
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
                llm_engine=StatefulEchoLLMEngine(),
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
                llm_engine=StatefulEchoLLMEngine(),
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
            llm_engine=StatefulEchoLLMEngine(),
            pre_invoke=pre_with_two_fields,
            extra_parameters=[
                ParamSpec(name="flag", index=0, kind=ParamSpec.KEYWORD_ONLY, type="bool", default=False),
            ],
        )
        names = [p.name for p in agent.parameters]
        assert names == ["subject", "style", "flag", "run_id"]


class TestFinalReservedGraftOrdering:
    def test_reserved_graft_valid_ordering_with_positional_only_pre(self) -> None:
        def pre_positional_only(subject: str, /) -> str:
            return subject

        agent = make_agent(pre_invoke=pre_positional_only)
        names = [p.name for p in agent.parameters]
        assert names == ["subject", "run_id"]
        assert agent.parameters[0].kind == "POSITIONAL_ONLY"
        assert agent.parameters[-1].kind == "KEYWORD_ONLY"

    def test_reserved_graft_after_extra_keyword_only_params(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            extra_parameters=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="English"),
            ],
        )
        names = [p.name for p in agent.parameters]
        # "prompt" (default identity_pre_tool, KEYWORD_ONLY) and "lang" (extra,
        # KEYWORD_ONLY) both sort at the same category tier; combined-vs-extra
        # reconciliation appends extra's remainder after pre's, and the final
        # reserved graft appends run_id last.
        assert names == ["prompt", "lang", "run_id"]


class TestUpdatePrompt:
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


class TestAgentRecordsAlwaysAppended:
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


class TestAgentRenderTurnGuards:
    """Tests for render_turn defensive checks."""

    def test_render_turn_final_source_on_draft_record_raises(self) -> None:
        engine = StatefulEchoLLMEngine()
        agent = make_agent(
            engine=engine,
            assistant_response_source="final",
        )
        draft = AgentRecord(
            user_prompt="hello",
            generated_response="raw text",
            final_result=None,
        )
        with pytest.raises(AgentInvocationError, match="final_result is None"):
            agent.render_turn(draft)


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


class TestInvocationLifecycle:
    """Non-destructive run_id read, reserved-name exclusion from pre/post
    slices, full inputs threading to _invoke, and AgentRecord.inputs widening."""

    def test_run_id_read_does_not_mutate_caller_inputs_dict(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
        )
        caller_inputs = {"prompt": "hello", "run_id": None}
        agent.invoke(caller_inputs)
        # The caller's own dict object is untouched (filter_inputs works on
        # a copy internally; this just confirms invoke never mutates in place).
        assert caller_inputs == {"prompt": "hello", "run_id": None}

    def test_invoke_receives_full_unsliced_inputs_in_invoke_arg(self) -> None:
        captured: dict[str, Any] = {}

        class CapturingAgent(_MinimalAgent):
            def _initialize_task(self, *, turns, prompt, inputs):  # type: ignore[override]
                captured.update(inputs)
                return super()._initialize_task(turns=turns, prompt=prompt, inputs=inputs)

        agent = CapturingAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            extra_parameters=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="English"),
            ],
        )
        agent.invoke({"prompt": "hi", "lang": "French"})

        assert captured == {"prompt": "hi", "lang": "French", "run_id": None}

    def test_reserved_name_excluded_from_pre_invoke_slice_even_if_declared_compatible(self) -> None:
        """A tool declaring a compatible same-named reserved param (warned, not
        raised, at construction) must not have the framework's reserved value
        silently slotted into it -- the exclusion applies at every invoke()."""
        seen: dict[str, Any] = {}

        def pre_with_compatible_run_id(prompt: str, *, run_id: Any = "unset") -> str:
            seen["run_id_seen_by_pre"] = run_id
            return prompt

        with pytest.warns(UserWarning):
            agent = _MinimalAgent(
                name="a",
                namespace="tests",
                description="d",
                llm_engine=StatefulEchoLLMEngine(),
                pre_invoke=pre_with_compatible_run_id,
            )
        agent.invoke({"prompt": "hi"})
        # pre_invoke's own declared default is used -- the framework's run_id
        # value is never routed into it.
        assert seen["run_id_seen_by_pre"] == "unset"

    def test_committed_agent_record_inputs_equals_full_filtered_inputs(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            extra_parameters=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="English"),
            ],
        )
        agent.invoke({"prompt": "hi", "lang": "French"})

        assert agent.records[0].inputs == {"prompt": "hi", "lang": "French", "run_id": None}

    def test_async_committed_agent_record_inputs_equals_full_filtered_inputs(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            extra_parameters=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="English"),
            ],
        )
        asyncio.run(agent.async_invoke({"prompt": "hi", "lang": "French"}))

        assert agent.records[0].inputs == {"prompt": "hi", "lang": "French", "run_id": None}

    def test_inputs_isolated_across_invocations(self) -> None:
        agent = _MinimalAgent(
            name="a",
            namespace="tests",
            description="d",
            llm_engine=StatefulEchoLLMEngine(),
            extra_parameters=[
                ParamSpec(name="lang", index=0, kind=ParamSpec.KEYWORD_ONLY, type="str", default="English"),
            ],
        )
        agent.invoke({"prompt": "hi"})
        agent.invoke({"prompt": "yo"})
        assert agent.records[0].inputs is not agent.records[1].inputs
