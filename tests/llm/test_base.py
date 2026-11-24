from __future__ import annotations

from typing import Any

import pytest

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import LLMEngine
from atomic_agentic.models.results import LLMModelData, LLMResult, TokenUsage
from atomic_agentic.utils.core import run_coro_sync
from ..fake_engines import FakeLLMEngine


class TestLLMEngineConstruction:
    def test_default_construction_uses_class_name_and_base_metadata(self) -> None:
        engine = FakeLLMEngine()

        assert engine.name == "FakeLLMEngine"
        assert engine.to_dict()["description"] == "LLM Engine"
        assert engine.description == (
            "LLM Engine\n\n"
            "- messages: A non-empty list of chat-message mappings. Each "
            "mapping must have a string 'role' key (one of 'system', "
            "'user', 'assistant') and a string 'content' key (free-form "
            "text).\n"
            "- output_structure: Optional JSON-Schema-shaped mapping "
            "requesting provider-native, schema-constrained structured "
            "output for this call. None (default) requests plain text."
        )
        assert engine.return_type == ("None", "bool", "dict[str, Any]", "float", "int", "list[Any]", "str")
        assert [(param.name, param.kind, param.type) for param in engine.parameters] == [
            ("messages", "POSITIONAL_OR_KEYWORD", ("list[dict[str, str]]",)),
            ("output_structure", "KEYWORD_ONLY", ("None", "dict[str, Any]")),
        ]

    def test_custom_construction_values_are_stored(self) -> None:
        engine = FakeLLMEngine(
            name="fake_engine",
            description="Fake test engine.",
        )

        data = engine.to_dict()

        assert engine.name == "fake_engine"
        assert data["description"] == "Fake test engine."
        assert engine.description == (
            "Fake test engine.\n\n"
            "- messages: A non-empty list of chat-message mappings. Each "
            "mapping must have a string 'role' key (one of 'system', "
            "'user', 'assistant') and a string 'content' key (free-form "
            "text).\n"
            "- output_structure: Optional JSON-Schema-shaped mapping "
            "requesting provider-native, schema-constrained structured "
            "output for this call. None (default) requests plain text."
        )
        assert data["runtime_parameters"] == {}
        for removed in (
            "timeout_seconds", "max_retries", "retry_backoff_base",
            "retry_backoff_max", "attachments",
        ):
            assert removed not in data

    def test_invalid_atomic_invokable_name_still_raises(self) -> None:
        with pytest.raises(ValueError):
            FakeLLMEngine(name="bad-name")


class TestLLMEngineMessagesAndInvoke:
    def test_invoke_filters_inputs_and_returns_stripped_text(self) -> None:
        engine = FakeLLMEngine()

        result = engine.invoke(
            {
                "messages": [{"role": "USER", "content": "Hello"}],
                "unused": "ignored",
            }
        )

        assert isinstance(result, LLMResult)
        assert result.result == "hello"
        assert engine.payloads == [
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "output_structure": None,
            }
        ]

    def test_invoke_result_carries_the_engine_full_name_as_invoker_id(self) -> None:
        engine = FakeLLMEngine(responses=["one", "two"])
        messages = {"messages": [{"role": "user", "content": "Hello"}]}

        assert engine.invoke(messages).invoker_id == engine.full_name
        assert run_coro_sync(engine.async_invoke(messages)).invoker_id == engine.full_name

    def test_invoke_is_inherited_and_make_result_is_gone(self) -> None:
        assert "invoke" not in LLMEngine.__dict__
        assert "async_invoke" not in LLMEngine.__dict__
        assert not hasattr(LLMEngine, "make_result")
        assert LLMEngine.result_cls is LLMResult

    def test_async_invoke_rejects_non_list_messages(self) -> None:
        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match=r"^FakeLLMEngine\.invoke: .*must be a list"):
            run_coro_sync(engine.async_invoke({"messages": "hello"}))

    def test_invoke_rejects_missing_messages(self) -> None:
        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match="messages"):
            engine.invoke({})

    def test_invoke_rejects_non_list_messages(self) -> None:
        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match="messages"):
            engine.invoke({"messages": "hello"})

    def test_invoke_rejects_empty_messages_list(self) -> None:
        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match="must not be empty"):
            engine.invoke({"messages": []})

    def test_normalize_messages_rejects_non_mapping_message(self) -> None:
        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match="not a mapping"):
            engine.invoke({"messages": ["bad"]})  # type: ignore[list-item]

    @pytest.mark.parametrize(
        "message",
        [
            {"role": 123, "content": "hello"},
            {"role": "user", "content": 123},
            {"role": "user"},
            {"content": "hello"},
        ],
    )
    def test_normalize_messages_requires_string_role_and_content(
        self,
        message: dict[str, Any],
    ) -> None:
        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match="role.*content"):
            engine.invoke({"messages": [message]})  # type: ignore[list-item]

    def test_extract_result_rejects_unsupported_type(self) -> None:
        # int/float/bool/None are all valid JSON-decodable result types now
        # (widened alongside response-schema-basic-agent's addendum) -- a
        # set is genuinely still outside the allowed closure.
        engine = FakeLLMEngine(responses=[{1, 2, 3}])

        with pytest.raises(LLMEngineError, match="must return str"):
            engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})

    def test_unexpected_provider_error_is_wrapped_by_invoke(self) -> None:
        engine = FakeLLMEngine(responses=[ValueError("provider failed")])

        with pytest.raises(LLMEngineError, match=r"^FakeLLMEngine\.invoke: provider failed$"):
            engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})

    def test_error_prefix_is_uniform_across_sync_and_async(self) -> None:
        messages = {"messages": [{"role": "assistant", "content": "x"}]}

        with pytest.raises(LLMEngineError, match=r"^FakeLLMEngine\.invoke: "):
            FakeLLMEngine().invoke(messages)
        with pytest.raises(LLMEngineError, match=r"^FakeLLMEngine\.invoke: "):
            run_coro_sync(FakeLLMEngine().async_invoke(messages))

    @pytest.mark.parametrize("role", ["tool", "developer", "bogus"])
    def test_normalize_messages_rejects_unsupported_role(self, role: str) -> None:
        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match=f"unsupported role '{role}'"):
            engine.invoke({"messages": [
                {"role": "user", "content": "hi"},
                {"role": role, "content": "x"},
            ]})

    def test_normalize_messages_lowercases_roles_and_accepts_the_three(self) -> None:
        engine = FakeLLMEngine()

        engine.invoke({"messages": [
            {"role": "SYSTEM", "content": "s"},
            {"role": "User", "content": "u"},
            {"role": "ASSISTANT", "content": "a"},
        ]})

        assert [m["role"] for m in engine.calls[0]] == ["system", "user", "assistant"]

    def test_normalize_messages_requires_a_user_message(self) -> None:
        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match="at least one user message"):
            engine.invoke({"messages": [
                {"role": "system", "content": "s"},
                {"role": "assistant", "content": "a"},
            ]})
        assert engine.call_count == 0

    def test_system_message_is_optional(self) -> None:
        engine = FakeLLMEngine()

        engine.invoke({"messages": [{"role": "user", "content": "u"}]})

        assert engine.call_count == 1


class TestLLMEngineNoRetry:
    """The base owns no retry loop: a provider failure is raised on the first call."""

    @pytest.mark.parametrize("error", [TimeoutError("temporary"), ConnectionError("temporary")])
    def test_transient_error_is_not_retried(self, error: Exception) -> None:
        engine = FakeLLMEngine(responses=[error, " recovered "])

        with pytest.raises(LLMEngineError, match="FakeLLMEngine.invoke: temporary"):
            engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})

        assert engine.call_count == 1

    @pytest.mark.parametrize("error", [TimeoutError("temporary"), ConnectionError("temporary")])
    def test_transient_error_is_not_retried_async(self, error: Exception) -> None:
        engine = FakeLLMEngine(responses=[error, " recovered "])

        with pytest.raises(LLMEngineError, match="FakeLLMEngine.invoke: temporary"):
            run_coro_sync(engine.async_invoke({"messages": [{"role": "user", "content": "Hello"}]}))

        assert engine.call_count == 1

    def test_llm_engine_error_passes_through_unwrapped(self) -> None:
        engine = FakeLLMEngine(responses=[LLMEngineError("normalized")])

        with pytest.raises(LLMEngineError, match="^normalized$"):
            engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})

        assert engine.call_count == 1

    def test_retry_machinery_is_gone(self) -> None:
        for name in ("_call_with_retries", "_call_with_retries_async", "_should_retry"):
            assert not hasattr(LLMEngine, name)
        engine = FakeLLMEngine()
        for name in ("_max_retries", "_retry_backoff_base", "_retry_backoff_max"):
            assert not hasattr(engine, name)

    def test_retry_constructor_arguments_are_gone(self) -> None:
        with pytest.raises(TypeError):
            FakeLLMEngine(max_retries=1)


class TestLLMEngineHasNoAttachmentApi:
    def test_base_has_no_attachment_surface(self) -> None:
        engine = FakeLLMEngine()

        for name in (
            "attach",
            "detach",
            "clear_attachments",
            "attachments",
            "_attachments",
            "_prepare_attachment",
            "_on_detach",
            "_validate_attachment_path",
            "illegal_attachment_exts",
            "allowed_attachment_exts",
        ):
            assert not hasattr(engine, name)


class TestLLMEngineAbstractHooks:
    def test_missing_abstract_hook_raises_type_error(self) -> None:
        class IncompleteEngine(LLMEngine):
            def _build_provider_payload(self, messages, output_structure=None):
                return {}

            def _call_provider(self, payload):
                return {"text": ""}

            def _extract_result(self, response, requested_structured):
                return ""

            def _extract_token_usage(self, response):
                return TokenUsage(
                    input_tokens=0, generated_tokens=0, total_tokens=0, response_tokens=0
                )

        with pytest.raises(TypeError):
            IncompleteEngine()


class TestLLMEngineOutputStructure:
    """Coverage for the output_structure/requested_structured contract."""

    def test_output_structure_none_threads_through_as_none(self) -> None:
        engine = FakeLLMEngine()

        engine.invoke({"messages": [{"role": "user", "content": "hi"}]})

        assert engine.payloads[-1]["output_structure"] is None
        assert engine.requested_structured_calls == [False]

    def test_output_structure_dict_threads_through_unmodified(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        engine = FakeLLMEngine(responses=['{"a": "x"}'])

        engine.invoke(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "output_structure": schema,
            }
        )

        # Identity, not just equality -- proves it was forwarded completely
        # unmodified, not copied or rebuilt.
        assert engine.payloads[-1]["output_structure"] is schema
        assert engine.requested_structured_calls == [True]

    def test_async_invoke_threads_output_structure_the_same_way(self) -> None:
        schema = {"type": "object"}
        engine = FakeLLMEngine(responses=[" hello ", '{"a": 1}'])

        run_coro_sync(
            engine.async_invoke({"messages": [{"role": "user", "content": "hi"}]})
        )
        run_coro_sync(
            engine.async_invoke(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "output_structure": schema,
                }
            )
        )

        assert engine.payloads[0]["output_structure"] is None
        assert engine.payloads[1]["output_structure"] is schema
        assert engine.requested_structured_calls == [False, True]

    def test_extract_accepts_dict_result(self) -> None:
        engine = FakeLLMEngine(responses=[{"a": 1}])

        result = engine.invoke(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "output_structure": {"type": "object"},
            }
        )

        # A dict result must not be passed through .strip() (which would
        # raise AttributeError on a dict) -- confirms extract() no longer
        # str-only-validates.
        assert result.result == {"a": 1}
        assert isinstance(result.result, dict)

    def test_extract_accepts_int_result(self) -> None:
        engine = FakeLLMEngine(responses=[7])

        result = engine.invoke(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "output_structure": {"type": "integer"},
            }
        )

        assert result.result == 7
        assert isinstance(result.result, int)

    def test_extract_accepts_bool_result(self) -> None:
        engine = FakeLLMEngine(responses=[True])

        result = engine.invoke(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "output_structure": {"type": "boolean"},
            }
        )

        assert result.result is True

    def test_extract_accepts_none_result(self) -> None:
        engine = FakeLLMEngine(responses=[None])

        result = engine.invoke(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "output_structure": {"type": "null"},
            }
        )

        assert result.result is None

    def test_extract_still_rejects_unsupported_type(self) -> None:
        # int/float/bool/None are all valid JSON-decodable result types now
        # (widened alongside response-schema-basic-agent's addendum) -- a
        # set is genuinely still outside the allowed closure.
        engine = FakeLLMEngine(responses=[{1, 2, 3}])

        with pytest.raises(LLMEngineError, match="must return str"):
            engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})


class TestLLMEngineNoTimeout:
    def test_base_has_no_timeout_surface(self) -> None:
        engine = FakeLLMEngine()

        assert not hasattr(engine, "timeout_seconds")
        assert not hasattr(engine, "_timeout_seconds")
        with pytest.raises(TypeError):
            FakeLLMEngine(timeout_seconds=12.5)  # type: ignore[call-arg]


class _ReservedEngine(FakeLLMEngine):
    _RESERVED_RUNTIME_KEYS = frozenset({"model", "messages"})


class TestLLMEngineRuntimeParameters:
    def test_store_starts_empty(self) -> None:
        assert FakeLLMEngine().runtime_parameters == {}

    def test_property_returns_a_copy(self) -> None:
        engine = FakeLLMEngine()
        engine.update_runtime_parameters(temperature=0.2)

        engine.runtime_parameters["temperature"] = 9.9

        assert engine.runtime_parameters == {"temperature": 0.2}

    def test_update_merges_and_overwrites(self) -> None:
        engine = FakeLLMEngine()

        engine.update_runtime_parameters(temperature=0.2, top_p=0.9)
        engine.update_runtime_parameters(temperature=0.5)

        assert engine.runtime_parameters == {"temperature": 0.5, "top_p": 0.9}

    def test_none_removes_a_key_and_absent_key_is_a_noop(self) -> None:
        engine = FakeLLMEngine()
        engine.update_runtime_parameters(temperature=0.2, top_p=0.9)

        result = engine.update_runtime_parameters(temperature=None, never_set=None)

        assert result is None
        assert engine.runtime_parameters == {"top_p": 0.9}

    def test_reserved_key_raises(self) -> None:
        engine = _ReservedEngine()

        with pytest.raises(LLMEngineError, match="model"):
            engine.update_runtime_parameters(model="other")

    def test_reserved_key_in_a_mixed_call_applies_nothing(self) -> None:
        engine = _ReservedEngine()
        engine.update_runtime_parameters(temperature=0.2)

        with pytest.raises(LLMEngineError):
            engine.update_runtime_parameters(temperature=0.9, messages=[])

        assert engine.runtime_parameters == {"temperature": 0.2}

    def test_base_reserves_nothing_by_default(self) -> None:
        engine = FakeLLMEngine()

        engine.update_runtime_parameters(model="anything")

        assert engine.runtime_parameters == {"model": "anything"}

    def test_property_returns_a_deep_copy(self) -> None:
        engine = FakeLLMEngine()
        engine.update_runtime_parameters(thinking={"type": "adaptive", "x": [1]})

        engine.runtime_parameters["thinking"]["x"].append(2)
        engine.runtime_parameters["thinking"]["type"] = "off"

        assert engine.runtime_parameters == {"thinking": {"type": "adaptive", "x": [1]}}

    def test_update_stores_a_copy_of_the_value(self) -> None:
        engine = FakeLLMEngine()
        value = {"type": "adaptive", "x": [1]}

        engine.update_runtime_parameters(thinking=value)
        value["x"].append(2)
        value["type"] = "off"

        assert engine.runtime_parameters == {"thinking": {"type": "adaptive", "x": [1]}}

    def test_to_dict_is_a_deep_copy(self) -> None:
        engine = FakeLLMEngine()
        engine.update_runtime_parameters(thinking={"type": "adaptive"})

        data = engine.to_dict()
        data["runtime_parameters"]["thinking"]["type"] = "off"

        assert engine.runtime_parameters == {"thinking": {"type": "adaptive"}}

    def test_to_dict_redacts_secret_named_keys(self) -> None:
        engine = FakeLLMEngine()
        engine.update_runtime_parameters(
            temperature=0.2,
            max_tokens=64,
            api_key="sk-1",
            Access_Token="t",
            client_secret="s",
            authorization="Bearer x",
            password="p",
            credentials="c",
            base_url="http://localhost:1234",
            extra_headers={"X-Id": "abc"},
            extra_body={"nested": {"api_key": "k2", "plain": 1}, "items": [{"token": "t2", "n": 1}]},
        )

        rp = engine.to_dict()["runtime_parameters"]

        assert rp["temperature"] == 0.2
        assert rp["max_tokens"] == 64
        assert rp["base_url"] == "http://localhost:1234"
        for key in ("api_key", "Access_Token", "client_secret", "authorization",
                    "password", "credentials", "extra_headers"):
            assert rp[key] == "***"
        assert rp["extra_body"] == {
            "nested": {"api_key": "***", "plain": 1},
            "items": [{"token": "***", "n": 1}],
        }
        # the store itself is untouched
        assert engine.runtime_parameters["api_key"] == "sk-1"
        assert engine.runtime_parameters["extra_headers"] == {"X-Id": "abc"}

    def test_to_dict_carries_a_copy(self) -> None:
        engine = FakeLLMEngine()
        engine.update_runtime_parameters(temperature=0.2)

        data = engine.to_dict()
        data["runtime_parameters"]["temperature"] = 9.9

        assert engine.runtime_parameters == {"temperature": 0.2}


class TestLLMEngineRequireUsage:
    def test_defaults_to_true_and_is_dumped(self) -> None:
        engine = FakeLLMEngine()

        assert engine.require_usage is True
        assert engine.to_dict()["require_usage"] is True

    def test_settable_bool(self) -> None:
        engine = FakeLLMEngine()

        engine.require_usage = False

        assert engine.require_usage is False
        assert engine.to_dict()["require_usage"] is False

    @pytest.mark.parametrize("bad", [1, "yes", None])
    def test_bad_type_raises(self, bad: Any) -> None:
        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match="require_usage must be a bool"):
            engine.require_usage = bad

        assert engine.require_usage is True


class TestLLMEngineModelData:
    def test_model_data_returns_engine_model_data(self) -> None:
        data = FakeLLMEngine().model_data

        assert isinstance(data, LLMModelData)
        assert data == LLMModelData(provider="fake")

    def test_model_data_needs_no_call(self) -> None:
        engine = FakeLLMEngine()

        engine.model_data

        assert engine.requested_structured_calls == []

    def test_model_data_reflects_get_model_data(self) -> None:
        class _OtherEngine(FakeLLMEngine):
            def _get_model_data(self) -> LLMModelData:
                return LLMModelData(provider="other")

        assert _OtherEngine().model_data.provider == "other"
