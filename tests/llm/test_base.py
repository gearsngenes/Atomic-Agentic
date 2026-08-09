from __future__ import annotations

import importlib
from typing import Any, Mapping

import pytest

base_module = importlib.import_module("atomic_agentic.llm.base")

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import LLMEngine
from atomic_agentic.models.results import LLMModelData, LLMResult, TokenUsage
from fake_engines import FakeLLMEngine


class TestLLMEngineConstruction:
    def test_default_construction_uses_class_name_and_base_metadata(self) -> None:
        engine = FakeLLMEngine()

        assert engine.name == "FakeLLMEngine"
        assert engine.description == "LLM Engine"
        assert engine.return_type == "str"
        assert [(param.name, param.kind, param.type) for param in engine.parameters] == [
            ("messages", "POSITIONAL_OR_KEYWORD", ("list[dict[str, str]]",))
        ]

    def test_custom_construction_values_are_stored(self) -> None:
        engine = FakeLLMEngine(
            name="fake_engine",
            description="Fake test engine.",
            timeout_seconds=12.5,
            max_retries=3,
            retry_backoff_base=0.25,
            retry_backoff_max=2.0,
        )

        data = engine.to_dict()

        assert engine.name == "fake_engine"
        assert engine.description == "Fake test engine."
        assert data["timeout_seconds"] == 12.5
        assert data["max_retries"] == 3
        assert data["retry_backoff_base"] == 0.25
        assert data["retry_backoff_max"] == 2.0
        assert data["attachments"] == {}

    def test_invalid_atomic_invokable_name_still_raises(self) -> None:
        with pytest.raises(ValueError):
            FakeLLMEngine(name="bad-name")


class TestLLMEngineNamespace:
    def test_llm_engine_namespace_default(self) -> None:
        engine = FakeLLMEngine()
        assert engine.namespace == "llm"

    def test_llm_engine_namespace_explicit(self) -> None:
        engine = FakeLLMEngine(namespace="prod_llm")
        assert engine.namespace == "prod_llm"

    def test_llm_engine_namespace_in_to_dict(self) -> None:
        engine = FakeLLMEngine(namespace="prod_llm")
        assert engine.to_dict()["namespace"] == "prod_llm"


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
                "attachments": {},
            }
        ]

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

    def test_extract_text_must_return_string(self) -> None:
        engine = FakeLLMEngine(responses=[123])

        with pytest.raises(LLMEngineError, match="must return str"):
            engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})

    def test_unexpected_provider_error_is_wrapped_by_invoke(self) -> None:
        engine = FakeLLMEngine(responses=[ValueError("provider failed")])

        with pytest.raises(LLMEngineError, match="invoke failed"):
            engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})


class TestLLMEngineRetries:
    def test_timeout_error_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(base_module.time, "sleep", lambda _: None)
        monkeypatch.setattr(base_module.random, "uniform", lambda _low, _high: 1.0)

        engine = FakeLLMEngine(
            max_retries=1,
            responses=[
                TimeoutError("temporary"),
                " recovered ",
            ],
        )

        result = engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})

        assert result.result == "recovered"
        assert engine.call_count == 2

    def test_connection_error_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(base_module.time, "sleep", lambda _: None)
        monkeypatch.setattr(base_module.random, "uniform", lambda _low, _high: 1.0)

        engine = FakeLLMEngine(
            max_retries=1,
            responses=[
                ConnectionError("temporary"),
                " recovered ",
            ],
        )

        result = engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})

        assert result.result == "recovered"
        assert engine.call_count == 2

    def test_llm_engine_error_does_not_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(base_module.time, "sleep", lambda _: None)

        engine = FakeLLMEngine(
            max_retries=3,
            responses=[LLMEngineError("normalized")],
        )

        with pytest.raises(LLMEngineError, match="normalized"):
            engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})

        assert engine.call_count == 1

    def test_non_retryable_error_does_not_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(base_module.time, "sleep", lambda _: None)

        engine = FakeLLMEngine(
            max_retries=3,
            responses=[ValueError("bad request")],
        )

        with pytest.raises(LLMEngineError, match="invoke failed"):
            engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})

        assert engine.call_count == 1


class TestLLMEngineAttachments:
    def test_attach_rejects_blank_and_non_string_paths(self) -> None:
        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match="path"):
            engine.attach("")

        with pytest.raises(LLMEngineError, match="path"):
            engine.attach(123)  # type: ignore[arg-type]

    def test_attach_rejects_nonexistent_path(self) -> None:
        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match="does not exist"):
            engine.attach("missing_file.txt")

    def test_attach_rejects_illegal_extension(self, tmp_path: Any) -> None:
        path = tmp_path / "payload.zip"
        path.write_text("bad")

        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match="not allowed"):
            engine.attach(str(path))

    def test_attach_rejects_extension_not_in_allow_list(self, tmp_path: Any) -> None:
        path = tmp_path / "payload.md"
        path.write_text("hello")

        engine = FakeLLMEngine()
        engine.allowed_attachment_exts = {".txt"}

        with pytest.raises(LLMEngineError, match="not supported"):
            engine.attach(str(path))

    def test_attach_stores_prepare_metadata_and_caches_same_path(self, tmp_path: Any) -> None:
        path = tmp_path / "payload.txt"
        path.write_text("hello")

        engine = FakeLLMEngine(prepare_result={"kind": "text"})

        first = engine.attach(str(path))
        second = engine.attach(str(path))

        assert first == {"kind": "text"}
        assert second == {"kind": "text"}
        assert engine.prepare_calls == [str(path)]
        assert engine.attachments == {str(path): {"kind": "text"}}

    def test_attach_requires_prepare_attachment_to_return_mapping(self, tmp_path: Any) -> None:
        path = tmp_path / "payload.txt"
        path.write_text("hello")

        engine = FakeLLMEngine(prepare_result=["bad"])

        with pytest.raises(LLMEngineError, match="must return a mapping"):
            engine.attach(str(path))

    def test_attachments_property_returns_top_level_copy(self, tmp_path: Any) -> None:
        path = tmp_path / "payload.txt"
        path.write_text("hello")

        engine = FakeLLMEngine(prepare_result={"kind": "text"})
        engine.attach(str(path))

        snapshot = engine.attachments
        snapshot["other.txt"] = {"kind": "other"}

        assert "other.txt" not in engine.attachments

    def test_detach_and_clear_attachments(self, tmp_path: Any) -> None:
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("one")
        second.write_text("two")

        engine = FakeLLMEngine()
        engine.attach(str(first))
        engine.attach(str(second))

        assert engine.detach(str(first)) is True
        assert engine.detach(str(first)) is False
        assert len(engine.detach_calls) == 1

        engine.clear_attachments()

        assert engine.attachments == {}
        assert len(engine.detach_calls) == 2


class TestLLMEngineAbstractShouldRetry:
    def test_missing_should_retry_override_raises_type_error(self) -> None:
        class IncompleteEngine(LLMEngine):
            def _build_provider_payload(self, messages, attachments):
                return {}

            def _call_provider(self, payload):
                return {"text": ""}

            def _extract_text(self, response):
                return ""

            def _extract_token_usage(self, response):
                return TokenUsage(
                    input_tokens=0, generated_tokens=0, total_tokens=0, response_tokens=0
                )

            def _get_model_data(self):
                return LLMModelData(provider="incomplete")

        with pytest.raises(TypeError):
            IncompleteEngine()


class TestLLMEngineImmutability:
    def test_timeout_seconds_is_read_only(self) -> None:
        engine = FakeLLMEngine(timeout_seconds=12.5)

        assert engine.timeout_seconds == 12.5

        with pytest.raises(AttributeError):
            engine.timeout_seconds = 99.0  # type: ignore[misc]
