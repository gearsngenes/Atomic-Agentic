# tests/engines/test_llm_engines.py

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
from google.genai import types as _real_genai_types

base_module = importlib.import_module("atomic_agentic.llm.base")
openai_module = importlib.import_module("atomic_agentic.llm.openai_engine")
gemini_module = importlib.import_module("atomic_agentic.llm.gemini_engine")
mistral_module = importlib.import_module("atomic_agentic.llm.mistral_engine")
llama_module = importlib.import_module("atomic_agentic.llm.llama_engine")

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import (
    GeminiEngine,
    LlamaCppEngine,
    LLMEngine,
    MistralEngine,
    OpenAIEngine,
)
from atomic_agentic.models.results import LLMModelData, LLMResult, TokenUsage
from atomic_agentic.utils.core import run_coro_sync


class FakeLLMEngine(LLMEngine):
    """Concrete test engine for the provider-independent LLMEngine contract."""

    allowed_attachment_exts: set[str] | None = None

    def __init__(
        self,
        *,
        provider_results: list[Any] | None = None,
        prepare_result: Mapping[str, Any] | Any | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.provider_results = list(provider_results or [{"text": " hello "}])
        self.prepare_result = prepare_result
        self.payloads: list[Any] = []
        self.prepare_calls: list[str] = []
        self.detach_calls: list[Mapping[str, Any]] = []
        self.call_count = 0

    def _build_provider_payload(
        self,
        messages: list[dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "messages": messages,
            "attachments": dict(attachments),
        }

    def _call_provider(self, payload: Any) -> Any:
        self.call_count += 1
        self.payloads.append(payload)

        result = self.provider_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def _extract_text(self, response: Any) -> Any:
        return response["text"]

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        return TokenUsage(input_tokens=1, generated_tokens=1, total_tokens=2)

    def _get_model_data(self) -> LLMModelData:
        return LLMModelData(provider="fake")

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        self.prepare_calls.append(path)
        if self.prepare_result is not None:
            return self.prepare_result
        return {"path": path, "prepared": True}

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        self.detach_calls.append(meta)


class TestLLMEngineConstruction:
    def test_default_construction_uses_class_name_and_base_metadata(self) -> None:
        engine = FakeLLMEngine()

        assert engine.name == "FakeLLMEngine"
        assert engine.description == "LLM Engine"
        assert engine.return_type == "str"
        assert engine.filter_extraneous_inputs is True
        assert [(param.name, param.kind, param.type) for param in engine.parameters] == [
            ("messages", "POSITIONAL_OR_KEYWORD", "List[Dict[str, str]]")
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
        engine = FakeLLMEngine(provider_results=[{"text": 123}])

        with pytest.raises(LLMEngineError, match="must return str"):
            engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})

    def test_unexpected_provider_error_is_wrapped_by_invoke(self) -> None:
        engine = FakeLLMEngine(provider_results=[ValueError("provider failed")])

        with pytest.raises(LLMEngineError, match="invoke failed"):
            engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})


class TestLLMEngineRetries:
    def test_timeout_error_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(base_module.time, "sleep", lambda _: None)
        monkeypatch.setattr(base_module.random, "uniform", lambda _low, _high: 1.0)

        engine = FakeLLMEngine(
            max_retries=1,
            provider_results=[
                TimeoutError("temporary"),
                {"text": " recovered "},
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
            provider_results=[
                ConnectionError("temporary"),
                {"text": " recovered "},
            ],
        )

        result = engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})

        assert result.result == "recovered"
        assert engine.call_count == 2

    def test_llm_engine_error_does_not_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(base_module.time, "sleep", lambda _: None)

        engine = FakeLLMEngine(
            max_retries=3,
            provider_results=[LLMEngineError("normalized")],
        )

        with pytest.raises(LLMEngineError, match="normalized"):
            engine.invoke({"messages": [{"role": "user", "content": "Hello"}]})

        assert engine.call_count == 1

    def test_non_retryable_error_does_not_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(base_module.time, "sleep", lambda _: None)

        engine = FakeLLMEngine(
            max_retries=3,
            provider_results=[ValueError("bad request")],
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


class TestLLMEngineImmutability:
    def test_timeout_seconds_is_read_only(self) -> None:
        engine = FakeLLMEngine(timeout_seconds=12.5)

        assert engine.timeout_seconds == 12.5

        with pytest.raises(AttributeError):
            engine.timeout_seconds = 99.0  # type: ignore[misc]

    def test_openai_inline_cutoff_chars_is_read_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)
        engine = OpenAIEngine(model="gpt-4o-mini", inline_cutoff_chars=500)

        assert engine.inline_cutoff_chars == 500

        with pytest.raises(AttributeError):
            engine.inline_cutoff_chars = 999  # type: ignore[misc]

    def test_mistral_inline_cutoff_chars_is_read_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)
        engine = MistralEngine(model="mistral-small-latest", inline_cutoff_chars=300)

        assert engine.inline_cutoff_chars == 300

        with pytest.raises(AttributeError):
            engine.inline_cutoff_chars = 999  # type: ignore[misc]



class FakeOpenAIClient:
    instances: list["FakeOpenAIClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.responses = SimpleNamespace(create=self._create_response)
        self.files = SimpleNamespace(
            create=self._create_file,
            delete=self._delete_file,
        )
        self.response_calls: list[dict[str, Any]] = []
        self.deleted_files: list[str] = []
        FakeOpenAIClient.instances.append(self)

    def _create_response(self, **kwargs: Any) -> Any:
        self.response_calls.append(kwargs)
        return SimpleNamespace(output_text=" openai text ")

    def _create_file(self, file: Any, purpose: str) -> Any:
        return SimpleNamespace(id="file_123")

    def _delete_file(self, file_id: str) -> None:
        self.deleted_files.append(file_id)


class FakeAsyncOpenAIClient:
    instances: list["FakeAsyncOpenAIClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.responses = SimpleNamespace(create=self._create_response)
        self.response_calls: list[dict[str, Any]] = []
        FakeAsyncOpenAIClient.instances.append(self)

    async def _create_response(self, **kwargs: Any) -> Any:
        self.response_calls.append(kwargs)
        return SimpleNamespace(output_text=" openai async text ")


class TestOpenAIEngine:
    def test_missing_openai_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_module, "OpenAI", None)

        with pytest.raises(LLMEngineError, match="openai"):
            OpenAIEngine(model="gpt_test")

    def test_constructor_uses_fake_client_and_sanitizes_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        FakeOpenAIClient.instances.clear()
        FakeAsyncOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(
            model="gpt-4o-mini",
            api_key="secret",        # flows through client_kwargs
            timeout_seconds=12.0,
        )

        # Only ONE sync client constructed — no async client built.
        assert len(FakeOpenAIClient.instances) == 1
        assert len(FakeAsyncOpenAIClient.instances) == 0

        fake = FakeOpenAIClient.instances[-1]
        assert engine.name == "openai_gpt_4o_mini"
        assert fake.kwargs["api_key"] == "secret"
        assert fake.kwargs["timeout"] == 12.0   # setdefault from timeout_seconds
        assert engine._client is fake

    def test_sync_client_injection_skips_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        injected = FakeOpenAIClient()
        count_before = len(FakeOpenAIClient.instances)

        engine = OpenAIEngine(model="gpt-4o-mini", client=injected)

        # No additional sync client constructed.
        assert len(FakeOpenAIClient.instances) == count_before
        assert engine._client is injected

    def test_async_client_injection_skips_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeAsyncOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        injected = FakeAsyncOpenAIClient()
        count_before = len(FakeAsyncOpenAIClient.instances)

        engine = OpenAIEngine(model="gpt-4o-mini", client=injected)

        # No additional async client constructed.
        assert len(FakeAsyncOpenAIClient.instances) == count_before
        assert engine._client is injected

    def test_async_client_routes_sync_via_run_coro_sync(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeAsyncOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        injected = FakeAsyncOpenAIClient()
        engine = OpenAIEngine(model="gpt-4o-mini", client=injected)

        response = engine._call_provider({"blocks": [], "instructions": None})

        # Async client's response_calls populated via run_coro_sync.
        assert injected.response_calls
        assert engine._extract_text(response) == " openai async text "

    def test_openai_payload_helpers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(model="gpt-4o-mini")

        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        assert engine._collect_instructions(messages) == "Be concise."
        assert engine._build_role_blocks(messages) == [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi"}],
            },
        ]

        blocks: list[dict[str, Any]] = []
        assert engine._ensure_user_block(blocks) == 0
        assert blocks == [{"role": "user", "content": []}]

    def test_openai_call_provider_and_extract_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(model="gpt-4o-mini", temperature=0.25)
        response = engine._call_provider(
            {
                "blocks": [{"role": "user", "content": []}],
                "instructions": "system",
            }
        )

        fake = FakeOpenAIClient.instances[-1]

        assert engine._extract_text(response) == " openai text "
        assert fake.response_calls[-1]["model"] == "gpt-4o-mini"
        assert fake.response_calls[-1]["instructions"] == "system"
        assert fake.response_calls[-1]["temperature"] == 0.25

    def test_call_provider_kwargs_temperature_and_new_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        FakeAsyncOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(
            model="gpt-4o-mini",
            temperature=0.5,
            max_output_tokens=512,
            truncation="auto",
        )
        engine._call_provider({"blocks": [], "instructions": None})
        call = FakeOpenAIClient.instances[-1].response_calls[-1]

        assert call["temperature"] == 0.5
        assert call["max_output_tokens"] == 512
        assert call["truncation"] == "auto"
        assert "reasoning" not in call

    def test_call_provider_reasoning_suppresses_temperature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        FakeAsyncOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(
            model="o3-mini",
            temperature=0.7,             # set, but should be suppressed
            reasoning={"effort": "high"},
        )
        engine._call_provider({"blocks": [], "instructions": None})
        call = FakeOpenAIClient.instances[-1].response_calls[-1]

        assert "temperature" not in call
        assert call["reasoning"] == {"effort": "high"}

    def test_call_provider_temperature_none_not_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        FakeAsyncOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(model="gpt-4o-mini", temperature=None)
        engine._call_provider({"blocks": [], "instructions": None})
        call = FakeOpenAIClient.instances[-1].response_calls[-1]

        assert "temperature" not in call

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("doc.pdf", "pdf"),
            ("image.png", "image"),
            ("notes.txt", "text"),
        ],
    )
    def test_openai_classify_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        filename: str,
        expected: str,
    ) -> None:
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(model="gpt-4o-mini")

        assert engine._classify_path(filename) == expected

    def test_openai_to_dict_includes_non_secret_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(
            model="gpt-4o-mini",
            api_key="secret",
            temperature=0.33,
            inline_cutoff_chars=123,
        )

        data = engine.to_dict()

        assert data["type"] == "OpenAIEngine"
        assert data["model"] == "gpt-4o-mini"
        assert data["temperature"] == 0.33
        assert data["max_output_tokens"] is None    # default
        assert data["reasoning"] is None            # default
        assert data["truncation"] is None           # default
        assert data["inline_cutoff_chars"] == 123
        assert "secret" not in str(data)


class FakeGenAIClient:
    instances: list["FakeGenAIClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.models = SimpleNamespace(generate_content=self._generate_content)
        self.aio = SimpleNamespace(
            models=SimpleNamespace(generate_content=self._generate_content_async)
        )
        self.files = SimpleNamespace(
            upload=self._upload_file,
            delete=self._delete_file,
        )
        self.generate_calls: list[dict[str, Any]] = []
        self.aio_generate_calls: list[dict[str, Any]] = []
        self.deleted_files: list[str] = []
        FakeGenAIClient.instances.append(self)

    def _generate_content(self, **kwargs: Any) -> Any:
        self.generate_calls.append(kwargs)
        return SimpleNamespace(text=" gemini text ", usage_metadata=None)

    async def _generate_content_async(self, **kwargs: Any) -> Any:
        self.aio_generate_calls.append(kwargs)
        return SimpleNamespace(text=" gemini async text ", usage_metadata=None)

    def _upload_file(self, *, file: str) -> Any:
        return SimpleNamespace(name="gemini_file", uri="gs://fake/gemini_file", mime_type="text/plain")

    def _delete_file(self, *, name: str) -> None:
        self.deleted_files.append(name)


class TestOpenAIExtractTokenUsage:
    def _engine(self, monkeypatch: pytest.MonkeyPatch) -> OpenAIEngine:
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)
        return OpenAIEngine(model="gpt-4o-mini")

    def _usage(
        self,
        *,
        input_tokens: int = 10,
        output_tokens: int = 5,
        total_tokens: int = 15,
        output_tokens_details: Any = None,
        input_tokens_details: Any = None,
    ) -> Any:
        return SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            output_tokens_details=output_tokens_details,
            input_tokens_details=input_tokens_details,
        )

    def test_both_details_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = self._engine(monkeypatch)
        response = SimpleNamespace(usage=self._usage(
            input_tokens=10,
            output_tokens=8,
            total_tokens=18,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
            input_tokens_details=SimpleNamespace(cached_tokens=2),
        ))

        result = engine._extract_token_usage(response)

        assert result.reasoning_tokens == 3
        assert result.response_tokens == 5          # 8 - 3
        assert result.cached_tokens == 2

    def test_output_tokens_details_none_defaults_reasoning_to_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = self._engine(monkeypatch)
        response = SimpleNamespace(usage=self._usage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            output_tokens_details=None,
            input_tokens_details=SimpleNamespace(cached_tokens=1),
        ))

        result = engine._extract_token_usage(response)

        assert result.reasoning_tokens == 0
        assert result.response_tokens == 5          # 5 - 0
        assert result.cached_tokens == 1

    def test_input_tokens_details_none_defaults_cached_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = self._engine(monkeypatch)
        response = SimpleNamespace(usage=self._usage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            input_tokens_details=None,
        ))

        result = engine._extract_token_usage(response)

        assert result.cached_tokens is None

    def test_usage_none_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = self._engine(monkeypatch)
        response = SimpleNamespace(usage=None)

        with pytest.raises(LLMEngineError, match="did not include usage"):
            engine._extract_token_usage(response)


class FakeGenAI:
    Client = FakeGenAIClient

    class types:
        GenerateContentConfig = _real_genai_types.GenerateContentConfig
        Content = _real_genai_types.Content
        Part = _real_genai_types.Part
        FileData = _real_genai_types.FileData
        ThinkingConfig = _real_genai_types.ThinkingConfig


class TestGeminiEngine:
    def test_missing_genai_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(gemini_module, "genai", None)

        with pytest.raises(LLMEngineError, match="google-genai"):
            GeminiEngine(model="gemini_test")

    def test_constructor_builds_client_from_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)

        engine = GeminiEngine(
            model="gemini-2.5-flash",
            timeout_seconds=7.0,
            api_key="secret",
        )
        fake = FakeGenAIClient.instances[-1]

        assert engine.name == "gemini_gemini_2_5_flash"
        assert fake.kwargs["api_key"] == "secret"
        assert fake.kwargs["http_options"] == {"timeout": 7000}

    def test_constructor_client_injection_skips_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)

        injected = FakeGenAIClient()
        count_before = len(FakeGenAIClient.instances)
        engine = GeminiEngine(model="gemini-2.5-flash", client=injected)

        assert len(FakeGenAIClient.instances) == count_before
        assert engine._client is injected

    def test_build_content_list_maps_roles_and_skips_system(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)

        engine = GeminiEngine(model="gemini-2.5-flash")
        messages = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        contents = engine._build_content_list(messages, {})

        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Hello"
        assert contents[1].role == "model"
        assert contents[1].parts[0].text == "Hi"

    def test_build_content_list_injects_attachments_into_first_user_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)

        engine = GeminiEngine(model="gemini-2.5-flash")
        file_obj = SimpleNamespace(name="f1", uri="gs://fake/f1", mime_type="text/plain")
        attachments = {
            "file.txt": {"uploaded": True, "file_obj": file_obj, "mime": "text/plain"},
            "note.txt": {"inlined": True, "inlined_text": "Inline!"},
        }
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        contents = engine._build_content_list(messages, attachments)

        assert len(contents) == 2
        first_parts = contents[0].parts
        assert len(first_parts) == 3  # file Part + inline Part + text Part
        assert first_parts[0].file_data.file_uri == "gs://fake/f1"
        assert first_parts[1].text == "Inline!"
        assert first_parts[2].text == "Hello"
        assert contents[1].role == "model"

    def test_call_provider_uses_sync_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)

        engine = GeminiEngine(model="gemini-2.5-flash", temperature=0.4)
        messages = [
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Hello"},
        ]
        payload = engine._build_provider_payload(messages, {})

        assert payload["system_instruction"] == "Sys."
        assert len(payload["contents"]) == 1
        assert payload["contents"][0].role == "user"

        response = engine._call_provider(payload)
        fake = FakeGenAIClient.instances[-1]
        call = fake.generate_calls[-1]

        assert call["model"] == "gemini-2.5-flash"
        assert call["config"].temperature == 0.4
        assert call["config"].system_instruction == "Sys."
        assert engine._extract_text(response) == " gemini text "

    def test_call_provider_async_uses_aio_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)

        engine = GeminiEngine(model="gemini-2.5-flash")
        payload = engine._build_provider_payload(
            [{"role": "user", "content": "Hi"}], {}
        )
        response = run_coro_sync(engine._call_provider_async(payload))
        fake = FakeGenAIClient.instances[-1]

        assert fake.aio_generate_calls        # aio path was hit
        assert not fake.generate_calls         # sync path was NOT hit
        assert engine._extract_text(response) == " gemini async text "

    def test_temperature_none_omits_from_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)

        engine = GeminiEngine(model="gemini-2.5-flash", temperature=None)
        cfg = engine._build_generate_config(None)

        assert cfg.temperature is None

    def test_thinking_config_passed_to_generate_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)

        engine = GeminiEngine(
            model="gemini-2.5-flash",
            thinking_config={"thinking_budget": 1024},
        )
        cfg = engine._build_generate_config(None)

        assert cfg.thinking_config.thinking_budget == 1024

    def test_gemini_to_dict_includes_non_secret_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gemini_module, "genai", FakeGenAI)

        engine = GeminiEngine(
            model="gemini-2.5-flash",
            api_key="secret",
            temperature=0.2,
            max_output_tokens=512,
            thinking_config={"thinking_budget": 100},
        )
        data = engine.to_dict()

        assert data["type"] == "GeminiEngine"
        assert data["model"] == "gemini-2.5-flash"
        assert data["temperature"] == 0.2
        assert data["max_output_tokens"] == 512
        assert data["thinking_config"] == {"thinking_budget": 100}
        assert "secret" not in str(data)


class FakeMistralClient:
    instances: list["FakeMistralClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.chat = SimpleNamespace(
            complete=self._complete,
            complete_async=self._complete_async,
        )
        self.files = SimpleNamespace(
            upload=self._upload,
            get_signed_url=self._get_signed_url,
            delete=self._delete,
        )
        self.complete_calls: list[dict[str, Any]] = []
        self.aio_complete_calls: list[dict[str, Any]] = []
        self.deleted_files: list[str] = []
        FakeMistralClient.instances.append(self)

    def _complete(self, **kwargs: Any) -> Any:
        self.complete_calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" mistral text "))]
        )

    async def _complete_async(self, **kwargs: Any) -> Any:
        self.aio_complete_calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" mistral async text "))]
        )

    def _upload(self, **kwargs: Any) -> Any:
        return SimpleNamespace(id="mistral_file")

    def _get_signed_url(self, *, file_id: str, **kwargs: Any) -> Any:
        return SimpleNamespace(url=f"https://signed.example/{file_id}")

    def _delete(self, *, file_id: str, **kwargs: Any) -> None:
        self.deleted_files.append(file_id)


class TestMistralEngine:
    def test_missing_mistral_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mistral_module, "Mistral", None)

        with pytest.raises(LLMEngineError, match="mistralai"):
            MistralEngine(model="mistral_test")

    def test_constructor_builds_client_from_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeMistralClient.instances.clear()
        monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)

        engine = MistralEngine(
            model="mistral-small-latest",
            api_key="secret",
            timeout_seconds=5.0,
        )

        fake = FakeMistralClient.instances[-1]

        assert engine.name == "mistral_mistral_small_latest"
        assert fake.kwargs["api_key"] == "secret"
        assert fake.kwargs["timeout_ms"] == 5000

    def test_constructor_client_injection_skips_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeMistralClient.instances.clear()
        monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)
        injected = FakeMistralClient()
        FakeMistralClient.instances.clear()

        engine = MistralEngine(model="mistral-small-latest", client=injected)

        assert FakeMistralClient.instances == []
        assert engine._client is injected

    def test_call_provider_async_uses_aio_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeMistralClient.instances.clear()
        monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)

        engine = MistralEngine(model="mistral-small-latest", temperature=0.3)
        payload = {"messages": [{"role": "user", "content": "hello"}]}

        result = run_coro_sync(engine._call_provider_async(payload))

        fake = FakeMistralClient.instances[-1]
        assert fake.aio_complete_calls
        call = fake.aio_complete_calls[-1]
        assert call["model"] == "mistral-small-latest"
        assert call["temperature"] == 0.3
        assert engine._extract_text(result) == "mistral async text"

    def test_temperature_none_omits_from_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeMistralClient.instances.clear()
        monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)

        engine = MistralEngine(model="mistral-small-latest", temperature=None)
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        engine._call_provider(payload)

        fake = FakeMistralClient.instances[-1]
        assert "temperature" not in fake.complete_calls[-1]

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("doc.pdf", "pdf"),
            ("image.png", "image"),
            ("notes.txt", "text"),
        ],
    )
    def test_mistral_classify_kind(
        self,
        monkeypatch: pytest.MonkeyPatch,
        filename: str,
        expected: str,
    ) -> None:
        monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)

        engine = MistralEngine(model="mistral-small-latest")

        assert engine._classify_kind(filename) == expected

    def test_mistral_ensure_user_parts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)

        engine = MistralEngine(model="mistral-small-latest")
        messages = [{"role": "assistant", "content": "hello"}]

        index = engine._ensure_user_parts(messages)

        assert index == 1
        assert messages[1] == {"role": "user", "content": []}

        messages = [{"role": "user", "content": "hello"}]
        index = engine._ensure_user_parts(messages)

        assert index == 0
        assert messages[0]["content"] == [{"type": "text", "text": "hello"}]

    def test_mistral_payload_and_extract_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeMistralClient.instances.clear()
        monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)

        engine = MistralEngine(
            model="mistral-small-latest",
            temperature=0.6,
            inline_cutoff_chars=100,
        )

        payload = engine._build_provider_payload(
            [{"role": "user", "content": "Question"}],
            {
                "notes.txt": {"kind": "text", "inlined_text": "Inline notes"},
                "doc.pdf": {"kind": "pdf", "signed_url": "https://doc"},
                "image.png": {"kind": "image", "signed_url": "https://image"},
            },
        )

        user_parts = payload["messages"][0]["content"]

        assert user_parts[0] == {"type": "text", "text": "Question"}
        assert any(part["type"] == "text" and "Inline notes" in part["text"] for part in user_parts)
        assert {"type": "document_url", "document_url": "https://doc"} in user_parts
        assert {"type": "image_url", "image_url": "https://image"} in user_parts

        response = engine._call_provider(payload)
        fake = FakeMistralClient.instances[-1]

        assert engine._extract_text(response) == "mistral text"
        assert fake.complete_calls[-1]["model"] == "mistral-small-latest"

    def test_mistral_extract_text_from_chunk_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)

        engine = MistralEngine(model="mistral-small-latest")
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=[
                            {"text": "hello "},
                            {"text": "world"},
                        ]
                    )
                )
            ]
        )

        assert engine._extract_text(response) == "hello world"

    def test_mistral_to_dict_includes_non_secret_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)

        engine = MistralEngine(
            model="mistral-small-latest",
            temperature=0.7,
            inline_cutoff_chars=321,
        )

        data = engine.to_dict()

        assert data["type"] == "MistralEngine"
        assert data["model"] == "mistral-small-latest"
        assert data["temperature"] == 0.7
        assert data["inline_cutoff_chars"] == 321


class FakeLlama:
    instances: list["FakeLlama"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.chat_completion_calls: list[dict[str, Any]] = []
        FakeLlama.instances.append(self)

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.chat_completion_calls.append(kwargs)
        return {
            "choices": [{"message": {"content": " llama text "}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }


class TestLlamaCppEngine:
    def test_missing_llama_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llama_module, "Llama", None)

        with pytest.raises(LLMEngineError, match="llama-cpp-python"):
            LlamaCppEngine(model_path="model.gguf")

    def test_requires_model_path_or_repo_and_filename(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        with pytest.raises(LLMEngineError, match="requires either"):
            LlamaCppEngine()

    def test_constructor_uses_local_model_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeLlama.instances.clear()
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        engine = LlamaCppEngine(
            model_path="model.gguf",
            n_ctx=4096,
            verbose=True,
            n_threads=4,
        )

        fake = FakeLlama.instances[-1]

        assert engine.name == "llama_cpp"
        assert fake.kwargs["model_path"] == "model.gguf"
        assert fake.kwargs["n_ctx"] == 4096
        assert fake.kwargs["verbose"] is True
        assert fake.kwargs["n_threads"] == 4

    def test_constructor_resolves_repo_and_filename_via_hf_hub_download(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        FakeLlama.instances.clear()
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        download_calls: list[dict[str, Any]] = []
        resolved_path = "/cache/org/repo/model.gguf"

        def fake_hf_hub_download(**kwargs: Any) -> str:
            download_calls.append(kwargs)
            return resolved_path

        monkeypatch.setattr(llama_module, "hf_hub_download", fake_hf_hub_download)

        engine = LlamaCppEngine(
            repo_id="org/repo",
            filename="model.gguf",
        )

        fake = FakeLlama.instances[-1]

        assert engine.repo_id == "org/repo"
        assert engine.filename == "model.gguf"
        assert engine.model_path == resolved_path
        assert download_calls[-1]["repo_id"] == "org/repo"
        assert download_calls[-1]["filename"] == "model.gguf"
        assert "subfolder" in download_calls[-1]
        assert fake.kwargs["model_path"] == resolved_path

    def test_llama_payload_call_and_extract_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeLlama.instances.clear()
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        engine = LlamaCppEngine(model_path="model.gguf")
        messages = [{"role": "user", "content": "Hello"}]
        payload = engine._build_provider_payload(messages, {})

        response = engine._call_provider(payload)
        fake = FakeLlama.instances[-1]

        assert payload == {"messages": messages}
        assert fake.chat_completion_calls[-1] == {"messages": messages}
        assert engine._extract_text(response) == "llama text"

    def test_llama_extract_text_rejects_bad_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        engine = LlamaCppEngine(model_path="model.gguf")

        with pytest.raises(LLMEngineError, match="unexpected response shape"):
            engine._extract_text({"bad": "shape"})

    def test_llama_attachments_are_not_supported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        engine = LlamaCppEngine(model_path="model.gguf")

        with pytest.raises(LLMEngineError, match="does not support attachments"):
            engine._prepare_attachment("payload.txt")

    def test_llama_to_dict_includes_non_secret_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        engine = LlamaCppEngine(
            model_path="model.gguf",
            n_ctx=1024,
            verbose=True,
            n_gpu_layers=32,
            temperature=0.7,
            top_k=40,
        )

        data = engine.to_dict()

        assert data["type"] == "LlamaCppEngine"
        assert data["model_path"] == "model.gguf"
        assert data["repo_id"] is None
        assert data["filename"] is None
        assert data["n_ctx"] == 1024
        assert data["verbose"] is True
        assert data["llama_kwargs"] == {"n_gpu_layers": 32}
        assert data["temperature"] == 0.7
        assert data["top_k"] == 40
        assert "n_threads" not in data
        assert "n_gpu_layers" not in data

    def test_llamacpp_source_params_are_read_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)
        engine = LlamaCppEngine(model_path="model.gguf")

        assert engine.model_path == "model.gguf"
        assert engine.repo_id is None
        assert engine.filename is None

        for attr in ("model_path", "repo_id", "filename"):
            with pytest.raises(AttributeError):
                setattr(engine, attr, None)

    def test_llamacpp_model_load_params_are_read_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)
        engine = LlamaCppEngine(model_path="model.gguf", n_ctx=2048, verbose=True)

        assert engine.n_ctx == 2048
        assert engine.verbose is True

        for attr in ("n_ctx", "verbose"):
            with pytest.raises(AttributeError):
                setattr(engine, attr, None)

    def test_llama_kwargs_forwarded_to_llama_constructor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeLlama.instances.clear()
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        LlamaCppEngine(
            model_path="model.gguf",
            n_gpu_layers=32,
            flash_attn=True,
        )

        fake = FakeLlama.instances[-1]
        assert fake.kwargs["n_gpu_layers"] == 32
        assert fake.kwargs["flash_attn"] is True

    def test_generation_params_set_sent_to_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeLlama.instances.clear()
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        engine = LlamaCppEngine(
            model_path="model.gguf",
            temperature=0.5,
            top_k=40,
            top_p=0.9,
            min_p=0.05,
            max_tokens=256,
            repeat_penalty=1.1,
            seed=42,
            stop=["</s>"],
        )
        engine._call_provider({"messages": [{"role": "user", "content": "hi"}]})

        call = FakeLlama.instances[-1].chat_completion_calls[-1]
        assert call["temperature"] == 0.5
        assert call["top_k"] == 40
        assert call["top_p"] == 0.9
        assert call["min_p"] == 0.05
        assert call["max_tokens"] == 256
        assert call["repeat_penalty"] == 1.1
        assert call["seed"] == 42
        assert call["stop"] == ["</s>"]

    def test_generation_params_none_omit_from_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeLlama.instances.clear()
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        engine = LlamaCppEngine(model_path="model.gguf")
        engine._call_provider({"messages": [{"role": "user", "content": "hi"}]})

        call = FakeLlama.instances[-1].chat_completion_calls[-1]
        for param in ("temperature", "top_k", "top_p", "min_p",
                      "max_tokens", "repeat_penalty", "seed", "stop"):
            assert param not in call, f"{param!r} should be omitted when None"

    def test_hf_subfolder_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeLlama.instances.clear()
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        download_calls: list[dict[str, Any]] = []

        def fake_hf_hub_download(**kwargs: Any) -> str:
            download_calls.append(kwargs)
            return "/cache/model.gguf"

        monkeypatch.setattr(llama_module, "hf_hub_download", fake_hf_hub_download)

        LlamaCppEngine(
            repo_id="org/repo",
            filename="model.gguf",
            subfolder="gguf",
        )

        assert download_calls[-1]["subfolder"] == "gguf"
