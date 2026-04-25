# tests/engines/test_llm_engines.py

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

llm_module = importlib.import_module("atomic_agentic.engines.LLMEngines")

from atomic_agentic.core.Exceptions import LLMEngineError
from atomic_agentic.engines.LLMEngines import (
    GeminiEngine,
    LlamaCppEngine,
    LLMEngine,
    MistralEngine,
    OpenAIEngine,
)


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


class TestLLMEngineMessagesAndInvoke:
    def test_invoke_filters_inputs_and_returns_stripped_text(self) -> None:
        engine = FakeLLMEngine()

        result = engine.invoke(
            {
                "messages": [{"role": "USER", "content": "Hello"}],
                "unused": "ignored",
            }
        )

        assert result == "hello"
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

    def test_invoke_messages_rejects_empty_messages(self) -> None:
        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match="must not be empty"):
            engine.invoke_messages([])

    def test_normalize_messages_rejects_non_mapping_message(self) -> None:
        engine = FakeLLMEngine()

        with pytest.raises(LLMEngineError, match="not a mapping"):
            engine.invoke_messages(["bad"])  # type: ignore[list-item]

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
            engine.invoke_messages([message])  # type: ignore[list-item]

    def test_extract_text_must_return_string(self) -> None:
        engine = FakeLLMEngine(provider_results=[{"text": 123}])

        with pytest.raises(LLMEngineError, match="must return str"):
            engine.invoke_messages([{"role": "user", "content": "Hello"}])

    def test_unexpected_provider_error_is_wrapped_by_invoke_messages(self) -> None:
        engine = FakeLLMEngine(provider_results=[ValueError("provider failed")])

        with pytest.raises(LLMEngineError, match="invoke failed"):
            engine.invoke_messages([{"role": "user", "content": "Hello"}])


class TestLLMEngineRetries:
    def test_timeout_error_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_module.time, "sleep", lambda _: None)
        monkeypatch.setattr(llm_module.random, "uniform", lambda _low, _high: 1.0)

        engine = FakeLLMEngine(
            max_retries=1,
            provider_results=[
                TimeoutError("temporary"),
                {"text": " recovered "},
            ],
        )

        result = engine.invoke_messages([{"role": "user", "content": "Hello"}])

        assert result == "recovered"
        assert engine.call_count == 2

    def test_connection_error_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_module.time, "sleep", lambda _: None)
        monkeypatch.setattr(llm_module.random, "uniform", lambda _low, _high: 1.0)

        engine = FakeLLMEngine(
            max_retries=1,
            provider_results=[
                ConnectionError("temporary"),
                {"text": " recovered "},
            ],
        )

        result = engine.invoke_messages([{"role": "user", "content": "Hello"}])

        assert result == "recovered"
        assert engine.call_count == 2

    def test_llm_engine_error_does_not_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_module.time, "sleep", lambda _: None)

        engine = FakeLLMEngine(
            max_retries=3,
            provider_results=[LLMEngineError("normalized")],
        )

        with pytest.raises(LLMEngineError, match="normalized"):
            engine.invoke_messages([{"role": "user", "content": "Hello"}])

        assert engine.call_count == 1

    def test_non_retryable_error_does_not_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_module.time, "sleep", lambda _: None)

        engine = FakeLLMEngine(
            max_retries=3,
            provider_results=[ValueError("bad request")],
        )

        with pytest.raises(LLMEngineError, match="invoke failed"):
            engine.invoke_messages([{"role": "user", "content": "Hello"}])

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


class FakeOpenAIClient:
    instances: list["FakeOpenAIClient"] = []

    def __init__(self, api_key: str | None = None, timeout: float | None = None) -> None:
        self.api_key = api_key
        self.timeout = timeout
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


class TestOpenAIEngine:
    def test_missing_openai_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_module, "OpenAI", None)

        with pytest.raises(RuntimeError, match="openai"):
            OpenAIEngine(model="gpt_test")

    def test_constructor_uses_fake_client_and_sanitizes_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        FakeOpenAIClient.instances.clear()
        monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAIClient)

        engine = OpenAIEngine(
            model="gpt-4o-mini",
            api_key="secret",
            timeout_seconds=12.0,
        )

        fake = FakeOpenAIClient.instances[-1]

        assert engine.name == "openai_gpt_4o_mini"
        assert fake.api_key == "secret"
        assert fake.timeout == 12.0

    def test_openai_payload_helpers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeOpenAIClient.instances.clear()
        monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAIClient)

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
        monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAIClient)

        engine = OpenAIEngine(model="gpt-4o-mini", temperature=0.25)
        response = engine._call_provider(
            {
                "blocks": [{"role": "user", "content": []}],
                "instructions": "system",
            }
        )

        fake = FakeOpenAIClient.instances[-1]

        assert engine._extract_text(response) == "openai text"
        assert fake.response_calls[-1]["model"] == "gpt-4o-mini"
        assert fake.response_calls[-1]["instructions"] == "system"
        assert fake.response_calls[-1]["temperature"] == 0.25

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
        monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAIClient)

        engine = OpenAIEngine(model="gpt-4o-mini")

        assert engine._classify_path(filename) == expected

    def test_openai_to_dict_includes_non_secret_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(llm_module, "OpenAI", FakeOpenAIClient)

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
        assert data["inline_cutoff_chars"] == 123
        assert "secret" not in str(data)


class FakeGenerateContentConfig:
    def __init__(
        self,
        *,
        temperature: float,
        system_instruction: str | None = None,
    ) -> None:
        self.temperature = temperature
        self.system_instruction = system_instruction


class FakeGenAIClient:
    instances: list["FakeGenAIClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.models = SimpleNamespace(generate_content=self._generate_content)
        self.files = SimpleNamespace(
            upload=self._upload_file,
            delete=self._delete_file,
        )
        self.generate_calls: list[dict[str, Any]] = []
        self.deleted_files: list[str] = []
        FakeGenAIClient.instances.append(self)

    def _generate_content(self, **kwargs: Any) -> Any:
        self.generate_calls.append(kwargs)
        return SimpleNamespace(text=" gemini text ")

    def _upload_file(self, file: str) -> Any:
        return SimpleNamespace(name="gemini_file")

    def _delete_file(self, name: str) -> None:
        self.deleted_files.append(name)


class FakeGenAI:
    Client = FakeGenAIClient

    class types:
        GenerateContentConfig = FakeGenerateContentConfig


class TestGeminiEngine:
    def test_missing_genai_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_module, "genai", None)

        with pytest.raises(RuntimeError, match="google-genai"):
            GeminiEngine(model="gemini_test")

    def test_constructor_uses_fake_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(llm_module, "genai", FakeGenAI)

        engine = GeminiEngine(
            model="gemini-2.5-flash",
            api_key="secret",
            timeout_seconds=7.0,
        )

        fake = FakeGenAIClient.instances[-1]

        assert engine.name == "gemini_gemini_2_5_flash"
        assert fake.kwargs["api_key"] == "secret"
        assert fake.kwargs["http_options"] == {"timeout": 7000}

    def test_gemini_payload_helpers_and_call_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        FakeGenAIClient.instances.clear()
        monkeypatch.setattr(llm_module, "genai", FakeGenAI)

        engine = GeminiEngine(model="gemini-2.5-flash", temperature=0.4)

        messages = [
            {"role": "system", "content": "System one."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        file_obj = SimpleNamespace(name="file_1")

        payload = engine._build_provider_payload(
            messages,
            {
                "file.txt": {"uploaded": True, "file_obj": file_obj},
                "inline.txt": {"inlined": True, "inlined_text": "Inline content"},
            },
        )

        assert payload["system_instruction"] == "System one."
        assert payload["contents"] == [
            file_obj,
            "Inline content",
            "Hello",
            "Hi",
        ]

        response = engine._call_provider(payload)
        fake = FakeGenAIClient.instances[-1]
        call = fake.generate_calls[-1]

        assert engine._extract_text(response) == " gemini text "
        assert call["model"] == "gemini-2.5-flash"
        assert call["contents"] == payload["contents"]
        assert call["config"].temperature == 0.4
        assert call["config"].system_instruction == "System one."

    def test_gemini_to_dict_includes_non_secret_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(llm_module, "genai", FakeGenAI)

        engine = GeminiEngine(
            model="gemini-2.5-flash",
            api_key="secret",
            temperature=0.2,
        )

        data = engine.to_dict()

        assert data["type"] == "GeminiEngine"
        assert data["model"] == "gemini-2.5-flash"
        assert data["temperature"] == 0.2
        assert "secret" not in str(data)


class FakeMistralClient:
    instances: list["FakeMistralClient"] = []

    def __init__(self, api_key: str = "", client: Any = None) -> None:
        self.api_key = api_key
        self.http_client = client
        self.chat = SimpleNamespace(complete=self._complete)
        self.files = SimpleNamespace(
            upload=self._upload,
            get_signed_url=self._get_signed_url,
            delete=self._delete,
        )
        self.complete_calls: list[dict[str, Any]] = []
        self.deleted_files: list[str] = []
        FakeMistralClient.instances.append(self)

    def _complete(self, **kwargs: Any) -> Any:
        self.complete_calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=" mistral text ")
                )
            ]
        )

    def _upload(self, **kwargs: Any) -> Any:
        return SimpleNamespace(id="mistral_file")

    def _get_signed_url(self, file_id: str) -> Any:
        return SimpleNamespace(url=f"https://signed.example/{file_id}")

    def _delete(self, file_id: str) -> None:
        self.deleted_files.append(file_id)


class TestMistralEngine:
    def test_missing_mistral_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_module, "Mistral", None)

        with pytest.raises(RuntimeError, match="mistralai"):
            MistralEngine(model="mistral_test")

    def test_constructor_uses_fake_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeMistralClient.instances.clear()
        monkeypatch.setattr(llm_module, "Mistral", FakeMistralClient)

        engine = MistralEngine(
            model="mistral-small-latest",
            api_key="secret",
            timeout_seconds=5.0,
        )

        fake = FakeMistralClient.instances[-1]

        assert engine.name == "mistral_mistral_small_latest"
        assert fake.api_key == "secret"

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
        monkeypatch.setattr(llm_module, "Mistral", FakeMistralClient)

        engine = MistralEngine(model="mistral-small-latest")

        assert engine._classify_kind(filename) == expected

    def test_mistral_ensure_user_parts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_module, "Mistral", FakeMistralClient)

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
        monkeypatch.setattr(llm_module, "Mistral", FakeMistralClient)

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
        assert fake.complete_calls[-1]["temperature"] == 0.6

    def test_mistral_extract_text_from_chunk_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_module, "Mistral", FakeMistralClient)

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
        monkeypatch.setattr(llm_module, "Mistral", FakeMistralClient)

        engine = MistralEngine(
            model="mistral-small-latest",
            api_key="secret",
            temperature=0.7,
            inline_cutoff_chars=321,
        )

        data = engine.to_dict()

        assert data["type"] == "MistralEngine"
        assert data["model"] == "mistral-small-latest"
        assert data["temperature"] == 0.7
        assert data["inline_cutoff_chars"] == 321
        assert "secret" not in str(data)


class FakeLlama:
    instances: list["FakeLlama"] = []
    pretrained_calls: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.chat_completion_calls: list[dict[str, Any]] = []
        FakeLlama.instances.append(self)

    @classmethod
    def from_pretrained(cls, **kwargs: Any) -> "FakeLlama":
        cls.pretrained_calls.append(kwargs)
        return cls(**kwargs)

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.chat_completion_calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": " llama text ",
                    }
                }
            ]
        }


class TestLlamaCppEngine:
    def test_missing_llama_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_module, "Llama", None)

        with pytest.raises(RuntimeError, match="llama-cpp-python"):
            LlamaCppEngine(model_path="model.gguf")

    def test_requires_model_path_or_repo_and_filename(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(llm_module, "Llama", FakeLlama)

        with pytest.raises(LLMEngineError, match="requires either"):
            LlamaCppEngine()

    def test_constructor_uses_local_model_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeLlama.instances.clear()
        monkeypatch.setattr(llm_module, "Llama", FakeLlama)

        engine = LlamaCppEngine(
            model_path="model.gguf",
            n_ctx=4096,
            n_threads=4,
            verbose=True,
        )

        fake = FakeLlama.instances[-1]

        assert engine.name == "llama_cpp"
        assert fake.kwargs["model_path"] == "model.gguf"
        assert fake.kwargs["n_ctx"] == 4096
        assert fake.kwargs["n_threads"] == 4
        assert fake.kwargs["verbose"] is True

    def test_constructor_uses_from_pretrained(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeLlama.pretrained_calls.clear()
        monkeypatch.setattr(llm_module, "Llama", FakeLlama)

        engine = LlamaCppEngine(
            repo_id="org/repo",
            filename="model.gguf",
        )

        assert engine.repo_id == "org/repo"
        assert engine.filename == "model.gguf"
        assert FakeLlama.pretrained_calls[-1]["repo_id"] == "org/repo"
        assert FakeLlama.pretrained_calls[-1]["filename"] == "model.gguf"

    def test_llama_payload_call_and_extract_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeLlama.instances.clear()
        monkeypatch.setattr(llm_module, "Llama", FakeLlama)

        engine = LlamaCppEngine(model_path="model.gguf")
        messages = [{"role": "user", "content": "Hello"}]
        payload = engine._build_provider_payload(messages, {})

        response = engine._call_provider(payload)
        fake = FakeLlama.instances[-1]

        assert payload == {"messages": messages}
        assert fake.chat_completion_calls[-1] == {"messages": messages}
        assert engine._extract_text(response) == "llama text"

    def test_llama_extract_text_rejects_bad_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_module, "Llama", FakeLlama)

        engine = LlamaCppEngine(model_path="model.gguf")

        with pytest.raises(LLMEngineError, match="unexpected response shape"):
            engine._extract_text({"bad": "shape"})

    def test_llama_attachments_are_not_supported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_module, "Llama", FakeLlama)

        engine = LlamaCppEngine(model_path="model.gguf")

        with pytest.raises(LLMEngineError, match="does not support attachments"):
            engine._prepare_attachment("payload.txt")

    def test_llama_to_dict_includes_non_secret_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(llm_module, "Llama", FakeLlama)

        engine = LlamaCppEngine(
            model_path="model.gguf",
            n_ctx=1024,
            verbose=True,
        )

        data = engine.to_dict()

        assert data["type"] == "LlamaCppEngine"
        assert data["model_path"] == "model.gguf"
        assert data["repo_id"] is None
        assert data["filename"] is None
        assert data["n_ctx"] == 1024
        assert data["verbose"] is True
