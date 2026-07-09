from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

mistral_module = importlib.import_module("atomic_agentic.llm.mistral_engine")

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import MistralEngine
from atomic_agentic.utils.core import run_coro_sync


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


def _make_engine(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> MistralEngine:
    FakeMistralClient.instances.clear()
    monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)
    return MistralEngine(model="mistral-small-latest", **kwargs)


class TestMistralImmutability:
    def test_inline_cutoff_chars_is_read_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch, inline_cutoff_chars=300)
        assert engine.inline_cutoff_chars == 300
        with pytest.raises(AttributeError):
            engine.inline_cutoff_chars = 999  # type: ignore[misc]


class TestMistralEngine:
    def test_missing_mistral_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mistral_module, "Mistral", None)
        with pytest.raises(LLMEngineError, match="mistralai"):
            MistralEngine(model="mistral_test")

    def test_constructor_builds_client_from_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch, api_key="secret", timeout_seconds=5.0)
        fake = FakeMistralClient.instances[-1]

        assert engine.name == "mistral_mistral_small_latest"
        assert fake.kwargs["api_key"] == "secret"
        assert fake.kwargs["timeout_ms"] == 5000

    def test_constructor_client_injection_skips_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mistral_module, "Mistral", FakeMistralClient)
        injected = FakeMistralClient()
        FakeMistralClient.instances.clear()

        engine = MistralEngine(model="mistral-small-latest", client=injected)

        assert FakeMistralClient.instances == []
        assert engine._client is injected

    def test_call_provider_async_uses_aio_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch, temperature=0.3)
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
        engine = _make_engine(monkeypatch, temperature=None)
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
        engine = _make_engine(monkeypatch)
        assert engine._classify_kind(filename) == expected

    def test_mistral_ensure_user_parts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        messages = [{"role": "assistant", "content": "hello"}]

        index = engine._ensure_user_parts(messages)

        assert index == 1
        assert messages[1] == {"role": "user", "content": []}

        messages = [{"role": "user", "content": "hello"}]
        index = engine._ensure_user_parts(messages)

        assert index == 0
        assert messages[0]["content"] == [{"type": "text", "text": "hello"}]

    def test_mistral_payload_and_extract_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch, temperature=0.6, inline_cutoff_chars=100)

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
        engine = _make_engine(monkeypatch)
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
        engine = _make_engine(monkeypatch, temperature=0.7, inline_cutoff_chars=321)
        data = engine.to_dict()

        assert data["type"] == "MistralEngine"
        assert data["model"] == "mistral-small-latest"
        assert data["temperature"] == 0.7
        assert data["inline_cutoff_chars"] == 321

    def test_get_model_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        model_data = engine._get_model_data()

        assert model_data.provider == "mistral"
        assert model_data.model_name == "mistral-small-latest"


class TestMistralAttachments:
    def test_validate_attachment_path_rejects_illegal_ext(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "payload.zip"
        path.write_bytes(b"data")

        with pytest.raises(LLMEngineError, match="not permitted"):
            engine._validate_attachment_path(str(path))

    def test_validate_attachment_path_passes_for_txt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "notes.txt"
        path.write_text("hello")
        engine._validate_attachment_path(str(path))  # no exception

    def test_prepare_attachment_pdf_uploads_and_signs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "report.pdf"
        path.write_bytes(b"%PDF data")

        meta = engine._prepare_attachment(str(path))

        assert meta["kind"] == "pdf"
        assert meta["uploaded"] is True
        assert meta["file_id"] == "mistral_file"
        assert "signed.example/mistral_file" in meta["signed_url"]

    def test_prepare_attachment_image_uploads_and_signs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "photo.png"
        path.write_bytes(b"\x89PNG data")

        meta = engine._prepare_attachment(str(path))

        assert meta["kind"] == "image"
        assert meta["uploaded"] is True
        assert meta["file_id"] == "mistral_file"

    def test_prepare_attachment_text_inlines(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "notes.txt"
        path.write_text("Hello from text!")

        meta = engine._prepare_attachment(str(path))

        assert meta["kind"] == "text"
        assert meta["inlined"] is True
        assert "Hello from text!" in meta["inlined_text"]

    def test_prepare_attachment_text_truncates_at_cutoff(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch, inline_cutoff_chars=10)
        path = tmp_path / "long.txt"
        path.write_text("X" * 100)

        meta = engine._prepare_attachment(str(path))

        assert meta["kind"] == "text"
        assert "truncated" in meta["inlined_text"]

    def test_on_detach_with_file_id_calls_delete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        fake = FakeMistralClient.instances[-1]

        engine._on_detach({"file_id": "abc123"})

        assert "abc123" in fake.deleted_files

    def test_on_detach_without_file_id_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        fake = FakeMistralClient.instances[-1]

        engine._on_detach({})

        assert fake.deleted_files == []


class TestMistralTokenUsage:
    def _make_usage(
        self,
        *,
        prompt_tokens: int | None = 10,
        completion_tokens: int | None = 5,
        total_tokens: int | None = 15,
        prompt_tokens_details: Any = None,
    ) -> Any:
        return SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=prompt_tokens_details,
        )

    def test_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        response = SimpleNamespace(usage=self._make_usage())

        result = engine._extract_token_usage(response)

        assert result.input_tokens == 10
        assert result.generated_tokens == 5
        assert result.total_tokens == 15
        assert result.cached_tokens is None

    def test_usage_none_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="usage"):
            engine._extract_token_usage(SimpleNamespace(usage=None))

    def test_missing_prompt_tokens_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="prompt_tokens"):
            engine._extract_token_usage(
                SimpleNamespace(usage=self._make_usage(prompt_tokens=None))
            )

    def test_missing_completion_tokens_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="completion_tokens"):
            engine._extract_token_usage(
                SimpleNamespace(usage=self._make_usage(completion_tokens=None))
            )

    def test_missing_total_tokens_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="total_tokens"):
            engine._extract_token_usage(
                SimpleNamespace(usage=self._make_usage(total_tokens=None))
            )

    def test_cached_tokens_from_mapping_details(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        details = {"cached_tokens": 3}
        response = SimpleNamespace(usage=self._make_usage(prompt_tokens_details=details))

        result = engine._extract_token_usage(response)

        assert result.cached_tokens == 3

    def test_cached_tokens_from_namespace_details(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        details = SimpleNamespace(cached_tokens=7)
        response = SimpleNamespace(usage=self._make_usage(prompt_tokens_details=details))

        result = engine._extract_token_usage(response)

        assert result.cached_tokens == 7
