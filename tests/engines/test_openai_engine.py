from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any

import pytest

openai_module = importlib.import_module("atomic_agentic.llm.openai_engine")

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import OpenAIEngine
from atomic_agentic.utils.core import run_coro_sync


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
        self.files = SimpleNamespace(
            create=self._create_file_async,
            delete=self._delete_file_async,
        )
        self.response_calls: list[dict[str, Any]] = []
        self.deleted_files: list[str] = []
        FakeAsyncOpenAIClient.instances.append(self)

    async def _create_response(self, **kwargs: Any) -> Any:
        self.response_calls.append(kwargs)
        return SimpleNamespace(output_text=" openai async text ")

    async def _create_file_async(self, file: Any, purpose: str) -> Any:
        return SimpleNamespace(id="async_file_456")

    async def _delete_file_async(self, file_id: str) -> None:
        self.deleted_files.append(file_id)


def _make_engine(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> OpenAIEngine:
    monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
    monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)
    return OpenAIEngine(model="gpt-4o-mini", **kwargs)


class TestOpenAIImmutability:
    def test_inline_cutoff_chars_is_read_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch, inline_cutoff_chars=500)
        assert engine.inline_cutoff_chars == 500
        with pytest.raises(AttributeError):
            engine.inline_cutoff_chars = 999  # type: ignore[misc]


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
            api_key="secret",
            timeout_seconds=12.0,
        )

        assert len(FakeOpenAIClient.instances) == 1
        assert len(FakeAsyncOpenAIClient.instances) == 0

        fake = FakeOpenAIClient.instances[-1]
        assert engine.name == "openai_gpt_4o_mini"
        assert fake.kwargs["api_key"] == "secret"
        assert fake.kwargs["timeout"] == 12.0
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

        assert injected.response_calls
        assert engine._extract_text(response) == " openai async text "

    def test_openai_payload_helpers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
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
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(
            model="o3-mini",
            temperature=0.7,
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
        engine = _make_engine(monkeypatch)
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
        assert data["max_output_tokens"] is None
        assert data["reasoning"] is None
        assert data["truncation"] is None
        assert data["inline_cutoff_chars"] == 123
        assert "secret" not in str(data)

    def test_call_provider_async_with_async_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeAsyncOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        injected = FakeAsyncOpenAIClient()
        engine = OpenAIEngine(model="gpt-4o-mini", client=injected)

        response = run_coro_sync(engine._call_provider_async({"blocks": [], "instructions": None}))

        assert injected.response_calls
        assert engine._extract_text(response) == " openai async text "

    def test_call_provider_async_with_sync_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(model="gpt-4o-mini")
        response = run_coro_sync(engine._call_provider_async({"blocks": [], "instructions": None}))

        assert engine._extract_text(response) == " openai text "


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
        assert result.response_tokens == 5
        assert result.cached_tokens == 2

    def test_output_tokens_details_none_defaults_reasoning_to_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = self._engine(monkeypatch)
        response = SimpleNamespace(usage=self._usage(
            output_tokens_details=None,
            input_tokens_details=SimpleNamespace(cached_tokens=1),
        ))

        result = engine._extract_token_usage(response)

        assert result.reasoning_tokens == 0
        assert result.response_tokens == 5
        assert result.cached_tokens == 1

    def test_input_tokens_details_none_defaults_cached_to_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = self._engine(monkeypatch)
        response = SimpleNamespace(usage=self._usage(
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

    def test_negative_response_tokens_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = self._engine(monkeypatch)
        response = SimpleNamespace(usage=self._usage(
            output_tokens=2,
            output_tokens_details=SimpleNamespace(reasoning_tokens=5),
        ))

        with pytest.raises(LLMEngineError, match="negative"):
            engine._extract_token_usage(response)


class TestOpenAIAttachments:
    def test_validate_attachment_path_rejects_illegal_ext(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "payload.zip"
        path.write_bytes(b"data")

        with pytest.raises(LLMEngineError, match="not permitted"):
            engine._validate_attachment_path(str(path))

    def test_validate_attachment_path_rejects_unsupported_ext(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "doc.docx"
        path.write_bytes(b"data")

        with pytest.raises(LLMEngineError, match="not supported"):
            engine._validate_attachment_path(str(path))

    def test_validate_attachment_path_passes_for_txt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "notes.txt"
        path.write_text("hello")
        engine._validate_attachment_path(str(path))  # no exception

    def test_prepare_attachment_inlines_text_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "notes.txt"
        path.write_text("Hello world!")

        meta = engine._prepare_attachment(str(path))

        assert meta["kind"] == "text"
        assert meta["inlined"] is True
        assert "Hello world!" in meta["inlined_text"]

    def test_prepare_attachment_truncates_long_text(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch, inline_cutoff_chars=10)
        path = tmp_path / "long.txt"
        path.write_text("A" * 50)

        meta = engine._prepare_attachment(str(path))

        assert meta["kind"] == "text"
        assert "truncated" in meta["inlined_text"]
        assert len(meta["inlined_text"]) < 100

    def test_prepare_attachment_pdf_uploads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        FakeOpenAIClient.instances.clear()
        engine = _make_engine(monkeypatch)
        path = tmp_path / "report.pdf"
        path.write_bytes(b"%PDF fake")

        meta = engine._prepare_attachment(str(path))

        assert meta["kind"] == "pdf"
        assert meta["uploaded"] is True
        assert meta["file_id"] == "file_123"

    def test_prepare_attachment_image_uploads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        FakeOpenAIClient.instances.clear()
        engine = _make_engine(monkeypatch)
        path = tmp_path / "photo.png"
        path.write_bytes(b"\x89PNG")

        meta = engine._prepare_attachment(str(path))

        assert meta["kind"] == "image"
        assert meta["uploaded"] is True
        assert meta["file_id"] == "file_123"

    def test_on_detach_calls_delete_with_file_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        engine = _make_engine(monkeypatch)
        fake = FakeOpenAIClient.instances[-1]

        engine._on_detach({"file_id": "file_abc"})

        assert fake.deleted_files == ["file_abc"]

    def test_on_detach_noop_when_no_file_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        engine = _make_engine(monkeypatch)
        fake = FakeOpenAIClient.instances[-1]

        engine._on_detach({})
        engine._on_detach({"kind": "text"})

        assert fake.deleted_files == []

    def test_on_detach_with_async_client_uses_run_coro_sync(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeAsyncOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        injected = FakeAsyncOpenAIClient()
        engine = OpenAIEngine(model="gpt-4o-mini", client=injected)
        engine._on_detach({"file_id": "file_xyz"})

        assert "file_xyz" in injected.deleted_files

    def test_build_provider_payload_with_inlined_attachment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        messages = [{"role": "user", "content": "Q?"}]
        attachments = {
            "notes.txt": {"kind": "text", "inlined": True, "inlined_text": "important note"},
        }

        payload = engine._build_provider_payload(messages, attachments)
        user_block = next(b for b in payload["blocks"] if b["role"] == "user")
        texts = [p["text"] for p in user_block["content"] if p.get("type") == "input_text"]

        assert any("important note" in t for t in texts)

    def test_build_provider_payload_with_uploaded_pdf(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        messages = [{"role": "user", "content": "Q?"}]
        attachments = {
            "doc.pdf": {"kind": "pdf", "uploaded": True, "file_id": "file_pdf"},
        }

        payload = engine._build_provider_payload(messages, attachments)
        user_block = next(b for b in payload["blocks"] if b["role"] == "user")
        file_parts = [p for p in user_block["content"] if p.get("type") == "input_file"]

        assert file_parts == [{"type": "input_file", "file_id": "file_pdf"}]

    def test_build_provider_payload_with_uploaded_image(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        messages = [{"role": "user", "content": "Q?"}]
        attachments = {
            "photo.png": {"kind": "image", "uploaded": True, "file_id": "file_img"},
        }

        payload = engine._build_provider_payload(messages, attachments)
        user_block = next(b for b in payload["blocks"] if b["role"] == "user")
        img_parts = [p for p in user_block["content"] if p.get("type") == "input_image"]

        assert img_parts == [{"type": "input_image", "file_id": "file_img"}]
