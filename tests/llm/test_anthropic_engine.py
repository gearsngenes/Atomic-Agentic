from __future__ import annotations

import base64
import pathlib
from types import SimpleNamespace
from typing import Any

import pytest

import atomic_agentic.llm.anthropic_engine as engine_module
from atomic_agentic.llm.anthropic_engine import AnthropicEngine
from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.models.results.llm import AnthropicTokenUsage, RemoteLLMModelData
from atomic_agentic.utils.core import run_coro_sync


# ── Fake response helpers ──────────────────────────────────────────────────────

def _fake_usage(
    input_tokens: int = 10,
    output_tokens: int = 20,
    cache_creation: int = 0,
    cache_read: int = 0,
    thinking_tokens: int = 0,
) -> Any:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
        output_tokens_details={"thinking_tokens": thinking_tokens},
    )


def _fake_response(text: str = "hello", **usage_kwargs: Any) -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        model="claude-opus-4-8",
        usage=_fake_usage(**usage_kwargs),
    )


# ── Fake sync client ───────────────────────────────────────────────────────────

class FakeAnthropicClient:
    instances: list["FakeAnthropicClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.create_calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)
        FakeAnthropicClient.instances.append(self)

    def _create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return _fake_response()


# ── Fake async client ──────────────────────────────────────────────────────────

class FakeAsyncAnthropicClient:
    instances: list["FakeAsyncAnthropicClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.create_calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)
        FakeAsyncAnthropicClient.instances.append(self)

    async def _create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return _fake_response(text="async hello")


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestAnthropicEngine:

    # -- SDK guard -------------------------------------------------------------

    def test_missing_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", None)
        with pytest.raises(LLMEngineError, match="anthropic"):
            AnthropicEngine(model="claude-opus-4-8")

    # -- Constructor: client build ---------------------------------------------

    def test_constructor_builds_sync_client_from_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeAnthropicClient.instances.clear()
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)

        engine = AnthropicEngine(
            model="claude-opus-4-8",
            api_key="test-key",
            timeout_seconds=30.0,
        )

        fake = FakeAnthropicClient.instances[-1]
        assert engine._client is fake
        assert fake.kwargs["api_key"] == "test-key"
        assert fake.kwargs["timeout"] == 30.0

    def test_constructor_client_injection_skips_build(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeAnthropicClient.instances.clear()
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        injected = FakeAnthropicClient()
        FakeAnthropicClient.instances.clear()

        engine = AnthropicEngine(model="claude-opus-4-8", client=injected)

        assert FakeAnthropicClient.instances == []
        assert engine._client is injected

    # -- Constructor: name + params --------------------------------------------

    def test_constructor_name_sanitization(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        assert engine.name == "anthropic_claude_opus_4_8"

    def test_constructor_custom_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8", name="my_engine")
        assert engine.name == "my_engine"

    def test_constructor_stores_request_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(
            model="claude-sonnet-4-6",
            max_output_tokens=1024,
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            stop_sequences=["STOP"],
            thinking_config={"type": "adaptive"},
        )
        assert engine.model == "claude-sonnet-4-6"
        assert engine.max_output_tokens == 1024
        assert engine.temperature == 0.5
        assert engine.top_p == 0.9
        assert engine.top_k == 40
        assert engine.stop_sequences == ["STOP"]
        assert engine.thinking_config == {"type": "adaptive"}

    # -- _call_provider routing ------------------------------------------------

    def test_call_provider_sync_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        fake: FakeAnthropicClient = engine._client  # type: ignore[assignment]
        payload = {
            "model": "claude-opus-4-8",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }

        engine._call_provider(payload)

        assert fake.create_calls
        assert fake.create_calls[-1]["model"] == "claude-opus-4-8"

    def test_call_provider_with_async_client_uses_run_coro_sync(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        monkeypatch.setattr(engine_module, "AsyncAnthropic", FakeAsyncAnthropicClient)
        injected = FakeAsyncAnthropicClient()
        engine = AnthropicEngine(model="claude-opus-4-8", client=injected)
        payload = {
            "model": "claude-opus-4-8",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }

        result = engine._call_provider(payload)

        assert injected.create_calls
        assert result.content[0].text == "async hello"

    def test_call_provider_async_with_async_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        monkeypatch.setattr(engine_module, "AsyncAnthropic", FakeAsyncAnthropicClient)
        injected = FakeAsyncAnthropicClient()
        engine = AnthropicEngine(model="claude-opus-4-8", client=injected)
        payload = {
            "model": "claude-opus-4-8",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }

        result = run_coro_sync(engine._call_provider_async(payload))

        assert injected.create_calls
        assert result.content[0].text == "async hello"

    def test_call_provider_async_with_sync_client_uses_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        fake: FakeAnthropicClient = engine._client  # type: ignore[assignment]
        payload = {
            "model": "claude-opus-4-8",
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }

        result = run_coro_sync(engine._call_provider_async(payload))

        assert fake.create_calls
        assert result.content[0].text == "hello"

    # -- _build_provider_payload -----------------------------------------------

    def test_build_payload_required_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8", max_output_tokens=512)
        msgs = [{"role": "user", "content": "hello"}]

        payload = engine._build_provider_payload(msgs, {})

        assert payload["model"] == "claude-opus-4-8"
        assert payload["max_tokens"] == 512
        assert payload["messages"][0]["role"] == "user"

    def test_build_payload_system_extracted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        payload = engine._build_provider_payload(msgs, {})

        assert payload["system"] == "You are helpful."
        roles = [m["role"] for m in payload["messages"]]
        assert "system" not in roles

    def test_build_payload_multiple_system_messages_concatenated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        msgs = [
            {"role": "system", "content": "Part A."},
            {"role": "system", "content": "Part B."},
            {"role": "user", "content": "Hi"},
        ]

        payload = engine._build_provider_payload(msgs, {})

        assert payload["system"] == "Part A.\n\nPart B."

    def test_build_payload_no_system_omits_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        msgs = [{"role": "user", "content": "hello"}]

        payload = engine._build_provider_payload(msgs, {})

        assert "system" not in payload

    def test_build_payload_optional_params_omitted_when_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        msgs = [{"role": "user", "content": "hi"}]

        payload = engine._build_provider_payload(msgs, {})

        for key in ("temperature", "top_p", "top_k", "stop_sequences", "thinking"):
            assert key not in payload

    def test_build_payload_optional_params_included_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(
            model="claude-opus-4-8",
            temperature=0.7,
            top_p=0.9,
            top_k=20,
            stop_sequences=["END"],
            thinking_config={"type": "adaptive"},
        )
        msgs = [{"role": "user", "content": "hi"}]

        payload = engine._build_provider_payload(msgs, {})

        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9
        assert payload["top_k"] == 20
        assert payload["stop_sequences"] == ["END"]
        assert payload["thinking"] == {"type": "adaptive"}

    # -- _extract_text ---------------------------------------------------------

    def test_extract_text_from_single_text_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Hello world")],
            usage=_fake_usage(),
        )
        assert engine._extract_text(response) == "Hello world"

    def test_extract_text_skips_thinking_blocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="secret thoughts"),
                SimpleNamespace(type="text", text="Answer here"),
            ],
            usage=_fake_usage(),
        )
        assert engine._extract_text(response) == "Answer here"

    def test_extract_text_empty_content_returns_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        response = SimpleNamespace(content=[], usage=_fake_usage())
        assert engine._extract_text(response) == ""

    # -- _extract_token_usage --------------------------------------------------

    def test_extract_token_usage_basic_mapping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        response = SimpleNamespace(
            content=[],
            usage=_fake_usage(input_tokens=10, output_tokens=20),
        )
        usage = engine._extract_token_usage(response)
        assert isinstance(usage, AnthropicTokenUsage)
        assert usage.input_tokens == 10
        assert usage.generated_tokens == 20
        assert usage.total_tokens == 30

    def test_extract_token_usage_derives_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        response = SimpleNamespace(
            content=[],
            usage=_fake_usage(input_tokens=7, output_tokens=13),
        )
        usage = engine._extract_token_usage(response)
        assert usage.total_tokens == 20

    def test_extract_token_usage_cache_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        response = SimpleNamespace(
            content=[],
            usage=_fake_usage(cache_creation=50, cache_read=100),
        )
        usage = engine._extract_token_usage(response)
        assert usage.cache_creation_input_tokens == 50
        assert usage.cache_read_input_tokens == 100

    def test_extract_token_usage_thinking_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        response = SimpleNamespace(
            content=[],
            usage=_fake_usage(output_tokens=500, thinking_tokens=300),
        )
        usage = engine._extract_token_usage(response)
        assert usage.thinking_tokens == 300

    # -- _get_model_data -------------------------------------------------------

    def test_get_model_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        md = engine._get_model_data()
        assert isinstance(md, RemoteLLMModelData)
        assert md.provider == "anthropic"
        assert md.model_name == "claude-opus-4-8"

    # -- to_dict ---------------------------------------------------------------

    def test_to_dict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(
            model="claude-opus-4-8",
            max_output_tokens=1024,
            temperature=0.3,
            stop_sequences=["END"],
        )
        d = engine.to_dict()
        assert d["type"] == "AnthropicEngine"
        assert d["model"] == "claude-opus-4-8"
        assert d["max_output_tokens"] == 1024
        assert d["temperature"] == 0.3
        assert d["stop_sequences"] == ["END"]
        assert "client" not in d

    # -- Attachment pipeline ---------------------------------------------------

    def test_prepare_attachment_image(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n")

        block = engine._prepare_attachment(str(img))

        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"
        assert block["source"]["data"] == base64.b64encode(b"\x89PNG\r\n").decode()

    def test_prepare_attachment_pdf(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        block = engine._prepare_attachment(str(pdf))

        assert block["type"] == "document"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "application/pdf"
        assert block["source"]["data"] == base64.b64encode(b"%PDF-1.4").decode()

    def test_prepare_attachment_text_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        txt = tmp_path / "notes.txt"
        txt.write_text("Hello notes", encoding="utf-8")

        block = engine._prepare_attachment(str(txt))

        assert block["type"] == "document"
        assert block["source"]["type"] == "text"
        assert block["source"]["data"] == "Hello notes"

    def test_attachment_blocks_prepended_to_first_user_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        fake_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc"},
        }
        msgs = [{"role": "user", "content": "Describe this."}]

        payload = engine._build_provider_payload(msgs, {"img.png": fake_block})

        first_content = payload["messages"][0]["content"]
        assert first_content[0] is fake_block
        assert first_content[1] == {"type": "text", "text": "Describe this."}

    def test_attachment_inserts_bare_user_turn_when_none_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        fake_block = {"type": "document", "source": {"type": "text", "data": "notes"}}
        msgs = [{"role": "system", "content": "You are helpful."}]

        payload = engine._build_provider_payload(msgs, {"notes.txt": fake_block})

        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == [fake_block]

    def test_on_detach_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        engine._on_detach({"type": "image", "source": {}})  # must not raise

    def test_validate_attachment_rejects_illegal_ext(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        bad = tmp_path / "model.exe"
        bad.write_bytes(b"MZ")

        with pytest.raises(LLMEngineError):
            engine._validate_attachment_path(str(bad))

    def test_validate_attachment_rejects_unsupported_ext(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        monkeypatch.setattr(engine_module, "Anthropic", FakeAnthropicClient)
        engine = AnthropicEngine(model="claude-opus-4-8")
        docx = tmp_path / "report.docx"
        docx.write_bytes(b"PK")

        with pytest.raises(LLMEngineError):
            engine._validate_attachment_path(str(docx))
