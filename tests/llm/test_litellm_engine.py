from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import pytest

import atomic_agentic.llm.litellm_engine as engine_module
from atomic_agentic.llm.litellm_engine import LiteLLMEngine
from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.models.results.llm import LiteLLMTokenUsage, RemoteLLMModelData
from atomic_agentic.utils.core import run_coro_sync


# ── Fake response helpers ──────────────────────────────────────────────────────

class _FakePromptTokensDetails:
    """
    Mirrors litellm's real ``PromptTokensDetailsWrapper`` behavior: when
    ``cache_creation_tokens`` is ``None``, the attribute is deleted from the
    instance entirely (not just set to ``None``) — plain attribute access
    raises ``AttributeError``, exactly like the real class.
    """

    def __init__(
        self, cached_tokens: int | None, cache_creation_tokens: int | None
    ) -> None:
        self.cached_tokens = cached_tokens
        if cache_creation_tokens is not None:
            self.cache_creation_tokens = cache_creation_tokens


def _fake_usage(
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    cached_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> Any:
    prompt_details = None
    if cached_tokens is not None or cache_creation_tokens is not None:
        prompt_details = _FakePromptTokensDetails(
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
    completion_details = None
    if reasoning_tokens is not None:
        completion_details = SimpleNamespace(reasoning_tokens=reasoning_tokens)
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=prompt_details,
        completion_tokens_details=completion_details,
    )


def _fake_response(text: str = "hello", **usage_kwargs: Any) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=_fake_usage(**usage_kwargs),
    )


class FakeLiteLLM:
    """Stand-in for the `litellm` module — fakes `completion`/`acompletion`
    directly since there is no client object to fake."""

    def __init__(self) -> None:
        self.completion_calls: list[dict[str, Any]] = []
        self.acompletion_calls: list[dict[str, Any]] = []

    def completion(self, **kwargs: Any) -> Any:
        self.completion_calls.append(kwargs)
        return _fake_response()

    async def acompletion(self, **kwargs: Any) -> Any:
        self.acompletion_calls.append(kwargs)
        return _fake_response(text="async hello")


class TestLiteLLMEngine:

    # ── SDK guard ───────────────────────────────────────────────────────────

    def test_missing_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "litellm", None)
        with pytest.raises(LLMEngineError, match="litellm"):
            LiteLLMEngine(model="gpt-4o-mini")

    # ── Constructor — naming, defaults, param storage ──────────────────────

    def test_constructor_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        assert engine.provider == "litellm_proxy"
        assert engine.model == "gpt-4o-mini"
        assert engine.drop_params is False

    def test_constructor_name_auto_derivation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini", provider="openai")
        assert engine.name == "openai_gpt_4o_mini"

    def test_constructor_custom_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini", name="my_engine")
        assert engine.name == "my_engine"

    def test_constructor_stores_generation_knobs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(
            model="gpt-4o-mini",
            provider="openai",
            temperature=0.5,
            top_p=0.9,
            max_tokens=100,
            max_completion_tokens=200,
            stop=["END"],
            seed=42,
            reasoning_effort="medium",
            thinking={"type": "enabled"},
            drop_params=True,
            api_key="test-key",
            base_url="https://example.com",
            api_version="2024-01-01",
        )
        assert engine.temperature == 0.5
        assert engine.top_p == 0.9
        assert engine.max_tokens == 100
        assert engine.max_completion_tokens == 200
        assert engine.stop == ["END"]
        assert engine.seed == 42
        assert engine.reasoning_effort == "medium"
        assert engine.thinking == {"type": "enabled"}
        assert engine.drop_params is True
        assert engine.api_key == "test-key"
        assert engine.base_url == "https://example.com"
        assert engine.api_version == "2024-01-01"

    # ── _call_provider / _call_provider_async ──────────────────────────────

    def test_call_provider_calls_litellm_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeLiteLLM()
        monkeypatch.setattr(engine_module, "litellm", fake)
        engine = LiteLLMEngine(model="gpt-4o-mini", provider="openai")

        response = engine._call_provider({"model": "openai/gpt-4o-mini", "messages": []})

        assert fake.completion_calls
        assert response.choices[0].message.content == "hello"

    def test_call_provider_async_calls_litellm_acompletion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeLiteLLM()
        monkeypatch.setattr(engine_module, "litellm", fake)
        engine = LiteLLMEngine(model="gpt-4o-mini", provider="openai")

        response = run_coro_sync(
            engine._call_provider_async({"model": "openai/gpt-4o-mini", "messages": []})
        )

        assert fake.acompletion_calls
        assert response.choices[0].message.content == "async hello"

    # ── _build_provider_payload ─────────────────────────────────────────────

    def test_build_payload_composes_model_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="claude-haiku-4-5-20251001", provider="anthropic")
        msgs = [{"role": "user", "content": "hi"}]

        payload = engine._build_provider_payload(msgs, {})

        assert payload["model"] == "anthropic/claude-haiku-4-5-20251001"

    def test_build_payload_model_string_reflects_mutation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini", provider="openai")
        engine.provider = "azure"
        engine.model = "gpt-4o"
        msgs = [{"role": "user", "content": "hi"}]

        payload = engine._build_provider_payload(msgs, {})

        assert payload["model"] == "azure/gpt-4o"

    def test_build_payload_string_content_passthrough_without_attachments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        msgs = [{"role": "user", "content": "hello"}]

        payload = engine._build_provider_payload(msgs, {})

        assert payload["messages"][0]["content"] == "hello"

    def test_build_payload_always_includes_drop_params_and_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini", drop_params=True, timeout_seconds=45.0)
        msgs = [{"role": "user", "content": "hi"}]

        payload = engine._build_provider_payload(msgs, {})

        assert payload["drop_params"] is True
        assert payload["timeout"] == 45.0

    def test_build_payload_optional_knobs_omitted_when_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")  # all None defaults
        msgs = [{"role": "user", "content": "hi"}]

        payload = engine._build_provider_payload(msgs, {})

        for key in (
            "temperature", "top_p", "max_tokens", "max_completion_tokens",
            "stop", "seed", "reasoning_effort", "thinking",
            "api_key", "base_url", "api_version",
        ):
            assert key not in payload

    def test_build_payload_optional_knobs_included_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(
            model="gpt-4o-mini",
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            max_completion_tokens=200,
            stop="STOP",
            seed=7,
            reasoning_effort="low",
            thinking={"type": "enabled"},
            api_key="k",
            base_url="https://x",
            api_version="v1",
        )
        msgs = [{"role": "user", "content": "hi"}]

        payload = engine._build_provider_payload(msgs, {})

        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9
        assert payload["max_tokens"] == 100
        assert payload["max_completion_tokens"] == 200
        assert payload["stop"] == "STOP"
        assert payload["seed"] == 7
        assert payload["reasoning_effort"] == "low"
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["api_key"] == "k"
        assert payload["base_url"] == "https://x"
        assert payload["api_version"] == "v1"

    def test_build_payload_litellm_kwargs_forwarded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini", organization="org-123")
        msgs = [{"role": "user", "content": "hi"}]

        payload = engine._build_provider_payload(msgs, {})

        assert payload["organization"] == "org-123"

    def test_attachment_blocks_prepended_convert_string_content_to_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        fake_block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        msgs = [{"role": "user", "content": "Describe this."}]

        payload = engine._build_provider_payload(msgs, {"img.png": fake_block})

        first_content = payload["messages"][0]["content"]
        assert first_content[0] is fake_block
        assert first_content[1] == {"type": "text", "text": "Describe this."}

    def test_attachment_inserts_bare_user_turn_when_none_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        fake_block = {"type": "text", "text": "notes"}
        msgs = [{"role": "system", "content": "You are helpful."}]

        payload = engine._build_provider_payload(msgs, {"notes.txt": fake_block})

        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == [fake_block]

    # ── _extract_text ────────────────────────────────────────────────────────

    def test_extract_text_returns_message_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        response = _fake_response(text="Hello world")
        assert engine._extract_text(response) == "Hello world"

    def test_extract_text_none_content_returns_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=_fake_usage(),
        )
        assert engine._extract_text(response) == ""

    # ── _extract_token_usage ─────────────────────────────────────────────────

    def test_extract_token_usage_basic_mapping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        response = _fake_response(prompt_tokens=10, completion_tokens=20)

        usage = engine._extract_token_usage(response)

        assert isinstance(usage, LiteLLMTokenUsage)
        assert usage.input_tokens == 10
        assert usage.generated_tokens == 20
        assert usage.total_tokens == 30

    def test_extract_token_usage_all_none_when_details_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        response = _fake_response()  # both *_details are None

        usage = engine._extract_token_usage(response)

        assert usage.cached_tokens is None
        assert usage.cache_creation_tokens is None
        assert usage.reasoning_tokens is None

    def test_extract_token_usage_missing_usage_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        response = SimpleNamespace(choices=[])  # no usage attribute at all

        with pytest.raises(LLMEngineError, match="usage"):
            engine._extract_token_usage(response)

    def test_extract_token_usage_cache_creation_tokens_deleted_attribute(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Regression test: PromptTokensDetailsWrapper deletes
        cache_creation_tokens from the instance (not just sets it to None)
        when there is no cache-creation activity — e.g. a plain OpenAI
        response. _extract_token_usage must not crash with AttributeError.
        """
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        response = _fake_response(cached_tokens=0)  # cache_creation_tokens omitted

        usage = engine._extract_token_usage(response)

        assert usage.cache_creation_tokens is None

    def test_extract_token_usage_cache_and_reasoning_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        response = _fake_response(
            cached_tokens=50, cache_creation_tokens=25, reasoning_tokens=15,
        )

        usage = engine._extract_token_usage(response)

        assert usage.cached_tokens == 50
        assert usage.cache_creation_tokens == 25
        assert usage.reasoning_tokens == 15

    # ── _get_model_data ──────────────────────────────────────────────────────

    def test_get_model_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini", provider="openai")
        md = engine._get_model_data()
        assert isinstance(md, RemoteLLMModelData)
        assert md.provider == "openai"
        assert md.model_name == "gpt-4o-mini"

    # ── to_dict ──────────────────────────────────────────────────────────────

    def test_to_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(
            model="gpt-4o-mini", provider="openai", temperature=0.3, api_key="secret",
        )
        d = engine.to_dict()
        assert d["type"] == "LiteLLMEngine"
        assert d["provider"] == "openai"
        assert d["model"] == "gpt-4o-mini"
        assert d["temperature"] == 0.3
        assert "api_key" not in d

    # ── Attachment pipeline ──────────────────────────────────────────────────

    def test_prepare_attachment_image(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n")

        block = engine._prepare_attachment(str(img))

        assert block["type"] == "image_url"
        expected = base64.b64encode(b"\x89PNG\r\n").decode()
        assert block["image_url"]["url"] == f"data:image/png;base64,{expected}"

    def test_prepare_attachment_pdf_non_mistral(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini", provider="openai")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        block = engine._prepare_attachment(str(pdf))

        assert block["type"] == "file"
        expected = base64.b64encode(b"%PDF-1.4").decode()
        assert block["file"]["file_data"] == f"data:application/pdf;base64,{expected}"

    def test_prepare_attachment_pdf_mistral_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="mistral-small-latest", provider="mistral")
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with pytest.raises(LLMEngineError, match="mistral"):
            engine._prepare_attachment(str(pdf))

    def test_prepare_attachment_text_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        txt = tmp_path / "notes.txt"
        txt.write_text("Hello notes", encoding="utf-8")

        block = engine._prepare_attachment(str(txt))

        assert block == {"type": "text", "text": "Hello notes"}

    def test_on_detach_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        engine._on_detach({"type": "text", "text": "x"})  # must not raise

    def test_validate_attachment_rejects_illegal_ext(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        bad = tmp_path / "model.exe"
        bad.write_bytes(b"MZ")

        with pytest.raises(LLMEngineError):
            engine._validate_attachment_path(str(bad))

    def test_validate_attachment_rejects_unsupported_ext(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        docx = tmp_path / "report.docx"
        docx.write_bytes(b"PK")

        with pytest.raises(LLMEngineError):
            engine._validate_attachment_path(str(docx))
