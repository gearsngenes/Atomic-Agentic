from __future__ import annotations

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

        payload = engine._build_provider_payload(msgs)

        assert payload["model"] == "anthropic/claude-haiku-4-5-20251001"

    def test_build_payload_model_string_reflects_mutation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini", provider="openai")
        engine.provider = "azure"
        engine.model = "gpt-4o"
        msgs = [{"role": "user", "content": "hi"}]

        payload = engine._build_provider_payload(msgs)

        assert payload["model"] == "azure/gpt-4o"

    def test_build_payload_string_content_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        msgs = [{"role": "user", "content": "hello"}]

        payload = engine._build_provider_payload(msgs)

        assert payload["messages"][0]["content"] == "hello"

    def test_build_payload_always_includes_drop_params_and_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini", drop_params=True, timeout_seconds=45.0)
        msgs = [{"role": "user", "content": "hi"}]

        payload = engine._build_provider_payload(msgs)

        assert payload["drop_params"] is True
        assert payload["timeout"] == 45.0

    def test_build_payload_optional_knobs_omitted_when_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")  # all None defaults
        msgs = [{"role": "user", "content": "hi"}]

        payload = engine._build_provider_payload(msgs)

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

        payload = engine._build_provider_payload(msgs)

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

        payload = engine._build_provider_payload(msgs)

        assert payload["organization"] == "org-123"

    # ── _extract_text ────────────────────────────────────────────────────────

    def test_extract_text_returns_message_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        response = _fake_response(text="Hello world")
        assert engine._extract_result(response, requested_structured=False) == "Hello world"

    def test_extract_text_none_content_returns_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=_fake_usage(),
        )
        assert engine._extract_result(response, requested_structured=False) == ""

    # ── structured-output payload/extraction ────────────────────────────────

    def test_build_provider_payload_raises_when_structured_output_unsupported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeLiteLLM()
        fake.supports_response_schema = lambda **kwargs: False
        monkeypatch.setattr(engine_module, "litellm", fake)
        engine = LiteLLMEngine(model="gpt-4o-mini", provider="openai")

        with pytest.raises(LLMEngineError, match="structured output"):
            engine._build_provider_payload(
                [{"role": "user", "content": "hi"}], {"type": "object"}
            )

    def test_build_provider_payload_sets_response_format_when_supported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeLiteLLM()
        fake.supports_response_schema = lambda **kwargs: True
        monkeypatch.setattr(engine_module, "litellm", fake)
        engine = LiteLLMEngine(model="gpt-4o-mini", provider="openai")
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}

        payload = engine._build_provider_payload(
            [{"role": "user", "content": "hi"}], schema
        )

        assert payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {"schema": schema, "name": "output_structure", "strict": True},
        }

    def test_build_provider_payload_omits_response_format_when_not_structured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeLiteLLM()
        monkeypatch.setattr(engine_module, "litellm", fake)
        engine = LiteLLMEngine(model="gpt-4o-mini", provider="openai")

        payload = engine._build_provider_payload([{"role": "user", "content": "hi"}])

        assert "response_format" not in payload

    def test_extract_result_requested_parses_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        response = _fake_response(text='{"a": 1}')

        assert engine._extract_result(response, requested_structured=True) == {"a": 1}

    def test_extract_result_requested_parse_failure_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        response = _fake_response(text="not json")

        assert engine._extract_result(response, requested_structured=True) == "not json"

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
        assert usage.response_tokens == 20

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
        # response_tokens is plain subtraction: generated_tokens (20) - reasoning (15)
        assert usage.response_tokens == 5

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
        for removed in ("inline_cutoff_chars", "max_retries", "attachments"):
            assert removed not in d
        assert "api_key" not in d

    def test_timeout_default_is_600_until_pass_9(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        payload = engine._build_provider_payload([{"role": "user", "content": "hi"}])
        assert payload["timeout"] == 600.0
        assert not hasattr(engine, "timeout_seconds")

    # ── Removed surface ───────────────────────────────────────────────────────

    def test_attachment_and_retry_surface_is_gone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")

        for name in ("attach", "detach", "clear_attachments", "attachments",
                     "_prepare_attachment", "_on_detach", "_validate_attachment_path",
                     "inline_cutoff_chars", "_inline_cutoff_chars",
                     "_should_retry", "_max_retries"):
            assert not hasattr(engine, name)

    def test_max_retries_is_forwarded_to_litellm_through_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini", max_retries=0)

        payload = engine._build_provider_payload([{"role": "user", "content": "hi"}])

        assert payload["max_retries"] == 0
