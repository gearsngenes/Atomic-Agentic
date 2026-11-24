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



class FakeRouter:
    """Stand-in for ``litellm.Router``: records completion/acompletion calls."""

    def __init__(self) -> None:
        self.completion_calls: list[dict[str, Any]] = []
        self.acompletion_calls: list[dict[str, Any]] = []

    def completion(self, **kwargs: Any) -> Any:
        self.completion_calls.append(kwargs)
        return _fake_response(text="router hello")

    async def acompletion(self, **kwargs: Any) -> Any:
        self.acompletion_calls.append(kwargs)
        return _fake_response(text="router async hello")


class FakeLiteLLM:
    """Stand-in for the `litellm` module: `completion`/`acompletion`, a
    `Router` class, `get_llm_provider` and the `suppress_debug_info` flag."""

    Router = FakeRouter

    def __init__(self, bad_models: tuple[str, ...] = ()) -> None:
        self.completion_calls: list[dict[str, Any]] = []
        self.acompletion_calls: list[dict[str, Any]] = []
        self.suppress_debug_info = False
        self.seen_suppress: list[bool] = []
        self._bad_models = bad_models

    def completion(self, **kwargs: Any) -> Any:
        self.completion_calls.append(kwargs)
        return _fake_response()

    async def acompletion(self, **kwargs: Any) -> Any:
        self.acompletion_calls.append(kwargs)
        return _fake_response(text="async hello")

    def get_llm_provider(self, model: str) -> tuple[str, str, None, None]:
        self.seen_suppress.append(self.suppress_debug_info)
        if model in self._bad_models:
            raise ValueError("unknown model")
        provider = model.split("/", 1)[0] if "/" in model else "openai"
        return (model, provider, None, None)


MSGS = [{"role": "user", "content": "hi"}]


class TestLiteLLMEngine:

    # ── SDK guard ───────────────────────────────────────────────────────────

    def test_missing_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "litellm", None)
        with pytest.raises(LLMEngineError, match="litellm"):
            LiteLLMEngine(model="gpt-4o-mini")

    # ── Constructor ─────────────────────────────────────────────────────────

    def test_constructor_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="openai/gpt-4o-mini")
        assert engine.name == "LiteLLMEngine"
        assert engine.description.startswith("LiteLLM provider-agnostic language model engine.")
        assert engine.model == "openai/gpt-4o-mini"
        assert engine.runtime_parameters == {}

    def test_constructor_custom_name_is_kept(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini", name="my_engine")
        assert engine.name == "my_engine"

    def test_wrong_client_type_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        with pytest.raises(LLMEngineError, match="must be a litellm.Router"):
            LiteLLMEngine(model="grp", client=object())

    def test_router_with_kwargs_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        with pytest.raises(LLMEngineError, match="either 'client' or call kwargs"):
            LiteLLMEngine(model="grp", client=FakeRouter(), api_key="k")

    def test_removed_arguments_flow_into_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(
            model="gpt-4o-mini", provider="openai", timeout_seconds=5.0, temperature=0.5,
        )
        assert engine._kwargs == {
            "provider": "openai", "timeout_seconds": 5.0, "temperature": 0.5,
        }

    # ── Call routing ────────────────────────────────────────────────────────

    def test_no_client_calls_litellm_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeLiteLLM()
        monkeypatch.setattr(engine_module, "litellm", fake)
        engine = LiteLLMEngine(model="gpt-4o-mini")
        payload = {"model": "gpt-4o-mini", "messages": []}

        response = engine._call_provider(payload)

        assert fake.completion_calls == [payload]
        assert response.choices[0].message.content == "hello"

    def test_no_client_calls_litellm_acompletion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeLiteLLM()
        monkeypatch.setattr(engine_module, "litellm", fake)
        engine = LiteLLMEngine(model="gpt-4o-mini")
        payload = {"model": "gpt-4o-mini", "messages": []}

        response = run_coro_sync(engine._call_provider_async(payload))

        assert fake.acompletion_calls == [payload]
        assert response.choices[0].message.content == "async hello"

    def test_router_client_calls_router_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeLiteLLM()
        router = FakeRouter()
        monkeypatch.setattr(engine_module, "litellm", fake)
        engine = LiteLLMEngine(model="group", client=router)
        payload = {"model": "group", "messages": []}

        response = engine._call_provider(payload)

        assert router.completion_calls == [payload]
        assert fake.completion_calls == []
        assert response.choices[0].message.content == "router hello"

    def test_router_client_calls_router_acompletion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeLiteLLM()
        router = FakeRouter()
        monkeypatch.setattr(engine_module, "litellm", fake)
        engine = LiteLLMEngine(model="group", client=router)
        payload = {"model": "group", "messages": []}

        response = run_coro_sync(engine._call_provider_async(payload))

        assert router.acompletion_calls == [payload]
        assert fake.acompletion_calls == []
        assert response.choices[0].message.content == "router async hello"

    # ── _build_provider_payload ─────────────────────────────────────────────

    def test_build_payload_uses_model_as_given_and_reflects_mutation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="anthropic/claude-haiku-4-5-20251001")

        assert engine._build_provider_payload(MSGS)["model"] == "anthropic/claude-haiku-4-5-20251001"
        engine.model = "azure/gpt-4o"
        assert engine._build_provider_payload(MSGS)["model"] == "azure/gpt-4o"

    def test_build_payload_exact_default_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")

        payload = engine._build_provider_payload(MSGS)

        assert payload == {"model": "gpt-4o-mini", "messages": MSGS}
        assert "timeout" not in payload

    def test_build_payload_messages_are_role_content_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        msgs = [{"role": "user", "content": "hello", "extra": 1}]

        assert engine._build_provider_payload(msgs)["messages"] == [
            {"role": "user", "content": "hello"}
        ]

    def test_build_payload_layering(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(
            model="gpt-4o-mini", api_key="k", timeout=60, temperature=0.1, num_retries=2,
        )
        engine.update_runtime_parameters(temperature=0.9, drop_params=True, max_tokens=50)

        payload = engine._build_provider_payload(MSGS)

        assert payload["api_key"] == "k"
        assert payload["timeout"] == 60
        assert payload["num_retries"] == 2
        assert payload["temperature"] == 0.9          # runtime over kwargs
        assert payload["drop_params"] is True
        assert payload["max_tokens"] == 50

    def test_engine_owned_fields_beat_kwargs_and_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini", model_list=1)
        engine._kwargs.update(model="kw", messages="kw", response_format="kw")
        engine._runtime_parameters.update(model="rt", messages="rt", response_format="rt")
        schema = {"type": "object"}

        plain = engine._build_provider_payload(MSGS)
        structured = engine._build_provider_payload(MSGS, schema)

        assert plain["model"] == "gpt-4o-mini"
        assert plain["messages"] == MSGS
        assert structured["model"] == "gpt-4o-mini"
        assert structured["messages"] == MSGS
        assert structured["response_format"]["json_schema"]["schema"] is schema

    def test_structured_call_sets_strict_envelope_without_capability_check(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The fake has no supports_response_schema at all: no gate may call it.
        fake = FakeLiteLLM()
        assert not hasattr(fake, "supports_response_schema")
        monkeypatch.setattr(engine_module, "litellm", fake)
        engine = LiteLLMEngine(model="some/unsupported-model")
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}

        payload = engine._build_provider_payload(MSGS, schema)

        assert payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {"schema": schema, "name": "output_structure", "strict": True},
        }
        assert payload["response_format"]["json_schema"]["schema"] is schema

    def test_plain_call_omits_response_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        assert "response_format" not in engine._build_provider_payload(MSGS)

    # ── update_runtime_parameters ───────────────────────────────────────────

    @pytest.mark.parametrize("key", ["model", "messages", "response_format", "stream"])
    def test_reserved_runtime_keys_raise(
        self, monkeypatch: pytest.MonkeyPatch, key: str
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        with pytest.raises(LLMEngineError, match="managed by the engine"):
            engine.update_runtime_parameters(**{key: 1})

    def test_unknown_runtime_keys_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")
        engine.update_runtime_parameters(some_provider_knob=3)
        assert engine.runtime_parameters == {"some_provider_knob": 3}


    # ── extraction ───────────────────────────────────────────────────────────

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


    # ── structured-output extraction ─────────────────────────────────────────

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

    def test_get_model_data_provider_from_litellm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="anthropic/claude-x")
        md = engine._get_model_data()
        assert isinstance(md, RemoteLLMModelData)
        assert md.provider == "anthropic"
        assert md.model_name == "anthropic/claude-x"

    def test_get_model_data_unknown_when_litellm_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM(bad_models=("grp",)))
        engine = LiteLLMEngine(model="grp")
        md = engine._get_model_data()
        assert md.provider == "unknown"
        assert md.model_name == "grp"

    def test_get_model_data_restores_suppress_debug_info(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeLiteLLM(bad_models=("grp",))
        monkeypatch.setattr(engine_module, "litellm", fake)

        LiteLLMEngine(model="ok/model")._get_model_data()
        assert fake.suppress_debug_info is False
        LiteLLMEngine(model="grp")._get_model_data()
        assert fake.suppress_debug_info is False
        assert fake.seen_suppress == [True, True]

        fake.suppress_debug_info = True
        LiteLLMEngine(model="grp")._get_model_data()
        assert fake.suppress_debug_info is True

    def test_get_model_data_caches_the_provider_lookup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeLiteLLM()
        monkeypatch.setattr(engine_module, "litellm", fake)
        engine = LiteLLMEngine(model="anthropic/claude-x")

        first = engine._get_model_data()
        second = engine._get_model_data()

        assert first.provider == second.provider == "anthropic"
        assert fake.seen_suppress == [True]  # one lookup only
        assert fake.suppress_debug_info is False

        engine.model = "mistral/small"
        third = engine._get_model_data()

        assert third.provider == "mistral"
        assert third.model_name == "mistral/small"
        assert fake.seen_suppress == [True, True]

    def test_constructor_rejects_stream_kwarg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        with pytest.raises(LLMEngineError, match="stream"):
            LiteLLMEngine(model="gpt-4o-mini", stream=True)

    # ── to_dict ──────────────────────────────────────────────────────────────

    def test_to_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="openai/gpt-4o-mini", api_key="secret", timeout=60)
        engine.update_runtime_parameters(temperature=0.3)
        d = engine.to_dict()
        assert d["type"] == "LiteLLMEngine"
        assert d["runtime_parameters"] == {"temperature": 0.3}
        assert d["model"] == "openai/gpt-4o-mini"
        for removed in ("api_key", "kwargs", "provider", "timeout_seconds", "timeout",
                        "inline_cutoff_chars", "max_retries", "attachments"):
            assert removed not in d
        assert "secret" not in str(d)

    # ── Removed surface ───────────────────────────────────────────────────────

    def test_attachment_and_retry_surface_is_gone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini")

        for name in ("attach", "detach", "clear_attachments", "attachments",
                     "_prepare_attachment", "_on_detach", "_validate_attachment_path",
                     "inline_cutoff_chars", "_inline_cutoff_chars",
                     "_should_retry", "_max_retries",
                     "provider", "temperature", "drop_params", "api_key",
                     "_timeout_seconds", "_litellm_kwargs"):
            assert not hasattr(engine, name)

    def test_num_retries_is_forwarded_to_litellm_through_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine_module, "litellm", FakeLiteLLM())
        engine = LiteLLMEngine(model="gpt-4o-mini", num_retries=0)

        payload = engine._build_provider_payload(MSGS)

        assert payload["num_retries"] == 0
