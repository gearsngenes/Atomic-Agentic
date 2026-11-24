from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any

import pytest

openai = pytest.importorskip("openai")

openai_module = importlib.import_module("atomic_agentic.llm.openai_engine")

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import OpenAIEngine
from atomic_agentic.models.results import OpenAITokenUsage
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


class TestOpenAIRemovedSurface:
    def test_cutoff_and_retry_surface_is_gone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)

        for name in ("inline_cutoff_chars", "_inline_cutoff_chars", "_should_retry",
                     "_max_retries", "_on_detach", "timeout_seconds", "_timeout_seconds",
                     "temperature", "max_output_tokens", "reasoning", "truncation"):
            assert not hasattr(engine, name)

    @pytest.mark.parametrize(
        "removed", ["temperature", "max_output_tokens", "reasoning", "truncation",
                    "strict", "timeout_seconds"],
    )
    def test_removed_constructor_arguments_are_not_named_parameters(
        self, monkeypatch: pytest.MonkeyPatch, removed: str
    ) -> None:
        # They are no longer named parameters, so they flow into client_kwargs
        # and reach the SDK client constructor untouched.
        FakeOpenAIClient.instances.clear()
        _make_engine(monkeypatch, **{removed: 1})

        assert FakeOpenAIClient.instances[-1].kwargs[removed] == 1

    def test_max_retries_goes_to_the_sdk_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeOpenAIClient.instances.clear()
        _make_engine(monkeypatch, max_retries=0)

        assert FakeOpenAIClient.instances[-1].kwargs["max_retries"] == 0


class TestOpenAIValidatingProperties:
    def test_strict_setter_rejects_non_bool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="strict"):
            engine.strict = "not-a-bool"  # type: ignore[assignment]

    def test_strict_defaults_true_and_is_settable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        assert engine.strict is True

        engine.strict = False

        assert engine.strict is False
        assert "strict" not in engine.runtime_parameters


class TestOpenAIEngine:
    def test_missing_openai_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_module, "OpenAI", None)
        with pytest.raises(LLMEngineError, match="openai"):
            OpenAIEngine(model="gpt_test")

    def test_constructor_uses_fake_client_and_default_name(
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
            timeout=12.0,
        )

        assert len(FakeOpenAIClient.instances) == 1
        assert len(FakeAsyncOpenAIClient.instances) == 0

        fake = FakeOpenAIClient.instances[-1]
        assert engine.name == "OpenAIEngine"
        assert fake.kwargs == {"api_key": "secret", "timeout": 12.0}
        assert engine._client is fake

    def test_no_timeout_is_injected_into_the_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        _make_engine(monkeypatch)

        assert "timeout" not in FakeOpenAIClient.instances[-1].kwargs

    def test_given_name_is_stored_as_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch, name="my_engine", description="mine")

        assert engine.name == "my_engine"
        assert engine.to_dict()["description"] == "mine"

    def test_client_with_client_kwargs_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        with pytest.raises(LLMEngineError, match="either 'client' or client kwargs"):
            OpenAIEngine(model="gpt-4o-mini", client=FakeOpenAIClient(), api_key="secret")

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
        response = engine._call_provider({"model": "gpt-4o-mini", "input": []})

        assert injected.response_calls
        assert engine._extract_result(response, requested_structured=False) == " openai async text "

    def test_openai_payload_helpers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

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

        assert not hasattr(engine, "_collect_instructions")
        assert not hasattr(engine, "_ensure_user_block")
        assert engine._build_provider_payload(messages)["instructions"] == "Be concise."

    def test_openai_call_provider_and_extract_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(model="gpt-4o-mini")
        engine.update_runtime_parameters(temperature=0.25)
        response = engine._call_provider(engine._build_provider_payload(
            [{"role": "system", "content": "system"}, {"role": "user", "content": "hi"}]
        ))

        fake = FakeOpenAIClient.instances[-1]
        assert engine._extract_result(response, requested_structured=False) == " openai text "
        assert fake.response_calls[-1]["model"] == "gpt-4o-mini"
        assert fake.response_calls[-1]["instructions"] == "system"
        assert fake.response_calls[-1]["temperature"] == 0.25

    def test_runtime_parameters_reach_the_responses_call_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(
            temperature=0.5, max_output_tokens=512, truncation="auto"
        )

        engine._call_provider(engine._build_provider_payload(
            [{"role": "user", "content": "hi"}]
        ))
        call = FakeOpenAIClient.instances[-1].response_calls[-1]

        assert call["temperature"] == 0.5
        assert call["max_output_tokens"] == 512
        assert call["truncation"] == "auto"
        assert call["model"] == "gpt-4o-mini"
        assert "reasoning" not in call

    def test_reasoning_does_not_suppress_temperature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.7, reasoning={"effort": "high"})

        engine._call_provider(engine._build_provider_payload(
            [{"role": "user", "content": "hi"}]
        ))
        call = FakeOpenAIClient.instances[-1].response_calls[-1]

        assert call["temperature"] == 0.7
        assert call["reasoning"] == {"effort": "high"}

    def test_removed_runtime_key_is_not_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.7)
        engine.update_runtime_parameters(temperature=None)

        engine._call_provider(engine._build_provider_payload(
            [{"role": "user", "content": "hi"}]
        ))
        call = FakeOpenAIClient.instances[-1].response_calls[-1]

        assert "temperature" not in call

    def test_payload_snapshots_runtime_parameters_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.1)
        payload = engine._build_provider_payload([{"role": "user", "content": "hi"}])

        engine.update_runtime_parameters(temperature=0.9)
        engine._call_provider(payload)

        assert FakeOpenAIClient.instances[-1].response_calls[-1]["temperature"] == 0.1

    @pytest.mark.parametrize("key", ["model", "input", "instructions", "stream"])
    def test_reserved_runtime_keys_raise(
        self, monkeypatch: pytest.MonkeyPatch, key: str
    ) -> None:
        engine = _make_engine(monkeypatch)

        with pytest.raises(LLMEngineError, match=key):
            engine.update_runtime_parameters(**{key: "x"})

    def test_payload_is_the_sdk_call_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.3)

        payload = engine._build_provider_payload(
            [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]
        )

        assert payload == {
            "temperature": 0.3,
            "model": "gpt-4o-mini",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "instructions": "Be brief.",
        }

    def test_engine_owned_fields_win_over_runtime_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine._runtime_parameters.update(model="smuggled", input="smuggled")

        payload = engine._build_provider_payload([{"role": "user", "content": "hi"}])

        assert payload["model"] == "gpt-4o-mini"
        assert isinstance(payload["input"], list)

    def test_unrecognized_attachment_metadata_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine._attachments["odd.bin"] = {"kind": "text"}

        payload = engine._build_provider_payload([{"role": "user", "content": "hi"}])

        assert payload["input"][-1]["content"] == [{"type": "input_text", "text": "hi"}]

    def test_text_option_survives_a_plain_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(text={"verbosity": "low"})

        engine._call_provider(engine._build_provider_payload(
            [{"role": "user", "content": "hi"}]
        ))

        assert FakeOpenAIClient.instances[-1].response_calls[-1]["text"] == {"verbosity": "low"}

    def test_structured_call_merges_format_into_runtime_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(text={"verbosity": "low"})
        schema = {"type": "object"}

        engine._call_provider(engine._build_provider_payload(
            [{"role": "user", "content": "hi"}], schema
        ))
        text = FakeOpenAIClient.instances[-1].response_calls[-1]["text"]

        assert text["verbosity"] == "low"
        assert text["format"]["schema"] == schema
        assert engine.runtime_parameters["text"] == {"verbosity": "low"}

    def test_openai_to_dict_includes_non_secret_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(model="gpt-4o-mini", api_key="secret")
        engine.update_runtime_parameters(temperature=0.33)

        data = engine.to_dict()

        assert data["type"] == "OpenAIEngine"
        assert data["model"] == "gpt-4o-mini"
        assert data["runtime_parameters"] == {"temperature": 0.33}
        assert data["strict"] is True
        for removed in ("inline_cutoff_chars", "max_retries", "attachments",
                        "timeout_seconds", "temperature"):
            assert removed not in data
        assert "secret" not in str(data)

    def test_call_provider_async_with_async_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeAsyncOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        injected = FakeAsyncOpenAIClient()
        engine = OpenAIEngine(model="gpt-4o-mini", client=injected)

        response = run_coro_sync(engine._call_provider_async({"model": "gpt-4o-mini", "input": []}))

        assert injected.response_calls
        assert engine._extract_result(response, requested_structured=False) == " openai async text "

    def test_call_provider_async_with_sync_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(model="gpt-4o-mini")
        response = run_coro_sync(engine._call_provider_async({"model": "gpt-4o-mini", "input": []}))

        assert engine._extract_result(response, requested_structured=False) == " openai text "

    def test_call_provider_kwargs_include_structured_output_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(model="gpt-4o-mini")
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "additionalProperties": False,
        }
        engine._call_provider(
            engine._build_provider_payload([{"role": "user", "content": "hi"}], schema)
        )

        call = FakeOpenAIClient.instances[-1].response_calls[-1]
        assert call["text"] == {
            "format": {
                "type": "json_schema",
                "name": "output_structure",
                "schema": schema,
                "strict": True,
            }
        }

    def test_call_provider_kwargs_omit_text_format_when_not_structured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        engine = OpenAIEngine(model="gpt-4o-mini")
        engine._call_provider({"model": "gpt-4o-mini", "input": []})

        call = FakeOpenAIClient.instances[-1].response_calls[-1]
        assert "text" not in call

    def test_extract_result_not_requested_returns_output_text_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = SimpleNamespace(output_text=" openai text ")

        assert engine._extract_result(response, requested_structured=False) == " openai text "

    def test_extract_result_requested_parses_json_object(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = SimpleNamespace(output_text='{"a": 1}')

        assert engine._extract_result(response, requested_structured=True) == {"a": 1}

    def test_extract_result_requested_parse_failure_falls_back_to_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = SimpleNamespace(output_text="not json")

        assert engine._extract_result(response, requested_structured=True) == "not json"


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
            input_tokens_details=SimpleNamespace(cached_tokens=2, cache_write_tokens=7),
        ))

        result = engine._extract_token_usage(response)

        assert result.reasoning_tokens == 3
        assert result.response_tokens == 5
        assert result.cached_tokens == 2
        assert result.cache_write_tokens == 7

    def test_output_tokens_details_none_defaults_reasoning_to_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = self._engine(monkeypatch)
        response = SimpleNamespace(usage=self._usage(
            output_tokens_details=None,
            input_tokens_details=SimpleNamespace(cached_tokens=1, cache_write_tokens=0),
        ))

        result = engine._extract_token_usage(response)

        assert result.reasoning_tokens == 0
        assert result.response_tokens == 5
        assert result.cached_tokens == 1
        assert result.cache_write_tokens == 0

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
        assert result.cache_write_tokens is None

    def test_usage_none_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = self._engine(monkeypatch)
        response = SimpleNamespace(usage=None)

        with pytest.raises(LLMEngineError, match="did not include usage"):
            engine._extract_token_usage(response)

    def test_usage_none_returns_zeros_when_not_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = self._engine(monkeypatch)
        engine.require_usage = False

        result = engine._extract_token_usage(SimpleNamespace(usage=None))

        assert isinstance(result, OpenAITokenUsage)
        assert (result.input_tokens, result.generated_tokens,
                result.total_tokens, result.response_tokens) == (0, 0, 0, 0)

    def test_require_usage_rejects_a_non_bool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = self._engine(monkeypatch)

        with pytest.raises(LLMEngineError, match="require_usage"):
            engine.require_usage = "no"  # type: ignore[assignment]

    def test_partially_malformed_usage_still_raises_when_not_required(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = self._engine(monkeypatch)
        engine.require_usage = False
        response = SimpleNamespace(usage=self._usage(
            output_tokens=2,
            output_tokens_details=SimpleNamespace(reasoning_tokens=5),
        ))

        with pytest.raises(ValueError, match="response_tokens must be >= 0"):
            engine._extract_token_usage(response)

    def test_negative_response_tokens_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A negative response_tokens value now fails TokenUsage's own
        # >= 0 field validation (base class), not an OpenAIEngine-level
        # LLMEngineError — it never reaches the separate "exceeds
        # generated_tokens" invariant since -3 is caught first.
        engine = self._engine(monkeypatch)
        response = SimpleNamespace(usage=self._usage(
            output_tokens=2,
            output_tokens_details=SimpleNamespace(reasoning_tokens=5),
        ))

        with pytest.raises(ValueError, match="response_tokens must be >= 0"):
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

    def test_prepare_attachment_inlines_long_text_whole(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "long.txt"
        path.write_text("A" * 300_000)

        meta = engine._prepare_attachment(str(path))

        assert meta["inlined_text"] == "A" * 300_000

    def test_prepare_attachment_failure_names_the_cause(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        missing = tmp_path / "gone.txt"

        with pytest.raises(LLMEngineError, match=r"_prepare_attachment failed for .*: .*gone\.txt"):
            engine._prepare_attachment(str(missing))

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

    def test_attach_rejects_blank_and_non_string_paths(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        with pytest.raises(LLMEngineError, match="non-empty string"):
            engine.attach("")
        with pytest.raises(LLMEngineError, match="non-empty string"):
            engine.attach(123)  # type: ignore[arg-type]

    def test_attach_rejects_missing_and_illegal_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        illegal = tmp_path / "payload.zip"
        illegal.write_bytes(b"data")

        with pytest.raises(LLMEngineError, match="does not exist"):
            engine.attach(str(tmp_path / "missing.txt"))
        with pytest.raises(LLMEngineError, match="not permitted"):
            engine.attach(str(illegal))
        assert engine.attachments == {}

    def test_attach_stores_metadata_and_caches_the_same_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "notes.txt"
        path.write_text("hello")

        first = engine.attach(str(path))
        second = engine.attach(str(path))

        assert first is second
        assert engine.attachments == {str(path): first}

    def test_attachments_property_returns_a_copy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        engine = _make_engine(monkeypatch)
        path = tmp_path / "notes.txt"
        path.write_text("hello")
        engine.attach(str(path))

        engine.attachments.clear()  # type: ignore[attr-defined]

        assert str(path) in engine.attachments

    def test_detach_returns_false_for_unknown_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        assert engine.detach("never_attached.txt") is False

    def test_detach_deletes_uploaded_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        engine = _make_engine(monkeypatch)
        fake = FakeOpenAIClient.instances[-1]
        engine._attachments["doc.pdf"] = {"file_id": "file_abc"}

        assert engine.detach("doc.pdf") is True

        assert fake.deleted_files == ["file_abc"]
        assert engine.attachments == {}

    def test_detach_without_file_id_deletes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        engine = _make_engine(monkeypatch)
        fake = FakeOpenAIClient.instances[-1]
        engine._attachments["notes.txt"] = {"kind": "text"}

        assert engine.detach("notes.txt") is True

        assert fake.deleted_files == []

    def test_detach_swallows_a_failing_delete(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        engine = _make_engine(monkeypatch)

        def _boom(file_id: str) -> None:
            raise RuntimeError("delete failed")

        engine._client.files.delete = _boom  # type: ignore[attr-defined]
        engine._attachments["doc.pdf"] = {"file_id": "file_abc"}

        assert engine.detach("doc.pdf") is True
        assert engine.attachments == {}

    def test_detach_with_async_client_uses_run_coro_sync(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeAsyncOpenAIClient.instances.clear()
        monkeypatch.setattr(openai_module, "OpenAI", FakeOpenAIClient)
        monkeypatch.setattr(openai_module, "AsyncOpenAI", FakeAsyncOpenAIClient)

        injected = FakeAsyncOpenAIClient()
        engine = OpenAIEngine(model="gpt-4o-mini", client=injected)
        engine._attachments["doc.pdf"] = {"file_id": "file_xyz"}
        engine.detach("doc.pdf")

        assert "file_xyz" in injected.deleted_files

    def test_clear_attachments_detaches_everything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakeOpenAIClient.instances.clear()
        engine = _make_engine(monkeypatch)
        fake = FakeOpenAIClient.instances[-1]
        engine._attachments["a.pdf"] = {"file_id": "file_a"}
        engine._attachments["b.txt"] = {"kind": "text"}

        engine.clear_attachments()

        assert engine.attachments == {}
        assert fake.deleted_files == ["file_a"]

    def test_build_provider_payload_with_inlined_attachment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        messages = [{"role": "user", "content": "Q?"}]
        attachments = {
            "notes.txt": {"kind": "text", "inlined": True, "inlined_text": "important note"},
        }

        engine._attachments.update(attachments)
        payload = engine._build_provider_payload(messages)
        user_block = next(b for b in payload["input"] if b["role"] == "user")
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

        engine._attachments.update(attachments)
        payload = engine._build_provider_payload(messages)
        user_block = next(b for b in payload["input"] if b["role"] == "user")
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

        engine._attachments.update(attachments)
        payload = engine._build_provider_payload(messages)
        user_block = next(b for b in payload["input"] if b["role"] == "user")
        img_parts = [p for p in user_block["content"] if p.get("type") == "input_image"]

        assert img_parts == [{"type": "input_image", "file_id": "file_img"}]
