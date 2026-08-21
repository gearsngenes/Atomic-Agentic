from __future__ import annotations

import importlib
from typing import Any, Mapping

import pytest

llama_module = importlib.import_module("atomic_agentic.llm.llama_engine")

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import LlamaCppEngine


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


def _make_engine(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> LlamaCppEngine:
    FakeLlama.instances.clear()
    monkeypatch.setattr(llama_module, "Llama", FakeLlama)
    return LlamaCppEngine(model_path="model.gguf", **kwargs)


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

    def test_missing_hf_hub_download_raises_when_repo_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)
        monkeypatch.setattr(llama_module, "hf_hub_download", None)

        with pytest.raises(LLMEngineError, match="huggingface_hub"):
            LlamaCppEngine(repo_id="org/repo", filename="model.gguf")

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

        engine = LlamaCppEngine(repo_id="org/repo", filename="model.gguf")
        fake = FakeLlama.instances[-1]

        assert engine.repo_id == "org/repo"
        assert engine.filename == "model.gguf"
        assert engine.model_path == resolved_path
        assert download_calls[-1]["repo_id"] == "org/repo"
        assert download_calls[-1]["filename"] == "model.gguf"
        assert "subfolder" in download_calls[-1]
        assert fake.kwargs["model_path"] == resolved_path

    def test_llama_payload_call_and_extract_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        messages = [{"role": "user", "content": "Hello"}]
        payload = engine._build_provider_payload(messages, {})
        response = engine._call_provider(payload)
        fake = FakeLlama.instances[-1]

        assert payload == {"messages": messages}
        assert fake.chat_completion_calls[-1] == {"messages": messages}
        assert engine._extract_result(response, requested_structured=False) == "llama text"

    def test_llama_extract_result_rejects_bad_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="unexpected response shape"):
            engine._extract_result({"bad": "shape"}, requested_structured=False)

    def test_llama_payload_and_call_include_response_format_when_structured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        messages = [{"role": "user", "content": "Hello"}]

        payload = engine._build_provider_payload(messages, {}, schema)
        assert payload == {
            "messages": messages,
            "response_format": {"type": "json_object", "schema": schema},
        }

        engine._call_provider(payload)
        fake = FakeLlama.instances[-1]
        assert fake.chat_completion_calls[-1]["response_format"] == {
            "type": "json_object",
            "schema": schema,
        }

    def test_llama_payload_omits_response_format_when_not_structured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        messages = [{"role": "user", "content": "Hello"}]

        payload = engine._build_provider_payload(messages, {})

        assert "response_format" not in payload

    def test_llama_extract_result_requested_parses_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = {"choices": [{"message": {"content": '{"a": 1}'}}]}

        assert engine._extract_result(response, requested_structured=True) == {"a": 1}

    def test_llama_extract_result_requested_parse_failure_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = {"choices": [{"message": {"content": "not json"}}]}

        assert engine._extract_result(response, requested_structured=True) == "not json"

    def test_llama_attachments_are_not_supported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
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
        engine = _make_engine(monkeypatch)

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

        LlamaCppEngine(model_path="model.gguf", n_gpu_layers=32, flash_attn=True)

        fake = FakeLlama.instances[-1]
        assert fake.kwargs["n_gpu_layers"] == 32
        assert fake.kwargs["flash_attn"] is True

    def test_generation_params_set_sent_to_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        engine = _make_engine(monkeypatch)
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

        LlamaCppEngine(repo_id="org/repo", filename="model.gguf", subfolder="gguf")

        assert download_calls[-1]["subfolder"] == "gguf"

    def test_get_model_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        model_data = engine._get_model_data()

        assert model_data.provider == "llama_cpp"
        assert model_data.model_path == "model.gguf"


class TestLlamaCppTokenUsage:
    def test_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        response = {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 4,
                "total_tokens": 12,
            },
        }

        result = engine._extract_token_usage(response)

        assert result.input_tokens == 8
        assert result.generated_tokens == 4
        assert result.total_tokens == 12
        assert result.response_tokens == 4

    def test_missing_usage_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="missing usage"):
            engine._extract_token_usage({"choices": []})

    def test_non_mapping_usage_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="must be a mapping"):
            engine._extract_token_usage({"choices": [], "usage": "bad"})

    def test_bad_usage_shape_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="unexpected usage shape"):
            engine._extract_token_usage({"choices": [], "usage": {"wrong_key": 1}})


class TestLlamaCppShouldRetry:
    def test_retries_blindly_within_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch, max_retries=2)
        assert engine._should_retry(ValueError("anything"), attempt=1) is True
        assert engine._should_retry(RuntimeError("anything"), attempt=2) is True

    def test_stops_once_budget_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch, max_retries=2)
        assert engine._should_retry(ValueError("anything"), attempt=3) is False
