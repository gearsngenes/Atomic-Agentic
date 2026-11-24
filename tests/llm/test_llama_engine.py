from __future__ import annotations

import importlib
from typing import Any

import pytest

llama_module = importlib.import_module("atomic_agentic.llm.llama_engine")

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import LlamaCppEngine


class FakeLlama:
    instances: list["FakeLlama"] = []
    pretrained_calls: list[dict[str, Any]] = []
    pretrained_error: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.model_path = kwargs["model_path"]
        self.chat_completion_calls: list[dict[str, Any]] = []
        FakeLlama.instances.append(self)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        filename: str,
        additional_files: list[str] | None = None,
        local_dir: str | None = None,
        local_dir_use_symlinks: Any = "auto",
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> "FakeLlama":
        cls.pretrained_calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "additional_files": additional_files,
                "local_dir": local_dir,
                "cache_dir": cache_dir,
                "kwargs": kwargs,
            }
        )
        if cls.pretrained_error is not None:
            raise cls.pretrained_error
        return cls(model_path=f"/cache/{repo_id}/{filename}", **kwargs)

    def create_chat_completion(
        self,
        messages: Any,
        response_format: Any = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        stop: Any = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        self.chat_completion_calls.append(
            {"messages": messages, "response_format": response_format}
        )
        return {
            "choices": [{"message": {"content": " llama text "}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    FakeLlama.instances.clear()
    FakeLlama.pretrained_calls.clear()
    FakeLlama.pretrained_error = None


def _make_engine(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> LlamaCppEngine:
    monkeypatch.setattr(llama_module, "Llama", FakeLlama)
    return LlamaCppEngine(model="model.gguf", **kwargs)


class TestConstructor:
    def test_missing_llama_sdk_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llama_module, "Llama", None)
        with pytest.raises(LLMEngineError, match="llama-cpp-python"):
            LlamaCppEngine(model="model.gguf")

    def test_neither_source_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)
        with pytest.raises(LLMEngineError, match="exactly one of 'model' or 'llm'"):
            LlamaCppEngine()

    def test_both_sources_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)
        preloaded = FakeLlama(model_path="a.gguf")
        with pytest.raises(LLMEngineError, match="exactly one of 'model' or 'llm'"):
            LlamaCppEngine(model="model.gguf", llm=preloaded)

    def test_llm_with_client_kwargs_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)
        preloaded = FakeLlama(model_path="a.gguf")
        with pytest.raises(LLMEngineError, match="either 'llm' or client kwargs"):
            LlamaCppEngine(llm=preloaded, n_gpu_layers=1)

    def test_model_loads_with_defaults_and_caller_kwargs_win(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        LlamaCppEngine(model="model.gguf")
        defaults = FakeLlama.instances[-1].kwargs
        assert defaults == {"model_path": "model.gguf", "n_ctx": 2048, "verbose": False}

        LlamaCppEngine(model="model.gguf", n_ctx=4096, verbose=True, n_threads=4)
        override = FakeLlama.instances[-1].kwargs
        assert override["n_ctx"] == 4096
        assert override["verbose"] is True
        assert override["n_threads"] == 4

    def test_llm_used_as_is(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)
        preloaded = FakeLlama(model_path="pre.gguf")
        count = len(FakeLlama.instances)

        engine = LlamaCppEngine(llm=preloaded)

        assert len(FakeLlama.instances) == count
        assert engine._llm is preloaded
        assert engine.model_path == "pre.gguf"
        assert engine._client_kwargs == {}

    def test_default_and_custom_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        assert engine.name == "LlamaCppEngine"
        assert engine.description.startswith("Llama.cpp LLM Engine")

        custom = LlamaCppEngine(model="model.gguf", name="my_llama", description="d")
        assert custom.name == "my_llama"
        assert custom.description.startswith("d")

    def test_model_path_property_is_read_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        assert engine.model_path == "model.gguf"
        with pytest.raises(AttributeError):
            engine.model_path = "other"  # type: ignore[misc]

    def test_removed_arguments_flow_into_client_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch, timeout_seconds=5.0, n_gpu_layers=32)

        assert FakeLlama.instances[-1].kwargs["timeout_seconds"] == 5.0
        assert engine._client_kwargs == {"timeout_seconds": 5.0, "n_gpu_layers": 32}


class TestFromPretrained:
    def test_forwards_arguments_and_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        engine = LlamaCppEngine.from_pretrained(
            "org/repo",
            "m.gguf",
            additional_files=["m-2.gguf"],
            local_dir="/ld",
            cache_dir="/cd",
            name="n",
            description="d",
        )

        call = FakeLlama.pretrained_calls[-1]
        assert call["repo_id"] == "org/repo"
        assert call["filename"] == "m.gguf"
        assert call["additional_files"] == ["m-2.gguf"]
        assert call["local_dir"] == "/ld"
        assert call["cache_dir"] == "/cd"
        assert call["kwargs"] == {"n_ctx": 2048, "verbose": False}
        assert engine.name == "n"
        assert engine.description.startswith("d")
        assert engine.model_path == "/cache/org/repo/m.gguf"
        assert engine._llm is FakeLlama.instances[-1]

    def test_caller_kwargs_win_and_are_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)

        engine = LlamaCppEngine.from_pretrained(
            "org/repo", "m.gguf", n_ctx=512, n_threads=8
        )

        assert FakeLlama.pretrained_calls[-1]["kwargs"] == {
            "n_ctx": 512,
            "verbose": False,
            "n_threads": 8,
        }
        assert engine._client_kwargs == {"n_ctx": 512, "n_threads": 8}
        assert engine.name == "LlamaCppEngine"

    def test_missing_huggingface_hub_raises_engine_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(llama_module, "Llama", FakeLlama)
        FakeLlama.pretrained_error = ImportError("no hub")

        with pytest.raises(LLMEngineError, match=r"huggingface-hub.*atomic-agentic\[llama\]"):
            LlamaCppEngine.from_pretrained("org/repo", "m.gguf")

    def test_sdk_check_runs_before_download(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llama_module, "Llama", None)

        with pytest.raises(LLMEngineError, match="llama-cpp-python"):
            LlamaCppEngine.from_pretrained("org/repo", "m.gguf")
        assert FakeLlama.pretrained_calls == []


class TestRuntimeParameters:
    def test_start_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _make_engine(monkeypatch).runtime_parameters == {}

    def test_valid_keys_accepted_and_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.5, max_tokens=64, seed=7, stop=["x"])
        assert engine.runtime_parameters == {
            "temperature": 0.5, "max_tokens": 64, "seed": 7, "stop": ["x"],
        }

        messages = [{"role": "user", "content": "hi"}]
        payload = engine._build_provider_payload(messages)
        assert payload == {
            "temperature": 0.5, "max_tokens": 64, "seed": 7, "stop": ["x"],
            "messages": messages,
        }

    def test_unknown_key_raises_naming_key_and_valid_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match=r"unknown keys \['bogus'\].*valid keys") as exc:
            engine.update_runtime_parameters(bogus=1)
        assert "temperature" in str(exc.value)
        assert "messages" not in str(exc.value).split("valid keys:")[1]

    @pytest.mark.parametrize("key", ["messages", "response_format"])
    def test_reserved_keys_raise(self, monkeypatch: pytest.MonkeyPatch, key: str) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="managed by the engine"):
            engine.update_runtime_parameters(**{key: 1})

    def test_bad_key_in_batch_applies_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError):
            engine.update_runtime_parameters(temperature=0.1, bogus=1)
        assert engine.runtime_parameters == {}

    def test_none_removes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.5)
        engine.update_runtime_parameters(temperature=None)
        assert engine.runtime_parameters == {}

    def test_snapshot_semantics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        engine.update_runtime_parameters(temperature=0.5)
        payload = engine._build_provider_payload([{"role": "user", "content": "hi"}])
        engine.update_runtime_parameters(temperature=0.9)
        assert payload["temperature"] == 0.5


class TestPayloadAndCall:
    def test_payload_call_and_extract_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        messages = [{"role": "user", "content": "Hello"}]
        payload = engine._build_provider_payload(messages)
        response = engine._call_provider(payload)
        fake = FakeLlama.instances[-1]

        assert payload == {"messages": messages}
        assert fake.chat_completion_calls[-1]["messages"] == messages
        assert engine._extract_result(response, requested_structured=False) == "llama text"

    def test_structured_call_sets_response_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        messages = [{"role": "user", "content": "Hello"}]

        payload = engine._build_provider_payload(messages, schema)
        assert payload == {
            "messages": messages,
            "response_format": {"type": "json_object", "schema": schema},
        }

        engine._call_provider(payload)
        assert FakeLlama.instances[-1].chat_completion_calls[-1]["response_format"] == {
            "type": "json_object",
            "schema": schema,
        }

    def test_payload_omits_response_format_when_not_structured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        payload = engine._build_provider_payload([{"role": "user", "content": "Hello"}])
        assert "response_format" not in payload

    def test_engine_owned_fields_beat_smuggled_runtime_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        engine._runtime_parameters.update(messages="x", response_format="y")
        messages = [{"role": "user", "content": "Hello"}]
        schema = {"type": "object"}

        plain = engine._build_provider_payload(messages)
        structured = engine._build_provider_payload(messages, schema)

        assert plain["messages"] == messages
        assert structured["messages"] == messages
        assert structured["response_format"] == {"type": "json_object", "schema": schema}

    def test_call_provider_passes_payload_as_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        messages = [{"role": "user", "content": "hi"}]
        engine._call_provider({"messages": messages, "temperature": 0.3})

        # The fake records only messages/response_format; a TypeError would
        # surface here if the payload were not unpacked as keyword arguments.
        assert FakeLlama.instances[-1].chat_completion_calls[-1]["messages"] == messages

    def test_extract_result_rejects_bad_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)
        with pytest.raises(LLMEngineError, match="unexpected response shape"):
            engine._extract_result({"bad": "shape"}, requested_structured=False)

    def test_extract_result_requested_parses_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = {"choices": [{"message": {"content": '{"a": 1}'}}]}
        assert engine._extract_result(response, requested_structured=True) == {"a": 1}

    def test_extract_result_requested_parse_failure_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)
        response = {"choices": [{"message": {"content": "not json"}}]}
        assert engine._extract_result(response, requested_structured=True) == "not json"

    def test_has_no_attachment_or_retry_surface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine = _make_engine(monkeypatch)

        for name in ("attach", "detach", "clear_attachments", "attachments",
                     "_prepare_attachment", "_should_retry", "_max_retries"):
            assert not hasattr(engine, name)

    def test_removed_surface_is_gone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch)

        for name in ("repo_id", "filename", "n_ctx", "verbose", "temperature",
                     "_timeout_seconds", "_llama_kwargs"):
            assert not hasattr(engine, name)


class TestToDictAndModelData:
    def test_to_dict_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine = _make_engine(monkeypatch, n_gpu_layers=32)
        engine.update_runtime_parameters(temperature=0.7)
        data = engine.to_dict()

        assert data["type"] == "LlamaCppEngine"
        assert data["runtime_parameters"] == {"temperature": 0.7}
        assert data["model_path"] == "model.gguf"
        for removed in ("llama_kwargs", "client_kwargs", "timeout_seconds", "repo_id",
                        "n_gpu_layers", "n_ctx"):
            assert removed not in data

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
