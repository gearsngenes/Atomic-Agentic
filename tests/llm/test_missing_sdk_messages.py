from __future__ import annotations

import importlib
from importlib import metadata

import pytest

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import (
    AnthropicEngine,
    GeminiEngine,
    LiteLLMEngine,
    LlamaCppEngine,
    MistralEngine,
    OpenAIEngine,
)

# (module, factory, guarded name, import-error attribute, distribution, extra)
ENGINES = [
    ("openai_engine", lambda: OpenAIEngine(model="m"), "OpenAI", "_SDK_IMPORT_ERROR", "openai", "openai"),
    ("anthropic_engine", lambda: AnthropicEngine(model="m"), "Anthropic", "_SDK_IMPORT_ERROR", "anthropic", "anthropic"),
    ("gemini_engine", lambda: GeminiEngine(model="m"), "genai", "_SDK_IMPORT_ERROR", "google-genai", "gemini"),
    ("mistral_engine", lambda: MistralEngine(model="m"), "Mistral", "_SDK_IMPORT_ERROR", "mistralai", "mistral"),
    ("litellm_engine", lambda: LiteLLMEngine(model="m"), "litellm", "_SDK_IMPORT_ERROR", "litellm", "litellm"),
    ("llama_engine", lambda: LlamaCppEngine(model="x.gguf"), "Llama", "_LLAMA_IMPORT_ERROR", "llama-cpp-python", "llama"),
]
IDS = [row[0] for row in ENGINES]


def _module(name: str):
    return importlib.import_module(f"atomic_agentic.llm.{name}")


@pytest.mark.parametrize("module_name,factory,attr,err_attr,dist,extra", ENGINES, ids=IDS)
class TestMissingSdkMessages:
    def test_missing_sdk_names_package_and_extra(
        self, monkeypatch, module_name, factory, attr, err_attr, dist, extra
    ) -> None:
        monkeypatch.setattr(_module(module_name), attr, None)
        monkeypatch.setattr(_module(module_name), err_attr, None)

        with pytest.raises(LLMEngineError) as excinfo:
            factory()

        assert dist in str(excinfo.value)
        assert f"pip install atomic-agentic[{extra}]" in str(excinfo.value)

    def test_broken_import_is_reported_and_chained(
        self, monkeypatch, module_name, factory, attr, err_attr, dist, extra
    ) -> None:
        module = _module(module_name)
        boom = ImportError("boom")
        monkeypatch.setattr(module, attr, None)
        monkeypatch.setattr(module, err_attr, boom)
        monkeypatch.setattr(metadata, "version", lambda package: "1.0.0")

        with pytest.raises(LLMEngineError) as excinfo:
            factory()

        assert "installed but could not be imported" in str(excinfo.value)
        assert "boom" in str(excinfo.value)
        assert excinfo.value.__cause__ is boom


def test_llama_hf_hub_missing_names_huggingface_hub_and_extra(monkeypatch) -> None:
    module = _module("llama_engine")

    class _NoHubLlama:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("no huggingface_hub")

    monkeypatch.setattr(module, "Llama", _NoHubLlama)

    with pytest.raises(LLMEngineError) as excinfo:
        LlamaCppEngine.from_pretrained("org/repo", "m.gguf")

    assert "huggingface-hub" in str(excinfo.value)
    assert "pip install atomic-agentic[llama]" in str(excinfo.value)
