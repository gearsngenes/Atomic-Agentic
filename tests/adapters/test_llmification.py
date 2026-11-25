from __future__ import annotations

import contextlib
import importlib
from typing import Any

import pytest

from atomic_agentic.adapters import to_llm
from atomic_agentic.constants.llm import LLM_CLIENT_ROUTES
from atomic_agentic.llm import (
    AnthropicEngine,
    GeminiEngine,
    LiteLLMEngine,
    LlamaCppEngine,
    MistralEngine,
    OpenAIEngine,
)

def _openai() -> Any:
    return importlib.import_module("openai").OpenAI(api_key="dummy")


def _async_openai() -> Any:
    return importlib.import_module("openai").AsyncOpenAI(api_key="dummy")


def _anthropic() -> Any:
    return importlib.import_module("anthropic").Anthropic(api_key="dummy")


def _async_anthropic() -> Any:
    return importlib.import_module("anthropic").AsyncAnthropic(api_key="dummy")


def _gemini() -> Any:
    return importlib.import_module("google.genai").Client(api_key="dummy")


def _mistral() -> Any:
    return importlib.import_module("mistralai.client").Mistral(api_key="dummy")


def _router() -> Any:
    litellm = importlib.import_module("litellm")
    return litellm.Router(
        model_list=[{"model_name": "m", "litellm_params": {"model": "openai/gpt-4o", "api_key": "dummy"}}]
    )


# (SDK module to require, client factory, expected engine)
CLIENT_CASES = {
    "openai": ("openai", _openai, OpenAIEngine),
    "async_openai": ("openai", _async_openai, OpenAIEngine),
    "anthropic": ("anthropic", _anthropic, AnthropicEngine),
    "async_anthropic": ("anthropic", _async_anthropic, AnthropicEngine),
    "gemini": ("google.genai", _gemini, GeminiEngine),
    "mistral": ("mistralai.client", _mistral, MistralEngine),
    "router": ("litellm", _router, LiteLLMEngine),
}


def _build(case: str) -> tuple[Any, type]:
    module, factory, engine_cls = CLIENT_CASES[case]
    pytest.importorskip(module)
    return factory(), engine_cls


@pytest.fixture
def llama_stub() -> Any:
    """A real ``Llama`` instance built without loading a model."""
    llama_cpp = pytest.importorskip("llama_cpp")
    stub = object.__new__(llama_cpp.Llama)
    stub.model_path = "/models/stub.gguf"
    stub._stack = contextlib.ExitStack()  # Llama.__del__ closes it
    return stub


class TestRouting:
    @pytest.mark.parametrize("case", list(CLIENT_CASES))
    def test_client_routes_to_engine(self, case: str) -> None:
        client, engine_cls = _build(case)

        engine = to_llm(client, "some-model")

        assert type(engine) is engine_cls
        assert engine._client is client

    def test_llama_stub_routes_to_llama_engine(self, llama_stub: Any) -> None:
        engine = to_llm(llama_stub)

        assert type(engine) is LlamaCppEngine
        assert engine._llm is llama_stub

    def test_subclass_of_supported_client_routes(self) -> None:
        openai = pytest.importorskip("openai")

        class MyOpenAI(openai.OpenAI):
            pass

        client = MyOpenAI(api_key="dummy")

        assert type(to_llm(client, "gpt-4o")) is OpenAIEngine


class TestNameAndDescription:
    def test_given_values_are_used(self) -> None:
        client, _ = _build("openai")

        engine = to_llm(client, "gpt-4o", name="mine", description="my engine")

        assert engine.name == "mine"
        assert engine.description.startswith("my engine")

    def test_omitted_values_use_engine_defaults(self) -> None:
        client, _ = _build("openai")

        engine = to_llm(client, "gpt-4o")

        assert engine.name == "OpenAIEngine"

    def test_llama_name_and_description(self, llama_stub: Any) -> None:
        engine = to_llm(llama_stub, name="local", description="local model")

        assert engine.name == "local"
        assert engine.description.startswith("local model")


class TestModelRules:
    def test_llama_with_model_raises(self, llama_stub: Any) -> None:
        with pytest.raises(TypeError, match="already carries its model path"):
            to_llm(llama_stub, "other")

    def test_missing_model_raises_naming_client(self) -> None:
        client, _ = _build("openai")

        with pytest.raises(TypeError, match="`model` is required for a OpenAI client"):
            to_llm(client)

    def test_non_str_model_raises(self) -> None:
        client, _ = _build("openai")

        with pytest.raises(TypeError, match="`model` is required"):
            to_llm(client, 3)  # type: ignore[arg-type]


class TestUnsupported:
    @pytest.mark.parametrize("value", [object(), None, "openai", 3])
    def test_unsupported_raises_listing_table(self, value: Any) -> None:
        with pytest.raises(TypeError) as excinfo:
            to_llm(value, "m")

        message = str(excinfo.value)
        assert type(value).__name__ in message
        for module_path, class_names, *_ in LLM_CLIENT_ROUTES:
            for class_name in class_names:
                assert f"{module_path}.{class_name}" in message

    def test_engine_is_unsupported(self) -> None:
        client, _ = _build("openai")
        engine = to_llm(client, "gpt-4o")

        with pytest.raises(TypeError, match="unsupported client type 'OpenAIEngine'"):
            to_llm(engine, "gpt-4o")

    def test_lookalike_class_does_not_match(self) -> None:
        fake = type("OpenAI", (), {"__module__": "openai"})()

        with pytest.raises(TypeError):
            to_llm(fake, "m")


class TestMissingSdk:
    def test_unimportable_sdk_is_reported_and_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import atomic_agentic.adapters.llmification as module

        real = module.import_client_types

        def fake_import(module_path: str, class_names: Any) -> Any:
            if module_path == "openai":
                return (), ImportError("no openai")
            return real(module_path, class_names)

        monkeypatch.setattr(module, "import_client_types", fake_import)
        client, _ = _build("anthropic")

        assert type(to_llm(client, "claude")) is AnthropicEngine
        with pytest.raises(TypeError) as excinfo:
            to_llm(object(), "m")
        assert "pip install atomic-agentic[openai]" in str(excinfo.value)

    def test_missing_sdk_client_cannot_route(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import atomic_agentic.adapters.llmification as module

        client, _ = _build("openai")
        monkeypatch.setattr(
            module, "import_client_types", lambda path, names: ((), ImportError("gone"))
        )

        with pytest.raises(TypeError, match="Not importable here"):
            to_llm(client, "gpt-4o")
