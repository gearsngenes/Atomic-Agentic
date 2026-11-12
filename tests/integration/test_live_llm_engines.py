from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from huggingface_hub import try_to_load_from_cache
except ImportError:  # pragma: no cover
    try_to_load_from_cache = None

from atomic_agentic.exceptions import LLMEngineError
from atomic_agentic.llm import (
    AnthropicEngine,
    GeminiEngine,
    LiteLLMEngine,
    LLMEngine,
    LlamaCppEngine,
    MistralEngine,
    OpenAIEngine,
)
from atomic_agentic.models.results import LLMResult

LLAMACPP_MODEL_REPO = "unsloth/phi-4-GGUF"
LLAMACPP_MODEL_FILENAME = "phi-4-Q4_K_M.gguf"

# Shared structured-output schema, reused across all six engines' new test.
STRUCTURED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "acknowledgement": {"type": "string"},
    },
    "required": ["acknowledgement"],
    "additionalProperties": False,
}


pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.network,
    pytest.mark.slow,
]


def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv()

    if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def _live_tests_enabled() -> bool:
    _load_env()
    return os.getenv("AA_RUN_LIVE_LLM_TESTS") == "1"


def _skip_if_live_tests_disabled() -> None:
    if not _live_tests_enabled():
        pytest.skip("Set AA_RUN_LIVE_LLM_TESTS=1 to run live LLM integration tests.")


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a terse test responder."},
        {"role": "user", "content": "Reply with a very short acknowledgement."},
    ]


def _assert_live_text_response(result: Any) -> None:
    assert isinstance(result, str)
    assert result.strip()


def _structured_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a terse test responder.",
        },
        {
            "role": "user",
            "content": (
                "Reply with a JSON object containing a short acknowledgement "
                "string in the 'acknowledgement' field."
            ),
        },
    ]


def _openai_engine() -> LLMEngine:
    _skip_if_live_tests_disabled()

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set.")

    try:
        return OpenAIEngine(
            model=os.getenv("AA_TEST_OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            timeout_seconds=60,
            max_retries=0,
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _gemini_engine() -> LLMEngine:
    _skip_if_live_tests_disabled()

    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GOOGLE_API_KEY or GEMINI_API_KEY is not set.")

    try:
        return GeminiEngine(
            model=os.getenv("AA_TEST_GEMINI_MODEL", "gemini-2.5-flash-lite"),
            temperature=0,
            timeout_seconds=60,
            max_retries=0,
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _mistral_engine() -> LLMEngine:
    _skip_if_live_tests_disabled()

    if not os.getenv("MISTRAL_API_KEY"):
        pytest.skip("MISTRAL_API_KEY is not set.")

    try:
        return MistralEngine(
            model=os.getenv("AA_TEST_MISTRAL_MODEL", "mistral-small-latest"),
            temperature=0,
            timeout_seconds=60,
            max_retries=0,
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _anthropic_engine() -> LLMEngine:
    _skip_if_live_tests_disabled()

    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set.")

    try:
        return AnthropicEngine(
            model=os.getenv("AA_TEST_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            temperature=None,
            timeout_seconds=60,
            max_retries=0,
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _litellm_engine() -> LLMEngine:
    _skip_if_live_tests_disabled()

    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY is not set.")

    try:
        return LiteLLMEngine(
            model=os.getenv("AA_TEST_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            provider="anthropic",
            temperature=0,
            timeout_seconds=60,
            max_retries=0,
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _llamacpp_engine() -> LLMEngine:
    _skip_if_live_tests_disabled()

    if try_to_load_from_cache is None:
        pytest.skip("huggingface_hub is not installed.")

    cached = try_to_load_from_cache(LLAMACPP_MODEL_REPO, LLAMACPP_MODEL_FILENAME)
    if not isinstance(cached, str):
        pytest.skip(f"{LLAMACPP_MODEL_REPO}/{LLAMACPP_MODEL_FILENAME} is not cached locally.")

    try:
        return LlamaCppEngine(
            repo_id=LLAMACPP_MODEL_REPO,
            filename=LLAMACPP_MODEL_FILENAME,
            timeout_seconds=120,
            max_retries=0,
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))


ENGINE_BUILDERS: list[tuple[str, Callable[[], LLMEngine]]] = [
    ("openai", _openai_engine),
    ("gemini", _gemini_engine),
    ("mistral", _mistral_engine),
    ("anthropic", _anthropic_engine),
    ("litellm", _litellm_engine),
    ("llamacpp", _llamacpp_engine),
]


@pytest.mark.parametrize("provider,build_engine", ENGINE_BUILDERS)
def test_live_llm_engine_returns_non_empty_text(
    provider: str,
    build_engine: Callable[[], LLMEngine],
) -> None:
    engine = build_engine()

    result = engine.invoke({"messages": _messages()})

    assert isinstance(result, LLMResult)
    _assert_live_text_response(result.result)


@pytest.mark.parametrize("provider,build_engine", ENGINE_BUILDERS)
def test_live_llm_engine_rejects_invalid_messages_before_provider_call(
    provider: str,
    build_engine: Callable[[], LLMEngine],
) -> None:
    engine = build_engine()

    with pytest.raises(LLMEngineError, match="messages"):
        engine.invoke({"messages": "not a message list"})


@pytest.mark.parametrize("provider,build_engine", ENGINE_BUILDERS)
def test_live_llm_engine_to_dict_exposes_non_secret_runtime_snapshot(
    provider: str,
    build_engine: Callable[[], LLMEngine],
) -> None:
    engine = build_engine()

    data = engine.to_dict()

    # LlamaCpp's builder uses a longer timeout (local model load/generation
    # can exceed the 60s the five remote-engine builders share).
    expected_timeout = 120.0 if provider == "llamacpp" else 60.0

    assert data["type"] == type(engine).__name__
    assert data["timeout_seconds"] == expected_timeout
    assert data["max_retries"] == 0
    assert "attachments" in data
    assert "api_key" not in data


@pytest.mark.parametrize("provider,build_engine", ENGINE_BUILDERS)
def test_live_llm_engine_returns_structured_result(
    provider: str,
    build_engine: Callable[[], LLMEngine],
) -> None:
    engine = build_engine()

    result = engine.invoke(
        {"messages": _structured_messages(), "output_structure": STRUCTURED_SCHEMA}
    )

    assert isinstance(result, LLMResult)
    assert isinstance(result.result, dict)
    assert "acknowledgement" in result.result
    assert isinstance(result.result["acknowledgement"], str)
    assert result.result["acknowledgement"].strip()
