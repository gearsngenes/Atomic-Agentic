from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from atomic_agentic.core.Exceptions import LLMEngineError
from atomic_agentic.engines.LLMEngines import (
    GeminiEngine,
    LLMEngine,
    MistralEngine,
    OpenAIEngine,
)


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


ENGINE_BUILDERS: list[tuple[str, Callable[[], LLMEngine]]] = [
    ("openai", _openai_engine),
    ("gemini", _gemini_engine),
    ("mistral", _mistral_engine),
]


@pytest.mark.parametrize("provider,build_engine", ENGINE_BUILDERS)
def test_live_llm_engine_returns_non_empty_text(
    provider: str,
    build_engine: Callable[[], LLMEngine],
) -> None:
    engine = build_engine()

    result = engine.invoke({"messages": _messages()})

    _assert_live_text_response(result)


@pytest.mark.parametrize("provider,build_engine", ENGINE_BUILDERS)
def test_live_llm_engine_invoke_messages_returns_non_empty_text(
    provider: str,
    build_engine: Callable[[], LLMEngine],
) -> None:
    engine = build_engine()

    result = engine.invoke_messages(_messages())

    _assert_live_text_response(result)


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

    assert data["type"] == type(engine).__name__
    assert data["timeout_seconds"] == 60.0
    assert data["max_retries"] == 0
    assert "attachments" in data
    assert "api_key" not in data