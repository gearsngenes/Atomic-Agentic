from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

from atomic_agentic.models.results.llm import (
    LLMModelData,
    LLMResult,
    LlamaCppModelData,
    OpenAITokenUsage,
    RemoteLLMModelData,
    TokenUsage,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_token_usage(*, input_tokens: int = 10, generated_tokens: int = 5) -> TokenUsage:
    return TokenUsage(
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        total_tokens=input_tokens + generated_tokens,
        response_tokens=generated_tokens,
    )


def make_model_data(*, provider: str = "openai") -> LLMModelData:
    return LLMModelData(provider=provider)


# ── TestTokenUsage ───────────────────────────────────────────────────────────

class TestTokenUsage:
    def test_valid_construction_and_to_dict(self) -> None:
        usage = make_token_usage(input_tokens=10, generated_tokens=5)
        assert usage.to_dict() == {
            "type": "TokenUsage",
            "input_tokens": 10,
            "generated_tokens": 5,
            "total_tokens": 15,
            "response_tokens": 5,
        }

    def test_rejects_non_int_token_count(self) -> None:
        with pytest.raises(TypeError, match="input_tokens"):
            TokenUsage(
                input_tokens="10",  # type: ignore[arg-type]
                generated_tokens=5,
                total_tokens=15,
                response_tokens=5,
            )

    def test_rejects_negative_token_count(self) -> None:
        with pytest.raises(ValueError, match="generated_tokens"):
            TokenUsage(
                input_tokens=10,
                generated_tokens=-1,
                total_tokens=9,
                response_tokens=0,
            )

    def test_rejects_total_tokens_mismatch(self) -> None:
        with pytest.raises(ValueError, match="total_tokens"):
            TokenUsage(
                input_tokens=10,
                generated_tokens=5,
                total_tokens=100,
                response_tokens=5,
            )

    def test_rejects_response_tokens_exceeding_generated(self) -> None:
        with pytest.raises(ValueError, match="response_tokens"):
            TokenUsage(
                input_tokens=10,
                generated_tokens=5,
                total_tokens=15,
                response_tokens=6,
            )

    def test_is_frozen(self) -> None:
        usage = make_token_usage()
        with pytest.raises(FrozenInstanceError):
            usage.input_tokens = 99  # type: ignore[misc]


# ── TestOpenAITokenUsage ─────────────────────────────────────────────────────
# Representative TokenUsage subclass — proves the relocated staticmethod is
# correctly inherited and still validates optional fields. Provider-specific
# field semantics are already covered by tests/llm/test_openai_engine.py.

class TestOpenAITokenUsage:
    def test_valid_construction_with_optional_fields(self) -> None:
        usage = OpenAITokenUsage(
            input_tokens=10,
            generated_tokens=5,
            total_tokens=15,
            response_tokens=5,
            cached_tokens=2,
            reasoning_tokens=1,
        )
        assert usage.cached_tokens == 2
        assert usage.reasoning_tokens == 1
        d = usage.to_dict()
        assert d["cached_tokens"] == 2
        assert d["reasoning_tokens"] == 1

    def test_optional_fields_default_to_none(self) -> None:
        usage = OpenAITokenUsage(
            input_tokens=10,
            generated_tokens=5,
            total_tokens=15,
            response_tokens=5,
        )
        assert usage.cached_tokens is None
        assert usage.reasoning_tokens is None

    def test_rejects_negative_optional_field(self) -> None:
        with pytest.raises(ValueError, match="cached_tokens"):
            OpenAITokenUsage(
                input_tokens=10,
                generated_tokens=5,
                total_tokens=15,
                response_tokens=5,
                cached_tokens=-1,
            )


# ── TestLLMModelData ─────────────────────────────────────────────────────────

class TestLLMModelData:
    def test_valid_construction_and_to_dict(self) -> None:
        data = LLMModelData(provider="openai")
        assert data.to_dict() == {"type": "LLMModelData", "provider": "openai"}

    def test_strips_whitespace(self) -> None:
        data = LLMModelData(provider="  openai  ")
        assert data.provider == "openai"

    def test_rejects_empty_provider(self) -> None:
        with pytest.raises(ValueError, match="provider"):
            LLMModelData(provider="   ")

    def test_rejects_non_string_provider(self) -> None:
        with pytest.raises(TypeError, match="provider"):
            LLMModelData(provider=123)  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        data = make_model_data()
        with pytest.raises(FrozenInstanceError):
            data.provider = "other"  # type: ignore[misc]


# ── TestRemoteLLMModelData ───────────────────────────────────────────────────

class TestRemoteLLMModelData:
    def test_valid_construction_and_to_dict(self) -> None:
        data = RemoteLLMModelData(provider="openai", model_name="gpt-5")
        d = data.to_dict()
        assert d["provider"] == "openai"
        assert d["model_name"] == "gpt-5"

    def test_rejects_empty_model_name(self) -> None:
        with pytest.raises(ValueError, match="model_name"):
            RemoteLLMModelData(provider="openai", model_name="   ")


# ── TestLlamaCppModelData ────────────────────────────────────────────────────

class TestLlamaCppModelData:
    def test_valid_construction_and_to_dict(self) -> None:
        data = LlamaCppModelData(provider="llama_cpp", model_path="/models/model.gguf")
        d = data.to_dict()
        assert d["provider"] == "llama_cpp"
        assert d["model_path"] == "/models/model.gguf"

    def test_rejects_empty_model_path(self) -> None:
        with pytest.raises(ValueError, match="model_path"):
            LlamaCppModelData(provider="llama_cpp", model_path="")


# ── TestLLMResult ─────────────────────────────────────────────────────────────

class TestLLMResult:
    def _make(self, **overrides: object) -> LLMResult:
        started_at = datetime.now(timezone.utc)
        kwargs = dict(
            result="hello",
            invoker_id="engine-1",
            started_at=started_at,
            ended_at=started_at + timedelta(seconds=1),
            token_usage=make_token_usage(),
            model_data=make_model_data(),
        )
        kwargs.update(overrides)
        return LLMResult(**kwargs)  # type: ignore[arg-type]

    def test_valid_construction_and_to_dict(self) -> None:
        result = self._make()
        d = result.to_dict()
        assert d["result"] == "hello"
        assert d["token_usage"] == result.token_usage.to_dict()
        assert d["model_data"] == result.model_data.to_dict()

    def test_rejects_non_string_result(self) -> None:
        with pytest.raises(TypeError, match="result"):
            self._make(result=123)

    def test_rejects_wrong_token_usage_type(self) -> None:
        with pytest.raises(TypeError, match="token_usage"):
            self._make(token_usage="not-a-token-usage")

    def test_rejects_wrong_model_data_type(self) -> None:
        with pytest.raises(TypeError, match="model_data"):
            self._make(model_data="not-a-model-data")
