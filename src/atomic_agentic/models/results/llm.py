from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from .atomic import AtomicResult

__all__ = [
    "TokenUsage",
    "OpenAITokenUsage",
    "GeminiTokenUsage",
    "MistralTokenUsage",
    "LlamaCppTokenUsage",
    "AnthropicTokenUsage",
    "LiteLLMTokenUsage",
    "LLMModelData",
    "RemoteLLMModelData",
    "LocalLLMModelData",
    "LlamaCppModelData",
    "LLMResult",
]


# --------------------------------------------------------------------------- #
# Shared validation / serialization helpers
# --------------------------------------------------------------------------- #
def _validate_token_count(field_name: str, value: int) -> None:
    """Validate one required token-count field."""
    if type(value) is not int:
        raise TypeError(
            f"{field_name} must be an int, got {type(value).__name__}."
        )
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0.")


def _validate_optional_token_count(field_name: str, value: int | None) -> None:
    """Validate one optional provider-specific token-count field."""
    if value is None:
        return
    _validate_token_count(field_name, value)


def _normalize_required_string(field_name: str, value: str) -> str:
    """Validate and normalize a required non-empty string field."""
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a str, got {type(value).__name__}."
        )

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")

    return normalized


def _dataclass_record_to_dict(record: Any) -> dict[str, Any]:
    """Serialize a dataclass record including its concrete class name."""
    data = {field.name: getattr(record, field.name) for field in fields(record)}
    return {"type": type(record).__name__, **data}


# --------------------------------------------------------------------------- #
# Token usage records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TokenUsage:
    """
    Normalized base token-usage record for one LLM generation.

    The base record contains only token metrics that are consistently available
    across the currently supported providers, either as native usage fields or
    by exact derivation from native usage fields.

    Fields
    ------
    input_tokens:
        Tokens consumed from the initial prompt/messages/input context.

    generated_tokens:
        All non-input tokens counted by the provider for the generation side.
        This may include visible response tokens plus hidden reasoning/thought,
        prediction, tool-use, or other provider-specific generated-side tokens.

    total_tokens:
        Total usage count. Must equal ``input_tokens + generated_tokens``.

    response_tokens:
        Visible-reply token count — the subset of ``generated_tokens``
        actually surfaced to the caller, excluding hidden reasoning/thinking
        tokens. Required and always populated: trivially equal to
        ``generated_tokens`` for providers with no reasoning-token concept,
        or derived by subtraction where one exists. Must not exceed
        ``generated_tokens``.
    """

    input_tokens: int
    generated_tokens: int
    total_tokens: int
    response_tokens: int

    def __post_init__(self) -> None:
        _validate_token_count("input_tokens", self.input_tokens)
        _validate_token_count("generated_tokens", self.generated_tokens)
        _validate_token_count("total_tokens", self.total_tokens)
        _validate_token_count("response_tokens", self.response_tokens)

        expected_total = self.input_tokens + self.generated_tokens
        if self.total_tokens != expected_total:
            raise ValueError(
                "total_tokens must equal input_tokens + generated_tokens; "
                f"expected {expected_total}, got {self.total_tokens}."
            )

        if self.response_tokens > self.generated_tokens:
            raise ValueError(
                "response_tokens must not exceed generated_tokens; "
                f"got {self.response_tokens} > {self.generated_tokens}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return _dataclass_record_to_dict(self)


@dataclass(frozen=True, slots=True)
class OpenAITokenUsage(TokenUsage):
    """
    OpenAI Responses API token-usage details.

    ``cached_tokens`` is an input-token subset and is not additive to
    ``input_tokens``.
    """

    cached_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        TokenUsage.__post_init__(self)
        _validate_optional_token_count("cached_tokens", self.cached_tokens)
        _validate_optional_token_count("reasoning_tokens", self.reasoning_tokens)


@dataclass(frozen=True, slots=True)
class GeminiTokenUsage(TokenUsage):
    """
    Gemini token-usage details preserved from GenerateContent usage metadata.

    The base ``response_tokens`` field represents what was previously
    ``candidates_token_count`` here — Gemini natively reports this value
    rather than requiring derivation, unlike OpenAI/Anthropic/LiteLLM.
    """

    thoughts_token_count: int | None = None
    tool_use_prompt_token_count: int | None = None
    cached_content_token_count: int | None = None

    def __post_init__(self) -> None:
        TokenUsage.__post_init__(self)
        _validate_optional_token_count("thoughts_token_count", self.thoughts_token_count)
        _validate_optional_token_count(
            "tool_use_prompt_token_count",
            self.tool_use_prompt_token_count,
        )
        _validate_optional_token_count(
            "cached_content_token_count",
            self.cached_content_token_count,
        )


@dataclass(frozen=True, slots=True)
class MistralTokenUsage(TokenUsage):
    """Mistral token-usage details preserved from chat completion usage."""

    cached_tokens: int | None = None

    def __post_init__(self) -> None:
        TokenUsage.__post_init__(self)
        _validate_optional_token_count("cached_tokens", self.cached_tokens)


@dataclass(frozen=True, slots=True)
class LlamaCppTokenUsage(TokenUsage):
    """llama-cpp-python token-usage details preserved from chat completion usage."""


@dataclass(frozen=True, slots=True)
class AnthropicTokenUsage(TokenUsage):
    """
    Anthropic Messages API token-usage details.

    ``cache_creation_input_tokens`` and ``cache_read_input_tokens`` are
    prompt-cache accounting fields; they are subsets of billed input cost
    and are not additive to ``input_tokens``. ``thinking_tokens`` is a
    subset of ``generated_tokens`` produced by extended thinking.
    """

    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    thinking_tokens: int | None = None

    def __post_init__(self) -> None:
        TokenUsage.__post_init__(self)
        _validate_optional_token_count(
            "cache_creation_input_tokens", self.cache_creation_input_tokens
        )
        _validate_optional_token_count(
            "cache_read_input_tokens", self.cache_read_input_tokens
        )
        _validate_optional_token_count("thinking_tokens", self.thinking_tokens)


@dataclass(frozen=True, slots=True)
class LiteLLMTokenUsage(TokenUsage):
    """
    LiteLLM-normalized token-usage details.

    litellm's own ``Usage`` object already normalizes raw per-provider usage
    into one shape before it reaches engine code — no per-provider subclass
    is needed here, unlike the raw-SDK case that justified
    ``AnthropicTokenUsage``. ``cached_tokens`` and ``cache_creation_tokens``
    are prompt-cache accounting fields read from
    ``usage.prompt_tokens_details``; both are subsets of billed input cost
    and are not additive to ``input_tokens``. ``reasoning_tokens`` is a
    subset of ``generated_tokens`` read from
    ``usage.completion_tokens_details``. For Anthropic models specifically,
    litellm computes ``reasoning_tokens`` as its own token-count estimate off
    extracted thinking text, not a value Anthropic's API reports directly.
    """

    cached_tokens: int | None = None
    cache_creation_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        TokenUsage.__post_init__(self)
        _validate_optional_token_count("cached_tokens", self.cached_tokens)
        _validate_optional_token_count(
            "cache_creation_tokens", self.cache_creation_tokens
        )
        _validate_optional_token_count("reasoning_tokens", self.reasoning_tokens)


# --------------------------------------------------------------------------- #
# Model data records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LLMModelData:
    """
    Base model-identity record for an LLM generation.

    ``provider`` is the framework-facing provider/backend identifier, such as
    ``"openai"``, ``"gemini"``, ``"mistral"``, or ``"llama_cpp"``.
    """

    provider: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider",
            _normalize_required_string("provider", self.provider),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return _dataclass_record_to_dict(self)


@dataclass(frozen=True, slots=True)
class RemoteLLMModelData(LLMModelData):
    """Model data for hosted/provider API-backed LLMs."""

    model_name: str

    def __post_init__(self) -> None:
        LLMModelData.__post_init__(self)
        object.__setattr__(
            self,
            "model_name",
            _normalize_required_string("model_name", self.model_name),
        )


@dataclass(frozen=True, slots=True)
class LocalLLMModelData(LLMModelData):
    """Base model data for local/backend-loaded LLMs."""


@dataclass(frozen=True, slots=True)
class LlamaCppModelData(LocalLLMModelData):
    """Model data for a llama-cpp-python backed local model."""

    model_path: str

    def __post_init__(self) -> None:
        LocalLLMModelData.__post_init__(self)
        object.__setattr__(
            self,
            "model_path",
            _normalize_required_string("model_path", self.model_path),
        )


# --------------------------------------------------------------------------- #
# LLM result record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class LLMResult(AtomicResult):
    """
    Successful string-only LLM generation result.

    ``LLMResult.result`` is always the generated text string. Token accounting
    and configured model identity are stored as explicit nested result records.
    This class does not model structured generation, parsed output, or provider
    raw responses.
    """

    token_usage: TokenUsage
    model_data: LLMModelData

    def __post_init__(self) -> None:
        if not isinstance(self.result, str):
            raise TypeError(
                f"result must be a str, got {type(self.result).__name__}."
            )

        if not isinstance(self.token_usage, TokenUsage):
            raise TypeError(
                "token_usage must be a TokenUsage instance, "
                f"got {type(self.token_usage).__name__}."
            )

        if not isinstance(self.model_data, LLMModelData):
            raise TypeError(
                "model_data must be an LLMModelData instance, "
                f"got {type(self.model_data).__name__}."
            )

        AtomicResult.__post_init__(self)

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        data = AtomicResult.to_dict(self)
        data.update(
            {
                "token_usage": self.token_usage.to_dict(),
                "model_data": self.model_data.to_dict(),
            }
        )
        return data
