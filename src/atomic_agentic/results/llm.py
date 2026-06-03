from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .atomic import AtomicResult

__all__ = ["LLMUsage", "LLMGenerationResult"]


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """
    Minimal normalized token-usage record for an LLM generation.

    Provider-specific usage payloads are intentionally not stored here. Add
    normalized fields only when they are stable across supported providers.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        self._validate_token_count("input_tokens", self.input_tokens)
        self._validate_token_count("output_tokens", self.output_tokens)
        self._validate_token_count("total_tokens", self.total_tokens)

    @staticmethod
    def _validate_token_count(field_name: str, value: int | None) -> None:
        """Validate one optional token-count field."""
        if value is None:
            return
        if type(value) is not int:
            raise TypeError(
                f"{field_name} must be an int or None, got {type(value).__name__}."
            )
        if value < 0:
            raise ValueError(f"{field_name} must be >= 0.")

    def to_dict(self) -> dict[str, int | None]:
        """Return the explicit serialized dictionary representation."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class LLMGenerationResult(AtomicResult):
    """
    Successful string-only LLM generation result.

    ``LLMGenerationResult.result`` is always the generated text string. This
    class does not model structured generation, parsed output, or provider raw
    responses.
    """

    provider: str
    model: str
    usage: LLMUsage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.result, str):
            raise TypeError(
                f"result must be a str, got {type(self.result).__name__}."
            )

        normalized_provider = self._normalize_non_empty_string(
            field_name="provider",
            value=self.provider,
        )
        normalized_model = self._normalize_non_empty_string(
            field_name="model",
            value=self.model,
        )

        if self.usage is not None and not isinstance(self.usage, LLMUsage):
            raise TypeError(
                f"usage must be an LLMUsage or None, got {type(self.usage).__name__}."
            )

        AtomicResult.__post_init__(self)
        object.__setattr__(self, "provider", normalized_provider)
        object.__setattr__(self, "model", normalized_model)

    @staticmethod
    def _normalize_non_empty_string(*, field_name: str, value: str) -> str:
        """Validate and normalize a required non-empty string field."""
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a str, got {type(value).__name__}."
            )

        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must be a non-empty string.")

        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        data = AtomicResult.to_dict(self)
        data.update(
            {
                "provider": self.provider,
                "model": self.model,
                "usage": self.usage.to_dict() if self.usage is not None else None,
            }
        )
        return data
