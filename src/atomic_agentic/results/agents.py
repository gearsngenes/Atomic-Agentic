from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .atomic import AtomicResult
from .llm import LLMModelData, TokenUsage

__all__ = ["AgentResult"]


@dataclass(frozen=True, slots=True)
class AgentResult(AtomicResult):
    """
    Successful Agent invocation result.

    ``AgentResult.result`` is the final caller-facing payload produced by the
    Agent after post-invoke processing has completed.

    This result intentionally stores only compact LLM accounting/model context.
    Full LLM activity, triggering prompts, protected ``_invoke`` payloads, and
    memory/rendering records belong to AgentRecord-style history objects rather
    than the public result envelope.

    Fields
    ------
    llm_token_usage:
        Token-usage records for each LLM generation that contributed to this
        Agent result. Stored as an immutable tuple after construction.

    llm_model_data:
        Model identity associated with the LLM activity that produced this
        Agent result.
    """

    llm_token_usage: tuple[TokenUsage, ...]
    llm_model_data: LLMModelData

    def __post_init__(self) -> None:
        normalized_llm_token_usage = self._normalize_llm_token_usage(
            self.llm_token_usage
        )

        if not isinstance(self.llm_model_data, LLMModelData):
            raise TypeError(
                "llm_model_data must be an LLMModelData instance, "
                f"got {type(self.llm_model_data).__name__}."
            )

        object.__setattr__(
            self,
            "llm_token_usage",
            normalized_llm_token_usage,
        )

        AtomicResult.__post_init__(self)

    @staticmethod
    def _normalize_llm_token_usage(
        value: Sequence[TokenUsage],
    ) -> tuple[TokenUsage, ...]:
        """Validate and normalize Agent LLM token usage records."""
        if not isinstance(value, Sequence) or isinstance(
            value,
            (str, bytes, bytearray),
        ):
            raise TypeError(
                "llm_token_usage must be a non-empty sequence of TokenUsage "
                f"instances, got {type(value).__name__}."
            )

        normalized = tuple(value)
        if not normalized:
            raise ValueError("llm_token_usage must not be empty.")

        for index, usage in enumerate(normalized):
            if not isinstance(usage, TokenUsage):
                raise TypeError(
                    "llm_token_usage must contain only TokenUsage instances; "
                    f"item {index} is {type(usage).__name__}."
                )

        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        data = AtomicResult.to_dict(self)
        data.update(
            {
                "llm_token_usage": [
                    usage.to_dict()
                    for usage in self.llm_token_usage
                ],
                "llm_model_data": self.llm_model_data.to_dict(),
            }
        )
        return data