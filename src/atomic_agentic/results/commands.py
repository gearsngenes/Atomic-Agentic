from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .atomic import AtomicResult

__all__ = ["CommandResult"]


@dataclass(frozen=True, slots=True)
class CommandResult(AtomicResult):
    """
    Successful Command invocation result.

    ``CommandResult.result`` is the caller-facing payload produced by the
    Command. ``executor_run_id`` optionally links to the delegated executor run
    once executor results are AtomicResult-family objects.
    """

    executor_run_id: str | None = None

    def __post_init__(self) -> None:
        normalized_executor_run_id = self._normalize_optional_run_id(
            self.executor_run_id
        )
        AtomicResult.__post_init__(self)
        object.__setattr__(self, "executor_run_id", normalized_executor_run_id)

    @staticmethod
    def _normalize_optional_run_id(value: str | None) -> str | None:
        """Validate and normalize an optional child executor run id."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"executor_run_id must be a str or None, got {type(value).__name__}."
            )

        normalized = value.strip()
        if not normalized:
            raise ValueError("executor_run_id must be a non-empty string when provided.")

        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        data = AtomicResult.to_dict(self)
        data["executor_run_id"] = self.executor_run_id
        return data
