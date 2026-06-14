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
    Command — the unwrapped ``.result`` extracted from the delegated
    executor's AtomicResult. ``executor_run_id`` links to that executor run
    (``AtomicResult.run_id``) and ``executor_id`` links to the executor
    invokable's identity (``AtomicInvokable.instance_id``), together giving
    full cross-envelope tracing back to "what produced this, and which run".
    """

    executor_run_id: str | None = None
    executor_id: str | None = None

    def __post_init__(self) -> None:
        normalized_executor_run_id = self._normalize_optional_id(
            self.executor_run_id, field_name="executor_run_id"
        )
        normalized_executor_id = self._normalize_optional_id(
            self.executor_id, field_name="executor_id"
        )
        AtomicResult.__post_init__(self)
        object.__setattr__(self, "executor_run_id", normalized_executor_run_id)
        object.__setattr__(self, "executor_id", normalized_executor_id)

    @staticmethod
    def _normalize_optional_id(value: str | None, *, field_name: str) -> str | None:
        """Validate and normalize an optional id-shaped string field."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a str or None, got {type(value).__name__}."
            )

        normalized = value.strip()
        if not normalized:
            raise ValueError(f"{field_name} must be a non-empty string when provided.")

        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        data = AtomicResult.to_dict(self)
        data["executor_run_id"] = self.executor_run_id
        data["executor_id"] = self.executor_id
        return data
