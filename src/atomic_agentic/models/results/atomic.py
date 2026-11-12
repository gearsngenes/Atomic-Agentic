from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

__all__ = ["AtomicResult"]


@dataclass(frozen=True, slots=True)
class AtomicResult:
    """
    Base successful-invocation result envelope.

    ``AtomicResult`` carries the caller-facing payload produced by one completed
    invocation plus minimal run identity and timing metadata.

    Fields
    ------
    result:
        Caller-facing payload produced by the invocation.

    invoker_id:
        Stable instance identifier of the invokable that produced this result.

    started_at:
        Timezone-aware invocation start timestamp. Stored normalized to UTC.

    ended_at:
        Timezone-aware invocation end timestamp. Stored normalized to UTC.

    run_id:
        Result-owned unique run identifier. If omitted or None, generated during
        construction using ``uuid4().hex``. If provided, validated and normalized
        before storage.

    elapsed_s:
        Invocation duration in seconds, derived from ``ended_at - started_at``.
    """

    result: Any
    invoker_id: str
    started_at: datetime
    ended_at: datetime

    run_id: str | None = field(default=None, kw_only=True)
    elapsed_s: float = field(init=False)

    def __post_init__(self) -> None:
        normalized_invoker_id = self._normalize_invoker_id(self.invoker_id)
        normalized_started_at = self._normalize_datetime(
            field_name="started_at",
            value=self.started_at,
        )
        normalized_ended_at = self._normalize_datetime(
            field_name="ended_at",
            value=self.ended_at,
        )

        if normalized_ended_at < normalized_started_at:
            raise ValueError("ended_at must be greater than or equal to started_at.")

        raw_run_id = uuid4().hex if self.run_id is None else self.run_id
        normalized_run_id = self._normalize_run_id(raw_run_id)

        object.__setattr__(self, "invoker_id", normalized_invoker_id)
        object.__setattr__(self, "started_at", normalized_started_at)
        object.__setattr__(self, "ended_at", normalized_ended_at)
        object.__setattr__(self, "run_id", normalized_run_id)
        object.__setattr__(
            self,
            "elapsed_s",
            (normalized_ended_at - normalized_started_at).total_seconds(),
        )

    @staticmethod
    def _normalize_invoker_id(value: str) -> str:
        """Validate and normalize the invoker instance identifier."""
        if not isinstance(value, str):
            raise TypeError(
                f"invoker_id must be a str, got {type(value).__name__}."
            )

        normalized = value.strip()
        if not normalized:
            raise ValueError("invoker_id must be a non-empty string.")

        return normalized

    @staticmethod
    def _normalize_run_id(value: str) -> str:
        """Validate and normalize a result run identifier."""
        if not isinstance(value, str):
            raise TypeError(
                f"run_id must be a str or None, got {type(value).__name__}."
            )

        normalized = value.strip()
        if not normalized:
            raise ValueError("run_id must be a non-empty string when provided.")

        return normalized

    @staticmethod
    def _normalize_datetime(*, field_name: str, value: datetime) -> datetime:
        """Validate a timezone-aware datetime and normalize it to UTC."""
        if not isinstance(value, datetime):
            raise TypeError(
                f"{field_name} must be a datetime, got {type(value).__name__}."
            )

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{field_name} must be timezone-aware.")

        return value.astimezone(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        return {
            "run_id": self.run_id,
            "invoker_id": self.invoker_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "elapsed_s": self.elapsed_s,
            "result": self.result,
        }