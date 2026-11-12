from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .atomic import AtomicResult

__all__ = ["StructuredResult"]


@dataclass(frozen=True, slots=True)
class StructuredResult(AtomicResult):
    """
    Successful StructuredInvokable invocation result.

    ``StructuredResult.result`` is the packaged ``dict[str, Any]`` produced by
    schema-shaping and missing-value handling. ``unpackaged_result`` preserves
    the value that was actually fed into packaging (the wrapped component's
    AtomicResult.result, unwrapped before shaping) for diagnostics/tracing.
    ``missing_keys`` records which named output fields came back unresolved
    from packaging, before RAISE/DROP/FILL remediation — cross-reference with
    ``result`` and the invokable's ``absent_value_mode`` to tell whether each
    was filled or dropped. ``component_run_id`` links to the wrapped
    component's AtomicResult.run_id for cross-envelope tracing.
    """

    unpackaged_result: Any
    missing_keys: tuple[str, ...] = ()
    component_run_id: str | None = None

    def __post_init__(self) -> None:
        normalized_missing_keys = self._normalize_missing_keys(self.missing_keys)
        normalized_component_run_id = self._normalize_optional_component_run_id(
            self.component_run_id
        )
        AtomicResult.__post_init__(self)
        object.__setattr__(self, "missing_keys", normalized_missing_keys)
        object.__setattr__(self, "component_run_id", normalized_component_run_id)

    @staticmethod
    def _normalize_missing_keys(value: Any) -> tuple[str, ...]:
        """Validate and normalize the missing-keys tuple."""
        if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
            raise TypeError(f"missing_keys must be a tuple of str, got {value!r}.")
        return value

    @staticmethod
    def _normalize_optional_component_run_id(value: str | None) -> str | None:
        """Validate and normalize an optional wrapped-component run id."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"component_run_id must be a str or None, got {type(value).__name__}."
            )

        normalized = value.strip()
        if not normalized:
            raise ValueError("component_run_id must be a non-empty string when provided.")

        return normalized

    def to_dict(self) -> dict[str, Any]:
        """Return the explicit serialized dictionary representation."""
        data = AtomicResult.to_dict(self)
        data["unpackaged_result"] = self.unpackaged_result
        data["missing_keys"] = list(self.missing_keys)
        data["component_run_id"] = self.component_run_id
        return data
