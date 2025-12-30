from __future__ import annotations

from typing import Any


class _NoValSentinel:
    """Shared sentinel to represent an absent value (NO_VAL).

    This object is intentionally opaque and single-instanced. Use `is NO_VAL`
    to test for absence.
    """
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "NO_VAL"


NO_VAL: Any = _NoValSentinel()

__all__ = ["NO_VAL"]
