from __future__ import annotations

import re
from typing import Any, TypeAlias, TypeVar


# =============================================================================
# Sentinel / absence
# =============================================================================
# Used by:
# - models/parameters.py: ParamSpec defaults
# - core/Invokable.py: signature/default rendering and input filtering
#
# Contract:
# - NO_VAL is identity-checkable. Use `is NO_VAL`, not equality.
# - repr(NO_VAL) is stable and should remain "NO_VAL".


NO_VAL_REPR = "NO_VAL"


class _NoValSentinel:
    """Shared sentinel to represent an absent value.

    This object is intentionally opaque and single-instanced at module import
    time. Use `is NO_VAL` to test for absence.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return NO_VAL_REPR


NO_VAL: Any = _NoValSentinel()


# =============================================================================
# Generic identifier/name validation
# =============================================================================
# Used by:
# - core/Invokable.py: invokable names and parameter names
# - utils/parameters.py: schema-derived parameter names
#
# This is intentionally Python-identifier-like, not a full Python keyword check.
# Individual modules remain responsible for deciding which exception type/message
# to raise when validation fails.


IDENTIFIER_PATTERN_TEXT = r"[A-Za-z_][A-Za-z0-9_]*"
IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(
    rf"^{IDENTIFIER_PATTERN_TEXT}$"
)


# =============================================================================
# Shared utility typing
# =============================================================================
# Used by:
# - utils/core.py: generic sync/async bridge and header normalization
#
# Keep this section intentionally small. These names support narrow shared
# utilities and should not become a dumping ground for module-specific type aliases.


T = TypeVar("T")

HeaderValue: TypeAlias = str | int | float | bool | bytes | bytearray


# =============================================================================
# Explicit public export list
# =============================================================================
# Keep this explicit so adding local helper names or imports cannot accidentally
# widen the module's public surface.


__all__ = [
    "NO_VAL_REPR",
    "NO_VAL",
    "IDENTIFIER_PATTERN_TEXT",
    "IDENTIFIER_PATTERN",
    "T",
    "HeaderValue",
]