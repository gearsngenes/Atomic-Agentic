from __future__ import annotations

"""Backward-compatible import path for the shared ``NO_VAL`` sentinel.

The canonical sentinel definition lives in :mod:`atomic_agentic.core.Constants`.
This module remains intentionally small so existing imports such as:

    from atomic_agentic.core.sentinels import NO_VAL

continue to resolve to the same singleton object used by the rest of the package.
"""

from .Constants import NO_VAL

__all__ = ["NO_VAL"]
