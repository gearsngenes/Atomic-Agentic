"""Parameter-kind vocabulary and ordering, shared by ``ParamSpec`` and its
consumers.

Lives in ``constants/`` rather than ``models/parameters.py`` so that
``utils/parameters.py``'s kind-priority table can be defined here too,
directly alongside the vocabulary it orders. ``constants/`` sits *below*
``models/`` in the layered dependency topology
(``{exceptions, constants} -> models -> utils -> core -> ...``), so a
``models/``-owned ``ParamSpec`` could never be imported from here without a
cycle -- the vocabulary has to live at this lower layer for both
``ParamSpec`` and ``utils/parameters.py`` to share one real source of truth
instead of ``utils/`` re-deriving it from ``models/`` each time.

``ParamSpec.POSITIONAL_ONLY`` (and its four siblings, plus
``ParamSpec._VALID_KINDS``) become read-only shims onto these same values --
the public surface `ParamSpec.POSITIONAL_ONLY` etc. is unchanged for every
existing caller; only where the canonical definition lives moves.
"""
from __future__ import annotations

POSITIONAL_ONLY = "POSITIONAL_ONLY"
POSITIONAL_OR_KEYWORD = "POSITIONAL_OR_KEYWORD"
VAR_POSITIONAL = "VAR_POSITIONAL"
KEYWORD_ONLY = "KEYWORD_ONLY"
VAR_KEYWORD = "VAR_KEYWORD"

VALID_KINDS: tuple[str, ...] = (
    POSITIONAL_ONLY,
    POSITIONAL_OR_KEYWORD,
    VAR_POSITIONAL,
    KEYWORD_ONLY,
    VAR_KEYWORD,
)

# Priority order for Python-compatible parameter ordering validation --
# position in VALID_KINDS is the priority itself, no separate literal list.
KIND_PRIORITY: dict[str, int] = {
    kind: priority for priority, kind in enumerate(VALID_KINDS)
}

__all__ = [
    "POSITIONAL_ONLY",
    "POSITIONAL_OR_KEYWORD",
    "VAR_POSITIONAL",
    "KEYWORD_ONLY",
    "VAR_KEYWORD",
    "VALID_KINDS",
    "KIND_PRIORITY",
]
