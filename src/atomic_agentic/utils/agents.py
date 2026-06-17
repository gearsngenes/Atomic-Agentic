from __future__ import annotations

import re
from typing import Any

__all__ = ["extract_dependencies"]


def extract_dependencies(obj: Any, placeholder_pattern: re.Pattern[str]) -> set[int]:
    """
    Recursively extract all placeholder references from an object.

    Scans the object for occurrences of a given placeholder pattern (e.g., ``<<__sN__>>``)
    and returns the set of all referenced indices. Used during planning to extract
    dependencies between steps.

    Parameters
    ----------
    obj : Any
        Object to scan. Typically a dict (tool args) but can be any nested structure
        (lists, tuples, dicts, sets, scalars).
    placeholder_pattern : re.Pattern[str]
        Compiled regex pattern matching placeholders. Usually:
        - ``STEP_REF_PATTERN`` for step refs (``<<__sN__>>``)
        - ``CACHE_REF_PATTERN`` for cache refs (``<<__cN__>>``)

    Returns
    -------
    set[int]
        Set of all indices found (0-based). Empty set if no placeholders found.

    Validation
    ~~~~~~~~~~
    This method performs **NO validation** of the found indices:
    - Does NOT check bounds (N might be >= blackboard length)
    - Does NOT check execution status (referenced slot might not be executed yet)
    - Purely structural scanning

    Validation happens later in ``_resolve_placeholders()`` at prepare time.

    Examples
    --------
    >>> pattern = STEP_REF_PATTERN  # Matches <<__sN__>>
    >>> obj = {"query": "<<__s0__>>", "context": ["<<__s1__>>", "<<__s0__>>"]}
    >>> extract_dependencies(obj, pattern)
    {0, 1}

    >>> obj = {"static": "no placeholders here"}
    >>> extract_dependencies(obj, pattern)
    set()
    """
    deps: set[int] = set()

    def walk(x: Any) -> None:
        if isinstance(x, str):
            for m in placeholder_pattern.finditer(x):
                deps.add(int(m.group(1)))
            return
        if isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)
            return

    walk(obj)
    return deps
