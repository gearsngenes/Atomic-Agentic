from __future__ import annotations

import re
from typing import Any

from ..exceptions import AgentError
from ..models.parameters import ParamSpec
from ..models.agents.prompts import PromptConfig
from ..constants.core import NO_VAL

__all__ = [
    "build_context_description",
    "extract_dependencies",
    "normalize_context_properties",
    "normalize_role_prompt",
]


def normalize_role_prompt(
    value: str | PromptConfig | None,
    default_template: str,
) -> PromptConfig:
    """Coerce a role-prompt value to a ``PromptConfig``."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return PromptConfig(
            template=default_template,
            description="Default assistant role prompt",
        )
    if isinstance(value, str):
        return PromptConfig(template=value.strip(), description="Role prompt")
    if isinstance(value, PromptConfig):
        return value
    raise TypeError(
        f"role_prompt must be str, PromptConfig, or None; got {type(value).__name__}."
    )


def build_context_description(params: list[ParamSpec]) -> str:
    """Build the auto-generated description for the ``context`` schema param."""
    required = [p for p in params if p.default is NO_VAL]
    optional = [p for p in params if p.default is not NO_VAL]
    parts: list[str] = []
    if required:
        entries = "; ".join(
            f"'{p.name}' ({p.type})" + (f" — {p.description}" if p.description else "")
            for p in required
        )
        parts.append(f"Required: {entries}")
    if optional:
        entries = "; ".join(
            f"'{p.name}' ({p.type}, default={p.default!r})"
            + (f" — {p.description}" if p.description else "")
            for p in optional
        )
        parts.append(f"Optional: {entries}")
    return ". ".join(parts)


def normalize_context_properties(
    context_properties: list[str] | list[ParamSpec] | None,
) -> list[ParamSpec]:
    """Normalize ``context_properties`` to a list of KEYWORD_ONLY ParamSpecs.

    ``list[str]``       → KEYWORD_ONLY ParamSpecs with ``default=NO_VAL``.
    ``list[ParamSpec]`` → coerced to KEYWORD_ONLY (variadic items rejected).
    ``None``            → ``[]``.
    Duplicate names and empty strings are rejected.
    """
    if context_properties is None:
        return []
    if not isinstance(context_properties, list):
        raise AgentError(
            "context_properties must be a list of str, a list of ParamSpec, or None."
        )
    variadic_kinds = {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}
    result: list[ParamSpec] = []
    seen: set[str] = set()
    for i, item in enumerate(context_properties):
        if isinstance(item, str):
            name = item.strip()
            if not name:
                raise AgentError(
                    f"context_properties[{i}] must be a non-empty string."
                )
            param = ParamSpec(
                name=name, index=i, kind=ParamSpec.KEYWORD_ONLY,
                type="Any", default=NO_VAL,
            )
        elif isinstance(item, ParamSpec):
            if item.kind in variadic_kinds:
                raise AgentError(
                    f"context_properties[{i}] ({item.name!r}) must not be variadic; "
                    f"got kind {item.kind!r}."
                )
            param = ParamSpec(
                name=item.name, index=i, kind=ParamSpec.KEYWORD_ONLY,
                type=item.type, default=item.default, description=item.description,
            )
        else:
            raise AgentError(
                f"context_properties items must be str or ParamSpec; "
                f"got {type(item).__name__} at index {i}."
            )
        if param.name in seen:
            raise AgentError(
                f"context_properties contains duplicate name {param.name!r}."
            )
        seen.add(param.name)
        result.append(param)
    return result


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
