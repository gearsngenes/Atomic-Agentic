from __future__ import annotations

import json
import re
from typing import Any

from ..constants.agents import THOUGHT_CATEGORIES
from ..constants.core import NO_VAL
from ..models.agents.prompts import PromptConfig
from ..models.agents.thought_models2 import AgentThought2

__all__ = [
    "extract_dependencies",
    "extract_json_object",
    "normalize_role_prompt",
    "normalize_thinking_instructions",
    "parse_thoughts",
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


def extract_json_object(raw_text: str, *, source_label: str) -> Any:
    """
    Extract the largest decodable JSON array/object from a possibly noisy string.

    Promoted from ``ToolAgent._extract_from_json_string`` — shared by any
    caller that needs to pull structured output out of free-form LLM text,
    not just ``ToolAgent`` and its subclasses. Behavior is unchanged from
    the original method except that the non-string-input case now raises a
    plain ``TypeError`` (an internal-contract violation, not a
    ``ToolAgent``-specific concern) instead of ``ToolAgentError``.

    This helper is intentionally shape-neutral:
    - It does not require the decoded value to be a list.
    - It does not require the decoded value to be a dict.
    - It does not validate any particular schema's fields.

    Parsing steps
    -------------
    1. Strip a single common markdown fence wrapper if present.
    2. Scan for candidate JSON array/object starts.
    3. Decode with ``json.JSONDecoder().raw_decode(...)``.
    4. Return the candidate with the largest decoded span.

    Parameters
    ----------
    raw_text : str
        Raw LLM output that may contain a JSON array/object surrounded by
        prose, markdown fences, or other text.
    source_label : str
        Identifies the caller in error messages (e.g.
        ``f"{type(self).__name__}.{self.name}"``).

    Returns
    -------
    Any
        The decoded Python value for the largest valid JSON array/object found.

    Raises
    ------
    TypeError
        If ``raw_text`` is not a string.
    json.JSONDecodeError
        If ``raw_text`` is empty or contains no decodable JSON array/object.
    """
    if not isinstance(raw_text, str):
        raise TypeError(f"{source_label}: LLM returned non-string output.")
    if not raw_text.strip():
        raise json.JSONDecodeError("LLM returned empty output", "", 0)

    text = raw_text.strip()

    # Strip a single fenced block wrapper if present.
    text = re.sub(r"^\s*```[a-zA-Z0-9]*\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text).strip()

    decoder = json.JSONDecoder()

    best_val: Any = NO_VAL
    best_span_len: int = -1

    # Candidate starts: JSON arrays or objects.
    for match in re.finditer(r"[\[{]", text):
        start = match.start()
        try:
            val, end_rel = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue

        if end_rel > best_span_len:
            best_span_len = end_rel
            best_val = val

    if best_val is NO_VAL:
        raise json.JSONDecodeError("no valid JSON array or object found in LLM output", text, 0)

    return best_val


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


def normalize_thinking_instructions(value: str | PromptConfig | None) -> PromptConfig:
    """
    Coerce an ``Agent2`` ``thinking_instructions`` value to a ``PromptConfig``.

    Structurally mirrors ``normalize_role_prompt`` but is kept as its own
    function rather than a shared/generalized one: reusing
    ``normalize_role_prompt`` directly would surface its hardcoded
    ``"role_prompt must be..."`` message for a ``thinking_instructions``
    caller, which would be actively misleading. Unlike a role prompt, a
    ``None``/empty ``thinking_instructions`` defaults to an empty template
    (no persona filler needed -- the base ``"think"`` prompt stands alone).
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return PromptConfig(template="", description="No thinking instructions")
    if isinstance(value, str):
        return PromptConfig(template=value.strip(), description="Thinking instructions")
    if isinstance(value, PromptConfig):
        return value
    raise TypeError(
        f"thinking_instructions must be str, PromptConfig, or None; got {type(value).__name__}."
    )


_THOUGHT_MARKER_PATTERN = re.compile(
    r"^\s*\[(" + "|".join(THOUGHT_CATEGORIES) + r")\]\s*",
    re.MULTILINE | re.IGNORECASE,
)


def parse_thoughts(text: str) -> list[AgentThought2]:
    """
    Parse one thinking round's raw text into a list of ``AgentThought2``.

    Line-based, lax format: each category marker (``[CATEGORY]``, any
    casing, no colon, anchored at a line start) begins a new thought; its
    content runs until the next marker or the end of ``text``. Bracketed,
    colon-free, to match ``_format_thoughts``'s own rendering of prior
    thoughts exactly -- what's shown back to the model round after round as
    its own history is what it's asked to keep producing, closing the loop
    a bare colon-terminated form (``CATEGORY:``) previously left open (a
    model imitating its own bracketed history would drift away from a
    colon-based instructed format and fail to parse). If no marker is found
    anywhere, the entire (stripped) text becomes a single ``OTHER``-category
    thought -- unless it's empty/whitespace-only, in which case no thought
    is produced at all (an empty prefix isn't unparseable content, it's
    simply no content).

    Does not know about ``|STOP_THINKING|`` -- callers (``Agent2.think``)
    strip that before calling this function, keeping parsing pure and
    independently testable.
    """
    matches = list(_THOUGHT_MARKER_PATTERN.finditer(text))

    if not matches:
        stripped = text.strip()
        return [AgentThought2(category="OTHER", content=stripped)] if stripped else []

    thoughts: list[AgentThought2] = []
    for index, match in enumerate(matches):
        category = match.group(1).upper()
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()
        thoughts.append(AgentThought2(category=category, content=content))
    return thoughts
