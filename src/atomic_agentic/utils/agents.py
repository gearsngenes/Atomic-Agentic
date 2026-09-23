from __future__ import annotations

import ast
import json
import re
from typing import Any


from ..constants.agents import (
    CODE_FENCE_PATTERN,
    DUNDER_ATTRIBUTE_PATTERN,
    LEADING_CODE_FENCE_PATTERN,
    TRAILING_CODE_FENCE_PATTERN,
    UNSUPPORTED_EXPR_LABELS,
)
from ..constants.core import NO_VAL
from ..exceptions import BlackboardParseError
from ..models.agents.prompts import PromptConfig

__all__ = [
    "evaluate_expr",
    "extract_dependencies",
    "extract_identifiers",
    "extract_json_object",
    "normalize_role_prompt",
    "normalize_thinking_instructions",
    "reject_unsupported_forms",
    "stringify_result",
    "strip_code_fence",
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


def normalize_thinking_instructions(
    value: str | PromptConfig | None,
    default_template: str,
) -> PromptConfig:
    """Coerce a thinking-instructions value to a ``PromptConfig``.

    Mirrors ``normalize_role_prompt`` exactly, including now taking a
    caller-supplied ``default_template`` -- ``None``/blank resolves to
    that default (``ThinkingAgent.DEFAULT_THINKING_PROMPT``) rather than a
    hardcoded empty string, matching how ``normalize_role_prompt`` always
    resolved to a real default persona sentence.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return PromptConfig(
            template=default_template,
            description="Default thinking instructions.",
        )
    if isinstance(value, str):
        return PromptConfig(template=value.strip(), description="Thinking instructions")
    if isinstance(value, PromptConfig):
        return value
    raise TypeError(
        f"thinking_instructions must be str, PromptConfig, or None; got {type(value).__name__}."
    )


def stringify_result(value: str | int | float | bool | list | dict | None) -> str:
    """Render an agent-produced result value as LLM-facing display text --
    a ``str`` value used as-is, any other JSON-decodable value
    ``json.dumps``-rendered. Shared by every render path that replays a
    structured (``response_schema``/``thinking_schema``) result back into
    text (``Agent.render_turn``, ``ThinkingAgent._stringify_thought``), so
    they can't drift on how a non-str value gets stringified."""
    return value if isinstance(value, str) else json.dumps(value)


def extract_json_object(raw_text: str, *, source_label: str) -> Any:
    """
    Extract the largest decodable JSON array/object from a possibly noisy string.

    Originally promoted from the old ``ToolAgent._extract_from_json_string``
    — shared by any caller that needs to pull structured output out of
    free-form LLM text. That original method has since been removed
    entirely (see `json-tool-agent-rename`'s lifecycle-slimming addendum —
    ``output_structure`` strict mode makes free-text JSON extraction
    unnecessary for the current ``JsonToolAgent`` family), but this
    promoted utility remains available to any other caller that still
    needs it. Behavior is unchanged from the original method except that
    the non-string-input case raises a plain ``TypeError`` (an
    internal-contract violation, not caller-specific) instead of
    ``ToolAgentError``.

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


def extract_identifiers(
    source: ast.expr | dict[str, Any] | tuple[Any, ...] | list[Any],
) -> list[str]:
    """
    Walk any ``ast.Name`` reference in ``source`` and return every referenced
    identifier, deduplicated in first-seen order (not a set: multiplicity
    isn't meaningful for a dependency list, but a list keeps a stable,
    orderable contract). Used by ``ScriptAgent`` (``utils/script.py``) to
    find a statement's real dependencies from its parsed argument tree.

    Accepts a single parsed expression node, a slot's ``kwargs`` dict, or a
    slot's ``args`` tuple/list -- in the dict/tuple/list forms, only values
    that are still unresolved ``ast.expr`` nodes contribute identifiers; an
    already-folded raw literal value contributes none. A ``CodeStatement``
    with both containers calls this once per container and merges the
    results -- this function stays single-container. An ``ast.Starred``
    entry (a ``*expr`` unpack in ``args``) is itself an ``ast.expr``
    subtype, so it's picked up by the plain expression branch below with no
    special-casing: ``ast.walk`` already recurses into its ``.value``.
    """
    raw: list[str] = []

    if isinstance(source, dict):
        for value in source.values():
            if isinstance(value, ast.expr):
                raw.extend(extract_identifiers(value))
    elif isinstance(source, (tuple, list)):
        for value in source:
            if isinstance(value, ast.expr):
                raw.extend(extract_identifiers(value))
    else:
        raw.extend(
            node.id
            for node in ast.walk(source)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )

    return list(dict.fromkeys(raw))


def strip_code_fence(raw_text: str) -> str:
    """
    Strip a markdown code fence wrapping generated text, if present --
    defensive against a model wrapping otherwise-valid output in a code
    fence despite being told not to. Generic to any language tag (or none)
    on the opening fence line. Used by ``ScriptAgent``'s statement parsing
    (``utils/script.py``).

    Tries a fully matched pair first (``CODE_FENCE_PATTERN``) -- unambiguous,
    so its captured inner text is used as-is. If that doesn't match (a model
    emitting only one side), falls back to stripping a leading and/or
    trailing fence line independently. Either way, a fence appearing only
    mid-text is left alone (``ast.parse`` will reject that on its own terms,
    as a real structural problem).
    """
    full_match = CODE_FENCE_PATTERN.match(raw_text)
    if full_match:
        return full_match.group(1)
    text = LEADING_CODE_FENCE_PATTERN.sub("", raw_text, count=1)
    text = TRAILING_CODE_FENCE_PATTERN.sub("", text, count=1)
    return text


def evaluate_expr(node: ast.expr, namespace: dict[str, Any]) -> Any:
    """
    Evaluate one parsed expression node against a namespace, with no
    builtins available. Used by ``ScriptAgent`` (``utils/script.py``, safe
    because every arg reaching this function is guaranteed Call-free by its
    hoisting rule) -- nothing reachable through ``namespace`` can itself be
    invoked.

    Raises whatever the evaluation naturally raises (``TypeError``,
    ``ZeroDivisionError``, ``NameError``, ``KeyError``, ...), uncaught --
    callers decide whether to wrap (parse-time constant folding) or let it
    surface naturally (``resolve_slot_args``/``resolve_call_args``).
    """
    expr_wrapper = ast.Expression(body=node)
    ast.fix_missing_locations(expr_wrapper)
    code = compile(expr_wrapper, filename="<blackboard-slot-v2>", mode="eval")
    return eval(code, {"__builtins__": {}}, namespace)


def reject_unsupported_forms(node: ast.expr) -> None:
    """
    Walk ``node`` and raise on any of: a ternary (``ast.IfExp``) whose
    either branch contains a ``Call``, an ``ast.Await`` anywhere, a
    comprehension/lambda (``UNSUPPORTED_EXPR_LABELS``) anywhere, or an
    ``ast.Attribute`` whose ``.attr`` matches ``DUNDER_ATTRIBUTE_PATTERN``
    anywhere -- the exact set ``ScriptAgent`` (``utils/script.py``) needs.

    Raises before any hoisting/unparsing proceeds -- every check here is
    unconditional over the whole tree passed in, at any depth, regardless
    of whether it actually contains the form being checked for.
    """
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.IfExp) and (
            any(isinstance(n, ast.Call) for n in ast.walk(candidate.body))
            or any(isinstance(n, ast.Call) for n in ast.walk(candidate.orelse))
        ):
            raise BlackboardParseError(
                "conditional expression branches must not contain tool "
                "calls (wastes budget evaluating the untaken branch): "
                f"{ast.unparse(candidate)!r} -- restructure as separate "
                "statements or a pause."
            )

        if isinstance(candidate, ast.Await):
            raise BlackboardParseError(
                "'await' is not supported: "
                f"{ast.unparse(candidate)!r} -- write the call as an "
                "ordinary statement; execution order is inferred "
                "automatically from data dependencies."
            )

        label = UNSUPPORTED_EXPR_LABELS.get(type(candidate))
        if label is not None:
            raise BlackboardParseError(
                f"{label} expressions are not supported: "
                f"{ast.unparse(candidate)!r} -- rewrite as explicit "
                "statements instead."
            )

        if isinstance(candidate, ast.Attribute) and DUNDER_ATTRIBUTE_PATTERN.fullmatch(candidate.attr):
            raise BlackboardParseError(
                f"dunder attribute access is not permitted: {ast.unparse(candidate)!r}."
            )
