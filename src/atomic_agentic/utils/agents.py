from __future__ import annotations

import ast
import json
import re
from typing import Any


from ..constants.core import NO_VAL
from ..models.agents.prompts import PromptConfig
from ..models.agents.thought_models import AgentThought
from ..constants.agents import (
    THOUGHT_MARKER_PATTERN,
    PLACEHOLDER_SHAPE_PATTERN,
    REGEX_BLOCK_TAGS,
    REGEX_STEP_TAGS,
    REGEX_STEP_TAG_TO_FIELD,
    CALL_TAG,
    RETURN_TAG,
    TOOL_FIELD,
    ARGS_FIELD,
    REASON_FIELD,
    RETURN_TOOL_FULL_NAME,
    RETURN_TOOL_REASON_TEXT,
    RETURN_VALUE_FIELD,
)

__all__ = [
    "extract_dependencies",
    "extract_json_object",
    "extract_regex_steps",
    "format_generation_issues",
    "normalize_role_prompt",
    "normalize_thinking_instructions",
]

# Mirrors THOUGHT_MARKER_PATTERN's shape (constants/agents.py) exactly:
# line-anchored, case-insensitive, optional leading whitespace.
_REGEX_BLOCK_PATTERN = re.compile(
    r"^\s*\[(" + "|".join(REGEX_BLOCK_TAGS) + r")\]\s*",
    re.MULTILINE | re.IGNORECASE,
)
_REGEX_STEP_TAG_PATTERN = re.compile(
    r"^\s*\[(" + "|".join(REGEX_STEP_TAGS) + r")\]\s*",
    re.MULTILINE | re.IGNORECASE,
)


def _placeholder_quoting_hint(text: str) -> str:
    """Return an extra hint sentence when ``text`` contains an unquoted
    placeholder-shaped token (e.g. ``|STEP.0|``), else ``""``."""
    if PLACEHOLDER_SHAPE_PATTERN.search(text):
        return (
            " This looks like it contains a placeholder (e.g. |STEP.0|) "
            "written unquoted -- placeholders must be written as, or "
            "inside, a quoted Python string."
        )
    return ""


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
) -> PromptConfig:
    """Coerce a thinking-instructions value to a ``PromptConfig``.

    Mirrors ``normalize_role_prompt`` exactly, but with an empty template
    as the default rather than a persona sentence — ``None``/blank means
    "no additional thinking instructions," and the caller's own
    header/footer-wrapping logic (``SelfAskAgent._render_system_message``)
    already treats an empty rendered result as invisible, so an empty
    template here is sufficient, not a special case.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return PromptConfig(
            template="",
            description="No additional thinking instructions.",
        )
    if isinstance(value, str):
        return PromptConfig(template=value.strip(), description="Thinking instructions")
    if isinstance(value, PromptConfig):
        return value
    raise TypeError(
        f"thinking_instructions must be str, PromptConfig, or None; got {type(value).__name__}."
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


def extract_regex_steps(raw_text: str, *, source_label: str) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Parse a possibly-noisy regex-mode LLM output string into step dicts.

    Peer to ``extract_json_object`` -- same "no family/class knowledge"
    purity, same non-string-input contract, same tolerance of surrounding
    prose. Unlike ``extract_json_object``, never raises for "nothing
    recognizable found" -- that case returns an empty list, since a
    tag-grammar has no equivalent of JSON's "syntactically valid empty
    array" to distinguish from "totally unparseable."

    Comprehensive, not fail-fast: a malformed per-tag literal payload drops
    only that one key from its block's dict and adds one issue string
    explaining why -- it does not drop the whole block, and it does not
    stop the scan of the rest of ``raw_text``. A ``[CALL]`` block whose tool
    line isn't a parseable call expression at all is the one exception --
    the tool identity itself is unrecoverable, so the whole block is
    dropped (see below).

    A bare ``[RETURN] <literal>`` block desugars into a return-tool call
    dict -- the literal itself is never a ``{"val": ...}`` dict; the model
    writes only the bare value and this function performs the wrapping.

    A ``[CALL]`` block's tool line is a keyword-only Python call expression,
    ``tool.id(key=value, ...)`` (a zero-argument call still needs empty
    parentheses: ``tool.id()``), parsed via a single ``ast.parse(...,
    mode="eval")`` attempt -- no repair, no retry; a parse failure reports
    the raw exception and drops the block. Positional arguments are
    silently ignored (no tool schema at this layer to map position to
    parameter name). ``**`` unpacking is accepted only when it
    literal_evals to an actual dict -- its keys are merged in as an
    ordinary keyword would be; anything else is an issue.

    Parameters
    ----------
    raw_text : str
        Raw LLM output that may contain one or more ``[CALL]``/``[RETURN]``
        blocks surrounded by prose.
    source_label : str
        Identifies the caller in the ``TypeError`` message (e.g.
        ``f"{type(self).__name__}.{self.name}"``).

    Returns
    -------
    tuple[list[dict[str, Any]], list[str]]
        ``(steps, issues)``. ``steps`` has one dict per recognized,
        successfully tool-identified block, in the order encountered; each
        dict has a key per recognized field that was actually present and
        parsed successfully -- absence is meaningful and left for
        downstream required-field validation to report. ``issues`` has one
        string per detected parse problem, referencing the 0-based block
        index it occurred in; empty when nothing went wrong.

    Raises
    ------
    TypeError
        If ``raw_text`` is not a ``str``.
    """
    if not isinstance(raw_text, str):
        raise TypeError(f"{source_label}: LLM returned non-string output.")

    block_matches = list(_REGEX_BLOCK_PATTERN.finditer(raw_text))
    if not block_matches:
        return [], []

    steps: list[dict[str, Any]] = []
    issues: list[str] = []

    for i, match in enumerate(block_matches):
        block_tag = match.group(1).upper()
        content_start = match.end()
        content_end = (
            block_matches[i + 1].start() if i + 1 < len(block_matches) else len(raw_text)
        )
        content = raw_text[content_start:content_end]

        if block_tag == RETURN_TAG:
            raw = content.strip()
            if not raw:
                # Inferred, not taught: an empty payload becomes an explicit
                # [RETURN] None rather than being rejected -- a silent
                # code-level backstop, never a documented model-facing
                # affordance.
                value = None
            else:
                try:
                    value = ast.literal_eval(raw)
                except (SyntaxError, ValueError):
                    # Retry against the first line only, so trailing text a
                    # weaker model appended after the real value isn't
                    # folded in. If that still fails, keep the first line as
                    # plain text -- return_tool's "val: Any" and lack of any
                    # downstream consumer make that an acceptable outcome.
                    first_line = raw.splitlines()[0].strip()
                    try:
                        value = ast.literal_eval(first_line)
                    except (SyntaxError, ValueError):
                        value = first_line
            steps.append({
                TOOL_FIELD: RETURN_TOOL_FULL_NAME,
                ARGS_FIELD: {RETURN_VALUE_FIELD: value},
                REASON_FIELD: RETURN_TOOL_REASON_TEXT,
            })
            continue

        assert block_tag == CALL_TAG  # only other member of REGEX_BLOCK_TAGS
        sub_matches = list(_REGEX_STEP_TAG_PATTERN.finditer(content))
        tool_text = (content[: sub_matches[0].start()] if sub_matches else content).strip()

        try:
            call_expr = ast.parse(tool_text, mode="eval").body
        except SyntaxError as exc:
            # No repair, no retry: unlike a scalar RETURN value, a call's
            # name+args structure can't be stringified without losing it,
            # so any parse failure here is unconditionally terminal for
            # this block.
            issues.append(
                f"block {i}: [{CALL_TAG}] payload {tool_text!r} could not be parsed "
                f"as a call expression like tool.id(key=value, ...). "
                f"{type(exc).__name__}: {exc}."
                + _placeholder_quoting_hint(tool_text)
            )
            continue
        if not isinstance(call_expr, ast.Call):
            issues.append(
                f"block {i}: [{CALL_TAG}] payload {tool_text!r} did not evaluate to "
                f"a call expression; got {type(call_expr).__name__}."
            )
            continue

        step_dict: dict[str, Any] = {TOOL_FIELD: ast.unparse(call_expr.func), ARGS_FIELD: {}}

        # Positional arguments are silently ignored, not flagged -- no tool
        # schema is available at this layer to map position to parameter
        # name, so there's nothing actionable to report beyond what the
        # prompt already teaches (keyword-only). A tool call missing a
        # parameter it needed surfaces naturally and visibly when the tool
        # executes.

        for kw in call_expr.keywords:
            if kw.arg is None:
                # arg=None only means "** was used" -- it does NOT guarantee
                # the unpacked value is dict-shaped (Python's parser accepts
                # ** before any expression; "must be a mapping" is a
                # runtime-only constraint). Safety net: merge only if it
                # actually literal_evals to a dict.
                try:
                    unpacked = ast.literal_eval(kw.value)
                except (SyntaxError, ValueError):
                    unpacked = None
                if isinstance(unpacked, dict):
                    step_dict[ARGS_FIELD].update(unpacked)
                else:
                    issues.append(
                        f"block {i}: [{CALL_TAG}] does not support ** unpacking "
                        f"({ast.unparse(kw.value)}); pass a dict-shaped parameter as "
                        f"an ordinary keyword argument instead (e.g. tool.id(extra={{'k': 'v'}}))."
                    )
                continue
            try:
                step_dict[ARGS_FIELD][kw.arg] = ast.literal_eval(kw.value)
            except (SyntaxError, ValueError) as exc:
                # Some other syntactically valid but non-literal expression
                # (a bare name, a nested call). The stringified source is
                # still recovered into step_dict for observability, but
                # this generation attempt is rejected via `issues` either
                # way, so the fallback value never reaches execution.
                segment = ast.get_source_segment(tool_text, kw.value)
                fallback_text = segment.strip() if segment else ""
                step_dict[ARGS_FIELD][kw.arg] = fallback_text
                issues.append(
                    f"block {i}: [{CALL_TAG}] argument {kw.arg!r} could not be parsed "
                    f"as a Python literal: {type(exc).__name__}: {exc}."
                    + _placeholder_quoting_hint(fallback_text)
                )

        for j, sub_match in enumerate(sub_matches):
            field_tag = sub_match.group(1).upper()
            field = REGEX_STEP_TAG_TO_FIELD[field_tag]
            payload_start = sub_match.end()
            payload_end = (
                sub_matches[j + 1].start() if j + 1 < len(sub_matches) else len(content)
            )
            payload = content[payload_start:payload_end].strip()

            if field == REASON_FIELD:
                step_dict[field] = payload or None
                continue

            try:
                step_dict[field] = ast.literal_eval(payload)
            except (SyntaxError, ValueError) as exc:
                issues.append(
                    f"block {i}: [{field_tag}] payload {payload!r} did not parse: {exc}"
                    + _placeholder_quoting_hint(payload)
                )

        steps.append(step_dict)

    return steps, issues


def extract_dependencies(obj: Any, placeholder_pattern: re.Pattern[str]) -> set[int]:
    """
    Recursively extract all placeholder references from an object.

    Scans the object for occurrences of a given placeholder pattern (e.g., ``|STEP.N|``)
    and returns the set of all referenced indices. Used during planning to extract
    dependencies between steps.

    Parameters
    ----------
    obj : Any
        Object to scan. Typically a dict (tool args) but can be any nested structure
        (lists, tuples, dicts, sets, scalars).
    placeholder_pattern : re.Pattern[str]
        Compiled regex pattern matching placeholders. Usually:
        - ``STEP_REF_PATTERN`` for step refs (``|STEP.N|``)
        - ``CACHE_REF_PATTERN`` for cache refs (``|CACHE.N|``)

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
    >>> pattern = STEP_REF_PATTERN  # Matches |STEP.N|
    >>> obj = {"query": "|STEP.0|", "context": ["|STEP.1|", "|STEP.0|"]}
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

def format_generation_issues(issues: list[str], *, category_header: str | None = None) -> str:
    """
    Join one or more detected generation-output problems into a single
    LLM-facing feedback message, ready to inject as a retry turn verbatim.

    Raises ``ValueError`` if ``issues`` is empty -- callers only invoke this
    once they have at least one real issue; a hollow success-shaped message
    would be a caller bug, not a legitimate empty-feedback case.

    category_header : str | None
        When supplied, this exact text (already including any trailing
        colon/newline the caller wants) is prepended to the formatted body
        -- regardless of whether there are 1 or many issues, matching the
        calling convention of a caller that always frames its category
        message. When multiple issues are present, the built-in "Multiple
        problems were found in your output:" line is *not* also emitted --
        ``category_header`` replaces it rather than stacking with it. When
        ``None`` (default), single-issue passthrough and the built-in
        multi-issue wrapper apply exactly as before.
    """
    if not issues:
        raise ValueError("format_generation_issues requires at least one issue.")

    if len(issues) == 1:
        body = issues[0]
    else:
        numbered = "\n".join(f"{i}. {issue}" for i, issue in enumerate(issues, start=1))
        if category_header is None:
            body = (
                "Multiple problems were found in your output:\n"
                f"{numbered}\n\n"
                "Correct all of the above and resubmit."
            )
        else:
            body = f"{numbered}\n\nCorrect all of the above and resubmit."

    return f"{category_header}{body}" if category_header is not None else body


def parse_thoughts(text: str) -> list[AgentThought]:
    """
    Parse one thinking round's raw text into a list of ``AgentThought``.

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

    Does not know about ``|STOP_THINKING|`` -- callers (``SelfAskAgent.think``)
    strip that before calling this function, keeping parsing pure and
    independently testable.
    """
    matches = list(THOUGHT_MARKER_PATTERN.finditer(text))

    if not matches:
        stripped = text.strip()
        return [AgentThought(category="OTHER", content=stripped)] if stripped else []

    thoughts: list[AgentThought] = []
    for index, match in enumerate(matches):
        category = match.group(1).upper()
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()
        thoughts.append(AgentThought(category=category, content=content))
    return thoughts
