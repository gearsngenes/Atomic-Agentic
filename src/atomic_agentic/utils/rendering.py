from __future__ import annotations

import json
from collections.abc import Mapping
from pprint import pformat
from typing import Any

from ..constants.agents import (
    BACKTICK_RUN_PATTERN,
    CONTEXT_END,
    CONTEXT_ENTRY_SEPARATOR,
    CONTEXT_MARKERS,
    CONTEXT_PREAMBLE,
    CONTEXT_START,
    FULL_VALUE_POINTER_TEMPLATE,
    HIDDEN_VALUE_TEMPLATE,
    MARKER_ELIDED_TEXT,
    RENDER_MODES,
    TASK_PROMPT_END,
    TASK_PROMPT_START,
    TRUNCATION_TEMPLATE,
)
from ..models.agents.context import ContextVariable

__all__ = [
    "stringify_value",
    "elide_markers",
    "render_context_entries",
    "render_task_block",
]


def stringify_value(
    value: Any,
    *,
    mode: str | None = None,
    limit: int | None = None,
    ref: str | None = None,
) -> str:
    """Render ``value`` as LLM-facing text, then apply ``limit``.

    1. ``mode`` must be ``None`` or one of ``RENDER_MODES`` (``ValueError``).
    2. Format. ``None`` (history replay and tool-agent previews; a
       ``ContextVariable`` never uses it): a ``str`` as-is, anything else
       compact ``json.dumps`` (byte-identical to the retired
       ``stringify_result``). ``s``/``r``/``a``:
       ``str``/``repr``/``ascii``. ``j``: ``json.dumps(indent=2,
       ensure_ascii=False)``. ``p``: ``pformat(indent=2, sort_dicts=False)``.
       Both JSON paths fall back to ``(TypeName) repr`` when the value cannot
       be encoded -- a deliberate exception to "natural exceptions", since a
       context value is unrestricted and a presentation failure must not kill
       the LLM call.
    3. Limit: ``None`` leaves the text; ``0`` gives the hidden text; a longer
       text is cut and the truncation marker appended. With ``ref`` both
       markers end with a pointer to the full value.

    Already-rendered text is limited by passing it with ``mode=None``.
    """
    if mode is not None and mode not in RENDER_MODES:
        raise ValueError(
            f"stringify_value: mode must be None or one of {RENDER_MODES}, got {mode!r}."
        )

    if mode == "s":
        text = str(value)
    elif mode == "r":
        text = repr(value)
    elif mode == "a":
        text = ascii(value)
    elif mode == "p":
        text = pformat(value, indent=2, sort_dicts=False)
    elif mode is None and isinstance(value, str):
        text = value
    else:
        try:
            if mode is None:
                text = json.dumps(value)
            else:
                text = json.dumps(value, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            text = f"({type(value).__name__}) {value!r}"

    if limit is None:
        return text
    pointer = (
        FULL_VALUE_POINTER_TEMPLATE.format(ref=ref) if ref is not None else ""
    )
    if limit == 0:
        return HIDDEN_VALUE_TEMPLATE.format(pointer=pointer)
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATION_TEMPLATE.format(
        shown=limit, total=len(text), pointer=pointer
    )


def elide_markers(text: str) -> str:
    """Replace every exact occurrence of a framing marker with a placeholder.

    Structural protection only: a variant with different spacing or case
    passes through untouched.
    """
    for marker in CONTEXT_MARKERS:
        text = text.replace(marker, MARKER_ELIDED_TEXT)
    return text


def render_context_entries(
    context: Mapping[str, ContextVariable],
    *,
    limit: int | None,
    ref_prefix: str | None = None,
) -> str:
    """Render ``context`` as ``---``-separated labeled entries, or ``""``.

    1. An empty mapping renders nothing.
    2. Order: readonly first, then writable; described before undescribed
       within each group; insertion order within a group (stable sort).
    3. Each entry is labeled lines -- ``name``, ``type``, ``readonly: yes``
       (only when readonly), ``description`` (only when present, collapsed to
       one line), then ``value:`` and the value text. The value comes from
       ``stringify_value`` with the variable's own ``render_mode`` and
       ``limit``; a multi-line value is fenced with backticks longer than any
       backtick run inside it. Description and value are marker-elided.
    4. ``ref_prefix`` ``None`` renders no full-value pointer and a plain name;
       ``""`` or ``"$"`` prefixes the name and points to it.

    Pure function of its arguments: the same input renders byte-identically.
    """
    if not context:
        return ""

    ordered = sorted(
        context.values(), key=lambda v: (not v.readonly, v.description is None)
    )
    entries: list[str] = []
    for variable in ordered:
        shown_name = variable.name if ref_prefix is None else f"{ref_prefix}{variable.name}"
        lines = [f"name: {shown_name}", f"type: {variable.type}"]
        if variable.readonly:
            lines.append("readonly: yes")
        if variable.description is not None:
            one_line = " ".join(variable.description.split())
            lines.append(f"description: {elide_markers(one_line)}")
        lines.append("value:")

        text = elide_markers(
            stringify_value(
                variable.value,
                mode=variable.render_mode,
                limit=limit,
                ref=None if ref_prefix is None else shown_name,
            )
        )
        if "\n" in text or "\r" in text:
            longest = max(
                (len(run) for run in BACKTICK_RUN_PATTERN.findall(text)), default=0
            )
            fence = "`" * max(3, longest + 1)
            text = f"{fence}\n{text}\n{fence}"
        lines.append(text)
        entries.append("\n".join(lines))

    separator = CONTEXT_ENTRY_SEPARATOR
    return f"{separator}\n" + f"\n{separator}\n".join(entries) + f"\n{separator}"


def render_task_block(
    prompt: str,
    entries_text: str,
    *,
    context_first: bool,
) -> str:
    """Wrap ``prompt`` in the task markers and join the context block to it.

    The task block is always present. The context block (markers, preamble,
    entries) is present only when ``entries_text`` is non-empty; its entries
    arrive already elided from ``render_context_entries``, so only the prompt
    is elided here.
    """
    task = f"{TASK_PROMPT_START}\n{elide_markers(prompt)}\n{TASK_PROMPT_END}"
    if not entries_text:
        return task
    context = f"{CONTEXT_START}\n{CONTEXT_PREAMBLE}\n\n{entries_text}\n{CONTEXT_END}"
    return f"{context}\n\n{task}" if context_first else f"{task}\n\n{context}"
