from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pprint import pformat
from typing import Any

from ..constants.agents import (
    CONTEXT_END,
    CONTEXT_PREAMBLE,
    CONTEXT_START,
    TASK_PROMPT_END,
    TASK_PROMPT_START,
    THOUGHTS_END,
    THOUGHTS_START,
)
from ..models.agents.context import ContextVariable

__all__ = [
    "stringify_value",
    "elide_markers",
    "render_context_entries",
    "render_task_block",
    "render_thoughts_block",
]


def stringify_value(
    value: Any,
    *,
    mode: str,
    limit: int | None = None,
) -> str:
    """Render ``value`` as LLM-facing text, then apply ``limit``.

    1. Format by ``mode`` (one of ``RENDER_MODES``; anything else raises
       ``ValueError``): ``s`` ``str``, ``r`` ``repr``, ``a`` ``ascii``,
       ``j`` ``json.dumps(indent=2, ensure_ascii=False)`` with ``repr`` for
       any sub-value JSON cannot encode (and for the whole value when JSON
       fails outright, e.g. a circular reference or non-basic dict keys -- a
       deliberate exception to "natural exceptions", since a context value is
       unrestricted and a presentation failure must not kill the LLM call),
       ``p`` ``pformat(indent=2, sort_dicts=False)``.
    2. Limit: ``None`` leaves the text; ``0`` gives ``"value hidden"``; a
       longer text is cut and the truncation marker appended.

    Already-rendered text is limited by passing it with ``mode="s"``. A context
    entry never passes ``0`` here: ``render_context_entries`` replaces the whole
    value row instead.
    """
    if mode == "s":
        text = str(value)
    elif mode == "r":
        text = repr(value)
    elif mode == "a":
        text = ascii(value)
    elif mode == "j":
        try:
            text = json.dumps(value, indent=2, ensure_ascii=False, default=repr)
        except (TypeError, ValueError):
            text = repr(value)
    elif mode == "p":
        text = pformat(value, indent=2, sort_dicts=False)
    else:
        raise ValueError(f"stringify_value: unknown mode {mode!r}.")

    if limit is None:
        return text
    if limit == 0:
        return "value hidden"
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated: {limit:,} of {len(text):,} chars shown]"


def elide_markers(text: str) -> str:
    """Replace every exact occurrence of a framing marker with a placeholder.

    1. For each framing marker in (CONTEXT_START, CONTEXT_END,
       TASK_PROMPT_START, TASK_PROMPT_END, THOUGHTS_START, THOUGHTS_END),
       replace every exact occurrence with ``[marker elided]``.
    2. Return the text.

    Structural protection only: a variant with different spacing or case
    passes through untouched.
    """
    for marker in (
        CONTEXT_START, CONTEXT_END,
        TASK_PROMPT_START, TASK_PROMPT_END,
        THOUGHTS_START, THOUGHTS_END,
    ):
        text = text.replace(marker, "[marker elided]")
    return text


def render_context_entries(
    context: Mapping[str, ContextVariable],
    *,
    limit: int | None,
    name_prefix: str = "",
) -> str:
    """Render ``context`` as ``---``-separated labeled entries, or ``""``.

    1. An empty mapping renders nothing.
    2. Order: readonly first, then writable; described before undescribed
       within each group; insertion order within a group (stable sort).
    3. Each entry is labeled lines -- ``name``, ``type``, ``readonly: yes``
       (only when readonly), ``description`` (only when present, collapsed to
       one line), then ``value:`` and the value text. The value comes from
       ``stringify_value`` with the variable's own ``render_mode`` and the
       effective limit -- the smaller of ``limit`` and the variable's own
       ``preview_limit`` (``None`` is unlimited). An effective limit of ``0``
       replaces the value row with ``value hidden``. A multi-line value is
       fenced with backticks longer than any backtick run inside it.
       Description and value are marker-elided.
    4. ``name_prefix`` is prepended to each shown name (the agent family
       decides: ``"$"`` for sigil-referencing families, ``""`` otherwise).

    Pure function of its arguments: the same input renders byte-identically.
    """
    if not context:
        return ""

    ordered = sorted(
        context.values(), key=lambda v: (not v.readonly, v.description is None)
    )
    entries: list[str] = []
    for variable in ordered:
        lines = [f"name: {name_prefix}{variable.name}", f"type: {variable.type}"]
        if variable.readonly:
            lines.append("readonly: yes")
        if variable.description is not None:
            one_line = " ".join(variable.description.split())
            lines.append(f"description: {elide_markers(one_line)}")

        caps = [c for c in (limit, variable.preview_limit) if c is not None]
        effective = min(caps, default=None)
        if effective == 0:
            lines.append("value hidden")
            entries.append("\n".join(lines))
            continue

        lines.append("value:")
        text = elide_markers(
            stringify_value(variable.value, mode=variable.render_mode, limit=effective)
        )
        if "\n" in text or "\r" in text:
            longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
            fence = "`" * max(3, longest + 1)
            text = f"{fence}\n{text}\n{fence}"
        lines.append(text)
        entries.append("\n".join(lines))

    return "---\n" + "\n---\n".join(entries) + "\n---"


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


def render_thoughts_block(body: str) -> str:
    """Wrap ``body`` in the thoughts markers.

    1. Elide forged framing markers in ``body`` (``elide_markers``).
    2. Return ``THOUGHTS_START``, the body and ``THOUGHTS_END``, each on its own
       line.

    Pure wrap: no truncation, no formatting of the body. The caller decides
    whether to call it at all (the agent omits the block when there are no
    thoughts). An empty body yields the two markers with an empty line between
    them.
    """
    return f"{THOUGHTS_START}\n{elide_markers(body)}\n{THOUGHTS_END}"
