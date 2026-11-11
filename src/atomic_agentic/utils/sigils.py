from __future__ import annotations

import ast
import json
from copy import deepcopy
from typing import Any, Iterable, NamedTuple, Optional

from ..constants.agents import (
    PLANACT_OUTPUT_SCHEMA,
    REACT_OUTPUT_SCHEMA,
    RETURN_ALIAS,
    RETURN_VALUE_FIELD,
    SIGIL_REF_PATTERN,
    TASK_RESULT_PREFIX,
)
from ..constants.core import IDENTIFIER_PATTERN
from ..models.agents.blackboard_models import ToolStatement
from .agents import extract_identifiers

__all__ = [
    "build_planact_schema",
    "build_react_schema",
    "parse_generation",
    "parse_react_call",
    "translate_calls",
    "find_cascade_failures",
    "serialize_call",
    "render_completed_as_json",
    "render_failed_as_json",
]


class _DraftCall(NamedTuple):
    """
    Private intermediate shape: one plan/step entry's raw, untranslated
    fields straight from the JSON payload -- identifier/tool as given,
    args/kwargs as plain already-typed JSON scalars (never ``ast``). Exists
    only between ``parse_generation``/``parse_react_call`` and
    ``translate_calls``; never constructed or consumed anywhere else.
    """
    identifier: Optional[str]
    tool: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    duplicate_kwargs: tuple[str, ...] = ()


def build_planact_schema(tool_names: Iterable[str]) -> dict[str, Any]:
    """
    Return a fresh, per-call copy of ``PLANACT_OUTPUT_SCHEMA`` with
    ``call``'s ``enum`` populated from the currently-registered toolset --
    makes an unregistered ``call`` structurally impossible under strict-mode
    generation. Deep-copied so no two calls ever alias the same mutable
    dict, and the module-level template itself is never mutated.
    """
    schema = deepcopy(PLANACT_OUTPUT_SCHEMA)
    schema["properties"]["plan"]["items"]["properties"]["call"]["enum"] = sorted(tool_names)
    return schema


def build_react_schema(tool_names: Iterable[str]) -> dict[str, Any]:
    """
    Sibling to ``build_planact_schema`` -- identical pattern, built off
    ``REACT_OUTPUT_SCHEMA`` instead. No restriction parameter -- a caller
    wanting the enum narrowed to ``return`` only (the budget-boundary case)
    passes a pre-narrowed ``tool_names`` iterable itself (e.g.
    ``[RETURN_TOOL_NAME]``); this function always does exactly one thing
    with whatever it's given.
    """
    schema = deepcopy(REACT_OUTPUT_SCHEMA)
    schema["properties"]["call"]["enum"] = sorted(tool_names)
    return schema


def parse_generation(payload: dict[str, Any]) -> list[_DraftCall]:
    """
    Normalize ``output_structure``'s already-schema-validated payload (today:
    ``PLANACT_OUTPUT_SCHEMA``) into a flat list of draft calls. Pure shape
    unpacking only -- no decoding, no ``ast`` translation of any kind
    happens here (that's ``translate_calls``'s job, next). Every
    ``arguments[].value``/non-null ``return`` is carried through byte-for-
    byte as the schema handed it back.

    No malformed-shape failure mode -- every required key is schema-
    guaranteed present and type-correct; this function never raises.

    1. One ``_DraftCall`` is built per ``plan`` entry. The top-level
       ``summary`` field is never read -- decoding-order scaffold only.
    2. A keyword name repeated within one entry's ``arguments`` is recorded
       in that draft's ``duplicate_kwargs`` (once per repeated name, in
       first-repeat order); the LAST value is kept in ``kwargs``.
    3. A trailing ``RETURN_ALIAS`` draft is always synthesized and appended,
       carrying ``payload["return"]`` under ``RETURN_VALUE_FIELD`` -- a
       one-shot planner has no deferral concept, so synthesis is
       unconditional. ``payload["return"]`` stays a bare subscript,
       deliberately -- the schema requires this key, so a missing/malformed
       value here is a real schema-contract violation that should surface
       as a natural ``KeyError``, not be defensively guarded against.
    """
    drafts: list[_DraftCall] = []
    for entry in payload["plan"]:
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        duplicates: list[str] = []
        for arg in entry["arguments"]:
            if arg["name"] is None:
                args.append(arg["value"])
            else:
                if arg["name"] in kwargs and arg["name"] not in duplicates:
                    duplicates.append(arg["name"])
                kwargs[arg["name"]] = arg["value"]
        drafts.append(
            _DraftCall(
                identifier=entry["result_name"],
                tool=entry["call"],
                args=tuple(args),
                kwargs=kwargs,
                duplicate_kwargs=tuple(duplicates),
            )
        )

    drafts.append(
        _DraftCall(
            identifier=None,
            tool=RETURN_ALIAS,
            args=(),
            kwargs={RETURN_VALUE_FIELD: payload["return"]},
        )
    )

    return drafts


def parse_react_call(payload: dict[str, Any]) -> _DraftCall:
    """
    Normalize ``REACT_OUTPUT_SCHEMA``'s already-schema-validated flat
    payload into the one ``_DraftCall`` it describes. Pure shape unpacking,
    no decoding of any kind (matches ``parse_generation``'s own contract).

    No malformed-shape failure mode -- every required key is schema-
    guaranteed present and type-correct; this function never raises.

    ``payload["summary"]`` is never read -- decoding-order scaffold only,
    same treatment ``PLANACT_OUTPUT_SCHEMA``'s own ``"summary"`` field
    already gets. No ``plan``-array loop (there is none -- exactly one call
    per payload) and no ``RETURN_ALIAS`` synthesis (``return`` is an
    ordinary, really-dispatched ``call`` value in this family, never a
    separate top-level field to synthesize a sentinel call from).

    A repeated keyword name is recorded in ``duplicate_kwargs`` exactly as
    ``parse_generation`` does.
    """
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    duplicates: list[str] = []
    for arg in payload["arguments"]:
        if arg["name"] is None:
            args.append(arg["value"])
        else:
            if arg["name"] in kwargs and arg["name"] not in duplicates:
                duplicates.append(arg["name"])
            kwargs[arg["name"]] = arg["value"]

    return _DraftCall(
        identifier=payload["result_name"],
        tool=payload["call"],
        args=tuple(args),
        kwargs=kwargs,
        duplicate_kwargs=tuple(duplicates),
    )


def translate_calls(
    drafts: list[_DraftCall],
    tool_calls_limit: Optional[int],
    context_names: frozenset[str],
    readonly_names: frozenset[str],
    tool_names: frozenset[str],
) -> tuple[list[str], list[ToolStatement]]:
    """
    Does both validation and ast-translation in one walk, since both need
    the identical growing "available names" set.

    ``context_names`` is the full reference-*resolution* seed -- every
    variable in the task's context. ``readonly_names`` is the subset of
    those that cannot be reassigned. ``tool_names`` is every registered tool
    id, which a ``result_name`` may not take.

    1. ``available = set(context_names)``. ``issues = []``. ``statements = []``.
    2. For each draft, in order:
       a. Normalize identifier: if not ``None``, strip whitespace, strip one
          leading ``$`` if present, collapse blank-after-stripping to
          ``None``.
       b. If normalized identifier is not ``None``, check in order (first
          match wins): ``IDENTIFIER_PATTERN`` legality; the reserved
          ``task_result_`` prefix; membership in ``readonly_names``;
          membership in ``tool_names``; dunder-shape (``__...__``). A name in ``context_names`` that is
          not readonly passes (it updates that variable).
       c. If the draft repeated a keyword name, append an issue.
       d. Translate every arg/kwarg value (see "Value translation" below),
          using ``available`` as it stands *before* this draft's own
          identifier is added to it (a call can never reference its own
          result_name).
       e. Build a ``ToolStatement`` and append to ``statements`` --
          built unconditionally (even on an issue), so the full list is
          available for inspection, but only meaningful to the caller once
          ``issues`` is empty.
       f. If normalized identifier is not ``None`` and legal: add it to
          ``available``.
    3. Budget check: count drafts whose ``tool != RETURN_ALIAS``; if
       ``tool_calls_limit`` is not ``None`` and that count exceeds it,
       append a budget-exceeded issue message.
    4. Return ``(issues, statements)``.

    Value translation (applied to each arg/kwarg value during step 2d):
    - Non-string (int/float/bool/None): ``ast.Constant(value=value)``
      directly.
    - String, whole ``SIGIL_REF_PATTERN`` match:
      - name in ``available``: ``ast.Name(id=name, ctx=ast.Load())``.
      - name not in ``available``: append an "unresolved reference" issue
        -- AND still build ``ast.Constant(value=original_string)`` as a
        placeholder (discarded by the caller once issues is non-empty).
    - String, no whole match, one or more embedded ``SIGIL_REF_PATTERN``
      matches: build ``ast.JoinedStr`` whose parts alternate
      ``ast.Constant`` (literal text segments) and
      ``ast.FormattedValue(ast.Name(id=name))`` for every embedded
      occurrence whose name IS in ``available``; an embedded occurrence
      whose name is NOT in ``available`` is left as literal text (folded
      into the adjacent Constant segment, sigil included) -- never flagged
      as an issue.
    - String, no match at all (whole or embedded): ``ast.Constant(value=
      original_string)``.

    Never raises. Comprehensive issue collection (not fail-fast).
    """
    issues: list[str] = []
    statements: list[ToolStatement] = []
    available: set[str] = set(context_names)

    def translate_value(value: Any, label: str) -> ast.expr:
        if not isinstance(value, str):
            return ast.Constant(value=value)

        whole = SIGIL_REF_PATTERN.fullmatch(value)
        if whole is not None:
            name = whole.group(1)
            if name in available:
                return ast.Name(id=name, ctx=ast.Load())
            issues.append(
                f"{label}: reference '${name}' does not match any context "
                "variable or earlier result_name (only checked for a value "
                "that is *entirely* one '$name' token -- a '$name' embedded "
                "in a longer string is never flagged). Reference only names "
                "shown in the RUNTIME CONTEXT block or bound by an earlier "
                "call this run."
            )
            return ast.Constant(value=value)

        matches = list(SIGIL_REF_PATTERN.finditer(value))
        if not matches:
            return ast.Constant(value=value)

        # Embedded references: build a JoinedStr alternating literal text
        # segments and FormattedValue(Name) interpolations for every
        # occurrence whose name resolves against `available`. An
        # unresolved embedded name is left as literal text (folded into
        # the adjacent Constant segment, sigil included) -- never an
        # issue, matching the prior permissive embedded-reference
        # behavior exactly.
        parts: list[ast.expr] = []
        literal_buffer = ""
        cursor = 0
        for m in matches:
            name = m.group(1)
            if name in available:
                literal_buffer += value[cursor:m.start()]
                if literal_buffer:
                    parts.append(ast.Constant(value=literal_buffer))
                    literal_buffer = ""
                parts.append(ast.FormattedValue(
                    value=ast.Name(id=name, ctx=ast.Load()),
                    conversion=-1,
                ))
            else:
                literal_buffer += value[cursor:m.end()]
            cursor = m.end()
        literal_buffer += value[cursor:]

        if not any(isinstance(part, ast.FormattedValue) for part in parts):
            # Every embedded occurrence was unresolved -- nothing actually
            # interpolates, so this is just the original literal text.
            # Falling through to a plain ast.Constant (rather than a
            # trivial single-part JoinedStr) avoids ast.unparse rendering
            # it with a spurious f-prefix (f'...') for a string that never
            # actually interpolates anything.
            return ast.Constant(value=value)

        if literal_buffer:
            parts.append(ast.Constant(value=literal_buffer))

        return ast.JoinedStr(values=parts)

    for draft in drafts:
        label = draft.identifier if draft.identifier is not None else "(unassigned)"

        # 2a. Identifier normalization -- strip whitespace, strip one
        # leading '$', collapse blank to None.
        normalized_identifier = draft.identifier
        if normalized_identifier is not None:
            stripped = normalized_identifier.strip()
            if stripped.startswith("$"):
                stripped = stripped[1:].strip()
            normalized_identifier = stripped or None

        # 2b. Identifier legality -- first match wins.
        identifier_legal = True
        if normalized_identifier is not None:
            if not IDENTIFIER_PATTERN.fullmatch(normalized_identifier):
                issues.append(
                    f"result_name {normalized_identifier!r} is not a valid identifier."
                )
                identifier_legal = False
            elif normalized_identifier.startswith(TASK_RESULT_PREFIX):
                issues.append(
                    f"result_name {normalized_identifier!r} starts with the reserved "
                    f"{TASK_RESULT_PREFIX!r} prefix; choose a different name."
                )
            elif normalized_identifier in readonly_names:
                issues.append(
                    f"result_name {normalized_identifier!r} is a read-only context "
                    "variable and cannot be reassigned."
                )
            elif normalized_identifier in tool_names:
                issues.append(
                    f"result_name {normalized_identifier!r} is a registered tool id "
                    "and cannot be used as a result name."
                )
            elif normalized_identifier.startswith("__") and normalized_identifier.endswith("__"):
                issues.append(
                    f"result_name {normalized_identifier!r} is reserved -- a name "
                    "starting and ending with '__' is reserved for "
                    "framework-assigned identifiers."
                )

        # 2c. Repeated keyword names (collapsed by parse_*, recorded there).
        if draft.duplicate_kwargs:
            issues.append(
                f"{label}: argument name(s) {list(draft.duplicate_kwargs)!r} given "
                "more than once; each keyword argument may appear only once."
            )

        # 2d. Translate every arg/kwarg value against `available` as it
        # stands before this draft's own identifier is added.
        translated_args = tuple(translate_value(v, label) for v in draft.args)
        translated_kwargs = {k: translate_value(v, label) for k, v in draft.kwargs.items()}

        # 2e. Build unconditionally -- meaningful only once issues is empty.
        # An illegal identifier is dropped from the placeholder statement
        # (ToolStatement would raise on it; the issue is already recorded).
        statements.append(
            ToolStatement(
                identifier=normalized_identifier if identifier_legal else None,
                tool=draft.tool,
                args=translated_args,
                kwargs=translated_kwargs,
            )
        )

        # 2f. A call's own identifier is only added to `available` after
        # its own references were checked against the pre-call set.
        if normalized_identifier is not None and identifier_legal:
            available.add(normalized_identifier)

    real_call_count = sum(1 for draft in drafts if draft.tool != RETURN_ALIAS)
    if tool_calls_limit is not None and real_call_count > tool_calls_limit:
        issues.append(
            f"this makes {real_call_count} tool call(s), exceeding the "
            f"{tool_calls_limit} still allowed by the tool-call limit."
        )

    return issues, statements


def find_cascade_failures(
    failed_identifiers: set[str],
    pending: list[list[ToolStatement]],
) -> set[str]:
    """
    Given the identifiers of calls that just failed, return the full
    "poisoned" name set: ``failed_identifiers`` plus the ``identifier`` of
    every call in ``pending`` that transitively references one of those
    names (directly, or through a chain of intermediate poisoned calls).

    Used by callers with no continuation round to fall back to (a one-shot
    planner): rather than abandoning the whole plan on any failure, this
    lets independent branches of the same plan keep running while only the
    calls that actually depend on the failure are skipped.

    The returned set is the caller's filter key, not a list of calls to
    remove directly: to find every call that must be skipped (named or
    not), the caller re-scans ``pending`` for any call whose ``args``/
    ``kwargs`` reference a name in the returned set via
    ``extract_identifiers`` -- an unnamed call can be poisoned this way
    without ever itself being added to the set (nothing can reference an
    unnamed result later, so it never needs to propagate further, only to
    be skipped once). Never raises.
    """
    poisoned: set[str] = set(failed_identifiers)
    changed = True
    while changed:
        changed = False
        for batch in pending:
            for call in batch:
                if call.identifier is None or call.identifier in poisoned:
                    continue
                refs = set(extract_identifiers(call.args)) | set(extract_identifiers(call.kwargs))
                if refs & poisoned:
                    poisoned.add(call.identifier)
                    changed = True

    return poisoned


def serialize_call(call: ToolStatement) -> dict[str, Any]:
    """
    Reconstruct ``call`` in the **wire schema's own shape**
    (``call``/``arguments: [{"name", "value"}, ...]``/``result_name``), the
    model's own generation vocabulary -- distinct from ``to_dict()``'s
    internal-field debug view. A standalone function, usable on any
    ``ToolStatement`` regardless of origin.

    1. Build each ``arguments[]`` entry from ``call.args``
       (``name=None``) and ``call.kwargs`` (``name=key``), classifying the
       ast node:
       a. ``ast.Name`` -> ``{"name": name, "value": f"${node.id}"}``
       b. ``ast.Constant`` -> ``{"name": name, "value": node.value}``
       c. ``ast.JoinedStr`` where every ``ast.FormattedValue.value`` is a
          bare ``ast.Name`` -> ``{"name": name, "value": <string with each
          {name} replaced by $name>}``
       d. anything else (``BinOp``, ``Attribute``, ``Subscript``,
          ``Compare``, ``BoolOp``, ``IfExp``, a ``JoinedStr`` with any
          non-``Name`` ``FormattedValue``, container literals, ``Starred``)
          -> ``{"name": name, "raw_code": ast.unparse(node)}`` instead of a
          "value" key -- no partial/hybrid representation, the whole value
          falls back as one unit.
    2. Return ``{"call": call.tool, "arguments": [...], "result_name":
       call.identifier}``.

    Never raises -- ``ast.unparse`` never fails on a well-formed node tree.
    """
    def classify(node: ast.expr, name: Optional[str]) -> dict[str, Any]:
        if isinstance(node, ast.Name):
            return {"name": name, "value": f"${node.id}"}
        if isinstance(node, ast.Constant):
            return {"name": name, "value": node.value}
        if isinstance(node, ast.JoinedStr) and all(
            isinstance(part, ast.Constant)
            or (isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name))
            for part in node.values
        ):
            text = "".join(
                part.value if isinstance(part, ast.Constant) else f"${part.value.id}"
                for part in node.values
            )
            return {"name": name, "value": text}
        return {"name": name, "raw_code": ast.unparse(node)}

    arguments = (
        [classify(value, None) for value in call.args]
        + [classify(value, key) for key, value in call.kwargs.items()]
    )

    return {"call": call.tool, "arguments": arguments, "result_name": call.identifier}


def render_completed_as_json(calls: list[ToolStatement]) -> str:
    """
    Render ``calls`` as a JSON array of leaner per-call dicts, in the
    **wire schema's own shape** (``call``/``arguments``/``result_name``) --
    a thin wrapper over ``serialize_call``, the one authoritative home for
    that reconstruction.

    Generic over any ``list[ToolStatement]`` -- reused for both a full
    "work completed so far" snapshot and a single batch. **Never used for
    regen-repair feedback** -- that path replays the raw engine output
    directly, since reconstructing through ``serialize_call`` would
    silently drop every ``summary``-adjacent reasoning the model wrote for
    that specific failed attempt.

    Returns ``""`` for an empty ``calls`` list -- the caller supplies its
    own fallback text.
    """
    if not calls:
        return ""
    return json.dumps([serialize_call(call) for call in calls], indent=2)


def render_failed_as_json(calls: list[ToolStatement]) -> str:
    """
    Render ``calls`` as a JSON array of leaner per-call dicts, identical
    shape to ``render_completed_as_json``'s own wire-schema-vocabulary
    reconstruction (via ``serialize_call``), plus one additional key per
    entry: ``"error"``, the failed call's own ``str(call.exception)``.

    For the views that must carry the failure reason (``ReActAgent``'s
    last-failure steps entry, ``PlanActAgent``'s exhausted-plan error) --
    neither ``serialize_call`` nor ``render_completed_as_json`` carries
    exception text, and nothing else in this file does either.

    Returns ``""`` for an empty ``calls`` list, matching
    ``render_completed_as_json``'s own empty-input contract -- the caller
    supplies its own fallback text.
    """
    if not calls:
        return ""
    return json.dumps(
        [{**serialize_call(call), "error": str(call.exception)} for call in calls],
        indent=2,
    )
