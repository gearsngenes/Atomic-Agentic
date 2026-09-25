from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Iterable, Optional

from ..constants.agents import (
    DAG_OUTPUT_SCHEMA,
    DAG_REF_PATTERN,
    PLANACT_OUTPUT_SCHEMA,
    REACT_OUTPUT_SCHEMA,
    RETURN_ALIAS,
    RETURN_VALUE_FIELD,
    TASK_RESULT_PREFIX,
)
from ..constants.core import IDENTIFIER_PATTERN
from ..models.agents.blackboard_models import DagToolCall

__all__ = [
    "build_dag_schema",
    "build_planact_schema",
    "build_react_schema",
    "parse_generation",
    "parse_react_call",
    "find_sigil_refs",
    "find_cascade_failures",
    "resolve_sigil_value",
    "is_dispatched_call",
    "validate_calls",
    "compile_batches",
    "resolve_call_args",
    "render_completed_as_json",
    "render_failed_as_json",
    "render_completed_as_code",
    "render_cache_snapshot",
]


def build_dag_schema(tool_names: Iterable[str]) -> dict[str, Any]:
    """
    Return a fresh, per-call copy of ``DAG_OUTPUT_SCHEMA`` with ``call``'s
    ``enum`` populated from the currently-registered toolset -- makes an
    unregistered ``call`` structurally impossible under strict-mode
    generation. Deep-copied so no two calls ever alias the same mutable
    dict, and the module-level template itself is never mutated.
    """
    schema = deepcopy(DAG_OUTPUT_SCHEMA)
    schema["properties"]["plan"]["items"]["properties"]["call"]["enum"] = sorted(tool_names)
    return schema


def build_planact_schema(tool_names: Iterable[str]) -> dict[str, Any]:
    """
    Sibling to ``build_dag_schema`` -- identical pattern, built off
    ``PLANACT_OUTPUT_SCHEMA`` instead: a fresh, per-call copy with
    ``call``'s ``enum`` populated from the currently-registered toolset.
    """
    schema = deepcopy(PLANACT_OUTPUT_SCHEMA)
    schema["properties"]["plan"]["items"]["properties"]["call"]["enum"] = sorted(tool_names)
    return schema


def build_react_schema(tool_names: Iterable[str]) -> dict[str, Any]:
    """
    Sibling to ``build_dag_schema``/``build_planact_schema`` -- identical
    pattern, built off ``REACT_OUTPUT_SCHEMA`` instead. No restriction
    parameter -- a caller wanting the enum narrowed to ``return`` only (the
    budget-boundary case) passes a pre-narrowed ``tool_names`` iterable
    itself (e.g. ``[RETURN_TOOL_NAME]``); this function always does exactly
    one thing with whatever it's given.
    """
    schema = deepcopy(REACT_OUTPUT_SCHEMA)
    schema["properties"]["call"]["enum"] = sorted(tool_names)
    return schema


def parse_generation(payload: dict[str, Any]) -> tuple[list[DagToolCall], Optional[str]]:
    """
    Normalize ``output_structure``'s already-schema-validated payload into a
    flat call sequence plus the round's completion signal. Pure shape
    unpacking only -- no decoding of any kind happens here. Every
    ``arguments[].value``/non-null ``return`` is carried through byte-for-
    byte as the schema handed it back -- a scalar needs no further
    processing at all; a string may contain ``$name`` sigil references,
    resolved later, at prepare time (``utils.dag.resolve_call_args``),
    never here.

    No malformed-shape failure mode -- every required key is schema-
    guaranteed present and type-correct; this function never raises.

    1. One ``DagToolCall`` is built per ``plan`` entry. The top-level
       ``summary`` field is never read -- decoding-order scaffold only.
    2. ``remaining_work`` is read via ``payload.get("remaining_work")`` --
       not a bare subscript -- since it's a genuinely optional key: absent
       entirely from ``PLANACT_OUTPUT_SCHEMA`` (no continuation round
       exists for a one-shot planner), always present in
       ``DAG_OUTPUT_SCHEMA``. ``.get`` returns the identical value a bare
       subscript would for ``DagAgent``'s own payload, and ``None`` for a
       payload that never had the key at all -- behavior-preserving either
       way. Normalized once, here, to the single canonical shape every
       downstream caller consumes: ``None`` stays ``None``; a string is
       stripped, and an all-whitespace/empty result also collapses to
       ``None``; a non-empty stripped string passes through as-is.
    3. A trailing ``RETURN_ALIAS`` call is synthesized and appended,
       carrying ``payload["return"]`` under ``RETURN_VALUE_FIELD``, unless
       the round is actively deferring (``return`` is JSON ``null`` *and*
       ``remaining_work`` is truthy -- the only combination
       ``validate_calls`` still permits when nothing should be dispatched
       as a return signal this round). A ``null`` return is not itself a
       reason to skip synthesis: it's a legitimate finished value (the task
       genuinely has nothing to hand back) whenever it isn't paired with a
       real ``remaining_work`` note. For ``PLANACT_OUTPUT_SCHEMA`` payloads,
       ``remaining_work`` is always ``None`` (no such field exists), so a
       ``RETURN_ALIAS`` call is synthesized unconditionally -- correct,
       since a one-shot planner has no deferral to skip it for.
       ``payload["return"]`` stays a bare subscript, deliberately -- both
       schemas require this key, so a missing/malformed value here is a
       real schema-contract violation that should surface as a natural
       ``KeyError``, not be defensively guarded against.
    """
    calls: list[DagToolCall] = []
    for entry in payload["plan"]:
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        for arg in entry["arguments"]:
            if arg["name"] is None:
                args.append(arg["value"])
            else:
                kwargs[arg["name"]] = arg["value"]
        calls.append(
            DagToolCall(
                identifier=entry["result_name"],
                tool=entry["call"],
                args=tuple(args),
                kwargs=kwargs,
            )
        )

    raw_remaining_work = payload.get("remaining_work")
    remaining_work: Optional[str] = (
        raw_remaining_work.strip() or None
        if isinstance(raw_remaining_work, str)
        else None
    )

    raw_return = payload["return"]
    if raw_return is not None or not remaining_work:
        calls.append(
            DagToolCall(
                identifier=None,
                tool=RETURN_ALIAS,
                kwargs={RETURN_VALUE_FIELD: raw_return},
            )
        )

    return calls, remaining_work


def parse_react_call(payload: dict[str, Any]) -> DagToolCall:
    """
    Normalize ``REACT_OUTPUT_SCHEMA``'s already-schema-validated flat
    payload into the one ``DagToolCall`` it describes. Pure shape
    unpacking, no decoding of any kind (matches ``parse_generation``'s own
    contract) -- every ``arguments[].value`` is carried through byte-for-
    byte; a string may contain a ``$name`` sigil, resolved later at prepare
    time (``resolve_call_args``), never here.

    No malformed-shape failure mode -- every required key is schema-
    guaranteed present and type-correct; this function never raises.

    ``payload["summary"]`` is never read -- decoding-order scaffold only,
    same treatment ``DAG_OUTPUT_SCHEMA``/``PLANACT_OUTPUT_SCHEMA``'s own
    ``"summary"`` field already gets. No ``plan``-array loop (there is none
    -- exactly one call per payload) and no ``RETURN_ALIAS`` synthesis
    (``return`` is an ordinary, really-dispatched ``call`` value in this
    family, never a separate top-level field to synthesize a sentinel call
    from).
    """
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for arg in payload["arguments"]:
        if arg["name"] is None:
            args.append(arg["value"])
        else:
            kwargs[arg["name"]] = arg["value"]

    return DagToolCall(
        identifier=payload["result_name"],
        tool=payload["call"],
        args=tuple(args),
        kwargs=kwargs,
    )


def find_sigil_refs(value: Any) -> frozenset[str]:
    """
    Given one already-typed args/kwargs value (never a container -- see
    ``DagToolCall``'s own docstring, args/kwargs entries are always flat
    scalars now), return every name referenced by a ``$name`` sigil found
    anywhere in it. Used for batch dependency detection only (``compile_
    batches``) -- whole-string or embedded, doesn't matter which for this
    purpose, only *that* a name is referenced.

    A non-``str`` value can never contain a sigil -- returns ``frozenset()``
    immediately. Never raises -- pure regex scan over an already-guaranteed
    ``str``.
    """
    if not isinstance(value, str):
        return frozenset()
    return frozenset(m.group(1) for m in DAG_REF_PATTERN.finditer(value))


def find_cascade_failures(
    failed_identifiers: set[str],
    pending: list[list[DagToolCall]],
) -> set[str]:
    """
    Given the identifiers of calls that just failed, return the full
    "poisoned" name set: ``failed_identifiers`` plus the ``result_name`` of
    every call in ``pending`` that transitively references one of those
    names (directly, or through a chain of intermediate poisoned calls).

    Used by callers with no continuation round to fall back to (a one-shot
    planner): unlike ``DagAgent``'s own failure handling, which simply
    abandons everything and requests a fresh round, this lets independent
    branches of the same plan keep running while only the calls that
    actually depend on the failure are skipped.

    The returned set is the caller's filter key, not a list of calls to
    remove directly: to find every call that must be skipped (named or
    not), the caller re-scans ``pending`` for any call whose ``args``/
    ``kwargs`` reference a name in the returned set via
    ``find_sigil_refs`` -- an unnamed call can be poisoned this way without
    ever itself being added to the set (nothing can ``$``-reference an
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
                refs: set[str] = set()
                for value in call.args:
                    refs |= find_sigil_refs(value)
                for value in call.kwargs.values():
                    refs |= find_sigil_refs(value)
                if refs & poisoned:
                    poisoned.add(call.identifier)
                    changed = True

    return poisoned


def is_dispatched_call(call: DagToolCall) -> bool:
    """
    True iff ``call`` represents a real dispatched tool call rather than the
    ``RETURN_ALIAS`` sentinel, which is never dispatched at all. Shared by
    ``validate_calls`` (tool-call-budget accounting) and ``compile_batches``
    (concurrency-batch accounting).
    """
    return call.tool != RETURN_ALIAS


def validate_calls(
    calls: list[DagToolCall],
    remaining_work: Optional[str],
    tool_calls_limit: Optional[int],
    known_names: frozenset[str],
) -> list[str]:
    """
    Walk ``calls``, collecting every semantic issue found -- comprehensive,
    not fail-fast. Does NOT check tool registration -- the schema's ``call``
    enum already makes an unregistered tool structurally impossible.

    For each call with a non-``None`` ``identifier``: it must be
    ``IDENTIFIER_PATTERN``-legal (checked first), and, only if it already is,
    must not start with the reserved ``K_``/``task_result_`` prefixes.
    Separately: the dispatched-call count against ``tool_calls_limit``
    (``RETURN_ALIAS`` excluded); a ``RETURN_ALIAS`` call present alongside a
    truthy ``remaining_work`` -- a contradiction, exactly one way to end a
    round is permitted; and a truthy ``remaining_work`` with zero dispatched
    calls -- deferring with nothing dispatched is never valid, mirrors
    ``ScriptAgent``'s own "a pause cannot appear before any real work has
    been done" check. ``remaining_work`` is only ever checked for
    truthiness here -- an empty-vs-non-empty string was already resolved to
    ``None``-vs-real-text by ``parse_generation``, so this function never
    inspects the text content itself.

    Unbound-``$name``-reference check, narrowed to **whole-string** matches
    only -- an embedded ``$name`` (not the entire value) stays fully
    permissive and is never flagged here (silent fallback-to-literal at
    resolve time, genuinely ambiguous with intentional literal text, e.g. a
    dollar amount). Walked in ``calls``' own order (already plan order,
    pre-batching): a value that is *entirely* one ``$name`` token must
    already be bound -- in ``known_names`` (the caller's
    ``task.cache``/``task.constant_values`` keys), or by an earlier call
    already written in this same plan. A call's own identifier is only
    added to the available set *after* its own references are checked
    against the pre-call set -- a call can never reference its own
    ``result_name``, and a later call can only reference an earlier one's,
    never a forward reference within the same plan. This check exists
    specifically because ``compile_batches`` only detects a same-batch
    conflict against names already in the currently-open batch -- a forward
    reference within one plan would otherwise land both calls in one
    concurrently-dispatched batch and silently resolve to the literal
    ``"$name"`` string instead of erroring.

    Returns ``[]`` if every call is clean and the plan is within budget.
    """
    issues: list[str] = []

    for call in calls:
        if call.identifier is None:
            continue
        if not IDENTIFIER_PATTERN.fullmatch(call.identifier):
            issues.append(
                f"result_name {call.identifier!r} is not a valid identifier."
            )
        elif call.identifier.startswith("K_") or call.identifier.startswith(
            TASK_RESULT_PREFIX
        ):
            issues.append(
                f"result_name {call.identifier!r} uses a reserved prefix "
                "('K_' is reserved for constants, "
                f"{TASK_RESULT_PREFIX!r} for cross-invocation results)."
            )

    real_call_count = sum(1 for call in calls if is_dispatched_call(call))
    if tool_calls_limit is not None and real_call_count > tool_calls_limit:
        issues.append(
            f"the plan calls {real_call_count} tool(s), exceeding the "
            f"configured limit of {tool_calls_limit}."
        )

    has_return = any(call.tool == RETURN_ALIAS for call in calls)
    if has_return and remaining_work:
        issues.append(
            "the plan sets a return value while 'remaining_work' is also "
            "set -- pick exactly one way to end a round: return a final "
            "value with 'remaining_work' left null, or leave return null "
            "and describe what's left in 'remaining_work' to continue."
        )

    if remaining_work and real_call_count == 0:
        issues.append(
            "'remaining_work' is set but the plan calls no tools -- "
            "deferring with nothing dispatched is never valid; either call "
            "something whose result you need, or finish the round: leave "
            "'remaining_work' null and set 'return' to the final value."
        )

    def whole_refs(value: Any) -> Iterable[str]:
        if isinstance(value, str):
            m = DAG_REF_PATTERN.fullmatch(value)
            if m is not None:
                yield m.group(1)

    available: set[str] = set(known_names)
    for call in calls:
        label = call.identifier if call.identifier is not None else "(unassigned)"

        refs: set[str] = set()
        for value in call.args:
            refs |= set(whole_refs(value))
        for value in call.kwargs.values():
            refs |= set(whole_refs(value))

        unresolved = sorted(name for name in refs if name not in available)
        if unresolved:
            issues.append(
                f"{label}: reference(s) {unresolved!r} do not match any "
                "earlier result_name, constant, or cross-invocation "
                "result (only checked for a value that is *entirely* one "
                "'$name' token -- a '$name' embedded in a longer string is "
                "never flagged)."
            )
        if call.identifier is not None:
            available.add(call.identifier)

    return issues


def compile_batches(
    calls: list[DagToolCall],
    max_concurrency: Optional[int] = None,
    start_batch_index: int = 0,
) -> list[list[DagToolCall]]:
    """
    Group ``calls`` into dependency batches for concurrent execution, and
    stamp each call's ``.batch_index`` with the batch it landed in.

    Dependency source is ``find_sigil_refs``, scanned over each call's
    already-native ``args``/``.kwargs`` -- permissive over-collection (a
    name found here that never actually resolves to anything just never
    overlaps a real identifier in ``current_batch_identifiers``, harmless,
    not specially guarded against). The ``RETURN_ALIAS``-isolation rule is
    kept unchanged -- a ``RETURN_ALIAS`` call always closes the current
    batch, lands alone in a batch of its own, then closes that batch too,
    so a batch-partial failure elsewhere can never suppress an
    already-resolved return.

    ``max_concurrency=None`` means no cap. Every call in a batch is stamped
    with the same ``batch_index`` (``start_batch_index`` plus that batch's
    own 0-based position among the batches this call produces) the moment
    the batch closes.
    """
    batches: list[list[DagToolCall]] = []
    current_batch: list[DagToolCall] = []
    current_batch_identifiers: set[str] = set()
    current_batch_dispatched = 0

    def close_current() -> None:
        nonlocal current_batch, current_batch_identifiers, current_batch_dispatched
        if not current_batch:
            return
        index = start_batch_index + len(batches)
        for call in current_batch:
            call.batch_index = index
        batches.append(current_batch)
        current_batch = []
        current_batch_identifiers = set()
        current_batch_dispatched = 0

    for call in calls:
        if call.tool == RETURN_ALIAS:
            close_current()
            current_batch.append(call)
            close_current()
            continue

        deps: set[str] = set()
        for value in call.args:
            deps |= find_sigil_refs(value)
        for value in call.kwargs.values():
            deps |= find_sigil_refs(value)
        if any(name in current_batch_identifiers for name in deps):
            close_current()

        is_dispatched = is_dispatched_call(call)
        if (
            is_dispatched
            and max_concurrency is not None
            and current_batch_dispatched >= max_concurrency
        ):
            close_current()

        current_batch.append(call)
        if call.identifier is not None:
            current_batch_identifiers.add(call.identifier)
        if is_dispatched:
            current_batch_dispatched += 1

    close_current()
    return batches


def resolve_sigil_value(value: Any, resolved: dict[str, Any]) -> Any:
    """
    Given one already-typed args/kwargs value and the resolution namespace
    (``{**task.cache, **task.constant_values}``, built by the caller), return
    the value with every resolvable ``$name`` sigil substituted.

    A non-``str`` value passes through unchanged -- can never contain a
    sigil. A ``str`` that is a **whole** ``DAG_REF_PATTERN`` match against a
    name present in ``resolved`` returns ``resolved[name]`` directly, real
    type preserved (may be any scalar type, not just ``str``) -- the only
    path that can return a non-``str`` result. Otherwise, every embedded
    ``$name`` occurrence whose name is in ``resolved`` is stringified and
    spliced in place via a single ``DAG_REF_PATTERN.sub`` pass; an
    occurrence (whole or embedded) whose name is *not* in ``resolved`` is
    left untouched, sigil included -- one mechanism uniformly covering a
    plain literal (no-op), one or more embedded interpolations, and an
    unresolved sigil at any position.

    Never raises -- no failure mode exists at this layer. Note: a resolved
    value that itself contains a literal ``$name``-shaped substring (e.g.
    ``$price`` resolving to the string ``"$5"`` inside a larger interpolated
    string) is not re-scanned -- a single ``.sub()`` pass, not a fixed
    point -- so it lands verbatim in the output text. Intentional, not a
    bug.
    """
    if not isinstance(value, str):
        return value

    whole = DAG_REF_PATTERN.fullmatch(value)
    if whole is not None and whole.group(1) in resolved:
        return resolved[whole.group(1)]

    def _splice(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in resolved:
            return str(resolved[name])
        return match.group(0)

    return DAG_REF_PATTERN.sub(_splice, value)


def resolve_call_args(
    call: DagToolCall, resolved: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """
    Resolve one call's ``args``/``kwargs`` into a plain ``(positional,
    keyword)`` pair, ready to splat into
    ``tool._args_kwargs_to_dict(*positional, **keyword)`` -- or, for the
    ``RETURN_ALIAS`` call, for the caller to read
    ``keyword[RETURN_VALUE_FIELD]`` directly (that short-circuit lives in
    the caller, not here).

    Built around ``resolve_sigil_value`` -- identifier substitution only, no
    JSON decoding (that already happened, or didn't need to, by
    ``parse_generation`` time). Never raises: ``resolve_sigil_value`` has no
    failure mode, so this function inherits that -- there is no longer any
    precondition to document about a round having produced zero
    parse/validate issues first, because there's no parse step left to have
    failed.
    """
    positional = [resolve_sigil_value(value, resolved) for value in call.args]
    keyword = {key: resolve_sigil_value(value, resolved) for key, value in call.kwargs.items()}
    return positional, keyword


def render_completed_as_json(calls: list[DagToolCall]) -> str:
    """
    Render ``calls`` as a JSON array of leaner per-call dicts, in the
    **wire schema's own shape** (``call``/``arguments``/``result_name``) --
    a thin wrapper over ``DagToolCall.serialize()``, which is now the one
    authoritative home for that reconstruction (moved there so
    ``DagAgentRecord.serialize_statements()`` and this function share the
    identical logic instead of each rebuilding the shape independently).

    Generic over any ``list[DagToolCall]`` -- reused for both a full
    "work completed so far" snapshot and a single batch. **Never used for
    regen-repair feedback** -- that path replays the raw engine output
    directly, since reconstructing through ``DagToolCall`` would silently
    drop every ``summary``-adjacent reasoning the model wrote for that
    specific failed attempt.

    Returns ``""`` for an empty ``calls`` list -- the caller supplies its
    own fallback text.
    """
    if not calls:
        return ""
    return json.dumps([call.serialize() for call in calls], indent=2)


def render_failed_as_json(calls: list[DagToolCall]) -> str:
    """
    Render ``calls`` as a JSON array of leaner per-call dicts, identical
    shape to ``render_completed_as_json``'s own wire-schema-vocabulary
    reconstruction (via ``DagToolCall.serialize()``), plus one additional
    key per entry: ``"error"``, the failed call's own
    ``str(call.exception)``.

    Exists specifically for a family (``ReActAgent``) whose
    ``failed_statements`` persist into every future round's rendered
    snapshot, unlike ``DagAgent``'s one-time ``continuation_note`` --
    neither ``DagToolCall.serialize()`` nor ``render_completed_as_json``
    carries exception text, and nothing else in this file does either.

    Returns ``""`` for an empty ``calls`` list, matching
    ``render_completed_as_json``'s own empty-input contract -- the caller
    supplies its own fallback text.
    """
    if not calls:
        return ""
    return json.dumps(
        [{**call.serialize(), "error": str(call.exception)} for call in calls],
        indent=2,
    )


def render_completed_as_code(
    calls: list[DagToolCall], show_batches: bool = False,
) -> str:
    """
    Reconstruct a Python-source-formatted snapshot of ``calls``: one line
    per call, in commit order, via ``DagToolCall.to_python_code()``.

    Direct structural port of ``utils/script.py``'s
    ``render_completed_as_python``, simplified: no
    ``PY_BUILTIN_ALIAS``/``ATTR_CALL_ALIAS``/``RHS_ASSIGN_ALIAS``
    unsplicing branches needed (``DagAgent`` has no such sentinels --
    ``call_python_builtin`` renders like any other ordinary registered-tool
    call), no ``ast.Starred``/kwargs-unpack handling needed (this wire
    format has no unpack concept) -- the per-call rendering itself lives
    entirely on ``DagToolCall.to_python_code()``, this function only
    handles batch grouping and joining.

    ``show_batches`` (default ``False``, matching ``render_completed_as_python``'s
    own default for signature parity) controls whether output is grouped
    under a ``# Batch N:`` header per concurrently-dispatched batch. Never
    called from any model-facing render path in this codebase -- only
    ``DagAgentRecord.render_as_code()``, which always passes ``True``, the
    same standalone-human-inspection-only usage its ``ScriptAgent`` sibling
    has. Consecutive calls sharing the same ``.batch_index`` are already
    contiguous in ``calls`` (a batch drains fully before the next one
    starts), so grouping only needs to detect index changes, not sort.

    Returns the joined lines, or ``""`` for an empty ``calls``.
    """
    lines: list[str] = []
    current_index: Optional[int] = None
    for call in calls:
        if show_batches and call.batch_index != current_index:
            if current_index is not None:
                lines.append("")
            lines.append(f"# Batch {call.batch_index}:")
            current_index = call.batch_index
        lines.append(call.to_python_code())
    return "\n".join(lines)


def render_cache_snapshot(
    completed: list[DagToolCall],
    cache: dict[str, Any],
    preview_limit: Optional[int],
) -> str:
    """
    Retyped port of ``utils/script.py``'s ``render_cache_snapshot`` over
    ``DagToolCall`` -- structurally identical (same ordering, same fenced
    block, same ``name: type = value`` line shape, same truncation
    semantics), except for the value serialization itself. Renders the
    current value of every identifier bound by ``completed`` this round,
    each looked up fresh in ``cache`` (so a reassigned name shows its
    latest value), as its own fenced ``"Cached values:"`` block.
    ``preview_limit`` truncates each rendered value (``None`` means no
    truncation). Returns ``""`` when ``completed`` binds no identifiers at
    all.

    Unaffected by the expression-value grammar redesign -- ``cache`` only
    ever holds fully-resolved plain Python values, never ``ast.expr``
    nodes, regardless of how a value's source was written.

    Value serialization deliberately diverges from ``ScriptAgent``'s
    ``repr()``-based preview: this agent's whole continuation context is
    JSON-styled (see ``render_completed_as_json``, rendered directly above
    this block at the call site), so a ``repr()`` preview (Python's
    ``True``/``None``/single-quoted strings) would be a real vocabulary
    clash sitting next to valid JSON. ``json.dumps()`` is tried first;
    ``cache``'s values aren't guaranteed JSON-safe (a tool may legitimately
    return an arbitrary Python object, not just JSON primitives), so a
    ``TypeError`` falls back to ``repr()`` rather than propagating. The
    ``type(value).__name__`` annotation on each line is kept regardless of
    which serialization path a given value took -- useful context either
    way, independent of how the value itself gets rendered.
    """
    ordered: dict[str, Any] = {}
    for call in completed:
        if call.identifier is not None:
            ordered[call.identifier] = cache[call.identifier]

    if not ordered:
        return ""

    def preview(value: Any) -> str:
        try:
            text = json.dumps(value)
        except TypeError:
            text = repr(value)
        if preview_limit is not None and len(text) > preview_limit:
            text = text[:preview_limit] + "..."
        return text

    lines = [
        f"{name}: {type(value).__name__} = {preview(value)}"
        for name, value in ordered.items()
    ]
    return "```\nCached values:\n" + "\n".join(lines) + "\n```"
