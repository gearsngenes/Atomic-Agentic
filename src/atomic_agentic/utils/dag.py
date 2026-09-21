from __future__ import annotations

import ast
import json
from copy import deepcopy
from typing import Any, Iterable, Optional

from ..constants.agents import (
    DAG_OUTPUT_SCHEMA,
    RETURN_ALIAS,
    RETURN_VALUE_FIELD,
    TASK_RESULT_PREFIX,
)
from ..constants.core import IDENTIFIER_PATTERN
from ..exceptions import BlackboardParseError
from ..models.agents.blackboard_models import DagToolCall
from .agents import evaluate_expr, extract_identifiers, reject_unsupported_forms, strip_code_fence

__all__ = [
    "build_dag_schema",
    "parse_generation",
    "parse_call_expressions",
    "is_dispatched_call",
    "validate_calls",
    "compile_batches",
    "resolve_call_args",
    "render_completed_as_json",
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


def parse_generation(payload: dict[str, Any]) -> tuple[list[DagToolCall], bool]:
    """
    Normalize ``output_structure``'s already-schema-validated payload into a
    flat call sequence plus the round's completion signal. Pure shape
    unpacking only -- no decoding of any kind happens here. Every
    ``arguments[].value`` string, and a non-null top-level ``return``
    string, is carried through exactly as the model wrote it; turning that
    raw string into a parsed ``ast.expr`` is ``parse_call_expressions``'s
    job (a later, separate, fallible pass -- see its own docstring for why
    it isn't folded into this function).

    No malformed-shape failure mode -- every required key is schema-
    guaranteed present and type-correct; this function never raises.

    1. One ``DagToolCall`` is built per ``plan`` entry. The top-level
       ``summary`` field is never read -- decoding-order scaffold only.
    2. If the top-level ``return`` is not JSON ``null``, it's a raw string
       (same not-yet-parsed treatment as any argument value) -- a trailing
       ``RETURN_ALIAS`` call is synthesized and appended, carrying that raw
       string under ``RETURN_VALUE_FIELD``. ``null`` means nothing is
       returned this round; nothing is synthesized.
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
                identifier=entry["assign_to"],
                tool=entry["call"],
                args=tuple(args),
                kwargs=kwargs,
            )
        )

    more_planning_needed: bool = payload["more_planning_needed"]

    raw_return = payload["return"]
    if raw_return is not None:
        calls.append(
            DagToolCall(
                identifier=None,
                tool=RETURN_ALIAS,
                kwargs={RETURN_VALUE_FIELD: raw_return},
            )
        )

    return calls, more_planning_needed


def parse_call_expressions(calls: list[DagToolCall]) -> list[str]:
    """
    Second, fallible pass over ``calls`` -- kept separate from
    ``parse_generation`` so that function's own "never raises" contract
    stays literally true. Parses every still-raw-string args/kwargs value
    (including the synthesized ``RETURN_ALIAS`` call's ``RETURN_VALUE_FIELD``
    entry, if present -- no special-casing needed, it's just another call in
    the list) into a real ``ast.expr``, mutating each call's ``args``/
    ``.kwargs`` in place.

    Comprehensive, not fail-fast -- every value across every call is
    attempted, and every problem found is collected, matching
    ``validate_calls``' own convention. A value that fails at any step below
    is left as its original raw string and gets one issue string; nothing
    downstream of a round with any such issue ever runs (the combined issue
    list blocks progression before ``compile_batches``), so no other method
    needs to defensively handle a half-parsed value.

    Per value:
    1. Strip code fencing and surrounding whitespace
       (``strip_code_fence(raw).strip()``). Empty after that -> an issue,
       leave the raw string as-is.
    2. ``ast.parse(cleaned, mode="eval")`` -- a ``SyntaxError`` -> an issue,
       leave the raw string as-is.
    3. ``reject_unsupported_forms(tree.body, forbid_calls=True)`` -- a
       ``BlackboardParseError`` -> an issue, leave the raw string as-is.
    4. Otherwise replace the value with ``tree.body`` (the parsed
       ``ast.expr``).

    Returns the combined issues list (possibly empty).
    """
    issues: list[str] = []

    def parse_one(raw: str, label: str, position: str) -> Any:
        cleaned = strip_code_fence(raw).strip()
        if not cleaned:
            issues.append(
                f"{label}: {position} is empty after removing code "
                "fencing/whitespace."
            )
            return raw
        try:
            tree = ast.parse(cleaned, mode="eval")
        except SyntaxError as e:
            issues.append(f"{label}: {position} is not valid Python: {e}")
            return raw
        node = tree.body
        try:
            reject_unsupported_forms(node, forbid_calls=True)
        except BlackboardParseError as e:
            issues.append(f"{label}: {position}: {e}")
            return raw
        return node

    for call in calls:
        label = call.identifier if call.identifier is not None else "(unassigned)"

        new_args: list[Any] = []
        for index, raw in enumerate(call.args):
            if isinstance(raw, str):
                new_args.append(parse_one(raw, label, f"argument {index}"))
            else:
                new_args.append(raw)
        call.args = tuple(new_args)

        for key, raw in list(call.kwargs.items()):
            if isinstance(raw, str):
                call.kwargs[key] = parse_one(raw, label, f"argument {key!r}")

    return issues


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
    more_planning_needed: bool,
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
    (``RETURN_ALIAS`` excluded), and a ``RETURN_ALIAS`` call present
    alongside ``more_planning_needed=True`` -- a contradiction, exactly one
    way to end a round is permitted.

    New identifier-existence check, walked in ``calls``' own order (already
    plan order, pre-batching): every ``Name`` a call's args/kwargs
    reference must already be bound -- in ``known_names`` (the caller's
    ``task.cache``/``task.constant_values`` keys), or by an earlier call
    already written in this same plan. A call's own identifier is only
    added to the available set *after* its own references are checked
    against the pre-call set -- a call can never reference its own
    ``assign_to``, and a later call can only reference an earlier one's,
    never a forward reference within the same plan.
    ``extract_identifiers``' own dict/tuple/list-walking already skips any
    entry that isn't a parsed ``ast.expr`` (a value ``parse_call_expressions``
    failed to parse and left as a raw string), so a value with its own
    separate parse-failure issue contributes nothing here -- no
    double-reporting.

    Returns ``[]`` if every call is clean and the plan is within budget.
    """
    issues: list[str] = []

    for call in calls:
        if call.identifier is None:
            continue
        if not IDENTIFIER_PATTERN.fullmatch(call.identifier):
            issues.append(
                f"assign_to {call.identifier!r} is not a valid identifier."
            )
        elif call.identifier.startswith("K_") or call.identifier.startswith(
            TASK_RESULT_PREFIX
        ):
            issues.append(
                f"assign_to {call.identifier!r} uses a reserved prefix "
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
    if has_return and more_planning_needed:
        issues.append(
            "the plan sets a return value and more_planning_needed=true "
            "in the same round -- pick exactly one way to end: return a "
            "final value, or signal more planning is needed (with no "
            "return)."
        )

    available: set[str] = set(known_names)
    for call in calls:
        label = call.identifier if call.identifier is not None else "(unassigned)"
        refs = set(extract_identifiers(call.args)) | set(extract_identifiers(call.kwargs))
        unresolved = sorted(name for name in refs if name not in available)
        if unresolved:
            issues.append(
                f"{label}: references unbound name(s) {unresolved!r} -- not "
                "yet assigned earlier in this plan and not present in "
                "cache/constants."
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

    Dependency source is ``extract_identifiers`` (the same shared, ``ast``-
    aware helper ``ScriptAgent`` uses), available now that ``args``/
    ``.kwargs`` hold real ``ast.expr`` nodes -- unlike the old JSON-decoded-
    container shape, no permissive over-collecting string scan is needed
    anymore. The ``RETURN_ALIAS``-isolation rule is kept unchanged -- a
    ``RETURN_ALIAS`` call always closes the current batch, lands alone in a
    batch of its own, then closes that batch too, so a batch-partial
    failure elsewhere can never suppress an already-resolved return.

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

        deps = set(extract_identifiers(call.args)) | set(extract_identifiers(call.kwargs))
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

    Rewritten around the shared ``evaluate_expr`` (``utils.agents``) --
    every value is evaluated exactly once against ``resolved`` (identifier
    -> value), the same "parse once, evaluate once at prepare time"
    discipline ``CodeStatement``/``resolve_slot_args`` use. No exact-match
    string substitution anymore; a ``Name`` reference resolves through real
    Python evaluation instead.

    Assumes every value here is already a parsed ``ast.expr`` -- only true
    for a round that produced zero ``parse_call_expressions``/
    ``validate_calls`` issues, which is the only kind of round that ever
    reaches this function. Not defensively guarded against otherwise:
    ``evaluate_expr``'s own ``ast.Expression(body=node)`` construction
    raises a natural ``TypeError`` if handed a non-node, uncaught, same
    "let it surface" posture used throughout this module.
    """
    positional = [evaluate_expr(value, resolved) for value in call.args]
    keyword = {key: evaluate_expr(value, resolved) for key, value in call.kwargs.items()}
    return positional, keyword


def render_completed_as_json(calls: list[DagToolCall]) -> str:
    """
    Render ``calls`` as a JSON array of leaner per-call dicts, in the
    **wire schema's own shape** (``call``/``assign_to``/``arguments``) --
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
