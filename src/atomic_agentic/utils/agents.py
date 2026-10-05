from __future__ import annotations

import ast
import json
import re
from typing import Any, Optional


from ..constants.agents import (
    ATTR_CALL_ALIAS,
    CODE_FENCE_PATTERN,
    DUNDER_ATTRIBUTE_PATTERN,
    KWARGS_UNPACK_KEY,
    LEADING_CODE_FENCE_PATTERN,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
    TRAILING_CODE_FENCE_PATTERN,
    UNSUPPORTED_EXPR_LABELS,
)
from ..constants.core import NO_VAL
from ..exceptions import BlackboardParseError
from ..models.agents.blackboard_models import ToolStatement
from ..models.agents.prompts import PromptConfig

__all__ = [
    "evaluate_expr",
    "extract_identifiers",
    "extract_json_object",
    "normalize_prompt_config",
    "reject_unsupported_forms",
    "stringify_result",
    "strip_code_fence",
    "is_dispatched",
    "tool_identity",
    "compile_batches",
    "resolve_statement_args",
]


def normalize_prompt_config(
    value: str | PromptConfig | None,
    *,
    default_template: str | None,
    provided_description: str,
    error_label: str,
    default_description: str | None = None,
) -> PromptConfig | None:
    """Coerce a prompt-shaped construction value to a ``PromptConfig``, or
    ``None``.

    Unifies what were three near-identical coercion functions (role prompt,
    thinking instructions, tool instructions) into one, parameterized per
    call site rather than derived from a generic label, so existing callers
    get byte-identical ``PromptConfig``/error text.

    ``value`` is ``None`` or an all-whitespace ``str``:
        - ``default_template`` is ``None`` -- returns ``None``. The one
          branch a role prompt or thinking instructions never take (an
          agent always needs *some* version of either); a caller with no
          sensible generic fallback (e.g. tool instructions) passes
          ``None`` here to get real absence back instead of boilerplate.
        - ``default_template`` is a ``str`` -- returns
          ``PromptConfig(template=default_template, description=default_description)``.
    ``value`` is a non-blank ``str`` -- returns
        ``PromptConfig(template=value.strip(), description=provided_description)``.
    ``value`` is already a ``PromptConfig`` -- returned unchanged.
    Anything else -- raises ``TypeError``.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        if default_template is None:
            return None
        return PromptConfig(template=default_template, description=default_description)
    if isinstance(value, str):
        return PromptConfig(template=value.strip(), description=provided_description)
    if isinstance(value, PromptConfig):
        return value
    raise TypeError(
        f"{error_label} must be str, PromptConfig, or None; got {type(value).__name__}."
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

    Shared by any caller that needs to pull structured output out of
    free-form LLM text -- not used by ``PlanActAgent``/``ReActAgent``
    themselves (``output_structure`` strict mode makes free-text JSON
    extraction unnecessary for that family), but available to any other
    caller that still needs it. Raises a plain ``TypeError`` for a
    non-string input (an internal-contract violation, not caller-specific).

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


def extract_identifiers(
    source: ast.expr | dict[str, Any] | tuple[Any, ...] | list[Any],
) -> list[str]:
    """
    Walk any ``ast.Name`` reference in ``source`` and return every referenced
    identifier, deduplicated in first-seen order (not a set: multiplicity
    isn't meaningful for a dependency list, but a list keeps a stable,
    orderable contract). Used by ``ScriptActAgent`` (``utils/script.py``) to
    find a statement's real dependencies from its parsed argument tree.

    Accepts a single parsed expression node, a slot's ``kwargs`` dict, or a
    slot's ``args`` tuple/list -- in the dict/tuple/list forms, only values
    that are still unresolved ``ast.expr`` nodes contribute identifiers; an
    already-folded raw literal value contributes none. A ``ToolStatement``
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
    on the opening fence line. Used by ``ScriptActAgent``'s statement parsing
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
    builtins available. Used by ``ScriptActAgent`` (``utils/script.py``, safe
    because every arg reaching this function is guaranteed Call-free by its
    hoisting rule) -- nothing reachable through ``namespace`` can itself be
    invoked.

    Raises whatever the evaluation naturally raises (``TypeError``,
    ``ZeroDivisionError``, ``NameError``, ``KeyError``, ...), uncaught --
    callers decide whether to wrap (parse-time constant folding) or let it
    surface naturally (``resolve_statement_args``).
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
    anywhere -- the exact set ``ScriptActAgent`` (``utils/script.py``) needs.

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
                "statements."
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


def is_dispatched(call: ToolStatement) -> bool:
    """
    True iff ``call`` represents a real dispatched call rather than a
    non-dispatched sentinel (``RETURN_ALIAS`` or ``RHS_ASSIGN_ALIAS``).
    Shared across every ``ToolAgent`` family: a ``PlanActAgent``/
    ``ReActAgent``-sourced statement never has ``tool == RHS_ASSIGN_ALIAS``,
    so excluding it is a correct no-op for those families; a
    ``PY_BUILTIN_ALIAS``/``ATTR_CALL_ALIAS`` statement (``ScriptActAgent``-
    only) is still a real dispatch either way.
    """
    return call.tool not in (RETURN_ALIAS, RHS_ASSIGN_ALIAS)


def tool_identity(call: ToolStatement) -> str:
    """
    Resolve ``call``'s real, human-meaningful tool identity -- correctly
    unsplicing ``ScriptActAgent``'s two sentinel dispatch forms, which
    otherwise hide the real identity behind a fixed sentinel string in
    ``call.tool``.

    A ``PY_BUILTIN_ALIAS`` call's real builtin name lives in ``call.args[0]``
    (spliced in as a plain ``str`` by ``rewrite_builtin_calls``, not an ast
    node). An ``ATTR_CALL_ALIAS`` call's real method name lives in
    ``call.args[1]`` (also a plain ``str``, from ``_build_call_slot``'s own
    ``args=(obj_expr, method_name, *positional)`` construction). Any other
    call (a real registered tool's alias/full_name, or ``RETURN_ALIAS``)
    returns ``call.tool`` directly -- callers are expected to have already
    filtered to ``is_dispatched(call)`` before calling this, so
    ``RETURN_ALIAS``/``RHS_ASSIGN_ALIAS`` never actually reach this function
    in practice, but it is not itself responsible for that filtering.
    """
    if call.tool == PY_BUILTIN_ALIAS:
        return call.args[0]
    if call.tool == ATTR_CALL_ALIAS:
        return call.args[1]
    return call.tool


def compile_batches(
    calls: list[ToolStatement],
    max_concurrency: Optional[int] = None,
    start_batch_index: int = 0,
) -> list[list[ToolStatement]]:
    """
    Group ``calls`` into dependency batches for concurrent execution, and
    stamp each call's ``.batch_index`` with the batch it landed in. Generic
    over ``ToolStatement`` regardless of which agent family produced it.

    A call joins the currently-open batch only if none of its dependencies
    (``extract_identifiers(call.args) | extract_identifiers(call.kwargs)``)
    were bound by a call already sitting in that same open batch (i.e. every
    dependency is satisfiable from an earlier, already-closed batch, a
    registered tool, or a registered constant -- reference validity itself
    is assumed already checked upstream). Otherwise the open batch closes
    first and this call starts a new one.

    Additionally, when about to add a *dispatched* call (``is_dispatched``,
    this module) to a batch that already holds ``max_concurrency`` dispatched
    calls, the batch closes first -- a purely additive concurrency cap, never
    replacing the dependency-conflict closure rule above.
    ``max_concurrency=None`` means no cap (greedy default).

    A ``RETURN_ALIAS`` call is never grouped with anything else -- it always
    closes the current batch, lands alone in a batch of its own, then closes
    that batch too. This guarantees a batch-partial failure elsewhere can
    never suppress an already-resolved return.

    Every call in a batch is stamped with the same ``batch_index`` --
    ``start_batch_index`` plus that batch's own 0-based position among the
    batches this call produces -- the moment the batch closes. Lets a caller
    running multiple generation rounds in one invoke keep indices globally
    unique across rounds by passing the running total in as
    ``start_batch_index``.
    """
    batches: list[list[ToolStatement]] = []
    current_batch: list[ToolStatement] = []
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
            # A return is never grouped with anything else -- closing
            # before AND after guarantees it lands alone in its own batch,
            # so an unrelated failure elsewhere can never suppress it.
            close_current()
            current_batch.append(call)
            close_current()
            continue

        deps = (*extract_identifiers(call.args), *extract_identifiers(call.kwargs))
        if any(name in current_batch_identifiers for name in deps):
            close_current()

        dispatched = is_dispatched(call)
        if (
            dispatched
            and max_concurrency is not None
            and current_batch_dispatched >= max_concurrency
        ):
            close_current()

        current_batch.append(call)
        if call.identifier is not None:
            current_batch_identifiers.add(call.identifier)
        if dispatched:
            current_batch_dispatched += 1

    close_current()
    return batches


def resolve_statement_args(
    call: ToolStatement, resolved: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """
    Resolve one statement's ``args``/``kwargs`` into a plain
    ``(positional, keyword)`` pair ready to splat into
    ``tool._args_kwargs_to_dict(*positional, **keyword)`` (or, for a
    ``rhs_assign``/``return`` sentinel, to read ``keyword["val"]``
    directly). Shared across every ``ToolAgent`` family, including
    ``ast.Starred`` (``*expr`` unpack) and ``KWARGS_UNPACK_KEY``
    (``**expr`` unpack) handling. Those branches are ``ScriptActAgent``-only
    grammar features that simply never trigger for a ``PlanActAgent``/
    ``ReActAgent``-sourced statement (``translate_calls`` never produces a
    ``Starred`` arg or a ``KWARGS_UNPACK_KEY``-keyed kwarg), so sharing this
    one function is safe with zero behavior change for either family.

    Substitutes every ``ast.expr`` value with its concrete value from
    ``resolved`` (identifier -> value) via ``evaluate_expr``, passing
    through any already-plain (non-``ast.expr``) value unchanged (e.g. the
    spliced-in builtin name string from ``rewrite_builtin_calls``). Purely
    transient -- never persisted back onto a ``ToolStatement``.

    Assumes every dependency is already present in ``resolved``; does not
    itself check readiness. A missing identifier is not defensively guarded
    against here -- it surfaces as whatever ``evaluate_expr`` naturally
    raises.
    """

    def resolve_one(value: Any) -> Any:
        return evaluate_expr(value, resolved) if isinstance(value, ast.expr) else value

    positional: list[Any] = []
    for entry in call.args:
        if isinstance(entry, ast.Starred):
            positional.extend(resolve_one(entry.value))
        else:
            positional.append(resolve_one(entry))

    keyword: dict[str, Any] = {}
    for name, value in call.kwargs.items():
        if name == KWARGS_UNPACK_KEY:
            continue
        keyword[name] = resolve_one(value)

    if KWARGS_UNPACK_KEY in call.kwargs:
        unpacked = resolve_one(call.kwargs[KWARGS_UNPACK_KEY])
        overlap = set(unpacked) & set(keyword)
        if overlap:
            raise TypeError(
                f"got multiple values for keyword argument(s): {sorted(overlap)!r}"
            )
        keyword.update(unpacked)

    return positional, keyword
