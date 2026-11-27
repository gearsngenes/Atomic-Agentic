from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional


from ..constants.agents import (
    ATTR_CALL_ALIAS,
    CODE_FENCE_PATTERN,
    EMBEDDED_CODE_FENCE_PATTERN,
    DUNDER_ATTRIBUTE_PATTERN,
    KWARGS_UNPACK_KEY,
    LEADING_CODE_FENCE_PATTERN,
    PY_BUILTIN_ALIAS,
    TRAILING_CODE_FENCE_PATTERN,
    UNSUPPORTED_EXPR_LABELS,
)
from ..exceptions import BlackboardParseError
from ..models.agents.blackboard_models import Statement, Step
from ..models.results.agents import ToolUsageRecord, ToolUsageReport

if TYPE_CHECKING:
    from ..invokables.atomic_invokable import AtomicInvokable

__all__ = [
    "evaluate_expr",
    "extract_identifiers",
    "reject_unsupported_forms",
    "strip_code_fence",
    "tool_identity",
    "build_usage_report",
    "resolve_statement_args",
    "check_call_bindings",
]


def extract_identifiers(
    source: ast.expr | dict[str, Any] | tuple[Any, ...] | list[Any],
) -> list[str]:
    """
    Walk any ``ast.Name`` reference in ``source`` and return every referenced
    identifier, deduplicated in first-seen order (not a set: multiplicity
    isn't meaningful for a dependency list, but a list keeps a stable,
    orderable contract). Used by ``PlanActAgent`` (``utils/script.py``) to
    find a statement's real dependencies from its parsed argument tree.

    Accepts a single parsed expression node, a step's ``kwargs`` dict, or a
    step's ``args`` tuple/list -- in the dict/tuple/list forms, only values
    that are still unresolved ``ast.expr`` nodes contribute identifiers; an
    already-folded raw literal value contributes none. A ``Step``
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
    Extract the code from a markdown code fence around generated text, if
    present. ``PlanActAgent`` asks for its plan as a single fenced block;
    this also tolerates a missing or one-sided fence and prose around the
    block. Generic to any language tag (or none) on the opening fence line.
    Used by ``PlanActAgent``'s statement parsing (``utils/script.py``).

    1. A fully matched pair wrapping the whole text (``CODE_FENCE_PATTERN``)
       -- its inner text is used as-is.
    2. Exactly one fenced block elsewhere in the text, e.g. with a sentence
       before or after it (``EMBEDDED_CODE_FENCE_PATTERN``) -- its inner text;
       the prose outside is dropped. Two or more blocks are ambiguous and fall
       through to step 3 unchanged, so ``ast.parse`` rejects them.
    3. Otherwise a leading and/or trailing fence line is stripped
       independently (a model emitting only one side). A fence appearing only
       mid-text is left alone (``ast.parse`` will reject that on its own
       terms, as a real structural problem).
    """
    full_match = CODE_FENCE_PATTERN.match(raw_text)
    if full_match:
        return full_match.group(1)
    embedded = EMBEDDED_CODE_FENCE_PATTERN.findall(raw_text)
    if len(embedded) == 1:
        return embedded[0]
    text = LEADING_CODE_FENCE_PATTERN.sub("", raw_text, count=1)
    text = TRAILING_CODE_FENCE_PATTERN.sub("", text, count=1)
    return text


def evaluate_expr(node: ast.expr, namespace: dict[str, Any]) -> Any:
    """
    Evaluate one parsed expression node against a namespace, with no
    builtins available. Used by ``PlanActAgent`` (``utils/script.py``, safe
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
    anywhere -- the exact set ``PlanActAgent`` (``utils/script.py``) needs.

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


def tool_identity(step: Step) -> str:
    """The real tool name of a tool step: the builtin name (``args[0]``) for
    ``PY_BUILTIN_ALIAS``, the method name (``args[1]``) for
    ``ATTR_CALL_ALIAS``, else ``step.tool``."""
    if step.tool == PY_BUILTIN_ALIAS:
        return step.args[0]
    if step.tool == ATTR_CALL_ALIAS:
        return step.args[1]
    return step.tool


def build_usage_report(statements: Iterable[Statement]) -> ToolUsageReport:
    """
    One ``ToolUsageRecord`` per distinct real tool identity among the tool
    steps that ran or failed, in first-seen order across ``statements`` as
    given. ``total_dispatched`` counts those steps and ``total_failed`` the
    failed ones. Shared by ``ToolAgentRecord.usage_report()`` and each
    family's ``result_payload_from_task``.
    """
    counts: dict[str, int] = {}
    total_failed = 0
    for statement in statements:
        for step in statement.steps:
            if step.kind != "tool" or step.status == "not_run":
                continue
            identity = tool_identity(step)
            counts[identity] = counts.get(identity, 0) + 1
            total_failed += step.status == "failed"
    by_tool = tuple(ToolUsageRecord(tool_name=name, call_count=n) for name, n in counts.items())
    return ToolUsageReport(
        by_tool=by_tool,
        total_dispatched=sum(counts.values()),
        total_failed=total_failed,
    )


def resolve_statement_args(
    step: Step, resolved: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """
    Resolve one step's ``args``/``kwargs`` into a plain
    ``(positional, keyword)`` pair ready to splat into
    ``tool._args_kwargs_to_dict(*positional, **keyword)`` (or, for an eval
    step, to read ``positional[0]`` directly). Handles ``ast.Starred`` (``*expr`` unpack) and
    ``KWARGS_UNPACK_KEY`` (``**expr`` unpack).

    Substitutes every ``ast.expr`` value with its concrete value from
    ``resolved`` (identifier -> value) via ``evaluate_expr``, passing
    through any already-plain (non-``ast.expr``) value unchanged (e.g. the
    spliced-in builtin or method name string). Purely transient -- never
    persisted back onto the step.

    Assumes every dependency is already present in ``resolved``; does not
    itself check readiness. A missing identifier is not defensively guarded
    against here -- it surfaces as whatever ``evaluate_expr`` naturally
    raises.
    """

    def resolve_one(value: Any) -> Any:
        return evaluate_expr(value, resolved) if isinstance(value, ast.expr) else value

    positional: list[Any] = []
    for entry in step.args:
        if isinstance(entry, ast.Starred):
            positional.extend(resolve_one(entry.value))
        else:
            positional.append(resolve_one(entry))

    keyword: dict[str, Any] = {}
    for name, value in step.kwargs.items():
        if name == KWARGS_UNPACK_KEY:
            continue
        keyword[name] = resolve_one(value)

    if KWARGS_UNPACK_KEY in step.kwargs:
        unpacked = resolve_one(step.kwargs[KWARGS_UNPACK_KEY])
        overlap = set(unpacked) & set(keyword)
        if overlap:
            raise TypeError(
                f"got multiple values for keyword argument(s): {sorted(overlap)!r}"
            )
        keyword.update(unpacked)

    return positional, keyword


def check_call_bindings(
    statements: Sequence[Statement],
    toolbox: Mapping[str, AtomicInvokable],
) -> list[str]:
    """
    Dry bind: catch argument-shape mistakes at plan time. For each tool step
    whose ``tool`` is a registered tool id (eval, builtin and method steps are
    skipped), bind placeholder values through the tool's
    ``_args_kwargs_to_dict`` and report a ``TypeError`` (too many positionals,
    unknown keyword, duplicate argument) as an issue.

    A step with a ``*`` unpack or a ``**`` unpack is skipped (arity unknown).
    No values are evaluated and nothing is dispatched. A missing required
    argument is not reported (binding does not check it); it still surfaces
    at invoke time. Returns ``[]`` when every checked call binds.
    """
    issues: list[str] = []
    for statement in statements:
        for step in statement.steps:
            if step.kind != "tool" or step.tool not in toolbox:
                continue
            if KWARGS_UNPACK_KEY in step.kwargs or any(
                isinstance(arg, ast.Starred) for arg in step.args
            ):
                continue
            try:
                toolbox[step.tool]._args_kwargs_to_dict(
                    *([None] * len(step.args)), **{key: None for key in step.kwargs}
                )
            except TypeError as e:
                label = step.name or statement.target or "(unassigned)"
                issues.append(
                    f"statement producing {label} calls {step.tool!r} with arguments "
                    f"that do not fit its signature: {e}"
                )
    return issues
