from __future__ import annotations

import ast
from typing import Any

from ..constants.toolagent2 import HOISTED_NAME_PREFIX, RHS_ASSIGN_ALIAS
from ..exceptions import BlackboardParseError
from ..models.agents.toolagent2_models import BlackboardSlotV2

__all__ = [
    "parse_statement_to_slots",
    "extract_dependencies_v2",
    "resolve_slot_args",
]


def _evaluate_expr(node: ast.expr, namespace: dict[str, Any]) -> Any:
    """
    Evaluate one parsed expression node against a namespace, with no
    builtins available. Safe because every arg reaching this function is
    guaranteed Call-free by the hoisting rule -- nothing reachable through
    ``namespace`` can itself be invoked.

    Raises whatever the evaluation naturally raises (``TypeError``,
    ``ZeroDivisionError``, ``NameError``, ``KeyError``, ...), uncaught --
    callers decide whether to wrap (parse-time constant folding) or let it
    surface naturally (``resolve_slot_args``).
    """
    expr_wrapper = ast.Expression(body=node)
    ast.fix_missing_locations(expr_wrapper)
    code = compile(expr_wrapper, filename="<blackboard-slot-v2>", mode="eval")
    return eval(code, {"__builtins__": {}}, namespace)


def extract_dependencies_v2(source: ast.expr | dict[str, Any]) -> list[str]:
    """
    Generalized replacement for ``utils/agents.py``'s ``extract_dependencies``
    -- walks any ``ast.Name`` reference rather than matching a fixed
    placeholder regex. Returns a deduplicated, first-seen-order list (not a
    set): multiplicity is not meaningful for a dependency list, but a list
    is kept for a stable, orderable contract.

    Accepts either a single parsed expression node, or a slot's ``args``
    dict -- in the dict form, only values that are still unresolved
    ``ast.expr`` nodes contribute dependencies; an already-folded raw
    literal value contributes none.
    """
    raw: list[str] = []

    if isinstance(source, dict):
        for value in source.values():
            if isinstance(value, ast.expr):
                raw.extend(extract_dependencies_v2(value))
    else:
        raw.extend(
            node.id
            for node in ast.walk(source)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )

    return list(dict.fromkeys(raw))


def resolve_slot_args(args: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any]:
    """
    Substitute every unresolved ``ast.expr`` value in ``args`` with its
    concrete value from ``resolved`` (identifier -> value), leaving
    already-folded raw literals untouched. Purely transient -- never
    persisted back onto a ``BlackboardSlotV2``.

    Assumes every dependency in ``args`` is already present in ``resolved``;
    does not itself check readiness (a future ``prepare()``-phase caller's
    job, combining ``extract_dependencies_v2`` with an
    all-dependencies-have-results check before ever calling this). A
    missing identifier is not defensively guarded against here -- it
    surfaces as whatever `_evaluate_expr` naturally raises.
    """
    output: dict[str, Any] = {}
    for name, value in args.items():
        if isinstance(value, ast.expr):
            output[name] = _evaluate_expr(value, resolved)
        else:
            output[name] = value
    return output


def _process_call_args(
    call_node: ast.Call,
    *,
    counter: list[int],
    start_index: int,
    hoisted: list[BlackboardSlotV2],
) -> dict[str, Any]:
    """
    Build one call's final args dict: hoists any nested call found in each
    keyword value (appending synthesized slots to ``hoisted``), then
    eagerly constant-folds each resulting dependency-free argument. Shared
    by both the top-level call (case A) and every recursively hoisted call.
    """
    if call_node.args:
        raise BlackboardParseError(
            "positional call arguments are not supported; use keyword arguments only."
        )
    for kw in call_node.keywords:
        if kw.arg is None:
            raise BlackboardParseError(
                "**kwargs-style double-star unpacking in a call is not supported."
            )

    args: dict[str, Any] = {}
    for kw in call_node.keywords:
        processed_value = _hoist_calls(
            kw.value, counter=counter, start_index=start_index, hoisted=hoisted
        )
        deps = extract_dependencies_v2(processed_value)
        if not deps:
            try:
                args[kw.arg] = _evaluate_expr(processed_value, {})
            except Exception as e:
                raise BlackboardParseError(
                    f"argument {kw.arg!r} is a constant expression that failed "
                    f"to evaluate: {e!r}"
                ) from e
        else:
            args[kw.arg] = processed_value

    return args


def _hoist_calls(
    node: ast.expr,
    *,
    counter: list[int],
    start_index: int,
    hoisted: list[BlackboardSlotV2],
) -> ast.expr:
    """
    Post-order rewrite: replaces every ``Call`` node found anywhere within
    ``node`` (at any depth -- a ``BinOp`` operand, another call's keyword
    value, an f-string's embedded expression, a container literal element,
    ...) with a ``Name`` reference to a newly synthesized, hoisted
    ``BlackboardSlotV2``, appended to ``hoisted`` in discovery order.

    ``ast.NodeTransformer.generic_visit`` recurses into a call's own
    children before ``visit_Call`` builds that call's own hoisted slot, so
    doubly/triply-nested calls flatten correctly bottom-up.
    """

    class _CallHoister(ast.NodeTransformer):
        def visit_Call(self, call_node: ast.Call) -> ast.Name:
            # Only recurse into keyword values, not `func` -- a call's own
            # callable is always a plain dotted-name chain in this grammar,
            # never itself a nested call.
            for kw in call_node.keywords:
                kw.value = self.visit(kw.value)

            index = counter[0] + start_index
            counter[0] += 1
            hoisted_identifier = f"{HOISTED_NAME_PREFIX}{index}"

            hoisted_args = _process_call_args(
                call_node, counter=counter, start_index=start_index, hoisted=hoisted
            )
            tool_name = ast.unparse(call_node.func)
            hoisted.append(
                BlackboardSlotV2(identifier=hoisted_identifier, tool=tool_name, args=hoisted_args)
            )

            replacement = ast.Name(id=hoisted_identifier, ctx=ast.Load())
            return ast.copy_location(replacement, call_node)

    return _CallHoister().visit(node)


def parse_statement_to_slots(statement: str, start_index: int = 0) -> list[BlackboardSlotV2]:
    """
    Parse one raw generated statement string into an ordered list of
    ``BlackboardSlotV2`` objects: any auto-hoisted slots first (in
    discovery/post-order), the statement's own slot last.

    ``start_index`` lets a future whole-block caller avoid ``_HOIST_N``
    collisions across multiple calls within the same subtask -- pass the
    running count of hoisted slots already emitted so far for this block;
    this function is otherwise pure/stateless.

    Raises ``BlackboardParseError`` for every rejection category (always via
    ``raise ... from e`` where an underlying exception exists) -- never
    silently repaired, never returned as an issues list.
    """
    try:
        tree = ast.parse(statement, mode="exec")
    except SyntaxError as e:
        raise BlackboardParseError(str(e)) from e

    if len(tree.body) != 1:
        raise BlackboardParseError(f"expected exactly one statement; got {len(tree.body)}.")

    stmt = tree.body[0]
    # isinstance, not a looser check -- ast.AugAssign ("x += 1") is a
    # distinct node type and never satisfies this, rejecting it for free.
    if not isinstance(stmt, ast.Assign):
        raise BlackboardParseError(
            f"expected a single assignment statement; got {type(stmt).__name__}."
        )

    if len(stmt.targets) != 1:
        raise BlackboardParseError("chained assignment (multiple targets) is not supported.")

    target = stmt.targets[0]
    if not isinstance(target, ast.Name):
        raise BlackboardParseError(
            "the assignment target must be a single bare identifier; got "
            f"{type(target).__name__}."
        )

    identifier = target.id
    if identifier.startswith(HOISTED_NAME_PREFIX):
        raise BlackboardParseError(
            f"identifier {identifier!r} uses the reserved hoisted-name prefix "
            f"{HOISTED_NAME_PREFIX!r}."
        )

    # Unwrap an optional top-level `await`.
    rhs = stmt.value
    awaited = False
    if isinstance(rhs, ast.Await):
        awaited = True
        rhs = rhs.value

    hoisted: list[BlackboardSlotV2] = []
    counter = [0]

    if isinstance(rhs, ast.Call):
        # Case A: bare top-level call -- the only shape `awaited` stays True for.
        tool = ast.unparse(rhs.func)
        args = _process_call_args(rhs, counter=counter, start_index=start_index, hoisted=hoisted)
    else:
        # Case B: rhs_assign. Also where an `await` wrapping anything other
        # than a bare call lands -- forced False regardless of the unwrap
        # above, since this branch means rhs was never a bare call at all.
        awaited = False
        tool = RHS_ASSIGN_ALIAS
        processed = _hoist_calls(rhs, counter=counter, start_index=start_index, hoisted=hoisted)
        deps = extract_dependencies_v2(processed)
        if not deps:
            try:
                val: Any = _evaluate_expr(processed, {})
            except Exception as e:
                raise BlackboardParseError(
                    f"expression is a constant that failed to evaluate: {e!r}"
                ) from e
        else:
            val = processed
        args = {"val": val}

    final_slot = BlackboardSlotV2(identifier=identifier, tool=tool, args=args, awaited=awaited)
    return [*hoisted, final_slot]
