from __future__ import annotations

import ast
import builtins
import io
import re
import tokenize
from typing import Any, Optional

from ..constants.agents import (
    ATTR_CALL_ALIAS,
    CODE_FENCE_PATTERN,
    DUNDER_ATTRIBUTE_PATTERN,
    EXCLUDED_PY_BUILTINS,
    KWARGS_UNPACK_KEY,
    LEADING_CODE_FENCE_PATTERN,
    PAUSE_PATTERN,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
    SUB_NAME_PREFIX,
    TASK_RESULT_PREFIX,
    TRAILING_CODE_FENCE_PATTERN,
    UNSUPPORTED_EXPR_LABELS,
)
from ..exceptions import BlackboardParseError
from ..models.agents.blackboard_models import CodeStatement
from .agents import extract_identifiers

__all__ = [
    "parse_statement_to_slots",
    "resolve_slot_args",
    "parse_generation",
    "rewrite_builtin_calls",
    "is_dispatched_slot",
    "validate_references",
    "compile_batches",
    "render_completed_as_python",
    "render_cache_snapshot",
]


def _strip_code_fence(raw_text: str) -> str:
    """
    Strip a markdown code fence wrapping the generation, if present --
    defensive against a model wrapping otherwise-valid output in a code
    fence despite being told not to. Generic to any language tag (or none)
    on the opening fence line.

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


def resolve_slot_args(
    statement: CodeStatement, resolved: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """
    Resolve one statement's ``args``/``kwargs`` into a plain
    ``(positional, keyword)`` pair ready to splat into
    ``tool._args_kwargs_to_dict(*positional, **keyword)`` (or, for a
    ``rhs_assign``/``return`` sentinel, to read ``keyword["val"]``
    directly). Substitutes every unresolved ``ast.expr`` value with its
    concrete value from ``resolved`` (identifier -> value), passing through
    any already-plain (non-``ast.expr``) value unchanged (e.g. the
    spliced-in builtin name string from ``rewrite_builtin_calls``). Purely
    transient -- never persisted back onto a ``CodeStatement``.

    Assumes every dependency is already present in ``resolved``; does not
    itself check readiness (a ``prepare()``-phase caller's job, combining
    ``extract_identifiers`` with an all-dependencies-have-results check
    before ever calling this). A missing identifier is not defensively
    guarded against here -- it surfaces as whatever ``_evaluate_expr``
    naturally raises.

    A positional entry that is an ``ast.Starred`` (a ``*expr`` unpack) has
    its ``.value`` resolved and the result spliced into ``positional`` via
    ``list.extend`` -- raises naturally (``TypeError``) if the resolved
    value isn't iterable, uncaught here, same "let it surface" precedent
    as everything else in this function. A keyword entry stored under
    ``KWARGS_UNPACK_KEY`` (a ``**expr`` unpack) is resolved, checked for a
    colliding key against the already-resolved named keywords -- raising
    ``TypeError`` on overlap, matching real Python's own runtime behavior
    for this exact collision (CPython raises rather than silently
    favoring one side) -- then merged in.
    """

    def resolve_one(value: Any) -> Any:
        return _evaluate_expr(value, resolved) if isinstance(value, ast.expr) else value

    positional: list[Any] = []
    for entry in statement.args:
        if isinstance(entry, ast.Starred):
            positional.extend(resolve_one(entry.value))
        else:
            positional.append(resolve_one(entry))

    keyword: dict[str, Any] = {}
    for name, value in statement.kwargs.items():
        if name == KWARGS_UNPACK_KEY:
            continue
        keyword[name] = resolve_one(value)

    if KWARGS_UNPACK_KEY in statement.kwargs:
        unpacked = resolve_one(statement.kwargs[KWARGS_UNPACK_KEY])
        overlap = set(unpacked) & set(keyword)
        if overlap:
            raise TypeError(
                f"got multiple values for keyword argument(s): {sorted(overlap)!r}"
            )
        keyword.update(unpacked)

    return positional, keyword


def _process_call_args(
    call_node: ast.Call,
    *,
    counter: list[int],
    start_index: int,
    hoisted: list[CodeStatement],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Build one call's final ``(args, kwargs)`` pair: hoists any nested call
    found in each positional or keyword value (appending synthesized slots
    to ``hoisted``), then validates each resulting dependency-free argument
    via a dry-run evaluation (raising early on a guaranteed-bad constant
    expression) without folding it -- the original expression form is
    always what gets stored. Shared by both the top-level call (case A)
    and every recursively hoisted call.

    A positional entry that is an ``ast.Starred`` (a ``*expr`` unpack) is
    never eagerly folded, regardless of whether its own inner expr has
    dependencies -- its Starred-ness must survive to resolve time
    (``resolve_slot_args``), and an already-folded plain value has no way
    to carry that tag. It still gets the same dependency-free dry-run
    validation as every other argument category, applied to its inner
    expr before it's wrapped back in ``ast.Starred``. A keyword entry whose
    ``kw.arg is None`` (a ``**expr`` unpack) is stored under
    ``KWARGS_UNPACK_KEY`` -- at most one per call; a second one raises
    immediately.
    """
    args: list[Any] = []
    for arg_node in call_node.args:
        if isinstance(arg_node, ast.Starred):
            inner = _hoist_calls(
                arg_node.value, counter=counter, start_index=start_index, hoisted=hoisted
            )
            deps = extract_identifiers(inner)
            if not deps:
                try:
                    _evaluate_expr(inner, {})
                except Exception as e:
                    raise BlackboardParseError(
                        "* unpack is a constant expression that failed to "
                        f"evaluate: {e!r}"
                    ) from e
            args.append(ast.Starred(value=inner, ctx=ast.Load()))
            continue

        processed = _hoist_calls(
            arg_node, counter=counter, start_index=start_index, hoisted=hoisted
        )
        deps = extract_identifiers(processed)
        if not deps:
            try:
                _evaluate_expr(processed, {})
            except Exception as e:
                raise BlackboardParseError(
                    "positional argument is a constant expression that "
                    f"failed to evaluate: {e!r}"
                ) from e
        args.append(processed)

    kwargs: dict[str, Any] = {}
    seen_unpack = False
    for kw in call_node.keywords:
        processed_value = _hoist_calls(
            kw.value, counter=counter, start_index=start_index, hoisted=hoisted
        )
        if kw.arg is None:
            if seen_unpack:
                raise BlackboardParseError(
                    "at most one ** unpack is supported per call."
                )
            seen_unpack = True
            key = KWARGS_UNPACK_KEY
        else:
            key = kw.arg

        deps = extract_identifiers(processed_value)
        if not deps:
            try:
                _evaluate_expr(processed_value, {})
            except Exception as e:
                raise BlackboardParseError(
                    f"argument {key!r} is a constant expression that failed "
                    f"to evaluate: {e!r}"
                ) from e
        kwargs[key] = processed_value

    return tuple(args), kwargs


def _build_call_slot(
    call_node: ast.Call,
    *,
    identifier: Optional[str],
    counter: list[int],
    start_index: int,
    hoisted: list[CodeStatement],
) -> CodeStatement:
    """
    Build one ``CodeStatement`` for ``call_node``, branching on whether its
    ``func`` is a plain dotted-name chain (a registered tool/builtin id) or
    an ``ast.Attribute`` (a method call on some object). Shared by the
    top-level bare-unassigned-call and assignment (``name = call(...)``)
    statement shapes in ``parse_statement_to_slots``, and by
    ``_hoist_calls``'s own nested-call hoisting -- the one place this
    branch lives, so all three call sites can never drift out of sync.

    For an attribute/method call: the method name (``call_node.func.attr``)
    is checked against ``DUNDER_ATTRIBUTE_PATTERN`` here -- the one
    position ``_hoist_calls``'s own rejection scan never sees, since it
    only ever walks ``call_node.func.value``, not ``call_node.func``
    itself. The object sub-expression (``call_node.func.value``) is run
    through ``_hoist_calls`` exactly like any other operand: a call-free
    chain (``x.y.z``) passes through unchanged (no new binding created),
    an embedded call (``x.y.method1()`` inside
    ``x.y.method1().z.method2()``) gets hoisted into its own prior slot
    first, recursively, to any depth.
    """
    if isinstance(call_node.func, ast.Attribute):
        method_name = call_node.func.attr
        if DUNDER_ATTRIBUTE_PATTERN.fullmatch(method_name):
            raise BlackboardParseError(
                f"dunder attribute access is not permitted: "
                f"{ast.unparse(call_node.func)!r}."
            )
        obj_expr = _hoist_calls(
            call_node.func.value, counter=counter, start_index=start_index, hoisted=hoisted
        )
        positional, keyword = _process_call_args(
            call_node, counter=counter, start_index=start_index, hoisted=hoisted
        )
        return CodeStatement(
            identifier=identifier,
            tool=ATTR_CALL_ALIAS,
            args=(obj_expr, method_name, *positional),
            kwargs=keyword,
        )

    tool_name = ast.unparse(call_node.func)
    positional, keyword = _process_call_args(
        call_node, counter=counter, start_index=start_index, hoisted=hoisted
    )
    return CodeStatement(identifier=identifier, tool=tool_name, args=positional, kwargs=keyword)


def _reject_await(node: ast.expr) -> None:
    """
    Raise the one dedicated `await`-no-longer-supported parse error. Shared
    by `_hoist_calls`'s rejection scan (an `await` nested anywhere, or used
    as a bare/rhs-assign top-level expression that still routes through
    `_hoist_calls`) and `parse_statement_to_slots`'s top-level bare-call
    shape (the sole position that never reaches `_hoist_calls`, since it
    reads the statement's own `Expr.value` directly rather than one of a
    call's own argument expressions). Keeps wording identical across both
    sites rather than risking drift between two independent raises.
    """
    raise BlackboardParseError(
        "'await' is not supported: "
        f"{ast.unparse(node)!r} -- write the call as an ordinary "
        "statement; execution order is inferred automatically from data "
        "dependencies."
    )


def _hoist_calls(
    node: ast.expr,
    *,
    counter: list[int],
    start_index: int,
    hoisted: list[CodeStatement],
) -> ast.expr:
    """
    Post-order rewrite: replaces every ``Call`` node found anywhere within
    ``node`` (at any depth -- a ``BinOp`` operand, another call's keyword
    value, an f-string's embedded expression, a container literal element,
    ...) with a ``Name`` reference to a newly synthesized, hoisted
    ``CodeStatement``, appended to ``hoisted`` in discovery order.

    ``ast.NodeTransformer.generic_visit`` recurses into a call's own
    children before ``visit_Call`` builds that call's own hoisted slot, so
    doubly/triply-nested calls flatten correctly bottom-up.

    Rejects (before any hoisting) an ``ast.IfExp`` anywhere in ``node``
    whose either branch contains a ``Call`` -- both branches would
    otherwise be hoisted and eagerly executed regardless of the condition,
    defeating the ternary's short-circuit semantics and wasting tool-call
    budget on the untaken branch. Checked here, the single choke point
    every caller (a bare rhs_assign, a top-level call's own keyword
    arguments via ``_process_call_args``, and any nested/hoisted call's
    keyword arguments) funnels through -- a per-call-site check would miss
    a ternary buried inside a call argument.

    Also rejects, at this same choke point, any ``ast.Await`` found
    anywhere in ``node`` via ``_reject_await`` -- there is no more
    await-aware execution semantics left in this grammar (concurrency is
    now inferred from data dependencies alone, bounded by the agent's own
    ``tool_concurrency_limit``, never signaled by the model), so any use
    (bare, nested inside a call argument, an RHS assignment target) is
    rejected uniformly, before hoisting proceeds -- covers arbitrary
    nesting depth for the same reason the comprehension/lambda scan below
    does.

    Also rejects, at this same choke point and just as unconditionally as
    the ``IfExp`` check below, any comprehension (``ast.ListComp``/
    ``ast.SetComp``/``ast.DictComp``/``ast.GeneratorExp``) or ``ast.Lambda``
    found anywhere in ``node`` -- regardless of whether it contains a call.
    Both constructs introduce a local binding scope (a comprehension's loop
    variable(s), a lambda's parameters) that neither this function nor
    ``extract_identifiers`` has any awareness of: a call-free comprehension's
    loop variable would otherwise be misclassified as an unresolved external
    dependency (a misleading error), and worse, one whose bound name happens
    to collide with an already-bound identifier elsewhere in the plan would
    silently resolve against that unrelated value instead of erroring at
    all. Checked before hoisting proceeds, so this also catches the
    construct nested arbitrarily deep (inside a call's own keyword argument,
    inside another hoisted call) -- the scan walks the whole original tree
    before any rewriting happens.

    Also rejects, at this same choke point, any ``ast.Attribute`` node
    anywhere in ``node`` whose ``.attr`` matches ``DUNDER_ATTRIBUTE_PATTERN``
    -- a bare dunder read (``x.__class__``) or one buried mid-chain
    (``x.__class__.y``), at any depth, including inside a to-be-hoisted
    call's own object expression. Closes the classic attribute-chaining
    sandbox-escape class (``().__class__.__bases__[0].__subclasses__()``-
    style), which the ``{"__builtins__": {}}`` eval lockout elsewhere in
    this module does not defend against on its own. A method-call's own
    method name (``call_node.func.attr``) sits outside this scan (it's
    never itself walked as a standalone ``ast.Attribute`` node here) and is
    checked separately, in ``_build_call_slot``.
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
            _reject_await(candidate)

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

    def _hoist_one_call(call_node: ast.Call) -> ast.Name:
        """Build one hoisted slot for `call_node` (appended to `hoisted`,
        via `_build_call_slot` -- handles both a plain tool/builtin id and
        an attribute/method call) and return a `Name` reference to it."""
        index = counter[0] + start_index
        counter[0] += 1
        hoisted_identifier = f"{SUB_NAME_PREFIX}{index}"

        hoisted.append(
            _build_call_slot(
                call_node,
                identifier=hoisted_identifier,
                counter=counter,
                start_index=start_index,
                hoisted=hoisted,
            )
        )

        replacement = ast.Name(id=hoisted_identifier, ctx=ast.Load())
        return ast.copy_location(replacement, call_node)

    class _CallHoister(ast.NodeTransformer):
        def visit_Call(self, call_node: ast.Call) -> ast.Name:
            # Only recurse into keyword values here, not `func` -- a plain
            # tool/builtin call's own callable is always a dotted-name
            # chain, never itself a nested call. An attribute/method call's
            # func.value CAN contain a nested call, but that's hoisted
            # explicitly inside _build_call_slot (a fresh _hoist_calls
            # entry), not via this transformer's own traversal.
            for kw in call_node.keywords:
                kw.value = self.visit(kw.value)
            return _hoist_one_call(call_node)

    return _CallHoister().visit(node)


def parse_statement_to_slots(statement: str, start_index: int = 0) -> list[CodeStatement]:
    """
    Parse one raw generated statement string into an ordered list of
    ``CodeStatement`` objects: any auto-hoisted slots first (in
    discovery/post-order), the statement's own slot last.

    Three top-level statement shapes are accepted: an assignment (case A/B
    below), a bare (unassigned) call, and a ``return`` statement. The
    latter two produce a slot with
    ``identifier=None``: nothing can ever reference either by name, so no
    synthesized name is needed the way hoisting needs one.

    ``start_index`` lets a future whole-block caller avoid ``_SUB_N``
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

    # Terminal case: a bare `return <expr>` (or bare `return`, treated as
    # `return None`). No Python-grammar obstacle to a module-level Return
    # node here -- the "return outside function" check only fires at
    # compile()-to-bytecode time, which this pipeline never does to a whole
    # statement (only to bare expressions, via _evaluate_expr).
    if isinstance(stmt, ast.Return):
        hoisted: list[CodeStatement] = []
        counter = [0]
        return_value = stmt.value if stmt.value is not None else ast.Constant(value=None)
        processed = _hoist_calls(
            return_value, counter=counter, start_index=start_index, hoisted=hoisted
        )
        deps = extract_identifiers(processed)
        if not deps:
            try:
                _evaluate_expr(processed, {})
            except Exception as e:
                raise BlackboardParseError(
                    f"return expression is a constant that failed to evaluate: {e!r}"
                ) from e
        val: Any = processed
        final_slot = CodeStatement(identifier=None, tool=RETURN_ALIAS, kwargs={"val": val})
        return [*hoisted, final_slot]

    # A bare top-level `await ...` never reaches `_hoist_calls` (this
    # branch reads `stmt.value` directly, not one of a call's own argument
    # expressions) -- rejected explicitly here via the same shared helper.
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Await):
        _reject_await(stmt.value)

    # Bare (unassigned) call -- void-style tool calls the model doesn't
    # need a name for, still logged as a real slot for budget/history.
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        bare_rhs = stmt.value
        hoisted = []
        counter = [0]
        final_slot = _build_call_slot(
            bare_rhs, identifier=None, counter=counter, start_index=start_index, hoisted=hoisted
        )
        return [*hoisted, final_slot]

    # isinstance, not a looser check -- ast.AugAssign ("x += 1") is a
    # distinct node type and never satisfies this, rejecting it for free.
    if not isinstance(stmt, ast.Assign):
        raise BlackboardParseError(
            "expected an assignment, a bare tool call, or a return "
            f"statement; got {type(stmt).__name__}."
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
    if identifier.startswith(SUB_NAME_PREFIX):
        raise BlackboardParseError(
            f"identifier {identifier!r} uses the reserved hoisted-name prefix "
            f"{SUB_NAME_PREFIX!r}."
        )
    if identifier.startswith(TASK_RESULT_PREFIX):
        raise BlackboardParseError(
            f"identifier {identifier!r} uses the reserved task-result-"
            f"addressing prefix {TASK_RESULT_PREFIX!r}; task_result_N names "
            "are fixed, read-only references to prior invocations and "
            "cannot be assigned to."
        )

    rhs = stmt.value
    hoisted: list[CodeStatement] = []
    counter = [0]

    if isinstance(rhs, ast.Call):
        # Case A: bare top-level call -- delegate the plain-tool-vs-
        # attribute-call branch to _build_call_slot (shared with the
        # bare-unassigned-call shape above and _hoist_calls's own
        # nested-call hoisting).
        final_slot = _build_call_slot(
            rhs, identifier=identifier, counter=counter, start_index=start_index, hoisted=hoisted
        )
        return [*hoisted, final_slot]

    # Case B: rhs_assign. An `ast.Await` here (e.g. `x = await f()`) is
    # no longer specially unwrapped -- it flows into `_hoist_calls`
    # below exactly like any other node, which rejects it via its own
    # rejection scan (`ast.walk` yields `rhs` itself before its
    # children, same precedent already established for a bare
    # comprehension/lambda RHS).
    tool = RHS_ASSIGN_ALIAS
    # Ternary-with-calls rejection lives inside _hoist_calls itself now
    # (the common choke point every call path funnels through) --
    # nothing extra needed here.
    processed = _hoist_calls(rhs, counter=counter, start_index=start_index, hoisted=hoisted)
    deps = extract_identifiers(processed)
    if not deps:
        try:
            _evaluate_expr(processed, {})
        except Exception as e:
            raise BlackboardParseError(
                f"expression is a constant that failed to evaluate: {e!r}"
            ) from e
    val: Any = processed
    args, kwargs = (), {"val": val}

    final_slot = CodeStatement(identifier=identifier, tool=tool, args=args, kwargs=kwargs)
    return [*hoisted, final_slot]


def _find_pause_marker(text: str) -> tuple[int, int] | None:
    """
    Locate the first real ``# PAUSE`` comment in ``text`` via ``tokenize``
    rather than a raw-text regex scan, so a legal (freely-interspersed,
    triple-quoted-preferred) reasoning-note string that happens to contain
    the text "# PAUSE" on one of its own lines can never be misread as the
    real sentinel -- ``tokenize`` never emits a ``COMMENT`` token from
    inside a ``STRING`` token, unlike a plain regex over raw text, which
    has no concept of "am I inside a string literal."

    Only a comment that is the sole content on its line (nothing but
    whitespace precedes it) counts, matching ``PAUSE_PATTERN``'s original
    line-anchored intent -- a trailing comment after real code on the same
    line is not a marker, unchanged from before this rewrite.

    Returns the marker's ``(row, col)`` start position (1-indexed row,
    ``tokenize``'s own convention), or ``None`` if no real marker exists
    (including when ``text`` fails to tokenize at all -- a genuine
    lexical error surfaces downstream via ``ast.parse``, exactly as it
    would have regardless of this function).
    """
    lines = text.splitlines()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type != tokenize.COMMENT or not PAUSE_PATTERN.match(tok.string):
                continue
            row, col = tok.start
            if lines[row - 1][:col].strip():
                continue
            return tok.start
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    return None


def parse_generation(
    raw_text: str,
) -> tuple[list[CodeStatement], bool]:
    """
    Parse one whole generation (a fresh plan, or a pause-triggered
    continuation) into a flat slot sequence and whether a further
    continuation round is needed.

    Pause-splitting happens before any AST parsing -- ``# PAUSE`` is a
    comment, and ``ast.parse`` strips comments, so a marker's position
    can't be recovered from a parsed tree. Located via ``_find_pause_marker``
    (``tokenize``-based, not a raw-text regex scan -- a legal reasoning-note
    string containing the text "# PAUSE" is never misread as the real
    sentinel). Only the FIRST marker matters: a generation has at most one
    meaningful pause, since reaching one always terminates it (mirroring
    how a ``return`` already terminates it) -- everything at or after it is
    discarded, only ``before`` (everything strictly preceding it) is ever
    parsed.

    ``before`` is parsed and dispatched statement-by-statement exactly as
    always, with two special any-position cases: a bare string-literal
    statement (a reasoning note -- inert, never stored or dispatched, legal
    anywhere, not just first; genuinely valid Python this grammar has no
    other use for, so there is nothing to validate beyond "it's a string"),
    and an ``ast.If`` node. The latter is a defensive backstop, not a taught
    convention -- the prompt tells the model never to write one -- so a
    model that does anyway is handled by silently truncating there (exactly
    like a ``return``) rather than failing the whole generation, UNLESS
    nothing real has been produced yet (``flat_slots`` still empty), in
    which case there is no confident partial work to fall back to and this
    is treated as a genuine structural error instead, feeding regen-repair.
    The identical "nothing real yet" check applies to an explicit pause
    marker found with an empty ``before`` -- both represent the same waste
    (a whole planning round spent for zero progress).

    A ``return`` also terminates immediately (whatever follows it in
    ``before``, if anything, is never even parsed) -- for the same reason
    a later real call must never land in the same batch as the return and
    execute anyway, and a second `return` must never silently overwrite the
    first. If a pause marker was ALSO found anywhere in the raw text
    (``len(parts) == 2``), this is a structural error, not silently
    resolved in ``return``'s favor: a generation writing both terminals is
    self-contradictory (observed live -- a model hedging between "return
    this" and "pause to reconsider" in the same breath), and letting
    ``return`` silently win discards the pause with zero signal, risking a
    premature/unverified final answer. Raises, feeding regen-repair so the
    model is told directly to pick exactly one.

    Whatever follows a found marker is never inspected at all -- ``# PAUSE``
    is a bare, complete sentinel; no trailing note is expected, taught, or
    parsed. Whatever a model writes past the marker is discarded without
    complaint, same as any other post-terminal content.

    Returns ``(flat_slots, continue_planning)``. Raises
    ``BlackboardParseError`` on any structural failure (propagated from
    ``parse_statement_to_slots``, a genuine ``ast.parse`` syntax error in
    ``before``, or one of the two "nothing real yet" cases above).
    """
    text = _strip_code_fence(raw_text)
    marker = _find_pause_marker(text)
    if marker is None:
        before = text
    else:
        row, col = marker
        lines = text.splitlines(keepends=True)
        before = "".join(lines[: row - 1]) + lines[row - 1][:col]
    marker_found = marker is not None

    flat_slots: list[CodeStatement] = []
    hoist_index = 0

    if before.strip():
        try:
            tree = ast.parse(before, mode="exec")
        except SyntaxError as e:
            raise BlackboardParseError(str(e)) from e

        for node in tree.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue

            if isinstance(node, ast.If):
                if not flat_slots:
                    raise BlackboardParseError(
                        "a plan cannot open with a conditional statement "
                        "and no real work done yet; compute or check "
                        "whatever the condition depends on first, as a "
                        "real statement."
                    )
                return flat_slots, True

            stmt_source = ast.unparse(node)
            slots = parse_statement_to_slots(stmt_source, start_index=hoist_index)
            flat_slots.extend(slots)
            hoist_index += len(slots)

            if slots and slots[-1].tool == RETURN_ALIAS:
                if marker_found:
                    raise BlackboardParseError(
                        "a generation cannot contain both a return statement "
                        "and a # PAUSE marker -- pick exactly one way to end: "
                        "return a final value, or # PAUSE (with no return) to "
                        "continue next round."
                    )
                return flat_slots, False

    if not marker_found:
        # No pause marker anywhere -- completes normally (or falls off the
        # end with an inferred `None` result if no `return` ran).
        return flat_slots, False

    if not flat_slots:
        raise BlackboardParseError(
            "a pause cannot appear before any real work has been done; "
            "write at least one real statement first, then pause only if "
            "what follows still depends on something not yet known."
        )

    return flat_slots, True


def rewrite_builtin_calls(slots: list[CodeStatement]) -> list[str]:
    """
    Rewrite eligible builtin-call slots in place to dispatch through the
    ``PY_BUILTIN_ALIAS`` sentinel, mutating ``slot.tool``/``.args`` directly.
    Run once, over the full flat slot list, between ``parse_generation()``
    and ``validate_references()``.

    A slot is eligible when its ``tool`` isn't already a sentinel and names
    a real Python builtin. A registered tool can no longer share a name
    with a real, non-excluded builtin at all (enforced at registration time
    by ``ScriptAgent._validate_tool_alias``), so there is no precedence
    rule to apply here -- a name reaching this function is either a
    registered tool (never a builtin) or not, mutually exclusive by
    construction. An eligible slot gets its builtin name spliced in as a
    new leading positional arg and its ``tool`` replaced with
    ``PY_BUILTIN_ALIAS``.

    A name that resolves to a real but excluded builtin is left unrewritten
    and reported as its own issue, distinct from
    ``validate_references``'s generic "unregistered tool" message that an
    unrecognized name (a real typo/hallucination) still falls through to.

    Returns the list of excluded-builtin issues found (``[]`` if none).
    """
    issues: list[str] = []
    for slot in slots:
        if slot.tool in (RHS_ASSIGN_ALIAS, RETURN_ALIAS):
            continue
        if not hasattr(builtins, slot.tool):
            continue

        label = slot.identifier if slot.identifier is not None else "(unassigned)"
        if slot.tool in EXCLUDED_PY_BUILTINS:
            issues.append(
                f"statement producing {label} calls python builtin "
                f"{slot.tool!r}, which is excluded and cannot be used."
            )
            continue

        slot.args = (slot.tool, *slot.args)
        slot.tool = PY_BUILTIN_ALIAS

    return issues


def is_dispatched_slot(slot: CodeStatement) -> bool:
    """
    True iff ``slot`` represents a real dispatched call (a registered tool
    or an approved Python builtin) rather than an ``rhs_assign``/``return``
    sentinel, which are never dispatched at all. Shared by
    ``compile_batches`` (concurrency-batch accounting) and
    ``validate_references`` (tool-call-budget accounting) -- both use the
    identical predicate, since builtins count toward the budget the same
    as registered tools (no per-category exemption).
    """
    return slot.tool not in (RHS_ASSIGN_ALIAS, RETURN_ALIAS)


def validate_references(
    slots: list[CodeStatement],
    known_tools: frozenset[str],
    known_constants: frozenset[str],
    known_history: frozenset[str],
    tool_calls_limit: Optional[int],
) -> list[str]:
    """
    Walk ``slots`` in order, tracking every previously-bound identifier;
    also check the whole plan's dispatched-call count against a budget.

    For each slot: its own ``tool`` must be a registered tool id unless
    it's the ``rhs_assign``/``return``/``py_builtin``/``attr_call``
    sentinel, or a real but excluded builtin name (``rewrite_builtin_calls``
    already reported that case with a specific message -- this generic
    check must not double-report it); every still-unresolved
    (``ast.expr``-typed) dependency in its ``args`` must already be bound
    by an earlier slot in this same walk, a registered tool id, a
    registered constant, or a name in ``known_history`` (a prior
    invocation's ``task_result_N`` result -- checked identically to a
    registered constant: always externally known, never introduced
    mid-walk). An assignment target (``slot.identifier`` not ``None``) that
    names a registered constant is also rejected here -- constants are
    reserved, read-only bindings, the same protection ``task_result_*``/
    ``_SUB_*`` prefixes already get via ``parse_statement_to_slots``'s own
    LHS check (that check is prefix-based and registry-free by design; this
    one needs ``known_constants``, so it lives here instead). Comprehensive,
    not fail-fast: every unresolvable reference across the whole sequence is
    collected and returned, never just the first (cheaper than discovering
    one issue per regeneration round when a real tool call could have spent
    budget in between).

    Separately, independent of the per-slot walk: if ``tool_calls_limit``
    is not ``None`` and the count of dispatched slots (registered tool
    calls and approved-builtin calls combined, via ``is_dispatched_slot``;
    ``rhs_assign``/``return`` excluded, hoisted calls included) exceeds it,
    that's also collected as an issue -- one regen-repair round can report
    both a bad reference and an excess call count together. Tools and
    builtins are counted identically; there is no separate budget or
    exemption for either category.

    Returns ``[]`` if every reference resolves and the plan is within
    budget.
    """
    bound: set[str] = set()
    issues: list[str] = []

    for slot in slots:
        label = slot.identifier if slot.identifier is not None else "(unassigned)"

        if (
            slot.tool not in (RHS_ASSIGN_ALIAS, RETURN_ALIAS, PY_BUILTIN_ALIAS, ATTR_CALL_ALIAS)
            and slot.tool not in known_tools
            and slot.tool not in EXCLUDED_PY_BUILTINS
        ):
            issues.append(f"statement producing {label} calls unregistered tool {slot.tool!r}.")

        for name in (*extract_identifiers(slot.args), *extract_identifiers(slot.kwargs)):
            if (
                name not in bound
                and name not in known_tools
                and name not in known_constants
                and name not in known_history
            ):
                issues.append(
                    f"statement producing {label} references undefined name {name!r}."
                )

        if slot.identifier is not None:
            if slot.identifier in known_constants:
                issues.append(
                    f"statement assigns to {slot.identifier!r}, a registered "
                    "constant's name; constants are read-only and cannot be "
                    "reassigned."
                )
            bound.add(slot.identifier)

    real_call_count = sum(1 for slot in slots if is_dispatched_slot(slot))
    if tool_calls_limit is not None and real_call_count > tool_calls_limit:
        issues.append(
            f"the plan calls {real_call_count} tool(s)/builtin(s), "
            f"exceeding the configured limit of {tool_calls_limit}."
        )

    return issues


def compile_batches(
    slots: list[CodeStatement],
    max_concurrency: Optional[int] = None,
    start_batch_index: int = 0,
) -> list[list[CodeStatement]]:
    """
    Group ``slots`` into dependency batches for concurrent execution, and
    stamp each slot's ``.batch_index`` with the batch it landed in.

    A slot joins the currently-open batch only if none of its dependencies
    were bound by a slot already sitting in that same open batch (i.e. every
    dependency is satisfiable from an earlier, already-closed batch, a
    registered tool, or a registered constant -- reference validity itself
    is assumed already checked by ``validate_references``). Otherwise the
    open batch closes first and this slot starts a new one.

    Additionally, when about to add a *dispatched* slot (``is_dispatched_slot``
    -- the same predicate ``validate_references`` now uses for its own
    budget accounting) to a batch that already holds ``max_concurrency``
    dispatched slots, the batch closes first -- a purely additive
    concurrency cap, never replacing the dependency-conflict closure rule
    above. ``max_concurrency=None`` means no cap (today's greedy default).

    A ``RETURN_ALIAS`` slot is never grouped with anything else -- it always
    closes the current batch, lands alone in a batch of its own, then closes
    that batch too. This guarantees ``_apply_batch_results`` can never see a
    return slot sharing a batch with an unrelated failing call, which would
    otherwise let that failure suppress an already-resolved return.

    Every slot in a batch is stamped with the same ``batch_index`` --
    ``start_batch_index`` plus that batch's own 0-based position among the
    batches this call produces -- the moment the batch closes. Lets a
    caller running multiple generation rounds in one invoke
    (``ScriptAgentTask.batch_counter``) keep indices globally unique across
    rounds by passing the running total in as ``start_batch_index``.
    """
    batches: list[list[CodeStatement]] = []
    current_batch: list[CodeStatement] = []
    current_batch_identifiers: set[str] = set()
    current_batch_dispatched = 0

    def close_current() -> None:
        nonlocal current_batch, current_batch_identifiers, current_batch_dispatched
        if not current_batch:
            return
        index = start_batch_index + len(batches)
        for slot in current_batch:
            slot.batch_index = index
        batches.append(current_batch)
        current_batch = []
        current_batch_identifiers = set()
        current_batch_dispatched = 0

    for slot in slots:
        if slot.tool == RETURN_ALIAS:
            # A return is never grouped with anything else -- closing
            # before AND after guarantees it lands alone in its own batch,
            # so an unrelated failure elsewhere can never suppress it (a
            # partial-batch failure can only ever apply to slots that were
            # actually batched alongside the failure).
            close_current()
            current_batch.append(slot)
            close_current()
            continue

        deps = (*extract_identifiers(slot.args), *extract_identifiers(slot.kwargs))
        if any(name in current_batch_identifiers for name in deps):
            close_current()

        is_dispatched = is_dispatched_slot(slot)
        if (
            is_dispatched
            and max_concurrency is not None
            and current_batch_dispatched >= max_concurrency
        ):
            close_current()

        current_batch.append(slot)
        if slot.identifier is not None:
            current_batch_identifiers.add(slot.identifier)
        if is_dispatched:
            current_batch_dispatched += 1

    close_current()
    return batches


def render_completed_as_python(
    completed: list[CodeStatement],
    show_batches: bool = False,
) -> str:
    """
    Reconstruct a Python-source-formatted snapshot of already-completed
    slots: one line per slot, in commit order, mirroring the statement
    that originally produced it.

    ``show_batches`` (default ``False``) controls whether output is
    grouped under a ``# Batch N:`` header per concurrently-dispatched
    batch. Model-facing callers (``ScriptAgent``'s own continuation-message
    building) must leave this ``False`` -- ``# Batch N:`` headers appearing
    in text shown to the model were found, empirically, to get echoed and
    fabricated back into later generations. The grouped form remains
    available, opt-in, for standalone human inspection
    (``ScriptAgentRecord.render_as_code``), where there is no such risk.
    Consecutive slots sharing the same ``.batch_index`` are already
    contiguous in ``completed`` (a batch drains fully before the next one
    starts), so grouping only needs to detect index changes, not sort.

    No per-slot result preview or ``await`` echo -- both judged noise
    cluttering the reconstructed code itself. A caller needing actual
    resolved values (a continuation round genuinely needs this -- see
    ``render_cache_snapshot``) renders them as a separate block instead of
    interleaving them per line. A ``return`` slot is never expected here
    (it always ends the invoke, so no continuation is ever rendered
    afterward) but is handled defensively rather than crashing. Returns
    the joined lines, or ``""`` for an empty ``completed``.
    """

    def render_value(value: Any) -> str:
        return ast.unparse(value) if isinstance(value, ast.expr) else repr(value)

    lines: list[str] = []
    current_index: Optional[int] = None
    for slot in completed:
        if show_batches and slot.batch_index != current_index:
            if current_index is not None:
                lines.append("")
            lines.append(f"# Batch {slot.batch_index}:")
            current_index = slot.batch_index

        if slot.tool == RETURN_ALIAS:
            lines.append(f"return {render_value(slot.kwargs['val'])}")
            continue

        prefix = f"{slot.identifier} = " if slot.identifier is not None else ""
        if slot.tool == RHS_ASSIGN_ALIAS:
            lines.append(f"{prefix}{render_value(slot.kwargs['val'])}")
            continue

        # A py_builtin slot renders back as the original natural call
        # syntax (`len(x)`), never the internal rewritten form
        # (`py_builtin('len', x)`) -- unsplice the builtin name before
        # falling into the same generic rendering as any real tool call. An
        # attr_call slot gets the same treatment: `obj.method(args)`, not
        # the internal (obj, "method", *args) shape.
        if slot.tool == PY_BUILTIN_ALIAS:
            call_name, call_args = slot.args[0], slot.args[1:]
        elif slot.tool == ATTR_CALL_ALIAS:
            call_name, call_args = f"{render_value(slot.args[0])}.{slot.args[1]}", slot.args[2:]
        else:
            call_name, call_args = slot.tool, slot.args

        positional_tokens = [
            f"*{render_value(entry.value)}" if isinstance(entry, ast.Starred) else render_value(entry)
            for entry in call_args
        ]
        keyword_tokens = [
            f"**{render_value(value)}" if name == KWARGS_UNPACK_KEY else f"{name}={render_value(value)}"
            for name, value in slot.kwargs.items()
        ]
        args_source = ", ".join(positional_tokens + keyword_tokens)
        lines.append(f"{prefix}{call_name}({args_source})")

    return "\n".join(lines)


def render_cache_snapshot(
    completed: list[CodeStatement],
    cache: dict[str, Any],
    preview_limit: Optional[int],
) -> str:
    """
    Render the current value of every identifier bound by ``completed``
    this round, as its own fenced block, separate from the reconstructed
    code (``render_completed_as_python``) -- keeps per-statement lines free
    of inline value noise while still giving a continuation round real
    visibility into what a prior dispatched call actually returned (the
    one thing a bare ``name = tool(...)`` statement can never reveal on
    its own; a reactive, content-driven pause decision -- e.g. reading a
    reviewer's actual verdict -- depends on this).

    Identifiers are taken from ``completed`` in first-occurrence order,
    each looked up fresh in ``cache`` (so a reassigned name shows its
    latest value, not its first -- every identifier bound by a slot in
    ``completed`` is guaranteed present in ``cache``, populated the moment
    that slot lands there). Each line is ``name: type = value``
    (``type(value).__name__``, a runtime snapshot, not a declared/static
    type), under a leading ``"Cached values:"`` label, wrapped in a
    triple-backtick fence -- visually distinct from the reconstructed-code
    block above it, and unambiguous to a model that this section is data,
    not code to continue writing. ``preview_limit`` truncates each
    rendered value the same way ``Agent.render_turn`` truncates a rendered
    response (``None`` means no truncation) -- guards against dumping an
    excessively long value into every subsequent generation call. Returns
    ``""`` when ``completed`` binds no identifiers at all (nothing to
    show) -- the caller omits the block entirely rather than rendering an
    empty fence.
    """
    ordered: dict[str, Any] = {}
    for slot in completed:
        if slot.identifier is not None:
            ordered[slot.identifier] = cache[slot.identifier]

    if not ordered:
        return ""

    def preview(value: Any) -> str:
        text = repr(value)
        if preview_limit is not None and len(text) > preview_limit:
            text = text[:preview_limit] + "..."
        return text

    lines = [
        f"{name}: {type(value).__name__} = {preview(value)}"
        for name, value in ordered.items()
    ]
    return "```\nCached values:\n" + "\n".join(lines) + "\n```"
