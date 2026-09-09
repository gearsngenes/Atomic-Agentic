from __future__ import annotations

import ast
import re
from typing import Any, Optional

from ..constants.agents import (
    DEFAULT_CONTINUATION_NOTE,
    HOISTED_NAME_PREFIX,
    KWARGS_UNPACK_KEY,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
    TASK_RESULT_PREFIX,
)
from ..exceptions import BlackboardParseError
from ..models.agents.blackboard_models import CodeStatement
from .agents import extract_identifiers

__all__ = [
    "parse_statement_to_slots",
    "resolve_slot_args",
    "parse_generation",
    "validate_references",
    "compile_batches",
    "render_completed_as_python",
]

# Matches a `#`-comment line whose content is (case-insensitively) the word
# PAUSE -- line-anchored so a tool argument that happens to contain the
# text is never misread as a real marker.
_PAUSE_PATTERN = re.compile(r"^\s*#\s*PAUSE\b", re.IGNORECASE | re.MULTILINE)

# Matches an optional single markdown code fence wrapping the *entire*
# generation -- any (or no) language tag on the opening fence line
# (```python, ```py, ```text, a bare ```, ...), not just ```python.
_CODE_FENCE_PATTERN = re.compile(r"^\s*```[^\n]*\n(.*?)\n?```\s*$", re.DOTALL)


def _strip_code_fence(raw_text: str) -> str:
    """
    Strip a single markdown code fence wrapping the whole generation, if
    present -- defensive against a model wrapping otherwise-valid output in
    a code fence despite being told not to. Generic to any language tag
    (or none) on the opening fence line. Only strips a fence wrapping the
    *entire* text; a fence appearing only partway through is left alone
    (``ast.parse`` will reject that on its own terms, as a real structural
    problem).
    """
    match = _CODE_FENCE_PATTERN.match(raw_text)
    return match.group(1) if match else raw_text


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
    concrete value from ``resolved`` (identifier -> value), leaving
    already-folded raw literals untouched. Purely transient -- never
    persisted back onto a ``CodeStatement``.

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
    to ``hoisted``), then eagerly constant-folds each resulting
    dependency-free argument. Shared by both the top-level call (case A)
    and every recursively hoisted call.

    A positional entry that is an ``ast.Starred`` (a ``*expr`` unpack) is
    never eagerly folded, regardless of whether its own inner expr has
    dependencies -- its Starred-ness must survive to resolve time
    (``resolve_slot_args``), and an already-folded plain value has no way
    to carry that tag. A keyword entry whose ``kw.arg is None`` (a
    ``**expr`` unpack) is stored under ``KWARGS_UNPACK_KEY`` -- at most
    one per call; a second one raises immediately.
    """
    args: list[Any] = []
    for arg_node in call_node.args:
        if isinstance(arg_node, ast.Starred):
            inner = _hoist_calls(
                arg_node.value, counter=counter, start_index=start_index, hoisted=hoisted
            )
            args.append(ast.Starred(value=inner, ctx=ast.Load()))
            continue

        processed = _hoist_calls(
            arg_node, counter=counter, start_index=start_index, hoisted=hoisted
        )
        deps = extract_identifiers(processed)
        if not deps:
            try:
                args.append(_evaluate_expr(processed, {}))
            except Exception as e:
                raise BlackboardParseError(
                    "positional argument is a constant expression that "
                    f"failed to evaluate: {e!r}"
                ) from e
        else:
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
                kwargs[key] = _evaluate_expr(processed_value, {})
            except Exception as e:
                raise BlackboardParseError(
                    f"argument {key!r} is a constant expression that failed "
                    f"to evaluate: {e!r}"
                ) from e
        else:
            kwargs[key] = processed_value

    return tuple(args), kwargs


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

    A nested ``ast.Await`` (anywhere ``node`` isn't itself one of the two
    top-level await positions ``parse_statement_to_slots`` already unwraps
    before calling here -- e.g. inside a call's own keyword argument, or a
    bare ``return await f()``) is hoisted exactly like an ordinary nested
    call, except the synthesized slot itself carries ``awaited=True`` --
    the awaited-ness travels with the call to wherever it lands, rather
    than being discarded or left dangling on a now-bare ``Name``. An
    ``await`` wrapping anything other than a call (nested or not) still
    raises, since there is nothing else in this grammar to await.
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

    def _hoist_one_call(call_node: ast.Call, *, awaited: bool) -> ast.Name:
        """Build one hoisted slot for `call_node` (appended to `hoisted`)
        and return a `Name` reference to it. Shared tail for both
        `visit_Call` (awaited=False) and `visit_Await` (awaited=True) --
        the only difference between an ordinary and an awaited hoist."""
        index = counter[0] + start_index
        counter[0] += 1
        hoisted_identifier = f"{HOISTED_NAME_PREFIX}{index}"

        hoisted_positional, hoisted_keyword = _process_call_args(
            call_node, counter=counter, start_index=start_index, hoisted=hoisted
        )
        tool_name = ast.unparse(call_node.func)
        hoisted.append(
            CodeStatement(
                identifier=hoisted_identifier,
                tool=tool_name,
                args=hoisted_positional,
                kwargs=hoisted_keyword,
                awaited=awaited,
            )
        )

        replacement = ast.Name(id=hoisted_identifier, ctx=ast.Load())
        return ast.copy_location(replacement, call_node)

    class _CallHoister(ast.NodeTransformer):
        def visit_Call(self, call_node: ast.Call) -> ast.Name:
            # Only recurse into keyword values, not `func` -- a call's own
            # callable is always a plain dotted-name chain in this grammar,
            # never itself a nested call.
            for kw in call_node.keywords:
                kw.value = self.visit(kw.value)
            return _hoist_one_call(call_node, awaited=False)

        def visit_Await(self, await_node: ast.Await) -> ast.Name:
            if not isinstance(await_node.value, ast.Call):
                raise BlackboardParseError(
                    "`await` must directly wrap a call; got "
                    f"{type(await_node.value).__name__}."
                )
            call_node = await_node.value
            for kw in call_node.keywords:
                kw.value = self.visit(kw.value)
            return _hoist_one_call(call_node, awaited=True)

    return _CallHoister().visit(node)


def parse_statement_to_slots(statement: str, start_index: int = 0) -> list[CodeStatement]:
    """
    Parse one raw generated statement string into an ordered list of
    ``CodeStatement`` objects: any auto-hoisted slots first (in
    discovery/post-order), the statement's own slot last.

    Three top-level statement shapes are accepted: an assignment (case A/B
    below), a bare (unassigned) call -- optionally ``await``-wrapped -- and
    a ``return`` statement. The latter two produce a slot with
    ``identifier=None``: nothing can ever reference either by name, so no
    synthesized name is needed the way hoisting needs one.

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
                val: Any = _evaluate_expr(processed, {})
            except Exception as e:
                raise BlackboardParseError(
                    f"return expression is a constant that failed to evaluate: {e!r}"
                ) from e
        else:
            val = processed
        final_slot = CodeStatement(
            identifier=None, tool=RETURN_ALIAS, kwargs={"val": val}, awaited=False
        )
        return [*hoisted, final_slot]

    # Bare (unassigned) call, optionally `await`-wrapped -- void-style tool
    # calls the model doesn't need a name for, still logged as a real slot
    # for ordering/budget/history.
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Call, ast.Await)):
        bare_rhs = stmt.value
        bare_awaited = False
        if isinstance(bare_rhs, ast.Await):
            bare_awaited = True
            bare_rhs = bare_rhs.value
        if not isinstance(bare_rhs, ast.Call):
            raise BlackboardParseError(
                f"a bare `await` expression must wrap a call; got {type(bare_rhs).__name__}."
            )
        hoisted = []
        counter = [0]
        tool = ast.unparse(bare_rhs.func)
        args, kwargs = _process_call_args(
            bare_rhs, counter=counter, start_index=start_index, hoisted=hoisted
        )
        final_slot = CodeStatement(
            identifier=None, tool=tool, args=args, kwargs=kwargs, awaited=bare_awaited
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
    if identifier.startswith(HOISTED_NAME_PREFIX):
        raise BlackboardParseError(
            f"identifier {identifier!r} uses the reserved hoisted-name prefix "
            f"{HOISTED_NAME_PREFIX!r}."
        )
    if identifier.startswith(TASK_RESULT_PREFIX):
        raise BlackboardParseError(
            f"identifier {identifier!r} uses the reserved task-result-"
            f"addressing prefix {TASK_RESULT_PREFIX!r}; task_result_N names "
            "are fixed, read-only references to prior invocations and "
            "cannot be assigned to."
        )

    # Unwrap an optional top-level `await`.
    rhs = stmt.value
    awaited = False
    if isinstance(rhs, ast.Await):
        awaited = True
        rhs = rhs.value

    hoisted: list[CodeStatement] = []
    counter = [0]

    if isinstance(rhs, ast.Call):
        # Case A: bare top-level call -- the only shape `awaited` stays True for.
        tool = ast.unparse(rhs.func)
        args, kwargs = _process_call_args(
            rhs, counter=counter, start_index=start_index, hoisted=hoisted
        )
    else:
        # Case B: rhs_assign. Also where an `await` wrapping anything other
        # than a bare call lands -- forced False regardless of the unwrap
        # above, since this branch means rhs was never a bare call at all.
        awaited = False
        tool = RHS_ASSIGN_ALIAS
        # Ternary-with-calls rejection lives inside _hoist_calls itself now
        # (the common choke point every call path funnels through) --
        # nothing extra needed here.
        processed = _hoist_calls(rhs, counter=counter, start_index=start_index, hoisted=hoisted)
        deps = extract_identifiers(processed)
        if not deps:
            try:
                val: Any = _evaluate_expr(processed, {})
            except Exception as e:
                raise BlackboardParseError(
                    f"expression is a constant that failed to evaluate: {e!r}"
                ) from e
        else:
            val = processed
        args, kwargs = (), {"val": val}

    final_slot = CodeStatement(
        identifier=identifier, tool=tool, args=args, kwargs=kwargs, awaited=awaited
    )
    return [*hoisted, final_slot]


def parse_generation(
    raw_text: str,
) -> tuple[list[CodeStatement], list[str], bool, Optional[str]]:
    """
    Parse one whole generation (a fresh plan, or a pause-triggered
    continuation) into a flat slot sequence, its annotation blocks, and
    whether/why a further continuation round is needed.

    Pause-splitting happens on raw text, before any AST parsing --
    ``# PAUSE`` is a comment, and ``ast.parse`` strips comments, so a
    marker's position can't be recovered from a parsed tree. Only the FIRST
    marker matters: a generation has at most one meaningful pause, since
    reaching one always terminates it (mirroring how a ``return`` already
    terminates it) -- ``maxsplit=1`` produces at most two pieces,
    ``before``/``after``.

    ``before`` is parsed and dispatched statement-by-statement exactly as
    always, with two special first-statement/any-position cases: an opening
    reasoning block (a bare string-literal statement at position 0), and an
    ``ast.If`` node. The latter is a defensive backstop, not a taught
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
    first.

    ``after`` (present only when a pause marker was found) is read only to
    look for its own leading bare string-literal statement, which becomes
    the continuation note; anything else in ``after`` -- a missing note, a
    syntax error, or genuine further statements -- is discarded without
    complaint, and ``DEFAULT_CONTINUATION_NOTE`` is used in place of a
    missing note.

    Returns ``(flat_slots, annotations, continue_planning,
    continuation_note)``. Raises ``BlackboardParseError`` on any structural
    failure (propagated from ``parse_statement_to_slots``, a genuine
    ``ast.parse`` syntax error in ``before``, or one of the two
    "nothing real yet" cases above).
    """
    text = _strip_code_fence(raw_text)
    parts = _PAUSE_PATTERN.split(text, maxsplit=1)
    before = parts[0]

    flat_slots: list[CodeStatement] = []
    annotations: list[str] = []
    hoist_index = 0

    if before.strip():
        try:
            tree = ast.parse(before, mode="exec")
        except SyntaxError as e:
            raise BlackboardParseError(str(e)) from e

        for j, node in enumerate(tree.body):
            if (
                j == 0
                and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                annotations.append(node.value.value)
                continue

            if isinstance(node, ast.If):
                if not flat_slots:
                    raise BlackboardParseError(
                        "a plan cannot open with a conditional statement "
                        "and no real work done yet; compute or check "
                        "whatever the condition depends on first, as a "
                        "real statement."
                    )
                return flat_slots, annotations, True, None

            stmt_source = ast.unparse(node)
            slots = parse_statement_to_slots(stmt_source, start_index=hoist_index)
            flat_slots.extend(slots)
            hoist_index += len(slots)

            if slots and slots[-1].tool == RETURN_ALIAS:
                return flat_slots, annotations, False, None

    if len(parts) == 1:
        # No pause marker anywhere -- completes normally (or falls off the
        # end with an inferred `None` result if no `return` ran).
        return flat_slots, annotations, False, None

    if not flat_slots:
        raise BlackboardParseError(
            "a pause cannot appear before any real work has been done; "
            "write at least one real statement first, then pause only if "
            "what follows still depends on something not yet known."
        )

    continuation_note = DEFAULT_CONTINUATION_NOTE
    try:
        after_tree = ast.parse(parts[1], mode="exec")
        if (
            after_tree.body
            and isinstance(after_tree.body[0], ast.Expr)
            and isinstance(after_tree.body[0].value, ast.Constant)
            and isinstance(after_tree.body[0].value.value, str)
        ):
            continuation_note = after_tree.body[0].value.value
    except SyntaxError:
        pass  # malformed trailing content -- fall back to the default note

    return flat_slots, annotations, True, continuation_note


def validate_references(
    slots: list[CodeStatement],
    known_tools: frozenset[str],
    known_constants: frozenset[str],
    known_history: frozenset[str],
    tool_calls_limit: Optional[int],
) -> list[str]:
    """
    Walk ``slots`` in order, tracking every previously-bound identifier;
    also check the whole plan's real-tool-call count against a budget.

    For each slot: its own ``tool`` must be a registered tool id unless
    it's the ``rhs_assign``/``return`` sentinel; every still-unresolved
    (``ast.expr``-typed) dependency in its ``args`` must already be bound
    by an earlier slot in this same walk, a registered tool id, a
    registered constant, or a name in ``known_history`` (a prior
    invocation's ``task_result_N`` result -- checked identically to a
    registered constant: always externally known, never introduced
    mid-walk). Comprehensive, not fail-fast: every unresolvable reference
    across the whole sequence is collected and returned, never just the
    first (cheaper than discovering one issue per regeneration round when a
    real tool call could have spent budget in between).

    Separately, independent of the per-slot walk: if ``tool_calls_limit``
    is not ``None`` and the count of real tool-call slots (excluding
    ``rhs_assign``/``return``, hoisted calls included) exceeds it, that's
    also collected as an issue -- one regen-repair round can report both
    a bad reference and an excess tool-call count together.

    Returns ``[]`` if every reference resolves and the plan is within
    budget.
    """
    bound: set[str] = set()
    issues: list[str] = []

    for slot in slots:
        label = slot.identifier if slot.identifier is not None else "(unassigned)"

        if slot.tool not in (RHS_ASSIGN_ALIAS, RETURN_ALIAS) and slot.tool not in known_tools:
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
            bound.add(slot.identifier)

    real_call_count = sum(1 for slot in slots if slot.tool not in (RHS_ASSIGN_ALIAS, RETURN_ALIAS))
    if tool_calls_limit is not None and real_call_count > tool_calls_limit:
        issues.append(
            f"the plan calls {real_call_count} real tool(s), exceeding the "
            f"configured limit of {tool_calls_limit}."
        )

    return issues


def compile_batches(slots: list[CodeStatement]) -> list[list[CodeStatement]]:
    """
    Group ``slots`` into dependency batches for concurrent execution.

    A slot joins the currently-open batch only if none of its dependencies
    were bound by a slot already sitting in that same open batch (i.e. every
    dependency is satisfiable from an earlier, already-closed batch, a
    registered tool, or a registered constant -- reference validity itself
    is assumed already checked by ``validate_references``). Otherwise the
    open batch closes first and this slot starts a new one.

    A batch always closes right after an ``awaited=True`` slot is added --
    a forward barrier, so nothing textually after it can share that batch
    regardless of real dependency edges (this is also why a batch can never
    hold more than one awaited call: the first one already forces closure
    before a second could ever join). There is no longer a pause-driven
    closure case: a pause (or an if-cutoff) always sits at the very end of
    ``slots`` now, since ``parse_generation`` terminates the sequence there
    -- nothing structurally follows it to force a boundary against.
    """
    batches: list[list[CodeStatement]] = []
    current_batch: list[CodeStatement] = []
    current_batch_identifiers: set[str] = set()

    def close_current() -> None:
        nonlocal current_batch, current_batch_identifiers
        if not current_batch:
            return
        batches.append(current_batch)
        current_batch = []
        current_batch_identifiers = set()

    for slot in slots:
        deps = (*extract_identifiers(slot.args), *extract_identifiers(slot.kwargs))
        if any(name in current_batch_identifiers for name in deps):
            close_current()

        current_batch.append(slot)
        if slot.identifier is not None:
            current_batch_identifiers.add(slot.identifier)

        if slot.awaited:
            close_current()

    close_current()
    return batches


def render_completed_as_python(
    completed: list[CodeStatement],
    preview_limit: Optional[int],
) -> str:
    """
    Reconstruct a Python-source-formatted snapshot of already-completed
    slots, for a pause-triggered continuation round's rendered context:
    one line per slot, in commit order, mirroring the statement that
    originally produced it.

    A dispatched real tool call gets a trailing ``# Equals: <preview>``
    comment showing its resolved value -- truncated the same way
    ``Agent.render_turn`` truncates a rendered response (via
    ``preview_limit``; that logic lives on ``Agent``, in ``agents/``, which
    sits above ``utils/`` in this project's layering, so it's replicated
    here rather than imported). A plain ``rhs_assign`` slot needs no such
    comment -- its value is already the literal shown. A ``return`` slot is
    never expected here (it always ends the invoke, so no continuation is
    ever rendered afterward) but is handled defensively rather than
    crashing. Returns the joined lines, or ``""`` for an empty ``completed``.
    """

    def render_value(value: Any) -> str:
        return ast.unparse(value) if isinstance(value, ast.expr) else repr(value)

    def preview(value: Any) -> str:
        text = str(value)
        if preview_limit is not None and len(text) > preview_limit:
            text = text[:preview_limit] + "..."
        return text

    lines: list[str] = []
    for slot in completed:
        if slot.tool == RETURN_ALIAS:
            lines.append(f"return {render_value(slot.kwargs['val'])}")
            continue

        prefix = f"{slot.identifier} = " if slot.identifier is not None else ""
        if slot.tool == RHS_ASSIGN_ALIAS:
            lines.append(f"{prefix}{render_value(slot.kwargs['val'])}")
            continue

        positional_tokens = [
            f"*{render_value(entry.value)}" if isinstance(entry, ast.Starred) else render_value(entry)
            for entry in slot.args
        ]
        keyword_tokens = [
            f"**{render_value(value)}" if name == KWARGS_UNPACK_KEY else f"{name}={render_value(value)}"
            for name, value in slot.kwargs.items()
        ]
        args_source = ", ".join(positional_tokens + keyword_tokens)
        resolved_value = slot.result.result if slot.result is not None else None
        call_source = f"{slot.tool}({args_source})"
        if slot.awaited:
            call_source = f"await {call_source}"
        lines.append(f"{prefix}{call_source}  # Equals: {preview(resolved_value)}")

    return "\n".join(lines)
