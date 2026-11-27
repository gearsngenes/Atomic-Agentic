from __future__ import annotations

import ast
import builtins
import keyword
from collections.abc import Collection, Iterable, Sequence
from typing import Any, Optional

from ..constants.agents import (
    ALLOWED_ATTR_METHODS,
    MUTATING_ATTR_METHODS,
    ATTR_CALL_ALIAS,
    DUNDER_ATTRIBUTE_PATTERN,
    ALLOWED_PY_BUILTINS,
    KWARGS_UNPACK_KEY,
    PY_BUILTIN_ALIAS,
    SUB_NAME_PREFIX,
    TASK_RESULT_PREFIX,
)
from ..exceptions import BlackboardParseError, ToolRegistrationError
from ..invokables.atomic_invokable import AtomicInvokable
from ..models.agents.blackboard_models import Statement, Step
from .agents import (
    evaluate_expr,
    extract_identifiers,
    reject_unsupported_forms,
    strip_code_fence,
)

__all__ = [
    "parse_statement",
    "parse_generation",
    "parse_first_statement",
    "dispatch_inputs",
    "check_script_tool_name",
    "script_context_clashes",
    "validate_references",
]
# resolve_statement_args, check_call_bindings and build_usage_report live in
# utils/agents.py; the statement runner and renderers in utils/statements.py.


def _process_call_args(
    call_node: ast.Call,
    *,
    counter: list[int],
    start_index: int,
    hoisted: list[Step],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Build one call's final ``(args, kwargs)`` pair: hoists any nested call
    found in each positional or keyword value (appending synthesized steps
    to ``hoisted``), then validates each resulting dependency-free argument
    via a dry-run evaluation (raising early on a guaranteed-bad constant
    expression) without folding it -- the original expression form is
    always what gets stored. Shared by both the top-level call (case A)
    and every recursively hoisted call.

    A positional entry that is an ``ast.Starred`` (a ``*expr`` unpack) is
    never eagerly folded, regardless of whether its own inner expr has
    dependencies -- its Starred-ness must survive to resolve time
    (``resolve_statement_args``), and an already-folded plain value has no way
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
                    evaluate_expr(inner, {})
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
                evaluate_expr(processed, {})
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
                evaluate_expr(processed_value, {})
            except Exception as e:
                raise BlackboardParseError(
                    f"argument {key!r} is a constant expression that failed "
                    f"to evaluate: {e!r}"
                ) from e
        kwargs[key] = processed_value

    return tuple(args), kwargs


def _build_call_step(
    call_node: ast.Call,
    *,
    name: Optional[str],
    counter: list[int],
    start_index: int,
    hoisted: list[Step],
) -> Step:
    """
    Build one tool ``Step`` for ``call_node``, branching on whether its
    ``func`` is a plain dotted-name chain (a registered tool/builtin id) or
    an ``ast.Attribute`` (a method call on some object). Shared by the
    top-level bare-unassigned-call and assignment (``name = call(...)``)
    statement shapes in ``parse_statement``, and by
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
    ``x.y.method1().z.method2()``) gets hoisted into its own prior step
    first, recursively, to any depth.
    """
    if isinstance(call_node.func, ast.Attribute):
        method_name = call_node.func.attr
        if DUNDER_ATTRIBUTE_PATTERN.fullmatch(method_name):
            raise BlackboardParseError(
                f"dunder attribute access is not permitted: "
                f"{ast.unparse(call_node.func)!r}."
            )
        if not any(method_name in methods for methods in ALLOWED_ATTR_METHODS.values()):
            raise BlackboardParseError(
                f"method {method_name!r} is not an allowed method call: "
                f"{ast.unparse(call_node.func)!r}."
            )
        obj_expr = _hoist_calls(
            call_node.func.value, counter=counter, start_index=start_index, hoisted=hoisted
        )
        # Same dependency-free dry-run validation every other argument
        # category gets in _process_call_args -- without this, a guaranteed-
        # bad constant object expression (e.g. `(1/0).to_bytes(...)`) isn't
        # caught here at parse time; it instead surfaces later as an opaque
        # resolution failure in prepare().
        obj_deps = extract_identifiers(obj_expr)
        if not obj_deps:
            try:
                evaluate_expr(obj_expr, {})
            except Exception as e:
                raise BlackboardParseError(
                    "attribute/method call's object expression is a constant "
                    f"expression that failed to evaluate: {e!r}"
                ) from e
        positional, keyword = _process_call_args(
            call_node, counter=counter, start_index=start_index, hoisted=hoisted
        )
        return Step(
            kind="tool",
            tool=ATTR_CALL_ALIAS,
            name=name,
            args=(obj_expr, method_name, *positional),
            kwargs=keyword,
        )

    # call_node.func never itself reaches _hoist_calls in this branch (it's
    # used verbatim, via ast.unparse, as the tool id -- never hoisted or
    # rewritten) -- without this explicit call, a lambda/comprehension/
    # ternary-with-call/await/dunder-attribute used directly as a call's
    # own callee (e.g. `(lambda x: x)(1)`) would silently produce a
    # nonsense tool id instead of a clean rejection here.
    reject_unsupported_forms(call_node.func)
    tool_name = ast.unparse(call_node.func)
    positional, keyword = _process_call_args(
        call_node, counter=counter, start_index=start_index, hoisted=hoisted
    )
    # An allowed builtin dispatches through PY_BUILTIN_ALIAS with its name
    # spliced in as the leading argument. A registered tool cannot share a
    # name with an allowed builtin (ToolAgent._validate_tool_name), so there
    # is no precedence rule to apply.
    if tool_name in ALLOWED_PY_BUILTINS:
        return Step(
            kind="tool", tool=PY_BUILTIN_ALIAS, name=name,
            args=(tool_name, *positional), kwargs=keyword,
        )
    return Step(kind="tool", tool=tool_name, name=name, args=positional, kwargs=keyword)


def _reject_await(node: ast.expr) -> None:
    """
    Raise the one dedicated `await`-no-longer-supported parse error. Shared
    by `_hoist_calls`'s rejection scan (an `await` nested anywhere, or used
    as a bare/rhs-assign top-level expression that still routes through
    `_hoist_calls`) and `parse_statement`'s top-level bare-call
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
    hoisted: list[Step],
) -> ast.expr:
    """
    Post-order rewrite: replaces every ``Call`` node found anywhere within
    ``node`` (at any depth -- a ``BinOp`` operand, another call's keyword
    value, an f-string's embedded expression, a container literal element,
    ...) with a ``Name`` reference to a newly synthesized, hoisted
    ``Step``, appended to ``hoisted`` in discovery order.

    ``ast.NodeTransformer.generic_visit`` recurses into a call's own
    children before ``visit_Call`` builds that call's own hoisted step, so
    doubly/triply-nested calls flatten correctly bottom-up.

    Rejects (before any hoisting) every form ``reject_unsupported_forms``
    covers -- a ternary whose either branch contains a ``Call``, an
    ``ast.Await`` anywhere, a comprehension/lambda anywhere, and a dunder
    attribute access anywhere (including inside a to-be-hoisted call's own
    object expression; closes the classic attribute-chaining sandbox-escape
    class, ``().__class__.__bases__[0].__subclasses__()``-style, which the
    ``{"__builtins__": {}}`` eval lockout elsewhere in this module does not
    defend against on its own). A method-call's own method name
    (``call_node.func.attr``) sits outside this scan (it's never itself
    walked as a standalone ``ast.Attribute`` node here) and is checked
    separately, in ``_build_call_step``.
    """
    reject_unsupported_forms(node)

    def _hoist_one_call(call_node: ast.Call) -> ast.Name:
        """Build one hoisted step for `call_node` (appended to `hoisted`,
        via `_build_call_step` -- handles both a plain tool/builtin id and
        an attribute/method call) and return a `Name` reference to it."""
        index = counter[0] + start_index
        counter[0] += 1
        hoisted_identifier = f"{SUB_NAME_PREFIX}{index}"

        hoisted.append(
            _build_call_step(
                call_node,
                name=hoisted_identifier,
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
            # explicitly inside _build_call_step (a fresh _hoist_calls
            # entry), not via this transformer's own traversal.
            for kw in call_node.keywords:
                kw.value = self.visit(kw.value)
            return _hoist_one_call(call_node)

    return _CallHoister().visit(node)


def parse_statement(source: str, *, ordinal: int = 0, start_index: int = 0) -> Statement:
    """
    Parse one raw generated statement string into a ``Statement``: any
    auto-hoisted steps first (in discovery/post-order), the statement's own
    step last.

    Three top-level statement shapes are accepted: an assignment, a bare
    (unassigned) call, and a ``return`` statement. Their ``kind`` is
    ``"assign"``, ``"call"`` and ``"return"``; the latter two have no
    ``target`` (nothing can reference either by name). An own step is a tool
    step when the statement's value is a call, else an eval step carrying the
    expression as ``args[0]``.

    ``start_index`` keeps ``_SUB_N`` names unique across the statements of one
    generation: pass the running count of steps already emitted.

    Every call anywhere in the statement is hoisted and later executed,
    post-order and left to right, including the operands of ``and``/``or``,
    comparisons and chained comparisons: nothing short-circuits.

    Raises ``BlackboardParseError`` for every rejection category (always via
    ``raise ... from e`` where an underlying exception exists) -- never
    silently repaired, never returned as an issues list.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as e:
        raise BlackboardParseError(str(e)) from e

    if len(tree.body) != 1:
        raise BlackboardParseError(f"expected exactly one statement; got {len(tree.body)}.")

    stmt = tree.body[0]
    normalized = ast.unparse(stmt)
    hoisted: list[Step] = []
    counter = [0]

    def build(kind: str, target: Optional[str], own: Step) -> Statement:
        return Statement(
            ordinal=ordinal, source=normalized, kind=kind, target=target,
            steps=(*hoisted, own),
        )

    def eval_step(expr: ast.expr, what: str) -> Step:
        processed = _hoist_calls(expr, counter=counter, start_index=start_index, hoisted=hoisted)
        if not extract_identifiers(processed):
            try:
                evaluate_expr(processed, {})
            except Exception as e:
                raise BlackboardParseError(
                    f"{what} is a constant that failed to evaluate: {e!r}"
                ) from e
        return Step(kind="eval", args=(processed,))

    # Terminal case: a bare `return <expr>` (or bare `return`, treated as
    # `return None`). No Python-grammar obstacle to a module-level Return
    # node here -- the "return outside function" check only fires at
    # compile()-to-bytecode time, which this pipeline never does to a whole
    # statement (only to bare expressions, via evaluate_expr).
    if isinstance(stmt, ast.Return):
        value = stmt.value if stmt.value is not None else ast.Constant(value=None)
        return build("return", None, eval_step(value, "return expression"))

    # A bare top-level `await ...` never reaches `_hoist_calls` (this
    # branch reads `stmt.value` directly, not one of a call's own argument
    # expressions) -- rejected explicitly here via the same shared helper.
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Await):
        _reject_await(stmt.value)

    # Bare (unassigned) call -- void-style tool calls the model doesn't
    # need a name for, still run and logged as a real statement.
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        own = _build_call_step(
            stmt.value, name=None, counter=counter, start_index=start_index, hoisted=hoisted
        )
        return build("call", None, own)

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
    if isinstance(rhs, ast.Call):
        # A call on the right: its own step is that call (plain tool,
        # builtin or method), shared with the bare-call shape above.
        own = _build_call_step(
            rhs, name=None, counter=counter, start_index=start_index, hoisted=hoisted
        )
        return build("assign", identifier, own)

    # Any other expression: hoist the calls inside it (rejections, including
    # an `await`, a ternary with a call and a comprehension, live in
    # _hoist_calls) and evaluate what is left in place.
    return build("assign", identifier, eval_step(rhs, "expression"))


def _compile_gate(nodes: list[ast.stmt]) -> None:
    """Compile ``nodes`` inside a function body, raising ``BlackboardParseError``
    on what the statement-level checks cannot catch (a duplicate keyword
    argument, assignment to ``__debug__``)."""
    shell = ast.parse("def _script():\n    pass")
    shell.body[0].body = nodes
    try:
        compile(shell, "<generation>", "exec")
    except SyntaxError as e:
        raise BlackboardParseError(f"line {e.lineno}: {e.msg}") from e


_MSG_NO_BOOLOP = (
    "`and`/`or` are not available: every call in a statement runs, in order, "
    "with no short-circuiting. Use a comparison or `not`, or read the value "
    "first and decide in the next statement."
)
_MSG_NO_TERNARY = (
    "conditional expressions (`a if b else c`) are not available. Read the "
    "value first; the next statement is your decision."
)


def parse_first_statement(
    raw_text: str, *, ordinal: int = 0, start_index: int = 0,
) -> tuple[Statement, int]:
    """
    Parse one generation into the ``Statement`` of its FIRST statement. Raises
    ``BlackboardParseError`` for every rejection.

    Returns ``(statement, dropped)``: ``dropped`` is how many later statements
    were ignored.

    1. ``strip_code_fence``, ``ast.parse``.
    2. Drop leading bare-string statements (inert notes); none left is an
       error.
    3. Take the first statement; every later one is ignored unparsed.
    4. ReAct-only form check: ``and``/``or`` and a ternary are rejected.
    5. ``parse_statement``, then the compile gate.
    """
    text = strip_code_fence(raw_text)
    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError as e:
        raise BlackboardParseError(str(e)) from e

    body = list(tree.body)
    while (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body.pop(0)
    if not body:
        raise BlackboardParseError("no statement found in the reply.")

    first, dropped = body[0], len(body) - 1
    for node in ast.walk(first):
        if isinstance(node, ast.BoolOp):
            raise BlackboardParseError(_MSG_NO_BOOLOP)
        if isinstance(node, ast.IfExp):
            raise BlackboardParseError(_MSG_NO_TERNARY)

    statement = parse_statement(ast.unparse(first), ordinal=ordinal, start_index=start_index)
    _compile_gate([first])
    return statement, dropped


def dispatch_inputs(
    step: Step,
    tool: AtomicInvokable,
    positional: list[Any],
    keyword: dict[str, Any],
) -> dict[str, Any]:
    """Bind a resolved ``(positional, keyword)`` pair into the input dict of
    the tool that runs ``step``. A builtin or method step packs the real
    call's own arguments as one tuple and one dict, so a real argument named
    like the dispatcher's own parameter cannot collide."""
    if step.tool == PY_BUILTIN_ALIAS:
        return tool._args_kwargs_to_dict(positional[0], tuple(positional[1:]), keyword)
    if step.tool == ATTR_CALL_ALIAS:
        return tool._args_kwargs_to_dict(
            positional[0], positional[1], tuple(positional[2:]), keyword
        )
    return tool._args_kwargs_to_dict(*positional, **keyword)


def check_script_tool_name(name: str) -> None:
    """Raise ``ToolRegistrationError`` for a tool name script source cannot
    reach: a parser sentinel, an allowed Python builtin, or a Python keyword."""
    if name in (PY_BUILTIN_ALIAS, ATTR_CALL_ALIAS):
        raise ToolRegistrationError(
            f"tool name {name!r} is reserved for the parser's own "
            "sentinel tool names and cannot be used as a registered "
            "tool name."
        )
    if name in ALLOWED_PY_BUILTINS:
        raise ToolRegistrationError(
            f"tool name {name!r} collides with a real Python builtin "
            "of the same name -- a tool with this name would be "
            "permanently unreachable (the builtin always resolves "
            "first). Build the tool with a different name via to_tool."
        )
    if keyword.iskeyword(name):
        raise ToolRegistrationError(
            f"tool name {name!r} is a Python keyword; a script could "
            "never call it. Build the tool with a different name via "
            "to_tool."
        )


def script_context_clashes(names: Iterable[str], tool_ids: Collection[str]) -> dict[str, str]:
    """Context-variable names a script could not use unambiguously, mapped to
    the reason: a Python keyword, or a registered tool id (a name in call
    position would resolve to the tool)."""
    return {
        n: ("a Python keyword" if keyword.iskeyword(n) else "a registered tool id")
        for n in sorted(names)
        if keyword.iskeyword(n) or n in tool_ids
    }


def parse_generation(raw_text: str, *, start_ordinal: int = 0) -> list[Statement]:
    """
    Parse one whole generation (a fresh plan, or a repair-triggered
    continuation -- both use this same grammar; there is no separate
    "continuation mode") into its ``Statement``s, ordinals from
    ``start_ordinal``.

    The entire generation is parsed statement by statement, with two special
    any-position cases: a bare string-literal statement (a reasoning note --
    inert, never stored or dispatched, legal anywhere), and an ``ast.If``
    node, which always produces an error -- conditionals are permanently
    forbidden in this grammar.

    A ``return`` terminates immediately (whatever follows it is never
    parsed or compiled) -- a later real call must never run alongside the
    return, and a second ``return`` must never silently overwrite the first.

    Per-statement errors are collected, not fail-fast: one
    ``BlackboardParseError`` carries every error, one per line, as
    ``line N: ...`` (line numbers relative to the code after fence removal).
    A genuine ``ast.parse`` syntax error is fatal and reported alone. When no
    statement error was found, the consumed statements are compiled inside a
    function body as a final gate, catching what the statement-level checks
    cannot (a duplicate keyword argument, assignment to ``__debug__``);
    skipping the gate otherwise keeps a statement-level message from being
    shadowed by the compiler's wording.

    Falling off the end of the generation with neither a ``return`` nor an
    error simply means the plan is done as written.
    """
    text = strip_code_fence(raw_text)

    statements: list[Statement] = []
    hoist_index = 0

    if not text.strip():
        return statements

    try:
        tree = ast.parse(text, mode="exec")
    except SyntaxError as e:
        raise BlackboardParseError(str(e)) from e

    errors: list[str] = []
    consumed: list[ast.stmt] = []

    for node in tree.body:
        consumed.append(node)

        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue

        if isinstance(node, ast.If):
            errors.append(
                f"line {node.lineno}: conditional statements are not permitted "
                "in this grammar -- compute both branches' own work directly, "
                "or restructure without a conditional."
            )
            continue

        try:
            statement = parse_statement(
                ast.unparse(node),
                ordinal=start_ordinal + len(statements),
                start_index=hoist_index,
            )
        except BlackboardParseError as e:
            errors.append(f"line {node.lineno}: {e}")
            continue

        statements.append(statement)
        hoist_index += len(statement.steps)

        if statement.kind == "return":
            break

    if not errors:
        try:
            _compile_gate(consumed)
        except BlackboardParseError as e:
            errors.append(str(e))

    if errors:
        raise BlackboardParseError("\n".join(errors))

    return statements


def validate_references(
    statements: Sequence[Statement],
    known_tools: frozenset[str],
    context_names: frozenset[str],
    readonly_names: frozenset[str],
) -> list[str]:
    """
    Walk ``statements`` step by step, tracking every previously-bound name
    (a hoisted step binds its ``_SUB_n`` name, an own step the statement's
    target).

    For each tool step: its ``tool`` must be a registered tool id unless it is
    the builtin or method sentinel (a real but disallowed builtin name gets
    its own specific message instead of the generic "unregistered tool" one);
    every dependency in its ``args``/``kwargs`` must already be bound by an
    earlier step in this same walk or be a name in ``context_names`` (every
    variable in ``task.context``: conversation, runtime, input-bound,
    earlier-round bindings, ``task_result_N``), and must not be a registered
    tool id. An assignment target that names a read-only context variable
    (``readonly_names``) or a registered tool id is rejected; a writable
    context name is allowed (it updates the variable). A mutating method call
    (``MUTATING_ATTR_METHODS``) on a read-only variable, or on a name assigned
    from one (directly or via ``get``/``pop``/``setdefault``), is also
    rejected. Comprehensive, not fail-fast: every unresolvable reference
    across the whole sequence is collected and returned, never just the
    first. The plan's action budget is checked separately, by
    ``utils.statements.check_statements``.

    Returns ``[]`` if every reference resolves.
    """
    bound: set[str] = set()
    issues: list[str] = []
    # Names a mutating method call must not touch: the read-only context
    # variables plus any name that aliases one (see the end of the loop).
    protected: set[str] = set(readonly_names)

    for statement in statements:
        for step in statement.steps:
            bound_name = step.name if step.name is not None else statement.target
            label = bound_name if bound_name is not None else "(unassigned)"

            if (
                step.kind == "tool"
                and step.tool not in (PY_BUILTIN_ALIAS, ATTR_CALL_ALIAS)
                and step.tool not in known_tools
            ):
                if hasattr(builtins, step.tool):
                    issues.append(
                        f"statement producing {label} calls python builtin "
                        f"{step.tool!r}, which is not an allowed builtin."
                    )
                else:
                    issues.append(
                        f"statement producing {label} calls unregistered tool {step.tool!r}."
                    )

            for ref in (*extract_identifiers(step.args), *extract_identifiers(step.kwargs)):
                if ref in known_tools:
                    # A registered tool id is never a resolvable value -- only
                    # reachable here since a method step made a bare tool name
                    # into a legal object expression (`sometool.attr()`).
                    issues.append(
                        f"statement producing {label} references {ref!r}, a "
                        "registered tool id, as a value -- a tool cannot be used "
                        "as an argument or as the object of an attribute/method "
                        "call; call it directly instead."
                    )
                elif ref not in bound and ref not in context_names:
                    issues.append(
                        f"statement producing {label} references undefined name {ref!r}."
                    )

            if step.tool == ATTR_CALL_ALIAS and step.args[1] in MUTATING_ATTR_METHODS:
                for ref in sorted(protected & set(extract_identifiers(step.args[0]))):
                    issues.append(
                        f"statement producing {label} calls mutating method "
                        f"{step.args[1]!r} on {ref!r}, a read-only context variable "
                        "(or a name bound to one); copy it first, e.g. list(x)."
                    )

            if bound_name is not None:
                if bound_name in readonly_names:
                    issues.append(
                        f"statement assigns to {bound_name!r}, a read-only context variable."
                    )
                elif bound_name in known_tools:
                    issues.append(
                        f"statement assigns to {bound_name!r}, a registered tool id."
                    )
                # An assignment, or a get/pop/setdefault, from a protected name
                # may hand back the very same object, so the new name is
                # protected too. Methods that return a new object (copy, split,
                # keys, ...) and builtin calls are not propagated.
                if step.kind == "eval":
                    source = step.args[0]
                elif step.tool == ATTR_CALL_ALIAS and step.args[1] in ("get", "pop", "setdefault"):
                    source = step.args[0]
                else:
                    source = None
                if source is not None and protected & set(extract_identifiers(source)):
                    protected.add(bound_name)
                elif bound_name not in readonly_names:
                    protected.discard(bound_name)
                bound.add(bound_name)

    return issues
