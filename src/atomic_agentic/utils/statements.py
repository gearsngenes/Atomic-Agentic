"""
The statement core shared by ``PlanActAgent`` and ``ReActAgent``: compile the
model's text into ``Statement``s, validate them, run one statement, cost
them, order them into concurrent waves, and render them for the model.

A statement is the unit of execution and concurrency. Its steps (hoisted calls
first, its own step last) run in order against a snapshot of the context plus
the statement's own scratch results, so a failure is local to the statement and
``run_statement`` never touches a task or the context: the caller commits the
named results. Parsing is the existing ``utils/script.py`` parser (no fork); the
slots it builds are regrouped into ``Statement``/``Step`` here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Callable, Optional

from ..constants.agents import (
    ATTR_CALL_ALIAS,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
    SUB_NAME_PREFIX,
)
from ..invokables.atomic_invokable import AtomicInvokable
from ..models.agents.blackboard_models import Statement, Step, ToolStatement
from .agents import (
    check_call_bindings,
    extract_identifiers,
    resolve_statement_args,
)
from .script import (
    dispatch_inputs,
    parse_first_statement,
    parse_generation,
    rewrite_builtin_calls,
    validate_references,
)

__all__ = [
    "compile_statements",
    "check_statements",
    "run_statement",
    "async_run_statement",
    "planned_cost",
    "actions_used",
    "plan_waves",
    "render_statement_failure",
    "render_statements",
]


# --------------------------------------------------------------------------- #
# Compile
# --------------------------------------------------------------------------- #
def _is_hoisted(slot: ToolStatement) -> bool:
    return slot.identifier is not None and slot.identifier.startswith(SUB_NAME_PREFIX)


def _step_from_slot(slot: ToolStatement) -> Step:
    return Step(
        text=replace(slot, identifier=None).to_code(),
        kind="eval" if slot.tool in (RHS_ASSIGN_ALIAS, RETURN_ALIAS) else "tool",
        tool=slot.tool,
        name=slot.identifier if _is_hoisted(slot) else None,
        args=slot.args,
        kwargs=dict(slot.kwargs),
    )


def compile_statements(
    raw_text: str, *, limit: Optional[int], start_ordinal: int = 0,
) -> tuple[list[Statement], int]:
    """
    Parse a generation into ``Statement``s. ``limit=1`` keeps only the first
    statement (the rest are dropped unparsed); ``limit=None`` keeps them all.
    Returns ``(statements, dropped)`` where ``dropped`` counts the ignored
    later statements. Raises ``BlackboardParseError`` for every rejection.

    1. Parse with ``parse_first_statement`` or ``parse_generation``.
    2. ``rewrite_builtin_calls`` over the slots.
    3. Group the slots: hoisted slots join the next own slot.
    4. Build the steps and statements.
    """
    if limit is None:
        slots, sources = parse_generation(raw_text, with_sources=True)
        dropped = 0
    elif limit == 1:
        slots, source, dropped = parse_first_statement(raw_text)
        sources = [source]
    else:
        raise ValueError(f"compile_statements: limit must be 1 or None; got {limit!r}.")

    rewrite_builtin_calls(slots)

    groups: list[list[ToolStatement]] = []
    pending: list[ToolStatement] = []
    for slot in slots:
        pending.append(slot)
        if not _is_hoisted(slot):
            groups.append(pending)
            pending = []

    statements: list[Statement] = []
    for offset, (group, source) in enumerate(zip(groups, sources)):
        own = group[-1]
        statements.append(Statement(
            ordinal=start_ordinal + offset,
            source=source,
            kind=(
                "return" if own.tool == RETURN_ALIAS
                else "assign" if own.identifier is not None
                else "call"
            ),
            target=own.identifier,
            steps=tuple(_step_from_slot(slot) for slot in group),
        ))
    return statements, dropped


def _slot(step: Step, identifier: Optional[str]) -> ToolStatement:
    """The parser-shaped slot the shared helpers take (goes away when those
    helpers take steps)."""
    return ToolStatement(
        identifier=identifier, tool=step.tool, args=step.args, kwargs=step.kwargs,
    )


def check_statements(
    statements: Sequence[Statement],
    toolbox: Mapping[str, AtomicInvokable],
    context_names: frozenset[str],
    readonly_names: frozenset[str],
    remaining_budget: Optional[int] = None,
) -> list[str]:
    """
    Validate names, tools and argument bindings across ``statements`` (the
    existing ``validate_references`` and ``check_call_bindings`` over the
    steps, in order). ``remaining_budget`` is ``validate_references``' own
    dispatched-call limit; an agent that enforces the action cost itself passes
    ``None``. Returns the issue list, ``[]`` when clean.
    """
    slots = [
        _slot(step, step.name if step.name is not None else statement.target)
        for statement in statements
        for step in statement.steps
    ]
    return (
        validate_references(
            slots, frozenset(toolbox), context_names, readonly_names, remaining_budget,
        )
        + check_call_bindings(slots, toolbox)
    )


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def run_statement(
    statement: Statement,
    namespace: Mapping[str, Any],
    resolve_tool: Callable[[str], AtomicInvokable],
) -> Statement:
    """
    Run ``statement``'s steps in order and return the executed statement.

    For each step: resolve its arguments against ``namespace`` plus the
    statement's scratch results (the resolved values are not stored); evaluate
    an eval step in place, or dispatch a tool step through
    ``resolve_tool(step.tool)``; record its output (the full result for a tool
    step, the value for an eval step), and (for a hoisted step) the value in
    the scratch. The first failure (a resolution
    or tool error) is recorded on its step; the later steps stay ``not_run``
    and the statement reads as ``failed``. Reads and writes no task or
    context. A ``BaseException`` propagates.
    """
    scratch: dict[str, Any] = {}
    steps = list(statement.steps)
    for index, step in enumerate(steps):
        slot = _slot(step, None)
        try:
            positional, keyword = resolve_statement_args(slot, {**namespace, **scratch})
            if step.kind == "eval":
                output = keyword["val"]
            else:
                tool = resolve_tool(step.tool)
                output = tool.invoke(dispatch_inputs(slot, tool, positional, keyword))
        except Exception as error:
            steps[index] = replace(step, error=error, status="failed")
            break
        steps[index] = ran = replace(step, output=output, status="ran")
        if step.name is not None:
            scratch[step.name] = ran.value
    return replace(statement, steps=tuple(steps))


async def async_run_statement(
    statement: Statement,
    namespace: Mapping[str, Any],
    resolve_tool: Callable[[str], AtomicInvokable],
) -> Statement:
    """Async mirror of ``run_statement``: a tool step awaits ``async_invoke``."""
    scratch: dict[str, Any] = {}
    steps = list(statement.steps)
    for index, step in enumerate(steps):
        slot = _slot(step, None)
        try:
            positional, keyword = resolve_statement_args(slot, {**namespace, **scratch})
            if step.kind == "eval":
                output = keyword["val"]
            else:
                tool = resolve_tool(step.tool)
                output = await tool.async_invoke(dispatch_inputs(slot, tool, positional, keyword))
        except Exception as error:
            steps[index] = replace(step, error=error, status="failed")
            break
        steps[index] = ran = replace(step, output=output, status="ran")
        if step.name is not None:
            scratch[step.name] = ran.value
    return replace(statement, steps=tuple(steps))


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
def _cost(statement: Statement, tool_steps: int) -> int:
    """A statement costs its tool steps, at least one; a ``return`` is exempt
    from the minimum so an agent at its limit can still finish."""
    return tool_steps if statement.kind == "return" else max(1, tool_steps)


def planned_cost(statement: Statement) -> int:
    """Actions ``statement`` spends if every step runs (the budget pre-check)."""
    return _cost(statement, sum(1 for step in statement.steps if step.kind == "tool"))


def actions_used(statements: Iterable[Statement]) -> int:
    """Actions already spent: the same rule over the tool steps that ran or
    failed (a failed call counts); a statement that never started costs 0."""
    return sum(
        _cost(statement, sum(
            1 for step in statement.steps if step.kind == "tool" and step.status != "not_run"
        ))
        for statement in statements
        if statement.status != "not_run"
    )


# --------------------------------------------------------------------------- #
# Concurrency waves
# --------------------------------------------------------------------------- #
def _reads_writes(statement: Statement) -> tuple[set[str], set[str]]:
    """Names the statement reads and writes. Its own ``_SUB_`` scratch names
    are excluded; a method call writes its receiver's names; the target is a
    write."""
    own = {step.name for step in statement.steps if step.name is not None}
    reads: set[str] = set()
    writes: set[str] = set()
    if statement.target is not None:
        writes.add(statement.target)
    for step in statement.steps:
        reads.update(extract_identifiers(step.args))
        reads.update(extract_identifiers(step.kwargs))
        if step.tool == ATTR_CALL_ALIAS:
            writes.update(extract_identifiers(step.args[0]))
    return reads - own, writes - own


def plan_waves(
    statements: Sequence[Statement], concurrency: Optional[int],
) -> list[list[Statement]]:
    """
    Group ``statements`` into waves that may run concurrently, in source order.

    A statement joins the open (last) wave unless the wave is full
    (``concurrency``; ``None`` is unbounded), it or a statement already in the
    wave is a bare call or a ``return`` (those run alone: a bare call is a
    barrier, a return is last), or it conflicts with a statement in the wave:
    its reads against their writes, its writes against their reads, or writes
    against writes.
    """
    waves: list[list[Statement]] = []
    open_rw: list[tuple[set[str], set[str]]] = []
    open_alone = False
    for statement in statements:
        reads, writes = _reads_writes(statement)
        alone = statement.kind != "assign"
        joins = (
            bool(waves)
            and not alone
            and not open_alone
            and (concurrency is None or len(waves[-1]) < concurrency)
            and not any(
                (reads & w) or (writes & r) or (writes & w) for r, w in open_rw
            )
        )
        if joins:
            waves[-1].append(statement)
            open_rw.append((reads, writes))
        else:
            waves.append([statement])
            open_rw = [(reads, writes)]
            open_alone = alone
    return waves


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_statement_failure(statement: Statement) -> str:
    """The failure note for a failed statement: the one-line form for a
    single-step statement, otherwise the source, the failing sub-step and the
    sub-steps that already ran with their values."""
    failed = next(step for step in statement.steps if step.status == "failed")
    if len(statement.steps) == 1:
        return f"{statement.source}  # FAILED: {failed.error!r}"
    note = f"{statement.source}\n# sub-step `{failed.text}` failed: {failed.error!r}"
    ran = [step for step in statement.steps if step.status == "ran"]
    if ran:
        done = ", ".join(f"`{step.text}` -> {step.value!r:.80}" for step in ran)
        note += f"\n# sub-steps that already ran: {done}"
    return note


def render_statements(statements: Iterable[Statement]) -> str:
    """One block per statement, in order: its source when done, the failure
    note when failed, its source plus ``# NOT RUN`` when it never ran."""
    blocks: list[str] = []
    for statement in statements:
        if statement.status == "done":
            blocks.append(statement.source)
        elif statement.status == "failed":
            blocks.append(render_statement_failure(statement))
        else:
            blocks.append(f"{statement.source}\n# NOT RUN")
    return "\n".join(blocks)
