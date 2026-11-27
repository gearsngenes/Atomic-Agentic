"""
The statement core shared by ``PlanActAgent`` and ``ReActAgent``: compile the
model's text into ``Statement``s, validate them, run one statement, cost
them, order them into concurrent waves, and render them for the model.

A statement is the unit of execution and concurrency. Its steps (hoisted calls
first, its own step last) run in order against a snapshot of the context plus
the statement's own scratch results, so a failure is local to the statement and
``run_statement`` never touches a task or the context: the caller commits the
named results. Parsing is ``utils/script.py``, which builds the
``Statement``/``Step`` objects directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Callable, Optional

from ..constants.agents import ATTR_CALL_ALIAS
from ..invokables.atomic_invokable import AtomicInvokable
from ..models.agents.blackboard_models import Statement
from .agents import (
    check_call_bindings,
    extract_identifiers,
    resolve_statement_args,
)
from .script import (
    dispatch_inputs,
    parse_first_statement,
    parse_generation,
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
def compile_statements(
    raw_text: str, *, limit: Optional[int], start_ordinal: int = 0,
) -> tuple[list[Statement], int]:
    """
    Parse a generation into ``Statement``s. ``limit=1`` keeps only the first
    statement (the rest are dropped unparsed); ``limit=None`` keeps them all.
    Returns ``(statements, dropped)`` where ``dropped`` counts the ignored
    later statements. Raises ``BlackboardParseError`` for every rejection.
    """
    if limit is None:
        return parse_generation(raw_text, start_ordinal=start_ordinal), 0
    if limit == 1:
        statement, dropped = parse_first_statement(raw_text, ordinal=start_ordinal)
        return [statement], dropped
    raise ValueError(f"compile_statements: limit must be 1 or None; got {limit!r}.")


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
    steps, in order), then, when ``remaining_budget`` is not ``None``, that
    the statements' planned cost fits it. An agent that enforces the action
    cost itself passes ``None``. Returns the issue list, ``[]`` when clean.
    """
    issues = (
        validate_references(statements, frozenset(toolbox), context_names, readonly_names)
        + check_call_bindings(statements, toolbox)
    )
    if remaining_budget is not None:
        cost = sum(planned_cost(statement) for statement in statements)
        if cost > remaining_budget:
            issues.append(
                f"the plan costs {cost} action(s), exceeding the remaining "
                f"budget of {remaining_budget}."
            )
    return issues


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
        try:
            positional, keyword = resolve_statement_args(step, {**namespace, **scratch})
            if step.kind == "eval":
                output = positional[0]
            else:
                tool = resolve_tool(step.tool)
                output = tool.invoke(dispatch_inputs(step, tool, positional, keyword))
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
        try:
            positional, keyword = resolve_statement_args(step, {**namespace, **scratch})
            if step.kind == "eval":
                output = positional[0]
            else:
                tool = resolve_tool(step.tool)
                output = await tool.async_invoke(dispatch_inputs(step, tool, positional, keyword))
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
