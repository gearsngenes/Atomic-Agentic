"""The statement core: compile, check, run, cost, waves, renderers.

Uses a scripted tool set (plain ``Tool``s over small functions); no model calls.
"""

from __future__ import annotations

import asyncio

import pytest

from atomic_agentic.agents.tools import attr_call_tool, builtin_call_tool
from atomic_agentic.constants.agents import ATTR_CALL_ALIAS, PY_BUILTIN_ALIAS
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.exceptions import BlackboardParseError
from atomic_agentic.tools import Tool
from atomic_agentic.utils.statements import (
    actions_used,
    async_run_statement,
    check_statements,
    compile_statements,
    plan_waves,
    planned_cost,
    render_statement_failure,
    render_statements,
    run_statement,
)

CALLS: list[str] = []


def add(a: int, b: int) -> int:
    return a + b


def mul(a: int, b: int) -> int:
    return a * b


def boom(x: int) -> int:
    raise ValueError("bad")


def log(text: str) -> None:
    CALLS.append(text)


def make_list(x: int) -> list:
    return [x]


TOOLBOX = {
    name: Tool(function=fn, name=name, description=name)
    for name, fn in [("add", add), ("mul", mul), ("boom", boom), ("log", log), ("make_list", make_list)]
}


def resolve_tool(tool_id: str):
    if tool_id == PY_BUILTIN_ALIAS:
        return builtin_call_tool
    if tool_id == ATTR_CALL_ALIAS:
        return attr_call_tool
    return TOOLBOX[tool_id]


def one(text: str):
    statements, _ = compile_statements(text, limit=1)
    return statements[0]


def plan(text: str):
    statements, _ = compile_statements(text, limit=None)
    return statements


# ----------------------------------------------------------------------- #
# compile
# ----------------------------------------------------------------------- #
def test_composed_statement_has_hoisted_steps_then_its_own():
    st = one("x = add(mul(2, 3), 4)")
    assert st.kind == "assign" and st.target == "x"
    assert [s.text for s in st.steps] == ["mul(2, 3)", "add(_SUB_0, 4)"]
    assert st.steps[0].name == "_SUB_0" and st.steps[1].name is None
    assert all(s.kind == "tool" and s.status == "not_run" for s in st.steps)


def test_call_free_assignment_is_one_eval_step():
    st = one("y = a + 1")
    assert [s.kind for s in st.steps] == ["eval"]
    assert st.kind == "assign" and st.target == "y"


def test_kinds_and_targets():
    assert one("log('hi')").kind == "call" and one("log('hi')").target is None
    r = one("return 5")
    assert r.kind == "return" and r.target is None and r.steps[0].kind == "eval"


def test_builtin_and_method_calls_are_tool_steps():
    assert one("n = len(s)").steps[-1].tool == PY_BUILTIN_ALIAS
    assert one("u = s.upper()").steps[-1].tool == ATTR_CALL_ALIAS


def test_limit_one_keeps_the_first_statement_and_counts_dropped():
    statements, dropped = compile_statements("a = add(1, 2)\nb = add(3, 4)\nreturn b", limit=1)
    assert len(statements) == 1 and statements[0].source == "a = add(1, 2)"
    assert dropped == 2


def test_plan_mode_keeps_every_statement_with_its_source_and_ordinal():
    statements, dropped = compile_statements("a = add(1, 2)\nb = add(a, 4)\nreturn b", limit=None, start_ordinal=5)
    assert dropped == 0
    assert [s.ordinal for s in statements] == [5, 6, 7]
    assert [s.source for s in statements] == ["a = add(1, 2)", "b = add(a, 4)", "return b"]


def test_parse_errors_raise():
    with pytest.raises(BlackboardParseError):
        compile_statements("for x in y: pass", limit=1)
    with pytest.raises(BlackboardParseError):
        compile_statements("x = a or b", limit=1)


def test_bad_limit_raises():
    with pytest.raises(ValueError):
        compile_statements("x = 1", limit=2)


# ----------------------------------------------------------------------- #
# check
# ----------------------------------------------------------------------- #
def check(text: str, context=(), readonly=(), budget=None):
    return check_statements(
        plan(text), TOOLBOX, frozenset(context), frozenset(readonly), budget,
    )


def test_check_clean_and_unknown_tool_and_undefined_name():
    assert check("x = add(1, 2)\nreturn x") == []
    assert check("x = nope(1)")
    assert check("x = add(y, 2)")


def test_check_readonly_target_binding_and_budget():
    assert check("PI = add(1, 2)", context={"PI"}, readonly={"PI"})
    assert check("x = add(1, 2, 3)")
    assert check("a = add(1, 2)\nb = add(a, 3)", budget=1)


# ----------------------------------------------------------------------- #
# run
# ----------------------------------------------------------------------- #
def test_composed_statement_runs_innermost_first_and_binds_scratch():
    out = run_statement(one("x = add(mul(2, 3), 4)"), {}, resolve_tool)
    assert out.status == "done" and out.value == 10
    assert [s.status for s in out.steps] == ["ran", "ran"]
    assert out.steps[0].value == 6 and out.steps[1].value == 10


def test_arithmetic_around_a_call_is_evaluated_in_the_consuming_step():
    out = run_statement(one("x = add(1 + mul(2, 3), 0)"), {}, resolve_tool)
    assert out.value == 7
    assert len(out.steps) == 2


def test_names_come_from_the_namespace_and_it_is_not_mutated():
    ns = {"base": 5}
    out = run_statement(one("x = add(base, 1)"), ns, resolve_tool)
    assert out.value == 6 and ns == {"base": 5}


def test_tool_step_keeps_full_result_and_value_is_the_same_object():
    out = run_statement(one("x = make_list(3)"), {}, resolve_tool)
    step = out.steps[0]
    assert step.result is not None and step.result.invoker_id
    assert step.value is step.result.result


def test_eval_step_has_no_result():
    out = run_statement(one("y = base + 1"), {"base": 2}, resolve_tool)
    assert out.value == 3 and out.steps[0].result is None and out.steps[0].value == 3


def test_bare_call_returning_none_is_done_with_value_none():
    out = run_statement(one("log('a')"), {}, resolve_tool)
    assert out.status == "done" and out.value is None


def test_return_statement_value():
    out = run_statement(one("return base + 1"), {"base": 1}, resolve_tool)
    assert out.status == "done" and out.value == 2


def test_builtin_and_method_calls_run():
    assert run_statement(one("n = len(s)"), {"s": "abcd"}, resolve_tool).value == 4
    assert run_statement(one("u = s.upper()"), {"s": "ab"}, resolve_tool).value == "AB"


def test_mid_statement_failure_keeps_what_ran_and_stops():
    out = run_statement(one("x = add(mul(2, 3), boom(1))"), {}, resolve_tool)
    assert out.status == "failed" and out.value is NO_VAL
    statuses = [s.status for s in out.steps]
    assert statuses == ["ran", "failed", "not_run"]
    assert out.steps[0].value == 6 and out.steps[0].result is not None
    assert isinstance(out.steps[1].error, Exception)
    assert out.steps[2].error is None and out.steps[2].result is None


def test_resolution_failure_records_the_error():
    out = run_statement(one("x = add(missing, 1)"), {}, resolve_tool)
    assert out.status == "failed"
    step = out.steps[0]
    assert step.status == "failed" and step.error is not None and step.result is None


def test_run_returns_a_new_statement_and_leaves_the_planned_one():
    planned = one("x = add(1, 2)")
    out = run_statement(planned, {}, resolve_tool)
    assert out is not planned
    assert planned.status == "not_run" and planned.steps[0].status == "not_run"


def test_base_exception_propagates():
    def stop(x: int) -> int:
        raise KeyboardInterrupt

    tool = Tool(function=stop, name="stop", description="stop")
    with pytest.raises(KeyboardInterrupt):
        run_statement(one("y = stop(1)"), {}, lambda _id: tool)


def test_async_run_matches_sync():
    st = one("x = add(mul(2, 3), 4)")
    sync_out = run_statement(st, {}, resolve_tool)
    async_out = asyncio.run(async_run_statement(st, {}, resolve_tool))
    assert async_out.value == sync_out.value == 10
    assert [s.status for s in async_out.steps] == [s.status for s in sync_out.steps]
    failed = asyncio.run(async_run_statement(one("x = add(boom(1), 2)"), {}, resolve_tool))
    assert [s.status for s in failed.steps] == ["failed", "not_run"]


# ----------------------------------------------------------------------- #
# cost
# ----------------------------------------------------------------------- #
def test_planned_cost_rules():
    assert planned_cost(one("x = add(mul(2, 3), 4)")) == 2
    assert planned_cost(one("x = add(1, 2) + mul(3, 4)")) == 2
    assert planned_cost(one("y = a + 1")) == 1
    assert planned_cost(one("return a + 1")) == 0
    assert planned_cost(one("return 5")) == 0
    assert planned_cost(one("return add(1, 2)")) == 1
    assert planned_cost(one("log('x')")) == 1


def test_actions_used_counts_ran_and_failed_but_not_unrun():
    failed = run_statement(one("x = add(mul(2, 3), boom(1))"), {}, resolve_tool)
    assert actions_used([failed]) == 2
    planned = one("x = add(1, 2)")
    assert actions_used([planned]) == 0
    done = run_statement(one("y = a + 1"), {"a": 1}, resolve_tool)
    assert actions_used([failed, done]) == 3
    assert actions_used([run_statement(one("return base + 1"), {"base": 1}, resolve_tool)]) == 0


# ----------------------------------------------------------------------- #
# waves
# ----------------------------------------------------------------------- #
def ordinals(waves):
    return [[s.ordinal for s in w] for w in waves]


def test_independent_statements_share_a_wave_and_dependents_do_not():
    sts = plan("a = add(1, 2)\nb = add(3, 4)\nc = add(a, b)\nreturn c")
    assert ordinals(plan_waves(sts, None)) == [[0, 1], [2], [3]]


def test_concurrency_limit_caps_a_wave_and_one_is_sequential():
    sts = plan("a = add(1, 2)\nb = add(3, 4)\nc = add(5, 6)")
    assert ordinals(plan_waves(sts, 2)) == [[0, 1], [2]]
    assert ordinals(plan_waves(sts, 1)) == [[0], [1], [2]]


def test_write_after_read_and_write_after_write_split():
    war = plan("y = add(x, 1)\nx = add(2, 2)")
    assert ordinals(plan_waves(war, None)) == [[0], [1]]
    waw = plan("x = add(1, 1)\nx = add(2, 2)")
    assert ordinals(plan_waves(waw, None)) == [[0], [1]]


def test_method_call_on_a_name_counts_as_a_write_to_it():
    sts = plan("a = len(items)\nb = items.count(1)")
    assert ordinals(plan_waves(sts, None)) == [[0], [1]]


def test_bare_call_is_a_barrier_and_return_runs_alone():
    sts = plan("a = add(1, 2)\nlog('x')\nb = add(3, 4)\nreturn a")
    assert ordinals(plan_waves(sts, None)) == [[0], [1], [2], [3]]


# ----------------------------------------------------------------------- #
# render
# ----------------------------------------------------------------------- #
def test_failure_note_single_and_multi_step():
    single = run_statement(one("x = boom(1)"), {}, resolve_tool)
    assert render_statement_failure(single).startswith("x = boom(1)  # FAILED: ")
    multi = run_statement(one("x = add(mul(2, 3), boom(1))"), {}, resolve_tool)
    note = render_statement_failure(multi)
    assert "# sub-step `boom(1)` failed:" in note
    assert "# sub-steps that already ran: `mul(2, 3)` -> 6" in note


def test_render_statements_done_failed_and_not_run():
    done = run_statement(one("a = add(1, 2)"), {}, resolve_tool)
    failed = run_statement(one("b = boom(1)"), {}, resolve_tool)
    not_run = one("c = add(a, 1)")
    text = render_statements([done, failed, not_run])
    lines = text.split("\n")
    assert lines[0] == "a = add(1, 2)"
    assert lines[1].startswith("b = boom(1)  # FAILED")
    assert lines[2:] == ["c = add(a, 1)", "# NOT RUN"]
