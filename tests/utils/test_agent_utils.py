from __future__ import annotations

import ast

import pytest

from atomic_agentic.constants.agents import ATTR_CALL_ALIAS, KWARGS_UNPACK_KEY, PY_BUILTIN_ALIAS
from atomic_agentic.models.agents.blackboard_models import Statement, Step
from atomic_agentic.tools import Tool
from atomic_agentic.utils.agents import (
    build_usage_report,
    check_call_bindings,
    resolve_statement_args,
    tool_identity,
)


def _name(identifier: str) -> ast.Name:
    return ast.Name(id=identifier, ctx=ast.Load())


def _const(value: object) -> ast.Constant:
    return ast.Constant(value=value)


def _add(a: int, b: int = 0) -> int:
    return a + b


_TOOLBOX = {"add": Tool(function=_add, name="add", description="Adds.")}


def _tool_step(tool: str, *args, name=None, **kwargs) -> Step:
    return Step(kind="tool", tool=tool, name=name, args=tuple(args), kwargs=kwargs)


def _statement(step: Step, target=None, ordinal=0) -> Statement:
    return Statement(
        ordinal=ordinal, source="", kind="assign" if target else "call",
        target=target, steps=(step,),
    )


class TestCheckCallBindings:
    def test_clean_call_returns_empty(self) -> None:
        st = _statement(_tool_step("add", _const(1), b=_const(2)), "y")

        assert check_call_bindings([st], _TOOLBOX) == []

    def test_too_many_positionals(self) -> None:
        st = _statement(_tool_step("add", _const(1), _const(2), _const(3)), "y")

        issues = check_call_bindings([st], _TOOLBOX)

        assert len(issues) == 1
        assert "'add'" in issues[0] and "do not fit its signature" in issues[0]
        assert issues[0].startswith("statement producing y ")

    def test_unknown_keyword(self) -> None:
        st = _statement(_tool_step("add", zzz=_const(1)))

        issues = check_call_bindings([st], _TOOLBOX)

        assert len(issues) == 1
        assert "(unassigned)" in issues[0]

    def test_duplicate_argument_via_position_and_keyword(self) -> None:
        st = _statement(_tool_step("add", _const(1), a=_const(2)), "y")

        assert len(check_call_bindings([st], _TOOLBOX)) == 1

    def test_star_unpack_is_skipped(self) -> None:
        starred = ast.Starred(value=_name("xs"), ctx=ast.Load())
        st = _statement(_tool_step("add", starred, _const(1), _const(2), _const(3)), "y")

        assert check_call_bindings([st], _TOOLBOX) == []

    def test_kwargs_unpack_is_skipped(self) -> None:
        step = _tool_step("add", _const(1), _const(2), _const(3), **{KWARGS_UNPACK_KEY: _name("kw")})

        assert check_call_bindings([_statement(step, "y")], _TOOLBOX) == []

    def test_builtin_method_eval_and_unregistered_steps_are_skipped(self) -> None:
        statements = [
            _statement(_tool_step(PY_BUILTIN_ALIAS, "len", _const(1), _const(2)), "a"),
            _statement(_tool_step(ATTR_CALL_ALIAS, _name("x"), "m", _const(1)), "b"),
            _statement(Step(kind="eval", args=(_const(1),)), "c"),
            _statement(_tool_step("not_registered", _const(1)), "d"),
        ]

        assert check_call_bindings(statements, _TOOLBOX) == []

    def test_hoisted_step_is_labelled_by_its_scratch_name(self) -> None:
        hoisted = _tool_step("add", _const(1), _const(2), _const(3), name="_SUB_0")
        own = _tool_step("add", _name("_SUB_0"))
        st = Statement(ordinal=0, source="", kind="assign", target="y", steps=(hoisted, own))

        issues = check_call_bindings([st], _TOOLBOX)

        assert len(issues) == 1 and issues[0].startswith("statement producing _SUB_0 ")


class TestToolIdentity:
    def test_registered_builtin_and_method(self) -> None:
        assert tool_identity(_tool_step("add")) == "add"
        assert tool_identity(_tool_step(PY_BUILTIN_ALIAS, "len", _const(1))) == "len"
        assert tool_identity(_tool_step(ATTR_CALL_ALIAS, _name("x"), "upper")) == "upper"


def _ran(step: Step, status: str = "ran") -> Step:
    from dataclasses import replace
    return replace(step, status=status)


class TestBuildUsageReport:
    def test_counts_ran_and_failed_tool_steps_by_real_identity(self) -> None:
        sts = [
            _statement(_ran(_tool_step("add")), "a", 0),
            _statement(_ran(_tool_step(PY_BUILTIN_ALIAS, "len", _const(1))), "b", 1),
            _statement(_ran(_tool_step("add"), "failed"), "c", 2),
            _statement(Step(kind="eval", args=(_const(1),), status="ran"), "d", 3),
            _statement(_tool_step("add"), "e", 4),
        ]

        report = build_usage_report(sts)

        assert [(r.tool_name, r.call_count) for r in report.by_tool] == [("add", 2), ("len", 1)]
        assert report.total_dispatched == 3
        assert report.total_failed == 1

    def test_empty_is_an_empty_report(self) -> None:
        report = build_usage_report([])

        assert report.by_tool == () and report.total_dispatched == 0 and report.total_failed == 0


class TestResolveStatementArgs:
    def test_resolves_names_and_unpacks(self) -> None:
        step = _tool_step(
            "add", _name("x"), ast.Starred(value=_name("xs"), ctx=ast.Load()),
            b=_name("y"), **{KWARGS_UNPACK_KEY: _name("kw")},
        )

        positional, keyword = resolve_statement_args(
            step, {"x": 1, "xs": [2, 3], "y": 4, "kw": {"c": 5}},
        )

        assert positional == [1, 2, 3] and keyword == {"b": 4, "c": 5}

    def test_eval_step_resolves_to_its_expression_value(self) -> None:
        step = Step(kind="eval", args=(ast.parse("a + 1", mode="eval").body,))

        positional, keyword = resolve_statement_args(step, {"a": 2})

        assert positional == [3] and keyword == {}

    def test_plain_values_pass_through(self) -> None:
        positional, _ = resolve_statement_args(_tool_step(PY_BUILTIN_ALIAS, "len", _const(1)), {})

        assert positional == ["len", 1]

    def test_starred_positional_resolving_to_non_iterable_raises(self) -> None:
        step = _tool_step("add", ast.Starred(value=_const(5), ctx=ast.Load()))

        with pytest.raises(TypeError):
            resolve_statement_args(step, {})

    def test_overlapping_unpacked_keyword_raises(self) -> None:
        step = _tool_step("add", a=_const(1), **{KWARGS_UNPACK_KEY: _name("kw")})

        with pytest.raises(TypeError):
            resolve_statement_args(step, {"kw": {"a": 2}})
