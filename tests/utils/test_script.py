from __future__ import annotations

import ast

import pytest

from atomic_agentic.agents.tools import builtin_call_tool
from atomic_agentic.constants.agents import ATTR_CALL_ALIAS, PY_BUILTIN_ALIAS
from atomic_agentic.exceptions import BlackboardParseError, ToolRegistrationError
from atomic_agentic.models.agents.blackboard_models import Statement, Step
from atomic_agentic.tools import Tool
from atomic_agentic.utils.agents import extract_identifiers
from atomic_agentic.utils.script import (
    check_script_tool_name,
    dispatch_inputs,
    parse_first_statement,
    parse_generation,
    parse_statement,
    script_context_clashes,
    validate_references,
)


def _name(identifier: str) -> ast.Name:
    return ast.Name(id=identifier, ctx=ast.Load())


def _const(value: object) -> ast.Constant:
    return ast.Constant(value=value)


def _own(statement: Statement) -> Step:
    return statement.steps[-1]


class TestParseStatementAssignment:
    def test_plain_assignment_produces_an_eval_step(self) -> None:
        statement = parse_statement("x = 5")

        assert statement.kind == "assign" and statement.target == "x"
        assert statement.source == "x = 5"
        assert len(statement.steps) == 1
        step = _own(statement)
        assert step.kind == "eval" and step.tool is None and step.name is None
        assert isinstance(step.args[0], ast.Constant) and step.args[0].value == 5

    def test_constant_expression_is_not_folded(self) -> None:
        assert isinstance(_own(parse_statement("x = 5 * 1.5")).args[0], ast.BinOp)

    @pytest.mark.parametrize(
        "statement",
        [
            "x, y = 1, 2",
            "x = y = 1",
            "x += 1",
            "obj.attr = 1",
            "x[0] = 1",
            "*x, = (1,)",
        ],
    )
    def test_illegal_target_shapes_raise(self, statement: str) -> None:
        with pytest.raises(BlackboardParseError):
            parse_statement(statement)

    def test_reserved_sub_name_prefix_on_target_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="_SUB_"):
            parse_statement("_SUB_0 = 1")

    def test_reserved_task_result_prefix_on_target_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="task_result_"):
            parse_statement("task_result_0 = 1")

    def test_constant_expression_that_fails_to_evaluate_raises(self) -> None:
        with pytest.raises(BlackboardParseError) as exc_info:
            parse_statement("x = 1 / 0")

        assert isinstance(exc_info.value.__cause__, ZeroDivisionError)

    def test_dependency_bearing_expression_is_not_dry_evaluated(self) -> None:
        step = _own(parse_statement("x = y + 1"))

        assert step.kind == "eval"
        assert "y" in extract_identifiers(step.args)

    def test_ordinal_is_recorded(self) -> None:
        assert parse_statement("x = 1", ordinal=7).ordinal == 7


class TestParseStatementCallShapes:
    def test_assigned_call(self) -> None:
        statement = parse_statement("y = add(1, 2)")
        step = _own(statement)

        assert statement.kind == "assign" and statement.target == "y"
        assert step.kind == "tool" and step.tool == "add" and step.name is None
        assert [a.value for a in step.args] == [1, 2]

    def test_bare_unassigned_call(self) -> None:
        statement = parse_statement("add(1, 2)")

        assert statement.kind == "call" and statement.target is None
        assert len(statement.steps) == 1 and _own(statement).tool == "add"

    def test_return_with_value(self) -> None:
        statement = parse_statement("return 5")
        step = _own(statement)

        assert statement.kind == "return" and statement.target is None
        assert step.kind == "eval" and step.args[0].value == 5

    def test_bare_return_treated_as_return_none(self) -> None:
        step = _own(parse_statement("return"))

        assert isinstance(step.args[0], ast.Constant) and step.args[0].value is None

    def test_return_of_a_call_hoists_it(self) -> None:
        statement = parse_statement("return add(1, 2)")

        assert statement.kind == "return"
        assert [s.kind for s in statement.steps] == ["tool", "eval"]
        assert statement.steps[0].name == "_SUB_0"

    def test_return_of_constant_that_fails_to_evaluate_raises(self) -> None:
        with pytest.raises(BlackboardParseError) as exc_info:
            parse_statement("return 1 / 0")

        assert isinstance(exc_info.value.__cause__, ZeroDivisionError)

    def test_keyword_arguments(self) -> None:
        step = _own(parse_statement("y = add(a=1, b=2)"))

        assert set(step.kwargs) == {"a", "b"}
        assert step.kwargs["a"].value == 1 and step.kwargs["b"].value == 2

    def test_single_nested_call_is_hoisted_before_the_own_step(self) -> None:
        statement = parse_statement("y = add(inner(1), 2)")

        assert len(statement.steps) == 2
        hoisted, own = statement.steps
        assert hoisted.name == "_SUB_0" and hoisted.tool == "inner"
        assert statement.target == "y" and own.name is None and own.tool == "add"
        assert isinstance(own.args[0], ast.Name) and own.args[0].id == "_SUB_0"

    def test_doubly_nested_calls_flatten_bottom_up_in_discovery_order(self) -> None:
        statement = parse_statement("y = add(inner(mid(1)), 2)")

        # Append order is bottom-up (mid must resolve before inner is built),
        # but _SUB_N numbering follows outer-to-inner discovery order, so the
        # innermost call is numbered higher despite coming first.
        assert [s.name for s in statement.steps] == ["_SUB_1", "_SUB_0", None]
        assert statement.steps[0].tool == "mid"
        assert statement.steps[1].tool == "inner"
        assert statement.steps[1].args[0].id == "_SUB_1"

    def test_start_index_offsets_hoisted_names(self) -> None:
        statement = parse_statement("y = add(inner(1), 2)", start_index=5)

        assert statement.steps[0].name == "_SUB_5"

    def test_allowed_builtin_call_dispatches_through_the_builtin_sentinel(self) -> None:
        step = _own(parse_statement("n = len(x)"))

        assert step.tool == PY_BUILTIN_ALIAS and step.args[0] == "len"
        assert isinstance(step.args[1], ast.Name) and step.args[1].id == "x"

    @pytest.mark.parametrize("tool", ["eval", "__loader__", "object"])
    def test_disallowed_builtin_is_left_as_a_plain_tool_name(self, tool: str) -> None:
        assert _own(parse_statement(f"y = {tool}(x)")).tool == tool

    def test_non_builtin_tool_name_is_untouched(self) -> None:
        assert _own(parse_statement("y = my_registered_tool(1)")).tool == "my_registered_tool"


class TestParseStatementAttributeCalls:
    def test_attribute_call_shape(self) -> None:
        step = _own(parse_statement("y = obj.get(1)"))

        assert step.tool == ATTR_CALL_ALIAS
        obj_expr, method_name, *rest = step.args
        assert isinstance(obj_expr, ast.Name) and obj_expr.id == "obj"
        assert method_name == "get"
        assert [a.value for a in rest] == [1]

    def test_attribute_call_keyword_argument_stored_plainly(self) -> None:
        step = _own(parse_statement("y = obj.get(x=1)"))

        assert step.tool == ATTR_CALL_ALIAS and step.kwargs["x"].value == 1

    def test_call_free_object_chain_passes_through_unchanged(self) -> None:
        statement = parse_statement("y = obj.attr.get(1)")

        assert len(statement.steps) == 1
        assert ast.unparse(_own(statement).args[0]) == "obj.attr"

    def test_embedded_call_in_object_position_is_hoisted(self) -> None:
        statement = parse_statement("y = obj.strip().attr.get(1)")

        assert len(statement.steps) == 2
        hoisted, own = statement.steps
        assert hoisted.name == "_SUB_0" and hoisted.tool == ATTR_CALL_ALIAS
        assert own.tool == ATTR_CALL_ALIAS
        assert ast.unparse(own.args[0]) == "_SUB_0.attr"

    @pytest.mark.parametrize("method", ["format", "format_map", "ljust", "encode", "no_such_method"])
    def test_method_in_no_permitted_class_raises_at_parse_time(self, method: str) -> None:
        with pytest.raises(BlackboardParseError, match="not an allowed method call"):
            parse_statement(f"y = obj.{method}(1)")

    def test_dunder_method_call_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="dunder"):
            parse_statement("y = obj.__reduce__()")

    def test_bare_dunder_access_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="dunder"):
            parse_statement("y = obj.__class__")

    def test_mid_chain_dunder_access_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="dunder"):
            parse_statement("y = obj.__class__.attr")

    def test_constant_object_expression_that_fails_to_evaluate_raises(self) -> None:
        with pytest.raises(BlackboardParseError) as exc_info:
            parse_statement("y = (1/0).to_bytes(2, 'big')")

        assert isinstance(exc_info.value.__cause__, ZeroDivisionError)

    def test_dependency_bearing_object_expression_is_not_dry_evaluated(self) -> None:
        assert _own(parse_statement("y = obj.get(1)")).tool == ATTR_CALL_ALIAS


class TestParseStatementRejectedForms:
    def test_ternary_call_in_body_branch_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="conditional expression"):
            parse_statement("y = f() if flag else 2")

    def test_ternary_call_in_orelse_branch_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="conditional expression"):
            parse_statement("y = 1 if flag else g()")

    def test_ternary_call_in_both_branches_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="conditional expression"):
            parse_statement("y = f() if True else g()")

    def test_ternary_with_no_call_anywhere_is_legal(self) -> None:
        step = _own(parse_statement("y = 1 if flag else 2"))

        assert step.kind == "eval" and isinstance(step.args[0], ast.IfExp)

    def test_ternary_call_only_in_condition_is_legal_and_hoisted(self) -> None:
        statement = parse_statement("y = 1 if cond() else 2")

        assert len(statement.steps) == 2
        assert statement.steps[0].name == "_SUB_0" and statement.steps[0].tool == "cond"

    def test_await_in_assignment_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="await"):
            parse_statement("y = await f()")

    def test_bare_top_level_await_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="await"):
            parse_statement("await f()")

    @pytest.mark.parametrize(
        "statement",
        [
            "y = [i for i in range(3)]",
            "y = {i for i in range(3)}",
            "y = {i: i for i in range(3)}",
            "y = (i for i in range(3))",
            "y = (lambda x: x + 1)",
        ],
    )
    def test_comprehensions_and_lambdas_raise(self, statement: str) -> None:
        with pytest.raises(BlackboardParseError):
            parse_statement(statement)

    def test_lambda_used_directly_as_a_calls_own_callee_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="lambda"):
            parse_statement("y = (lambda x: x)(1)")

    def test_if_statement_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="expected an assignment"):
            parse_statement("if True:\n    pass")

    def test_for_statement_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="expected an assignment"):
            parse_statement("for i in range(3):\n    pass")

    def test_multiple_statements_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="exactly one statement"):
            parse_statement("x = 1\ny = 2")

    def test_genuine_syntax_error_raises(self) -> None:
        with pytest.raises(BlackboardParseError):
            parse_statement("x = 1 +")


class TestParseFirstStatement:
    def test_returns_the_first_statement_and_the_dropped_count(self) -> None:
        statement, dropped = parse_first_statement("a = add(1, 2)\nb = add(3, 4)\nreturn b")

        assert statement.source == "a = add(1, 2)" and dropped == 2

    def test_ordinal_and_start_index_are_forwarded(self) -> None:
        statement, _ = parse_first_statement("y = add(inner(1), 2)", ordinal=3, start_index=4)

        assert statement.ordinal == 3 and statement.steps[0].name == "_SUB_4"

    def test_leading_notes_are_skipped(self) -> None:
        statement, dropped = parse_first_statement('"thinking"\nx = add(1, 2)')

        assert statement.source == "x = add(1, 2)" and dropped == 0

    def test_only_notes_is_an_error(self) -> None:
        with pytest.raises(BlackboardParseError, match="no statement"):
            parse_first_statement('"just a note"')

    @pytest.mark.parametrize("text", ["x = a or b", "x = 1 if a else 2"])
    def test_boolop_and_ternary_are_rejected(self, text: str) -> None:
        with pytest.raises(BlackboardParseError):
            parse_first_statement(text)

    def test_fence_is_stripped(self) -> None:
        statement, _ = parse_first_statement("```python\nx = add(1, 2)\n```")

        assert statement.source == "x = add(1, 2)"


class TestParseGeneration:
    """
    A flat list of ``Statement``s; a ``return`` terminates the generation and an
    ``ast.If`` always raises (there is no silent-truncation tolerance).
    """

    def test_return_terminates_the_generation(self) -> None:
        statements = parse_generation("x = add(1, 2)\nreturn x")

        assert len(statements) == 2
        assert statements[-1].kind == "return"

    def test_ordinals_run_from_start_ordinal(self) -> None:
        statements = parse_generation("x = add(1, 2)\nreturn x", start_ordinal=4)

        assert [s.ordinal for s in statements] == [4, 5]

    def test_hoisted_names_do_not_collide_across_statements(self) -> None:
        statements = parse_generation("a = add(f(1), 2)\nb = add(g(1), 2)")

        names = [s.name for st in statements for s in st.steps if s.name is not None]
        assert len(names) == len(set(names)) == 2

    def test_return_terminates_before_anything_that_follows_is_processed(self) -> None:
        statements = parse_generation("return 1\nif True:\n    y = 1")

        assert len(statements) == 1

    def test_reasoning_string_anywhere_is_skipped(self) -> None:
        statements = parse_generation('"thinking about this"\nx = add(1, 2)')

        assert len(statements) == 1
        assert _own(statements[0]).tool == "add"

    def test_if_statement_always_raises_with_prior_work(self) -> None:
        text = "x = add(1, 2)\nif x > 0:\n    y = add(1, 2)"

        with pytest.raises(BlackboardParseError, match="conditional statements"):
            parse_generation(text)

    def test_if_statement_always_raises_with_no_prior_work(self) -> None:
        with pytest.raises(BlackboardParseError, match="conditional statements"):
            parse_generation("if True:\n    y = 1")

    def test_falling_off_the_end_with_no_return_simply_means_plan_is_done(self) -> None:
        statements = parse_generation("x = add(1, 2)\ny = add(x, 1)")

        assert len(statements) == 2
        assert all(st.kind != "return" for st in statements)

    def test_fence_wrapped_generation_parses_like_unfenced(self) -> None:
        statements = parse_generation("```python\nx = add(1, 2)\nreturn x\n```")

        assert len(statements) == 2

    def test_unmatched_leading_fence_is_still_stripped(self) -> None:
        assert len(parse_generation("```python\nx = add(1, 2)\nreturn x")) == 2

    def test_prose_around_a_single_fenced_block_is_ignored(self) -> None:
        text = "Here is the plan:\n```python\nx = add(1, 2)\nreturn x\n```\nHope that helps!"
        statements = parse_generation(text)

        assert len(statements) == 2
        assert statements[-1].kind == "return"

    def test_two_fenced_blocks_are_rejected_not_merged(self) -> None:
        text = "```python\nx = add(1, 2)\n```\nthen\n```python\nreturn x\n```"

        with pytest.raises(BlackboardParseError):
            parse_generation(text)

    def test_backticks_inside_a_string_do_not_close_the_fence(self) -> None:
        text = "```python\nx = add('```', 2)\nreturn x\n```"

        assert len(parse_generation(text)) == 2

    def test_empty_generation(self) -> None:
        assert parse_generation("") == []

    def test_all_statement_errors_collected_with_line_numbers(self) -> None:
        text = "x = add(1, 2)\nif x:\n    y = 1\nz += 1\nw = add(x, 1)"

        with pytest.raises(BlackboardParseError) as excinfo:
            parse_generation(text)

        lines = str(excinfo.value).split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("line 2: ") and "conditional statements" in lines[0]
        assert lines[1].startswith("line 4: ")

    def test_duplicate_keyword_rejected_by_compile_gate(self) -> None:
        with pytest.raises(BlackboardParseError, match=r"line 1: .*keyword argument repeated"):
            parse_generation("x = add(a=1, a=2)")

    def test_debug_assignment_rejected_by_compile_gate(self) -> None:
        with pytest.raises(BlackboardParseError, match=r"line 2: .*__debug__"):
            parse_generation("x = add(1, 2)\n__debug__ = 1")

    def test_error_after_return_is_not_reported(self) -> None:
        assert len(parse_generation("return 1\nz += 1\nx = add(a=1, a=2)")) == 1

    def test_compile_gate_skipped_when_a_statement_error_exists(self) -> None:
        text = "await add(1, 2)\nx = add(a=1, a=2)"

        with pytest.raises(BlackboardParseError) as excinfo:
            parse_generation(text)

        message = str(excinfo.value)
        assert "'await' is not supported" in message
        assert "keyword argument repeated" not in message

    def test_walrus_rejected(self) -> None:
        with pytest.raises(BlackboardParseError, match="walrus assignment"):
            parse_generation("x = (y := 1)")

    def test_yield_rejected(self) -> None:
        with pytest.raises(BlackboardParseError, match="yield expressions are not supported"):
            parse_generation("x = yield 1")


class TestDispatchInputs:
    def test_registered_tool_splats_positional_and_keyword(self) -> None:
        def add(a: int, b: int = 0) -> int:
            return a + b

        tool = Tool(function=add, name="add", description="Adds.")
        step = Step(kind="tool", tool="add")

        assert dispatch_inputs(step, tool, [1], {"b": 2}) == {"a": 1, "b": 2}

    def test_builtin_step_packs_the_real_arguments(self) -> None:
        step = Step(kind="tool", tool=PY_BUILTIN_ALIAS)

        assert dispatch_inputs(step, builtin_call_tool, ["len", [1]], {}) == (
            builtin_call_tool._args_kwargs_to_dict("len", ([1],), {})
        )


class TestToolNameAndContextChecks:
    def test_registrable_name_passes(self) -> None:
        check_script_tool_name("add")

    @pytest.mark.parametrize("name", [PY_BUILTIN_ALIAS, ATTR_CALL_ALIAS, "len", "class"])
    def test_sentinel_builtin_and_keyword_names_are_rejected(self, name: str) -> None:
        with pytest.raises(ToolRegistrationError):
            check_script_tool_name(name)

    def test_return_is_a_keyword_so_it_is_still_rejected(self) -> None:
        with pytest.raises(ToolRegistrationError):
            check_script_tool_name("return")

    def test_context_clashes_name_keywords_and_tool_ids(self) -> None:
        clashes = script_context_clashes(["class", "add", "ok"], {"add"})

        assert set(clashes) == {"class", "add"}


def _stmt(kind: str, target, *steps: Step) -> Statement:
    return Statement(ordinal=0, source="", kind=kind, target=target, steps=steps)


def _eval(expr: ast.expr) -> Step:
    return Step(kind="eval", args=(expr,))


def _issues(statements, tools=(), context=(), readonly=()) -> list[str]:
    return validate_references(
        statements, frozenset(tools), frozenset(context), frozenset(readonly),
    )


class TestValidateReferences:
    def test_unregistered_tool_reference_is_an_issue(self) -> None:
        st = _stmt("assign", "y", Step(kind="tool", tool="unknown_tool"))

        issues = _issues([st])

        assert len(issues) == 1
        assert "unregistered tool" in issues[0] and "unknown_tool" in issues[0]

    def test_undefined_name_reference_is_an_issue(self) -> None:
        st = _stmt("assign", "y", Step(kind="tool", tool="add", args=(_name("z"),)))

        issues = _issues([st], tools={"add"})

        assert len(issues) == 1
        assert "undefined name 'z'" in issues[0]

    def test_reference_to_an_earlier_target_is_fine(self) -> None:
        first = _stmt("assign", "x", _eval(_const(1)))
        second = _stmt("assign", "y", Step(kind="tool", tool="add", args=(_name("x"),)))

        assert _issues([first, second], tools={"add"}) == []

    def test_reference_to_a_hoisted_name_within_a_statement_is_fine(self) -> None:
        hoisted = Step(kind="tool", tool="inner", name="_SUB_0")
        own = Step(kind="tool", tool="add", args=(_name("_SUB_0"),))

        assert _issues([_stmt("assign", "y", hoisted, own)], tools={"add", "inner"}) == []

    def test_reference_to_a_context_name_is_fine(self) -> None:
        st = _stmt("assign", "y", Step(kind="tool", tool="add", args=(_name("PI"), _name("task_result_0"))))

        assert _issues([st], tools={"add"}, context={"PI", "task_result_0"}, readonly={"PI"}) == []

    def test_assigning_to_a_readonly_context_name_is_rejected(self) -> None:
        issues = _issues([_stmt("assign", "PI", _eval(_const(1)))], context={"PI"}, readonly={"PI"})

        assert len(issues) == 1 and "read-only context variable" in issues[0]

    def test_assigning_to_a_tool_id_is_rejected(self) -> None:
        issues = _issues([_stmt("assign", "add", _eval(_const(1)))], tools={"add"})

        assert len(issues) == 1 and "registered tool id" in issues[0]

    def test_assigning_to_a_writable_context_name_is_allowed(self) -> None:
        assert _issues([_stmt("assign", "notes", _eval(_const(1)))], context={"notes"}) == []

    def test_disallowed_builtin_is_reported_specifically(self) -> None:
        issues = _issues([_stmt("assign", "y", Step(kind="tool", tool="eval"))])

        assert len(issues) == 1 and "not an allowed builtin" in issues[0]

    def test_builtin_and_method_steps_are_not_checked_against_the_toolbox(self) -> None:
        st = _stmt("assign", "y", Step(kind="tool", tool=PY_BUILTIN_ALIAS, args=("len", _const(1))))

        assert _issues([st]) == []

    def test_registered_tool_name_used_as_a_value_is_rejected(self) -> None:
        step = Step(kind="tool", tool=ATTR_CALL_ALIAS, args=(_name("sometool"), "attr"))

        issues = _issues([_stmt("assign", "y", step)], tools={"sometool"})

        assert len(issues) == 1
        assert "as a value" in issues[0] and "sometool" in issues[0]

    def test_comprehensive_not_fail_fast(self) -> None:
        statements = [
            _stmt("assign", "y", Step(kind="tool", tool="unknown_tool", args=(_name("z"),))),
            _stmt("assign", "w", Step(kind="tool", tool="add", args=(_name("q"),))),
        ]

        issues = _issues(statements, tools={"add"})

        assert any("unregistered tool" in i for i in issues)
        assert any("undefined name 'z'" in i for i in issues)
        assert any("undefined name 'q'" in i for i in issues)

    def test_an_unassigned_statement_is_labelled_as_such(self) -> None:
        issues = _issues([_stmt("call", None, Step(kind="tool", tool="nope"))])

        assert "(unassigned)" in issues[0]


class TestValidateReferencesMutatorsOnReadonly:
    @staticmethod
    def _check(script: str, readonly: frozenset[str] = frozenset({"ro"})) -> list[str]:
        return validate_references(
            parse_generation(script), frozenset(), frozenset({"ro", "rw"}) | readonly, readonly,
        )

    def test_direct_mutator_on_a_readonly_name_is_flagged(self) -> None:
        issues = self._check("ro.append(1)")

        assert any("mutating method 'append'" in i and "'ro'" in i for i in issues)

    def test_mutator_on_an_attribute_chain_of_a_readonly_name_is_flagged(self) -> None:
        assert any("mutating method" in i for i in self._check("ro.items.append(1)"))

    def test_non_mutating_methods_and_writable_names_pass(self) -> None:
        issues = self._check("a = ro.get(1)\nb = ro.copy()\nrw.append(1)")

        assert all("mutating" not in i for i in issues)

    def test_alias_by_assignment_is_protected(self) -> None:
        issues = self._check("a = ro\na.append(1)")

        assert any("mutating method 'append'" in i and "'a'" in i for i in issues)

    def test_reassigning_an_alias_to_a_new_value_clears_protection(self) -> None:
        issues = self._check("a = ro\na = [1]\na.append(2)")

        assert all("mutating" not in i for i in issues)

    def test_alias_by_get_is_protected(self) -> None:
        issues = self._check("v = ro.get(0)\nv.append(1)")

        assert any("mutating method" in i and "'v'" in i for i in issues)

    def test_copy_and_new_builtin_objects_are_not_protected(self) -> None:
        for setup in ("c = ro.copy()", "c = list(ro)", "c = ro.split(',')"):
            issues = self._check(setup + "\nc.append(1)")
            assert all("mutating" not in i for i in issues), setup
