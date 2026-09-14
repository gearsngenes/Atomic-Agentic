from __future__ import annotations

import ast

import pytest

from atomic_agentic.constants.agents import (
    ATTR_CALL_ALIAS,
    KWARGS_UNPACK_KEY,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
)
from atomic_agentic.exceptions import BlackboardParseError
from atomic_agentic.models.agents.blackboard_models import CodeStatement
from atomic_agentic.utils.agents import extract_identifiers
from atomic_agentic.utils.script import (
    compile_batches,
    is_dispatched_slot,
    parse_generation,
    parse_statement_to_slots,
    render_cache_snapshot,
    render_completed_as_python,
    resolve_slot_args,
    rewrite_builtin_calls,
    validate_references,
)


def _name(identifier: str) -> ast.Name:
    return ast.Name(id=identifier, ctx=ast.Load())


def _const(value: object) -> ast.Constant:
    return ast.Constant(value=value)


class TestParseStatementToSlotsAssignment:
    def test_plain_assignment_produces_rhs_assign_slot(self) -> None:
        slots = parse_statement_to_slots("x = 5")

        assert len(slots) == 1
        slot = slots[0]
        assert slot.identifier == "x"
        assert slot.tool == RHS_ASSIGN_ALIAS
        assert isinstance(slot.kwargs["val"], ast.Constant)
        assert slot.kwargs["val"].value == 5

    def test_constant_expression_is_not_folded(self) -> None:
        slot = parse_statement_to_slots("x = 5 * 1.5")[0]

        assert isinstance(slot.kwargs["val"], ast.BinOp)

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
            parse_statement_to_slots(statement)

    def test_reserved_sub_name_prefix_on_target_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="_SUB_"):
            parse_statement_to_slots("_SUB_0 = 1")

    def test_reserved_task_result_prefix_on_target_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="task_result_"):
            parse_statement_to_slots("task_result_0 = 1")

    def test_constant_expression_that_fails_to_evaluate_raises(self) -> None:
        with pytest.raises(BlackboardParseError) as exc_info:
            parse_statement_to_slots("x = 1 / 0")

        assert isinstance(exc_info.value.__cause__, ZeroDivisionError)

    def test_dependency_bearing_expression_is_not_dry_evaluated(self) -> None:
        slot = parse_statement_to_slots("x = y + 1")[0]

        assert slot.tool == RHS_ASSIGN_ALIAS
        assert "y" in extract_identifiers(slot.kwargs)


class TestParseStatementToSlotsCallShapes:
    def test_assigned_call(self) -> None:
        slot = parse_statement_to_slots("y = add(1, 2)")[0]

        assert slot.identifier == "y"
        assert slot.tool == "add"
        assert [a.value for a in slot.args] == [1, 2]

    def test_bare_unassigned_call(self) -> None:
        slots = parse_statement_to_slots("add(1, 2)")

        assert len(slots) == 1
        assert slots[0].identifier is None
        assert slots[0].tool == "add"

    def test_return_with_value(self) -> None:
        slot = parse_statement_to_slots("return 5")[0]

        assert slot.identifier is None
        assert slot.tool == RETURN_ALIAS
        assert slot.kwargs["val"].value == 5

    def test_bare_return_treated_as_return_none(self) -> None:
        slot = parse_statement_to_slots("return")[0]

        assert isinstance(slot.kwargs["val"], ast.Constant)
        assert slot.kwargs["val"].value is None

    def test_return_of_constant_that_fails_to_evaluate_raises(self) -> None:
        with pytest.raises(BlackboardParseError) as exc_info:
            parse_statement_to_slots("return 1 / 0")

        assert isinstance(exc_info.value.__cause__, ZeroDivisionError)

    def test_keyword_arguments(self) -> None:
        slot = parse_statement_to_slots("y = add(a=1, b=2)")[0]

        assert set(slot.kwargs.keys()) == {"a", "b"}
        assert slot.kwargs["a"].value == 1
        assert slot.kwargs["b"].value == 2

    def test_single_nested_call_is_hoisted_before_the_final_slot(self) -> None:
        slots = parse_statement_to_slots("y = add(inner(1), 2)")

        assert len(slots) == 2
        hoisted, final = slots
        assert hoisted.identifier == "_SUB_0"
        assert hoisted.tool == "inner"
        assert final.identifier == "y"
        assert final.tool == "add"
        assert isinstance(final.args[0], ast.Name) and final.args[0].id == "_SUB_0"

    def test_doubly_nested_calls_flatten_bottom_up_in_discovery_order(self) -> None:
        slots = parse_statement_to_slots("y = add(inner(mid(1)), 2)")

        # Append order is bottom-up (mid's own slot must fully resolve
        # before inner's CodeStatement can be built, so it lands in the
        # list first) -- but _SUB_N numbering follows outer-to-inner
        # discovery order (a call's own counter slot is reserved before its
        # arguments are ever recursed into), so the innermost call (mid)
        # ends up numbered HIGHER despite appearing FIRST in the list.
        assert [s.identifier for s in slots] == ["_SUB_1", "_SUB_0", "y"]
        assert slots[0].tool == "mid"
        assert slots[1].tool == "inner"
        assert isinstance(slots[1].args[0], ast.Name) and slots[1].args[0].id == "_SUB_1"

    def test_start_index_offsets_hoisted_names(self) -> None:
        slots = parse_statement_to_slots("y = add(inner(1), 2)", start_index=5)

        assert slots[0].identifier == "_SUB_5"


class TestParseStatementToSlotsAttributeCalls:
    def test_attribute_call_shape(self) -> None:
        slot = parse_statement_to_slots("y = obj.method(1)")[0]

        assert slot.tool == ATTR_CALL_ALIAS
        obj_expr, method_name, *rest = slot.args
        assert isinstance(obj_expr, ast.Name) and obj_expr.id == "obj"
        assert method_name == "method"
        assert [a.value for a in rest] == [1]

    def test_attribute_call_keyword_argument_stored_plainly(self) -> None:
        slot = parse_statement_to_slots("y = obj.method(x=1)")[0]

        assert slot.tool == ATTR_CALL_ALIAS
        assert slot.kwargs["x"].value == 1

    def test_call_free_object_chain_passes_through_unchanged(self) -> None:
        slots = parse_statement_to_slots("y = obj.attr.method(1)")

        assert len(slots) == 1
        final = slots[0]
        obj_expr = final.args[0]
        assert isinstance(obj_expr, ast.Attribute)
        assert ast.unparse(obj_expr) == "obj.attr"

    def test_embedded_call_in_object_position_is_hoisted(self) -> None:
        slots = parse_statement_to_slots("y = obj.method1().attr.method2(1)")

        assert len(slots) == 2
        hoisted, final = slots
        assert hoisted.identifier == "_SUB_0"
        assert hoisted.tool == ATTR_CALL_ALIAS
        assert final.tool == ATTR_CALL_ALIAS
        obj_expr = final.args[0]
        assert ast.unparse(obj_expr) == "_SUB_0.attr"

    def test_dunder_method_call_raises_via_build_call_slot(self) -> None:
        with pytest.raises(BlackboardParseError, match="dunder"):
            parse_statement_to_slots("y = obj.__reduce__()")

    def test_bare_dunder_access_raises_via_hoist_calls_walk(self) -> None:
        with pytest.raises(BlackboardParseError, match="dunder"):
            parse_statement_to_slots("y = obj.__class__")

    def test_mid_chain_dunder_access_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="dunder"):
            parse_statement_to_slots("y = obj.__class__.attr")

    def test_constant_object_expression_that_fails_to_evaluate_raises(self) -> None:
        # The attribute-call object sub-expression gets the same
        # dependency-free dry-eval check every other argument category
        # already gets.
        with pytest.raises(BlackboardParseError) as exc_info:
            parse_statement_to_slots("y = (1/0).to_bytes(2, 'big')")

        assert isinstance(exc_info.value.__cause__, ZeroDivisionError)

    def test_dependency_bearing_object_expression_is_not_dry_evaluated(self) -> None:
        # The dry-eval check only fires when extract_identifiers(obj_expr)
        # is empty -- a plain dependency-bearing object expression parses fine.
        slot = parse_statement_to_slots("y = obj.method(1)")[0]

        assert slot.tool == ATTR_CALL_ALIAS


class TestParseStatementToSlotsRejectedForms:
    def test_ternary_call_in_body_branch_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="conditional expression"):
            parse_statement_to_slots("y = f() if flag else 2")

    def test_ternary_call_in_orelse_branch_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="conditional expression"):
            parse_statement_to_slots("y = 1 if flag else g()")

    def test_ternary_call_in_both_branches_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="conditional expression"):
            parse_statement_to_slots("y = f() if True else g()")

    def test_ternary_with_no_call_anywhere_is_legal(self) -> None:
        slot = parse_statement_to_slots("y = 1 if flag else 2")[0]

        assert slot.tool == RHS_ASSIGN_ALIAS
        assert isinstance(slot.kwargs["val"], ast.IfExp)

    def test_ternary_call_only_in_condition_is_legal_and_hoisted(self) -> None:
        slots = parse_statement_to_slots("y = 1 if cond() else 2")

        assert len(slots) == 2
        assert slots[0].identifier == "_SUB_0"
        assert slots[0].tool == "cond"

    def test_await_in_assignment_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="await"):
            parse_statement_to_slots("y = await f()")

    def test_bare_top_level_await_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="await"):
            parse_statement_to_slots("await f()")

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
            parse_statement_to_slots(statement)

    def test_lambda_used_directly_as_a_calls_own_callee_raises(self) -> None:
        """
        Regression test: a lambda used directly as a call's own callee
        (`(lambda x: x)(1)`) is rejected the same as a lambda anywhere else
        (see test_comprehensions_and_lambdas_raise). Previously
        _build_call_slot's plain-tool branch never ran the rejection scan
        over `call_node.func` itself (only over each argument, via
        `_process_call_args`) -- fixed by having that branch call the
        shared `_reject_unsupported_forms` helper on `call_node.func`
        before treating it as a tool id.
        """
        with pytest.raises(BlackboardParseError, match="lambda"):
            parse_statement_to_slots("y = (lambda x: x)(1)")

    def test_if_statement_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="expected an assignment"):
            parse_statement_to_slots("if True:\n    pass")

    def test_for_statement_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="expected an assignment"):
            parse_statement_to_slots("for i in range(3):\n    pass")

    def test_multiple_statements_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="exactly one statement"):
            parse_statement_to_slots("x = 1\ny = 2")

    def test_genuine_syntax_error_raises(self) -> None:
        with pytest.raises(BlackboardParseError):
            parse_statement_to_slots("x = 1 +")


class TestParseGeneration:
    def test_return_terminates_without_continuation(self) -> None:
        slots, continue_planning = parse_generation("x = add(1, 2)\nreturn x")

        assert len(slots) == 2
        assert continue_planning is False

    def test_explicit_pause_requests_continuation(self) -> None:
        slots, continue_planning = parse_generation("x = add(1, 2)\n# PAUSE")

        assert len(slots) == 1
        assert continue_planning is True

    def test_reasoning_string_anywhere_is_skipped(self) -> None:
        slots, _ = parse_generation('"thinking about this"\nx = add(1, 2)')

        assert len(slots) == 1
        assert slots[0].tool == "add"

    def test_pause_text_inside_a_reasoning_string_is_not_a_real_marker(self) -> None:
        text = (
            '"""line one\n# PAUSE\nline three"""\n'
            "x = add(1, 2)\n"
            "return x"
        )
        slots, continue_planning = parse_generation(text)

        assert continue_planning is False
        assert len(slots) == 2

    def test_real_pause_as_trailing_comment_on_its_own_line(self) -> None:
        _, continue_planning = parse_generation("x = add(1, 2)\n# PAUSE")

        assert continue_planning is True

    def test_pause_shaped_trailing_comment_not_alone_on_its_line_is_not_a_marker(self) -> None:
        slots, continue_planning = parse_generation("x = 1  # PAUSE")

        assert continue_planning is False
        assert len(slots) == 1

    def test_if_cutoff_with_prior_work_truncates_silently(self) -> None:
        text = "x = add(1, 2)\nif x > 0:\n    y = add(1, 2)"
        slots, continue_planning = parse_generation(text)

        assert len(slots) == 1
        assert continue_planning is True

    def test_if_cutoff_with_no_prior_work_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="conditional statement"):
            parse_generation("if True:\n    y = 1")

    def test_pause_with_no_prior_work_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="before any real work"):
            parse_generation("# PAUSE")

    def test_return_and_pause_collision_raises(self) -> None:
        with pytest.raises(BlackboardParseError, match="return statement and a # PAUSE"):
            parse_generation("x = add(1, 2)\nreturn x\n# PAUSE")

    def test_fence_wrapped_generation_parses_like_unfenced(self) -> None:
        text = "```python\nx = add(1, 2)\nreturn x\n```"
        slots, continue_planning = parse_generation(text)

        assert len(slots) == 2
        assert continue_planning is False

    def test_unmatched_leading_fence_is_still_stripped(self) -> None:
        text = "```python\nx = add(1, 2)\nreturn x"
        slots, continue_planning = parse_generation(text)

        assert len(slots) == 2
        assert continue_planning is False

    def test_empty_generation(self) -> None:
        slots, continue_planning = parse_generation("")

        assert slots == []
        assert continue_planning is False


class TestIsDispatchedSlot:
    def test_rhs_assign_and_return_are_not_dispatched(self) -> None:
        rhs_slot = CodeStatement(identifier="y", tool=RHS_ASSIGN_ALIAS, kwargs={"val": _const(1)})
        return_slot = CodeStatement(identifier=None, tool=RETURN_ALIAS, kwargs={"val": _const(1)})

        assert is_dispatched_slot(rhs_slot) is False
        assert is_dispatched_slot(return_slot) is False

    @pytest.mark.parametrize("tool", ["add", PY_BUILTIN_ALIAS, ATTR_CALL_ALIAS])
    def test_everything_else_is_dispatched(self, tool: str) -> None:
        slot = CodeStatement(identifier="y", tool=tool)

        assert is_dispatched_slot(slot) is True


class TestRewriteBuiltinCalls:
    def test_eligible_builtin_is_rewritten(self) -> None:
        slot = CodeStatement(identifier="y", tool="len", args=(_name("x"),))

        issues = rewrite_builtin_calls([slot])

        assert issues == []
        assert slot.tool == PY_BUILTIN_ALIAS
        assert slot.args[0] == "len"
        assert isinstance(slot.args[1], ast.Name) and slot.args[1].id == "x"

    def test_excluded_builtin_is_left_unrewritten_and_reported(self) -> None:
        slot = CodeStatement(identifier="y", tool="eval", args=(_name("x"),))

        issues = rewrite_builtin_calls([slot])

        assert slot.tool == "eval"
        assert len(issues) == 1
        assert "eval" in issues[0]

    def test_non_builtin_tool_name_is_untouched(self) -> None:
        slot = CodeStatement(identifier="y", tool="my_registered_tool")

        issues = rewrite_builtin_calls([slot])

        assert issues == []
        assert slot.tool == "my_registered_tool"

    def test_rhs_assign_and_return_slots_are_skipped_unconditionally(self) -> None:
        rhs_slot = CodeStatement(identifier="y", tool=RHS_ASSIGN_ALIAS, kwargs={"val": _const(1)})
        return_slot = CodeStatement(identifier=None, tool=RETURN_ALIAS, kwargs={"val": _const(1)})

        issues = rewrite_builtin_calls([rhs_slot, return_slot])

        assert issues == []
        assert rhs_slot.tool == RHS_ASSIGN_ALIAS
        assert return_slot.tool == RETURN_ALIAS

    def test_mixed_batch_only_mutates_and_reports_the_relevant_slots(self) -> None:
        eligible = CodeStatement(identifier="a", tool="len", args=(_name("x"),))
        excluded = CodeStatement(identifier="b", tool="exec", args=(_name("x"),))
        untouched = CodeStatement(identifier="c", tool="my_tool")

        issues = rewrite_builtin_calls([eligible, excluded, untouched])

        assert eligible.tool == PY_BUILTIN_ALIAS
        assert excluded.tool == "exec"
        assert untouched.tool == "my_tool"
        assert len(issues) == 1


class TestResolveSlotArgs:
    def test_positional_and_keyword_resolution(self) -> None:
        slot = CodeStatement(identifier="y", tool="add", args=(_name("x"),), kwargs={"a": _name("x")})

        positional, keyword = resolve_slot_args(slot, {"x": 5})

        assert positional == [5]
        assert keyword == {"a": 5}

    def test_starred_positional_splices_iterable(self) -> None:
        slot = CodeStatement(identifier="y", tool="add", args=(ast.Starred(value=_name("nums"), ctx=ast.Load()),))

        positional, _ = resolve_slot_args(slot, {"nums": (1, 2, 3)})

        assert positional == [1, 2, 3]

    def test_starred_positional_resolving_to_non_iterable_raises(self) -> None:
        slot = CodeStatement(identifier="y", tool="add", args=(ast.Starred(value=_const(5), ctx=ast.Load()),))

        with pytest.raises(TypeError):
            resolve_slot_args(slot, {})

    def test_kwargs_unpack_merges_in(self) -> None:
        slot = CodeStatement(identifier="y", tool="add", kwargs={KWARGS_UNPACK_KEY: _name("kw")})

        _, keyword = resolve_slot_args(slot, {"kw": {"a": 1}})

        assert keyword == {"a": 1}

    def test_kwargs_unpack_collision_raises_type_error(self) -> None:
        slot = CodeStatement(
            identifier="y", tool="add",
            kwargs={"a": _const(1), KWARGS_UNPACK_KEY: _name("kw")},
        )

        with pytest.raises(TypeError, match="multiple values"):
            resolve_slot_args(slot, {"kw": {"a": 2}})

    def test_already_plain_value_passes_through_unchanged(self) -> None:
        slot = CodeStatement(identifier="y", tool=PY_BUILTIN_ALIAS, args=("len", _name("x")))

        positional, _ = resolve_slot_args(slot, {"x": [1, 2]})

        assert positional[0] == "len"
        assert positional[1] == [1, 2]


class TestValidateReferences:
    def test_unregistered_tool_reference_is_an_issue(self) -> None:
        slot = CodeStatement(identifier="y", tool="unknown_tool")

        issues = validate_references([slot], frozenset(), frozenset(), frozenset(), None)

        assert len(issues) == 1
        assert "unregistered tool" in issues[0]
        assert "unknown_tool" in issues[0]

    def test_undefined_name_reference_is_an_issue(self) -> None:
        slot = CodeStatement(identifier="y", tool="add", args=(_name("z"),))

        issues = validate_references([slot], frozenset({"add"}), frozenset(), frozenset(), None)

        assert len(issues) == 1
        assert "undefined name 'z'" in issues[0]

    def test_reference_to_earlier_bound_identifier_is_fine(self) -> None:
        first = CodeStatement(identifier="x", tool=RHS_ASSIGN_ALIAS, kwargs={"val": _const(1)})
        second = CodeStatement(identifier="y", tool="add", args=(_name("x"),))

        issues = validate_references([first, second], frozenset({"add"}), frozenset(), frozenset(), None)

        assert issues == []

    def test_reference_to_known_constant_is_fine(self) -> None:
        slot = CodeStatement(identifier="y", tool="add", args=(_name("K_X"),))

        issues = validate_references([slot], frozenset({"add"}), frozenset({"K_X"}), frozenset(), None)

        assert issues == []

    def test_reference_to_known_history_is_fine(self) -> None:
        slot = CodeStatement(identifier="y", tool="add", args=(_name("task_result_0"),))

        issues = validate_references([slot], frozenset({"add"}), frozenset(), frozenset({"task_result_0"}), None)

        assert issues == []

    def test_assigning_to_a_constant_name_is_rejected(self) -> None:
        slot = CodeStatement(identifier="K_X", tool=RHS_ASSIGN_ALIAS, kwargs={"val": _const(1)})

        issues = validate_references([slot], frozenset(), frozenset({"K_X"}), frozenset(), None)

        assert len(issues) == 1
        assert "read-only" in issues[0]

    def test_tool_call_budget_exceeded_is_reported(self) -> None:
        slots = [
            CodeStatement(identifier="a", tool="add"),
            CodeStatement(identifier="b", tool=PY_BUILTIN_ALIAS, args=("len", _name("a"))),
            CodeStatement(identifier="c", tool="add"),
        ]

        issues = validate_references(slots, frozenset({"add"}), frozenset(), frozenset(), 2)

        budget_issues = [i for i in issues if "exceeding the configured limit" in i]
        assert len(budget_issues) == 1
        assert "3" in budget_issues[0]

    def test_no_budget_issue_when_limit_is_none(self) -> None:
        slots = [CodeStatement(identifier="a", tool="add") for _ in range(5)]

        issues = validate_references(slots, frozenset({"add"}), frozenset(), frozenset(), None)

        assert issues == []

    def test_registered_tool_name_used_as_a_value_is_rejected(self) -> None:
        # A bare tool name used as a value (e.g. the object of an
        # attribute call) is rejected with its own specific message, not
        # silently treated as "known".
        slot = CodeStatement(
            identifier="y", tool=ATTR_CALL_ALIAS, args=(_name("sometool"), "attr"),
        )

        issues = validate_references([slot], frozenset({"sometool"}), frozenset(), frozenset(), None)

        assert len(issues) == 1
        assert "as a value" in issues[0]
        assert "sometool" in issues[0]

    def test_comprehensive_not_fail_fast(self) -> None:
        slots = [
            CodeStatement(identifier="y", tool="unknown_tool", args=(_name("z"),)),
            CodeStatement(identifier="w", tool="add"),
        ]

        issues = validate_references(slots, frozenset({"add"}), frozenset(), frozenset(), 0)

        assert any("unregistered tool" in i for i in issues)
        assert any("undefined name 'z'" in i for i in issues)
        assert any("exceeding the configured limit" in i for i in issues)


class TestCompileBatches:
    def test_independent_slots_share_a_batch(self) -> None:
        a = CodeStatement(identifier="a", tool="tool_a")
        b = CodeStatement(identifier="b", tool="tool_b")

        batches = compile_batches([a, b])

        assert batches == [[a, b]]
        assert a.batch_index == 0 and b.batch_index == 0

    def test_dependent_slot_starts_a_new_batch(self) -> None:
        a = CodeStatement(identifier="a", tool="tool_a")
        b = CodeStatement(identifier="b", tool="tool_b", args=(_name("a"),))

        batches = compile_batches([a, b])

        assert batches == [[a], [b]]
        assert a.batch_index == 0 and b.batch_index == 1

    def test_dependency_on_an_already_closed_batch_does_not_force_a_new_close(self) -> None:
        a = CodeStatement(identifier="a", tool="tool_a")
        b = CodeStatement(identifier="b", tool="tool_b", args=(_name("a"),))
        c = CodeStatement(identifier="c", tool="tool_c", args=(_name("a"),))

        batches = compile_batches([a, b, c])

        assert batches == [[a], [b, c]]

    def test_max_concurrency_forces_separate_batches(self) -> None:
        a = CodeStatement(identifier="a", tool="tool_a")
        b = CodeStatement(identifier="b", tool="tool_b")

        batches = compile_batches([a, b], max_concurrency=1)

        assert batches == [[a], [b]]

    def test_no_concurrency_cap_batches_independent_slots_together(self) -> None:
        a = CodeStatement(identifier="a", tool="tool_a")
        b = CodeStatement(identifier="b", tool="tool_b")

        batches = compile_batches([a, b], max_concurrency=None)

        assert batches == [[a, b]]

    def test_return_always_lands_alone_in_its_own_batch(self) -> None:
        a = CodeStatement(identifier="a", tool="tool_a")
        b = CodeStatement(identifier="b", tool="tool_b")
        ret = CodeStatement(identifier=None, tool=RETURN_ALIAS, kwargs={"val": _const(1)})

        batches = compile_batches([a, b, ret])

        assert batches == [[a, b], [ret]]

    def test_start_batch_index_offsets_stamped_indices(self) -> None:
        a = CodeStatement(identifier="a", tool="tool_a")

        compile_batches([a], start_batch_index=5)

        assert a.batch_index == 5

    def test_rhs_assign_slots_never_count_toward_concurrency_cap(self) -> None:
        slots = [
            CodeStatement(identifier=f"x{i}", tool=RHS_ASSIGN_ALIAS, kwargs={"val": _const(i)})
            for i in range(5)
        ]

        batches = compile_batches(slots, max_concurrency=1)

        assert batches == [slots]


class TestRenderCompletedAsPython:
    def test_empty_completed(self) -> None:
        assert render_completed_as_python([]) == ""

    def test_binop_renders_as_written_not_precomputed(self) -> None:
        slot = CodeStatement(identifier="name", tool=RHS_ASSIGN_ALIAS, kwargs={"val": ast.parse("5 * 1.5", mode="eval").body})

        assert render_completed_as_python([slot]) == "name = 5 * 1.5"

    def test_return_slot(self) -> None:
        slot = CodeStatement(identifier=None, tool=RETURN_ALIAS, kwargs={"val": _const(5)})

        assert render_completed_as_python([slot]) == "return 5"

    def test_registered_tool_call_with_identifier(self) -> None:
        slot = CodeStatement(identifier="y", tool="add", args=(_const(1), _const(2)))

        assert render_completed_as_python([slot]) == "y = add(1, 2)"

    def test_bare_call_has_no_leading_identifier(self) -> None:
        slot = CodeStatement(identifier=None, tool="log", args=(_const("hi"),))

        assert render_completed_as_python([slot]) == "log('hi')"

    def test_py_builtin_unsplices_to_natural_call_syntax(self) -> None:
        slot = CodeStatement(identifier="y", tool=PY_BUILTIN_ALIAS, args=("len", _name("x")))

        assert render_completed_as_python([slot]) == "y = len(x)"

    def test_attr_call_unsplices_to_dotted_call_syntax(self) -> None:
        slot = CodeStatement(identifier="y", tool=ATTR_CALL_ALIAS, args=(_name("obj"), "method", _const(1)))

        assert render_completed_as_python([slot]) == "y = obj.method(1)"

    def test_starred_positional_arg_renders_with_star_prefix(self) -> None:
        slot = CodeStatement(
            identifier="y", tool="add",
            args=(ast.Starred(value=_name("nums"), ctx=ast.Load()),),
        )

        assert render_completed_as_python([slot]) == "y = add(*nums)"

    def test_kwargs_unpack_renders_with_double_star_prefix(self) -> None:
        slot = CodeStatement(identifier="y", tool="add", kwargs={KWARGS_UNPACK_KEY: _name("kw")})

        assert render_completed_as_python([slot]) == "y = add(**kw)"

    def test_show_batches_groups_by_batch_index_with_headers(self) -> None:
        a = CodeStatement(identifier="a", tool="tool_a", batch_index=0)
        b = CodeStatement(identifier="b", tool="tool_b", batch_index=1)

        rendered = render_completed_as_python([a, b], show_batches=True)

        assert "# Batch 0:" in rendered
        assert "# Batch 1:" in rendered

        unrendered = render_completed_as_python([a, b], show_batches=False)
        assert "# Batch" not in unrendered


class TestRenderCacheSnapshot:
    def test_empty_completed(self) -> None:
        assert render_cache_snapshot([], {}, None) == ""

    def test_completed_with_no_bound_identifiers(self) -> None:
        slot = CodeStatement(identifier=None, tool="log", args=(_const("hi"),))

        assert render_cache_snapshot([slot], {}, None) == ""

    def test_two_bound_identifiers_in_first_occurrence_order(self) -> None:
        a = CodeStatement(identifier="a", tool="tool_a")
        b = CodeStatement(identifier="b", tool="tool_b")

        rendered = render_cache_snapshot([a, b], {"a": 1, "b": "two"}, None)

        assert rendered.index("a: int = 1") < rendered.index("b: str = 'two'")

    def test_reassigned_identifier_shows_only_its_latest_value(self) -> None:
        first = CodeStatement(identifier="a", tool="tool_a")
        second = CodeStatement(identifier="a", tool="tool_a")

        rendered = render_cache_snapshot([first, second], {"a": 2}, None)

        assert rendered.count("a:") == 1
        assert "a: int = 2" in rendered

    def test_preview_limit_truncates_long_values(self) -> None:
        slot = CodeStatement(identifier="a", tool="tool_a")

        rendered = render_cache_snapshot([slot], {"a": "x" * 20}, 5)

        # preview() truncates repr(value), and repr() of a string includes
        # its wrapping quote -- the first 5 characters are the quote plus 4
        # x's, not 5 x's.
        assert "'xxxx..." in rendered

    def test_no_preview_limit_means_no_truncation(self) -> None:
        slot = CodeStatement(identifier="a", tool="tool_a")

        rendered = render_cache_snapshot([slot], {"a": "x" * 20}, None)

        assert "x" * 20 in rendered
