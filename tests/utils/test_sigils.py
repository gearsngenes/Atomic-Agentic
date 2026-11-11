from __future__ import annotations

import ast

from atomic_agentic.constants.agents import RETURN_ALIAS, RETURN_VALUE_FIELD
from atomic_agentic.utils.sigils import (
    parse_generation,
    parse_react_call,
    translate_calls,
)
from atomic_agentic.utils.sigils import _DraftCall


def _draft(
    tool: str = "add",
    *,
    identifier: str | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
    duplicate_kwargs: tuple[str, ...] = (),
) -> _DraftCall:
    return _DraftCall(
        identifier=identifier,
        tool=tool,
        args=args,
        kwargs=kwargs or {},
        duplicate_kwargs=duplicate_kwargs,
    )


def _translate(drafts, *, limit=None, context=(), readonly=(), tools=()):
    return translate_calls(
        drafts, limit, frozenset(context), frozenset(readonly), frozenset(tools)
    )


class TestTranslateCallsReferences:
    def test_whole_reference_to_a_context_name_becomes_a_name_node(self) -> None:
        issues, calls = _translate([_draft(kwargs={"x": "$n"})], context={"n"})

        assert issues == []
        assert isinstance(calls[0].kwargs["x"], ast.Name) and calls[0].kwargs["x"].id == "n"

    def test_embedded_reference_to_a_context_name_becomes_a_joined_string(self) -> None:
        issues, calls = _translate([_draft(args=("n is $n",))], context={"n"})

        assert issues == []
        assert isinstance(calls[0].args[0], ast.JoinedStr)

    def test_unresolved_whole_reference_yields_one_issue_with_new_wording(self) -> None:
        issues, _ = _translate([_draft(identifier="a", kwargs={"x": "$ghost"})])

        assert len(issues) == 1
        assert "does not match any context variable or earlier result_name" in issues[0]
        assert "'$ghost'" in issues[0]
        assert "RUNTIME CONTEXT" in issues[0]
        assert "in this plan" not in issues[0]
        assert "constant" not in issues[0]

    def test_unresolved_embedded_reference_stays_text_without_an_issue(self) -> None:
        issues, calls = _translate([_draft(args=("hello $ghost",))])

        assert issues == []
        assert isinstance(calls[0].args[0], ast.Constant)
        assert calls[0].args[0].value == "hello $ghost"

    def test_earlier_result_name_is_referenceable_later(self) -> None:
        issues, _ = _translate([
            _draft(identifier="a", args=(1,)),
            _draft(identifier="b", kwargs={"x": "$a"}),
        ])

        assert issues == []

    def test_a_call_cannot_reference_its_own_result_name(self) -> None:
        issues, _ = _translate([_draft(identifier="a", kwargs={"x": "$a"})])

        assert len(issues) == 1


class TestTranslateCallsResultNames:
    def test_invalid_identifier_is_rejected(self) -> None:
        issues, _ = _translate([_draft(identifier="1bad")])

        assert issues == ["result_name '1bad' is not a valid identifier."]

    def test_task_result_prefix_is_rejected_even_without_such_a_variable(self) -> None:
        issues, _ = _translate([_draft(identifier="task_result_9")])

        assert len(issues) == 1 and "reserved 'task_result_' prefix" in issues[0]

    def test_readonly_context_name_is_rejected(self) -> None:
        issues, _ = _translate(
            [_draft(identifier="limit")], context={"limit"}, readonly={"limit"},
        )

        assert len(issues) == 1 and "read-only context variable" in issues[0]

    def test_dunder_name_is_rejected(self) -> None:
        issues, _ = _translate([_draft(identifier="__x__")])

        assert len(issues) == 1 and "reserved" in issues[0]

    def test_writable_existing_name_is_accepted(self) -> None:
        issues, _ = _translate([_draft(identifier="total")], context={"total"})

        assert issues == []

    def test_leading_sigil_and_whitespace_are_normalized(self) -> None:
        issues, calls = _translate([_draft(identifier="  $ total ")])

        assert issues == []
        assert calls[0].identifier == "total"

    def test_blank_identifier_collapses_to_none(self) -> None:
        issues, calls = _translate([_draft(identifier="  ")])

        assert issues == []
        assert calls[0].identifier is None


class TestTranslateCallsDuplicatesAndBudget:
    def test_duplicate_keyword_draft_yields_the_issue(self) -> None:
        issues, _ = _translate([
            _draft(identifier="a", kwargs={"x": 1}, duplicate_kwargs=("x",)),
        ])

        assert len(issues) == 1
        assert "['x']" in issues[0] and "given more than once" in issues[0]

    def test_budget_counts_real_calls_only(self) -> None:
        drafts = [_draft(), _draft(), _draft(RETURN_ALIAS, kwargs={RETURN_VALUE_FIELD: 1})]

        assert _translate(drafts, limit=2)[0] == []
        issues, _ = _translate(drafts, limit=1)
        assert len(issues) == 1 and "exceeding the 1 still allowed" in issues[0]


class TestParseRecordsDuplicateKeywords:
    def test_parse_generation_records_repeated_names_once_and_keeps_last_value(self) -> None:
        payload = {
            "plan": [{
                "call": "add",
                "arguments": [
                    {"name": "x", "value": 1},
                    {"name": "x", "value": 2},
                    {"name": "x", "value": 3},
                    {"name": "y", "value": 4},
                    {"name": "y", "value": 5},
                ],
                "result_name": None,
            }],
            "return": None,
        }

        drafts = parse_generation(payload)

        assert drafts[0].duplicate_kwargs == ("x", "y")
        assert drafts[0].kwargs == {"x": 3, "y": 5}
        assert drafts[-1].tool == RETURN_ALIAS and drafts[-1].duplicate_kwargs == ()

    def test_parse_react_call_records_repeated_names(self) -> None:
        payload = {
            "call": "add",
            "arguments": [{"name": "x", "value": 1}, {"name": "x", "value": 2}],
            "result_name": None,
        }

        draft = parse_react_call(payload)

        assert draft.duplicate_kwargs == ("x",)
        assert draft.kwargs == {"x": 2}


class TestTranslateCallsToolNamedResult:
    def test_result_name_equal_to_a_tool_id_is_rejected(self) -> None:
        issues, _ = _translate([_draft(identifier="add")], tools={"add"})

        assert any("registered tool id" in issue for issue in issues)

    def test_other_result_names_pass(self) -> None:
        issues, _ = _translate([_draft(identifier="total")], tools={"add"})

        assert issues == []
