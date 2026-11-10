from __future__ import annotations

import ast
import json
from typing import Any

import pytest

from atomic_agentic.constants.agents import (
    ATTR_CALL_ALIAS,
    KWARGS_UNPACK_KEY,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
)
from atomic_agentic.models.agents.blackboard_models import ToolStatement
from atomic_agentic.tools import Tool
from atomic_agentic.utils.agents import check_call_bindings, extract_json_object


def _name(identifier: str) -> ast.Name:
    return ast.Name(id=identifier, ctx=ast.Load())


def _const(value: object) -> ast.Constant:
    return ast.Constant(value=value)


class TestExtractJsonObject:
    def test_extracts_json_array_from_plain_text(self) -> None:
        value = extract_json_object(
            'Plan:\n[{"step": 0, "tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}]\nDone.',
            source_label="Test.agent",
        )

        assert value == [{"step": 0, "tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}]

    def test_extracts_json_array_from_markdown_fence(self) -> None:
        value = extract_json_object(
            '```json\n[{"step": 0, "tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}]\n```',
            source_label="Test.agent",
        )

        assert value == [{"step": 0, "tool": "Tool.tests.add", "args": {"x": 1, "y": 2}}]

    def test_extracts_json_object_from_plain_text(self) -> None:
        value = extract_json_object(
            'Before {"step": 0, "tool": "Tool.tests.add", "args": {}} after',
            source_label="Test.agent",
        )

        assert value == {"step": 0, "tool": "Tool.tests.add", "args": {}}

    def test_extracts_json_object_from_markdown_fence(self) -> None:
        value = extract_json_object(
            '```json\n{"step": 0, "tool": "Tool.tests.add", "args": {}}\n```',
            source_label="Test.agent",
        )

        assert value == {"step": 0, "tool": "Tool.tests.add", "args": {}}

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("[]", []),
            ("{}", {}),
        ],
    )
    def test_accepts_empty_json_array_or_object(self, raw: str, expected: Any) -> None:
        assert extract_json_object(raw, source_label="Test.agent") == expected

    @pytest.mark.parametrize("raw", ["", "   ", "not json"])
    def test_rejects_invalid_text(self, raw: str) -> None:
        with pytest.raises(json.JSONDecodeError):
            extract_json_object(raw, source_label="Test.agent")

    def test_skips_unparseable_candidates(self) -> None:
        value = extract_json_object(
            "garbage {not valid json then [1, 2, 3]", source_label="Test.agent"
        )

        assert value == [1, 2, 3]

    def test_rejects_non_string_input(self) -> None:
        with pytest.raises(TypeError, match="LLM returned non-string output"):
            extract_json_object(123, source_label="Test.agent")  # type: ignore[arg-type]

    def test_error_message_includes_source_label(self) -> None:
        with pytest.raises(TypeError, match=r"MyAgent\.instance-1"):
            extract_json_object(None, source_label="MyAgent.instance-1")  # type: ignore[arg-type]


def _add(a: int, b: int = 0) -> int:
    return a + b


_TOOLBOX = {"add": Tool(function=_add, name="add", namespace="tests", description="Adds.")}


class TestCheckCallBindings:
    def test_clean_call_returns_empty(self) -> None:
        slot = ToolStatement(identifier="y", tool="add", args=(_const(1),), kwargs={"b": _const(2)})

        assert check_call_bindings([slot], _TOOLBOX) == []

    def test_too_many_positionals(self) -> None:
        slot = ToolStatement(identifier="y", tool="add", args=(_const(1), _const(2), _const(3)))

        issues = check_call_bindings([slot], _TOOLBOX)

        assert len(issues) == 1
        assert "'add'" in issues[0] and "do not fit its signature" in issues[0]
        assert issues[0].startswith("statement producing y ")

    def test_unknown_keyword(self) -> None:
        slot = ToolStatement(identifier=None, tool="add", kwargs={"zzz": _const(1)})

        issues = check_call_bindings([slot], _TOOLBOX)

        assert len(issues) == 1
        assert "(unassigned)" in issues[0]

    def test_duplicate_argument_via_position_and_keyword(self) -> None:
        slot = ToolStatement(identifier="y", tool="add", args=(_const(1),), kwargs={"a": _const(2)})

        assert len(check_call_bindings([slot], _TOOLBOX)) == 1

    def test_star_unpack_is_skipped(self) -> None:
        starred = ast.Starred(value=_name("xs"), ctx=ast.Load())
        slot = ToolStatement(
            identifier="y", tool="add", args=(starred, _const(1), _const(2), _const(3)),
        )

        assert check_call_bindings([slot], _TOOLBOX) == []

    def test_kwargs_unpack_is_skipped(self) -> None:
        slot = ToolStatement(
            identifier="y", tool="add",
            args=(_const(1), _const(2), _const(3)), kwargs={KWARGS_UNPACK_KEY: _name("kw")},
        )

        assert check_call_bindings([slot], _TOOLBOX) == []

    def test_builtin_attr_and_sentinel_slots_are_skipped(self) -> None:
        slots = [
            ToolStatement(identifier="a", tool=PY_BUILTIN_ALIAS, args=("len", _const(1), _const(2))),
            ToolStatement(identifier="b", tool=ATTR_CALL_ALIAS, args=(_name("x"), "m", _const(1))),
            ToolStatement(identifier="c", tool=RHS_ASSIGN_ALIAS, kwargs={"val": _const(1)}),
            ToolStatement(identifier=None, tool=RETURN_ALIAS, kwargs={"val": _const(1)}),
            ToolStatement(identifier="d", tool="not_registered", args=(_const(1),)),
        ]

        assert check_call_bindings(slots, _TOOLBOX) == []
