from __future__ import annotations

import json
from typing import Any

import pytest

from atomic_agentic.utils.agents import extract_json_object, stringify_result


class TestStringifyResult:
    def test_str_value_returned_verbatim(self) -> None:
        assert stringify_result("plain text") == "plain text"

    def test_dict_value_rendered_as_json(self) -> None:
        value = {"summary": "ok", "confident": True, "score": None}
        assert stringify_result(value) == json.dumps(value)
        assert json.loads(stringify_result(value)) == value

    def test_list_value_rendered_as_json(self) -> None:
        assert stringify_result([1, 2, 3]) == "[1, 2, 3]"

    def test_int_float_bool_none_rendered_as_json_literals(self) -> None:
        assert stringify_result(3) == "3"
        assert stringify_result(3.5) == "3.5"
        assert stringify_result(True) == "true"
        assert stringify_result(None) == "null"


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
