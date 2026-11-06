from __future__ import annotations

import json
from typing import Any

import pytest

from atomic_agentic.constants.agents import STOP_EARLY_SCHEMA_TEMPLATE
from atomic_agentic.utils.agents import build_thinking_step_schema, extract_json_object


class TestBuildThinkingStepSchema:
    def test_none_types_the_thought_as_a_string(self) -> None:
        schema = build_thinking_step_schema(None)

        thought = schema["properties"]["thought"]
        assert thought["type"] == "string"
        assert thought["description"] == STOP_EARLY_SCHEMA_TEMPLATE["properties"]["thought"]["description"]

    def test_keeps_the_template_structure(self) -> None:
        schema = build_thinking_step_schema(None)

        assert schema["type"] == "object"
        assert schema["required"] == ["thought", "more_thinking_required"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["more_thinking_required"]["type"] == "boolean"

    def test_thought_schema_is_merged_into_the_thought_entry(self) -> None:
        user = {
            "type": "object",
            "properties": {"reasoning": {"type": "string"}, "next": {"type": "string"}},
            "required": ["reasoning", "next"],
        }

        thought = build_thinking_step_schema(user)["properties"]["thought"]

        assert thought["type"] == "object"
        assert thought["properties"] == user["properties"]
        assert thought["required"] == ["reasoning", "next"]
        assert "description" in thought

    def test_user_keys_win_including_the_description(self) -> None:
        thought = build_thinking_step_schema(
            {"type": "string", "description": "my own words"}
        )["properties"]["thought"]

        assert thought["description"] == "my own words"

    def test_does_not_alias_the_template_or_the_argument(self) -> None:
        user = {"type": "object", "properties": {"a": {"type": "string"}}}

        first = build_thinking_step_schema(user)
        first["properties"]["thought"]["properties"]["a"]["type"] = "integer"
        first["properties"]["more_thinking_required"]["type"] = "string"
        second = build_thinking_step_schema(user)

        assert user["properties"]["a"]["type"] == "string"
        assert second["properties"]["thought"]["properties"]["a"]["type"] == "string"
        assert STOP_EARLY_SCHEMA_TEMPLATE["properties"]["more_thinking_required"]["type"] == "boolean"
        assert "type" not in STOP_EARLY_SCHEMA_TEMPLATE["properties"]["thought"]


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
