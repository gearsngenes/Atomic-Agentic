from __future__ import annotations

import json
from typing import Any

import pytest

from atomic_agentic.utils.agents import extract_json_object


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
