from __future__ import annotations

import json
from typing import Any

import pytest

from atomic_agentic.utils.agents import (
    extract_json_object,
    extract_regex_steps,
    format_generation_issues,
)


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


class TestExtractRegexStepsNonLiteralArgs:
    def test_non_literal_kwarg_value_reports_issue_but_keeps_step(self) -> None:
        raw = (
            "[CALL] Tool.tests.add(a=1, b=STEP.field)\n"
            "[REASON] r\n"
            "[DURATION] 0"
        )
        steps, issues = extract_regex_steps(raw, source_label="Test.agent")

        assert len(steps) == 1
        assert steps[0]["args"]["a"] == 1
        assert steps[0]["args"]["b"] == "STEP.field"
        assert len(issues) == 1
        assert "argument 'b'" in issues[0]
        assert "could not be parsed as a Python literal" in issues[0]

    def test_non_literal_kwarg_value_without_placeholder_shape_has_no_hint(self) -> None:
        raw = "[CALL] Tool.tests.add(a=STEP.field)\n[REASON] r\n[DURATION] 0"
        _, issues = extract_regex_steps(raw, source_label="Test.agent")

        assert len(issues) == 1
        assert "placeholder" not in issues[0]

    def test_non_literal_kwarg_value_with_placeholder_shape_gets_hint(self) -> None:
        raw = "[CALL] Tool.tests.add(a=0|STEP.field|0)\n[REASON] r\n[DURATION] 0"
        _, issues = extract_regex_steps(raw, source_label="Test.agent")

        assert len(issues) == 1
        assert "unquoted" in issues[0]
        assert "|STEP.0|" in issues[0]  # the hint's own example text

    def test_whole_call_parse_failure_with_unquoted_placeholder_gets_hint(self) -> None:
        raw = "[CALL] Tool.tests.add(a=|STEP.0|)\n[REASON] r\n[DURATION] 0"
        steps, issues = extract_regex_steps(raw, source_label="Test.agent")

        assert steps == []
        assert len(issues) == 1
        assert "could not be parsed as a call expression" in issues[0]
        assert "unquoted" in issues[0]

    def test_whole_call_parse_failure_without_placeholder_has_no_hint(self) -> None:
        raw = "[CALL] Tool.tests.add(a=)\n[REASON] r\n[DURATION] 0"
        steps, issues = extract_regex_steps(raw, source_label="Test.agent")

        assert steps == []
        assert len(issues) == 1
        assert "unquoted" not in issues[0]

    def test_sub_tag_payload_failure_with_unquoted_placeholder_gets_hint(self) -> None:
        raw = "[CALL] Tool.tests.add()\n[REASON] r\n[DURATION] |STEP.0|"
        _, issues = extract_regex_steps(raw, source_label="Test.agent")

        assert len(issues) == 1
        assert "[DURATION] payload" in issues[0]
        assert "unquoted" in issues[0]

    def test_return_payload_parse_failure_stays_silent_no_hint(self) -> None:
        # RETURN's fallback-to-text behavior is unchanged by this pass:
        # no issue at all, regardless of placeholder-shaped content.
        raw = "[RETURN] |STEP.0|"
        steps, issues = extract_regex_steps(raw, source_label="Test.agent")

        assert issues == []
        assert steps[0]["args"]["val"] == "|STEP.0|"


class TestFormatGenerationIssues:
    def test_single_issue_no_header_unchanged(self) -> None:
        assert format_generation_issues(["only issue"]) == "only issue"

    def test_multi_issue_no_header_unchanged(self) -> None:
        result = format_generation_issues(["a", "b"])

        assert result.startswith("Multiple problems were found in your output:\n")
        assert "1. a" in result
        assert "2. b" in result
        assert result.endswith("Correct all of the above and resubmit.")

    def test_single_issue_with_category_header(self) -> None:
        result = format_generation_issues(["only issue"], category_header="Header:\n")

        assert result == "Header:\nonly issue"

    def test_multi_issue_with_category_header_replaces_generic_wrapper(self) -> None:
        result = format_generation_issues(["a", "b"], category_header="Header:\n")

        assert result.startswith("Header:\n")
        assert "Multiple problems were found" not in result
        assert "1. a" in result
        assert "2. b" in result
        assert result.endswith("Correct all of the above and resubmit.")

    def test_empty_issues_raises(self) -> None:
        with pytest.raises(ValueError, match="requires at least one issue"):
            format_generation_issues([])

    def test_error_message_includes_source_label(self) -> None:
        with pytest.raises(TypeError, match=r"MyAgent\.instance-1"):
            extract_json_object(None, source_label="MyAgent.instance-1")  # type: ignore[arg-type]
