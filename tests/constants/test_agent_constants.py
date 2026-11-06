from __future__ import annotations

import ast

import atomic_agentic.constants.agents as agent_constants

from atomic_agentic.constants.agents import (
    ATTR_CALL_ALIAS,
    CODE_FENCE_PATTERN,
    CONTEXT_END,
    CONTEXT_START,
    DUNDER_ATTRIBUTE_PATTERN,
    EXCLUDED_PY_BUILTINS,
    KWARGS_UNPACK_KEY,
    LEADING_CODE_FENCE_PATTERN,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    REPLY_DIRECTIVE,
    RHS_ASSIGN_ALIAS,
    STOP_EARLY_PROMPT_SECTION,
    STOP_EARLY_SCHEMA_TEMPLATE,
    SUB_NAME_PREFIX,
    TASK_PROMPT_END,
    TASK_PROMPT_START,
    TASK_RESULT_PREFIX,
    THINKING_DIRECTIVE,
    THINKING_INSTRUCTIONS_END,
    THINKING_INSTRUCTIONS_START,
    THOUGHTS_END,
    THOUGHTS_START,
    TRAILING_CODE_FENCE_PATTERN,
    UNSUPPORTED_EXPR_LABELS,
)


class TestScriptAgentSentinelLiterals:
    def test_reserved_call_sentinels(self) -> None:
        assert RHS_ASSIGN_ALIAS == "rhs_assign"
        assert RETURN_ALIAS == "return"
        assert PY_BUILTIN_ALIAS == "py_builtin"
        assert ATTR_CALL_ALIAS == "attr_call"

    def test_reserved_name_prefixes(self) -> None:
        assert SUB_NAME_PREFIX == "_SUB_"
        assert TASK_RESULT_PREFIX == "task_result_"

    def test_kwargs_unpack_key_is_not_a_valid_identifier(self) -> None:
        assert KWARGS_UNPACK_KEY == "**"
        assert not KWARGS_UNPACK_KEY.isidentifier()


class TestExcludedPyBuiltins:
    def test_is_a_non_empty_frozenset(self) -> None:
        assert isinstance(EXCLUDED_PY_BUILTINS, frozenset)
        assert EXCLUDED_PY_BUILTINS

    def test_contains_known_dangerous_builtins(self) -> None:
        for name in ("eval", "exec", "open", "__import__"):
            assert name in EXCLUDED_PY_BUILTINS


class TestDunderAttributePattern:
    def test_matches_dunder_names(self) -> None:
        assert DUNDER_ATTRIBUTE_PATTERN.fullmatch("__class__")
        assert DUNDER_ATTRIBUTE_PATTERN.fullmatch("__x__")

    def test_does_not_match_non_dunder_names(self) -> None:
        assert not DUNDER_ATTRIBUTE_PATTERN.fullmatch("_private")
        assert not DUNDER_ATTRIBUTE_PATTERN.fullmatch("public")


class TestCodeFencePatterns:
    def test_code_fence_pattern_matches_full_block(self) -> None:
        match = CODE_FENCE_PATTERN.match("```python\nx = 1\n```")
        assert match is not None
        assert match.group(1) == "x = 1"

    def test_leading_code_fence_pattern_matches_only_leading_half(self) -> None:
        assert LEADING_CODE_FENCE_PATTERN.match("```python\nx = 1")
        assert not TRAILING_CODE_FENCE_PATTERN.match("```python\nx = 1")

    def test_trailing_code_fence_pattern_matches_only_trailing_half(self) -> None:
        assert TRAILING_CODE_FENCE_PATTERN.search("x = 1\n```")
        assert not LEADING_CODE_FENCE_PATTERN.match("x = 1\n```")


class TestUnsupportedExprLabels:
    def test_covers_exactly_the_five_scope_introducing_expression_types(self) -> None:
        assert set(UNSUPPORTED_EXPR_LABELS.keys()) == {
            ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda,
        }

    def test_labels_are_non_empty_strings(self) -> None:
        for label in UNSUPPORTED_EXPR_LABELS.values():
            assert isinstance(label, str) and label.strip()


class TestThoughtsMarkers:
    def test_values(self) -> None:
        assert THOUGHTS_START == "==== THOUGHTS [START] ===="
        assert THOUGHTS_END == "==== THOUGHTS [END] ===="

    def test_framed_with_equals_runs(self) -> None:
        for marker in (THOUGHTS_START, THOUGHTS_END):
            assert marker.startswith("====")
            assert marker.endswith("====")

    def test_distinct_from_other_framing_markers(self) -> None:
        markers = (CONTEXT_START, CONTEXT_END, TASK_PROMPT_START, TASK_PROMPT_END, THOUGHTS_START, THOUGHTS_END)

        assert len(set(markers)) == len(markers)

    def test_exported(self) -> None:
        assert "THOUGHTS_START" in agent_constants.__all__
        assert "THOUGHTS_END" in agent_constants.__all__


class TestThinkingInstructionsMarkers:
    def test_values(self) -> None:
        assert THINKING_INSTRUCTIONS_START == "==== USER-DEFINED THINKING INSTRUCTIONS [START] ===="
        assert THINKING_INSTRUCTIONS_END == "==== USER-DEFINED THINKING INSTRUCTIONS [END] ===="

    def test_distinct_from_other_framing_markers(self) -> None:
        markers = (
            CONTEXT_START, CONTEXT_END, TASK_PROMPT_START, TASK_PROMPT_END,
            THOUGHTS_START, THOUGHTS_END,
            THINKING_INSTRUCTIONS_START, THINKING_INSTRUCTIONS_END,
        )

        assert len(set(markers)) == len(markers)

    def test_exported(self) -> None:
        for name in (
            "THINKING_INSTRUCTIONS_START",
            "THINKING_INSTRUCTIONS_END",
            "STOP_EARLY_PROMPT_SECTION",
            "STOP_EARLY_SCHEMA_TEMPLATE",
            "THINKING_DIRECTIVE",
            "REPLY_DIRECTIVE",
        ):
            assert name in agent_constants.__all__


class TestStopEarlySchemaTemplate:
    def test_properties_are_exactly_the_two_wrapper_keys(self) -> None:
        # ScratchpadAgent reads these two keys as literals; pin them here.
        assert set(STOP_EARLY_SCHEMA_TEMPLATE["properties"]) == {
            "thought", "more_thinking_required",
        }

    def test_both_required_and_no_extra_properties(self) -> None:
        assert STOP_EARLY_SCHEMA_TEMPLATE["required"] == ["thought", "more_thinking_required"]
        assert STOP_EARLY_SCHEMA_TEMPLATE["additionalProperties"] is False

    def test_flag_is_a_described_boolean(self) -> None:
        flag = STOP_EARLY_SCHEMA_TEMPLATE["properties"]["more_thinking_required"]
        assert flag["type"] == "boolean"
        assert flag["description"].strip()

    def test_thought_entry_is_a_described_string(self) -> None:
        thought = STOP_EARLY_SCHEMA_TEMPLATE["properties"]["thought"]
        assert thought["type"] == "string"
        assert thought["description"].strip()

    def test_prompt_section_names_both_fields(self) -> None:
        assert "thought" in STOP_EARLY_PROMPT_SECTION
        assert "more_thinking_required" in STOP_EARLY_PROMPT_SECTION

    def test_unsafe_schema_keys_constant_is_gone(self) -> None:
        assert not hasattr(agent_constants, "STOP_EARLY_UNSAFE_SCHEMA_KEYS")
        assert "STOP_EARLY_UNSAFE_SCHEMA_KEYS" not in agent_constants.__all__


class TestDirectives:
    def test_are_distinct_non_empty_strings(self) -> None:
        assert THINKING_DIRECTIVE.strip() and REPLY_DIRECTIVE.strip()
        assert THINKING_DIRECTIVE != REPLY_DIRECTIVE


class TestThinkingRoundsParamRemoved:
    def test_constant_is_gone(self) -> None:
        assert not hasattr(agent_constants, "THINKING_ROUNDS_PARAM")
        assert "THINKING_ROUNDS_PARAM" not in agent_constants.__all__
