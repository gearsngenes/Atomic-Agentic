from __future__ import annotations

import ast
import builtins

import atomic_agentic.constants.agents as agent_constants

from atomic_agentic.constants.agents import (
    ATTR_CALL_ALIAS,
    BUILTINS_PREAMBLE,
    CODE_FENCE_PATTERN,
    EMBEDDED_CODE_FENCE_PATTERN,
    DUNDER_ATTRIBUTE_PATTERN,
    ALLOWED_PY_BUILTINS,
    FRAME_SUBJECTS,
    KWARGS_UNPACK_KEY,
    LEADING_CODE_FENCE_PATTERN,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    REPLY_DIRECTIVE,
    RHS_ASSIGN_ALIAS,
    STOP_EARLY_PROMPT_SECTION,
    STOP_EARLY_SCHEMA_TEMPLATE,
    SUB_NAME_PREFIX,
    TASK_RESULT_PREFIX,
    THINKING_DIRECTIVE,
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


class TestAllowedPyBuiltins:
    def test_is_a_non_empty_frozenset(self) -> None:
        assert isinstance(ALLOWED_PY_BUILTINS, frozenset)
        assert ALLOWED_PY_BUILTINS

    def test_every_name_is_a_real_builtin(self) -> None:
        for name in ALLOWED_PY_BUILTINS:
            assert hasattr(builtins, name), name

    def test_excludes_dangerous_and_underscore_builtins(self) -> None:
        for name in ("eval", "exec", "open", "getattr", "object", "super"):
            assert name not in ALLOWED_PY_BUILTINS
        assert not [n for n in ALLOWED_PY_BUILTINS if n.startswith("_")]
        for name in dir(builtins):
            if name.startswith("_"):
                assert name not in ALLOWED_PY_BUILTINS

    def test_preamble_lists_exactly_the_allowlist(self) -> None:
        for name in ALLOWED_PY_BUILTINS:
            assert name in BUILTINS_PREAMBLE
        assert "eval" not in BUILTINS_PREAMBLE


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

    def test_embedded_pattern_finds_each_fenced_block_with_prose_around_it(self) -> None:
        text = "intro\n```python\nx = 1\n```\nmiddle\n```\ny = 2\n```\noutro"
        assert EMBEDDED_CODE_FENCE_PATTERN.findall(text) == ["x = 1", "y = 2"]

    def test_embedded_pattern_needs_the_closing_fence_on_its_own_line(self) -> None:
        assert EMBEDDED_CODE_FENCE_PATTERN.findall("```python\nx = '```'\n```") == ["x = '```'"]

    def test_leading_code_fence_pattern_matches_only_leading_half(self) -> None:
        assert LEADING_CODE_FENCE_PATTERN.match("```python\nx = 1")
        assert not TRAILING_CODE_FENCE_PATTERN.match("```python\nx = 1")

    def test_trailing_code_fence_pattern_matches_only_trailing_half(self) -> None:
        assert TRAILING_CODE_FENCE_PATTERN.search("x = 1\n```")
        assert not LEADING_CODE_FENCE_PATTERN.match("x = 1\n```")


class TestUnsupportedExprLabels:
    def test_covers_exactly_the_rejected_binding_introducing_expression_types(self) -> None:
        assert set(UNSUPPORTED_EXPR_LABELS.keys()) == {
            ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.Lambda,
            ast.NamedExpr, ast.Yield, ast.YieldFrom,
        }

    def test_labels_are_non_empty_strings(self) -> None:
        for label in UNSUPPORTED_EXPR_LABELS.values():
            assert isinstance(label, str) and label.strip()


class TestFrameSubjects:
    def test_values_and_order(self) -> None:
        assert FRAME_SUBJECTS == (
            "RUNTIME CONTEXT",
            "USER TASK PROMPT",
            "THOUGHTS",
            "STEPS",
            "TOOL-CALLING INSTRUCTIONS",
            "AVAILABLE TOOLS",
            "USER-DEFINED INSTRUCTIONS",
            "USER-DEFINED THINKING INSTRUCTIONS",
        )

    def test_is_a_tuple_of_unique_uppercase_single_line_strings(self) -> None:
        assert isinstance(FRAME_SUBJECTS, tuple)
        assert len(set(FRAME_SUBJECTS)) == len(FRAME_SUBJECTS)
        for subject in FRAME_SUBJECTS:
            assert isinstance(subject, str)
            assert subject == subject.upper()
            assert subject.strip() == subject and "\n" not in subject

    def test_removed_marker_constants_are_gone(self) -> None:
        for name in (
            "CONTEXT_START", "CONTEXT_END", "TASK_PROMPT_START", "TASK_PROMPT_END",
            "THOUGHTS_START", "THOUGHTS_END",
            "THINKING_INSTRUCTIONS_START", "THINKING_INSTRUCTIONS_END",
        ):
            assert not hasattr(agent_constants, name)
            assert name not in agent_constants.__all__

    def test_exported(self) -> None:
        for name in (
            "FRAME_SUBJECTS",
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

    def test_name_only_listed_frame_subjects(self) -> None:
        # The directives refer to frames by subject; keep text, call sites and
        # FRAME_SUBJECTS in sync.
        for directive in (THINKING_DIRECTIVE, REPLY_DIRECTIVE):
            for subject in ("USER TASK PROMPT", "THOUGHTS"):
                assert subject in directive
                assert subject in FRAME_SUBJECTS


class TestThinkingRoundsParamRemoved:
    def test_constant_is_gone(self) -> None:
        assert not hasattr(agent_constants, "THINKING_ROUNDS_PARAM")
        assert "THINKING_ROUNDS_PARAM" not in agent_constants.__all__
