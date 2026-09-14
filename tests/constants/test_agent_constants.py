from __future__ import annotations

import ast

from atomic_agentic.constants.agents import (
    ATTR_CALL_ALIAS,
    CODE_FENCE_PATTERN,
    DUNDER_ATTRIBUTE_PATTERN,
    EXCLUDED_PY_BUILTINS,
    FINAL_ROUND_WARNING,
    KWARGS_UNPACK_KEY,
    LEADING_CODE_FENCE_PATTERN,
    PAUSE_PATTERN,
    PY_BUILTIN_ALIAS,
    RETURN_ALIAS,
    RHS_ASSIGN_ALIAS,
    RUN_ID_PARAM,
    SUB_NAME_PREFIX,
    TASK_RESULT_PREFIX,
    TRAILING_CODE_FENCE_PATTERN,
    UNSUPPORTED_EXPR_LABELS,
)


class TestRunIdParam:
    def test_run_id_param_has_non_empty_description(self) -> None:
        assert isinstance(RUN_ID_PARAM.description, str)
        assert RUN_ID_PARAM.description.strip() != ""

    def test_run_id_param_identity(self) -> None:
        assert RUN_ID_PARAM.name == "run_id"
        assert RUN_ID_PARAM.kind == "KEYWORD_ONLY"
        assert RUN_ID_PARAM.default is None


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


class TestPausePattern:
    def test_matches_pause_comment_case_insensitively(self) -> None:
        assert PAUSE_PATTERN.match("# PAUSE")
        assert PAUSE_PATTERN.match("#PAUSE")
        assert PAUSE_PATTERN.match("# pause")

    def test_does_not_match_trailing_non_pause_comment(self) -> None:
        assert not PAUSE_PATTERN.match("x = 1  # not a pause")


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


class TestFinalRoundWarning:
    def test_is_a_non_empty_string_mentioning_pause(self) -> None:
        assert isinstance(FINAL_ROUND_WARNING, str)
        assert FINAL_ROUND_WARNING.strip()
        assert "# PAUSE" in FINAL_ROUND_WARNING
