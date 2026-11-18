from __future__ import annotations

import importlib

import pytest

import atomic_agentic.constants.agents as agent_constants
import atomic_agentic.constants.prompts as prompt_constants
from atomic_agentic.constants.agents import ALLOWED_PY_BUILTINS, FRAME_SUBJECTS
from atomic_agentic.constants.prompts import (
    BUILTINS_PREAMBLE,
    DEFAULT_INSTRUCTIONS,
    DEFAULT_PRE_INVOKE,
    DEFAULT_THINKING_INSTRUCTIONS,
    REPLY_DIRECTIVE,
    STOP_EARLY_PROMPT_SECTION,
    THINKING_DIRECTIVE,
)

EXPECTED_NAMES = {
    "CONTEXT_PREAMBLE",
    "TOOLS_PREAMBLE",
    "INSTRUCTIONS_PRECEDENCE",
    "BUILTINS_PREAMBLE",
    "THINKING_DIRECTIVE",
    "REPLY_DIRECTIVE",
    "STOP_EARLY_PROMPT_SECTION",
    "DEFAULT_INSTRUCTIONS",
    "DEFAULT_THINKING_INSTRUCTIONS",
    "DEFAULT_PRE_INVOKE",
    "PLANNER_PROMPT",
    "REACT_PROMPT",
    "ONESHOT_PLANNER_PROMPT",
}


class TestModuleSurface:
    def test_all_is_exactly_the_expected_names(self) -> None:
        assert set(prompt_constants.__all__) == EXPECTED_NAMES
        assert len(prompt_constants.__all__) == len(EXPECTED_NAMES)

    def test_every_exported_name_is_a_non_empty_string(self) -> None:
        for name in prompt_constants.__all__:
            value = getattr(prompt_constants, name)
            assert isinstance(value, str) and value.strip(), name

    def test_moved_names_are_gone_from_agent_constants(self) -> None:
        for name in EXPECTED_NAMES:
            assert not hasattr(agent_constants, name), name
            assert name not in agent_constants.__all__, name

    def test_old_prompts_module_is_gone(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("atomic_agentic.agents.prompts")


class TestBuiltinsPreamble:
    def test_preamble_lists_exactly_the_allowlist(self) -> None:
        for name in ALLOWED_PY_BUILTINS:
            assert name in BUILTINS_PREAMBLE
        assert "eval" not in BUILTINS_PREAMBLE


class TestStopEarlyPromptSection:
    def test_prompt_section_names_both_fields(self) -> None:
        assert "thought" in STOP_EARLY_PROMPT_SECTION
        assert "more_thinking_required" in STOP_EARLY_PROMPT_SECTION


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


class TestSlotDefaults:
    def test_pre_invoke_default_is_the_identity_template(self) -> None:
        assert DEFAULT_PRE_INVOKE == "{prompt}"

    def test_instruction_defaults_are_distinct_and_non_blank(self) -> None:
        assert DEFAULT_INSTRUCTIONS.strip() and DEFAULT_THINKING_INSTRUCTIONS.strip()
        assert DEFAULT_INSTRUCTIONS != DEFAULT_THINKING_INSTRUCTIONS
