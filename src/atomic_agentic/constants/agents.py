from __future__ import annotations
import re
from ..models.parameters import ParamSpec

# =============================================================================
# Agent framework-reserved parameters
# =============================================================================
# RUN_ID_PARAM is the canonical ParamSpec grafted onto every Agent subclass
# schema via Agent.get_reserved_parameters(). It is defined here so that the
# reserved-name reconciliation machinery in Agent.__init__ can compare
# caller-declared params against the authoritative definition without
# re-constructing it inline.

RUN_ID_PARAM: ParamSpec = ParamSpec(
    name="run_id", index=0, kind=ParamSpec.KEYWORD_ONLY,
    type=("None", "str"), default=None,
    description="Optional UUID hexstring used to point to a specific historical run of this agent. Do NOT provide natural-language instructions here; this is a reserved parameter to programatically select where in an agent's history to resume execution."
)

# =============================================================================
# ToolAgent LLM-output JSON fields
# =============================================================================
# Used by:
# - agents/planact.py, agents/react.py: generated step validation and BlackboardSlot creation
# - models/agents/blackboard_models.py: BlackboardSlot.from_dict support
#
# These fields are centralized because ToolAgent prompt contracts and parser/
# validator code need to agree on the same LLM-output protocol.
#
# Important runtime contract:
# - "tool" and "args" are the minimum required fields for executable tool calls.
# - "step" is allowed but advisory. Runtime owns the authoritative step index.
#   Prompts may still strongly instruct the LLM to include "step" because that
#   improves output regularity, but parser/runtime code must tolerate omission.


STEP_FIELD = "step"
TOOL_FIELD = "tool"
ARGS_FIELD = "args"
AWAIT_FIELD = "await"
DURATION_FIELD = "duration"
REASON_FIELD = "reason"

RETURN_VALUE_FIELD = "val"


BASE_STEP_FIELDS = frozenset(
    {
        STEP_FIELD,
        TOOL_FIELD,
        ARGS_FIELD,
    }
)

REQUIRED_BASE_STEP_FIELDS = frozenset(
    {
        TOOL_FIELD,
        ARGS_FIELD,
    }
)


PLAN_FIELDS = BASE_STEP_FIELDS | frozenset(
    {
        AWAIT_FIELD,
    }
)

REQUIRED_PLAN_FIELDS = REQUIRED_BASE_STEP_FIELDS


REACT_FIELDS = BASE_STEP_FIELDS | frozenset(
    {
        DURATION_FIELD,
        REASON_FIELD,
    }
)

REQUIRED_REACT_FIELDS = REQUIRED_BASE_STEP_FIELDS | frozenset(
    {
        DURATION_FIELD,
        REASON_FIELD,
    }
)


# Regex-mode-only field set for PlanAct: `reason` is mandatory on every
# generated block in regex-mode (unlike json-mode, where it stays
# deliberately deferred -- see PLAN_FIELDS above), mirroring REACT_FIELDS'
# existing shape rather than inventing a new mechanism.
REGEX_PLAN_FIELDS = PLAN_FIELDS | frozenset({REASON_FIELD})

REQUIRED_REGEX_PLAN_FIELDS = REQUIRED_PLAN_FIELDS | frozenset({REASON_FIELD})


# =============================================================================
# ToolAgent canonical return-tool identity
# =============================================================================
# Used by:
# - agents/toolagent.py: construction and registration of the executable return_tool
# - ToolAgent prompt finalization instructions requiring Tool.ToolAgents.return
# - tests around planner/ReAct final return behavior
#
# Do not put the executable Tool instance here; only the identity literals that
# must stay synchronized with ToolAgent prompt text.


RETURN_TOOL_NAME = "return"
RETURN_TOOL_NAMESPACE = "ToolAgents"
RETURN_TOOL_DESCRIPTION = (
    "Returns the passed-in value. Tool agents should use this to signal completion."
)
RETURN_TOOL_FULL_NAME = (
    f"Tool.{RETURN_TOOL_NAMESPACE}.{RETURN_TOOL_NAME}"
)

# =============================================================================
# ToolAgent regex-mode tag vocabulary (generation_format="regex")
# =============================================================================
# Bracket tag literals for the free-form, regex-tag-delimited generation
# format. Mirrors THOUGHT_MARKER_PATTERN's line-anchored, case-insensitive
# mechanism (see utils/agents.py :: parse_thoughts) but for ToolAgent's own
# step-block grammar. Prompt text teaches only the exact capitalized
# spelling below; parsing is case-insensitive purely as a defensive
# backstop, never a documented affordance.
#
# Used by:
# - utils/agents.py: extract_regex_steps (block-splitting and per-block
#   sub-tag scanning) -- generation-side tags only.
# - agents/toolagent.py: _render_regex_turn_body (render_turn's regex-mode
#   branch) -- render-only tags only, plus RETURN_TAG (reused verbatim for
#   the terminal step). CALL_TAG is not reused there: a rendered turn is
#   a record of what already executed, not an instruction for the model to
#   produce next, so it uses EVENT_TAG instead to avoid conflating the two.
#
# Both a utils/-layer function and an agents/-layer method need the
# generation-side tags, and only constants/ sits below both in the
# dependency layering (doc5 §1) -- do not move these onto ToolAgent as
# class attributes (unlike STEP_REF_PATTERN's precedent, which only ever
# needs to be visible inside agents/toolagent.py itself).

CALL_TAG = "CALL"
RETURN_TAG = "RETURN"
REASON_TAG = "REASON"
AWAIT_TAG = "AWAIT"
DURATION_TAG = "DURATION"
# [CALL]'s tool line is a keyword-only Python call expression,
# tool.id(key=value, ...), parsed via ast.Call -- there is no separate
# ARGS_TAG sub-tag. ARGS_FIELD (the dict key) is unaffected.

# Render-only -- never scanned/parsed by extract_regex_steps, never taught
# in a prompt as something to produce. RUN_ID_TAG is used unwrapped (e.g.
# f"RUN_ID={value}") inside render_turn's "** |CACHE.N| **" header rather
# than as its own bracket tag.
EVENT_TAG = "EVENT"  # render-only label for a fused "tool.id(key=value, ...)" line.
RUN_ID_TAG = "RUN_ID"
RESULT_TAG = "RESULT"
ERROR_TAG = "ERROR"

# Outer block-boundary tags -- each opens (and, for RETURN, is the entirety
# of) one step block. Order in this tuple has no significance beyond
# building the regex alternation in utils/agents.py.
REGEX_BLOCK_TAGS: tuple[str, ...] = (CALL_TAG, RETURN_TAG)

# Inner per-block sub-tags -- order-blind when scanned (see
# extract_regex_steps); canonical prompt-taught order is CALL(args) ->
# REASON -> AWAIT/DURATION, a Pass 2/3 concern, not encoded here.
REGEX_STEP_TAGS: tuple[str, ...] = (REASON_TAG, AWAIT_TAG, DURATION_TAG)

# Maps a recognized sub-tag's bracket spelling to the BlackboardSlot-shaped
# dict key it populates.
REGEX_STEP_TAG_TO_FIELD: dict[str, str] = {
    REASON_TAG: REASON_FIELD,
    AWAIT_TAG: AWAIT_FIELD,
    DURATION_TAG: DURATION_FIELD,
}

# Fixed reason text synthesized for a bare [RETURN] block's desugared slot
# (regex-mode requires `reason` on every slot; a bare [RETURN] block has no
# [REASON] tag of its own to supply one).
RETURN_TOOL_REASON_TEXT = "Final result for the completed task."

# =============================================================================
# Explicit public export list
# =============================================================================
# Keep this explicit so adding local helper names or imports cannot accidentally
# widen the module's public surface.

THOUGHT_CATEGORIES: tuple[str, ...] = (
    "OBSERVATION",
    "QUESTION",
    "CLARIFICATION",
    "ASSUMPTION",
    "REASON",
    "INSTRUCTION",
    "OTHER",
)

THOUGHT_MARKER_PATTERN = re.compile(
    r"^\s*\[(" + "|".join(THOUGHT_CATEGORIES) + r")\]\s*",
    re.MULTILINE | re.IGNORECASE,
)

# Deliberately discriminator-agnostic (matches |STEP.N|/|CACHE.N|/|K.NAME|
# generically) rather than reusing ToolAgent.STEP_REF_PATTERN/etc. -- those
# are compiled on ToolAgent, a layer above constants/, and importing them
# here would be a real circular import per the module dependency topology.
# Used by utils/agents.py's extract_regex_steps to detect a likely-unquoted
# placeholder token inside a failed literal-eval parse.
PLACEHOLDER_SHAPE_PATTERN = re.compile(r"\|[A-Za-z]+\.[^|]+\|")

STOP_THINKING_SENTINEL = "|STOP_THINKING|"

# Wraps a resolved (non-empty) thinking_instructions render into its own
# labeled section around SELF_ASK_PROMPT's {user_thinking_instructions}
# slot (agents/prompts.py). Concatenated around the resolved text, not part
# of any PromptConfig template -- plain literal wrapper text, not a prompt
# itself.
THINKING_ADDITIONAL_INSTRUCTIONS_HEADER = """\
# ADDITIONAL INSTRUCTIONS
Below are additional instructions provided by the user directly for \
tailored thinking instructions, WHILE ABIDING by the rules above.
===Additional Instructions Start===
"""
THINKING_ADDITIONAL_INSTRUCTIONS_FOOTER = "\n===Additional Instructions End===\n"

# Generation-retry feedback framing (generate/validate split). Multi-line
# LLM-facing prose, same precedent as THINKING_ADDITIONAL_INSTRUCTIONS_HEADER/
# FOOTER above -- not a short protocol literal, but still a pure constant.
PLAN_STRUCTURAL_ISSUE_HEADER = (
    "Your last output could not be validated: address the following "
    "structural issue(s) and resubmit your full plan:\n"
)
PLAN_SEMANTIC_ISSUE_HEADER = (
    "Your plan's tags all parsed correctly, but the following semantic "
    "issue(s) must be corrected before it can run:\n"
)
REACT_STRUCTURAL_ISSUE_HEADER = (
    "Your last output contained syntax errors: address the following "
    "and resubmit your next step:\n"
)
REACT_SEMANTIC_ISSUE_HEADER = (
    "Your step's tags all parsed correctly, but the following semantic "
    "issue(s) must be corrected before it can run:\n"
)
EXHAUSTED_ISSUE_PREFIX = "The following issues were identified in the LLM's output:\n"


__all__ = [
    # Framework-reserved parameters
    "RUN_ID_PARAM",
    # LLM step fields
    "STEP_FIELD",
    "TOOL_FIELD",
    "ARGS_FIELD",
    "AWAIT_FIELD",
    "DURATION_FIELD",
    "REASON_FIELD",
    "RETURN_VALUE_FIELD",
    # LLM step schemas
    "BASE_STEP_FIELDS",
    "REQUIRED_BASE_STEP_FIELDS",
    "PLAN_FIELDS",
    "REQUIRED_PLAN_FIELDS",
    "REACT_FIELDS",
    "REQUIRED_REACT_FIELDS",
    "REGEX_PLAN_FIELDS",
    "REQUIRED_REGEX_PLAN_FIELDS",
    # Canonical return tool
    "RETURN_TOOL_NAME",
    "RETURN_TOOL_NAMESPACE",
    "RETURN_TOOL_DESCRIPTION",
    "RETURN_TOOL_FULL_NAME",
    # Regex-mode tag vocabulary
    "CALL_TAG",
    "RETURN_TAG",
    "REASON_TAG",
    "AWAIT_TAG",
    "DURATION_TAG",
    "EVENT_TAG",
    "RUN_ID_TAG",
    "RESULT_TAG",
    "ERROR_TAG",
    "REGEX_BLOCK_TAGS",
    "REGEX_STEP_TAGS",
    "REGEX_STEP_TAG_TO_FIELD",
    "RETURN_TOOL_REASON_TEXT",
]