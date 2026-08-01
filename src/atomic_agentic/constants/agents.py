from __future__ import annotations
import re
from ..constants.agents import THOUGHT_CATEGORIES
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
    type="str | None", default=None,
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
DESCRIPTION_FIELD = "description"

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
        DESCRIPTION_FIELD,
    }
)

REQUIRED_REACT_FIELDS = REQUIRED_BASE_STEP_FIELDS | frozenset(
    {
        DURATION_FIELD,
        DESCRIPTION_FIELD,
    }
)


# =============================================================================
# ThinkingAgent LLM-output JSON fields
# =============================================================================
# Used by:
# - agents/thinking.py: shared reply-phase thought-snapshot rendering
# - agents/selfask.py, agents/planask.py (Pass 2/3): per-round JSON schemas
#
# THOUGHT_FIELDS is SelfAskAgent's fused per-round call schema (all four
# keys required every round). PLANNED_QUESTION_FIELDS is PlanAskAgent's
# upfront batch-item schema (no keep_thinking -- completion there is
# structural, not self-declared).


OBSERVATION_FIELD = "observation"
QUESTION_FIELD = "question"
ANSWER_FIELD = "answer"
KEEP_THINKING_FIELD = "keep_thinking"


THOUGHT_FIELDS = frozenset(
    {
        OBSERVATION_FIELD,
        QUESTION_FIELD,
        ANSWER_FIELD,
        KEEP_THINKING_FIELD,
    }
)

PLANNED_QUESTION_FIELDS = frozenset(
    {
        OBSERVATION_FIELD,
        QUESTION_FIELD,
        ANSWER_FIELD,
    }
)


# =============================================================================
# ThinkingAgent reserved prompt-template field
# =============================================================================
# Distinct category from the JSON-output fields above: this is a PromptConfig
# TEMPLATE placeholder name, not an LLM-output JSON key. A subclass's
# thinking-phase prompt (SelfAskAgent's "thinking"; PlanAskAgent's
# "ask_questions"/"answer_question") may reference {role_description} in its
# own internally-rendered context; ThinkingAgent.__init__ raises if any other
# parameter source (pre_invoke, post_invoke, role_prompt -- the only
# extra_parameters sources that exist) also declares a parameter with this
# name.

ROLE_DESCRIPTION_FIELD = "role_description"


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
# Explicit public export list
# =============================================================================
# Keep this explicit so adding local helper names or imports cannot accidentally
# widen the module's public surface.

THOUGHT_CATEGORIES: tuple[str, ...] = (
    "QUESTION",
    "CLARIFICATION",
    "OBSERVATION",
    "REASONING",
    "PLANNING",
    "ASSUMPTION",
    "OTHER",
)

THOUGHT_MARKER_PATTERN = re.compile(
    r"^\s*\[(" + "|".join(THOUGHT_CATEGORIES) + r")\]\s*",
    re.MULTILINE | re.IGNORECASE,
)

STOP_THINKING_SENTINEL = "|STOP_THINKING|"


__all__ = [
    # Framework-reserved parameters
    "RUN_ID_PARAM",
    # LLM step fields
    "STEP_FIELD",
    "TOOL_FIELD",
    "ARGS_FIELD",
    "AWAIT_FIELD",
    "DURATION_FIELD",
    "DESCRIPTION_FIELD",
    "RETURN_VALUE_FIELD",
    # LLM step schemas
    "BASE_STEP_FIELDS",
    "REQUIRED_BASE_STEP_FIELDS",
    "PLAN_FIELDS",
    "REQUIRED_PLAN_FIELDS",
    "REACT_FIELDS",
    "REQUIRED_REACT_FIELDS",
    # ThinkingAgent LLM-output fields
    "OBSERVATION_FIELD",
    "QUESTION_FIELD",
    "ANSWER_FIELD",
    "KEEP_THINKING_FIELD",
    "THOUGHT_FIELDS",
    "PLANNED_QUESTION_FIELDS",
    # ThinkingAgent reserved prompt-template field
    "ROLE_DESCRIPTION_FIELD",
    # Canonical return tool
    "RETURN_TOOL_NAME",
    "RETURN_TOOL_NAMESPACE",
    "RETURN_TOOL_DESCRIPTION",
    "RETURN_TOOL_FULL_NAME",
]