from __future__ import annotations

import re

from ..core.constants import IDENTIFIER_PATTERN_TEXT

# =============================================================================
# ToolAgent placeholder reference syntax
# =============================================================================
# Used by:
# - agents/tool_agents.py: placeholder dependency extraction and resolution
# - tests around PlanAct/ReAct placeholder parsing and cache references
#
# Runtime references:
# - <<__sN__>> references current-run step N
# - <<__cN__>> references cached blackboard step N
# - <<__k.NAME__>> references registered ToolAgent constant NAME
#
# This belongs in agents because it is ToolAgent runtime protocol, not a core
# AtomicInvokable / ParamSpec primitive.


STEP_REF_PATTERN_TEXT = r"<<__s(\d+)__>>"
CACHE_REF_PATTERN_TEXT = r"<<__c(\d+)__>>"
CONST_REF_PATTERN_TEXT = rf"<<__k\.({IDENTIFIER_PATTERN_TEXT})__>>"

STEP_REF_PATTERN: re.Pattern[str] = re.compile(
    STEP_REF_PATTERN_TEXT
)
CACHE_REF_PATTERN: re.Pattern[str] = re.compile(
    CACHE_REF_PATTERN_TEXT
)
CONST_REF_PATTERN: re.Pattern[str] = re.compile(
    CONST_REF_PATTERN_TEXT
)


# =============================================================================
# ToolAgent prompt template fields
# =============================================================================
# Used by:
# - agents/tool_agents.py: ToolAgent.role_prompt formatting and validation
# - ToolAgent-specific prompt templates
#
# These are str.format field names, not placeholder reference tokens.


TOOLS_FIELD = "TOOLS"
LIMIT_FIELD = "TOOL_CALLS_LIMIT"
CONSTANTS_FIELD = "CONSTANTS"

REQUIRED_PROMPT_FIELDS = frozenset(
    {
        TOOLS_FIELD,
        LIMIT_FIELD,
        CONSTANTS_FIELD,
    }
)


# =============================================================================
# ToolAgent LLM-output JSON fields
# =============================================================================
# Used by:
# - agents/tool_agents.py: generated step validation and BlackboardSlot creation
# - agents/data_classes.py: BlackboardSlot.from_dict support
#
# These fields are centralized because ToolAgent prompt contracts and parser/
# validator code need to agree on the same LLM-output protocol.


STEP_FIELD = "step"
TOOL_FIELD = "tool"
ARGS_FIELD = "args"
AWAIT_FIELD = "await"
DURATION_FIELD = "duration"
DESCRIPTION_FIELD = "description"

RETURN_VALUE_FIELD = "val"

PLAN_FIELDS = frozenset(
    {
        STEP_FIELD,
        TOOL_FIELD,
        ARGS_FIELD,
        AWAIT_FIELD,
    }
)

REQUIRED_PLAN_FIELDS = frozenset(
    {
        TOOL_FIELD,
        ARGS_FIELD,
    }
)

REACT_FIELDS = frozenset(
    {
        STEP_FIELD,
        TOOL_FIELD,
        ARGS_FIELD,
        DURATION_FIELD,
        DESCRIPTION_FIELD,
    }
)

REQUIRED_REACT_FIELDS = frozenset(
    {
        TOOL_FIELD,
        ARGS_FIELD,
        DURATION_FIELD,
        DESCRIPTION_FIELD,
    }
)


# =============================================================================
# ToolAgent canonical return-tool identity
# =============================================================================
# Used by:
# - agents/tool_agents.py: construction and registration of the executable return_tool
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


__all__ = [
    # Placeholder refs
    "STEP_REF_PATTERN_TEXT",
    "CACHE_REF_PATTERN_TEXT",
    "CONST_REF_PATTERN_TEXT",
    "STEP_REF_PATTERN",
    "CACHE_REF_PATTERN",
    "CONST_REF_PATTERN",
    # Prompt fields
    "TOOLS_FIELD",
    "LIMIT_FIELD",
    "CONSTANTS_FIELD",
    "REQUIRED_PROMPT_FIELDS",
    # LLM step fields
    "STEP_FIELD",
    "TOOL_FIELD",
    "ARGS_FIELD",
    "AWAIT_FIELD",
    "DURATION_FIELD",
    "DESCRIPTION_FIELD",
    "RETURN_VALUE_FIELD",
    # LLM step schemas
    "PLAN_FIELDS",
    "REQUIRED_PLAN_FIELDS",
    "REACT_FIELDS",
    "REQUIRED_REACT_FIELDS",
    # Canonical return tool
    "RETURN_TOOL_NAME",
    "RETURN_TOOL_NAMESPACE",
    "RETURN_TOOL_DESCRIPTION",
    "RETURN_TOOL_FULL_NAME",
]