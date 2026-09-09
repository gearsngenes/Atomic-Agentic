from __future__ import annotations
import re
from ..models.parameters import ParamSpec
from ..constants.core import IDENTIFIER_PATTERN_TEXT

# =============================================================================
# Agent conversation storage
# =============================================================================
# Used by:
# - agents/base.py: Agent's per-conversation storage (_conversations dict,
#   create_conversation, fork_conversation) and its name-shape validation.
#
# "default" is the always-present conversation key -- Agent seeds
# {"default": []} at construction and delete_conversation refuses to ever
# truly remove it, resetting its list to empty instead.

DEFAULT_CONVERSATION_NAME = "default"
"""The always-present conversation key. Never truly removable -- deleting
it resets its list to empty instead."""

CONVERSATION_NAME_PATTERN: re.Pattern[str] = re.compile(
    rf"^(?!.*_\d+$){IDENTIFIER_PATTERN_TEXT}$"
)
"""Valid shape for an explicitly-given conversation name (create_conversation's
`name`, or an explicit `fork_conversation(fork_name=...)`): alphanumeric +
underscore (same as IDENTIFIER_PATTERN), and must NOT end in `_<digits>` --
that suffix shape is reserved for auto-generated fork names, which
guarantees "ends in _<digits> iff auto-generated" holds everywhere."""

TRAILING_FORK_INDEX_PATTERN: re.Pattern[str] = re.compile(r"_(\d+)$")
"""Matches an existing auto-generated numeric suffix so it can be stripped
before a fresh one is appended (prevents suffix accumulation like
`branch_5_7_2`)."""

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


# =============================================================================
# ScriptAgent code-statement reserved literals
# =============================================================================
# Used by:
# - models/agents/blackboard_models.py: CodeStatement.tool default alias
# - utils/script.py: parse_statement_to_slots hoisting/rhs_assign/return/
#   task-result-reference logic
# - agents/script.py: render_turn/_initialize_task cross-invocation result
#   addressing (TASK_RESULT_PREFIX)
#
# Reserved namespaces: RHS_ASSIGN_ALIAS/RETURN_ALIAS can never be real
# registered tool aliases; HOISTED_NAME_PREFIX/TASK_RESULT_PREFIX can never
# be a model-chosen identifier. Enforcement points live outside this file's
# scope.

RHS_ASSIGN_ALIAS = "rhs_assign"
HOISTED_NAME_PREFIX = "_HOIST_"
RETURN_ALIAS = "return"
TASK_RESULT_PREFIX = "task_result_"

# Fallback continuation-note text (agents/script.py's checkpoint-triggered
# reactive continuation): used only when an explicit `# CHECKPOINT` marker is
# not followed by a triple-quoted explanation -- never used for a
# resolution/execution failure, which always surfaces its own real, dynamic
# reason instead of this generic text.
DEFAULT_CONTINUATION_NOTE = (
    "It was deemed necessary to pause code writing and execution to "
    "accurately determine the next steps for completing the task. Review "
    "the work completed so far and continue writing code based on what "
    "you can now reason."
)


__all__ = [
    # Conversation storage
    "DEFAULT_CONVERSATION_NAME",
    "CONVERSATION_NAME_PATTERN",
    "TRAILING_FORK_INDEX_PATTERN",
    # Framework-reserved parameters
    "RUN_ID_PARAM",
    # ScriptAgent code-statement reserved literals
    "RHS_ASSIGN_ALIAS",
    "HOISTED_NAME_PREFIX",
    "RETURN_ALIAS",
    "TASK_RESULT_PREFIX",
    "DEFAULT_CONTINUATION_NOTE",
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
    # Canonical return tool
    "RETURN_TOOL_NAME",
    "RETURN_TOOL_NAMESPACE",
    "RETURN_TOOL_DESCRIPTION",
    "RETURN_TOOL_FULL_NAME",
]