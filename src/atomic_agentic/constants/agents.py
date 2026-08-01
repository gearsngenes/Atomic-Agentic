from __future__ import annotations

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
# Agent2 thinking-phase constants
# =============================================================================
# Used by:
# - agents/base2.py: Agent2.think/_render_system_message/_render_task_messages
# - agents/prompts.py: THINKING_PROMPT/TOOL_THINKING_PROMPT (assembled at import time)
# - utils/agents.py: parse_thoughts
#
# Distinct from the ThinkingAgent LLM-output JSON fields above: these back
# the new line-based, category-tagged thinking format used by Agent2, not
# the old fused-JSON-call format. Both coexist until Pass 7 cutover retires
# the old family and its fields.

STOP_THINKING_SENTINEL = "|STOP_THINKING|"

THOUGHT_CATEGORIES: tuple[str, ...] = (
    "QUESTION",
    "CLARIFICATION",
    "OBSERVATION",
    "REASONING",
    "PLANNING",
    "ASSUMPTION",
    "OTHER",
)

# PromptConfig field names on Agent2._THINK_PROMPT's template. Not
# collision-guarded against caller parameters -- the render context these
# are drawn from is a purpose-built dict, never task.inputs.
THINKING_CONTENT_FIELD = "user_thinking_instructions"
THOUGHTS_PER_ROUND_FIELD = "thoughts_per_round"

# Resolved to a full sentence by the caller (Agent2._render_system_message),
# not a bare number -- mirrors THINKING_CONTENT_FIELD's own pattern of
# pre-resolving varying phrasing (bounded vs. unbounded) outside the
# template rather than branching inside it.
MAX_THINKING_ROUNDS_FIELD = "max_thinking_rounds"

# Hand-typed prose, deliberately NOT built by interpolating
# THOUGHT_CATEGORIES at runtime or import time -- category names/wording
# here and THOUGHT_CATEGORIES above must be kept in sync by hand if a
# category is ever added, renamed, or reworded. Not scaled dynamically
# until there's a real reason to.
#
# {user_thinking_instructions} is bare -- no header/markers of its own in
# this template. Agent2._render_system_message wraps it inline (or
# renders nothing) at render time, so the whole section is invisible when
# the caller supplied no thinking_instructions, rather than leaving an
# empty header. {thoughts_per_round} is the other reserved field.
THINKING_BASE_TEMPLATE = """\
# OBJECTIVE
You are a thinker who analyzes a view of a running/active task and 
produces a list of organized thoughts. This task view can contain a
description of the task itself, prior thoughts or messages,
instructions, and/or thoughts you have given for the current task..

# THINKING OUTPUT FORMAT
Return your thoughts as a block of lines, one thought per line, in this
EXACT format:

[CATEGORY] content
[CATEGORY] content
...

The category MUST be contained in `[` and `]`.

# THOUGHT CATEGORIES
Each thought's category must be exactly one of the following:

- QUESTION: An ambiguity or uncertainty about the task that needs to be
  resolved before proceeding. A question can potentially be answered by
  a follow-up thought of any of the below categories.
- CLARIFICATION: An answer to a question you or a prior thought raised, or an
  enhancement/refinement to the instructions or actions needed for the task.
- OBSERVATION: An emergent truth about the current state of the task --
  something you notice, not something you decide or ask.
- REASONING: Justification or explanation for why something needs to happen,
  or why a particular choice is being made.
- ASSUMPTION: A belief taken as true without confirmation, used only to let
  thinking move forward. Use sparingly -- only when genuinely necessary.
- PLANNING: A thought that helps with determining what to do next, a recommended
  action, or a way to break the task into smaller pieces.
- OTHER: Any thought that does not fit cleanly into the categories above.

# GUIDANCE
Think sparingly. Do not produce a thought for something that is already
obvious or self-explanatory from the task or your prior thoughts -- only
think when it genuinely helps advance or clarify the task.

QUESTION, CLARIFICATION, OBSERVATION, REASONING, and ASSUMPTION thoughts may
each occur multiple times within a single round -- they represent distinct,
independent considerations. PLANNING is different: it represents a
converged decision, not an open consideration. Produce AT MOST ONE PLANNING
thought per round, synthesizing everything you have reasoned through so
far -- typically as the last thought in that round, once you are ready to
commit rather than continue exploring.

# STOP CONDITION
After you produce your thoughts, if you determine no further thinking
is needed, then you MUST signal this by using the literal token
|STOP_THINKING|. Place this signal on its own line after your last thought.
Do NOT include this token if you are not done thinking.

# THOUGHT LIMIT
The block of thoughts you produce per round for a given task must contain
between (AT LEAST) 1 and {thoughts_per_round} (AT MOST) thoughts.

# ROUND LIMIT
{max_thinking_rounds}

{user_thinking_instructions}"""

# Wraps a resolved (non-empty) thinking_instructions render into its own
# labeled section -- see THINKING_BASE_TEMPLATE's note above. Concatenated
# around the resolved text, not part of any PromptConfig template.
THINKING_ADDITIONAL_INSTRUCTIONS_HEADER = """\
# ADDITIONAL INSTRUCTIONS
Below are additional instructions provided by the user directly for \
tailored thinking instructions, WHILE ABIDING by the rules above.
===Additional Instructions Start===
"""
THINKING_ADDITIONAL_INSTRUCTIONS_FOOTER = "\n===Additional Instructions End===\n"

# Sketch only -- not yet wired to any class (ToolAgent2 doesn't exist yet).
# Concatenated onto THINKING_BASE_TEMPLATE to form a tool-aware "think"
# prompt for that family, per prompts.py's TOOL_THINKING_PROMPT.
THINKING_TOOL_RESOURCES_BLOCK = """
# ADDITIONAL RESOURCES
You have access to the following tools, constants, and usage limits while
thinking about this task. Reference them in your thoughts as needed, but do
not attempt to invoke tools during thinking -- execution happens in a
separate phase.

Available tools:
{TOOLS}

Available constants:
{CONSTANTS}

Tool call budget (non-return calls): {TOOL_CALLS_LIMIT}
"""


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
    # Agent2 thinking-phase constants
    "STOP_THINKING_SENTINEL",
    "THOUGHT_CATEGORIES",
    "THINKING_CONTENT_FIELD",
    "THOUGHTS_PER_ROUND_FIELD",
    "MAX_THINKING_ROUNDS_FIELD",
    "THINKING_BASE_TEMPLATE",
    "THINKING_ADDITIONAL_INSTRUCTIONS_HEADER",
    "THINKING_ADDITIONAL_INSTRUCTIONS_FOOTER",
    "THINKING_TOOL_RESOURCES_BLOCK",
    # Canonical return tool
    "RETURN_TOOL_NAME",
    "RETURN_TOOL_NAMESPACE",
    "RETURN_TOOL_DESCRIPTION",
    "RETURN_TOOL_FULL_NAME",
]