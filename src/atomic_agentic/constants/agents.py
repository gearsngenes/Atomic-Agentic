from __future__ import annotations
import ast
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

THINKING_ROUNDS_PARAM: ParamSpec = ParamSpec(
    name="thinking_rounds", index=0, kind=ParamSpec.KEYWORD_ONLY,
    type=("int",), default=1,
    description="Number of thinking rounds ThinkingAgent runs before replying. Must be a concrete int >= 0; 0 skips thinking entirely and replies immediately."
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
# ScriptAgent code-statement reserved literals
# =============================================================================
# Used by:
# - models/agents/blackboard_models.py: CodeStatement.tool default alias
# - utils/script.py: parse_statement_to_slots hoisting/rhs_assign/return/
#   task-result-reference logic
# - agents/script.py: render_turn/_initialize_task cross-invocation result
#   addressing (TASK_RESULT_PREFIX)
#
# Reserved namespaces: RHS_ASSIGN_ALIAS/RETURN_ALIAS/PY_BUILTIN_ALIAS/
# ATTR_CALL_ALIAS can never be real registered tool aliases; SUB_NAME_PREFIX/
# TASK_RESULT_PREFIX can never be a model-chosen identifier. Enforcement
# points live outside this file's scope.

RHS_ASSIGN_ALIAS = "rhs_assign"
SUB_NAME_PREFIX = "_SUB_"
RETURN_ALIAS = "return"
TASK_RESULT_PREFIX = "task_result_"

PY_BUILTIN_ALIAS = "py_builtin"
"""Reserved CodeStatement.tool sentinel for a rewritten Python builtin call
-- joins RHS_ASSIGN_ALIAS/RETURN_ALIAS as a name no real registered tool
alias may ever equal (see agents/script.py's _validate_tool_alias). Unlike
those two sentinels, a PY_BUILTIN_ALIAS slot dispatches through a real Tool
(agents.tools.builtin_call_tool) instead of skipping dispatch entirely."""

ATTR_CALL_ALIAS = "attr_call"
"""Reserved CodeStatement.tool sentinel for an attribute/method call
(`obj.method(...)`) on a value the plan already holds -- same treatment as
PY_BUILTIN_ALIAS: reserved from real tool aliases, dispatches through a real
Tool (agents.tools.attr_call_tool), counts toward tool-call budget
accounting identically to a registered-tool call."""

EXCLUDED_PY_BUILTINS: frozenset[str] = frozenset(
    {
        # Code execution
        "eval", "exec", "compile", "__build_class__",
        # Module/scope access
        "__import__", "globals", "locals", "vars", "dir",
        # Attribute reflection by string
        "getattr", "setattr", "delattr",
        # Filesystem/stdin I/O
        "open", "input",
        # Process/interpreter control
        "exit", "quit", "breakpoint",
        # Interactive/blocking, site banners
        "help", "copyright", "credits", "license",
        # Return an awaitable -- the one case where "no approved builtin is
        # async" wouldn't hold if these were permitted
        "anext", "aiter",
    }
)
"""Builtins excluded from ScriptAgent's py_builtin dispatch. Checked by both
utils/script.py's rewrite_builtin_calls (parse-time eligibility) and
agents/tools.py's _call_py_builtin (runtime enforcement -- the authoritative
gate; the parse-time check exists so an excluded name gets a specific
regen-repair message instead of falling through to the generic
"unregistered tool" one)."""

# Reserved CodeStatement.kwargs key marking a `**expr` unpack in a real call.
# "**" is never a valid Python identifier, so it can never collide with a
# real keyword argument name -- no validation needed to guarantee this.
KWARGS_UNPACK_KEY = "**"

# Matches a `#`-comment whose content is (case-insensitively) the word
# PAUSE. Matched against a single tokenize COMMENT token's own string (by
# utils/script.py's _find_pause_marker), not scanned over raw multi-line
# text -- tokenize never emits a COMMENT token from inside a string
# literal, so a reasoning-note string containing this same text can never
# be misread as a real marker. Still line-anchored (^\s*) since a token's
# string always starts at its own "#".
PAUSE_PATTERN: re.Pattern[str] = re.compile(r"^\s*#\s*PAUSE\b", re.IGNORECASE | re.MULTILINE)

# Matches a single markdown code fence wrapping the *entire* generation --
# any (or no) language tag on the opening fence line (```python, ```py,
# ```text, a bare ```, ...), not just ```python. Tried first by
# utils/script.py's _strip_code_fence, since a matched pair unambiguously
# marks everything between them as the intended code.
CODE_FENCE_PATTERN: re.Pattern[str] = re.compile(r"^\s*```[^\n]*\n(.*?)\n?```\s*$", re.DOTALL)

# Fallback for when CODE_FENCE_PATTERN doesn't match (a model emitting only
# one side, unmatched) -- each stripped independently, never a fence
# appearing mid-text (that's a real structural problem, left for ast.parse
# to reject on its own terms). Same fence-line shape as CODE_FENCE_PATTERN.
# Used by utils/script.py's _strip_code_fence.
LEADING_CODE_FENCE_PATTERN: re.Pattern[str] = re.compile(r"^[ \t]*```[^\n]*\n")
TRAILING_CODE_FENCE_PATTERN: re.Pattern[str] = re.compile(r"\n[ \t]*```[ \t]*$")

# Matches a dunder-shaped attribute name (`__class__`, `__globals__`, ...).
# Rejected unconditionally by utils/script.py's _hoist_calls (for a bare
# `ast.Attribute` anywhere in an expression) and _build_call_slot (for a
# method-call's own method name, the one position _hoist_calls itself never
# scans) -- closes the classic attribute-chaining sandbox-escape class
# (`().__class__.__bases__[0].__subclasses__()`-style), which the
# `{"__builtins__": {}}` eval lockout alone does not defend against.
DUNDER_ATTRIBUTE_PATTERN: re.Pattern[str] = re.compile(r"^__.*__$")

# Expression node types utils/script.py's _hoist_calls rejects unconditionally
# (see its own docstring) -- each introduces a local binding scope neither
# that module nor extract_identifiers has any awareness of.
UNSUPPORTED_EXPR_LABELS: dict[type, str] = {
    ast.ListComp: "list comprehension",
    ast.SetComp: "set comprehension",
    ast.DictComp: "dict comprehension",
    ast.GeneratorExp: "generator expression",
    ast.Lambda: "lambda",
}

FINAL_ROUND_WARNING = (
    "This is your FINAL planning round -- you must complete the entire "
    "task now. Do not write # PAUSE."
)
"""Appended (space-separated) to a ScriptAgent continuation instruction when
ScriptAgent._is_final_round(task) is true -- shared by
_render_task_messages' round-1 and continuation branches so the two call
sites can never drift in wording."""


__all__ = [
    # Conversation storage
    "DEFAULT_CONVERSATION_NAME",
    "CONVERSATION_NAME_PATTERN",
    "TRAILING_FORK_INDEX_PATTERN",
    # Framework-reserved parameters
    "RUN_ID_PARAM",
    "THINKING_ROUNDS_PARAM",
    # ScriptAgent code-statement reserved literals
    "RHS_ASSIGN_ALIAS",
    "SUB_NAME_PREFIX",
    "RETURN_ALIAS",
    "TASK_RESULT_PREFIX",
    "PY_BUILTIN_ALIAS",
    "ATTR_CALL_ALIAS",
    "EXCLUDED_PY_BUILTINS",
    "KWARGS_UNPACK_KEY",
    "PAUSE_PATTERN",
    "CODE_FENCE_PATTERN",
    "LEADING_CODE_FENCE_PATTERN",
    "TRAILING_CODE_FENCE_PATTERN",
    "DUNDER_ATTRIBUTE_PATTERN",
    "UNSUPPORTED_EXPR_LABELS",
    "FINAL_ROUND_WARNING",
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