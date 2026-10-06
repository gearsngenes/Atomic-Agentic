from __future__ import annotations
import ast
import re
from typing import Any
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

# RETURN_VALUE_FIELD is the kwargs key PlanActAgent uses for its
# synthesized RETURN_ALIAS call's resolved value (utils/sigils.py's
# parse_generation/utils/agents.py's resolve_statement_args), and the
# parameter name return_tool's own real signature uses (agents/tools.py's
# `_return(val)`).
RETURN_VALUE_FIELD = "val"


# =============================================================================
# ReActAgent canonical return-tool identity
# =============================================================================
# Used by:
# - agents/react.py: construction and registration of the executable return_tool
#   (PlanActAgent never registers this -- its return value comes from the
#   synthesized RETURN_ALIAS call instead)
# - REACT_PROMPT's finalization instructions requiring Tool.ToolAgents.return
# - tests around planner/ReAct final return behavior
#
# Do not put the executable Tool instance here; only the identity literals that
# must stay synchronized with REACT_PROMPT text.


RETURN_TOOL_NAME = "return"
RETURN_TOOL_NAMESPACE = "ToolAgents"
RETURN_TOOL_DESCRIPTION = (
    "Returns the passed-in value. Tool agents should use this to signal completion."
)
RETURN_TOOL_FULL_NAME = (
    f"Tool.{RETURN_TOOL_NAMESPACE}.{RETURN_TOOL_NAME}"
)

# =============================================================================
# ScriptActAgent code-statement reserved literals
# =============================================================================
# Used by:
# - models/agents/blackboard_models.py: ToolStatement.tool default alias
# - utils/script.py: parse_statement_to_slots hoisting/rhs_assign/return/
#   task-result-reference logic
# - agents/toolagent.py: ToolAgent.render_turn cross-invocation result
#   addressing (TASK_RESULT_PREFIX); agents/scriptact.py: _initialize_task
#   seeds the same addressing into task.cache
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
"""Reserved ToolStatement.tool sentinel for a rewritten Python builtin call
-- joins RHS_ASSIGN_ALIAS/RETURN_ALIAS as a name no real registered tool
alias may ever equal (see agents/toolagent.py's ToolAgent._validate_tool_alias,
extended by agents/scriptact.py's ScriptActAgent._validate_effective_tool_id).
Unlike
those two sentinels, a PY_BUILTIN_ALIAS slot dispatches through a real Tool
(agents.tools.builtin_call_tool) instead of skipping dispatch entirely."""

ATTR_CALL_ALIAS = "attr_call"
"""Reserved ToolStatement.tool sentinel for an attribute/method call
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
"""Builtins excluded from ScriptActAgent's py_builtin dispatch. Checked by both
utils/script.py's rewrite_builtin_calls (parse-time eligibility) and
agents/tools.py's _call_py_builtin (runtime enforcement -- the authoritative
gate; the parse-time check exists so an excluded name gets a specific
regen-repair message instead of falling through to the generic
"unregistered tool" one)."""

# Reserved ToolStatement.kwargs key marking a `**expr` unpack in a real call.
# "**" is never a valid Python identifier, so it can never collide with a
# real keyword argument name -- no validation needed to guarantee this.
KWARGS_UNPACK_KEY = "**"

# Matches a single markdown code fence wrapping the *entire* generation --
# any (or no) language tag on the opening fence line (```python, ```py,
# ```text, a bare ```, ...), not just ```python. Tried first by
# utils/agents.py's strip_code_fence (ScriptActAgent's own code-statement
# parsing), since a matched pair unambiguously marks everything between
# them as the intended code.
CODE_FENCE_PATTERN: re.Pattern[str] = re.compile(r"^\s*```[^\n]*\n(.*?)\n?```\s*$", re.DOTALL)

# Fallback for when CODE_FENCE_PATTERN doesn't match (a model emitting only
# one side, unmatched) -- each stripped independently, never a fence
# appearing mid-text (that's a real structural problem, left for ast.parse
# to reject on its own terms). Same fence-line shape as CODE_FENCE_PATTERN.
# Used by utils/agents.py's strip_code_fence.
LEADING_CODE_FENCE_PATTERN: re.Pattern[str] = re.compile(r"^[ \t]*```[^\n]*\n")
TRAILING_CODE_FENCE_PATTERN: re.Pattern[str] = re.compile(r"\n[ \t]*```[ \t]*$")

# Matches a dunder-shaped attribute name (`__class__`, `__globals__`, ...).
# Rejected unconditionally by utils/agents.py's reject_unsupported_forms
# (for a bare `ast.Attribute` anywhere in an expression -- ScriptActAgent's
# own code-statement grammar) and utils/script.py's _build_call_slot (for
# a method-call's own method name, the one position
# reject_unsupported_forms's own walk never scans) -- closes the classic
# attribute-chaining sandbox-escape class
# (`().__class__.__bases__[0].__subclasses__()`-style), which the
# `{"__builtins__": {}}` eval lockout alone does not defend against.
DUNDER_ATTRIBUTE_PATTERN: re.Pattern[str] = re.compile(r"^__.*__$")

# Expression node types utils/agents.py's reject_unsupported_forms rejects
# unconditionally (see its own docstring) -- each introduces a local
# binding scope neither that function nor extract_identifiers has any
# awareness of.
UNSUPPORTED_EXPR_LABELS: dict[type, str] = {
    ast.ListComp: "list comprehension",
    ast.SetComp: "set comprehension",
    ast.DictComp: "dict comprehension",
    ast.GeneratorExp: "generator expression",
    ast.Lambda: "lambda",
}

# PlanActAgent's own output_structure schema (agent-taxonomy `planact-
# rewrite` design record) -- a one-shot planner has no continuation round
# to defer to, so there is no `remaining_work`-style field at all. `return`
# is unconditionally in `required` (always present), but its *value* may
# still legitimately be `null` when the task has nothing meaningful to hand
# back -- presence and nullability are separate concerns.
PLANACT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "plan", "return"],
    "properties": {
        "summary": {
            "type": "string",
            "description": "Briefly describe the work this plan accomplishes.",
        },
        "plan": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["call", "arguments", "result_name"],
                "properties": {
                    "call": {"type": "string", "enum": []},
                    "arguments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "value"],
                            "properties": {
                                "name": {"type": ["string", "null"]},
                                "value": {
                                    "type": ["number", "boolean", "null", "string"],
                                    "description": (
                                        "A literal value (any JSON scalar), or a "
                                        "string. A string that is *exactly* '$name' "
                                        "(nothing else) refers to an earlier "
                                        "result_name or a K_/task_result_ value, "
                                        "substituted with its real value and type. "
                                        "A '$name' appearing inside a longer string "
                                        "is spliced in as text (stringified) at that "
                                        "position. A '$name' that doesn't match "
                                        "anything is left as literal text, sigil "
                                        "included -- not an error. To build a "
                                        "list/tuple/set or dict, call "
                                        "make_sequence/make_dict instead of writing "
                                        "a container here."
                                    ),
                                },
                            },
                        },
                    },
                    "result_name": {"type": ["string", "null"]},
                },
            },
        },
        "return": {
            "type": ["number", "boolean", "null", "string"],
            "description": (
                "Follows the same rules as an argument's value (a literal, "
                "or a '$name' reference/interpolation). Always required in "
                "the response -- there is no continuation round to defer "
                "to, so decide it now. 'null' is a legitimate answer when "
                "the task genuinely has nothing to hand back; it does not "
                "mean 'come back later'."
            ),
        },
    },
}

# ReActAgent's own output_structure schema (agent-taxonomy `reactagent-
# models` design record) -- the flattened form of PLANACT_OUTPUT_SCHEMA's
# per-`plan`-item shape: `call`/`arguments`/`result_name` promoted to the
# top level directly, no `plan` array wrapper (exactly one call per round,
# never a list of them) and no `remaining_work` field (a per-step family has
# no continuation-round concept for a model to signal -- termination is
# simply calling the registered return tool, an ordinary `call` value, not a
# separate field). Field order matters (constrained decoding assigns zero
# probability to a token generated for a field declared after the one that
# would need it) -- `summary` first, same reasoning-before-decision
# principle as PLANACT_OUTPUT_SCHEMA.
REACT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "call", "arguments", "result_name"],
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "Briefly describe what this one call accomplishes and why "
                "it's needed now."
            ),
        },
        "call": {"type": "string", "enum": []},
        "arguments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "value"],
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "value": {
                        "type": ["number", "boolean", "null", "string"],
                        "description": (
                            "A literal value (any JSON scalar), or a "
                            "string. A string that is *exactly* '$name' "
                            "(nothing else) refers to an earlier "
                            "result_name or a K_/task_result_ value, "
                            "substituted with its real value and type. "
                            "A '$name' appearing inside a longer string "
                            "is spliced in as text (stringified) at that "
                            "position. A '$name' that doesn't match "
                            "anything is left as literal text, sigil "
                            "included -- not an error. To build a "
                            "list/tuple/set or dict, call "
                            "make_sequence/make_dict instead of writing "
                            "a container here."
                        ),
                    },
                },
            },
        },
        "result_name": {"type": ["string", "null"]},
    },
}

# Matches a `$`-sigil reference inside a PlanActAgent-/ReActAgent-generated
# `value`/`return` string -- used exclusively inside utils/sigils.py's
# translate_calls (candidate-token scanning during JSON -> ast translation:
# `SIGIL_REF_PATTERN.fullmatch(s)` for a whole-string reference,
# `SIGIL_REF_PATTERN.finditer(s)` for every embedded occurrence), not a
# resolve-time mechanism anymore. No anchors baked into the text itself
# (fullmatch anchors on its own, exactly like this file's own
# IDENTIFIER_PATTERN precedent elsewhere).
SIGIL_REF_PATTERN: re.Pattern[str] = re.compile(rf"\$({IDENTIFIER_PATTERN_TEXT})")


__all__ = [
    # Conversation storage
    "DEFAULT_CONVERSATION_NAME",
    "CONVERSATION_NAME_PATTERN",
    "TRAILING_FORK_INDEX_PATTERN",
    # Framework-reserved parameters
    "RUN_ID_PARAM",
    "THINKING_ROUNDS_PARAM",
    # ScriptActAgent code-statement reserved literals
    "RHS_ASSIGN_ALIAS",
    "SUB_NAME_PREFIX",
    "RETURN_ALIAS",
    "TASK_RESULT_PREFIX",
    "PY_BUILTIN_ALIAS",
    "ATTR_CALL_ALIAS",
    "EXCLUDED_PY_BUILTINS",
    "KWARGS_UNPACK_KEY",
    "CODE_FENCE_PATTERN",
    "LEADING_CODE_FENCE_PATTERN",
    "TRAILING_CODE_FENCE_PATTERN",
    "DUNDER_ATTRIBUTE_PATTERN",
    "UNSUPPORTED_EXPR_LABELS",
    "RETURN_VALUE_FIELD",
    # Canonical return tool
    "RETURN_TOOL_NAME",
    "RETURN_TOOL_NAMESPACE",
    "RETURN_TOOL_DESCRIPTION",
    "RETURN_TOOL_FULL_NAME",
    "SIGIL_REF_PATTERN",
    # PlanActAgent output_structure schema
    "PLANACT_OUTPUT_SCHEMA",
    # ReActAgent output_structure schema
    "REACT_OUTPUT_SCHEMA",
]