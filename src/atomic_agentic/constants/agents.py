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

# RETURN_VALUE_FIELD is the kwargs key DagAgent/PlanActAgent use for their
# synthesized RETURN_ALIAS call's resolved value (utils/dag.py's
# parse_generation/resolve_call_args), and the parameter name return_tool's
# own real signature uses (agents/tools.py's `_return(val)`).
RETURN_VALUE_FIELD = "val"


# =============================================================================
# JsonToolAgent canonical return-tool identity
# =============================================================================
# Used by:
# - agents/json_tool_agent.py: construction and registration of the executable return_tool
# - JsonToolAgent prompt finalization instructions requiring Tool.ToolAgents.return
# - tests around planner/ReAct final return behavior
#
# Do not put the executable Tool instance here; only the identity literals that
# must stay synchronized with JsonToolAgent prompt text.


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
# utils/agents.py's strip_code_fence (ScriptAgent's own code-statement
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
# (for a bare `ast.Attribute` anywhere in an expression -- ScriptAgent's
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

FINAL_ROUND_WARNING = (
    "This is your FINAL planning round -- you must complete the entire "
    "task now; you may not defer further."
)
"""Appended (space-separated) to a continuation instruction when
<agent>._is_final_round(task) is true -- shared verbatim by both
ScriptAgent and DagAgent (each family's own _render_task_messages' round-1
and continuation branches), so all four call sites can never drift in
wording. Deliberately grammar-neutral -- earlier revisions said "Do not
write # PAUSE", a ScriptAgent-specific instruction meaningless in
DagAgent's own grammar (there is no "# PAUSE" marker or equivalent; a
DagAgent round signals continuation via the remaining_work JSON
field instead) -- "you may not defer further" already fully covers the
same intent (deferring IS writing # PAUSE, for ScriptAgent) without
naming a mechanism that doesn't exist in DagAgent's own output schema."""

# =============================================================================
# DagAgent output_structure schema
# =============================================================================
# Used by:
# - utils/dag.py: build_dag_schema deep-copies this template and injects the
#   dynamic call.enum from the current toolbox.
# - agents/dag.py (Pass 3): passes the built schema as output_structure on
#   every planning-round engine call.
#
# Field names are plain string literals, not separate per-field-name
# constants (unlike STEP_FIELD/TOOL_FIELD/etc. above) -- this schema has
# exactly one producer (this constant) and one consumer (utils/dag.py's
# parse_generation), so there is no cross-file drift risk a shared constant
# would guard against. call.enum ships empty -- never sent to a provider
# as-is; build_dag_schema always populates it first.
#
# "summary" is a top-level field, not a per-plan-item "reason" (revised
# 2026-09-20, during Pass 3's prompt design -- superseding the original
# per-item reason field this schema shipped with in Pass 2). Only
# "properties" dict insertion order is load-bearing for constrained
# decoding (not "required"'s order) -- summary must be declared before
# plan/remaining_work/return so the model's stated reasoning can
# causally precede every one of them, the same field-order principle the
# original per-item reason design was built on, just applied once per
# round instead of once per call: cheaper (one reasoning blob, not N), and
# lets the model reason about the whole batch's strategy and its own
# halt/continue decision, neither of which a per-call reason ever
# actually informed.

DAG_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "plan", "remaining_work", "return"],
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "Briefly describe the work this round's plan accomplishes, "
                "and whether the task will be complete after it runs."
            ),
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
        "remaining_work": {
            "type": ["string", "null"],
            "description": (
                "Null (or blank) once this round's plan, together with "
                "'return', finishes the task. Otherwise, a non-empty "
                "string describing exactly what still needs to happen "
                "and what you need to inspect before deciding the next "
                "steps -- this text is shown back to you, verbatim, at "
                "the start of the next round, so write it as a note to "
                "your own future self, not just a status label."
            ),
        },
        "return": {
            "type": ["number", "boolean", "null", "string"],
            "description": (
                "A non-null value follows the same rules as an argument's "
                "value (literal, or a '$name' reference/interpolation). "
                "null means nothing is returned this round ('remaining_"
                "work' governs instead)."
            ),
        },
    },
}

# PlanActAgent's own output_structure schema (agent-taxonomy `planact-
# rewrite` design record) -- DAG_OUTPUT_SCHEMA's shape minus
# `remaining_work` entirely: a one-shot planner has no continuation round
# to defer to, so there is no field for it. `return` keeps DAG_OUTPUT_
# SCHEMA's own full value union, `null` included -- presence and
# nullability are separate concerns: `return` is unconditionally in
# `required` (always present, exactly like DAG_OUTPUT_SCHEMA's own
# `return`), but its *value* may still legitimately be `null` when the
# task has nothing meaningful to hand back. Never a "come back later"
# signal here, unlike `remaining_work`'s absence might suggest -- there is
# no later.
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
# principle as DAG_OUTPUT_SCHEMA/PLANACT_OUTPUT_SCHEMA.
REACT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "call", "arguments", "result_name"],
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "Briefly describe what this one step accomplishes and why "
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

# Matches a `$`-sigil reference inside a DagAgent-, PlanActAgent-, or
# ReActAgent-generated `value`/`return` string: `$` followed by an
# identifier-legal name,
# captured as group 1. Used two ways by utils/dag.py -- `DAG_REF_PATTERN.fullmatch(s)` (is the
# WHOLE string one reference?) and `DAG_REF_PATTERN.finditer(s)`/`.sub(s)`
# (find/splice every embedded occurrence) -- one pattern serves both, no
# anchors baked into the text itself (fullmatch anchors on its own, exactly
# like this file's own IDENTIFIER_PATTERN precedent elsewhere).
DAG_REF_PATTERN: re.Pattern[str] = re.compile(rf"\$({IDENTIFIER_PATTERN_TEXT})")


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
    "RETURN_VALUE_FIELD",
    # Canonical return tool
    "RETURN_TOOL_NAME",
    "RETURN_TOOL_NAMESPACE",
    "RETURN_TOOL_DESCRIPTION",
    "RETURN_TOOL_FULL_NAME",
    # DagAgent output_structure schema
    "DAG_OUTPUT_SCHEMA",
    "DAG_REF_PATTERN",
    # PlanActAgent output_structure schema
    "PLANACT_OUTPUT_SCHEMA",
    # ReActAgent output_structure schema
    "REACT_OUTPUT_SCHEMA",
]