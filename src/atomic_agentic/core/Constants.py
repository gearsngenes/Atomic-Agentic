from __future__ import annotations

import re
from typing import Any

# =============================================================================
# Sentinel / absence
# =============================================================================
# Used by:
# - core/sentinels.py: backwards-compatible NO_VAL re-export
# - core/Parameters.py: ParamSpec defaults
# - core/Invokable.py: signature/default rendering and input filtering
# - agents/data_classes.py: BlackboardSlot unset fields
# - agents/tool_agents.py: ToolAgentRunState return_value and placeholder state
# - workflows/metadata.py: absent child results
# - workflows/StructuredInvokable.py or future core.Invokable StructuredInvokable:
#   output default handling
#
# Contract:
# - NO_VAL is identity-checkable. Use `is NO_VAL`, not equality.
# - repr(NO_VAL) is stable and should remain "NO_VAL".


NO_VAL_REPR = "NO_VAL"


class _NoValSentinel:
    """Shared sentinel to represent an absent value.

    This object is intentionally opaque and single-instanced at module import
    time. Use `is NO_VAL` to test for absence.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return NO_VAL_REPR


NO_VAL: Any = _NoValSentinel()


# =============================================================================
# Generic identifier/name validation
# =============================================================================
# Used by:
# - core/Invokable.py: invokable names and parameter names
# - core/Parameters.py: schema-derived parameter names
# - agents/data_classes.py: ConstantSpec.name
# - workflows/parallel.py: output_names
# - ToolAgent constant placeholders below
#
# This is intentionally Python-identifier-like, not a full Python keyword check.
# Individual modules remain responsible for deciding which exception type/message
# to raise when validation fails.


IDENTIFIER_PATTERN_TEXT = r"[A-Za-z_][A-Za-z0-9_]*"
IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(
    rf"^{IDENTIFIER_PATTERN_TEXT}$"
)


# =============================================================================
# ToolAgent placeholder syntax
# =============================================================================
# Used by:
# - agents/tool_agents.py: placeholder dependency extraction and resolution
# - core/Prompts.py: LLM-facing instructions for placeholder output
# - tests around PlanAct/ReAct placeholder parsing and cache references
#
# Runtime placeholders:
# - <<__sN__>> references current-run step N
# - <<__cN__>> references cached blackboard step N
# - <<__k.NAME__>> references registered ToolAgent constant NAME
#
# This section belongs here because prompt text and runtime parsing must remain
# synchronized. Drift here would create real runtime bugs.


STEP_PLACEHOLDER_TEMPLATE = "<<__s{index}__>>"
CACHE_PLACEHOLDER_TEMPLATE = "<<__c{index}__>>"
CONSTANT_PLACEHOLDER_TEMPLATE = "<<__k.{name}__>>"

STEP_PLACEHOLDER_PATTERN_TEXT = r"<<__s(\d+)__>>"
CACHE_PLACEHOLDER_PATTERN_TEXT = r"<<__c(\d+)__>>"
CONSTANT_PLACEHOLDER_PATTERN_TEXT = rf"<<__k\.({IDENTIFIER_PATTERN_TEXT})__>>"

STEP_PLACEHOLDER_PATTERN: re.Pattern[str] = re.compile(
    STEP_PLACEHOLDER_PATTERN_TEXT
)
CACHE_PLACEHOLDER_PATTERN: re.Pattern[str] = re.compile(
    CACHE_PLACEHOLDER_PATTERN_TEXT
)
CONSTANT_PLACEHOLDER_PATTERN: re.Pattern[str] = re.compile(
    CONSTANT_PLACEHOLDER_PATTERN_TEXT
)


# =============================================================================
# ToolAgent prompt template fields
# =============================================================================
# Used by:
# - agents/tool_agents.py: ToolAgent.role_prompt formatting and validation
# - core/Prompts.py: PLANNER_PROMPT and ORCHESTRATOR_PROMPT templates
#
# These are str.format field names, not placeholder tokens.


TOOL_AGENT_TOOLS_FIELD = "TOOLS"
TOOL_AGENT_TOOL_CALLS_LIMIT_FIELD = "TOOL_CALLS_LIMIT"
TOOL_AGENT_CONSTANTS_FIELD = "CONSTANTS"

TOOL_AGENT_REQUIRED_ROLE_PROMPT_FIELDS = frozenset(
    {
        TOOL_AGENT_TOOLS_FIELD,
        TOOL_AGENT_TOOL_CALLS_LIMIT_FIELD,
        TOOL_AGENT_CONSTANTS_FIELD,
    }
)


# =============================================================================
# ToolAgent LLM-output JSON protocol
# =============================================================================
# Used by:
# - core/Prompts.py: expected planner/orchestrator JSON output contracts
# - agents/tool_agents.py: generated step validation and BlackboardSlot creation
# - agents/data_classes.py: BlackboardSlot.from_dict support
#
# These keys are centralized because prompt text and parser/validator code need
# to agree on the same LLM-output protocol.


PLAN_STEP_KEY = "step"
PLAN_TOOL_KEY = "tool"
PLAN_ARGS_KEY = "args"
PLAN_AWAIT_KEY = "await"
PLAN_DURATION_KEY = "duration"
PLAN_DESCRIPTION_KEY = "description"

RETURN_VALUE_ARG_KEY = "val"

PLANACT_ALLOWED_STEP_KEYS = frozenset(
    {
        PLAN_STEP_KEY,
        PLAN_TOOL_KEY,
        PLAN_ARGS_KEY,
        PLAN_AWAIT_KEY,
    }
)

PLANACT_REQUIRED_STEP_KEYS = frozenset(
    {
        PLAN_TOOL_KEY,
        PLAN_ARGS_KEY,
    }
)

REACT_ALLOWED_STEP_KEYS = frozenset(
    {
        PLAN_STEP_KEY,
        PLAN_TOOL_KEY,
        PLAN_ARGS_KEY,
        PLAN_DURATION_KEY,
        PLAN_DESCRIPTION_KEY,
    }
)

REACT_REQUIRED_STEP_KEYS = frozenset(
    {
        PLAN_TOOL_KEY,
        PLAN_ARGS_KEY,
        PLAN_DURATION_KEY,
        PLAN_DESCRIPTION_KEY,
    }
)


# =============================================================================
# ToolAgent canonical return-tool identity
# =============================================================================
# Used by:
# - agents/tool_agents.py: construction and registration of the executable return_tool
# - core/Prompts.py: finalization instructions requiring Tool.ToolAgents.return
# - tests around planner/ReAct final return behavior
#
# Do not put the executable Tool instance here; only the identity literals that
# must stay synchronized with prompt text.


TOOL_AGENT_RETURN_TOOL_NAME = "return"
TOOL_AGENT_RETURN_TOOL_NAMESPACE = "ToolAgents"
TOOL_AGENT_RETURN_TOOL_DESCRIPTION = (
    "Returns the passed-in value. Tool agents should use this to signal completion."
)
TOOL_AGENT_RETURN_FULL_NAME = (
    f"Tool.{TOOL_AGENT_RETURN_TOOL_NAMESPACE}.{TOOL_AGENT_RETURN_TOOL_NAME}"
)


# =============================================================================
# PyA2Atomic / A2A protocol constants
# =============================================================================
# Used by:
# - a2a/PyA2AtomicHost.py: reserved function dispatch, direct-call envelopes,
#   error envelopes, and metadata envelopes
# - a2a/PyA2AtomicClient.py: reserved function calls and payload validation
# - tools/a2a.py: PyA2AtomicTool remote metadata handling
#
# These are centralized because host/client/proxy code must agree on the same
# transport-level payload keys.


PYA2A_RESULT_KEY = "__py_a2a_result__"

PYA2A_LIST_INVOKABLES_FUNCTION = "list_invokables"
PYA2A_GET_INVOKABLE_METADATA_FUNCTION = "get_invokable_metadata"

PYA2A_METADATA_NAME_KEY = "name"
PYA2A_METADATA_DESCRIPTION_KEY = "description"
PYA2A_METADATA_PARAMETERS_KEY = "parameters"
PYA2A_METADATA_RETURN_TYPE_KEY = "return_type"
PYA2A_METADATA_FILTER_EXTRANEOUS_INPUTS_KEY = "filter_extraneous_inputs"
PYA2A_METADATA_INVOKABLE_TYPE_KEY = "invokable_type"

PYA2A_REQUIRED_METADATA_KEYS = frozenset(
    {
        PYA2A_METADATA_NAME_KEY,
        PYA2A_METADATA_DESCRIPTION_KEY,
        PYA2A_METADATA_PARAMETERS_KEY,
        PYA2A_METADATA_RETURN_TYPE_KEY,
        PYA2A_METADATA_FILTER_EXTRANEOUS_INPUTS_KEY,
        PYA2A_METADATA_INVOKABLE_TYPE_KEY,
    }
)

PYA2A_ERROR_KEY = "error"
PYA2A_ERROR_TYPE_KEY = "error_type"
PYA2A_ERROR_FUNCTION_NAME_KEY = "function_name"


# =============================================================================
# MCP transport/proxy protocol constants
# =============================================================================
# Used by:
# - mcp/MCPClientHub.py: transport validation and metadata envelopes
# - tools/mcp.py: MCPProxyTool construction, metadata validation, and result extraction
# - mcp/utils.py if metadata/result normalization helpers are split later
#
# These are literal transport/result/metadata labels shared across the MCP
# adapter boundary. Display metadata and local defaults stay local to their
# modules unless duplication becomes significant.


MCP_TRANSPORT_STDIO = "stdio"
MCP_TRANSPORT_SSE = "sse"
MCP_TRANSPORT_STREAMABLE_HTTP = "streamable_http"

MCP_TRANSPORT_MODES = frozenset(
    {
        MCP_TRANSPORT_STDIO,
        MCP_TRANSPORT_SSE,
        MCP_TRANSPORT_STREAMABLE_HTTP,
    }
)

MCP_METADATA_PARAMETERS_KEY = "parameters"
MCP_METADATA_RETURN_TYPE_KEY = "return_type"
MCP_METADATA_RAW_METADATA_KEY = "raw_metadata"
MCP_METADATA_EXTRACTION_MODE_KEY = "extraction_mode"

MCP_EXTRACTION_MODE_EXTRACT_RESULT = "extract_result"
MCP_EXTRACTION_MODE_STRUCTURED_CONTENT = "structured_content"
MCP_EXTRACTION_MODE_CONTENT_BLOCKS = "content_blocks"

MCP_EXTRACTION_MODES = frozenset(
    {
        MCP_EXTRACTION_MODE_EXTRACT_RESULT,
        MCP_EXTRACTION_MODE_STRUCTURED_CONTENT,
        MCP_EXTRACTION_MODE_CONTENT_BLOCKS,
    }
)

MCP_RESULT_IS_ERROR_KEY = "isError"
MCP_RESULT_STRUCTURED_CONTENT_KEY = "structuredContent"
MCP_RESULT_CONTENT_KEY = "content"


# =============================================================================
# Explicit public export list
# =============================================================================
# Keep this explicit so adding local helper names or imports cannot accidentally
# widen the module's public surface.


__all__ = [
    # Sentinel / absence
    "NO_VAL_REPR",
    "NO_VAL",
    # Identifier validation
    "IDENTIFIER_PATTERN_TEXT",
    "IDENTIFIER_PATTERN",
    # ToolAgent placeholders
    "STEP_PLACEHOLDER_TEMPLATE",
    "CACHE_PLACEHOLDER_TEMPLATE",
    "CONSTANT_PLACEHOLDER_TEMPLATE",
    "STEP_PLACEHOLDER_PATTERN_TEXT",
    "CACHE_PLACEHOLDER_PATTERN_TEXT",
    "CONSTANT_PLACEHOLDER_PATTERN_TEXT",
    "STEP_PLACEHOLDER_PATTERN",
    "CACHE_PLACEHOLDER_PATTERN",
    "CONSTANT_PLACEHOLDER_PATTERN",
    # ToolAgent prompt fields
    "TOOL_AGENT_TOOLS_FIELD",
    "TOOL_AGENT_TOOL_CALLS_LIMIT_FIELD",
    "TOOL_AGENT_CONSTANTS_FIELD",
    "TOOL_AGENT_REQUIRED_ROLE_PROMPT_FIELDS",
    # ToolAgent LLM-output JSON protocol
    "PLAN_STEP_KEY",
    "PLAN_TOOL_KEY",
    "PLAN_ARGS_KEY",
    "PLAN_AWAIT_KEY",
    "PLAN_DURATION_KEY",
    "PLAN_DESCRIPTION_KEY",
    "RETURN_VALUE_ARG_KEY",
    "PLANACT_ALLOWED_STEP_KEYS",
    "PLANACT_REQUIRED_STEP_KEYS",
    "REACT_ALLOWED_STEP_KEYS",
    "REACT_REQUIRED_STEP_KEYS",
    # ToolAgent return identity
    "TOOL_AGENT_RETURN_TOOL_NAME",
    "TOOL_AGENT_RETURN_TOOL_NAMESPACE",
    "TOOL_AGENT_RETURN_FULL_NAME",
    # PyA2Atomic / A2A
    "PYA2A_RESULT_KEY",
    "PYA2A_LIST_INVOKABLES_FUNCTION",
    "PYA2A_GET_INVOKABLE_METADATA_FUNCTION",
    "PYA2A_METADATA_NAME_KEY",
    "PYA2A_METADATA_DESCRIPTION_KEY",
    "PYA2A_METADATA_PARAMETERS_KEY",
    "PYA2A_METADATA_RETURN_TYPE_KEY",
    "PYA2A_METADATA_FILTER_EXTRANEOUS_INPUTS_KEY",
    "PYA2A_METADATA_INVOKABLE_TYPE_KEY",
    "PYA2A_REQUIRED_METADATA_KEYS",
    "PYA2A_ERROR_KEY",
    "PYA2A_ERROR_TYPE_KEY",
    "PYA2A_ERROR_FUNCTION_NAME_KEY",
    # MCP
    "MCP_TRANSPORT_STDIO",
    "MCP_TRANSPORT_SSE",
    "MCP_TRANSPORT_STREAMABLE_HTTP",
    "MCP_TRANSPORT_MODES",
    "MCP_METADATA_PARAMETERS_KEY",
    "MCP_METADATA_RETURN_TYPE_KEY",
    "MCP_METADATA_RAW_METADATA_KEY",
    "MCP_METADATA_EXTRACTION_MODE_KEY",
    "MCP_EXTRACTION_MODE_EXTRACT_RESULT",
    "MCP_EXTRACTION_MODE_STRUCTURED_CONTENT",
    "MCP_EXTRACTION_MODE_CONTENT_BLOCKS",
    "MCP_EXTRACTION_MODES",
    "MCP_RESULT_IS_ERROR_KEY",
    "MCP_RESULT_STRUCTURED_CONTENT_KEY",
    "MCP_RESULT_CONTENT_KEY",
]
