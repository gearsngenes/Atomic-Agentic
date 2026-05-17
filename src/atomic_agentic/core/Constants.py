from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any, Mapping

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
# - placeholder regexes below: constant placeholder names
#
# This is intentionally Python-identifier-like, not a full Python keyword check.


IDENTIFIER_PATTERN_TEXT = r"[A-Za-z_][A-Za-z0-9_]*"
IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(
    rf"^{IDENTIFIER_PATTERN_TEXT}$"
)


# =============================================================================
# Generic serialized/introspection keys
# =============================================================================
# Used by stable metadata/to_dict-style payloads across core, tools, A2A,
# MCP, and workflows. Prefer these only when the key is part of a stable
# serialized or protocol-like contract, not for every incidental local dict.


TYPE_KEY = "type"
NAME_KEY = "name"
DESCRIPTION_KEY = "description"
PARAMETERS_KEY = "parameters"
RETURN_TYPE_KEY = "return_type"
FILTER_EXTRANEOUS_INPUTS_KEY = "filter_extraneous_inputs"
INSTANCE_ID_KEY = "instance_id"
FULL_NAME_KEY = "full_name"
INPUTS_KEY = "inputs"
RESULT_KEY = "result"
METADATA_KEY = "metadata"
RUN_ID_KEY = "run_id"


# =============================================================================
# ParamSpec kinds and mapping keys
# =============================================================================
# Used by:
# - core/Parameters.py: ParamSpec class constants, to_dict/from_dict,
#   schema normalization, and parameter ordering validation
# - core/Invokable.py: signature rendering and vararg/varkwarg detection
# - tools/a2a.py and tools/mcp.py: remote schema reconstruction
# - workflows/parallel.py and StructuredInvokable: output/input schema handling
#
# ParamSpec should keep class aliases for public compatibility:
#     ParamSpec.POSITIONAL_ONLY = PARAM_KIND_POSITIONAL_ONLY
#     ...


PARAM_KIND_POSITIONAL_ONLY = "POSITIONAL_ONLY"
PARAM_KIND_POSITIONAL_OR_KEYWORD = "POSITIONAL_OR_KEYWORD"
PARAM_KIND_VAR_POSITIONAL = "VAR_POSITIONAL"
PARAM_KIND_KEYWORD_ONLY = "KEYWORD_ONLY"
PARAM_KIND_VAR_KEYWORD = "VAR_KEYWORD"

PARAM_KINDS = frozenset(
    {
        PARAM_KIND_POSITIONAL_ONLY,
        PARAM_KIND_POSITIONAL_OR_KEYWORD,
        PARAM_KIND_VAR_POSITIONAL,
        PARAM_KIND_KEYWORD_ONLY,
        PARAM_KIND_VAR_KEYWORD,
    }
)

PARAM_KIND_ORDER: Mapping[str, int] = MappingProxyType(
    {
        PARAM_KIND_POSITIONAL_ONLY: 0,
        PARAM_KIND_POSITIONAL_OR_KEYWORD: 1,
        PARAM_KIND_VAR_POSITIONAL: 2,
        PARAM_KIND_KEYWORD_ONLY: 3,
        PARAM_KIND_VAR_KEYWORD: 4,
    }
)

PARAM_NAME_KEY = NAME_KEY
PARAM_INDEX_KEY = "index"
PARAM_KIND_KEY = "kind"
PARAM_TYPE_KEY = TYPE_KEY
PARAM_DEFAULT_KEY = "default"

PARAMSPEC_REQUIRED_KEYS = frozenset(
    {
        PARAM_NAME_KEY,
        PARAM_INDEX_KEY,
        PARAM_KIND_KEY,
        PARAM_TYPE_KEY,
    }
)


# =============================================================================
# Parameter/schema grammar literals
# =============================================================================
# Used by:
# - core/Parameters.py: to_paramspec_list and schema parsing helpers
#
# These describe the small schema grammar used to turn strings/sequences/types
# into ParamSpec lists.


SCHEMA_POSITIONAL_ONLY_MARKER = "/"
SCHEMA_KEYWORD_ONLY_MARKER = "*"
SCHEMA_VAR_POSITIONAL_PREFIX = "*"
SCHEMA_VAR_KEYWORD_PREFIX = "**"
DEFAULT_TYPE_NAME = "Any"


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

TOOL_AGENT_UNLIMITED_TOOL_CALLS_TEXT = "unlimited"


# =============================================================================
# ToolAgent plan / ReAct JSON keys
# =============================================================================
# Used by:
# - core/Prompts.py: expected planner/orchestrator JSON output contracts
# - agents/tool_agents.py: generated step validation and BlackboardSlot creation
# - agents/data_classes.py: BlackboardSlot.from_dict support
#
# These keys are LLM-output protocol keys, so prompt text and parser/validator
# code should stay synchronized around these constants.


PLAN_STEP_KEY = "step"
PLAN_TOOL_KEY = "tool"
PLAN_ARGS_KEY = "args"
PLAN_AWAIT_KEY = "await"
PLAN_DURATION_KEY = "duration"
PLAN_DESCRIPTION_KEY = "description"

PLAN_RESULT_REF_KEY = "result_ref"
PLAN_OBSERVABLE_RESULT_KEY = "observable_result"
PLAN_RESULT_KEY = RESULT_KEY

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
# BlackboardSlot statuses and serialized keys
# =============================================================================
# Used by:
# - agents/data_classes.py: BlackboardSlot validation, convenience predicates,
#   to_dict/from_dict
# - agents/tool_agents.py: planning/execution state transitions
#
# BlackboardSlot can keep class aliases:
#     BlackboardSlot.EMPTY = BLACKBOARD_STATUS_EMPTY
#     BlackboardSlot.VALID_STATUSES = set(BLACKBOARD_VALID_STATUSES)


BLACKBOARD_STATUS_EMPTY = "empty"
BLACKBOARD_STATUS_PLANNED = "planned"
BLACKBOARD_STATUS_PREPARED = "prepared"
BLACKBOARD_STATUS_EXECUTED = "executed"
BLACKBOARD_STATUS_FAILED = "failed"

BLACKBOARD_VALID_STATUSES = frozenset(
    {
        BLACKBOARD_STATUS_EMPTY,
        BLACKBOARD_STATUS_PLANNED,
        BLACKBOARD_STATUS_PREPARED,
        BLACKBOARD_STATUS_EXECUTED,
        BLACKBOARD_STATUS_FAILED,
    }
)

BLACKBOARD_STEP_KEY = PLAN_STEP_KEY
BLACKBOARD_TOOL_KEY = PLAN_TOOL_KEY
BLACKBOARD_ARGS_KEY = PLAN_ARGS_KEY
BLACKBOARD_RESOLVED_ARGS_KEY = "resolved_args"
BLACKBOARD_RESULT_KEY = RESULT_KEY
BLACKBOARD_ERROR_KEY = "error"
BLACKBOARD_STATUS_KEY = "status"
BLACKBOARD_STEP_DEPENDENCIES_KEY = "step_dependencies"
BLACKBOARD_AWAIT_KEY = PLAN_AWAIT_KEY
BLACKBOARD_AWAIT_STEP_KEY = "await_step"

BLACKBOARD_FROM_DICT_ALLOWED_KEYS = frozenset(
    {
        BLACKBOARD_STEP_KEY,
        BLACKBOARD_TOOL_KEY,
        BLACKBOARD_ARGS_KEY,
        BLACKBOARD_RESOLVED_ARGS_KEY,
        BLACKBOARD_RESULT_KEY,
        BLACKBOARD_ERROR_KEY,
        BLACKBOARD_STATUS_KEY,
        BLACKBOARD_STEP_DEPENDENCIES_KEY,
        BLACKBOARD_AWAIT_KEY,
        BLACKBOARD_AWAIT_STEP_KEY,
    }
)


# =============================================================================
# ToolAgent canonical return tool and registration modes
# =============================================================================
# Used by:
# - agents/tool_agents.py: construction of the executable return_tool
# - core/Prompts.py: finalization instructions requiring Tool.ToolAgents.return
# - tests around planner/ReAct final return behavior
#
# Do not put the executable Tool instance here; only its literal metadata.


TOOL_AGENT_RETURN_TOOL_NAME = "return"
TOOL_AGENT_RETURN_TOOL_NAMESPACE = "ToolAgents"
TOOL_AGENT_RETURN_TOOL_DESCRIPTION = (
    "Returns the passed-in value. Tool agents should use this to signal completion."
)
TOOL_AGENT_RETURN_FULL_NAME = (
    f"Tool.{TOOL_AGENT_RETURN_TOOL_NAMESPACE}.{TOOL_AGENT_RETURN_TOOL_NAME}"
)

NAME_COLLISION_RAISE = "raise"
NAME_COLLISION_SKIP = "skip"
NAME_COLLISION_REPLACE = "replace"

VALID_NAME_COLLISION_MODES = frozenset(
    {
        NAME_COLLISION_RAISE,
        NAME_COLLISION_SKIP,
        NAME_COLLISION_REPLACE,
    }
)


# =============================================================================
# StructuredInvokable packaging constants
# =============================================================================
# Used by:
# - workflows/StructuredInvokable.py currently
# - future core/Invokable.py if StructuredInvokable moves there
#
# Keep StructuredInvokable.RAISE/DROP/FILL as class aliases to these constants.
# Do not move StructuredInvokable.PASSTHROUGH itself here because it contains
# a ParamSpec instance. Only move its literal pieces.


STRUCTURED_ABSENT_MODE_RAISE = "raise"
STRUCTURED_ABSENT_MODE_DROP = "drop"
STRUCTURED_ABSENT_MODE_FILL = "fill"

STRUCTURED_ABSENT_MODES = frozenset(
    {
        STRUCTURED_ABSENT_MODE_RAISE,
        STRUCTURED_ABSENT_MODE_DROP,
        STRUCTURED_ABSENT_MODE_FILL,
    }
)

STRUCTURED_RESULT_RETURN_TYPE = "StructuredResultDict[str, Any]"
STRUCTURED_PASSTHROUGH_MAPPING_NAME = "__passthrough_mapping__"
STRUCTURED_PASSTHROUGH_MAPPING_TYPE = "Mapping[str, Any]"


# =============================================================================
# Workflow result/topology constants
# =============================================================================
# Used by:
# - workflows/base.py: Workflow return_type
# - workflows/metadata.py: metadata kind values and OutputTopology labels
# - workflows/parallel.py: input/output shape and duplicate-key policy
#
# Existing class constants should become aliases for compatibility:
#     OutputTopology.NESTED = OUTPUT_TOPOLOGY_NESTED
#     ParallelFlow.BROADCAST = PARALLEL_INPUT_SHAPE_BROADCAST
#     ...


FLOW_RESULT_RETURN_TYPE = "FlowResultDict[str, Any]"

OUTPUT_TOPOLOGY_NESTED = "nested"
OUTPUT_TOPOLOGY_FLATTENED = "flattened"

PARALLEL_INPUT_SHAPE_BROADCAST = "broadcast"
PARALLEL_INPUT_SHAPE_NESTED = OUTPUT_TOPOLOGY_NESTED

PARALLEL_OUTPUT_SHAPE_NESTED = OUTPUT_TOPOLOGY_NESTED
PARALLEL_OUTPUT_SHAPE_FLATTENED = OUTPUT_TOPOLOGY_FLATTENED
PARALLEL_OUTPUT_SHAPE_ENVELOPED = OUTPUT_TOPOLOGY_NESTED  # compatibility alias

DUPLICATE_KEY_POLICY_RAISE = "raise"
DUPLICATE_KEY_POLICY_SKIP = "skip"
DUPLICATE_KEY_POLICY_UPDATE = "update"

VALID_PARALLEL_INPUT_SHAPES = frozenset(
    {
        PARALLEL_INPUT_SHAPE_BROADCAST,
        PARALLEL_INPUT_SHAPE_NESTED,
    }
)

VALID_PARALLEL_OUTPUT_SHAPES = frozenset(
    {
        PARALLEL_OUTPUT_SHAPE_NESTED,
        PARALLEL_OUTPUT_SHAPE_FLATTENED,
    }
)

VALID_DUPLICATE_KEY_POLICIES = frozenset(
    {
        DUPLICATE_KEY_POLICY_RAISE,
        DUPLICATE_KEY_POLICY_SKIP,
        DUPLICATE_KEY_POLICY_UPDATE,
    }
)

WORKFLOW_KIND_BASIC = "basic"
WORKFLOW_KIND_SEQUENTIAL = "sequential"
WORKFLOW_KIND_ROUTING = "routing"
WORKFLOW_KIND_ITERATIVE = "iterative"
WORKFLOW_KIND_PARALLEL = "parallel"

WORKFLOW_KINDS = frozenset(
    {
        WORKFLOW_KIND_BASIC,
        WORKFLOW_KIND_SEQUENTIAL,
        WORKFLOW_KIND_ROUTING,
        WORKFLOW_KIND_ITERATIVE,
        WORKFLOW_KIND_PARALLEL,
    }
)


# =============================================================================
# PyA2Atomic / A2A protocol constants
# =============================================================================
# Used by:
# - a2a/PyA2AtomicHost.py: reserved function dispatch, direct-call envelopes,
#   error envelopes, host defaults
# - a2a/PyA2AtomicClient.py: reserved function calls and payload validation
# - tools/a2a.py: PyA2AtomicTool remote metadata handling and default namespace
#
# These are centralized here because the user wants one constants file.
# If the package later splits constants by domain, these could move to
# a2a/Constants.py without changing their values.


PYA2A_RESULT_KEY = "__py_a2a_result__"

PYA2A_LIST_INVOKABLES_FUNCTION = "list_invokables"
PYA2A_GET_INVOKABLE_METADATA_FUNCTION = "get_invokable_metadata"
PYA2A_UNKNOWN_FUNCTION_NAME = "unknown_function"

PYA2A_METADATA_NAME_KEY = NAME_KEY
PYA2A_METADATA_DESCRIPTION_KEY = DESCRIPTION_KEY
PYA2A_METADATA_PARAMETERS_KEY = PARAMETERS_KEY
PYA2A_METADATA_RETURN_TYPE_KEY = RETURN_TYPE_KEY
PYA2A_METADATA_FILTER_EXTRANEOUS_INPUTS_KEY = FILTER_EXTRANEOUS_INPUTS_KEY
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

PYA2A_CONTENT_TYPE_TEXT = "text"
PYA2A_CONTENT_TYPE_FUNCTION_CALL = "function_call"
PYA2A_CONTENT_TYPE_FUNCTION_RESPONSE = "function_response"

PYA2A_DEFAULT_VERSION = "1.0.0"
PYA2A_DEFAULT_HOST = "localhost"
PYA2A_DEFAULT_PORT = 5000
PYA2A_DEFAULT_TOOL_NAMESPACE = "pya2a"


# =============================================================================
# MCP transport/proxy constants
# =============================================================================
# Used by:
# - mcp/MCPClientHub.py: transport validation and metadata envelopes
# - tools/mcp.py: MCPProxyTool construction, metadata validation, result extraction
# - mcp/utils.py if metadata/result normalization helpers are split later
#
# These are literal protocol/metadata labels. They should not introduce imports
# from the MCP SDK or Tool classes.


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

MCP_DEFAULT_TOOL_NAMESPACE = "mcp"

MCP_METADATA_NAME_KEY = NAME_KEY
MCP_METADATA_DESCRIPTION_KEY = DESCRIPTION_KEY
MCP_METADATA_PARAMETERS_KEY = PARAMETERS_KEY
MCP_METADATA_RETURN_TYPE_KEY = RETURN_TYPE_KEY
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
# LLM engine attachment/file-policy constants
# =============================================================================
# Used by:
# - engines/LLMEngines.py: base illegal attachment validation, OpenAI file
#   classification, metadata payloads, provider request block construction
#
# These are literal file-extension, MIME-prefix, metadata-key, and provider block
# labels. They are safe to centralize because they do not import provider SDKs.


ATTACHMENT_EXT_ZIP = ".zip"
ATTACHMENT_EXT_TAR = ".tar"
ATTACHMENT_EXT_GZ = ".gz"
ATTACHMENT_EXT_TGZ = ".tgz"
ATTACHMENT_EXT_RAR = ".rar"
ATTACHMENT_EXT_7Z = ".7z"

ATTACHMENT_EXT_EXE = ".exe"
ATTACHMENT_EXT_DLL = ".dll"
ATTACHMENT_EXT_SO = ".so"
ATTACHMENT_EXT_BIN = ".bin"
ATTACHMENT_EXT_O = ".o"

ATTACHMENT_EXT_DB = ".db"
ATTACHMENT_EXT_SQLITE = ".sqlite"

ATTACHMENT_EXT_H5 = ".h5"
ATTACHMENT_EXT_PT = ".pt"
ATTACHMENT_EXT_PTH = ".pth"
ATTACHMENT_EXT_ONNX = ".onnx"

ILLEGAL_ATTACHMENT_EXTS = frozenset(
    {
        ATTACHMENT_EXT_ZIP,
        ATTACHMENT_EXT_TAR,
        ATTACHMENT_EXT_GZ,
        ATTACHMENT_EXT_TGZ,
        ATTACHMENT_EXT_RAR,
        ATTACHMENT_EXT_7Z,
        ATTACHMENT_EXT_EXE,
        ATTACHMENT_EXT_DLL,
        ATTACHMENT_EXT_SO,
        ATTACHMENT_EXT_BIN,
        ATTACHMENT_EXT_O,
        ATTACHMENT_EXT_DB,
        ATTACHMENT_EXT_SQLITE,
        ATTACHMENT_EXT_H5,
        ATTACHMENT_EXT_PT,
        ATTACHMENT_EXT_PTH,
        ATTACHMENT_EXT_ONNX,
    }
)

IMAGE_ATTACHMENT_EXTS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".heic",
    }
)

TEXT_ATTACHMENT_EXTS = frozenset(
    {
        ".txt",
        ".md",
        ".rst",
        ".log",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".csv",
        ".tsv",
        ".py",
        ".ipynb",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rs",
        ".go",
        ".rb",
        ".php",
        ".cs",
        ".html",
        ".htm",
        ".xml",
    }
)

PDF_ATTACHMENT_EXT = ".pdf"

ILLEGAL_MIME_PREFIX_AUDIO = "audio/"
ILLEGAL_MIME_PREFIX_VIDEO = "video/"
ILLEGAL_MIME_PREFIXES = (
    ILLEGAL_MIME_PREFIX_AUDIO,
    ILLEGAL_MIME_PREFIX_VIDEO,
)

ATTACHMENT_KIND_PDF = "pdf"
ATTACHMENT_KIND_IMAGE = "image"
ATTACHMENT_KIND_TEXT = "text"
ATTACHMENT_KIND_FILE = "file"

ATTACHMENT_META_KIND_KEY = "kind"
ATTACHMENT_META_MIME_KEY = "mime"
ATTACHMENT_META_EXT_KEY = "ext"
ATTACHMENT_META_UPLOADED_KEY = "uploaded"
ATTACHMENT_META_FILE_ID_KEY = "file_id"
ATTACHMENT_META_FILE_OBJ_KEY = "file_obj"
ATTACHMENT_META_SIGNED_URL_KEY = "signed_url"
ATTACHMENT_META_INLINED_KEY = "inlined"
ATTACHMENT_META_INLINED_TEXT_KEY = "inlined_text"

OPENAI_INPUT_FILE_TYPE = "input_file"
OPENAI_INPUT_IMAGE_TYPE = "input_image"
OPENAI_INPUT_TEXT_TYPE = "input_text"
OPENAI_OUTPUT_TEXT_TYPE = "output_text"

MISTRAL_TEXT_PART_TYPE = "text"
MISTRAL_IMAGE_URL_PART_TYPE = "image_url"


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
    # Generic serialized/introspection keys
    "TYPE_KEY",
    "NAME_KEY",
    "DESCRIPTION_KEY",
    "PARAMETERS_KEY",
    "RETURN_TYPE_KEY",
    "FILTER_EXTRANEOUS_INPUTS_KEY",
    "INSTANCE_ID_KEY",
    "FULL_NAME_KEY",
    "INPUTS_KEY",
    "RESULT_KEY",
    "METADATA_KEY",
    "RUN_ID_KEY",
    # ParamSpec
    "PARAM_KIND_POSITIONAL_ONLY",
    "PARAM_KIND_POSITIONAL_OR_KEYWORD",
    "PARAM_KIND_VAR_POSITIONAL",
    "PARAM_KIND_KEYWORD_ONLY",
    "PARAM_KIND_VAR_KEYWORD",
    "PARAM_KINDS",
    "PARAM_KIND_ORDER",
    "PARAM_NAME_KEY",
    "PARAM_INDEX_KEY",
    "PARAM_KIND_KEY",
    "PARAM_TYPE_KEY",
    "PARAM_DEFAULT_KEY",
    "PARAMSPEC_REQUIRED_KEYS",
    # Parameter/schema grammar
    "SCHEMA_POSITIONAL_ONLY_MARKER",
    "SCHEMA_KEYWORD_ONLY_MARKER",
    "SCHEMA_VAR_POSITIONAL_PREFIX",
    "SCHEMA_VAR_KEYWORD_PREFIX",
    "DEFAULT_TYPE_NAME",
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
    "TOOL_AGENT_UNLIMITED_TOOL_CALLS_TEXT",
    # ToolAgent plan/ReAct JSON keys
    "PLAN_STEP_KEY",
    "PLAN_TOOL_KEY",
    "PLAN_ARGS_KEY",
    "PLAN_AWAIT_KEY",
    "PLAN_DURATION_KEY",
    "PLAN_DESCRIPTION_KEY",
    "PLAN_RESULT_REF_KEY",
    "PLAN_OBSERVABLE_RESULT_KEY",
    "PLAN_RESULT_KEY",
    "RETURN_VALUE_ARG_KEY",
    "PLANACT_ALLOWED_STEP_KEYS",
    "PLANACT_REQUIRED_STEP_KEYS",
    "REACT_ALLOWED_STEP_KEYS",
    "REACT_REQUIRED_STEP_KEYS",
    # Blackboard
    "BLACKBOARD_STATUS_EMPTY",
    "BLACKBOARD_STATUS_PLANNED",
    "BLACKBOARD_STATUS_PREPARED",
    "BLACKBOARD_STATUS_EXECUTED",
    "BLACKBOARD_STATUS_FAILED",
    "BLACKBOARD_VALID_STATUSES",
    "BLACKBOARD_STEP_KEY",
    "BLACKBOARD_TOOL_KEY",
    "BLACKBOARD_ARGS_KEY",
    "BLACKBOARD_RESOLVED_ARGS_KEY",
    "BLACKBOARD_RESULT_KEY",
    "BLACKBOARD_ERROR_KEY",
    "BLACKBOARD_STATUS_KEY",
    "BLACKBOARD_STEP_DEPENDENCIES_KEY",
    "BLACKBOARD_AWAIT_KEY",
    "BLACKBOARD_AWAIT_STEP_KEY",
    "BLACKBOARD_FROM_DICT_ALLOWED_KEYS",
    # ToolAgent return/collision modes
    "TOOL_AGENT_RETURN_TOOL_NAME",
    "TOOL_AGENT_RETURN_TOOL_NAMESPACE",
    "TOOL_AGENT_RETURN_TOOL_DESCRIPTION",
    "TOOL_AGENT_RETURN_FULL_NAME",
    "NAME_COLLISION_RAISE",
    "NAME_COLLISION_SKIP",
    "NAME_COLLISION_REPLACE",
    "VALID_NAME_COLLISION_MODES",
    # StructuredInvokable
    "STRUCTURED_ABSENT_MODE_RAISE",
    "STRUCTURED_ABSENT_MODE_DROP",
    "STRUCTURED_ABSENT_MODE_FILL",
    "STRUCTURED_ABSENT_MODES",
    "STRUCTURED_RESULT_RETURN_TYPE",
    "STRUCTURED_PASSTHROUGH_MAPPING_NAME",
    "STRUCTURED_PASSTHROUGH_MAPPING_TYPE",
    # Workflow
    "FLOW_RESULT_RETURN_TYPE",
    "OUTPUT_TOPOLOGY_NESTED",
    "OUTPUT_TOPOLOGY_FLATTENED",
    "PARALLEL_INPUT_SHAPE_BROADCAST",
    "PARALLEL_INPUT_SHAPE_NESTED",
    "PARALLEL_OUTPUT_SHAPE_NESTED",
    "PARALLEL_OUTPUT_SHAPE_FLATTENED",
    "PARALLEL_OUTPUT_SHAPE_ENVELOPED",
    "DUPLICATE_KEY_POLICY_RAISE",
    "DUPLICATE_KEY_POLICY_SKIP",
    "DUPLICATE_KEY_POLICY_UPDATE",
    "VALID_PARALLEL_INPUT_SHAPES",
    "VALID_PARALLEL_OUTPUT_SHAPES",
    "VALID_DUPLICATE_KEY_POLICIES",
    "WORKFLOW_KIND_BASIC",
    "WORKFLOW_KIND_SEQUENTIAL",
    "WORKFLOW_KIND_ROUTING",
    "WORKFLOW_KIND_ITERATIVE",
    "WORKFLOW_KIND_PARALLEL",
    "WORKFLOW_KINDS",
    # PyA2Atomic / A2A
    "PYA2A_RESULT_KEY",
    "PYA2A_LIST_INVOKABLES_FUNCTION",
    "PYA2A_GET_INVOKABLE_METADATA_FUNCTION",
    "PYA2A_UNKNOWN_FUNCTION_NAME",
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
    "PYA2A_CONTENT_TYPE_TEXT",
    "PYA2A_CONTENT_TYPE_FUNCTION_CALL",
    "PYA2A_CONTENT_TYPE_FUNCTION_RESPONSE",
    "PYA2A_DEFAULT_VERSION",
    "PYA2A_DEFAULT_HOST",
    "PYA2A_DEFAULT_PORT",
    "PYA2A_DEFAULT_TOOL_NAMESPACE",
    # MCP
    "MCP_TRANSPORT_STDIO",
    "MCP_TRANSPORT_SSE",
    "MCP_TRANSPORT_STREAMABLE_HTTP",
    "MCP_TRANSPORT_MODES",
    "MCP_DEFAULT_TOOL_NAMESPACE",
    "MCP_METADATA_NAME_KEY",
    "MCP_METADATA_DESCRIPTION_KEY",
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
    # Engine attachment/file policy
    "ATTACHMENT_EXT_ZIP",
    "ATTACHMENT_EXT_TAR",
    "ATTACHMENT_EXT_GZ",
    "ATTACHMENT_EXT_TGZ",
    "ATTACHMENT_EXT_RAR",
    "ATTACHMENT_EXT_7Z",
    "ATTACHMENT_EXT_EXE",
    "ATTACHMENT_EXT_DLL",
    "ATTACHMENT_EXT_SO",
    "ATTACHMENT_EXT_BIN",
    "ATTACHMENT_EXT_O",
    "ATTACHMENT_EXT_DB",
    "ATTACHMENT_EXT_SQLITE",
    "ATTACHMENT_EXT_H5",
    "ATTACHMENT_EXT_PT",
    "ATTACHMENT_EXT_PTH",
    "ATTACHMENT_EXT_ONNX",
    "ILLEGAL_ATTACHMENT_EXTS",
    "IMAGE_ATTACHMENT_EXTS",
    "TEXT_ATTACHMENT_EXTS",
    "PDF_ATTACHMENT_EXT",
    "ILLEGAL_MIME_PREFIX_AUDIO",
    "ILLEGAL_MIME_PREFIX_VIDEO",
    "ILLEGAL_MIME_PREFIXES",
    "ATTACHMENT_KIND_PDF",
    "ATTACHMENT_KIND_IMAGE",
    "ATTACHMENT_KIND_TEXT",
    "ATTACHMENT_KIND_FILE",
    "ATTACHMENT_META_KIND_KEY",
    "ATTACHMENT_META_MIME_KEY",
    "ATTACHMENT_META_EXT_KEY",
    "ATTACHMENT_META_UPLOADED_KEY",
    "ATTACHMENT_META_FILE_ID_KEY",
    "ATTACHMENT_META_FILE_OBJ_KEY",
    "ATTACHMENT_META_SIGNED_URL_KEY",
    "ATTACHMENT_META_INLINED_KEY",
    "ATTACHMENT_META_INLINED_TEXT_KEY",
    "OPENAI_INPUT_FILE_TYPE",
    "OPENAI_INPUT_IMAGE_TYPE",
    "OPENAI_INPUT_TEXT_TYPE",
    "OPENAI_OUTPUT_TEXT_TYPE",
    "MISTRAL_TEXT_PART_TYPE",
    "MISTRAL_IMAGE_URL_PART_TYPE",
]
