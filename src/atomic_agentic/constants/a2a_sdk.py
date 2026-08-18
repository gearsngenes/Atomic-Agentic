from __future__ import annotations

from a2a.types import TaskState

# =============================================================================
# a2a-sdk Transport Mode Literals
# =============================================================================
# Used by:
# - a2a/A2AClientHub.py: transport_mode constructor validation and
#   ClientConfig.supported_protocol_bindings selection
#
# Sibling to constants/python_a2a.py -- kept in a separate module, matching
# remote-subsystems' deliberate PyA2Atomic*-vs-A2A* naming split, so the two
# A2A tracks (the existing python_a2a-backed one and this new, additive
# a2a-sdk-backed one) don't bleed into each other.
#
# Values are verified byte-for-byte against the installed a2a-sdk's own
# a2a.utils.constants.TransportProtocol (a str, Enum):
# TransportProtocol.JSONRPC.value == "JSONRPC", .HTTP_JSON.value ==
# "HTTP+JSON", .GRPC.value == "GRPC" -- so a value from this module passes
# straight into ClientConfig.supported_protocol_bindings with zero
# translation.


TRANSPORT_JSON_RPC = "JSONRPC"
TRANSPORT_REST = "HTTP+JSON"
TRANSPORT_GRPC = "GRPC"

VALID_TRANSPORT_MODES: tuple[str, ...] = (
    TRANSPORT_JSON_RPC,
    TRANSPORT_REST,
    TRANSPORT_GRPC,
)


# =============================================================================
# a2a-sdk TaskState Terminal-Failure Set
# =============================================================================
# Used by:
# - a2a/A2AClientHub.py: send_parts/async_send_parts's terminal-failure check
#
# The TaskState values a remote task can end in that A2AClientHub treats as
# unrecoverable failures (raises A2AProxyError). Kept here, not as a
# module-local constant inside A2AClientHub.py, since it describes a2a-sdk's
# own TaskState semantics -- a domain constant, not implementation-private
# state.

TERMINAL_FAILURE_STATES: frozenset[TaskState] = frozenset(
    {
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_REJECTED,
    }
)


# =============================================================================
# A2AtomicExecutor / A2AClientHub Shared Conventions
# =============================================================================
# Used by:
# - a2a/A2AtomicExecutor.py: card extension publication (PARAM_SCHEMA_EXT_URI),
#   Message.metadata routing (SKILL_ROUTING_KEY), result wrapping (ATOMIC_RESULT_KEY)
# - a2a/A2AClientHub.py: extension detection, call_atomic_skill/
#   async_call_atomic_skill routing and result unwrapping
#
# PARAM_SCHEMA_EXT_URI identifies AA's own AgentExtension on a published
# AgentCard; its Struct payload is a skill_id -> A2AtomicSkillMetadata.to_dict()
# map. SKILL_ROUTING_KEY is the Message.metadata key naming which registered
# skill a call targets -- lives in metadata, not inside the DataPart's own
# content, since no skill-selection field exists anywhere on Message itself
# (confirmed against the full field list in a2a_pb2.pyi). ATOMIC_RESULT_KEY
# wraps the single returned value, the direct a2a-sdk-track analog of
# constants/python_a2a.py's PYA2A_RESULT_KEY.

PARAM_SCHEMA_EXT_URI = "atomic-agentic.dev/ext/param-schema/v1"
SKILL_ROUTING_KEY = "skill_id"
ATOMIC_RESULT_KEY = "__atomic_result__"


# =============================================================================
# A2AProxyTool generic-mode Part<->dict key vocabulary
# =============================================================================
# Used by:
# - utils/a2a.py: _dict_to_part/_part_to_dict conversion
# - tools/a2a_sdk.py: A2AProxyTool generic-mode signature description
#
# Each generic-mode `parts` item is a plain dict describing exactly one
# a2a-sdk Part. PART_CONTENT_KEYS names the four mutually-exclusive content
# keys -- exactly one must be non-None per dict.

PART_TEXT_KEY = "text"
PART_DATA_KEY = "data"
PART_RAW_B64_KEY = "raw_b64"
PART_URL_KEY = "url"
PART_FILENAME_KEY = "filename"
PART_MEDIA_TYPE_KEY = "media_type"

PART_CONTENT_KEYS: tuple[str, ...] = (
    PART_TEXT_KEY,
    PART_DATA_KEY,
    PART_RAW_B64_KEY,
    PART_URL_KEY,
)


# =============================================================================
# A2AProxyTool generic-mode ParamSpec descriptions
# =============================================================================
# Used by:
# - tools/a2a_sdk.py: A2AProxyTool._build_tool_signature's generic-mode
#   "parts"/"metadata" ParamSpecs
#
# Spelled out in full (every key, its type, when to set it) rather than a
# terse one-liner -- this text is the only schema documentation an LLM
# tool-agent sees before constructing a call, since generic mode's dict
# shape isn't enforced by any structured JSON-schema layer of its own.

GENERIC_PARTS_DESCRIPTION = (
    "Sequence of dicts; each dict describes exactly one A2A message part and "
    "should include all six keys: 'text', 'data', 'raw_b64', 'url', "
    "'filename', 'media_type'. Set exactly ONE of 'text'/'data'/'raw_b64'/"
    "'url' to a non-null value -- that is the part's content, and picks its "
    "kind -- and leave the other three of those four null. "
    "'text' (str): plain prose. "
    "'data' (any JSON value): a dict/list/str/number/bool/null payload. "
    "'raw_b64' (str): base64-encoded binary content. "
    "'url' (str): a URL reference to external content. "
    "'filename' (str, optional, null unless set): only meaningful alongside "
    "'raw_b64' or 'url'. "
    "'media_type' (str, optional, null unless set): a MIME type describing "
    "the content, e.g. 'text/plain', 'application/json', 'image/png', "
    "'application/pdf'; valid alongside any content kind. "
    "The call's result is a list of dicts in this exact same shape."
)

GENERIC_METADATA_DESCRIPTION = (
    "Optional dict of string keys/values, passed through unchanged as "
    "message-level metadata to the remote agent -- not part of any Part's "
    "content. Used only if the remote agent defines its own routing/session "
    "metadata convention; leave null (the default) otherwise."
)


# =============================================================================
# Explicit public export list
# =============================================================================
# Keep this explicit so adding local helper names or imports cannot accidentally
# widen the module's public surface.


__all__ = [
    "TRANSPORT_JSON_RPC",
    "TRANSPORT_REST",
    "TRANSPORT_GRPC",
    "VALID_TRANSPORT_MODES",
    "TERMINAL_FAILURE_STATES",
    "PARAM_SCHEMA_EXT_URI",
    "SKILL_ROUTING_KEY",
    "ATOMIC_RESULT_KEY",
    "PART_TEXT_KEY",
    "PART_DATA_KEY",
    "PART_RAW_B64_KEY",
    "PART_URL_KEY",
    "PART_FILENAME_KEY",
    "PART_MEDIA_TYPE_KEY",
    "PART_CONTENT_KEYS",
    "GENERIC_PARTS_DESCRIPTION",
    "GENERIC_METADATA_DESCRIPTION",
]
