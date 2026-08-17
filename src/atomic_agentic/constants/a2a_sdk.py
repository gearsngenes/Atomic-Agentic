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
]
