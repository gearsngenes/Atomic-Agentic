from __future__ import annotations

# =============================================================================
# A2A Protocol Function Names
# =============================================================================
# Used by:
# - a2a/PyA2AtomicHost.py: message dispatch and A2A function registry
# - a2a/PyA2AtomicClient.py: remote function call routing
# - tools/a2a.py: remote metadata fetching
#
# These function names form the core of the PyA2Atomic A2A protocol contract
# between host and client. The Host must define these as exact function names,
# and the Client must call them by these exact names to bootstrap remote
# introspection and function calling.
#
# Contract:
# - list_invokables(): returns dict[str, dict[str, Any]] of all registered invokables
# - get_invokable_metadata(name: str): returns dict[str, Any] for one invokable


LIST_INVOKABLES_FUNCTION = "list_invokables"
GET_INVOKABLE_METADATA_FUNCTION = "get_invokable_metadata"


# =============================================================================
# A2A Result Container Key
# =============================================================================
# Used by:
# - a2a/PyA2AtomicHost.py: wrapping invokable results in A2A response payloads
# - a2a/PyA2AtomicClient.py: extracting invokable results from A2A payloads
#
# When a PyA2AtomicHost executes an invokable, the result is wrapped inside
# a JSON object with this key to distinguish it from A2A infrastructure responses
# (which may contain error or metadata).
#
# Contract:
# - The Host places the raw invokable result at payload[PYA2A_RESULT_KEY]
# - The Client extracts the result by key, raising if key is missing
# - No other keys are expected in this payload during normal result delivery


PYA2A_RESULT_KEY = "__py_a2a_result__"


# =============================================================================
# Explicit public export list
# =============================================================================
# Keep this explicit so adding local helper names or imports cannot accidentally
# widen the module's public surface.


__all__ = [
    "LIST_INVOKABLES_FUNCTION",
    "GET_INVOKABLE_METADATA_FUNCTION",
    "PYA2A_RESULT_KEY",
]
