class LLMEngineError(RuntimeError):
    """Raised when an LLM engine fails to complete an invocation."""


class ToolError(Exception):
    """Base exception for Tool-related errors."""


class ToolDefinitionError(ToolError):
    """Raised when a callable is incompatible at Tool construction time."""


class ToolInvocationError(ToolError):
    """Raised when tool invocation fails, including input binding failures and
    execution-time exceptions thrown by the underlying callable."""


class AgentError(RuntimeError):
    """Base class for Agent-related errors."""


class AgentInvocationError(AgentError):
    """Raised when an Agent fails to prepare or process an invocation."""


class ToolAgentError(RuntimeError):
    """Base exception for ToolAgent-related errors."""


class ToolRegistrationError(ToolAgentError):
    """Raised when registering tools fails due to collisions or bad inputs."""


class ThinkingAgentError(RuntimeError):
    """Base exception for SelfAskAgent thinking-phase errors.

    Sibling to AgentError, not a subclass of it -- mirrors ToolAgentError's
    own convention (a domain error type superseding a bare RuntimeError;
    callers doing `except RuntimeError` upstream still catch these).
    """


class WorkflowError(Exception):
    """Base class for workflow-related errors."""


class ValidationError(WorkflowError, ValueError):
    """Raised for input/type validation failures."""


class SchemaError(ValidationError):
    """Raised when `output_schema` is malformed or incompatible with options."""


class PackagingError(ValidationError):
    """Raised when a raw result cannot be normalized to `output_schema`."""


class ExecutionError(WorkflowError, RuntimeError):
    """Raised when a workflow fails to execute at runtime."""


class RemoteInvocationError(Exception):
    """Raised when a remote host reports a host-side execution failure via error payload.

    Distinct from connection-level failures (which surface as RuntimeError).
    ``error_type`` carries the remote exception class name as a string.
    ``function_name`` identifies which invokable was being called.
    """

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        function_name: str,
    ) -> None:
        super().__init__(message)
        self.error_type: str = error_type
        self.function_name: str = function_name


class MCPError(RuntimeError):
    """Base exception for MCP transport/protocol errors.

    Subclasses RuntimeError to match this codebase's existing convention for
    domain error types that supersede a previously-bare RuntimeError (see
    LLMEngineError) — callers doing `except RuntimeError` upstream still
    catch these.
    """


class MCPConnectionError(MCPError):
    """Raised when MCP transport/session establishment fails, or a
    non-tool-specific protocol operation (e.g. listing tools) fails."""


class MCPToolError(MCPError):
    """Raised when a specific remote MCP tool invocation fails server-side."""


class PyA2AtomicError(RuntimeError):
    """Base exception for PyA2Atomic transport/protocol errors.

    Named PyA2Atomic (matching PyA2AtomicClient/PyA2AtomicHost/
    PyA2AtomicTool), not A2A* — the python_a2a dependency already exports
    its own A2AError/A2AConnectionError hierarchy with the same names but
    a different base (plain Exception, not RuntimeError); reusing those
    names here would make tracebacks genuinely ambiguous about which
    library raised what. Subclasses RuntimeError to match this codebase's
    existing convention for domain error types that supersede a
    previously-bare RuntimeError (see LLMEngineError, MCPError) — callers
    doing `except RuntimeError` upstream still catch these.
    """


class PyA2AtomicConnectionError(PyA2AtomicError):
    """Raised when PyA2Atomic agent-card refresh fails, message transport
    fails, or a remote response doesn't match the expected protocol shape."""


__all__ = [
    "LLMEngineError",
    "ToolError",
    "ToolDefinitionError",
    "ToolInvocationError",
    "AgentError",
    "AgentInvocationError",
    "ToolAgentError",
    "ToolRegistrationError",
    "ThinkingAgentError",
    "WorkflowError",
    "ValidationError",
    "SchemaError",
    "PackagingError",
    "ExecutionError",
    "RemoteInvocationError",
    "MCPError",
    "MCPConnectionError",
    "MCPToolError",
    "PyA2AtomicError",
    "PyA2AtomicConnectionError",
]
