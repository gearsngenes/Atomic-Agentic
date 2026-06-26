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


__all__ = [
    "LLMEngineError",
    "ToolError",
    "ToolDefinitionError",
    "ToolInvocationError",
    "AgentError",
    "AgentInvocationError",
    "ToolAgentError",
    "ToolRegistrationError",
    "WorkflowError",
    "ValidationError",
    "SchemaError",
    "PackagingError",
    "ExecutionError",
    "RemoteInvocationError",
]
