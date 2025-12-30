# ───────────────────────────────────────────────────────────────────────────────
# Exceptions
# ───────────────────────────────────────────────────────────────────────────────
class LLMEngineError(RuntimeError):
    """Raised when an LLM engine fails to complete an invocation."""

class ToolError(Exception):
    """Base exception for Tool-related errors."""


class ToolDefinitionError(ToolError):
    """Raised when a callable is incompatible at Tool construction time."""


class ToolInvocationError(ToolError):
    """Raised when inputs are invalid for invocation or binding fails."""


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
    """Raised when a workflow fails to execute in runtime"""
