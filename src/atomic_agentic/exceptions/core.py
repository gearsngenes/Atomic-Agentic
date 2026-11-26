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


class BlackboardParseError(RuntimeError):
    """Raised when parsing one raw statement into ToolStatement object(s)
    fails, or when the shared expression-parsing/rejection helpers in
    utils/agents.py (`evaluate_expr`, `reject_unsupported_forms`) reject an
    expression -- PlanActAgent's own statement-parsing mechanism.

    Subclasses RuntimeError to match this codebase's existing convention for
    domain error types superseding a bare RuntimeError (see LLMEngineError,
    MCPError) -- callers doing `except RuntimeError` upstream still catch
    these. Deliberately not rooted in ToolAgentError: a statement that fails
    to parse is regeneration feedback, not an agent failure.

    Unifies every rejection category from parse_statement_to_slots under one
    catchable type: genuine ast.parse SyntaxErrors, illegal assignment
    shapes (multiple targets, tuple/list-unpacking, augmented assignment, an
    attribute/subscript/starred LHS, positional or **-unpacked call
    arguments), a reserved hoisted-name-prefix collision on the input
    statement's own LHS, and a dependency-free expression that raises when
    eagerly evaluated during parsing (e.g. `1/0`). The original exception is
    always preserved via `raise BlackboardParseError(...) from e`.
    """


class DependencyFailedError(Exception):
    """Raised (by a future prepare()-phase caller, not by this release's own
    utils) when a ToolStatement's argument depends on another slot whose
    own resolution failed.

    Does not forward the upstream exception instance verbatim -- wraps it so
    the failure reads as "arg_name depends on dependency_identifier, which
    failed" without losing the root cause. `cause` may itself be a
    DependencyFailedError, so repeatedly walking `.cause` always bottoms out
    at the real originating exception regardless of cascade depth.
    """

    def __init__(self, arg_name: str, dependency_identifier: str, cause: Exception) -> None:
        super().__init__(
            f"argument {arg_name!r} depends on {dependency_identifier!r}, "
            f"which failed: {cause!r}"
        )
        self.arg_name = arg_name
        self.dependency_identifier = dependency_identifier
        self.cause = cause


class SchemaError(ValueError):
    """Raised when a declared parameter schema is malformed or breaks parameter-ordering rules."""


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


class A2AProxyError(RuntimeError):
    """Base exception for A2AClientHub transport/protocol errors.

    Unrelated to a2a-sdk's own A2AError/A2AClientError hierarchy despite the
    similar naming (reusing those names would make tracebacks ambiguous about
    which library raised what); that hierarchy is wrapped, not reused.
    Subclasses RuntimeError to match this codebase's convention for domain
    error types (see LLMEngineError, MCPError). A single exception, not a
    hierarchy; deep exception cleanup is deferred to the error-boundaries
    stage.
    """


__all__ = [
    "LLMEngineError",
    "ToolError",
    "ToolDefinitionError",
    "ToolInvocationError",
    "AgentError",
    "AgentInvocationError",
    "ToolAgentError",
    "ToolRegistrationError",
    "BlackboardParseError",
    "DependencyFailedError",
    "SchemaError",
    "RemoteInvocationError",
    "MCPError",
    "MCPConnectionError",
    "MCPToolError",
    "A2AProxyError",
]
