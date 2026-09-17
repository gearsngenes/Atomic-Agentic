from __future__ import annotations

import pytest

from atomic_agentic.exceptions import (
    AgentError,
    AgentInvocationError,
    BlackboardParseError,
    DependencyFailedError,
    ExecutionError,
    LLMEngineError,
    PackagingError,
    RemoteInvocationError,
    SchemaError,
    ToolAgentError,
    ToolDefinitionError,
    ToolError,
    ToolInvocationError,
    ToolRegistrationError,
    ValidationError,
    WorkflowError,
)
from atomic_agentic.utils.script import parse_statement_to_slots


class TestLLMEngineErrors:
    def test_llm_engine_error_is_runtime_error(self) -> None:
        assert issubclass(LLMEngineError, RuntimeError)

    def test_llm_engine_error_can_be_raised_and_caught(self) -> None:
        with pytest.raises(LLMEngineError):
            raise LLMEngineError("engine failed")


class TestToolErrors:
    def test_tool_definition_error_inherits_tool_error(self) -> None:
        assert issubclass(ToolDefinitionError, ToolError)

    def test_tool_invocation_error_inherits_tool_error(self) -> None:
        assert issubclass(ToolInvocationError, ToolError)

    @pytest.mark.parametrize(
        "error_type",
        [
            ToolDefinitionError,
            ToolInvocationError,
        ],
    )
    def test_tool_errors_can_be_caught_as_tool_error(
        self,
        error_type: type[Exception],
    ) -> None:
        with pytest.raises(ToolError):
            raise error_type("tool error")


class TestAgentErrors:
    def test_agent_invocation_error_inherits_agent_error(self) -> None:
        assert issubclass(AgentInvocationError, AgentError)

    def test_agent_error_inherits_runtime_error(self) -> None:
        assert issubclass(AgentError, RuntimeError)

    def test_agent_invocation_error_can_be_caught_as_agent_error(self) -> None:
        with pytest.raises(AgentError):
            raise AgentInvocationError("agent invocation failed")


class TestToolAgentErrors:
    def test_tool_registration_error_inherits_tool_agent_error(self) -> None:
        assert issubclass(ToolRegistrationError, ToolAgentError)

    def test_tool_agent_error_inherits_runtime_error(self) -> None:
        assert issubclass(ToolAgentError, RuntimeError)

    def test_tool_registration_error_can_be_caught_as_tool_agent_error(self) -> None:
        with pytest.raises(ToolAgentError):
            raise ToolRegistrationError("registration failed")


class TestWorkflowErrors:
    def test_validation_error_inherits_workflow_error_and_value_error(self) -> None:
        assert issubclass(ValidationError, WorkflowError)
        assert issubclass(ValidationError, ValueError)

    def test_schema_error_inherits_validation_error(self) -> None:
        assert issubclass(SchemaError, ValidationError)

    def test_packaging_error_inherits_validation_error(self) -> None:
        assert issubclass(PackagingError, ValidationError)

    def test_execution_error_inherits_workflow_error_and_runtime_error(self) -> None:
        assert issubclass(ExecutionError, WorkflowError)
        assert issubclass(ExecutionError, RuntimeError)

    @pytest.mark.parametrize(
        "error_type",
        [
            ValidationError,
            SchemaError,
            PackagingError,
        ],
    )
    def test_validation_family_can_be_caught_as_validation_error(
        self,
        error_type: type[Exception],
    ) -> None:
        with pytest.raises(ValidationError):
            raise error_type("validation failed")

    @pytest.mark.parametrize(
        "error_type",
        [
            ValidationError,
            SchemaError,
            PackagingError,
            ExecutionError,
        ],
    )
    def test_workflow_errors_can_be_caught_as_workflow_error(
        self,
        error_type: type[Exception],
    ) -> None:
        with pytest.raises(WorkflowError):
            raise error_type("workflow failed")


class TestRemoteInvocationError:
    def test_remote_invocation_error_is_exception(self) -> None:
        assert issubclass(RemoteInvocationError, Exception)

    def test_carries_message_error_type_and_function_name(self) -> None:
        exc = RemoteInvocationError(
            "something went wrong",
            error_type="ToolInvocationError",
            function_name="my_tool",
        )

        assert str(exc) == "something went wrong"
        assert exc.error_type == "ToolInvocationError"
        assert exc.function_name == "my_tool"

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(RemoteInvocationError) as exc_info:
            raise RemoteInvocationError(
                "remote failure",
                error_type="AgentError",
                function_name="run_agent",
            )

        assert exc_info.value.error_type == "AgentError"
        assert exc_info.value.function_name == "run_agent"


class TestBlackboardParseError:
    def test_is_a_runtime_error_not_a_tool_agent_error(self) -> None:
        # Sibling to ToolAgentError, not a subclass -- ScriptAgent is a new
        # sibling family, not a ToolAgent subclass.
        assert issubclass(BlackboardParseError, RuntimeError)
        assert not issubclass(BlackboardParseError, ToolAgentError)

    def test_can_be_raised_and_caught_with_message(self) -> None:
        with pytest.raises(BlackboardParseError, match="boom"):
            raise BlackboardParseError("boom")

    def test_chains_the_original_exception_via_cause(self) -> None:
        with pytest.raises(BlackboardParseError) as exc_info:
            parse_statement_to_slots("x = 1 / 0")

        assert isinstance(exc_info.value.__cause__, ZeroDivisionError)


class TestDependencyFailedError:
    """
    Standalone coverage only -- this class is exported but not wired into
    any live ScriptAgent code path yet (see its own docstring: "Raised by a
    future prepare()-phase caller, not by this release's own utils"). No
    test here asserts it's triggered by a real parse/resolve/dispatch call.
    """

    def test_message_format(self) -> None:
        cause = ValueError("boom")
        exc = DependencyFailedError("arg_name", "dep_id", cause)

        assert str(exc) == "argument 'arg_name' depends on 'dep_id', which failed: ValueError('boom')"

    def test_stores_arg_name_dependency_identifier_and_cause_unstringified(self) -> None:
        cause = ValueError("boom")
        exc = DependencyFailedError("arg_name", "dep_id", cause)

        assert exc.arg_name == "arg_name"
        assert exc.dependency_identifier == "dep_id"
        assert exc.cause is cause

    def test_cause_chaining_bottoms_out_at_the_real_originating_exception(self) -> None:
        root = ValueError("root")
        inner = DependencyFailedError("c", "d", root)
        outer = DependencyFailedError("a", "b", inner)

        assert outer.cause is inner
        assert outer.cause.cause is root
