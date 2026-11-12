from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from atomic_agentic.exceptions import ExecutionError
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.workflows.base import Workflow
from atomic_agentic.models.results.workflows import WorkflowResult


def make_value_param() -> ParamSpec:
    return ParamSpec(
        name="value",
        index=0,
        kind=ParamSpec.POSITIONAL_OR_KEYWORD,
        type="int",
    )


class EchoWorkflow(Workflow):
    """Minimal Workflow: echoes ``inputs["value"]`` back as the payload."""

    def __init__(
        self,
        *,
        name: str = "echo_workflow",
        namespace: str = "tests",
        description: str = "Echo workflow.",
        parameters: list[ParamSpec] | None = None,
        return_type: str = "dict[str, Any]",
        include_trace: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            parameters=parameters if parameters is not None else [make_value_param()],
            return_type=return_type,
            include_trace=include_trace,
        )

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"value": inputs["value"]}, {}


class NativeAsyncEchoWorkflow(EchoWorkflow):
    """EchoWorkflow with a directly-defined ``async def _async_run``."""

    async def _async_run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"value": inputs["value"]}, {}


class TraceEmittingWorkflow(Workflow):
    """Workflow whose ``_run``/``_async_run`` explicitly populate ``trace``."""

    def __init__(self, *, name: str = "trace_workflow", namespace: str = "tests") -> None:
        super().__init__(
            name=name,
            namespace=namespace,
            description="Trace-emitting workflow.",
            parameters=[make_value_param()],
            return_type="dict[str, Any]",
        )

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"value": inputs["value"]}, {"trace": ("child-a", "child-b")}

    async def _async_run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"value": inputs["value"]}, {"trace": ("child-a", "child-b")}


class ConfigurableWorkflow(Workflow):
    """Workflow whose run hooks return a configured result or raise."""

    def __init__(
        self,
        name: str,
        namespace: str = "tests",
        description: str = "Configurable workflow.",
        parameters: list[ParamSpec] | None = None,
        return_type: str = "Any",
        *,
        result: Any = None,
        run_error: Exception | None = None,
        async_error: Exception | None = None,
    ) -> None:
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            parameters=parameters if parameters is not None else [make_value_param()],
            return_type=return_type,
        )
        self._result = result
        self._run_error = run_error
        self._async_error = async_error

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        if self._run_error is not None:
            raise self._run_error
        return self._result, {}

    async def _async_run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        if self._async_error is not None:
            raise self._async_error
        return self._result, {}


class TestWorkflowNamespace:
    def test_workflow_namespace_default(self) -> None:
        workflow = EchoWorkflow()
        assert workflow.namespace == "tests"

    def test_workflow_namespace_explicit(self) -> None:
        workflow = ConfigurableWorkflow(
            name="wf",
            namespace="wf_ns",
        )
        assert workflow.namespace == "wf_ns"


class TestWorkflowConstruction:
    def test_initializes_name_description_parameters_return_type(self) -> None:
        params = [make_value_param()]
        workflow = EchoWorkflow(
            name="my_workflow",
            description="A workflow.",
            parameters=params,
            return_type="dict[str, Any]",
        )

        assert workflow.name == "my_workflow"
        assert workflow.description == "A workflow."
        assert workflow.parameters == params
        assert workflow.return_type == "dict[str, Any]"

    def test_include_trace_defaults_to_true(self) -> None:
        workflow = EchoWorkflow()

        assert workflow.include_trace is True


class TestWorkflowIncludeTrace:
    def test_include_trace_settable_post_construction(self) -> None:
        workflow = EchoWorkflow()

        workflow.include_trace = False

        assert workflow.include_trace is False

    def test_include_trace_setter_rejects_non_bool(self) -> None:
        workflow = EchoWorkflow()

        with pytest.raises(TypeError, match="include_trace must be a bool"):
            workflow.include_trace = "yes"  # type: ignore[assignment]

    def test_include_trace_accepted_at_construction(self) -> None:
        workflow = EchoWorkflow(include_trace=False)

        assert workflow.include_trace is False


class TestWorkflowSyncInvoke:
    def test_invoke_returns_workflow_result(self) -> None:
        workflow = EchoWorkflow()

        result = workflow.invoke({"value": 123})

        assert isinstance(result, WorkflowResult)
        assert result.result == {"value": 123}
        assert isinstance(result.run_id, str)
        assert result.started_at <= result.ended_at
        assert result.elapsed_s >= 0

    def test_invoke_filters_extraneous_inputs(self) -> None:
        workflow = EchoWorkflow()

        result = workflow.invoke({"value": 123, "extra": "ignored"})

        assert result.result == {"value": 123}

    def test_trace_defaults_to_none_when_run_omits_it(self) -> None:
        workflow = EchoWorkflow()

        result = workflow.invoke({"value": 1})

        assert result.trace is None

    def test_trace_populated_when_run_hook_supplies_it(self) -> None:
        workflow = TraceEmittingWorkflow()

        result = workflow.invoke({"value": 1})

        assert result.trace == ("child-a", "child-b")


class TestWorkflowAsyncInvoke:
    def test_async_invoke_with_native_async_run(self) -> None:
        workflow = NativeAsyncEchoWorkflow()

        result = asyncio.run(workflow.async_invoke({"value": 1}))

        assert isinstance(result, WorkflowResult)
        assert result.result == {"value": 1}

    def test_async_invoke_default_dispatches_sync_run_via_thread(self) -> None:
        workflow = EchoWorkflow()

        result = asyncio.run(workflow.async_invoke({"value": 1}))

        assert isinstance(result, WorkflowResult)
        assert result.result == {"value": 1}

    def test_async_trace_populated_when_async_run_hook_supplies_it(self) -> None:
        workflow = TraceEmittingWorkflow()

        result = asyncio.run(workflow.async_invoke({"value": 1}))

        assert result.trace == ("child-a", "child-b")


class TestWorkflowValidationAndErrors:
    def test_invoke_wraps_run_exception_as_execution_error(self) -> None:
        workflow = ConfigurableWorkflow(
            name="configurable_workflow",
            description="Configurable workflow.",
            parameters=[make_value_param()],
            return_type="dict[str, Any]",
            run_error=RuntimeError("boom"),
        )

        with pytest.raises(ExecutionError, match="_run failed"):
            workflow.invoke({"value": 1})

    def test_async_invoke_wraps_async_run_exception_as_execution_error(self) -> None:
        workflow = ConfigurableWorkflow(
            name="configurable_workflow",
            description="Configurable workflow.",
            parameters=[make_value_param()],
            return_type="dict[str, Any]",
            async_error=RuntimeError("boom"),
        )

        with pytest.raises(ExecutionError, match="_async_run failed"):
            asyncio.run(workflow.async_invoke({"value": 1}))


class TestWorkflowSerialization:
    def test_to_dict_includes_include_trace_and_type(self) -> None:
        workflow = EchoWorkflow(include_trace=False)

        data = workflow.to_dict()

        assert data["include_trace"] is False
        assert data["type"] == "EchoWorkflow"
        assert "checkpoints" not in data
        assert "checkpoint_count" not in data
        assert "runs" not in data
