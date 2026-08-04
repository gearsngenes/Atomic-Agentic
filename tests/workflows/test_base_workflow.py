from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from atomic_agentic.exceptions import ExecutionError
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.workflows.base import Workflow, WorkflowCheckpoint
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
    ) -> None:
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            parameters=parameters if parameters is not None else [make_value_param()],
            return_type=return_type,
        )

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"value": inputs["value"]}, {}


class NativeAsyncEchoWorkflow(EchoWorkflow):
    """EchoWorkflow with a directly-defined ``async def _async_run``."""

    async def _async_run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"value": inputs["value"]}, {}


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

    def test_checkpoints_initially_empty(self) -> None:
        workflow = EchoWorkflow()

        assert workflow.checkpoints == []


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

    def test_invoke_appends_checkpoint_matching_returned_result(self) -> None:
        workflow = EchoWorkflow()

        result = workflow.invoke({"value": 123})
        checkpoint = workflow.checkpoints[-1]

        assert isinstance(checkpoint, WorkflowCheckpoint)
        assert checkpoint.result is result
        assert checkpoint.inputs == {"value": 123}

    def test_get_checkpoint_by_index_run_id_and_result_identity(self) -> None:
        workflow = EchoWorkflow()

        result = workflow.invoke({"value": 123})
        checkpoint = workflow.checkpoints[0]

        assert workflow.get_checkpoint(0) is checkpoint
        assert workflow.get_checkpoint(result.run_id) is checkpoint
        assert workflow.get_checkpoint(result) is checkpoint
        assert workflow.get_checkpoint("missing-run-id") is None
        with pytest.raises(IndexError):
            workflow.get_checkpoint(5)

    def test_multiple_invokes_each_get_distinct_run_ids_and_ordered_checkpoints(
        self,
    ) -> None:
        workflow = EchoWorkflow()

        first = workflow.invoke({"value": 1})
        second = workflow.invoke({"value": 2})

        assert first.run_id != second.run_id
        assert [checkpoint.result.run_id for checkpoint in workflow.checkpoints] == [
            first.run_id,
            second.run_id,
        ]


class TestWorkflowMemoryAndSerialization:
    def test_checkpoints_property_is_shallow_copy(self) -> None:
        workflow = EchoWorkflow()
        workflow.invoke({"value": 1})

        snapshot = workflow.checkpoints
        snapshot.clear()

        assert len(snapshot) == 0
        assert len(workflow.checkpoints) == 1

    def test_clear_memory_empties_checkpoints(self) -> None:
        workflow = EchoWorkflow()
        workflow.invoke({"value": 1})

        workflow.clear_memory()

        assert workflow.checkpoints == []

    def test_to_dict_includes_checkpoint_count_and_runs_not_raw_checkpoints(
        self,
    ) -> None:
        workflow = EchoWorkflow()

        first = workflow.invoke({"value": 1})
        second = workflow.invoke({"value": 2})
        data = workflow.to_dict()

        assert data["checkpoint_count"] == 2
        assert data["runs"] == [first.run_id, second.run_id]
        assert "checkpoints" not in data
        assert data["type"] == "EchoWorkflow"


class TestWorkflowAsyncInvoke:
    def test_async_invoke_with_native_async_run(self) -> None:
        workflow = NativeAsyncEchoWorkflow()

        result = asyncio.run(workflow.async_invoke({"value": 1}))

        assert isinstance(result, WorkflowResult)
        assert result.result == {"value": 1}
        assert len(workflow.checkpoints) == 1
        assert workflow.checkpoints[-1].result is result

    def test_async_invoke_default_dispatches_sync_run_via_thread(self) -> None:
        workflow = EchoWorkflow()

        result = asyncio.run(workflow.async_invoke({"value": 1}))

        assert isinstance(result, WorkflowResult)
        assert result.result == {"value": 1}
        assert len(workflow.checkpoints) == 1


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

    def test_failed_invoke_does_not_append_checkpoint(self) -> None:
        workflow = ConfigurableWorkflow(
            name="configurable_workflow",
            description="Configurable workflow.",
            parameters=[make_value_param()],
            return_type="dict[str, Any]",
            run_error=RuntimeError("boom"),
        )

        with pytest.raises(ExecutionError):
            workflow.invoke({"value": 1})

        assert workflow.checkpoints == []
