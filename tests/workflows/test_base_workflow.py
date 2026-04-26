from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from atomic_agentic.core.Exceptions import ExecutionError, ValidationError
from atomic_agentic.core.Parameters import ParamSpec
from atomic_agentic.core.sentinels import NO_VAL
from atomic_agentic.workflows.base import FlowResultDict, Workflow
from atomic_agentic.workflows.metadata import WorkflowRunMetadata


def value_param() -> ParamSpec:
    return ParamSpec(
        name="value",
        index=0,
        kind=ParamSpec.POSITIONAL_OR_KEYWORD,
        type="Any",
        default=NO_VAL,
    )


class EchoWorkflow(Workflow[WorkflowRunMetadata]):
    def __init__(
        self,
        *,
        name: str = "echo_workflow",
        description: str = "Echo workflow.",
        filter_extraneous_inputs: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            parameters=[value_param()],
            filter_extraneous_inputs=filter_extraneous_inputs,
        )
        self.run_inputs: list[dict[str, Any]] = []
        self.async_run_inputs: list[dict[str, Any]] = []

    def _run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[WorkflowRunMetadata, Mapping[str, Any]]:
        self.run_inputs.append(dict(inputs))
        return WorkflowRunMetadata(kind="echo"), {"value": inputs["value"]}


class NativeAsyncEchoWorkflow(EchoWorkflow):
    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[WorkflowRunMetadata, Mapping[str, Any]]:
        self.async_run_inputs.append(dict(inputs))
        return WorkflowRunMetadata(kind="async_echo"), {"value": inputs["value"]}


class ConfigurableWorkflow(Workflow[WorkflowRunMetadata]):
    def __init__(
        self,
        *,
        metadata: Any = None,
        result: Any = None,
        run_error: Exception | None = None,
        async_error: Exception | None = None,
    ) -> None:
        super().__init__(
            name="configurable_workflow",
            description="Configurable workflow.",
            parameters=[value_param()],
            filter_extraneous_inputs=True,
        )
        self.metadata = (
            WorkflowRunMetadata(kind="configurable")
            if metadata is None
            else metadata
        )
        self.result = {"value": "ok"} if result is None else result
        self.run_error = run_error
        self.async_error = async_error

    def _run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        if self.run_error is not None:
            raise self.run_error
        return self.metadata, self.result

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        if self.async_error is not None:
            raise self.async_error
        return self.metadata, self.result


class TestFlowResultDict:
    def test_copy_preserves_items_and_run_id(self) -> None:
        result = FlowResultDict({"value": 1}, run_id="run_1")

        copied = result.copy()

        assert copied == {"value": 1}
        assert copied.run_id == "run_1"
        assert copied is not result

    def test_run_id_is_not_mapping_item(self) -> None:
        result = FlowResultDict({"value": 1}, run_id="run_1")

        assert "run_id" not in result
        assert result.run_id == "run_1"


class TestWorkflowConstruction:
    def test_initializes_as_atomic_invokable_with_workflow_return_type(self) -> None:
        workflow = EchoWorkflow()

        assert workflow.name == "echo_workflow"
        assert workflow.description == "Echo workflow."
        assert workflow.return_type == "FlowResultDict[str, Any]"
        assert [param.name for param in workflow.parameters] == ["value"]

    def test_checkpoints_initially_empty_and_latest_run_none(self) -> None:
        workflow = EchoWorkflow()

        assert workflow.checkpoints == []
        assert workflow.latest_run is None


class TestWorkflowSyncInvoke:
    def test_invoke_returns_flow_result_dict_with_generated_run_id(self) -> None:
        workflow = EchoWorkflow()

        result = workflow.invoke({"value": 123})

        assert isinstance(result, FlowResultDict)
        assert result == {"value": 123}
        assert isinstance(result.run_id, str)
        assert result.run_id

    def test_invoke_filters_inputs_before_run(self) -> None:
        workflow = EchoWorkflow(filter_extraneous_inputs=True)

        result = workflow.invoke({"value": 123, "extra": "ignored"})

        assert result == {"value": 123}
        assert workflow.run_inputs == [{"value": 123}]

    def test_invoke_records_checkpoint_matching_result(self) -> None:
        workflow = EchoWorkflow()

        result = workflow.invoke({"value": 123})
        checkpoint = workflow.checkpoints[0]

        assert checkpoint.run_id == result.run_id
        assert checkpoint.inputs == {"value": 123}
        assert checkpoint.result == {"value": 123}
        assert checkpoint.metadata.kind == "echo"
        assert checkpoint.started_at <= checkpoint.ended_at
        assert checkpoint.elapsed_s >= 0

    def test_latest_run_and_get_checkpoint_update_after_invoke(self) -> None:
        workflow = EchoWorkflow()

        result = workflow.invoke({"value": 123})

        assert workflow.latest_run == result.run_id
        assert workflow.get_checkpoint(result.run_id) is workflow.checkpoints[0]
        assert workflow.get_checkpoint("missing") is None

    def test_multiple_invokes_produce_multiple_distinct_run_ids(self) -> None:
        workflow = EchoWorkflow()

        first = workflow.invoke({"value": 1})
        second = workflow.invoke({"value": 2})

        assert first.run_id != second.run_id
        assert [checkpoint.run_id for checkpoint in workflow.checkpoints] == [
            first.run_id,
            second.run_id,
        ]
        assert workflow.latest_run == second.run_id


class TestWorkflowMemoryAndSerialization:
    def test_checkpoints_property_returns_shallow_copy(self) -> None:
        workflow = EchoWorkflow()
        workflow.invoke({"value": 1})

        snapshot = workflow.checkpoints
        snapshot.clear()

        assert len(snapshot) == 0
        assert len(workflow.checkpoints) == 1

    def test_clear_memory_clears_checkpoints_and_latest_run(self) -> None:
        workflow = EchoWorkflow()
        workflow.invoke({"value": 1})

        workflow.clear_memory()

        assert workflow.checkpoints == []
        assert workflow.latest_run is None

    def test_to_dict_includes_checkpoint_count_and_runs(self) -> None:
        workflow = EchoWorkflow()

        first = workflow.invoke({"value": 1})
        second = workflow.invoke({"value": 2})
        data = workflow.to_dict()

        assert data["type"] == "EchoWorkflow"
        assert data["checkpoint_count"] == 2
        assert data["runs"] == [first.run_id, second.run_id]
        assert "checkpoints" not in data


class TestWorkflowAsyncInvoke:
    def test_async_invoke_returns_flow_result_dict_and_records_checkpoint(self) -> None:
        workflow = NativeAsyncEchoWorkflow()

        result = asyncio.run(workflow.async_invoke({"value": 123}))

        assert isinstance(result, FlowResultDict)
        assert result == {"value": 123}
        assert workflow.async_run_inputs == [{"value": 123}]
        assert len(workflow.checkpoints) == 1
        assert workflow.checkpoints[0].run_id == result.run_id
        assert workflow.checkpoints[0].metadata.kind == "async_echo"

    def test_default_async_run_dispatches_to_sync_run(self) -> None:
        workflow = EchoWorkflow()

        result = asyncio.run(workflow.async_invoke({"value": 123}))

        assert result == {"value": 123}
        assert workflow.run_inputs == [{"value": 123}]
        assert len(workflow.checkpoints) == 1


class TestWorkflowValidationAndErrors:
    def test_run_exception_is_wrapped_as_execution_error(self) -> None:
        workflow = ConfigurableWorkflow(run_error=RuntimeError("boom"))

        with pytest.raises(ExecutionError, match="_run failed"):
            workflow.invoke({"value": 1})

    def test_async_run_exception_is_wrapped_as_execution_error(self) -> None:
        workflow = ConfigurableWorkflow(async_error=RuntimeError("async boom"))

        with pytest.raises(ExecutionError, match="_async_run failed"):
            asyncio.run(workflow.async_invoke({"value": 1}))

    def test_invalid_metadata_returned_by_run_raises_validation_error(self) -> None:
        workflow = ConfigurableWorkflow(metadata={"kind": "bad"})

        with pytest.raises(ValidationError, match="invalid metadata type"):
            workflow.invoke({"value": 1})

    def test_non_mapping_result_returned_by_run_raises_validation_error(self) -> None:
        workflow = ConfigurableWorkflow(result=["not", "mapping"])

        with pytest.raises(ValidationError, match="non-mapping result"):
            workflow.invoke({"value": 1})

    def test_invalid_metadata_returned_by_async_run_raises_validation_error(self) -> None:
        workflow = ConfigurableWorkflow(metadata={"kind": "bad"})

        with pytest.raises(ValidationError, match="invalid metadata type"):
            asyncio.run(workflow.async_invoke({"value": 1}))

    def test_non_mapping_result_returned_by_async_run_raises_validation_error(self) -> None:
        workflow = ConfigurableWorkflow(result=["not", "mapping"])

        with pytest.raises(ValidationError, match="non-mapping result"):
            asyncio.run(workflow.async_invoke({"value": 1}))
