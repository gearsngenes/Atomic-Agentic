from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import MethodType
from typing import Any

import pytest

from atomic_agentic.core.Exceptions import ExecutionError, ValidationError
from atomic_agentic.core.Parameters import ParamSpec
from atomic_agentic.core.sentinels import NO_VAL
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.StructuredInvokable import (
    StructuredInvokable,
    StructuredResultDict,
)
from atomic_agentic.workflows.base import FlowResultDict, Workflow
from atomic_agentic.workflows.basic import BasicFlow
from atomic_agentic.workflows.metadata import WorkflowRunMetadata


def value_param() -> ParamSpec:
    return ParamSpec(
        name="value",
        index=0,
        kind=ParamSpec.POSITIONAL_OR_KEYWORD,
        type="Any",
        default=NO_VAL,
    )


def return_mapping(value: Any) -> dict[str, Any]:
    """Return a structured mapping."""
    return {"value": value}


def return_scalar(value: Any) -> Any:
    """Return a raw scalar."""
    return value


def make_structured_component(
    *,
    filter_extraneous_inputs: bool = True,
) -> StructuredInvokable:
    tool = Tool(
        function=return_scalar,
        name="return_scalar",
        namespace="tests",
        description="Return raw scalar.",
        filter_extraneous_inputs=filter_extraneous_inputs,
    )
    return StructuredInvokable(
        component=tool,
        output_schema=["value"],
        name="structured_scalar",
        description="Structured scalar.",
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

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[WorkflowRunMetadata, Mapping[str, Any]]:
        self.async_run_inputs.append(dict(inputs))
        return WorkflowRunMetadata(kind="async_echo"), {"value": inputs["value"]}


class TestBasicFlowConstruction:
    def test_constructor_accepts_structured_invokable(self) -> None:
        component = make_structured_component()

        flow = BasicFlow(component=component)

        assert flow.component is component
        assert flow.name == component.name
        assert flow.parameters == component.parameters

    def test_constructor_accepts_workflow(self) -> None:
        component = EchoWorkflow()

        flow = BasicFlow(component=component)

        assert flow.component is component
        assert flow.name == component.name
        assert flow.parameters == component.parameters

    def test_constructor_rejects_raw_tool(self) -> None:
        tool = Tool(
            function=return_mapping,
            name="return_mapping",
            namespace="tests",
            description="Return mapping.",
        )

        with pytest.raises(TypeError, match="StructuredInvokable or Workflow"):
            BasicFlow(component=tool)  # type: ignore[arg-type]

    def test_inherits_component_metadata_and_filter_flag_by_default(self) -> None:
        component = EchoWorkflow(filter_extraneous_inputs=False)

        flow = BasicFlow(component=component)

        assert flow.name == "echo_workflow"
        assert flow.description == "Echo workflow."
        assert flow.parameters == component.parameters
        assert flow.filter_extraneous_inputs is False

    def test_explicit_name_description_and_filter_override_component(self) -> None:
        component = EchoWorkflow(filter_extraneous_inputs=False)

        flow = BasicFlow(
            component=component,
            name="wrapped_echo",
            description="Wrapped echo.",
            filter_extraneous_inputs=True,
        )

        assert flow.name == "wrapped_echo"
        assert flow.description == "Wrapped echo."
        assert flow.filter_extraneous_inputs is True


class TestBasicFlowStructuredChild:
    def test_invoke_delegates_to_structured_child_and_returns_outer_flow_result(
        self,
    ) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        result = flow.invoke({"value": 123})

        assert isinstance(result, FlowResultDict)
        assert result == {"value": 123}
        assert result.run_id == flow.latest_run
        assert len(flow.checkpoints) == 1

    def test_structured_child_metadata_records_raw_result(self) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        result = flow.invoke({"value": 123})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.kind == "basic"
        assert metadata.child_is_workflow is False
        assert metadata.child_id == component.instance_id
        assert metadata.child_full_name == component.full_name
        assert metadata.child_run_id is NO_VAL
        assert metadata.child_raw_result == 123
        assert metadata.has_child_raw_result is True
        assert metadata.child_raw_result_type == "int"

    def test_structured_child_async_metadata_mirrors_sync_path(self) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        result = asyncio.run(flow.async_invoke({"value": 123}))
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert result == {"value": 123}
        assert metadata.child_is_workflow is False
        assert metadata.child_raw_result == 123
        assert metadata.has_child_raw_result is True


class TestBasicFlowWorkflowChild:
    def test_invoke_delegates_to_workflow_child_and_returns_outer_flow_result(
        self,
    ) -> None:
        child = EchoWorkflow()
        flow = BasicFlow(component=child)

        result = flow.invoke({"value": 123})

        assert isinstance(result, FlowResultDict)
        assert result == {"value": 123}
        assert child.run_inputs == [{"value": 123}]
        assert len(child.checkpoints) == 1
        assert len(flow.checkpoints) == 1

    def test_workflow_child_metadata_records_child_run_id(self) -> None:
        child = EchoWorkflow()
        flow = BasicFlow(component=child)

        result = flow.invoke({"value": 123})
        child_result = child.checkpoints[0]
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.kind == "basic"
        assert metadata.child_is_workflow is True
        assert metadata.child_id == child.instance_id
        assert metadata.child_full_name == child.full_name
        assert metadata.child_run_id == child_result.run_id
        assert metadata.child_raw_result is NO_VAL
        assert metadata.has_child_raw_result is False
        assert metadata.child_raw_result_type == "Any"

    def test_workflow_child_async_metadata_mirrors_sync_path(self) -> None:
        child = EchoWorkflow()
        flow = BasicFlow(component=child)

        result = asyncio.run(flow.async_invoke({"value": 123}))
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert result == {"value": 123}
        assert child.async_run_inputs == [{"value": 123}]
        assert metadata.child_is_workflow is True
        assert metadata.child_run_id == child.checkpoints[0].run_id
        assert metadata.child_raw_result is NO_VAL


class TestBasicFlowValidationAndErrors:
    def test_run_raises_validation_error_if_component_returns_non_mapping(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        monkeypatch.setattr(component, "invoke", lambda inputs: 123)

        with pytest.raises(ValidationError, match="non-mapping result"):
            flow._run({"value": 123})

    def test_public_invoke_wraps_run_validation_error_as_execution_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        monkeypatch.setattr(component, "invoke", lambda inputs: 123)

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 123})

    def test_async_run_raises_validation_error_if_component_returns_non_mapping(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        async def bad_async_invoke(self: StructuredInvokable, inputs: Mapping[str, Any]) -> int:
            return 123

        monkeypatch.setattr(
            component,
            "async_invoke",
            MethodType(bad_async_invoke, component),
        )

        with pytest.raises(ValidationError, match="non-mapping async result"):
            asyncio.run(flow._async_run({"value": 123}))

    def test_build_metadata_rejects_plain_dict_result(self) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        with pytest.raises(ValidationError, match="expected StructuredResultDict or FlowResultDict"):
            flow._build_metadata({"value": 123})

    def test_build_metadata_rejects_flow_result_from_structured_child(self) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        with pytest.raises(ValidationError, match="wrapped structured child"):
            flow._build_metadata(FlowResultDict({"value": 123}, run_id="child_run"))

    def test_build_metadata_rejects_structured_result_from_workflow_child(self) -> None:
        child = EchoWorkflow()
        flow = BasicFlow(component=child)

        with pytest.raises(ValidationError, match="wrapped workflow child"):
            flow._build_metadata(
                StructuredResultDict({"value": 123}, raw_result=123)
            )


class TestBasicFlowSerialization:
    def test_to_dict_includes_component_snapshot(self) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        data = flow.to_dict()

        assert data["type"] == "BasicFlow"
        assert data["component"]["type"] == "StructuredInvokable"
        assert data["component"]["name"] == component.name

    def test_to_dict_includes_base_checkpoint_summary_after_invocation(self) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        result = flow.invoke({"value": 123})
        data = flow.to_dict()

        assert data["checkpoint_count"] == 1
        assert data["runs"] == [result.run_id]
        assert data["component"]["type"] == "StructuredInvokable"
