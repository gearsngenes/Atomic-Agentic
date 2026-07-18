from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from atomic_agentic.exceptions import ExecutionError
from atomic_agentic.core.Invokable import StructuredInvokable
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results.workflows import BasicFlowResult
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.base import Workflow
from atomic_agentic.workflows.basic import BasicFlow


def make_value_param() -> ParamSpec:
    return ParamSpec(
        name="value",
        index=0,
        kind=ParamSpec.POSITIONAL_OR_KEYWORD,
        type="int",
    )


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


class EchoWorkflow(Workflow):
    """Minimal Workflow: echoes ``inputs["value"]`` back as the payload."""

    def __init__(
        self,
        *,
        name: str = "echo_workflow",
        namespace: str = "tests",
        description: str = "Echo workflow.",
        filter_extraneous_inputs: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            parameters=[make_value_param()],
            return_type="dict[str, Any]",
            filter_extraneous_inputs=filter_extraneous_inputs,
        )

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"value": inputs["value"]}, {}


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

    def test_constructor_rejects_non_invokable_component(self) -> None:
        with pytest.raises(TypeError, match="AtomicInvokable"):
            BasicFlow(component=object())  # type: ignore[arg-type]

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


class TestBasicFlowExtraDescription:
    def test_surfaces_component_extra_description(self) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        assert flow._extra_description() == "Output schema: [value]"
        assert flow.description == "Structured scalar.\nOutput schema: [value]"

    def test_empty_when_component_has_no_extra_description(self) -> None:
        flow = BasicFlow(component=EchoWorkflow())

        assert flow._extra_description() == ""
        assert flow.description == flow._description


class TestBasicFlowStructuredChild:
    def test_invoke_returns_basic_flow_result_with_child_payload(self) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        result = flow.invoke({"value": 123})

        assert isinstance(result, BasicFlowResult)
        assert result.result == {"value": 123}

    def test_basic_flow_result_records_child_identity(self) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        result = flow.invoke({"value": 123})

        assert result.child_id == component.instance_id
        assert result.child_type == type(component).__name__
        assert isinstance(result.child_run_id, str)
        assert result.child_run_id != result.run_id

    def test_checkpoint_matches_returned_result(self) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        result = flow.invoke({"value": 123})

        assert flow.checkpoints[-1].result is result
        assert flow.checkpoints[-1].inputs == {"value": 123}

    def test_async_invoke_returns_basic_flow_result_with_child_payload(self) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        result = asyncio.run(flow.async_invoke({"value": 123}))

        assert isinstance(result, BasicFlowResult)
        assert result.result == {"value": 123}
        assert result.child_id == component.instance_id
        assert result.child_type == type(component).__name__
        assert isinstance(result.child_run_id, str)
        assert flow.checkpoints[-1].result is result


class TestBasicFlowWorkflowChild:
    def test_invoke_delegates_to_workflow_component(self) -> None:
        child = EchoWorkflow()
        flow = BasicFlow(component=child)

        result = flow.invoke({"value": 5})

        assert result.result == {"value": 5}
        assert result.child_type == "EchoWorkflow"
        assert result.child_run_id == child.checkpoints[-1].result.run_id

    def test_async_invoke_delegates_to_workflow_component(self) -> None:
        child = EchoWorkflow()
        flow = BasicFlow(component=child)

        result = asyncio.run(flow.async_invoke({"value": 5}))

        assert result.result == {"value": 5}
        assert result.child_type == "EchoWorkflow"
        assert result.child_run_id == child.checkpoints[-1].result.run_id


class TestBasicFlowValidationAndErrors:
    def test_invoke_wraps_component_exception_as_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        def raise_invoke(inputs: Mapping[str, Any]) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(component, "invoke", raise_invoke)

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 123})

    def test_async_invoke_wraps_component_exception_as_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        async def raise_async_invoke(inputs: Mapping[str, Any]) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(component, "async_invoke", raise_async_invoke)

        with pytest.raises(ExecutionError, match="_async_run failed"):
            asyncio.run(flow.async_invoke({"value": 123}))


class TestBasicFlowNamespace:
    def test_inherits_namespace_from_component(self) -> None:
        component = Tool(
            function=return_scalar,
            name="t",
            namespace="comp_ns",
            description="A tool.",
        )
        flow = BasicFlow(component=component)
        assert flow.namespace == "comp_ns"

    def test_explicit_namespace_overrides_component(self) -> None:
        component = Tool(
            function=return_scalar,
            name="t",
            namespace="comp_ns",
            description="A tool.",
        )
        flow = BasicFlow(component=component, namespace="flow_ns")
        assert flow.namespace == "flow_ns"


class TestBasicFlowSerialization:
    def test_to_dict_includes_component_and_checkpoint_summary(self) -> None:
        component = make_structured_component()
        flow = BasicFlow(component=component)

        result = flow.invoke({"value": 123})
        data = flow.to_dict()

        assert data["component"] == component.to_dict()
        assert data["checkpoint_count"] == 1
        assert data["runs"] == [result.run_id]
