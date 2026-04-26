from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from types import MethodType
from typing import Any

import pytest

from atomic_agentic.core.Exceptions import ExecutionError, ValidationError
from atomic_agentic.core.Parameters import ParamSpec
from atomic_agentic.core.sentinels import NO_VAL
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.StructuredInvokable import StructuredInvokable
from atomic_agentic.workflows.base import FlowResultDict, Workflow
from atomic_agentic.workflows.basic import BasicFlow
from atomic_agentic.workflows.metadata import WorkflowRunMetadata
from atomic_agentic.workflows.routing import RoutingFlow


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


def raise_from_router(value: Any) -> int:
    """Raise from a router tool."""
    raise RuntimeError("router boom")


def raise_from_branch(value: Any) -> dict[str, Any]:
    """Raise from a branch tool."""
    raise RuntimeError("branch boom")


def make_router(
    selection: Any,
    *,
    filter_extraneous_inputs: bool = True,
    events: list[str] | None = None,
) -> Tool:
    def route_constant(value: Any) -> Any:
        if events is not None:
            events.append("router")
        return selection

    return Tool(
        function=route_constant,
        name="route_constant",
        namespace="tests",
        description="Return a constant branch selection.",
        filter_extraneous_inputs=filter_extraneous_inputs,
    )


def make_value_router(*, filter_extraneous_inputs: bool = True) -> Tool:
    def route_by_value(value: Any) -> int:
        return 0 if value == "left" else 1

    return Tool(
        function=route_by_value,
        name="route_by_value",
        namespace="tests",
        description="Route left-like values to branch 0 and everything else to branch 1.",
        filter_extraneous_inputs=filter_extraneous_inputs,
    )


def make_raising_router() -> Tool:
    return Tool(
        function=raise_from_router,
        name="raise_from_router",
        namespace="tests",
        description="Router that raises.",
    )


def make_raw_tool() -> Tool:
    return Tool(
        function=return_mapping,
        name="return_mapping",
        namespace="tests",
        description="Return a mapping.",
    )


def make_structured_branch(
    label: str,
    *,
    filter_extraneous_inputs: bool = True,
    events: list[str] | None = None,
) -> StructuredInvokable:
    def branch(value: Any) -> dict[str, Any]:
        if events is not None:
            events.append(label)
        return {"branch": label, "value": value}

    tool = Tool(
        function=branch,
        name=f"branch_{label}",
        namespace="tests",
        description=f"Return the {label} branch result.",
        filter_extraneous_inputs=filter_extraneous_inputs,
    )
    return StructuredInvokable(
        component=tool,
        output_schema=["branch", "value"],
        name=f"structured_branch_{label}",
        description=f"Structured {label} branch.",
    )


def make_raising_branch() -> StructuredInvokable:
    tool = Tool(
        function=raise_from_branch,
        name="raise_from_branch",
        namespace="tests",
        description="Branch that raises.",
    )
    return StructuredInvokable(
        component=tool,
        output_schema=["branch", "value"],
        name="structured_raising_branch",
        description="Structured raising branch.",
    )


class RecordingBranchWorkflow(Workflow[WorkflowRunMetadata]):
    def __init__(
        self,
        *,
        label: str,
        name: str | None = None,
        filter_extraneous_inputs: bool = True,
    ) -> None:
        super().__init__(
            name=name or f"recording_branch_{label}",
            description=f"Recording branch {label}.",
            parameters=[value_param()],
            filter_extraneous_inputs=filter_extraneous_inputs,
        )
        self.label = label
        self.run_inputs: list[dict[str, Any]] = []
        self.async_run_inputs: list[dict[str, Any]] = []

    def _run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[WorkflowRunMetadata, Mapping[str, Any]]:
        self.run_inputs.append(dict(inputs))
        return WorkflowRunMetadata(kind=f"branch_{self.label}"), {
            "branch": self.label,
            "value": inputs["value"],
        }

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[WorkflowRunMetadata, Mapping[str, Any]]:
        self.async_run_inputs.append(dict(inputs))
        return WorkflowRunMetadata(kind=f"async_branch_{self.label}"), {
            "branch": self.label,
            "value": inputs["value"],
        }


def make_flow(
    selection: Any = 0,
    *,
    router: Tool | None = None,
    branches: list[Workflow | StructuredInvokable] | None = None,
    filter_extraneous_inputs: bool | None = None,
) -> RoutingFlow:
    kwargs: dict[str, Any] = {}
    if filter_extraneous_inputs is not None:
        kwargs["filter_extraneous_inputs"] = filter_extraneous_inputs

    return RoutingFlow(
        name="routing_flow",
        description="Routing flow under test.",
        branches=branches
        if branches is not None
        else [make_structured_branch("left"), make_structured_branch("right")],
        router=router if router is not None else make_router(selection),
        **kwargs,
    )


class TestRoutingFlowConstruction:
    def test_constructor_rejects_non_list_branches(self) -> None:
        with pytest.raises(TypeError, match="branches must be"):
            RoutingFlow(
                name="routing_flow",
                description="Routing flow.",
                branches=(make_structured_branch("left"),),  # type: ignore[arg-type]
                router=make_router(0),
            )

    def test_constructor_rejects_empty_branches(self) -> None:
        with pytest.raises(ValueError, match="branches must not be empty"):
            RoutingFlow(
                name="routing_flow",
                description="Routing flow.",
                branches=[],
                router=make_router(0),
            )

    def test_constructor_rejects_raw_tool_branch(self) -> None:
        with pytest.raises(TypeError, match="Workflow or StructuredInvokable"):
            RoutingFlow(
                name="routing_flow",
                description="Routing flow.",
                branches=[make_raw_tool()],  # type: ignore[list-item]
                router=make_router(0),
            )

    def test_constructor_rejects_non_atomic_router(self) -> None:
        with pytest.raises(TypeError, match="router must be an AtomicInvokable"):
            RoutingFlow(
                name="routing_flow",
                description="Routing flow.",
                branches=[make_structured_branch("left")],
                router=object(),  # type: ignore[arg-type]
            )

    def test_constructor_preserves_workflow_branches(self) -> None:
        left = RecordingBranchWorkflow(label="left")
        right = RecordingBranchWorkflow(label="right")

        flow = make_flow(branches=[left, right])

        assert flow.branches == (left, right)

    def test_constructor_wraps_structured_branches_in_basic_flow(self) -> None:
        left = make_structured_branch("left")
        right = make_structured_branch("right")

        flow = make_flow(branches=[left, right])

        assert all(isinstance(branch, BasicFlow) for branch in flow.branches)
        assert flow.branches[0].component is left  # type: ignore[attr-defined]
        assert flow.branches[1].component is right  # type: ignore[attr-defined]

    def test_constructor_normalizes_router_to_basic_flow(self) -> None:
        router = make_router(0)

        flow = make_flow(router=router)

        assert isinstance(flow.router, BasicFlow)
        assert isinstance(flow.router.component, StructuredInvokable)
        assert flow.router.component.component is router

    def test_router_structured_schema_is_branch_selection(self) -> None:
        flow = make_flow(selection=0)
        router_component = flow.router.component

        assert isinstance(router_component, StructuredInvokable)
        assert [spec.name for spec in router_component.output_schema] == [
            "branch_selection"
        ]

    def test_constructor_uses_router_parameters_as_outer_parameters(self) -> None:
        flow = make_flow(router=make_value_router())

        assert flow.parameters == flow.router.parameters
        assert [param.name for param in flow.parameters] == ["value"]

    def test_constructor_inherits_router_filter_flag_by_default(self) -> None:
        flow = make_flow(router=make_router(0, filter_extraneous_inputs=False))

        assert flow.filter_extraneous_inputs is False

    def test_constructor_allows_filter_flag_override(self) -> None:
        flow = make_flow(
            router=make_router(0, filter_extraneous_inputs=False),
            filter_extraneous_inputs=True,
        )

        assert flow.filter_extraneous_inputs is True


class TestRoutingFlowSyncInvoke:
    def test_invoke_routes_to_branch_zero(self) -> None:
        flow = make_flow(selection=0)

        result = flow.invoke({"value": 10})

        assert isinstance(result, FlowResultDict)
        assert result == {"branch": "left", "value": 10}

    def test_invoke_routes_to_branch_one(self) -> None:
        flow = make_flow(selection=1)

        result = flow.invoke({"value": 10})

        assert isinstance(result, FlowResultDict)
        assert result == {"branch": "right", "value": 10}

    def test_invoke_runs_router_before_branch(self) -> None:
        events: list[str] = []
        flow = make_flow(
            router=make_router(1, events=events),
            branches=[
                make_structured_branch("left", events=events),
                make_structured_branch("right", events=events),
            ],
        )

        result = flow.invoke({"value": 10})

        assert result == {"branch": "right", "value": 10}
        assert events == ["router", "right"]

    def test_invoke_runs_exactly_one_branch(self) -> None:
        flow = make_flow(selection=1)

        result = flow.invoke({"value": 10})

        assert result == {"branch": "right", "value": 10}
        assert len(flow.branches[0].checkpoints) == 0
        assert len(flow.branches[1].checkpoints) == 1

    def test_selected_branch_receives_original_filtered_inputs(self) -> None:
        left = RecordingBranchWorkflow(label="left")
        right = RecordingBranchWorkflow(label="right")
        flow = make_flow(selection=1, branches=[left, right])

        result = flow.invoke({"value": 10, "extra": "ignored"})

        assert result == {"branch": "right", "value": 10}
        assert left.run_inputs == []
        assert right.run_inputs == [{"value": 10}]

    def test_router_result_is_not_handed_to_branch(self) -> None:
        left = RecordingBranchWorkflow(label="left")
        right = RecordingBranchWorkflow(label="right")
        flow = make_flow(selection=1, branches=[left, right])

        flow.invoke({"value": 10})

        assert right.run_inputs == [{"value": 10}]
        assert "branch_selection" not in right.run_inputs[0]

    def test_invoke_returns_outer_flow_result_with_parent_run_id(self) -> None:
        flow = make_flow(selection=1)

        result = flow.invoke({"value": 10})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert isinstance(result, FlowResultDict)
        assert result.run_id == flow.latest_run
        assert result.run_id != metadata.router_run_id
        assert result.run_id != metadata.chosen_branch_record.run_id

    def test_parent_checkpoint_records_outer_inputs_and_selected_result(self) -> None:
        flow = make_flow(selection=1)

        result = flow.invoke({"value": 10, "extra": "ignored"})
        checkpoint = flow.get_checkpoint(result.run_id)

        assert checkpoint is not None
        assert checkpoint.inputs == {"value": 10}
        assert checkpoint.result == {"branch": "right", "value": 10}


class TestRoutingFlowRouterDecisionValidation:
    def test_run_raises_if_router_result_missing_branch_selection(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(selection=0)
        monkeypatch.setattr(
            flow.router,
            "invoke",
            lambda inputs: FlowResultDict({"other": 0}, run_id="router_run"),
        )

        with pytest.raises(ValidationError, match="branch_selection"):
            flow._run({"value": 10})

    def test_run_raises_if_branch_selection_is_string(self) -> None:
        flow = make_flow(selection="1")

        with pytest.raises(ValidationError, match="branch_selection must be an int"):
            flow._run({"value": 10})

    def test_run_raises_if_branch_selection_is_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(selection=0)
        monkeypatch.setattr(
            flow.router,
            "invoke",
            lambda inputs: FlowResultDict(
                {"branch_selection": None},
                run_id="router_run",
            ),
        )

        with pytest.raises(ValidationError, match="branch_selection must be an int"):
            flow._run({"value": 10})

    def test_run_raises_if_branch_selection_is_negative(self) -> None:
        flow = make_flow(selection=-1)

        with pytest.raises(ValidationError, match="out of range"):
            flow._run({"value": 10})

    def test_run_raises_if_branch_selection_is_out_of_range(self) -> None:
        flow = make_flow(selection=2)

        with pytest.raises(ValidationError, match="out of range"):
            flow._run({"value": 10})

    def test_run_rejects_bool_branch_selection_intentionally_fails_until_patch(
        self,
    ) -> None:
        flow = make_flow(selection=True)

        with pytest.raises(ValidationError, match="branch_selection"):
            flow._run({"value": 10})

    def test_public_invoke_wraps_invalid_router_decision_as_execution_error(
        self,
    ) -> None:
        flow = make_flow(selection="bad")

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 10})


class TestRoutingFlowMetadata:
    def test_metadata_kind_is_routing(self) -> None:
        flow = make_flow(selection=1)

        result = flow.invoke({"value": 10})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.kind == "routing"

    def test_metadata_records_router_run_id(self) -> None:
        flow = make_flow(selection=1)

        result = flow.invoke({"value": 10})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.router_run_id == flow.router.latest_run

    def test_metadata_records_router_instance_id(self) -> None:
        flow = make_flow(selection=1)

        result = flow.invoke({"value": 10})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.router_instance_id == flow.router.instance_id

    def test_metadata_records_chosen_index(self) -> None:
        flow = make_flow(selection=1)

        result = flow.invoke({"value": 10})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.chosen_index == 1

    def test_metadata_chosen_branch_record_matches_selected_branch(self) -> None:
        flow = make_flow(selection=1)

        result = flow.invoke({"value": 10})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]
        record = metadata.chosen_branch_record
        selected_branch = flow.branches[1]

        assert record.slot == 1
        assert record.instance_id == selected_branch.instance_id
        assert record.full_name == selected_branch.full_name
        assert record.run_id == selected_branch.latest_run

    def test_metadata_does_not_record_unchosen_branch(self) -> None:
        flow = make_flow(selection=1)

        result = flow.invoke({"value": 10})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.chosen_branch_record.slot == 1
        assert len(flow.branches[0].checkpoints) == 0
        assert len(flow.branches[1].checkpoints) == 1


class TestRoutingFlowRetrieval:
    def test_get_router_decision_returns_none_for_unknown_run(self) -> None:
        flow = make_flow(selection=1)

        assert flow.get_router_decision("missing") is None

    def test_get_router_decision_returns_chosen_index(self) -> None:
        flow = make_flow(selection=1)

        result = flow.invoke({"value": 10})

        assert flow.get_router_decision(result.run_id) == 1

    def test_get_router_decision_raises_if_stored_metadata_decision_is_not_int(
        self,
    ) -> None:
        flow = make_flow(selection=1)
        result = flow.invoke({"value": 10})
        checkpoint = flow.get_checkpoint(result.run_id)
        assert checkpoint is not None

        bad_metadata = replace(checkpoint.metadata, chosen_index="bad")  # type: ignore[arg-type]
        flow._checkpoints[0] = replace(checkpoint, metadata=bad_metadata)  # type: ignore[attr-defined]

        with pytest.raises(ValidationError, match="chosen_index"):
            flow.get_router_decision(result.run_id)

    def test_get_router_decision_after_multiple_runs_returns_per_run_decision(
        self,
    ) -> None:
        flow = make_flow(router=make_value_router())

        left = flow.invoke({"value": "left"})
        right = flow.invoke({"value": "right"})

        assert flow.get_router_decision(left.run_id) == 0
        assert flow.get_router_decision(right.run_id) == 1


class TestRoutingFlowAsyncInvoke:
    def test_async_invoke_routes_to_branch_zero(self) -> None:
        flow = make_flow(selection=0)

        result = asyncio.run(flow.async_invoke({"value": 10}))

        assert isinstance(result, FlowResultDict)
        assert result == {"branch": "left", "value": 10}

    def test_async_invoke_routes_to_branch_one(self) -> None:
        flow = make_flow(selection=1)

        result = asyncio.run(flow.async_invoke({"value": 10}))

        assert isinstance(result, FlowResultDict)
        assert result == {"branch": "right", "value": 10}

    def test_async_invoke_runs_exactly_one_branch(self) -> None:
        flow = make_flow(selection=1)

        result = asyncio.run(flow.async_invoke({"value": 10}))

        assert result == {"branch": "right", "value": 10}
        assert len(flow.branches[0].checkpoints) == 0
        assert len(flow.branches[1].checkpoints) == 1

    def test_async_selected_branch_receives_original_filtered_inputs(self) -> None:
        left = RecordingBranchWorkflow(label="left")
        right = RecordingBranchWorkflow(label="right")
        flow = make_flow(selection=1, branches=[left, right])

        result = asyncio.run(flow.async_invoke({"value": 10, "extra": "ignored"}))

        assert result == {"branch": "right", "value": 10}
        assert left.async_run_inputs == []
        assert right.async_run_inputs == [{"value": 10}]

    def test_async_metadata_records_router_and_chosen_branch(self) -> None:
        flow = make_flow(selection=1)

        result = asyncio.run(flow.async_invoke({"value": 10}))
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]
        selected_branch = flow.branches[1]

        assert metadata.kind == "routing"
        assert metadata.router_run_id == flow.router.latest_run
        assert metadata.router_instance_id == flow.router.instance_id
        assert metadata.chosen_index == 1
        assert metadata.chosen_branch_record.run_id == selected_branch.latest_run

    def test_async_invalid_router_decision_public_call_wraps_execution_error(
        self,
    ) -> None:
        flow = make_flow(selection="bad")

        with pytest.raises(ExecutionError, match="_async_run failed"):
            asyncio.run(flow.async_invoke({"value": 10}))


class TestRoutingFlowValidationAndErrors:
    def test_run_raises_validation_error_if_router_returns_non_flow_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(selection=0)
        monkeypatch.setattr(flow.router, "invoke", lambda inputs: {"branch_selection": 0})

        with pytest.raises(ValidationError, match="router returned"):
            flow._run({"value": 10})

    def test_public_invoke_wraps_router_non_flow_result_as_execution_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(selection=0)
        monkeypatch.setattr(flow.router, "invoke", lambda inputs: {"branch_selection": 0})

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 10})

    def test_run_raises_validation_error_if_selected_branch_returns_non_flow_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(selection=0)
        monkeypatch.setattr(flow.branches[0], "invoke", lambda inputs: {"branch": "left"})

        with pytest.raises(ValidationError, match="selected branch"):
            flow._run({"value": 10})

    def test_public_invoke_wraps_selected_branch_non_flow_result_as_execution_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(selection=0)
        monkeypatch.setattr(flow.branches[0], "invoke", lambda inputs: {"branch": "left"})

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 10})

    def test_public_invoke_wraps_router_runtime_error_as_execution_error(self) -> None:
        flow = make_flow(router=make_raising_router())

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 10})

    def test_public_invoke_wraps_selected_branch_runtime_error_as_execution_error(
        self,
    ) -> None:
        flow = make_flow(selection=0, branches=[make_raising_branch(), make_structured_branch("right")])

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 10})

    def test_async_run_raises_validation_error_if_router_returns_non_flow_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(selection=0)

        async def bad_async_invoke(self: BasicFlow, inputs: Mapping[str, Any]) -> dict[str, int]:
            return {"branch_selection": 0}

        monkeypatch.setattr(
            flow.router,
            "async_invoke",
            MethodType(bad_async_invoke, flow.router),
        )

        with pytest.raises(ValidationError, match="async router returned"):
            asyncio.run(flow._async_run({"value": 10}))

    def test_async_run_raises_validation_error_if_selected_branch_returns_non_flow_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(selection=0)
        branch = flow.branches[0]

        async def bad_async_invoke(self: Workflow, inputs: Mapping[str, Any]) -> dict[str, str]:
            return {"branch": "left"}

        monkeypatch.setattr(
            branch,
            "async_invoke",
            MethodType(bad_async_invoke, branch),
        )

        with pytest.raises(ValidationError, match="async selected branch"):
            asyncio.run(flow._async_run({"value": 10}))

    def test_public_async_invoke_wraps_router_failure_as_execution_error(self) -> None:
        flow = make_flow(router=make_raising_router())

        with pytest.raises(ExecutionError, match="_async_run failed"):
            asyncio.run(flow.async_invoke({"value": 10}))

    def test_public_async_invoke_wraps_selected_branch_failure_as_execution_error(
        self,
    ) -> None:
        flow = make_flow(selection=0, branches=[make_raising_branch(), make_structured_branch("right")])

        with pytest.raises(ExecutionError, match="_async_run failed"):
            asyncio.run(flow.async_invoke({"value": 10}))


class TestRoutingFlowSerialization:
    def test_to_dict_includes_router_snapshot(self) -> None:
        flow = make_flow(selection=1)

        data = flow.to_dict()

        assert data["type"] == "RoutingFlow"
        assert data["router"]["type"] == "BasicFlow"

    def test_to_dict_includes_branch_snapshots(self) -> None:
        flow = make_flow(selection=1)

        data = flow.to_dict()

        assert "branches" in data
        assert len(data["branches"]) == 2
        assert data["branches"][0]["type"] == "BasicFlow"
        assert data["branches"][1]["type"] == "BasicFlow"

    def test_to_dict_router_has_structured_branch_selection_component(self) -> None:
        flow = make_flow(selection=1)

        router_data = flow.to_dict()["router"]

        assert router_data["component"]["type"] == "StructuredInvokable"
        assert router_data["component"]["output_schema"][0]["name"] == "branch_selection"

    def test_to_dict_after_run_includes_base_checkpoint_summary(self) -> None:
        flow = make_flow(selection=1)

        result = flow.invoke({"value": 10})
        data = flow.to_dict()

        assert data["checkpoint_count"] == 1
        assert data["runs"] == [result.run_id]
        assert "checkpoints" not in data
        assert "router" in data
        assert "branches" in data
