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
from atomic_agentic.workflows.StructuredInvokable import StructuredInvokable
from atomic_agentic.workflows.base import FlowResultDict, Workflow
from atomic_agentic.workflows.basic import BasicFlow
from atomic_agentic.workflows.metadata import WorkflowRunMetadata
from atomic_agentic.workflows.sequential import SequentialFlow


def value_param() -> ParamSpec:
    return ParamSpec(
        name="value",
        index=0,
        kind=ParamSpec.POSITIONAL_OR_KEYWORD,
        type="Any",
        default=NO_VAL,
    )


def make_param(name: str, index: int) -> ParamSpec:
    return ParamSpec(
        name=name,
        index=index,
        kind=ParamSpec.POSITIONAL_OR_KEYWORD,
        type="Any",
        default=NO_VAL,
    )


def return_value(value: Any) -> Any:
    """Return the provided value."""
    return value


def make_raw_tool() -> Tool:
    return Tool(
        function=return_value,
        name="return_value",
        namespace="tests",
        description="Return the provided value.",
    )


def make_structured_component(
    function: Any,
    *,
    name: str,
    output_schema: list[str],
    filter_extraneous_inputs: bool = True,
) -> StructuredInvokable:
    tool = Tool(
        function=function,
        name=name,
        namespace="tests",
        description=f"Test tool {name}.",
        filter_extraneous_inputs=filter_extraneous_inputs,
    )
    return StructuredInvokable(
        component=tool,
        output_schema=output_schema,
        name=f"structured_{name}",
        description=f"Structured test component {name}.",
        ignore_unhandled=True,
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


def first_step(value: int) -> dict[str, int]:
    """Move value into the first step field."""
    return {"first": value + 1}


def second_step(first: int) -> dict[str, int]:
    """Move first into the second step field."""
    return {"second": first * 2}


def third_step(second: int) -> dict[str, str]:
    """Move second into the third step field."""
    return {"third": f"value={second}"}


def make_three_step_flow(*, return_index: int = -1) -> SequentialFlow:
    return SequentialFlow(
        name="sequential_flow",
        description="Sequential test flow.",
        steps=[
            make_structured_component(
                first_step,
                name="first_step",
                output_schema=["first"],
            ),
            make_structured_component(
                second_step,
                name="second_step",
                output_schema=["second"],
            ),
            make_structured_component(
                third_step,
                name="third_step",
                output_schema=["third"],
            ),
        ],
        return_index=return_index,
    )


class TestSequentialFlowConstruction:
    def test_constructor_rejects_non_list_steps(self) -> None:
        with pytest.raises(TypeError, match="steps must be"):
            SequentialFlow(
                name="bad_flow",
                description="Bad flow.",
                steps=(EchoWorkflow(),),  # type: ignore[arg-type]
            )

    def test_constructor_rejects_empty_steps(self) -> None:
        with pytest.raises(ValueError, match="steps must not be empty"):
            SequentialFlow(
                name="bad_flow",
                description="Bad flow.",
                steps=[],
            )

    def test_constructor_rejects_raw_tool_step(self) -> None:
        with pytest.raises(TypeError, match="Workflow or StructuredInvokable"):
            SequentialFlow(
                name="bad_flow",
                description="Bad flow.",
                steps=[make_raw_tool()],  # type: ignore[list-item]
            )

    def test_constructor_preserves_workflow_steps(self) -> None:
        child = EchoWorkflow()

        flow = SequentialFlow(
            name="sequential_flow",
            description="Sequential test flow.",
            steps=[child],
        )

        assert flow.steps == (child,)
        assert flow.steps[0] is child

    def test_constructor_wraps_structured_steps_in_basic_flow(self) -> None:
        component = make_structured_component(
            first_step,
            name="first_step",
            output_schema=["first"],
        )

        flow = SequentialFlow(
            name="sequential_flow",
            description="Sequential test flow.",
            steps=[component],
        )

        assert len(flow.steps) == 1
        assert isinstance(flow.steps[0], BasicFlow)
        assert flow.steps[0].component is component  # type: ignore[attr-defined]

    def test_constructor_uses_first_step_parameters(self) -> None:
        component = make_structured_component(
            first_step,
            name="first_step",
            output_schema=["first"],
        )

        flow = SequentialFlow(
            name="sequential_flow",
            description="Sequential test flow.",
            steps=[component],
        )

        assert flow.parameters == flow.steps[0].parameters
        assert [param.name for param in flow.parameters] == ["value"]

    def test_constructor_inherits_first_step_filter_flag_by_default(self) -> None:
        child = EchoWorkflow(filter_extraneous_inputs=False)

        flow = SequentialFlow(
            name="sequential_flow",
            description="Sequential test flow.",
            steps=[child],
        )

        assert flow.filter_extraneous_inputs is False

    def test_constructor_allows_filter_flag_override(self) -> None:
        child = EchoWorkflow(filter_extraneous_inputs=False)

        flow = SequentialFlow(
            name="sequential_flow",
            description="Sequential test flow.",
            steps=[child],
            filter_extraneous_inputs=True,
        )

        assert flow.filter_extraneous_inputs is True

    @pytest.mark.parametrize("return_index", [0, -1])
    def test_return_index_accepts_positive_and_negative_indices(
        self,
        return_index: int,
    ) -> None:
        flow = make_three_step_flow(return_index=return_index)

        assert flow.return_index == return_index

    @pytest.mark.parametrize("bad_index", ["0", 1.5, None])
    def test_return_index_rejects_non_int(self, bad_index: Any) -> None:
        flow = make_three_step_flow()

        with pytest.raises(TypeError, match="return_index must be an int"):
            flow.return_index = bad_index  # type: ignore[assignment]

    @pytest.mark.parametrize("bad_index", [3, -4])
    def test_return_index_rejects_out_of_range(self, bad_index: int) -> None:
        flow = make_three_step_flow()

        with pytest.raises(IndexError, match="out of range"):
            flow.return_index = bad_index


class TestSequentialFlowSyncInvoke:
    def test_invoke_runs_all_steps_in_order(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})

        assert result == {"third": "value=6"}
        assert flow.get_step_results(result.run_id) == [
            {"first": 3},
            {"second": 6},
            {"third": "value=6"},
        ]

    def test_first_step_receives_filtered_outer_inputs(self) -> None:
        calls: list[dict[str, Any]] = []

        def recording_first_step(value: int) -> dict[str, int]:
            calls.append({"value": value})
            return {"first": value + 1}

        flow = SequentialFlow(
            name="sequential_flow",
            description="Sequential test flow.",
            steps=[
                make_structured_component(
                    recording_first_step,
                    name="recording_first_step",
                    output_schema=["first"],
                ),
                make_structured_component(
                    second_step,
                    name="second_step",
                    output_schema=["second"],
                ),
            ],
        )

        flow.invoke({"value": 2, "extra": "ignored"})

        assert calls == [{"value": 2}]

    def test_each_step_receives_previous_step_result(self) -> None:
        second_calls: list[dict[str, Any]] = []

        def recording_second_step(first: int) -> dict[str, int]:
            second_calls.append({"first": first})
            return {"second": first * 2}

        flow = SequentialFlow(
            name="sequential_flow",
            description="Sequential test flow.",
            steps=[
                make_structured_component(
                    first_step,
                    name="first_step",
                    output_schema=["first"],
                ),
                make_structured_component(
                    recording_second_step,
                    name="recording_second_step",
                    output_schema=["second"],
                ),
            ],
        )

        flow.invoke({"value": 2})

        assert second_calls == [{"first": 3}]

    def test_default_return_index_returns_last_step_result(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})

        assert result == {"third": "value=6"}

    def test_non_default_return_index_returns_selected_earlier_result(self) -> None:
        flow = make_three_step_flow(return_index=0)

        result = flow.invoke({"value": 2})

        assert result == {"first": 3}
        assert flow.get_step_results(result.run_id) == [
            {"first": 3},
            {"second": 6},
            {"third": "value=6"},
        ]

    def test_invoke_returns_outer_flow_result_with_parent_run_id(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert isinstance(result, FlowResultDict)
        assert result.run_id == flow.latest_run
        assert result.run_id not in {record.run_id for record in metadata.step_records}

    def test_parent_checkpoint_records_outer_inputs_and_selected_result(self) -> None:
        flow = make_three_step_flow(return_index=1)

        result = flow.invoke({"value": 2, "extra": "ignored"})
        checkpoint = flow.get_checkpoint(result.run_id)

        assert checkpoint is not None
        assert checkpoint.inputs == {"value": 2}
        assert checkpoint.result == {"second": 6}


class TestSequentialFlowMetadata:
    def test_metadata_records_one_child_record_per_step(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.kind == "sequential"
        assert len(metadata.step_records) == 3

    def test_step_records_match_child_workflows(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        for index, record in enumerate(metadata.step_records):
            step = flow.steps[index]
            assert record.slot == index
            assert record.instance_id == step.instance_id
            assert record.full_name == step.full_name
            assert record.run_id == step.latest_run

    def test_metadata_resolves_negative_return_index_to_absolute_index(self) -> None:
        flow = make_three_step_flow(return_index=-1)

        result = flow.invoke({"value": 2})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.return_child_index == 2

    def test_metadata_return_child_run_id_matches_selected_step_record(self) -> None:
        flow = make_three_step_flow(return_index=1)

        result = flow.invoke({"value": 2})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.return_child_run_id == metadata.step_records[1].run_id


class TestSequentialFlowRetrieval:
    def test_get_step_records_returns_none_for_unknown_run(self) -> None:
        flow = make_three_step_flow()

        assert flow.get_step_records("unknown") is None

    def test_get_step_records_returns_metadata_records(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert flow.get_step_records(result.run_id) == metadata.step_records

    def test_get_step_results_returns_child_checkpoint_results(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})

        assert flow.get_step_results(result.run_id) == [
            {"first": 3},
            {"second": 6},
            {"third": "value=6"},
        ]

    def test_get_step_result_returns_selected_child_result(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})

        assert flow.get_step_result(result.run_id, 0) == {"first": 3}
        assert flow.get_step_result(result.run_id, -1) == {"third": "value=6"}

    def test_get_step_result_returns_none_for_unknown_parent_run(self) -> None:
        flow = make_three_step_flow()

        assert flow.get_step_result("unknown", 0) is None

    def test_get_step_result_rejects_non_int_index(self) -> None:
        flow = make_three_step_flow()

        with pytest.raises(TypeError, match="step index must be an int"):
            flow.get_step_result("unknown", "0")  # type: ignore[arg-type]

    def test_get_step_result_rejects_out_of_range_index(self) -> None:
        flow = make_three_step_flow()

        with pytest.raises(IndexError, match="out of range"):
            flow.get_step_result("unknown", 3)

    def test_get_step_results_returns_none_for_missing_child_checkpoint(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})
        flow.steps[1].clear_memory()

        assert flow.get_step_results(result.run_id) == [
            {"first": 3},
            None,
            {"third": "value=6"},
        ]


class TestSequentialFlowAsyncInvoke:
    def test_async_invoke_runs_all_steps_in_order(self) -> None:
        flow = make_three_step_flow()

        result = asyncio.run(flow.async_invoke({"value": 2}))

        assert result == {"third": "value=6"}
        assert flow.get_step_results(result.run_id) == [
            {"first": 3},
            {"second": 6},
            {"third": "value=6"},
        ]

    def test_async_invoke_returns_selected_return_index_result(self) -> None:
        flow = make_three_step_flow(return_index=0)

        result = asyncio.run(flow.async_invoke({"value": 2}))

        assert result == {"first": 3}
        assert flow.get_step_results(result.run_id) == [
            {"first": 3},
            {"second": 6},
            {"third": "value=6"},
        ]

    def test_async_metadata_matches_sync_shape(self) -> None:
        flow = make_three_step_flow(return_index=-1)

        result = asyncio.run(flow.async_invoke({"value": 2}))
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.kind == "sequential"
        assert len(metadata.step_records) == 3
        assert metadata.return_child_index == 2
        assert metadata.return_child_run_id == metadata.step_records[2].run_id

    def test_async_child_checkpoints_are_created(self) -> None:
        flow = make_three_step_flow()

        asyncio.run(flow.async_invoke({"value": 2}))

        assert [len(step.checkpoints) for step in flow.steps] == [1, 1, 1]


class TestSequentialFlowValidationAndErrors:
    def test_run_raises_validation_error_if_step_returns_non_flow_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_three_step_flow()

        monkeypatch.setattr(flow.steps[1], "invoke", lambda inputs: {"bad": True})

        with pytest.raises(ValidationError, match="expected FlowResultDict"):
            flow._run({"value": 2})

    def test_public_invoke_wraps_child_contract_failure_as_execution_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_three_step_flow()

        monkeypatch.setattr(flow.steps[1], "invoke", lambda inputs: {"bad": True})

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 2})

    def test_async_run_raises_validation_error_if_step_returns_non_flow_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_three_step_flow()

        async def bad_async_invoke(self: Workflow[Any], inputs: Mapping[str, Any]) -> dict[str, bool]:
            return {"bad": True}

        monkeypatch.setattr(
            flow.steps[1],
            "async_invoke",
            MethodType(bad_async_invoke, flow.steps[1]),
        )

        with pytest.raises(ValidationError, match="expected FlowResultDict"):
            asyncio.run(flow._async_run({"value": 2}))

    def test_public_async_invoke_wraps_child_contract_failure_as_execution_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_three_step_flow()

        async def bad_async_invoke(self: Workflow[Any], inputs: Mapping[str, Any]) -> dict[str, bool]:
            return {"bad": True}

        monkeypatch.setattr(
            flow.steps[1],
            "async_invoke",
            MethodType(bad_async_invoke, flow.steps[1]),
        )

        with pytest.raises(ExecutionError, match="_async_run failed"):
            asyncio.run(flow.async_invoke({"value": 2}))


class TestSequentialFlowSerialization:
    def test_to_dict_includes_steps_and_step_count(self) -> None:
        flow = make_three_step_flow()

        data = flow.to_dict()

        assert data["type"] == "SequentialFlow"
        assert data["step_count"] == 3
        assert len(data["steps"]) == 3

    def test_to_dict_includes_return_index(self) -> None:
        flow = make_three_step_flow(return_index=-2)

        data = flow.to_dict()

        assert data["return_index"] == -2

    def test_to_dict_after_run_includes_base_checkpoint_summary(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})
        data = flow.to_dict()

        assert data["checkpoint_count"] == 1
        assert data["runs"] == [result.run_id]
        assert data["step_count"] == 3
