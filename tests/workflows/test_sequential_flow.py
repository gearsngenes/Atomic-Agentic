from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import pytest

from atomic_agentic.core.Exceptions import ExecutionError, ValidationError
from atomic_agentic.core.Invokable import StructuredInvokable
from atomic_agentic.core.Parameters import ParamSpec
from atomic_agentic.models.results.workflows import SequentialFlowResult, WorkflowResult
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.base import Workflow
from atomic_agentic.workflows.basic import BasicFlow
from atomic_agentic.workflows.sequential import SequentialFlow


class EchoWorkflow(Workflow):
    """Same shape as other files' EchoWorkflow: _run returns ({"value": inputs["value"]}, {})."""

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
            parameters=[
                ParamSpec(
                    name="value",
                    index=0,
                    kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                    type="int",
                )
            ],
            return_type="dict[str, Any]",
            filter_extraneous_inputs=filter_extraneous_inputs,
        )

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"value": inputs["value"]}, {}


def first_step(value: int) -> dict[str, int]:
    """Move value into the first step field."""
    return {"first": value + 1}


def second_step(first: int) -> dict[str, int]:
    """Move first into the second step field."""
    return {"second": first * 2}


def third_step(second: int) -> dict[str, str]:
    """Move second into the third step field."""
    return {"third": f"value={second}"}


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
    )


def make_three_step_flow(*, return_index: int | None = None) -> SequentialFlow:
    return SequentialFlow(
        name="sequential_flow",
        description="Sequential test flow.",
        steps=[
            make_structured_component(first_step, name="first_step", output_schema=["first"]),
            make_structured_component(second_step, name="second_step", output_schema=["second"]),
            make_structured_component(third_step, name="third_step", output_schema=["third"]),
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

    def test_constructor_rejects_non_invokable_step(self) -> None:
        with pytest.raises(TypeError, match="Workflow or AtomicInvokable"):
            SequentialFlow(
                name="bad_flow",
                description="Bad flow.",
                steps=[object()],  # type: ignore[list-item]
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
        component = make_structured_component(first_step, name="first_step", output_schema=["first"])

        flow = SequentialFlow(
            name="sequential_flow",
            description="Sequential test flow.",
            steps=[component],
        )

        assert len(flow.steps) == 1
        assert isinstance(flow.steps[0], BasicFlow)
        assert flow.steps[0].component is component  # type: ignore[attr-defined]

    def test_return_index_defaults_to_last_step(self) -> None:
        flow = make_three_step_flow(return_index=None)

        assert flow.return_index == 2

    def test_return_index_accepts_in_range_int(self) -> None:
        flow = make_three_step_flow(return_index=0)

        assert flow.return_index == 0

    def test_return_index_rejects_non_int_at_construction(self) -> None:
        with pytest.raises(TypeError, match="return_index must be an int"):
            make_three_step_flow(return_index="0")  # type: ignore[arg-type]

    def test_return_index_rejects_negative_at_construction(self) -> None:
        with pytest.raises(IndexError, match="out of range"):
            make_three_step_flow(return_index=-1)

    def test_return_index_rejects_out_of_range_at_construction(self) -> None:
        with pytest.raises(IndexError, match="out of range"):
            make_three_step_flow(return_index=3)

    def test_return_index_has_no_setter(self) -> None:
        flow = make_three_step_flow()

        with pytest.raises(AttributeError):
            flow.return_index = 0  # type: ignore[misc]


class TestSequentialFlowSyncInvoke:
    def test_invoke_returns_sequential_flow_result(self) -> None:
        flow = make_three_step_flow(return_index=1)

        result = flow.invoke({"start": 1, "value": 2})

        assert isinstance(result, SequentialFlowResult)
        assert result.result == {"second": 6}
        assert result.return_index == 1
        assert len(result.step_runs) == 3
        assert result.run_id not in result.step_runs
        assert result.run_id == flow.checkpoints[-1].result.run_id

    def test_each_step_invoked_with_previous_steps_mapping_output(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})
        step_results = flow.get_step_results(result.run_id)

        assert step_results[0].result == {"first": 3}
        assert step_results[1].result == {"second": 6}
        assert step_results[2].result == {"third": "value=6"}

    def test_checkpoint_records_outer_inputs_and_selected_payload(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})

        assert flow.checkpoints[-1].inputs == {"value": 2}
        assert flow.checkpoints[-1].result.result == result.result


class TestSequentialFlowRetrieval:
    def test_get_step_results_returns_result_objects_in_order(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})
        results = flow.get_step_results(result.run_id)

        assert results is not None
        assert len(results) == 3
        assert all(isinstance(r, WorkflowResult) for r in results)
        assert [r.run_id for r in results] == list(result.step_runs)

    def test_get_step_results_returns_none_for_unknown_run_id(self) -> None:
        flow = make_three_step_flow()

        assert flow.get_step_results("nope") is None

    def test_get_step_result_supports_negative_index(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})

        assert flow.get_step_result(result.run_id, -1).run_id == result.step_runs[-1]

    def test_get_step_result_rejects_non_int_index(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})

        with pytest.raises(TypeError, match="step index must be an int"):
            flow.get_step_result(result.run_id, "0")  # type: ignore[arg-type]

    def test_get_step_result_rejects_out_of_range_index(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})

        with pytest.raises(IndexError, match="out of range"):
            flow.get_step_result(result.run_id, 3)

        with pytest.raises(IndexError, match="out of range"):
            flow.get_step_result("nope", 3)

    def test_get_step_result_returns_none_for_unknown_run_id_with_valid_index(self) -> None:
        flow = make_three_step_flow()

        assert flow.get_step_result("nope", 0) is None

    def test_get_step_results_per_slot_none_if_child_checkpoint_cleared(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})
        flow.steps[1].clear_memory()

        results = flow.get_step_results(result.run_id)

        assert results[1] is None
        assert results[0] is not None
        assert results[2] is not None


class TestSequentialFlowAsyncInvoke:
    def test_async_invoke_returns_sequential_flow_result(self) -> None:
        flow = make_three_step_flow(return_index=1)

        result = asyncio.run(flow.async_invoke({"value": 2}))

        assert isinstance(result, SequentialFlowResult)
        assert result.result == {"second": 6}
        assert result.return_index == 1
        assert len(result.step_runs) == 3
        assert result.run_id == flow.checkpoints[-1].result.run_id


class TestSequentialFlowValidationAndErrors:
    def test_non_final_step_non_mapping_result_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flow = make_three_step_flow()

        now = datetime.now(timezone.utc)
        bad_result = flow.steps[0].make_result(
            result=123,
            started_at=now,
            ended_at=now,
            child_id="x",
            child_type="x",
            child_run_id="x",
        )
        monkeypatch.setattr(flow.steps[0], "invoke", lambda inputs: bad_result)

        with pytest.raises(ValidationError, match="non-mapping result"):
            flow._run({"value": 1})

    def test_invoke_wraps_step_contract_failure_as_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flow = make_three_step_flow()

        now = datetime.now(timezone.utc)
        bad_result = flow.steps[0].make_result(
            result=123,
            started_at=now,
            ended_at=now,
            child_id="x",
            child_type="x",
            child_run_id="x",
        )
        monkeypatch.setattr(flow.steps[0], "invoke", lambda inputs: bad_result)

        with pytest.raises(ExecutionError, match="_run failed") as exc_info:
            flow.invoke({"value": 1})

        assert isinstance(exc_info.value.__cause__, ValidationError)

    def test_async_invoke_wraps_step_contract_failure_as_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flow = make_three_step_flow()

        now = datetime.now(timezone.utc)
        bad_result = flow.steps[0].make_result(
            result=123,
            started_at=now,
            ended_at=now,
            child_id="x",
            child_type="x",
            child_run_id="x",
        )

        async def bad_async_invoke(inputs: Mapping[str, Any]) -> Any:
            return bad_result

        monkeypatch.setattr(flow.steps[0], "async_invoke", bad_async_invoke)

        with pytest.raises(ExecutionError, match="_async_run failed") as exc_info:
            asyncio.run(flow.async_invoke({"value": 1}))

        assert isinstance(exc_info.value.__cause__, ValidationError)


class TestSequentialFlowSerialization:
    def test_to_dict_includes_steps_and_return_index(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})
        data = flow.to_dict()

        assert data["return_index"] == flow.return_index
        assert data["step_count"] == 3
        assert "steps" in data and len(data["steps"]) == 3
        assert data["checkpoint_count"] == 1
        assert data["runs"] == [result.run_id]
