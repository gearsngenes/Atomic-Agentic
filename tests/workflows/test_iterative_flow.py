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
from atomic_agentic.workflows.iterative import IterativeFlow
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


def raw_mapping(value: Any) -> dict[str, Any]:
    """Return a mapping."""
    return {"value": value}


def increment_value(value: int) -> int:
    """Increment the value."""
    return value + 1


def double_value(value: int) -> int:
    """Double the value."""
    return value * 2


def raising_body(value: Any) -> int:
    """Raise from a body tool."""
    raise RuntimeError("body boom")


def raising_judge(value: Any) -> bool:
    """Raise from a judge tool."""
    raise RuntimeError("judge boom")


def make_raw_tool() -> Tool:
    return Tool(
        function=raw_mapping,
        name="raw_mapping",
        namespace="tests",
        description="Return a mapping.",
    )


def make_structured_scalar_component(
    function: Any,
    *,
    name: str,
    output_key: str = "value",
    filter_extraneous_inputs: bool = True,
) -> StructuredInvokable:
    tool = Tool(
        function=function,
        name=name,
        namespace="tests",
        description=f"Tool {name}.",
        filter_extraneous_inputs=filter_extraneous_inputs,
    )
    return StructuredInvokable(
        component=tool,
        output_schema=[output_key],
        name=f"structured_{name}",
        description=f"Structured {name}.",
    )


def make_increment_component(
    *,
    filter_extraneous_inputs: bool = True,
) -> StructuredInvokable:
    return make_structured_scalar_component(
        increment_value,
        name="increment_value",
        filter_extraneous_inputs=filter_extraneous_inputs,
    )


def make_double_component() -> StructuredInvokable:
    return make_structured_scalar_component(double_value, name="double_value")


def make_marker_component(marker: str) -> StructuredInvokable:
    def mark_value(value: int) -> dict[str, Any]:
        return {"value": value + 1, "marker": marker}

    tool = Tool(
        function=mark_value,
        name=f"mark_{marker}",
        namespace="tests",
        description=f"Mark value with {marker}.",
    )
    return StructuredInvokable(
        component=tool,
        output_schema=["value", "marker"],
        name=f"structured_mark_{marker}",
        description=f"Structured marker {marker}.",
    )


def make_threshold_judge(threshold: int, *, seen: list[int] | None = None) -> Tool:
    def approve_at_threshold(value: int) -> bool:
        if seen is not None:
            seen.append(value)
        return value >= threshold

    return Tool(
        function=approve_at_threshold,
        name=f"approve_at_{threshold}",
        namespace="tests",
        description=f"Approve when value >= {threshold}.",
    )


def make_constant_judge(decision: Any, *, seen: list[Any] | None = None) -> Tool:
    def judge_value(value: Any) -> Any:
        if seen is not None:
            seen.append(value)
        return decision

    return Tool(
        function=judge_value,
        name="constant_judge",
        namespace="tests",
        description="Return a constant judge decision.",
    )


def make_marker_judge(expected_marker: str, *, seen: list[str] | None = None) -> Tool:
    def approve_marker(marker: str) -> bool:
        if seen is not None:
            seen.append(marker)
        return marker == expected_marker

    return Tool(
        function=approve_marker,
        name=f"approve_marker_{expected_marker}",
        namespace="tests",
        description=f"Approve marker {expected_marker}.",
    )


def make_raising_body_component() -> StructuredInvokable:
    return make_structured_scalar_component(raising_body, name="raising_body")


def make_raising_judge() -> Tool:
    return Tool(
        function=raising_judge,
        name="raising_judge",
        namespace="tests",
        description="Judge that raises.",
    )


def make_flow(
    *,
    body_steps: list[Workflow | StructuredInvokable] | None = None,
    judge: Tool | None = None,
    max_iterations: int = 3,
    return_index: int = -1,
    handoff_index: int = -1,
    evaluate_index: int = -1,
    filter_extraneous_inputs: bool | None = None,
) -> IterativeFlow:
    kwargs: dict[str, Any] = {}
    if filter_extraneous_inputs is not None:
        kwargs["filter_extraneous_inputs"] = filter_extraneous_inputs

    return IterativeFlow(
        name="iterative_flow",
        description="Iterative flow under test.",
        body_steps=body_steps if body_steps is not None else [make_increment_component()],
        judge=judge,
        max_iterations=max_iterations,
        return_index=return_index,
        handoff_index=handoff_index,
        evaluate_index=evaluate_index,
        **kwargs,
    )


class RecordingWorkflow(Workflow[WorkflowRunMetadata]):
    def __init__(
        self,
        *,
        name: str = "recording_workflow",
        filter_extraneous_inputs: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            description="Recording workflow.",
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
        return WorkflowRunMetadata(kind="recording"), {"value": inputs["value"] + 1}

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[WorkflowRunMetadata, Mapping[str, Any]]:
        self.async_run_inputs.append(dict(inputs))
        return WorkflowRunMetadata(kind="async_recording"), {"value": inputs["value"] + 1}


class TestIterativeFlowConstruction:
    def test_constructor_rejects_non_list_body_steps(self) -> None:
        with pytest.raises(TypeError, match="body_steps must be a list"):
            IterativeFlow(
                name="iterative_flow",
                description="Iterative flow.",
                body_steps=(make_increment_component(),),  # type: ignore[arg-type]
            )

    def test_constructor_rejects_empty_body_steps(self) -> None:
        with pytest.raises(ValueError, match="body_steps must not be empty"):
            IterativeFlow(
                name="iterative_flow",
                description="Iterative flow.",
                body_steps=[],
            )

    def test_constructor_rejects_invalid_body_step_item(self) -> None:
        with pytest.raises(TypeError, match="body_steps items"):
            IterativeFlow(
                name="iterative_flow",
                description="Iterative flow.",
                body_steps=[make_raw_tool()],  # type: ignore[list-item]
            )

    def test_constructor_rejects_non_atomic_judge(self) -> None:
        with pytest.raises(TypeError, match="judge must be an AtomicInvokable"):
            IterativeFlow(
                name="iterative_flow",
                description="Iterative flow.",
                body_steps=[make_increment_component()],
                judge=object(),  # type: ignore[arg-type]
            )

    def test_constructor_creates_loop_body_as_sequential_flow(self) -> None:
        flow = make_flow()

        assert isinstance(flow.loop_body, SequentialFlow)
        assert flow.loop_body.name == "iterative_flow_loop_body"

    def test_constructor_wraps_structured_body_steps_inside_loop_body(self) -> None:
        first = make_increment_component()
        second = make_double_component()

        flow = make_flow(body_steps=[first, second])

        assert all(isinstance(step, BasicFlow) for step in flow.loop_body.steps)
        assert flow.loop_body.steps[0].component is first  # type: ignore[attr-defined]
        assert flow.loop_body.steps[1].component is second  # type: ignore[attr-defined]

    def test_constructor_preserves_workflow_body_steps_inside_loop_body(self) -> None:
        first = RecordingWorkflow(name="first_workflow")
        second = RecordingWorkflow(name="second_workflow")

        flow = make_flow(body_steps=[first, second])

        assert flow.loop_body.steps == (first, second)

    def test_constructor_creates_default_fallback_judge_when_none(self) -> None:
        flow = make_flow(judge=None)

        assert isinstance(flow.judge, BasicFlow)
        assert isinstance(flow.judge.component, StructuredInvokable)
        assert flow.judge.component.component.name == "always_false_judge"

    def test_constructor_normalizes_custom_judge_to_basic_flow(self) -> None:
        judge = make_constant_judge(True)

        flow = make_flow(judge=judge)

        assert isinstance(flow.judge, BasicFlow)
        assert isinstance(flow.judge.component, StructuredInvokable)
        assert flow.judge.component.component is judge

    def test_judge_structured_schema_is_judge_decision(self) -> None:
        flow = make_flow(judge=make_constant_judge(True))
        judge_component = flow.judge.component

        assert isinstance(judge_component, StructuredInvokable)
        assert [spec.name for spec in judge_component.output_schema] == [
            "judge_decision"
        ]

    def test_constructor_uses_loop_body_parameters_as_outer_parameters(self) -> None:
        flow = make_flow()

        assert flow.parameters == flow.loop_body.parameters
        assert [param.name for param in flow.parameters] == ["value"]

    def test_constructor_inherits_loop_body_filter_flag_by_default(self) -> None:
        flow = make_flow(body_steps=[make_increment_component(filter_extraneous_inputs=False)])

        assert flow.filter_extraneous_inputs is False

    def test_constructor_allows_filter_flag_override(self) -> None:
        flow = make_flow(
            body_steps=[make_increment_component(filter_extraneous_inputs=False)],
            filter_extraneous_inputs=True,
        )

        assert flow.filter_extraneous_inputs is True


class TestIterativeFlowPolicySetters:
    def test_return_index_proxies_to_loop_body(self) -> None:
        flow = make_flow(body_steps=[make_increment_component(), make_double_component()])

        flow.return_index = 0

        assert flow.return_index == 0
        assert flow.loop_body.return_index == 0

    def test_return_index_accepts_negative_index(self) -> None:
        flow = make_flow(body_steps=[make_increment_component(), make_double_component()])

        flow.return_index = -1

        assert flow.return_index == -1

    def test_return_index_rejects_non_int(self) -> None:
        flow = make_flow()

        with pytest.raises(TypeError, match="return_index"):
            flow.return_index = "0"  # type: ignore[assignment]

    def test_return_index_rejects_out_of_range(self) -> None:
        flow = make_flow()

        with pytest.raises(IndexError, match="out of range"):
            flow.return_index = 1

    def test_handoff_index_accepts_positive_and_negative_indices(self) -> None:
        flow = make_flow(body_steps=[make_increment_component(), make_double_component()])

        flow.handoff_index = 0
        assert flow.handoff_index == 0
        flow.handoff_index = -1
        assert flow.handoff_index == -1

    def test_handoff_index_rejects_non_int(self) -> None:
        flow = make_flow()

        with pytest.raises(TypeError, match="handoff_index"):
            flow.handoff_index = "0"  # type: ignore[assignment]

    def test_handoff_index_rejects_out_of_range(self) -> None:
        flow = make_flow()

        with pytest.raises(IndexError, match="out of range"):
            flow.handoff_index = 1

    def test_evaluate_index_accepts_positive_and_negative_indices(self) -> None:
        flow = make_flow(body_steps=[make_increment_component(), make_double_component()])

        flow.evaluate_index = 0
        assert flow.evaluate_index == 0
        flow.evaluate_index = -1
        assert flow.evaluate_index == -1

    def test_evaluate_index_rejects_non_int(self) -> None:
        flow = make_flow()

        with pytest.raises(TypeError, match="evaluate_index"):
            flow.evaluate_index = "0"  # type: ignore[assignment]

    def test_evaluate_index_rejects_out_of_range(self) -> None:
        flow = make_flow()

        with pytest.raises(IndexError, match="out of range"):
            flow.evaluate_index = 1

    def test_max_iterations_accepts_positive_int(self) -> None:
        flow = make_flow(max_iterations=1)

        flow.max_iterations = 5

        assert flow.max_iterations == 5

    def test_max_iterations_rejects_non_int(self) -> None:
        flow = make_flow()

        with pytest.raises(TypeError, match="max_iterations"):
            flow.max_iterations = "5"  # type: ignore[assignment]

    @pytest.mark.parametrize("value", [0, -1])
    def test_max_iterations_rejects_zero_or_negative(self, value: int) -> None:
        with pytest.raises(ValueError, match="max_iterations must be > 0"):
            make_flow(max_iterations=value)


class TestIterativeFlowSyncInvokeMaxIterations:
    def test_default_fallback_judge_runs_until_max_iterations(self) -> None:
        flow = make_flow(judge=None, max_iterations=3)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert result == {"value": 3}
        assert metadata.iterations_completed == 3
        assert metadata.judge_approved_early is False

    def test_invoke_returns_final_body_result_after_max_iterations(self) -> None:
        flow = make_flow(judge=make_constant_judge(False), max_iterations=3)

        result = flow.invoke({"value": 0})

        assert result == {"value": 3}

    def test_each_iteration_hands_off_selected_step_result_to_next_iteration(self) -> None:
        flow = make_flow(
            body_steps=[make_increment_component(), make_double_component()],
            judge=make_constant_judge(False),
            max_iterations=2,
            return_index=-1,
            handoff_index=0,
            evaluate_index=-1,
        )

        result = flow.invoke({"value": 1})

        assert result == {"value": 6}

    def test_evaluate_index_controls_payload_given_to_judge(self) -> None:
        seen: list[str] = []
        flow = make_flow(
            body_steps=[make_marker_component("first"), make_marker_component("second")],
            judge=make_marker_judge("second", seen=seen),
            max_iterations=3,
            evaluate_index=1,
        )

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert seen == ["second"]
        assert metadata.iterations_completed == 1
        assert metadata.judge_approved_early is True

    def test_return_index_controls_body_result_that_becomes_outer_result(self) -> None:
        flow = make_flow(
            body_steps=[make_increment_component(), make_double_component()],
            judge=make_constant_judge(True),
            max_iterations=3,
            return_index=0,
            handoff_index=-1,
            evaluate_index=-1,
        )

        result = flow.invoke({"value": 1})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]
        step_results = flow.loop_body.get_step_results(metadata.iteration_records[0].body_run_id)

        assert result == {"value": 2}
        assert step_results == [{"value": 2}, {"value": 4}]

    def test_handoff_index_can_differ_from_return_index(self) -> None:
        flow = make_flow(
            body_steps=[make_increment_component(), make_double_component()],
            judge=make_constant_judge(False),
            max_iterations=2,
            return_index=0,
            handoff_index=1,
            evaluate_index=1,
        )

        result = flow.invoke({"value": 1})

        assert result == {"value": 5}

    def test_evaluate_index_can_differ_from_handoff_index(self) -> None:
        flow = make_flow(
            body_steps=[make_increment_component(), make_double_component()],
            judge=make_threshold_judge(6),
            max_iterations=5,
            return_index=0,
            handoff_index=0,
            evaluate_index=1,
        )

        result = flow.invoke({"value": 1})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert result == {"value": 3}
        assert metadata.iterations_completed == 2
        assert metadata.judge_approved_early is True

    def test_parent_checkpoint_records_filtered_outer_inputs_and_final_result(self) -> None:
        flow = make_flow(judge=make_constant_judge(True), max_iterations=3)

        result = flow.invoke({"value": 0, "extra": "ignored"})
        checkpoint = flow.get_checkpoint(result.run_id)

        assert checkpoint is not None
        assert checkpoint.inputs == {"value": 0}
        assert checkpoint.result == {"value": 1}


class TestIterativeFlowSyncInvokeEarlyStop:
    def test_always_true_judge_stops_after_one_iteration(self) -> None:
        flow = make_flow(judge=make_constant_judge(True), max_iterations=5)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert result == {"value": 1}
        assert metadata.iterations_completed == 1
        assert metadata.judge_approved_early is True

    def test_threshold_judge_stops_when_condition_met(self) -> None:
        flow = make_flow(judge=make_threshold_judge(3), max_iterations=5)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert result == {"value": 3}
        assert metadata.iterations_completed == 3
        assert metadata.judge_approved_early is True

    def test_early_stop_returns_approved_iteration_body_result(self) -> None:
        flow = make_flow(judge=make_threshold_judge(2), max_iterations=5)

        result = flow.invoke({"value": 0})

        assert result == {"value": 2}

    def test_early_stop_does_not_execute_remaining_iterations(self) -> None:
        flow = make_flow(judge=make_constant_judge(True), max_iterations=5)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert len(flow.loop_body.checkpoints) == 1
        assert len(flow.judge.checkpoints) == 1
        assert len(flow.loop_body.checkpoints) == metadata.iterations_completed

    def test_judge_receives_evaluate_result_from_each_iteration(self) -> None:
        seen: list[int] = []
        flow = make_flow(judge=make_threshold_judge(3, seen=seen), max_iterations=5)

        flow.invoke({"value": 0})

        assert seen == [1, 2, 3]


class TestIterativeFlowMetadata:
    def test_metadata_kind_is_iterative(self) -> None:
        flow = make_flow(judge=make_constant_judge(True))

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.kind == "iterative"

    def test_metadata_records_iterations_completed(self) -> None:
        flow = make_flow(judge=make_constant_judge(False), max_iterations=3)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.iterations_completed == 3

    def test_metadata_records_max_iterations(self) -> None:
        flow = make_flow(judge=make_constant_judge(False), max_iterations=4)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.max_iterations == 4

    def test_metadata_records_judge_approved_early_false_when_exhausted(self) -> None:
        flow = make_flow(judge=make_constant_judge(False), max_iterations=2)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.judge_approved_early is False

    def test_metadata_records_judge_approved_early_true_when_stopped(self) -> None:
        flow = make_flow(judge=make_constant_judge(True), max_iterations=2)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.judge_approved_early is True

    def test_metadata_resolves_negative_return_handoff_evaluate_indices(self) -> None:
        flow = make_flow(
            body_steps=[make_increment_component(), make_double_component()],
            judge=make_constant_judge(True),
            return_index=-1,
            handoff_index=-1,
            evaluate_index=-1,
        )

        result = flow.invoke({"value": 1})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.return_step_index == 1
        assert metadata.handoff_step_index == 1
        assert metadata.evaluate_step_index == 1

    def test_metadata_iteration_records_have_zero_based_iteration_numbers(self) -> None:
        flow = make_flow(judge=make_constant_judge(False), max_iterations=3)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert [record.iteration for record in metadata.iteration_records] == [0, 1, 2]

    def test_metadata_iteration_records_body_run_ids_match_loop_body_checkpoints(self) -> None:
        flow = make_flow(judge=make_constant_judge(False), max_iterations=3)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert [record.body_run_id for record in metadata.iteration_records] == [
            checkpoint.run_id for checkpoint in flow.loop_body.checkpoints
        ]

    def test_metadata_iteration_records_judge_run_ids_match_judge_checkpoints(self) -> None:
        flow = make_flow(judge=make_constant_judge(False), max_iterations=3)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert [record.judge_run_id for record in metadata.iteration_records] == [
            checkpoint.run_id for checkpoint in flow.judge.checkpoints
        ]

    def test_metadata_iteration_records_store_each_judge_decision(self) -> None:
        flow = make_flow(judge=make_threshold_judge(2), max_iterations=5)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert [record.judge_decision for record in metadata.iteration_records] == [
            False,
            True,
        ]

    def test_iteration_record_count_matches_iterations_completed(self) -> None:
        flow = make_flow(judge=make_constant_judge(False), max_iterations=4)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert len(metadata.iteration_records) == metadata.iterations_completed


class TestIterativeFlowBodyAndJudgeCheckpointing:
    def test_loop_body_has_one_checkpoint_per_iteration(self) -> None:
        flow = make_flow(judge=make_constant_judge(False), max_iterations=3)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert len(flow.loop_body.checkpoints) == metadata.iterations_completed

    def test_judge_has_one_checkpoint_per_iteration(self) -> None:
        flow = make_flow(judge=make_constant_judge(False), max_iterations=3)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert len(flow.judge.checkpoints) == metadata.iterations_completed

    def test_loop_body_step_results_can_be_retrieved_for_each_iteration(self) -> None:
        flow = make_flow(judge=make_constant_judge(False), max_iterations=3)

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert [
            flow.loop_body.get_step_result(record.body_run_id, 0)
            for record in metadata.iteration_records
        ] == [{"value": 1}, {"value": 2}, {"value": 3}]

    def test_require_body_step_result_returns_mapping_for_retained_checkpoint(self) -> None:
        flow = make_flow(judge=make_constant_judge(True))

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]
        body_run_id = metadata.iteration_records[0].body_run_id

        assert flow._require_body_step_result(
            body_run_id,
            0,
            purpose="test",
        ) == {"value": 1}

    def test_parent_run_id_differs_from_body_and_judge_run_ids(self) -> None:
        flow = make_flow(judge=make_constant_judge(True))

        result = flow.invoke({"value": 0})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]
        record = metadata.iteration_records[0]

        assert result.run_id != record.body_run_id
        assert result.run_id != record.judge_run_id
        assert record.body_run_id != record.judge_run_id


class TestIterativeFlowJudgeDecisionValidation:
    def test_extract_judge_decision_raises_when_missing(self) -> None:
        flow = make_flow()

        with pytest.raises(ValidationError, match="judge_decision"):
            flow._extract_judge_decision(FlowResultDict({}, run_id="judge_run"))

    @pytest.mark.parametrize("decision", ["yes", 1, None])
    def test_extract_judge_decision_raises_when_not_bool(self, decision: Any) -> None:
        flow = make_flow()

        with pytest.raises(ValidationError, match="judge_decision must be bool"):
            flow._extract_judge_decision(
                FlowResultDict({"judge_decision": decision}, run_id="judge_run")
            )

    def test_extract_judge_decision_accepts_false(self) -> None:
        flow = make_flow()

        assert flow._extract_judge_decision(
            FlowResultDict({"judge_decision": False}, run_id="judge_run")
        ) is False

    def test_extract_judge_decision_accepts_true(self) -> None:
        flow = make_flow()

        assert flow._extract_judge_decision(
            FlowResultDict({"judge_decision": True}, run_id="judge_run")
        ) is True

    def test_public_invoke_wraps_invalid_judge_decision_as_execution_error(self) -> None:
        flow = make_flow(judge=make_constant_judge("yes"))

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 0})


class TestIterativeFlowValidationAndErrors:
    def test_run_raises_validation_error_if_body_returns_non_flow_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(judge=make_constant_judge(True))
        monkeypatch.setattr(flow.loop_body, "invoke", lambda inputs: {"value": 1})

        with pytest.raises(ValidationError, match="body returned"):
            flow._run({"value": 0})

    def test_public_invoke_wraps_body_non_flow_result_as_execution_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(judge=make_constant_judge(True))
        monkeypatch.setattr(flow.loop_body, "invoke", lambda inputs: {"value": 1})

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 0})

    def test_run_raises_validation_error_if_judge_returns_non_flow_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(judge=make_constant_judge(True))
        monkeypatch.setattr(flow.judge, "invoke", lambda inputs: {"judge_decision": True})

        with pytest.raises(ValidationError, match="judge returned"):
            flow._run({"value": 0})

    def test_public_invoke_wraps_judge_non_flow_result_as_execution_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(judge=make_constant_judge(True))
        monkeypatch.setattr(flow.judge, "invoke", lambda inputs: {"judge_decision": True})

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 0})

    def test_require_body_step_result_raises_if_result_cannot_be_resolved(self) -> None:
        flow = make_flow()

        with pytest.raises(ValidationError, match="could not resolve handoff"):
            flow._require_body_step_result("missing", 0, purpose="handoff")

    def test_run_raises_validation_error_if_handoff_step_result_cannot_be_resolved(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(judge=make_constant_judge(True))
        monkeypatch.setattr(flow.loop_body, "get_step_result", lambda run_id, step_index: None)

        with pytest.raises(ValidationError, match="could not resolve handoff"):
            flow._run({"value": 0})

    def test_run_raises_validation_error_if_evaluate_step_result_cannot_be_resolved(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(
            body_steps=[make_increment_component(), make_double_component()],
            judge=make_constant_judge(True),
            handoff_index=0,
            evaluate_index=1,
        )

        def fake_get_step_result(run_id: str, step_index: int) -> dict[str, int] | None:
            return {"value": 1} if step_index == 0 else None

        monkeypatch.setattr(flow.loop_body, "get_step_result", fake_get_step_result)

        with pytest.raises(ValidationError, match="could not resolve evaluate"):
            flow._run({"value": 0})

    def test_public_invoke_wraps_body_runtime_error_as_execution_error(self) -> None:
        flow = make_flow(body_steps=[make_raising_body_component()], judge=make_constant_judge(True))

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 0})

    def test_public_invoke_wraps_judge_runtime_error_as_execution_error(self) -> None:
        flow = make_flow(judge=make_raising_judge())

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 0})

    def test_run_raises_validation_error_if_handoff_result_is_not_mapping(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(judge=make_constant_judge(True))
        monkeypatch.setattr(flow.loop_body, "get_step_result", lambda run_id, step_index: 123)

        with pytest.raises(ValidationError, match="handoff body step result must be mapping"):
            flow._run({"value": 0})


class TestIterativeFlowAsyncInvoke:
    def test_async_default_fallback_judge_runs_until_max_iterations(self) -> None:
        flow = make_flow(judge=None, max_iterations=3)

        result = asyncio.run(flow.async_invoke({"value": 0}))
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert result == {"value": 3}
        assert metadata.iterations_completed == 3
        assert metadata.judge_approved_early is False

    def test_async_always_true_judge_stops_after_one_iteration(self) -> None:
        flow = make_flow(judge=make_constant_judge(True), max_iterations=5)

        result = asyncio.run(flow.async_invoke({"value": 0}))
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert result == {"value": 1}
        assert metadata.iterations_completed == 1
        assert metadata.judge_approved_early is True

    def test_async_threshold_judge_stops_when_condition_met(self) -> None:
        flow = make_flow(judge=make_threshold_judge(3), max_iterations=5)

        result = asyncio.run(flow.async_invoke({"value": 0}))
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert result == {"value": 3}
        assert metadata.iterations_completed == 3
        assert metadata.judge_approved_early is True

    def test_async_return_handoff_evaluate_indices_match_sync_behavior(self) -> None:
        flow = make_flow(
            body_steps=[make_increment_component(), make_double_component()],
            judge=make_threshold_judge(6),
            max_iterations=5,
            return_index=0,
            handoff_index=0,
            evaluate_index=1,
        )

        result = asyncio.run(flow.async_invoke({"value": 1}))
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert result == {"value": 3}
        assert metadata.return_step_index == 0
        assert metadata.handoff_step_index == 0
        assert metadata.evaluate_step_index == 1
        assert metadata.iterations_completed == 2

    def test_async_metadata_records_iterations_and_child_run_ids(self) -> None:
        flow = make_flow(judge=make_constant_judge(False), max_iterations=3)

        result = asyncio.run(flow.async_invoke({"value": 0}))
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert [record.iteration for record in metadata.iteration_records] == [0, 1, 2]
        assert [record.body_run_id for record in metadata.iteration_records] == [
            checkpoint.run_id for checkpoint in flow.loop_body.checkpoints
        ]
        assert [record.judge_run_id for record in metadata.iteration_records] == [
            checkpoint.run_id for checkpoint in flow.judge.checkpoints
        ]

    def test_async_loop_body_and_judge_checkpoint_counts_match_iterations(self) -> None:
        flow = make_flow(judge=make_constant_judge(False), max_iterations=3)

        result = asyncio.run(flow.async_invoke({"value": 0}))
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert len(flow.loop_body.checkpoints) == metadata.iterations_completed
        assert len(flow.judge.checkpoints) == metadata.iterations_completed

    def test_async_invalid_judge_decision_public_call_wraps_execution_error(self) -> None:
        flow = make_flow(judge=make_constant_judge("yes"))

        with pytest.raises(ExecutionError, match="_async_run failed"):
            asyncio.run(flow.async_invoke({"value": 0}))

    def test_async_run_raises_validation_error_if_body_returns_non_flow_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(judge=make_constant_judge(True))

        async def bad_async_invoke(self: SequentialFlow, inputs: Mapping[str, Any]) -> dict[str, int]:
            return {"value": 1}

        monkeypatch.setattr(
            flow.loop_body,
            "async_invoke",
            MethodType(bad_async_invoke, flow.loop_body),
        )

        with pytest.raises(ValidationError, match="async body returned"):
            asyncio.run(flow._async_run({"value": 0}))

    def test_async_run_raises_validation_error_if_judge_returns_non_flow_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_flow(judge=make_constant_judge(True))

        async def bad_async_invoke(self: BasicFlow, inputs: Mapping[str, Any]) -> dict[str, bool]:
            return {"judge_decision": True}

        monkeypatch.setattr(
            flow.judge,
            "async_invoke",
            MethodType(bad_async_invoke, flow.judge),
        )

        with pytest.raises(ValidationError, match="async judge returned"):
            asyncio.run(flow._async_run({"value": 0}))


class TestIterativeFlowSerialization:
    def test_to_dict_includes_loop_body_snapshot(self) -> None:
        flow = make_flow()

        data = flow.to_dict()

        assert data["type"] == "IterativeFlow"
        assert data["loop_body"]["type"] == "SequentialFlow"

    def test_to_dict_includes_judge_snapshot(self) -> None:
        flow = make_flow(judge=make_constant_judge(True))

        data = flow.to_dict()

        assert data["judge"]["type"] == "BasicFlow"
        assert data["judge"]["component"]["type"] == "StructuredInvokable"

    def test_to_dict_includes_max_iterations(self) -> None:
        flow = make_flow(max_iterations=7)

        data = flow.to_dict()

        assert data["max_iterations"] == 7

    def test_to_dict_includes_return_handoff_and_evaluate_indices(self) -> None:
        flow = make_flow(
            body_steps=[make_increment_component(), make_double_component()],
            return_index=0,
            handoff_index=1,
            evaluate_index=-1,
        )

        data = flow.to_dict()

        assert data["return_index"] == 0
        assert data["handoff_index"] == 1
        assert data["evaluate_index"] == -1

    def test_to_dict_reflects_mutated_selection_policy(self) -> None:
        flow = make_flow(body_steps=[make_increment_component(), make_double_component()])

        flow.return_index = 0
        flow.handoff_index = 1
        flow.evaluate_index = 0
        flow.max_iterations = 5
        data = flow.to_dict()

        assert data["return_index"] == 0
        assert data["handoff_index"] == 1
        assert data["evaluate_index"] == 0
        assert data["max_iterations"] == 5

    def test_to_dict_after_run_includes_base_checkpoint_summary(self) -> None:
        flow = make_flow(judge=make_constant_judge(True))

        result = flow.invoke({"value": 0})
        data = flow.to_dict()

        assert data["checkpoint_count"] == 1
        assert data["runs"] == [result.run_id]
        assert "checkpoints" not in data
        assert "loop_body" in data
        assert "judge" in data
