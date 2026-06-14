from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from atomic_agentic.core.constants import NO_VAL
from atomic_agentic.core.Exceptions import ExecutionError, ValidationError
from atomic_agentic.core.Parameters import ParamSpec
from atomic_agentic.results.workflows import IterativeFlowResult
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.base import Workflow
from atomic_agentic.workflows.basic import BasicFlow
from atomic_agentic.workflows.iterative import IterativeFlow
from atomic_agentic.workflows.sequential import SequentialFlow


def make_count_param() -> ParamSpec:
    return ParamSpec(
        name="count",
        index=0,
        kind=ParamSpec.POSITIONAL_OR_KEYWORD,
        type="int",
    )


class CounterStepWorkflow(Workflow):
    """_run(inputs) -> ({"count": inputs["count"] + 1}, {})."""

    def __init__(self, *, name: str = "counter_step") -> None:
        super().__init__(
            name=name,
            description="Increment count by 1.",
            parameters=[make_count_param()],
            return_type="dict[str, Any]",
            filter_extraneous_inputs=True,
        )

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"count": inputs["count"] + 1}, {}


class ScalarStepWorkflow(Workflow):
    """_run(inputs) -> (inputs["count"], {}) — a non-mapping body-step result."""

    def __init__(self, *, name: str = "scalar_step") -> None:
        super().__init__(
            name=name,
            description="Return count as a raw scalar.",
            parameters=[make_count_param()],
            return_type="int",
            filter_extraneous_inputs=True,
        )

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return inputs["count"], {}


def make_threshold_judge(threshold: int) -> Tool:
    """Judge whose .invoke({"count": n}).result == (n >= threshold), a raw bool."""

    def at_least(count: int) -> bool:
        return count >= threshold

    return Tool(
        function=at_least,
        name="threshold_judge",
        namespace="tests",
        description="Return whether count has reached the threshold.",
    )


def make_raising_judge() -> Tool:
    def raise_judge(count: int) -> bool:
        raise RuntimeError("judge boom")

    return Tool(
        function=raise_judge,
        name="raising_judge",
        namespace="tests",
        description="Always raises.",
    )


def make_iterative_flow(
    *,
    body_steps: list[Any] | None = None,
    max_iterations: int = 3,
    judge: Any = None,
    approval_value: Any = NO_VAL,
    return_index: int | None = None,
    handoff_index: int | None = None,
    evaluate_index: int | None = None,
    name: str = "iterative_flow",
) -> IterativeFlow:
    return IterativeFlow(
        name=name,
        description="Iterative test flow.",
        body_steps=body_steps if body_steps is not None else [CounterStepWorkflow()],
        judge=judge,
        max_iterations=max_iterations,
        return_index=return_index,
        handoff_index=handoff_index,
        evaluate_index=evaluate_index,
        approval_value=approval_value,
    )


class TestIterativeFlowConstruction:
    def test_body_steps_must_be_non_empty_list(self) -> None:
        with pytest.raises(ValueError, match="body_steps must not be empty"):
            IterativeFlow(
                name="bad_flow",
                description="Bad flow.",
                body_steps=[],
                judge=make_threshold_judge(1),
                approval_value=True,
            )

    def test_non_workflow_body_steps_normalized_into_loop_body(self) -> None:
        component = Tool(
            function=lambda count: {"count": count + 1},
            name="raw_increment",
            namespace="tests",
            description="Increment count.",
        )

        flow = IterativeFlow(
            name="iterative_flow",
            description="Iterative test flow.",
            body_steps=[component],
            judge=make_threshold_judge(1),
            approval_value=True,
        )

        assert isinstance(flow.loop_body, SequentialFlow)
        assert isinstance(flow.loop_body.steps[0], BasicFlow)
        assert flow.loop_body.steps[0].component is component  # type: ignore[attr-defined]

    def test_loop_body_named_with_suffix(self) -> None:
        flow = make_iterative_flow(judge=make_threshold_judge(1), approval_value=True)

        assert flow.loop_body.name == f"{flow.name}_loop_body"

    def test_judge_none_installs_fallback_and_requires_no_val_approval(self) -> None:
        flow = make_iterative_flow(judge=None, approval_value=NO_VAL)

        assert isinstance(flow.judge, BasicFlow)
        assert flow.judge.component.name == "always_false_judge"  # type: ignore[attr-defined]

    def test_judge_none_with_non_no_val_approval_raises(self) -> None:
        with pytest.raises(ValueError):
            make_iterative_flow(judge=None, approval_value=True)

    def test_judge_given_requires_non_no_val_approval(self) -> None:
        with pytest.raises(ValueError):
            make_iterative_flow(judge=make_threshold_judge(2), approval_value=NO_VAL)

    def test_judge_given_with_approval_value_constructs_ok(self) -> None:
        judge = make_threshold_judge(2)
        flow = make_iterative_flow(judge=judge, approval_value=True)

        assert isinstance(flow.judge, BasicFlow)
        assert flow.judge.component is judge  # type: ignore[attr-defined]

    def test_indices_default_to_last_body_step(self) -> None:
        flow = IterativeFlow(
            name="iterative_flow",
            description="Iterative test flow.",
            body_steps=[CounterStepWorkflow(name="first"), CounterStepWorkflow(name="second")],
            judge=make_threshold_judge(1),
            approval_value=True,
        )

        assert flow.return_index == 1
        assert flow.handoff_index == 1
        assert flow.evaluate_index == 1

    def test_indices_accept_negative_resolution_at_construction(self) -> None:
        flow = IterativeFlow(
            name="iterative_flow",
            description="Iterative test flow.",
            body_steps=[CounterStepWorkflow(name="first"), CounterStepWorkflow(name="second")],
            judge=make_threshold_judge(1),
            approval_value=True,
            return_index=-1,
            handoff_index=-1,
            evaluate_index=-1,
        )

        assert flow.return_index == 1
        assert flow.handoff_index == 1
        assert flow.evaluate_index == 1

    def test_indices_reject_out_of_range_at_construction(self) -> None:
        with pytest.raises(IndexError, match="out of range"):
            IterativeFlow(
                name="iterative_flow",
                description="Iterative test flow.",
                body_steps=[CounterStepWorkflow(name="first"), CounterStepWorkflow(name="second")],
                judge=make_threshold_judge(1),
                approval_value=True,
                evaluate_index=5,
            )

    def test_indices_reject_non_int_at_construction(self) -> None:
        with pytest.raises(TypeError, match="step index must be an int"):
            IterativeFlow(
                name="iterative_flow",
                description="Iterative test flow.",
                body_steps=[CounterStepWorkflow(name="first"), CounterStepWorkflow(name="second")],
                judge=make_threshold_judge(1),
                approval_value=True,
                handoff_index="0",  # type: ignore[arg-type]
            )

    def test_return_handoff_evaluate_indices_have_no_setters(self) -> None:
        flow = make_iterative_flow(judge=make_threshold_judge(1), approval_value=True)

        with pytest.raises(AttributeError):
            flow.return_index = 0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            flow.handoff_index = 0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            flow.evaluate_index = 0  # type: ignore[misc]

    def test_max_iterations_setter_validates_positive_int(self) -> None:
        flow = make_iterative_flow(judge=make_threshold_judge(1), approval_value=True)

        flow.max_iterations = 5
        assert flow.max_iterations == 5

        with pytest.raises(TypeError):
            flow.max_iterations = "5"  # type: ignore[assignment]

        with pytest.raises(ValueError):
            flow.max_iterations = 0


class TestIterativeFlowSyncInvoke:
    def test_loop_runs_until_judge_approves(self) -> None:
        flow = make_iterative_flow(
            max_iterations=5, judge=make_threshold_judge(3), approval_value=True
        )

        result = flow.invoke({"count": 0})

        assert result.result == {"count": 3}
        assert isinstance(result, IterativeFlowResult)
        assert len(result.iteration_runs) == 3
        assert len(result.judge_runs) == 3

    def test_loop_stops_at_max_iterations_if_never_approved(self) -> None:
        flow = make_iterative_flow(
            max_iterations=2, judge=make_threshold_judge(100), approval_value=True
        )

        result = flow.invoke({"count": 0})

        assert len(result.iteration_runs) == 2
        assert result.result == {"count": 2}

    def test_fallback_judge_never_approves_runs_to_max_iterations(self) -> None:
        flow = make_iterative_flow(max_iterations=2, judge=None, approval_value=NO_VAL)

        result = flow.invoke({"count": 0})

        assert len(result.iteration_runs) == 2
        assert len(result.judge_runs) == 2
        assert result.result == {"count": 2}

    def test_iterative_flow_result_index_fields(self) -> None:
        flow = make_iterative_flow(
            max_iterations=3, judge=make_threshold_judge(3), approval_value=True
        )

        result = flow.invoke({"count": 0})

        assert result.return_step_index == flow.return_index
        assert result.handoff_step_index == flow.handoff_index
        assert result.evaluate_step_index == flow.evaluate_index
        assert result.max_iterations == flow.max_iterations


class TestIterativeFlowRetrieval:
    def test_get_iteration_results_returns_body_and_judge_pairs(self) -> None:
        flow = make_iterative_flow(
            max_iterations=3, judge=make_threshold_judge(3), approval_value=True
        )

        result = flow.invoke({"count": 0})
        pairs = flow.get_iteration_results(result.run_id)

        assert pairs is not None
        assert len(pairs) == len(result.iteration_runs)
        for body_result, judge_result in pairs:
            assert body_result is not None
            assert judge_result is not None
            assert isinstance(judge_result.result, bool)

    def test_get_iteration_results_returns_none_for_unknown_run_id(self) -> None:
        flow = make_iterative_flow(judge=make_threshold_judge(1), approval_value=True)

        flow.invoke({"count": 0})

        assert flow.get_iteration_results("nope") is None


class TestIterativeFlowAsyncInvoke:
    def test_async_invoke_runs_until_judge_approves(self) -> None:
        flow = make_iterative_flow(
            max_iterations=5, judge=make_threshold_judge(3), approval_value=True
        )

        result = asyncio.run(flow.async_invoke({"count": 0}))

        assert result.result == {"count": 3}
        assert isinstance(result, IterativeFlowResult)
        assert len(result.iteration_runs) == 3
        assert len(result.judge_runs) == 3


class TestIterativeFlowValidationAndErrors:
    def test_evaluate_step_non_mapping_result_raises_validation_error(self) -> None:
        flow = IterativeFlow(
            name="iterative_flow",
            description="Iterative test flow.",
            body_steps=[CounterStepWorkflow(), ScalarStepWorkflow()],
            judge=make_threshold_judge(3),
            approval_value=True,
            return_index=0,
            handoff_index=0,
            evaluate_index=1,
        )

        with pytest.raises(ExecutionError, match="_run failed") as exc_info:
            flow.invoke({"count": 0})

        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "evaluate step" in str(exc_info.value.__cause__)
        assert "must be mapping-shaped" in str(exc_info.value.__cause__)

    def test_handoff_step_non_mapping_result_raises_validation_error(self) -> None:
        flow = IterativeFlow(
            name="iterative_flow",
            description="Iterative test flow.",
            body_steps=[CounterStepWorkflow(), ScalarStepWorkflow()],
            judge=make_threshold_judge(0),
            approval_value=True,
            return_index=0,
            handoff_index=1,
            evaluate_index=0,
        )

        # judge approves on the first iteration (count starts at 0, threshold 0),
        # but the handoff step must still be validated unconditionally.
        with pytest.raises(ExecutionError, match="_run failed") as exc_info:
            flow.invoke({"count": 0})

        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "handoff step" in str(exc_info.value.__cause__)
        assert "must be mapping-shaped" in str(exc_info.value.__cause__)

    def test_judge_invoke_failure_wrapped_as_execution_error(self) -> None:
        flow = make_iterative_flow(judge=make_raising_judge(), approval_value=True)

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"count": 0})


class TestIterativeFlowSerialization:
    def test_to_dict_includes_loop_body_judge_and_policy_fields(self) -> None:
        flow = make_iterative_flow(
            max_iterations=3, judge=make_threshold_judge(3), approval_value=True
        )

        result = flow.invoke({"count": 0})
        data = flow.to_dict()

        assert data["loop_body"] == flow.loop_body.to_dict()
        assert data["judge"] == flow.judge.to_dict()
        assert data["max_iterations"] == flow.max_iterations
        assert data["return_index"] == flow.return_index
        assert data["handoff_index"] == flow.handoff_index
        assert data["evaluate_index"] == flow.evaluate_index
        assert "approval_value" in data
        assert data["checkpoint_count"] == 1
        assert data["runs"] == [result.run_id]
