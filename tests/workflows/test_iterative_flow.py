from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.core.Invokable import StructuredInvokable
from atomic_agentic.exceptions import ExecutionError, ValidationError
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results.workflows import IterativeFlowResult
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.base import Workflow
from atomic_agentic.workflows.iterative import IterativeFlow


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
            namespace="tests",
            description="Increment count by 1.",
            parameters=[make_count_param()],
            return_type="dict[str, Any]",
        )

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"count": inputs["count"] + 1}, {}


class ScalarStepWorkflow(Workflow):
    """_run(inputs) -> (inputs["count"], {}) — a non-mapping body-step result."""

    def __init__(self, *, name: str = "scalar_step") -> None:
        super().__init__(
            name=name,
            namespace="tests",
            description="Return count as a raw scalar.",
            parameters=[make_count_param()],
            return_type="int",
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
    checkers: list[tuple[Any, Any] | None] | None = None,
    result_setting_indices: list[int] | None = None,
    handoff_index: int | None = None,
    fallback_value: Any = NO_VAL,
    name: str = "iterative_flow",
) -> IterativeFlow:
    resolved_body_steps = body_steps if body_steps is not None else [CounterStepWorkflow()]
    resolved_result_setting = (
        result_setting_indices if result_setting_indices is not None else [0]
    )
    return IterativeFlow(
        name=name,
        namespace="tests",
        description="Iterative test flow.",
        body_steps=resolved_body_steps,
        max_iterations=max_iterations,
        checkers=checkers,
        result_setting_indices=resolved_result_setting,
        handoff_index=handoff_index,
        fallback_value=fallback_value,
    )


class TestIterativeFlowConstruction:
    def test_body_steps_must_be_list_type(self) -> None:
        with pytest.raises(TypeError, match="body_steps must be a list"):
            IterativeFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                body_steps=(CounterStepWorkflow(),),  # type: ignore[arg-type]
                result_setting_indices=[0],
            )

    def test_body_steps_must_be_non_empty_list(self) -> None:
        with pytest.raises(ValueError, match="body_steps must not be empty"):
            IterativeFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                body_steps=[],
            )

    def test_body_steps_items_must_be_atomic_invokable(self) -> None:
        with pytest.raises(TypeError, match="body_steps items must be AtomicInvokable"):
            IterativeFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                body_steps=[object()],  # type: ignore[list-item]
            )

    def test_body_steps_stored_exactly_as_configured_no_wrapping(self) -> None:
        component = Tool(
            function=lambda count: {"count": count + 1},
            name="raw_increment",
            namespace="tests",
            description="Increment count.",
        )

        flow = IterativeFlow(
            name="iterative_flow",
            namespace="tests",
            description="Iterative test flow.",
            body_steps=[component],
            result_setting_indices=[0],
        )

        assert flow.body_steps == (component,)
        assert flow.body_steps[0] is component

    def test_max_iterations_rejects_non_int_at_construction(self) -> None:
        with pytest.raises(TypeError, match="max_iterations must be an int"):
            make_iterative_flow(max_iterations="3")  # type: ignore[arg-type]

    def test_max_iterations_rejects_non_positive_at_construction(self) -> None:
        with pytest.raises(ValueError, match="max_iterations must be > 0"):
            make_iterative_flow(max_iterations=0)

    def test_result_setting_indices_default_empty(self) -> None:
        flow = IterativeFlow(
            name="iterative_flow",
            namespace="tests",
            description="Iterative test flow.",
            body_steps=[CounterStepWorkflow()],
            fallback_value=0,
        )

        assert flow.result_setting_indices == ()

    def test_result_setting_indices_resolve_negative_and_reject_out_of_range(self) -> None:
        flow = IterativeFlow(
            name="iterative_flow",
            namespace="tests",
            description="Iterative test flow.",
            body_steps=[CounterStepWorkflow(name="first"), CounterStepWorkflow(name="second")],
            result_setting_indices=[-1],
        )
        assert flow.result_setting_indices == (1,)

        with pytest.raises(IndexError, match="out of range"):
            IterativeFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                body_steps=[CounterStepWorkflow(name="first"), CounterStepWorkflow(name="second")],
                result_setting_indices=[5],
            )

    def test_result_setting_indices_reject_non_int(self) -> None:
        with pytest.raises(TypeError, match="result_setting_indices items must be int"):
            IterativeFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                body_steps=[CounterStepWorkflow()],
                result_setting_indices=["0"],  # type: ignore[list-item]
            )

    def test_handoff_index_defaults_to_last_step(self) -> None:
        flow = IterativeFlow(
            name="iterative_flow",
            namespace="tests",
            description="Iterative test flow.",
            body_steps=[CounterStepWorkflow(name="first"), CounterStepWorkflow(name="second")],
            result_setting_indices=[0],
        )
        assert flow.handoff_index == 1

    def test_handoff_index_accepts_negative_resolution(self) -> None:
        flow = IterativeFlow(
            name="iterative_flow",
            namespace="tests",
            description="Iterative test flow.",
            body_steps=[CounterStepWorkflow(name="first"), CounterStepWorkflow(name="second")],
            result_setting_indices=[0],
            handoff_index=-2,
        )
        assert flow.handoff_index == 0

    def test_handoff_index_rejects_non_int(self) -> None:
        with pytest.raises(TypeError, match="step index must be an int"):
            IterativeFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                body_steps=[CounterStepWorkflow(name="first"), CounterStepWorkflow(name="second")],
                result_setting_indices=[0],
                handoff_index="0",  # type: ignore[arg-type]
            )

    def test_handoff_index_rejects_out_of_range(self) -> None:
        with pytest.raises(IndexError, match="out of range"):
            IterativeFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                body_steps=[CounterStepWorkflow(name="first"), CounterStepWorkflow(name="second")],
                result_setting_indices=[0],
                handoff_index=5,
            )

    def test_empty_result_setting_indices_without_fallback_raises(self) -> None:
        with pytest.raises(ValueError, match="fallback_value is NO_VAL"):
            IterativeFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                body_steps=[CounterStepWorkflow()],
            )

    def test_empty_result_setting_indices_with_fallback_constructs_ok(self) -> None:
        flow = IterativeFlow(
            name="iterative_flow",
            namespace="tests",
            description="Iterative test flow.",
            body_steps=[CounterStepWorkflow()],
            fallback_value=0,
        )
        assert flow.fallback_value == 0
        assert flow.return_type == "int"

    def test_return_type_from_single_result_setting_step(self) -> None:
        flow = make_iterative_flow(result_setting_indices=[0])
        assert flow.return_type == "dict[str, Any]"

    def test_return_type_union_when_result_setting_steps_differ(self) -> None:
        flow = IterativeFlow(
            name="iterative_flow",
            namespace="tests",
            description="Iterative test flow.",
            body_steps=[CounterStepWorkflow(name="first"), ScalarStepWorkflow(name="second")],
            result_setting_indices=[0, 1],
            handoff_index=0,
        )
        assert flow.return_type == "dict[str, Any] | int"

    def test_checkers_bulk_construction_replays_add_checker(self) -> None:
        judge = make_threshold_judge(3)
        flow = make_iterative_flow(checkers=[(judge, True)])

        checker = flow.checkers[0]
        assert checker[0] is judge
        assert checker[1] is True

    def test_max_iterations_setter_validates_positive_int(self) -> None:
        flow = make_iterative_flow()

        flow.max_iterations = 5
        assert flow.max_iterations == 5

        with pytest.raises(TypeError):
            flow.max_iterations = "5"  # type: ignore[assignment]

        with pytest.raises(ValueError):
            flow.max_iterations = 0

    def test_include_trace_defaults_to_true_and_is_mutable(self) -> None:
        flow = make_iterative_flow()

        assert flow.include_trace is True
        flow.include_trace = False
        assert flow.include_trace is False


class TestIterativeFlowCheckerMutation:
    def test_add_checker_registers_and_visible_via_checkers_property(self) -> None:
        flow = make_iterative_flow(checkers=None)
        judge = make_threshold_judge(3)

        flow.add_checker(0, judge, True)

        assert flow.checkers[0] is not None
        assert flow.checkers[0][0] is judge
        assert flow.checkers[0][1] is True

    def test_add_checker_rejects_non_int_index(self) -> None:
        flow = make_iterative_flow()

        with pytest.raises(TypeError, match="checker index must be an int"):
            flow.add_checker("0", make_threshold_judge(3), True)  # type: ignore[arg-type]

    def test_add_checker_rejects_out_of_range_index_no_wraparound(self) -> None:
        flow = make_iterative_flow()

        with pytest.raises(IndexError, match="out of range"):
            flow.add_checker(-1, make_threshold_judge(3), True)

        with pytest.raises(IndexError, match="out of range"):
            flow.add_checker(5, make_threshold_judge(3), True)

    def test_add_checker_rejects_non_atomic_judge(self) -> None:
        flow = make_iterative_flow()

        with pytest.raises(TypeError, match="checker judge must be AtomicInvokable"):
            flow.add_checker(0, object(), True)  # type: ignore[arg-type]

    def test_add_checker_rejects_duplicate_index(self) -> None:
        flow = make_iterative_flow(checkers=[(make_threshold_judge(3), True)])

        with pytest.raises(ValueError, match="already registered"):
            flow.add_checker(0, make_threshold_judge(1), True)

    def test_remove_checker_removes_registered_checker(self) -> None:
        flow = make_iterative_flow(checkers=[(make_threshold_judge(3), True)])

        flow.remove_checker(0)

        assert flow.checkers[0] is None

    def test_remove_checker_rejects_unregistered_index(self) -> None:
        flow = make_iterative_flow()

        with pytest.raises(ValueError, match="no checker is registered"):
            flow.remove_checker(0)


def make_structured_step(name: str, output_schema: list[str]) -> StructuredInvokable:
    def step(count: int) -> dict[str, int]:
        return {"count": count}

    tool = Tool(
        function=step,
        name=name,
        namespace="tests",
        description=f"Step {name}.",
    )
    return StructuredInvokable(
        component=tool,
        output_schema=output_schema,
        name=f"structured_{name}",
        description=f"Structured step {name}.",
    )


class TestIterativeFlowExtraDescription:
    def test_surfaces_result_setting_step_extra_description_plus_iteration_bound(self) -> None:
        step = make_structured_step("counter", ["count"])
        flow = make_iterative_flow(body_steps=[step], max_iterations=5, result_setting_indices=[0])

        assert flow._extra_description() == (
            "Output schema: [count]\nRuns up to 5 iteration(s)."
        )

    def test_states_iteration_bound_only_when_no_result_setting_extra(self) -> None:
        flow = make_iterative_flow(max_iterations=2)

        assert flow._extra_description() == "Runs up to 2 iteration(s)."


class TestIterativeFlowSyncInvoke:
    def test_loop_runs_until_checker_approves(self) -> None:
        flow = make_iterative_flow(
            max_iterations=5,
            checkers=[(make_threshold_judge(3), True)],
        )

        result = flow.invoke({"count": 0})

        assert isinstance(result, IterativeFlowResult)
        assert result.result == {"count": 3}
        assert result.exited_early is True
        assert result.iterations_completed == 3
        assert result.triggering_step == 0
        assert len(result.trace) == 6
        assert [r.result for r in result.trace] == [
            {"count": 1}, False,
            {"count": 2}, False,
            {"count": 3}, True,
        ]

    def test_loop_stops_at_max_iterations_if_never_approved(self) -> None:
        flow = make_iterative_flow(
            max_iterations=2,
            checkers=[(make_threshold_judge(100), True)],
        )

        result = flow.invoke({"count": 0})

        assert result.exited_early is False
        assert result.triggering_step is None
        assert result.iterations_completed == 2
        assert result.result == {"count": 2}
        assert len(result.trace) == 4

    def test_no_checkers_runs_to_max_iterations(self) -> None:
        flow = make_iterative_flow(max_iterations=2, checkers=None)

        result = flow.invoke({"count": 0})

        assert result.exited_early is False
        assert result.iterations_completed == 2
        assert result.result == {"count": 2}
        assert len(result.trace) == 2

    def test_trace_none_when_include_trace_disabled(self) -> None:
        flow = make_iterative_flow(max_iterations=2, checkers=None)
        flow.include_trace = False

        result = flow.invoke({"count": 0})

        assert result.trace is None

    def test_iterative_flow_result_policy_fields(self) -> None:
        flow = make_iterative_flow(
            max_iterations=3,
            checkers=[(make_threshold_judge(3), True)],
        )

        result = flow.invoke({"count": 0})

        assert result.result_setting_indices == flow.result_setting_indices
        assert result.handoff_index == flow.handoff_index
        assert result.max_iterations == flow.max_iterations


class TestIterativeFlowAsyncInvoke:
    def test_async_invoke_runs_until_checker_approves(self) -> None:
        flow = make_iterative_flow(
            max_iterations=5,
            checkers=[(make_threshold_judge(3), True)],
        )

        result = asyncio.run(flow.async_invoke({"count": 0}))

        assert result.result == {"count": 3}
        assert isinstance(result, IterativeFlowResult)
        assert result.exited_early is True
        assert result.iterations_completed == 3
        assert len(result.trace) == 6


class TestIterativeFlowValidationAndErrors:
    def test_checker_step_non_mapping_result_raises_validation_error(self) -> None:
        flow = IterativeFlow(
            name="iterative_flow",
            namespace="tests",
            description="Iterative test flow.",
            body_steps=[ScalarStepWorkflow()],
            max_iterations=1,
            checkers=[(make_threshold_judge(3), True)],
            result_setting_indices=[0],
            handoff_index=0,
        )

        with pytest.raises(ExecutionError, match="_run failed") as exc_info:
            flow.invoke({"count": 0})

        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "checker at step 0" in str(exc_info.value.__cause__)
        assert "mapping-shaped result" in str(exc_info.value.__cause__)

    def test_handoff_step_non_mapping_result_raises_validation_error_when_continuing(self) -> None:
        flow = IterativeFlow(
            name="iterative_flow",
            namespace="tests",
            description="Iterative test flow.",
            body_steps=[ScalarStepWorkflow()],
            max_iterations=2,
            result_setting_indices=[0],
            handoff_index=0,
        )

        with pytest.raises(ExecutionError, match="_run failed") as exc_info:
            flow.invoke({"count": 0})

        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "handoff step 0" in str(exc_info.value.__cause__)
        assert "mapping-shaped result" in str(exc_info.value.__cause__)

    def test_non_final_step_non_mapping_result_raises_validation_error_when_chaining(self) -> None:
        flow = IterativeFlow(
            name="iterative_flow",
            namespace="tests",
            description="Iterative test flow.",
            body_steps=[ScalarStepWorkflow(), CounterStepWorkflow()],
            max_iterations=1,
            result_setting_indices=[1],
        )

        with pytest.raises(ExecutionError, match="_run failed") as exc_info:
            flow.invoke({"count": 0})

        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "must produce a mapping-shaped result to chain" in str(exc_info.value.__cause__)

    def test_judge_invoke_failure_wrapped_as_execution_error(self) -> None:
        flow = make_iterative_flow(
            checkers=[(make_raising_judge(), True)],
        )

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"count": 0})


class TestIterativeFlowSerialization:
    def test_to_dict_includes_body_steps_checkers_and_policy_fields(self) -> None:
        flow = make_iterative_flow(
            max_iterations=3,
            checkers=[(make_threshold_judge(3), True)],
        )

        flow.invoke({"count": 0})
        data = flow.to_dict()

        assert data["step_count"] == 1
        assert data["checkers"] == [
            {
                "index": 0,
                "judge": flow.checkers[0][0].to_dict(),
                "approval_value": True,
            }
        ]
        assert data["result_setting_indices"] == list(flow.result_setting_indices)
        assert data["handoff_index"] == flow.handoff_index
        assert data["fallback_value"] == flow.fallback_value
        assert data["max_iterations"] == flow.max_iterations
        assert "checkpoints" not in data
        assert "loop_body" not in data
