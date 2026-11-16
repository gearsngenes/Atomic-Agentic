from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import pytest

from atomic_agentic.exceptions import ExecutionError, ValidationError
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results import AtomicResult
from atomic_agentic.models.results.workflows import SequentialFlowResult
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.base import Workflow
from atomic_agentic.workflows.sequential import SequentialFlow


class EchoWorkflow(Workflow):
    """Same shape as other files' EchoWorkflow: _run returns ({"value": inputs["value"]}, {})."""

    def __init__(
        self,
        *,
        name: str = "echo_workflow",
        namespace: str = "tests",
        description: str = "Echo workflow.",
    ) -> None:
        super().__init__(
            name=name,
            namespace=namespace,
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


def make_step(function: Any, *, name: str) -> Tool:
    return Tool(
        function=function,
        name=name,
        namespace="tests",
        description=f"Test tool {name}.",
    )


class _ExtraDescriptionTool(Tool):
    """Tool whose _extra_description is a fixed string, for flow/Tool chaining tests."""

    def __init__(self, *args: Any, extra: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._extra = extra

    def _extra_description(self) -> str:
        return self._extra


def make_three_step_flow(
    *, return_index: int | None = None, second: Tool | None = None
) -> SequentialFlow:
    return SequentialFlow(
        name="sequential_flow",
        namespace="tests",
        description="Sequential test flow.",
        steps=[
            make_step(first_step, name="first_step"),
            second or make_step(second_step, name="second_step"),
            make_step(third_step, name="third_step"),
        ],
        return_index=return_index,
    )


class TestSequentialFlowConstruction:
    def test_constructor_rejects_non_list_steps(self) -> None:
        with pytest.raises(TypeError, match="steps must be"):
            SequentialFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                steps=(EchoWorkflow(),),  # type: ignore[arg-type]
            )

    def test_constructor_rejects_empty_steps(self) -> None:
        with pytest.raises(ValueError, match="steps must not be empty"):
            SequentialFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                steps=[],
            )

    def test_constructor_rejects_non_invokable_step(self) -> None:
        with pytest.raises(TypeError, match="AtomicInvokable"):
            SequentialFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                steps=[object()],  # type: ignore[list-item]
            )

    def test_constructor_stores_steps_exactly_as_configured(self) -> None:
        child = EchoWorkflow()

        flow = SequentialFlow(
            name="sequential_flow",
            namespace="tests",
            description="Sequential test flow.",
            steps=[child],
        )

        assert flow.steps == (child,)
        assert flow.steps[0] is child

    def test_constructor_stores_tool_step_unwrapped(self) -> None:
        component = make_step(first_step, name="first_step")

        flow = SequentialFlow(
            name="sequential_flow",
            namespace="tests",
            description="Sequential test flow.",
            steps=[component],
        )

        assert flow.steps == (component,)
        assert flow.steps[0] is component

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
        """No negative-index wraparound support -- -1 is simply out of range."""
        with pytest.raises(IndexError, match="out of range"):
            make_three_step_flow(return_index=-1)

    def test_return_index_rejects_out_of_range_at_construction(self) -> None:
        with pytest.raises(IndexError, match="out of range"):
            make_three_step_flow(return_index=3)

    def test_return_index_has_no_setter(self) -> None:
        flow = make_three_step_flow()

        with pytest.raises(AttributeError):
            flow.return_index = 0  # type: ignore[misc]

    def test_include_trace_defaults_to_true_and_is_mutable(self) -> None:
        flow = make_three_step_flow()

        assert flow.include_trace is True
        flow.include_trace = False
        assert flow.include_trace is False


class TestSequentialFlowExtraDescription:
    def test_surfaces_return_index_step_extra_description(self) -> None:
        second = _ExtraDescriptionTool(
            function=second_step,
            name="second_step",
            namespace="tests",
            description="Test tool second_step.",
            extra="second step extra",
        )
        flow = make_three_step_flow(return_index=1, second=second)

        assert flow._extra_description() == "second step extra"
        assert flow.description == "Sequential test flow.\nsecond step extra"

    def test_empty_when_return_index_step_has_no_extra_description(self) -> None:
        flow = SequentialFlow(
            name="sequential_flow",
            namespace="tests",
            description="Sequential test flow.",
            steps=[EchoWorkflow()],
        )

        assert flow._extra_description() == ""
        assert flow.description == flow._description


class TestSequentialFlowSyncInvoke:
    def test_invoke_returns_sequential_flow_result(self) -> None:
        flow = make_three_step_flow(return_index=1)

        result = flow.invoke({"start": 1, "value": 2})

        assert isinstance(result, SequentialFlowResult)
        assert result.result == {"second": 6}
        assert result.return_index == 1

    def test_trace_holds_every_step_result_in_order(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})

        assert result.trace is not None
        assert len(result.trace) == 3
        assert [r.result for r in result.trace] == [
            {"first": 3},
            {"second": 6},
            {"third": "value=6"},
        ]

    def test_trace_is_none_when_include_trace_disabled(self) -> None:
        flow = make_three_step_flow()
        flow.include_trace = False

        result = flow.invoke({"value": 2})

        assert result.trace is None

    def test_each_step_invoked_with_previous_steps_mapping_output(self) -> None:
        flow = make_three_step_flow()

        result = flow.invoke({"value": 2})

        assert result.trace[0].result == {"first": 3}
        assert result.trace[1].result == {"second": 6}
        assert result.trace[2].result == {"third": "value=6"}


class TestSequentialFlowAsyncInvoke:
    def test_async_invoke_returns_sequential_flow_result(self) -> None:
        flow = make_three_step_flow(return_index=1)

        result = asyncio.run(flow.async_invoke({"value": 2}))

        assert isinstance(result, SequentialFlowResult)
        assert result.result == {"second": 6}
        assert result.return_index == 1
        assert len(result.trace) == 3


class TestSequentialFlowValidationAndErrors:
    @staticmethod
    def _make_bad_result() -> Any:
        """A result object whose ``.result`` is a non-mapping payload."""
        now = datetime.now(timezone.utc)
        return AtomicResult(
            result=123, invoker_id="tests.bad_step", started_at=now, ended_at=now
        )

    def test_non_final_step_non_mapping_result_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flow = make_three_step_flow()

        bad_result = self._make_bad_result()
        monkeypatch.setattr(flow.steps[0], "invoke", lambda inputs: bad_result)

        with pytest.raises(ValidationError, match="non-mapping result"):
            flow._run({"value": 1})

    def test_invoke_wraps_step_contract_failure_as_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flow = make_three_step_flow()

        bad_result = self._make_bad_result()
        monkeypatch.setattr(flow.steps[0], "invoke", lambda inputs: bad_result)

        with pytest.raises(ExecutionError, match="_run failed") as exc_info:
            flow.invoke({"value": 1})

        assert isinstance(exc_info.value.__cause__, ValidationError)

    def test_async_invoke_wraps_step_contract_failure_as_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flow = make_three_step_flow()

        bad_result = self._make_bad_result()

        async def bad_async_invoke(inputs: Mapping[str, Any]) -> Any:
            return bad_result

        monkeypatch.setattr(flow.steps[0], "async_invoke", bad_async_invoke)

        with pytest.raises(ExecutionError, match="_async_run failed") as exc_info:
            asyncio.run(flow.async_invoke({"value": 1}))

        assert isinstance(exc_info.value.__cause__, ValidationError)


class TestSequentialFlowSerialization:
    def test_to_dict_includes_steps_and_return_index(self) -> None:
        flow = make_three_step_flow()

        flow.invoke({"value": 2})
        data = flow.to_dict()

        assert data["return_index"] == flow.return_index
        assert data["step_count"] == 3
        assert "steps" in data and len(data["steps"]) == 3
        assert "checkpoints" not in data


class TestSequentialFlowInvokeHook:
    def test_result_cls_and_no_make_result(self) -> None:
        assert SequentialFlow.result_cls is SequentialFlowResult
        assert not hasattr(SequentialFlow, "make_result")

    def test_sync_and_async_results_carry_return_index(self) -> None:
        flow = make_three_step_flow(return_index=1)

        assert flow.invoke({"value": 2}).return_index == flow.return_index == 1
        assert (
            asyncio.run(flow.async_invoke({"value": 2})).return_index
            == flow.return_index
        )
