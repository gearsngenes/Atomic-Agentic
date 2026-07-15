from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from atomic_agentic.exceptions import ExecutionError, ValidationError
from atomic_agentic.core.Invokable import StructuredInvokable
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results.workflows import RoutingFlowResult
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.base import Workflow
from atomic_agentic.workflows.basic import BasicFlow
from atomic_agentic.workflows.routing import RoutingFlow


def make_value_param() -> ParamSpec:
    return ParamSpec(
        name="value",
        index=0,
        kind=ParamSpec.POSITIONAL_OR_KEYWORD,
        type="int",
    )


class EchoWorkflow(Workflow):
    """_run(inputs) -> ({**inputs, "branch": self._tag}, {})."""

    def __init__(
        self,
        tag: str,
        *,
        name: str | None = None,
        namespace: str = "tests",
        description: str | None = None,
        return_type: str = "dict[str, Any]",
        filter_extraneous_inputs: bool = True,
    ) -> None:
        super().__init__(
            name=name or f"echo_{tag}",
            namespace=namespace,
            description=description or f"Echo workflow {tag}.",
            parameters=[make_value_param()],
            return_type=return_type,
            filter_extraneous_inputs=filter_extraneous_inputs,
        )
        self._tag = tag

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {**inputs, "branch": self._tag}, {}


def make_branch(tag: str, **kwargs: Any) -> EchoWorkflow:
    return EchoWorkflow(tag, **kwargs)


def make_structured_router(selector: Any) -> StructuredInvokable:
    """Router whose .invoke(inputs).result == selector (the raw selector)."""

    def select(value: Any) -> Any:
        return selector

    tool = Tool(
        function=select,
        name="select_constant",
        namespace="tests",
        description="Return a constant selector.",
    )
    return StructuredInvokable(
        component=tool,
        output_schema=None,
        name="structured_router",
        description="Structured router returning a raw selector.",
    )


class RouterWorkflow(Workflow):
    """_run(inputs) -> (selector, {}) — router whose raw result IS the selector."""

    def __init__(self, selector: Any, *, raise_error: bool = False) -> None:
        super().__init__(
            name="router_workflow",
            namespace="tests",
            description="Router workflow returning a constant selector.",
            parameters=[make_value_param()],
            return_type="Any",
            filter_extraneous_inputs=True,
        )
        self._selector = selector
        self._raise_error = raise_error

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        if self._raise_error:
            raise RuntimeError("router boom")
        return self._selector, {}


def make_router(selector: Any, **kwargs: Any) -> RouterWorkflow:
    return RouterWorkflow(selector, **kwargs)


class RaisingBranch(Workflow):
    """_run(inputs) raises RuntimeError unconditionally."""

    def __init__(self) -> None:
        super().__init__(
            name="raising_branch",
            namespace="tests",
            description="Branch that always raises.",
            parameters=[make_value_param()],
            return_type="dict[str, Any]",
            filter_extraneous_inputs=True,
        )

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        raise RuntimeError("branch boom")


class UnhashableSelectorRouter(Workflow):
    """_run(inputs) -> ([1, 2], {}) — an unhashable selector."""

    def __init__(self) -> None:
        super().__init__(
            name="unhashable_router",
            namespace="tests",
            description="Router returning an unhashable selector.",
            parameters=[make_value_param()],
            return_type="Any",
            filter_extraneous_inputs=True,
        )

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return [1, 2], {}


def make_list_routing_flow(selector: Any, **router_kwargs: Any) -> RoutingFlow:
    return RoutingFlow(
        name="routing_flow",
        namespace="tests",
        description="Routing test flow.",
        branches=[make_branch("a"), make_branch("b"), make_branch("c")],
        router=make_router(selector, **router_kwargs),
    )


def make_dict_routing_flow(selector: Any) -> RoutingFlow:
    return RoutingFlow(
        name="routing_flow",
        namespace="tests",
        description="Routing test flow.",
        branches={"left": make_branch("left"), "right": make_branch("right")},
        router=make_router(selector),
    )


class TestRoutingFlowConstruction:
    def test_branches_must_be_non_empty_list_tuple_or_dict(self) -> None:
        with pytest.raises(ValueError, match="branches must not be empty"):
            RoutingFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                branches=[],
                router=make_router(0),
            )

        with pytest.raises(TypeError, match="branches must be"):
            RoutingFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                branches="x",  # type: ignore[arg-type]
                router=make_router(0),
            )

    def test_list_branches_normalized_to_basic_flow(self) -> None:
        component = make_structured_router(0)
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[component, make_branch("b")],
            router=make_router(0),
        )

        assert isinstance(flow.branches[0], BasicFlow)
        assert flow.branches[0].component is component  # type: ignore[attr-defined]
        assert isinstance(flow.branches[1], EchoWorkflow)

    def test_dict_branches_normalized_per_value(self) -> None:
        component = make_structured_router(0)
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches={"a": component, "b": make_branch("b")},
            router=make_router("a"),
        )

        assert isinstance(flow.branches["a"], BasicFlow)
        assert flow.branches["a"].component is component  # type: ignore[attr-defined]
        assert isinstance(flow.branches["b"], EchoWorkflow)

    def test_router_normalized_workflow_kept_as_is_other_wrapped(self) -> None:
        router = make_router(0)
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[make_branch("a"), make_branch("b")],
            router=router,
        )
        assert flow.router is router

        component_router = make_structured_router(0)
        flow2 = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[make_branch("a"), make_branch("b")],
            router=component_router,
        )
        assert isinstance(flow2.router, BasicFlow)
        assert flow2.router.component is component_router  # type: ignore[attr-defined]

    def test_constructor_rejects_non_atomic_router(self) -> None:
        with pytest.raises(TypeError, match="Workflow or AtomicInvokable"):
            RoutingFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                branches=[make_branch("a"), make_branch("b")],
                router=object(),  # type: ignore[arg-type]
            )

    def test_parameters_default_to_router_parameters(self) -> None:
        router = make_router(0)
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[make_branch("a"), make_branch("b")],
            router=router,
        )

        assert flow.parameters == router.parameters

    def test_return_type_is_shared_or_union_of_branch_return_types(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[make_branch("a"), make_branch("b")],
            router=make_router(0),
        )
        assert flow.return_type == "dict[str, Any]"

        flow2 = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[
                make_branch("a", return_type="dict[str, Any]"),
                make_branch("b", return_type="str"),
            ],
            router=make_router(0),
        )
        assert flow2.return_type == "dict[str, Any] | str"


def make_structured_branch(tag: str, output_schema: list[str] | None) -> StructuredInvokable:
    def select(value: Any) -> Any:
        return {"branch": tag, "value": value}

    tool = Tool(
        function=select,
        name=f"branch_{tag}",
        namespace="tests",
        description=f"Branch {tag}.",
    )
    return StructuredInvokable(
        component=tool,
        output_schema=output_schema,
        name=f"structured_branch_{tag}",
        description=f"Structured branch {tag}.",
    )


class TestRoutingFlowExtraDescription:
    def test_unanimous_non_empty_extra_appends_shared_content(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[
                make_structured_branch("a", ["value"]),
                make_structured_branch("b", ["value"]),
            ],
            router=make_structured_router(0),
        )

        assert flow._extra_description() == (
            "Selects 1 of 2 branches at runtime.\nOutput schema: [value]"
        )

    def test_divergent_extras_state_branch_count_only(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[
                make_structured_branch("a", ["value"]),
                make_branch("b"),
            ],
            router=make_structured_router(0),
        )

        assert flow._extra_description() == "Selects 1 of 2 branches at runtime."

    def test_unanimous_empty_extra_states_branch_count_only(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[make_branch("a"), make_branch("b")],
            router=make_structured_router(0),
        )

        assert flow._extra_description() == "Selects 1 of 2 branches at runtime."


class TestRoutingFlowSyncInvokeListBranches:
    def test_invoke_routes_to_int_selected_branch(self) -> None:
        flow = make_list_routing_flow(1)

        result = flow.invoke({"value": 5})

        assert result.result == {"value": 5, "branch": "b"}

    def test_routing_flow_result_fields(self) -> None:
        flow = make_list_routing_flow(1)

        result = flow.invoke({"value": 5})

        assert isinstance(result, RoutingFlowResult)
        assert result.selected_key == 1
        assert result.chosen_branch_run == flow.branches[1].checkpoints[-1].result.run_id
        assert isinstance(result.router_run_id, str)

    def test_get_router_decision_returns_raw_selector(self) -> None:
        flow = make_list_routing_flow(1)

        result = flow.invoke({"value": 5})

        assert flow.get_router_decision(result.run_id) == 1

    def test_get_router_decision_returns_none_for_unknown_run_id(self) -> None:
        flow = make_list_routing_flow(1)

        flow.invoke({"value": 5})

        assert flow.get_router_decision("nope") is None


class TestRoutingFlowSyncInvokeDictBranches:
    def test_invoke_routes_to_dict_selected_branch(self) -> None:
        flow = make_dict_routing_flow("right")

        result = flow.invoke({"value": 5})

        assert result.result == {"value": 5, "branch": "right"}
        assert result.selected_key == "right"

    def test_dict_branch_selector_must_be_present_key(self) -> None:
        flow = make_dict_routing_flow("missing")

        with pytest.raises(ExecutionError, match="_run failed") as exc_info:
            flow.invoke({"value": 5})

        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "not among configured branch keys" in str(exc_info.value.__cause__)

    def test_dict_branch_selector_must_be_hashable(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches={"left": make_branch("left"), "right": make_branch("right")},
            router=UnhashableSelectorRouter(),
        )

        with pytest.raises(ExecutionError, match="_run failed") as exc_info:
            flow.invoke({"value": 5})

        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "not a valid (hashable) branch key" in str(exc_info.value.__cause__)


class TestRoutingFlowValidationAndErrors:
    @pytest.mark.parametrize("make_flow", [make_list_routing_flow, make_dict_routing_flow])
    def test_bool_selector_always_rejected(self, make_flow: Any) -> None:
        flow = make_flow(True)

        with pytest.raises(ExecutionError, match="_run failed") as exc_info:
            flow.invoke({"value": 5})

        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "must not be a bool" in str(exc_info.value.__cause__)

    def test_int_selector_out_of_range_for_list_branches(self) -> None:
        flow = make_list_routing_flow(5)

        with pytest.raises(ExecutionError, match="_run failed") as exc_info:
            flow.invoke({"value": 5})

        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "out of range" in str(exc_info.value.__cause__)

    def test_non_int_selector_for_list_branches(self) -> None:
        flow = make_list_routing_flow("x")

        with pytest.raises(ExecutionError, match="_run failed") as exc_info:
            flow.invoke({"value": 5})

        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "must be an int, got" in str(exc_info.value.__cause__)

    def test_router_invoke_failure_wrapped_as_execution_error(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[make_branch("a"), make_branch("b")],
            router=make_router(0, raise_error=True),
        )

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 5})

    def test_chosen_branch_invoke_failure_wrapped_as_execution_error(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[RaisingBranch(), make_branch("b")],
            router=make_router(0),
        )

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 5})

    def test_async_invoke_wraps_invalid_selector_as_execution_error(self) -> None:
        flow = make_list_routing_flow(5)

        with pytest.raises(ExecutionError, match="_async_run failed") as exc_info:
            asyncio.run(flow.async_invoke({"value": 5}))

        assert isinstance(exc_info.value.__cause__, ValidationError)

    def test_async_invoke_wraps_branch_failure_as_execution_error(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[RaisingBranch(), make_branch("b")],
            router=make_router(0),
        )

        with pytest.raises(ExecutionError, match="_async_run failed"):
            asyncio.run(flow.async_invoke({"value": 5}))


class TestRoutingFlowSerialization:
    def test_to_dict_includes_router_and_branches_list(self) -> None:
        flow = make_list_routing_flow(1)

        result = flow.invoke({"value": 5})
        data = flow.to_dict()

        assert data["router"] == flow.router.to_dict()
        assert isinstance(data["branches"], list)
        assert len(data["branches"]) == 3
        assert data["checkpoint_count"] == 1
        assert data["runs"] == [result.run_id]

    def test_to_dict_includes_branches_dict_for_dict_branches(self) -> None:
        flow = make_dict_routing_flow("right")

        result = flow.invoke({"value": 5})
        data = flow.to_dict()

        assert isinstance(data["branches"], dict)
        assert set(data["branches"].keys()) == {"left", "right"}
        assert data["checkpoint_count"] == 1
        assert data["runs"] == [result.run_id]
