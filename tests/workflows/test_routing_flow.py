from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pytest

from atomic_agentic.exceptions import ExecutionError, ValidationError, WorkflowError
from atomic_agentic.core.Invokable import StructuredInvokable
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results.workflows import RoutingFlowResult
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.base import Workflow
from atomic_agentic.workflows.routing import RoutingFlow


def make_value_param(*, description: str | None = None) -> ParamSpec:
    return ParamSpec(
        name="value",
        index=0,
        kind=ParamSpec.POSITIONAL_OR_KEYWORD,
        type="int",
        description=description,
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
        param_description: str | None = None,
    ) -> None:
        super().__init__(
            name=name or f"echo_{tag}",
            namespace=namespace,
            description=description or f"Echo workflow {tag}.",
            parameters=[make_value_param(description=param_description)],
            return_type=return_type,
        )
        self._tag = tag

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {**inputs, "branch": self._tag}, {}


def make_branch(tag: str, **kwargs: Any) -> EchoWorkflow:
    return EchoWorkflow(tag, **kwargs)


class OtherParamWorkflow(Workflow):
    """A branch declaring a param the router doesn't share, no default.

    Declares only ``other`` when ``with_overlap=False`` (zero overlap with a
    router whose sole param is ``value``); when ``with_overlap=True`` it also
    declares ``value``, satisfying PARTIAL's "at least one overlapping
    parameter" requirement while still introducing the new ``other`` param.
    """

    def __init__(self, *, tag: str = "other", with_overlap: bool = False) -> None:
        parameters = [
            ParamSpec(name="other", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="str"),
        ]
        if with_overlap:
            parameters = [make_value_param(), *[
                ParamSpec(name=p.name, index=p.index + 1, kind=p.kind, type=p.type)
                for p in parameters
            ]]
        super().__init__(
            name=f"other_param_{tag}",
            namespace="tests",
            description="Branch with a non-overlapping required param.",
            parameters=parameters,
            return_type="dict[str, Any]",
        )
        self._tag = tag

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"branch": self._tag}, {}


class RouterWorkflow(Workflow):
    """_run(inputs) -> (selector, {}) — router whose raw result IS the selector."""

    def __init__(self, selector: Any, *, raise_error: bool = False) -> None:
        super().__init__(
            name="router_workflow",
            namespace="tests",
            description="Router workflow returning a constant selector.",
            parameters=[make_value_param()],
            return_type="Any",
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

    def test_list_branches_stored_exactly_as_configured(self) -> None:
        b0, b1 = make_branch("a"), make_branch("b")
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[b0, b1],
            router=make_router(0),
        )

        assert flow.branches == (b0, b1)
        assert flow.branches[0] is b0

    def test_dict_branches_stored_exactly_as_configured(self) -> None:
        b0 = make_branch("a")
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches={"a": b0, "b": make_branch("b")},
            router=make_router("a"),
        )

        assert isinstance(flow.branches, MappingProxyType)
        assert flow.branches["a"] is b0

    def test_router_stored_exactly_as_configured(self) -> None:
        router = make_router(0)
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[make_branch("a"), make_branch("b")],
            router=router,
        )
        assert flow.router is router

    def test_constructor_rejects_non_atomic_router(self) -> None:
        with pytest.raises(TypeError, match="router must be AtomicInvokable"):
            RoutingFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                branches=[make_branch("a"), make_branch("b")],
                router=object(),  # type: ignore[arg-type]
            )

    def test_constructor_rejects_non_atomic_branch(self) -> None:
        with pytest.raises(TypeError, match="branches must be AtomicInvokable"):
            RoutingFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                branches=[object()],  # type: ignore[list-item]
                router=make_router(0),
            )

    def test_schema_mode_defaults_to_strict(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[make_branch("a"), make_branch("b")],
            router=make_router(0),
        )
        assert flow.schema_mode == RoutingFlow.STRICT

    def test_schema_mode_rejects_unknown_value(self) -> None:
        with pytest.raises(ValueError, match="schema_mode"):
            RoutingFlow(
                name="routing_flow",
                namespace="tests",
                description="Routing test flow.",
                branches=[make_branch("a"), make_branch("b")],
                router=make_router(0),
                schema_mode="weird",
            )

    def test_strict_parameters_equal_router_parameters(self) -> None:
        router = make_router(0)
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[make_branch("a"), make_branch("b")],
            router=router,
        )

        assert list(flow.parameters) == list(router.parameters)

    def test_strict_rejects_branch_param_not_in_router_schema(self) -> None:
        with pytest.raises(WorkflowError, match="not present in the router"):
            RoutingFlow(
                name="routing_flow",
                namespace="tests",
                description="Routing test flow.",
                branches=[OtherParamWorkflow(), make_branch("b")],
                router=make_router(0),
                schema_mode=RoutingFlow.STRICT,
            )

    def test_strict_compatible_but_not_identical_overlap_warns(self) -> None:
        with pytest.warns(UserWarning, match="compatible with the router's declaration"):
            RoutingFlow(
                name="routing_flow",
                namespace="tests",
                description="Routing test flow.",
                branches=[make_branch("a", param_description="different")],
                router=make_router(0),
            )

    def test_partial_requires_overlap_with_router(self) -> None:
        with pytest.raises(WorkflowError, match="shares no parameters with the router"):
            RoutingFlow(
                name="routing_flow",
                namespace="tests",
                description="Routing test flow.",
                branches=[OtherParamWorkflow(with_overlap=False)],
                router=make_router(0),
                schema_mode=RoutingFlow.PARTIAL,
            )

    def test_partial_folds_new_branch_param_as_optional(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[make_branch("a"), OtherParamWorkflow(with_overlap=True)],
            router=make_router(0),
            schema_mode=RoutingFlow.PARTIAL,
        )

        by_name = {p.name: p for p in flow.parameters}
        assert "other" in by_name
        assert by_name["other"].default is None

    def test_open_allows_branch_with_zero_overlap(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[OtherParamWorkflow()],
            router=make_router(0),
            schema_mode=RoutingFlow.OPEN,
        )

        by_name = {p.name: p for p in flow.parameters}
        assert "value" in by_name
        assert "other" in by_name
        assert by_name["other"].default is None

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

    def test_include_trace_defaults_to_true_and_is_mutable(self) -> None:
        flow = make_list_routing_flow(0)

        assert flow.include_trace is True
        flow.include_trace = False
        assert flow.include_trace is False


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
            schema_mode=RoutingFlow.OPEN,
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
            schema_mode=RoutingFlow.OPEN,
        )

        assert flow._extra_description() == "Selects 1 of 2 branches at runtime."

    def test_unanimous_empty_extra_states_branch_count_only(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[make_branch("a"), make_branch("b")],
            router=make_structured_router(0),
            schema_mode=RoutingFlow.OPEN,
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
        assert result.selected_branch == 1
        assert result.trace is not None
        assert len(result.trace) == 2
        assert result.trace[0].result == 1  # router's own raw result
        assert result.trace[1].result == {"value": 5, "branch": "b"}

    def test_trace_none_when_include_trace_disabled(self) -> None:
        flow = make_list_routing_flow(1)
        flow.include_trace = False

        result = flow.invoke({"value": 5})

        assert result.trace is None
        assert result.selected_branch == 1


class TestRoutingFlowSyncInvokeDictBranches:
    def test_invoke_routes_to_dict_selected_branch(self) -> None:
        flow = make_dict_routing_flow("right")

        result = flow.invoke({"value": 5})

        assert result.result == {"value": 5, "branch": "right"}
        assert result.selected_branch == "right"

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
    def test_to_dict_includes_router_branches_list_and_schema_mode(self) -> None:
        flow = make_list_routing_flow(1)

        flow.invoke({"value": 5})
        data = flow.to_dict()

        assert data["router"] == flow.router.to_dict()
        assert isinstance(data["branches"], list)
        assert len(data["branches"]) == 3
        assert data["schema_mode"] == RoutingFlow.STRICT
        assert "checkpoints" not in data

    def test_to_dict_includes_branches_dict_for_dict_branches(self) -> None:
        flow = make_dict_routing_flow("right")

        flow.invoke({"value": 5})
        data = flow.to_dict()

        assert isinstance(data["branches"], dict)
        assert set(data["branches"].keys()) == {"left", "right"}
