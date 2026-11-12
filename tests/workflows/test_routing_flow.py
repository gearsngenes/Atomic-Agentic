from __future__ import annotations

import asyncio
import warnings
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import pytest

from atomic_agentic.constants.core import NO_VAL
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
    declares ``value``, letting a branch share a router param while also
    introducing a new one the router doesn't have.
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


class ParamsWorkflow(Workflow):
    """A router or branch with an arbitrary caller-specified parameter list.

    ``is_router=True`` makes ``_run`` return a constant selector (``0``)
    instead of a dict payload, so this can stand in as either a router or a
    branch depending on the reconciliation scenario under test.
    """

    def __init__(self, *, tag: str, parameters: list[ParamSpec], is_router: bool = False) -> None:
        super().__init__(
            name=f"params_{tag}",
            namespace="tests",
            description=f"Workflow {tag} with a configurable parameter list.",
            parameters=parameters,
            return_type="Any" if is_router else "dict[str, Any]",
        )
        self._tag = tag
        self._is_router = is_router

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        if self._is_router:
            return 0, {}
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

    def test_identical_branch_declarations_leave_router_parameters_unchanged(self) -> None:
        router = make_router(0)
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[make_branch("a"), make_branch("b")],
            router=router,
        )

        assert list(flow.parameters) == list(router.parameters)

    def test_router_shared_compatible_but_not_identical_overlap_warns(self) -> None:
        with pytest.warns(UserWarning, match="compatible with the router's declaration"):
            RoutingFlow(
                name="routing_flow",
                namespace="tests",
                description="Routing test flow.",
                branches=[make_branch("a", param_description="different")],
                router=make_router(0),
            )

    def test_single_branch_non_router_param_defaults_to_none(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[make_branch("a"), OtherParamWorkflow(with_overlap=True)],
            router=make_router(0),
        )

        by_name = {p.name: p for p in flow.parameters}
        assert "other" in by_name
        assert by_name["other"].default is None
        assert "None" in by_name["other"].type

    def test_branch_with_zero_overlap_is_allowed(self) -> None:
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[OtherParamWorkflow()],
            router=make_router(0),
        )

        by_name = {p.name: p for p in flow.parameters}
        assert "value" in by_name
        assert "other" in by_name
        assert by_name["other"].default is None
        assert "None" in by_name["other"].type

    def test_router_shared_type_widens_via_witness(self) -> None:
        router = ParamsWorkflow(
            tag="router",
            parameters=[ParamSpec(name="value", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="dict[str, int]")],
            is_router=True,
        )
        branch = ParamsWorkflow(
            tag="a",
            parameters=[ParamSpec(name="value", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="dict")],
        )
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[branch],
            router=router,
        )

        by_name = {p.name: p for p in flow.parameters}
        assert by_name["value"].type == ("dict", "dict[str, int]")

    def test_router_shared_incompatible_type_raises(self) -> None:
        router = ParamsWorkflow(
            tag="router",
            parameters=[ParamSpec(name="value", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int")],
            is_router=True,
        )
        branch = ParamsWorkflow(
            tag="a",
            parameters=[ParamSpec(name="value", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="str")],
        )
        with pytest.raises(WorkflowError, match="incompatible with router"):
            RoutingFlow(
                name="routing_flow",
                namespace="tests",
                description="Routing test flow.",
                branches=[branch],
                router=router,
            )

    def test_router_shared_kind_conflict_raises(self) -> None:
        router = ParamsWorkflow(
            tag="router",
            parameters=[ParamSpec(name="value", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="Any")],
            is_router=True,
        )
        branch = ParamsWorkflow(
            tag="a",
            parameters=[ParamSpec(name="value", index=0, kind=ParamSpec.VAR_KEYWORD, type="Any")],
        )
        with pytest.raises(WorkflowError, match="incompatible with router"):
            RoutingFlow(
                name="routing_flow",
                namespace="tests",
                description="Routing test flow.",
                branches=[branch],
                router=router,
            )

    def test_non_router_multi_branch_agreeing_default_used_without_none(self) -> None:
        router = make_router(0)
        branch1 = ParamsWorkflow(
            tag="p1",
            parameters=[ParamSpec(
                name="extra", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                type="dict", default=5, description="first",
            )],
        )
        branch2 = ParamsWorkflow(
            tag="p2",
            parameters=[ParamSpec(
                name="extra", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                type="dict[str, int]", default=5, description="second",
            )],
        )
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[branch1, branch2],
            router=router,
        )

        extra = {p.name: p for p in flow.parameters}["extra"]
        assert extra.type == ("dict", "dict[str, int]")
        assert extra.description == "first"
        assert extra.default == 5

    def test_non_router_multi_branch_disagreeing_default_falls_back_to_none(self) -> None:
        router = make_router(0)
        branch1 = ParamsWorkflow(
            tag="p1",
            parameters=[ParamSpec(
                name="extra", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="dict", default=5,
            )],
        )
        branch2 = ParamsWorkflow(
            tag="p2",
            parameters=[ParamSpec(
                name="extra", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="dict", default=6,
            )],
        )
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[branch1, branch2],
            router=router,
        )

        extra = {p.name: p for p in flow.parameters}["extra"]
        assert extra.default is None
        assert "None" in extra.type

    def test_non_router_multi_branch_kind_conflict_raises(self) -> None:
        router = make_router(0)
        branch1 = ParamsWorkflow(
            tag="p1",
            parameters=[ParamSpec(name="extra2", index=0, kind=ParamSpec.VAR_POSITIONAL, type="Any")],
        )
        branch2 = ParamsWorkflow(
            tag="p2",
            parameters=[ParamSpec(name="extra2", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="Any")],
        )
        with pytest.raises(WorkflowError, match="incompatible kind"):
            RoutingFlow(
                name="routing_flow",
                namespace="tests",
                description="Routing test flow.",
                branches=[branch1, branch2],
                router=router,
            )

    def test_non_router_variadic_keeps_no_val_default_and_no_none_type(self) -> None:
        router = make_router(0)
        branch = ParamsWorkflow(
            tag="p1",
            parameters=[ParamSpec(name="extra_kwargs", index=0, kind=ParamSpec.VAR_KEYWORD, type="Any")],
        )
        flow = RoutingFlow(
            name="routing_flow",
            namespace="tests",
            description="Routing test flow.",
            branches=[branch],
            router=router,
        )

        extra_kwargs = {p.name: p for p in flow.parameters}["extra_kwargs"]
        assert extra_kwargs.default is NO_VAL
        assert extra_kwargs.type == ("Any",)
        assert extra_kwargs.kind == ParamSpec.VAR_KEYWORD

    def test_grouped_warning_covers_multiple_names_in_one_call(self) -> None:
        router = ParamsWorkflow(
            tag="router",
            parameters=[
                ParamSpec(name="value", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int"),
                ParamSpec(name="shared", index=1, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int"),
            ],
            is_router=True,
        )
        branch = ParamsWorkflow(
            tag="a",
            parameters=[
                ParamSpec(name="value", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int", description="value desc"),
                ParamSpec(name="shared", index=1, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int", description="shared desc"),
            ],
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RoutingFlow(
                name="routing_flow",
                namespace="tests",
                description="Routing test flow.",
                branches=[branch],
                router=router,
            )

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        message = str(user_warnings[0].message)
        assert "value" in message
        assert "shared" in message

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
    def test_to_dict_includes_router_and_branches_list(self) -> None:
        flow = make_list_routing_flow(1)

        flow.invoke({"value": 5})
        data = flow.to_dict()

        assert data["router"] == flow.router.to_dict()
        assert isinstance(data["branches"], list)
        assert len(data["branches"]) == 3
        assert "checkpoints" not in data

    def test_to_dict_includes_branches_dict_for_dict_branches(self) -> None:
        flow = make_dict_routing_flow("right")

        flow.invoke({"value": 5})
        data = flow.to_dict()

        assert isinstance(data["branches"], dict)
        assert set(data["branches"].keys()) == {"left", "right"}
