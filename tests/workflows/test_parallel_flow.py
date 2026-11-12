from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from atomic_agentic.exceptions import ExecutionError, WorkflowError
from atomic_agentic.core.Invokable import StructuredInvokable
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results.workflows import ParallelFlowResult
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.base import Workflow
from atomic_agentic.workflows.parallel import ParallelFlow


def make_value_param(*, description: str | None = None) -> ParamSpec:
    return ParamSpec(
        name="value",
        index=0,
        kind=ParamSpec.POSITIONAL_OR_KEYWORD,
        type="int",
        description=description,
    )


class EchoWorkflow(Workflow):
    """_run(inputs) -> ({**inputs, "tag": self._tag}, {}) — branches are
    distinguished by their configured tag."""

    def __init__(
        self,
        *,
        tag: str,
        name: str | None = None,
        namespace: str = "tests",
        description: str = "Echo workflow.",
        param_description: str | None = None,
    ) -> None:
        super().__init__(
            name=name if name is not None else f"echo_workflow_{tag}",
            namespace=namespace,
            description=description,
            parameters=[make_value_param(description=param_description)],
            return_type="dict[str, Any]",
        )
        self._tag = tag

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {**inputs, "tag": self._tag}, {}


def make_branch(tag: str, **kwargs: Any) -> EchoWorkflow:
    return EchoWorkflow(tag=tag, **kwargs)


def make_three_branch_flow(**kwargs: Any) -> ParallelFlow:
    return ParallelFlow(
        name="parallel_flow",
        namespace="tests",
        description="Parallel test flow.",
        branches=[make_branch("a"), make_branch("b"), make_branch("c")],
        **kwargs,
    )


class TypedParamWorkflow(Workflow):
    """A single-param branch whose param type is configurable, for collision tests."""

    def __init__(self, *, param_type: str, tag: str) -> None:
        super().__init__(
            name=f"typed_{tag}",
            namespace="tests",
            description="Typed-param branch.",
            parameters=[
                ParamSpec(name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type=param_type)
            ],
            return_type="dict[str, Any]",
        )
        self._tag = tag

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"tag": self._tag}, {}


class VariadicWorkflow(Workflow):
    """A single-param branch declaring an independently-named *args param."""

    def __init__(self, *, param_name: str, tag: str) -> None:
        super().__init__(
            name=f"variadic_{tag}",
            namespace="tests",
            description="Variadic-param branch.",
            parameters=[
                ParamSpec(
                    name=param_name,
                    index=0,
                    kind=ParamSpec.VAR_POSITIONAL,
                    type="int",
                )
            ],
            return_type="dict[str, Any]",
        )
        self._tag = tag

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {"tag": self._tag}, {}


class TestParallelFlowConstruction:
    def test_branches_must_be_non_empty_list_or_tuple(self) -> None:
        with pytest.raises(ValueError, match="branches must not be empty"):
            ParallelFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                branches=[],
            )

        with pytest.raises(TypeError, match="branches must be"):
            ParallelFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                branches="not-a-list",  # type: ignore[arg-type]
            )

    def test_branches_stored_exactly_as_configured_no_wrapping(self) -> None:
        b0, b1 = make_branch("a"), make_branch("b")

        flow = ParallelFlow(
            name="parallel_flow",
            namespace="tests",
            description="Parallel test flow.",
            branches=[b0, b1],
        )

        assert flow.branches == (b0, b1)
        assert flow.branches[0] is b0
        assert flow.branches[1] is b1

    def test_branch_type_validated_at_construction(self) -> None:
        with pytest.raises(TypeError, match="position 1"):
            ParallelFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                branches=[make_branch("a"), object()],  # type: ignore[list-item]
            )

    def test_parameters_derived_by_folding_branch_parameters(self) -> None:
        b0 = make_branch("a")
        b1 = make_branch("b")

        flow = ParallelFlow(
            name="parallel_flow",
            namespace="tests",
            description="Parallel test flow.",
            branches=[b0, b1],
        )

        assert list(flow.parameters) == list(b0.parameters)

    def test_colliding_incompatible_parameter_raises_workflow_error(self) -> None:
        with pytest.raises(WorkflowError, match="no compatible reconciliation"):
            ParallelFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                branches=[
                    TypedParamWorkflow(param_type="int", tag="a"),
                    TypedParamWorkflow(param_type="str", tag="b"),
                ],
            )

    def test_independently_named_variadic_conflict_raises_workflow_error(self) -> None:
        # Caught by the shared parameter-order validator (SchemaError, a
        # WorkflowError subclass) -- two independently-named same-kind
        # variadics survive N-way reconciliation as two distinct names, then
        # collide structurally on re-validation in insert_by_category.
        with pytest.raises(WorkflowError, match="Only one VAR_POSITIONAL parameter is allowed"):
            ParallelFlow(
                name="bad_flow",
                namespace="tests",
                description="Bad flow.",
                branches=[
                    VariadicWorkflow(param_name="args_a", tag="a"),
                    VariadicWorkflow(param_name="args_b", tag="b"),
                ],
            )

    def test_compatible_but_not_identical_overlap_warns_and_keeps_earlier(self) -> None:
        earlier = make_branch("a", param_description="earlier description")
        later = make_branch("b", param_description="later description")

        with pytest.warns(UserWarning, match="compatible across"):
            flow = ParallelFlow(
                name="parallel_flow",
                namespace="tests",
                description="Parallel test flow.",
                branches=[earlier, later],
            )

        assert flow.parameters[0].description == "earlier description"

    def test_type_witness_set_broader_than_any_single_branch_declaration(self) -> None:
        # Old left-to-right fold: branch 0's own "dict[str, int]"
        # declaration would win and be kept verbatim once branch 1's bare
        # "dict" is judged compatible-and-discarded, silently masking
        # branch 1's own bridging contribution. True N-way reconciliation
        # instead computes the full compatible-type witness set across all
        # three branches at once -- broader than any single branch's own
        # declared type.
        with pytest.warns(UserWarning, match="compatible across"):
            flow = ParallelFlow(
                name="parallel_flow",
                namespace="tests",
                description="Parallel test flow.",
                branches=[
                    TypedParamWorkflow(param_type="dict[str, int]", tag="a"),
                    TypedParamWorkflow(param_type="dict", tag="b"),
                    TypedParamWorkflow(param_type="dict[str, int]", tag="c"),
                ],
            )

        shared_param = next(p for p in flow.parameters if p.name == "x")
        assert shared_param.type == ("dict", "dict[str, int]")

    def test_result_mode_defaults_to_list(self) -> None:
        flow = make_three_branch_flow()

        assert flow.result_mode == ParallelFlow.LIST

    def test_result_mode_rejects_unknown_value(self) -> None:
        with pytest.raises(ValueError, match="result_mode"):
            make_three_branch_flow(result_mode="weird")

    def test_scalar_allows_at_most_one_selected_branch(self) -> None:
        with pytest.raises(ValueError, match="at most one selected branch"):
            make_three_branch_flow(result_mode=ParallelFlow.SCALAR, selected_branches=[0, 1])

    def test_scalar_allows_zero_selected_branches(self) -> None:
        flow = make_three_branch_flow(result_mode=ParallelFlow.SCALAR, selected_branches=[])

        assert flow.selected_indices == ()

    def test_selected_branches_none_and_empty_both_mean_nothing_selected(self) -> None:
        flow_none = make_three_branch_flow(selected_branches=None)
        flow_empty = make_three_branch_flow(selected_branches=[])

        assert flow_none.selected_indices == ()
        assert flow_empty.selected_indices == ()

    def test_selected_branches_rejects_non_int_items(self) -> None:
        with pytest.raises(TypeError, match="selected_branches items must be int"):
            make_three_branch_flow(selected_branches=["0"])  # type: ignore[list-item]

        with pytest.raises(TypeError, match="selected_branches items must be int"):
            make_three_branch_flow(selected_branches=[True])

    def test_selected_branches_rejects_negative_and_out_of_range(self) -> None:
        """No negative-index wraparound support."""
        with pytest.raises(IndexError, match="out of range"):
            make_three_branch_flow(selected_branches=[-1])

        with pytest.raises(IndexError, match="out of range"):
            make_three_branch_flow(selected_branches=[5])

    def test_selected_branches_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="same branch twice"):
            make_three_branch_flow(selected_branches=[0, 0])

    def test_dict_requires_result_keys_matching_selection_length(self) -> None:
        with pytest.raises(ValueError, match=r"result_keys must match"):
            make_three_branch_flow(
                result_mode=ParallelFlow.DICT,
                selected_branches=[0, 1],
                result_keys=["a"],
            )

    def test_dict_result_keys_must_be_unique_valid_identifiers(self) -> None:
        with pytest.raises(ValueError, match="must be unique"):
            make_three_branch_flow(
                result_mode=ParallelFlow.DICT,
                selected_branches=[0, 1, 2],
                result_keys=["a", "a", "b"],
            )

        with pytest.raises(ValueError, match="valid parameter-style name"):
            make_three_branch_flow(
                result_mode=ParallelFlow.DICT,
                selected_branches=[0, 1, 2],
                result_keys=["1bad", "ok", "ok2"],
            )

    def test_result_keys_forbidden_for_non_dict_modes(self) -> None:
        with pytest.raises(ValueError, match="result_keys must be None"):
            make_three_branch_flow(result_mode=ParallelFlow.LIST, result_keys=["a", "b", "c"])

    def test_include_trace_defaults_to_true_and_is_mutable(self) -> None:
        flow = make_three_branch_flow()

        assert flow.include_trace is True
        flow.include_trace = False
        assert flow.include_trace is False


def make_structured_branch(name: str, output_schema: list[str]) -> StructuredInvokable:
    def return_value(value: Any) -> Any:
        """Return the provided value."""
        return value

    tool = Tool(
        function=return_value,
        name=name,
        namespace="tests",
        description=f"Return the provided value ({name}).",
    )
    return StructuredInvokable(
        component=tool,
        output_schema=output_schema,
        name=f"structured_{name}",
        description=f"Structured {name}.",
    )


class TestParallelFlowExtraDescription:
    def test_scalar_output_surfaces_selected_branch_extra_description(self) -> None:
        structured = make_structured_branch("value_branch", ["value"])

        flow = ParallelFlow(
            name="parallel_flow",
            namespace="tests",
            description="Parallel test flow.",
            branches=[structured, make_branch("b")],
            result_mode=ParallelFlow.SCALAR,
            selected_branches=[0],
        )

        assert flow._extra_description() == "Output schema: [value]"
        assert flow.description == "Parallel test flow.\nOutput schema: [value]"

    def test_scalar_with_no_selection_states_none(self) -> None:
        flow = make_three_branch_flow(result_mode=ParallelFlow.SCALAR, selected_branches=[])

        assert flow._extra_description() == "Returns None (no branch selected)."

    def test_dict_output_lists_keys_and_branch_return_types(self) -> None:
        flow = make_three_branch_flow(
            result_mode=ParallelFlow.DICT,
            selected_branches=[0, 1, 2],
            result_keys=["first", "second", "third"],
        )

        assert flow._extra_description() == (
            "Returns a dict of 3 keys: first (dict[str, Any]), "
            "second (dict[str, Any]), third (dict[str, Any])"
        )

    def test_list_output_states_branch_count_and_omits_branch_content(self) -> None:
        structured = make_structured_branch("value_branch", ["value"])

        flow = ParallelFlow(
            name="parallel_flow",
            namespace="tests",
            description="Parallel test flow.",
            branches=[structured, make_branch("b"), make_branch("c")],
            selected_branches=[0, 1, 2],
        )

        assert flow._extra_description() == "Returns a list of 3 branch outputs."


class TestParallelFlowSyncInvoke:
    def test_all_branches_invoked_with_same_broadcast_inputs(self) -> None:
        flow = make_three_branch_flow(selected_branches=[0, 1, 2])

        result = flow.invoke({"value": 1})

        assert result.trace is not None
        assert [r.result for r in result.trace] == [
            {"value": 1, "tag": "a"},
            {"value": 1, "tag": "b"},
            {"value": 1, "tag": "c"},
        ]

    def test_all_branches_always_execute_regardless_of_selection(self) -> None:
        flow = make_three_branch_flow(result_mode=ParallelFlow.SCALAR, selected_branches=[])

        result = flow.invoke({"value": 1})

        assert result.result is None
        assert len(result.trace) == 3

    def test_result_mode_scalar_returns_single_branch_payload(self) -> None:
        flow = make_three_branch_flow(result_mode=ParallelFlow.SCALAR, selected_branches=[1])

        result = flow.invoke({"value": 1})

        assert result.result == {"value": 1, "tag": "b"}

    def test_result_mode_list_returns_ordered_list_of_payloads(self) -> None:
        flow = make_three_branch_flow(selected_branches=[0, 1, 2])

        result = flow.invoke({"value": 1})

        assert result.result == [
            {"value": 1, "tag": "a"},
            {"value": 1, "tag": "b"},
            {"value": 1, "tag": "c"},
        ]

    def test_result_mode_tuple_returns_ordered_tuple_of_payloads(self) -> None:
        flow = make_three_branch_flow(result_mode=ParallelFlow.TUPLE, selected_branches=[0, 1, 2])

        result = flow.invoke({"value": 1})

        assert result.result == (
            {"value": 1, "tag": "a"},
            {"value": 1, "tag": "b"},
            {"value": 1, "tag": "c"},
        )

    def test_result_mode_set_returns_set_of_payloads(self) -> None:
        # Use payloads reducible to hashable values for the SET case.
        class TaggedScalar(EchoWorkflow):
            def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
                return self._tag, {}

        flow = ParallelFlow(
            name="parallel_flow",
            namespace="tests",
            description="Parallel test flow.",
            branches=[TaggedScalar(tag="a"), TaggedScalar(tag="b"), TaggedScalar(tag="c")],
            result_mode=ParallelFlow.SET,
            selected_branches=[0, 1, 2],
        )

        result = flow.invoke({"value": 1})

        assert result.result == {"a", "b", "c"}

    def test_result_mode_dict_returns_named_mapping_of_payloads(self) -> None:
        flow = make_three_branch_flow(
            result_mode=ParallelFlow.DICT,
            selected_branches=[0, 2],
            result_keys=["first", "third"],
        )

        result = flow.invoke({"value": 1})

        assert result.result == {
            "first": {"value": 1, "tag": "a"},
            "third": {"value": 1, "tag": "c"},
        }

    def test_parallel_flow_result_fields(self) -> None:
        flow = make_three_branch_flow(selected_branches=[2, 0])

        result = flow.invoke({"value": 1})

        assert isinstance(result, ParallelFlowResult)
        assert len(result.trace) == 3
        assert result.result_mode == ParallelFlow.LIST
        assert result.selected_indices == (2, 0)
        assert result.result_keys == (2, 0)

    def test_result_keys_mirrors_selected_indices_for_non_dict_modes(self) -> None:
        flow = make_three_branch_flow(selected_branches=[1, 2])

        assert flow.result_keys == flow.selected_indices == (1, 2)


class TestParallelFlowAsyncInvoke:
    def test_async_invoke_list_output(self) -> None:
        flow = make_three_branch_flow(selected_branches=[0, 1, 2])

        result = asyncio.run(flow.async_invoke({"value": 1}))

        assert isinstance(result, ParallelFlowResult)
        assert result.result == [
            {"value": 1, "tag": "a"},
            {"value": 1, "tag": "b"},
            {"value": 1, "tag": "c"},
        ]
        assert len(result.trace) == 3

    def test_async_invoke_dict_output(self) -> None:
        flow = make_three_branch_flow(
            result_mode=ParallelFlow.DICT,
            selected_branches=[0, 2],
            result_keys=["first", "third"],
        )

        result = asyncio.run(flow.async_invoke({"value": 1}))

        assert result.result == {
            "first": {"value": 1, "tag": "a"},
            "third": {"value": 1, "tag": "c"},
        }


class TestParallelFlowValidationAndErrors:
    def test_branch_invoke_failure_wrapped_as_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flow = make_three_branch_flow()

        async def raise_async_invoke(inputs: Mapping[str, Any]) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(flow.branches[1], "async_invoke", raise_async_invoke)

        with pytest.raises(ExecutionError, match="_run failed") as exc_info:
            flow.invoke({"value": 1})

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "branch 1" in str(exc_info.value.__cause__)
        assert "failed during invoke" in str(exc_info.value.__cause__)

    def test_async_branch_invoke_failure_wrapped_as_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flow = make_three_branch_flow()

        async def raise_async_invoke(inputs: Mapping[str, Any]) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(flow.branches[1], "async_invoke", raise_async_invoke)

        with pytest.raises(ExecutionError, match="_async_run failed") as exc_info:
            asyncio.run(flow.async_invoke({"value": 1}))

        assert isinstance(exc_info.value.__cause__, RuntimeError)


class TestParallelFlowSerialization:
    def test_to_dict_includes_branches_result_mode_and_indices(self) -> None:
        flow = make_three_branch_flow(
            result_mode=ParallelFlow.DICT,
            selected_branches=[0, 2],
            result_keys=["first", "third"],
        )

        flow.invoke({"value": 1})
        data = flow.to_dict()

        assert data["branch_count"] == 3
        assert data["result_mode"] == flow.result_mode
        assert data["selected_indices"] == [0, 2]
        assert data["result_keys"] == ["first", "third"]
        assert "checkpoints" not in data
        assert "output_type" not in data
