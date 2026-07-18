from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from atomic_agentic.exceptions import ExecutionError
from atomic_agentic.core.Invokable import StructuredInvokable
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results.workflows import ParallelFlowResult
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.base import Workflow
from atomic_agentic.workflows.basic import BasicFlow
from atomic_agentic.workflows.parallel import ParallelFlow


def make_value_param() -> ParamSpec:
    return ParamSpec(
        name="value",
        index=0,
        kind=ParamSpec.POSITIONAL_OR_KEYWORD,
        type="int",
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
        filter_extraneous_inputs: bool = True,
    ) -> None:
        super().__init__(
            name=name if name is not None else f"echo_workflow_{tag}",
            namespace=namespace,
            description=description,
            parameters=[make_value_param()],
            return_type="dict[str, Any]",
            filter_extraneous_inputs=filter_extraneous_inputs,
        )
        self._tag = tag

    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        return {**inputs, "tag": self._tag}, {}


def make_branch(tag: str) -> EchoWorkflow:
    return EchoWorkflow(tag=tag)


def make_three_branch_flow(**kwargs: Any) -> ParallelFlow:
    return ParallelFlow(
        name="parallel_flow",
        namespace="tests",
        description="Parallel test flow.",
        branches=[make_branch("a"), make_branch("b"), make_branch("c")],
        **kwargs,
    )


class TestParallelFlowConstruction:
    def test_branches_must_be_non_empty_list(self) -> None:
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

    def test_non_workflow_branches_normalized_to_basic_flow(self) -> None:
        def return_value(value: Any) -> Any:
            """Return the provided value."""
            return value

        tool = Tool(
            function=return_value,
            name="return_value",
            namespace="tests",
            description="Return the provided value.",
        )
        component = StructuredInvokable(
            component=tool,
            output_schema=["value"],
            name="structured_value",
            description="Structured value.",
        )
        branch_b = make_branch("b")

        flow = ParallelFlow(
            name="parallel_flow",
            namespace="tests",
            description="Parallel test flow.",
            branches=[component, branch_b],
        )

        assert isinstance(flow.branches[0], BasicFlow)
        assert isinstance(flow.branches[1], EchoWorkflow)

    def test_parameters_default_to_first_branch_parameters(self) -> None:
        b0 = make_branch("a")
        b1 = make_branch("b")

        flow = ParallelFlow(
            name="parallel_flow",
            namespace="tests",
            description="Parallel test flow.",
            branches=[b0, b1],
            parameters=None,
        )

        assert flow.parameters == b0.parameters

    def test_output_type_defaults_to_list(self) -> None:
        flow = make_three_branch_flow()

        assert flow.output_type == ParallelFlow.LIST

    def test_output_type_rejects_unknown_value(self) -> None:
        with pytest.raises(ValueError, match="output_type"):
            make_three_branch_flow(output_type="weird")

    def test_scalar_requires_exactly_one_output_index(self) -> None:
        with pytest.raises(ValueError, match="exactly one output index"):
            make_three_branch_flow(output_type=ParallelFlow.SCALAR, output_indices=[0, 1])

        with pytest.raises(ValueError, match="exactly one output index"):
            make_three_branch_flow(output_type=ParallelFlow.SCALAR, output_indices=None)

    def test_scalar_forbids_output_names(self) -> None:
        with pytest.raises(ValueError, match="output_names"):
            make_three_branch_flow(
                output_type=ParallelFlow.SCALAR,
                output_indices=[0],
                output_names=["x"],
            )

    def test_dict_requires_output_names(self) -> None:
        with pytest.raises(ValueError, match="output_names"):
            make_three_branch_flow(output_type=ParallelFlow.DICT, output_names=None)

    def test_dict_output_names_must_match_selected_branch_count(self) -> None:
        with pytest.raises(ValueError, match=r"len\(output_names\)"):
            make_three_branch_flow(
                output_type=ParallelFlow.DICT,
                output_indices=[0, 1],
                output_names=["a"],
            )

    def test_dict_output_names_must_be_unique_identifiers(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            make_three_branch_flow(
                output_type=ParallelFlow.DICT,
                output_names=["a", "a", "b"],
            )

        with pytest.raises(ValueError, match="valid parameter-style name"):
            make_three_branch_flow(
                output_type=ParallelFlow.DICT,
                output_names=["1bad", "ok", "ok2"],
            )

    def test_list_tuple_set_forbid_output_names(self) -> None:
        with pytest.raises(ValueError, match="output_names"):
            make_three_branch_flow(output_type=ParallelFlow.LIST, output_names=["a", "b", "c"])

    def test_output_indices_and_output_range_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="either output_indices or output_range"):
            make_three_branch_flow(output_indices=[0, 1], output_range=(0, 2))

    def test_output_indices_supports_negative_resolution(self) -> None:
        flow = make_three_branch_flow(output_indices=[-1, 0])

        assert flow.output_indices == [2, 0]

    def test_output_indices_rejects_duplicates_and_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            make_three_branch_flow(output_indices=[0, 0])

        with pytest.raises(IndexError, match="out of range"):
            make_three_branch_flow(output_indices=[5])


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
            output_type=ParallelFlow.SCALAR,
            output_indices=[0],
        )

        assert flow._extra_description() == "Output schema: [value]"
        assert flow.description == "Parallel test flow.\nOutput schema: [value]"

    def test_dict_output_lists_keys_and_branch_return_types(self) -> None:
        flow = make_three_branch_flow(
            output_type=ParallelFlow.DICT,
            output_names=["first", "second", "third"],
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
        )

        assert flow._extra_description() == "Returns a list of 3 branch outputs."


class TestParallelFlowSyncInvoke:
    def test_all_branches_invoked_with_same_broadcast_inputs(self) -> None:
        flow = make_three_branch_flow()

        flow.invoke({"value": 1})

        for branch in flow.branches:
            assert branch.checkpoints[-1].inputs == {"value": 1}

    def test_output_type_scalar_returns_single_branch_payload(self) -> None:
        flow = make_three_branch_flow(output_type=ParallelFlow.SCALAR, output_indices=[1])

        result = flow.invoke({"value": 1})

        assert result.result == {"value": 1, "tag": "b"}

    def test_output_type_list_returns_ordered_list_of_payloads(self) -> None:
        flow = make_three_branch_flow()

        result = flow.invoke({"value": 1})

        assert result.result == [
            {"value": 1, "tag": "a"},
            {"value": 1, "tag": "b"},
            {"value": 1, "tag": "c"},
        ]

    def test_output_type_tuple_returns_ordered_tuple_of_payloads(self) -> None:
        flow = make_three_branch_flow(output_type=ParallelFlow.TUPLE)

        result = flow.invoke({"value": 1})

        assert result.result == (
            {"value": 1, "tag": "a"},
            {"value": 1, "tag": "b"},
            {"value": 1, "tag": "c"},
        )

    def test_output_type_set_returns_set_of_payloads(self) -> None:
        # Use payloads reducible to hashable values for the SET case.
        class TaggedScalar(EchoWorkflow):
            def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
                return self._tag, {}

        flow = ParallelFlow(
            name="parallel_flow",
            namespace="tests",
            description="Parallel test flow.",
            branches=[TaggedScalar(tag="a"), TaggedScalar(tag="b"), TaggedScalar(tag="c")],
            output_type=ParallelFlow.SET,
        )

        result = flow.invoke({"value": 1})

        assert result.result == {"a", "b", "c"}

    def test_output_type_dict_returns_named_mapping_of_payloads(self) -> None:
        flow = make_three_branch_flow(
            output_type=ParallelFlow.DICT,
            output_indices=[0, 2],
            output_names=["first", "third"],
        )

        result = flow.invoke({"value": 1})

        assert result.result == {
            "first": {"value": 1, "tag": "a"},
            "third": {"value": 1, "tag": "c"},
        }

    def test_parallel_flow_result_fields(self) -> None:
        flow = make_three_branch_flow(output_indices=[-1, 0])

        result = flow.invoke({"value": 1})

        assert isinstance(result, ParallelFlowResult)
        assert len(result.branch_runs) == 3
        assert result.output_indices == (2, 0)
        assert result.run_id == flow.checkpoints[-1].result.run_id


class TestParallelFlowAsyncInvoke:
    def test_async_invoke_list_output(self) -> None:
        flow = make_three_branch_flow()

        result = asyncio.run(flow.async_invoke({"value": 1}))

        assert isinstance(result, ParallelFlowResult)
        assert result.result == [
            {"value": 1, "tag": "a"},
            {"value": 1, "tag": "b"},
            {"value": 1, "tag": "c"},
        ]
        assert len(result.branch_runs) == 3

    def test_async_invoke_dict_output(self) -> None:
        flow = make_three_branch_flow(
            output_type=ParallelFlow.DICT,
            output_indices=[0, 2],
            output_names=["first", "third"],
        )

        result = asyncio.run(flow.async_invoke({"value": 1}))

        assert result.result == {
            "first": {"value": 1, "tag": "a"},
            "third": {"value": 1, "tag": "c"},
        }


class TestParallelFlowRetrieval:
    def test_get_branch_results_returns_result_objects_in_branch_order(self) -> None:
        flow = make_three_branch_flow()

        result = flow.invoke({"value": 1})
        results = flow.get_branch_results(result.run_id)

        assert results is not None
        assert len(results) == 3
        assert [r.run_id for r in results] == list(result.branch_runs)

    def test_get_branch_results_returns_none_for_unknown_run_id(self) -> None:
        flow = make_three_branch_flow()

        assert flow.get_branch_results("nope") is None

    def test_get_branch_result_returns_single_result_object(self) -> None:
        flow = make_three_branch_flow()

        result = flow.invoke({"value": 1})

        assert flow.get_branch_result(result.run_id, 1).run_id == result.branch_runs[1]

    def test_get_branch_result_rejects_non_int_index(self) -> None:
        flow = make_three_branch_flow()

        result = flow.invoke({"value": 1})

        with pytest.raises(TypeError, match="branch_index must be an int"):
            flow.get_branch_result(result.run_id, "1")  # type: ignore[arg-type]

    def test_get_branch_result_rejects_negative_index(self) -> None:
        flow = make_three_branch_flow()

        result = flow.invoke({"value": 1})

        with pytest.raises(IndexError, match="out of range"):
            flow.get_branch_result(result.run_id, -1)

    def test_get_branch_result_rejects_out_of_range_index(self) -> None:
        flow = make_three_branch_flow()

        result = flow.invoke({"value": 1})

        with pytest.raises(IndexError, match="out of range"):
            flow.get_branch_result(result.run_id, 3)


class TestParallelFlowValidationAndErrors:
    def test_branch_invoke_failure_wrapped_as_execution_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flow = make_three_branch_flow()

        def raise_invoke(inputs: Mapping[str, Any]) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(flow.branches[1], "invoke", raise_invoke)

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
    def test_to_dict_includes_branches_output_type_and_indices(self) -> None:
        flow = make_three_branch_flow(
            output_type=ParallelFlow.DICT,
            output_indices=[0, 2],
            output_names=["first", "third"],
        )

        result = flow.invoke({"value": 1})
        data = flow.to_dict()

        assert data["branch_count"] == 3
        assert data["output_type"] == flow.output_type
        assert data["output_indices"] == [0, 2]
        assert data["output_names"] == ["first", "third"]
        assert data["checkpoint_count"] == 1
        assert data["runs"] == [result.run_id]
