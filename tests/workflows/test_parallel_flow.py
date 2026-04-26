from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
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
from atomic_agentic.workflows.metadata import OutputTopology, WorkflowRunMetadata
from atomic_agentic.workflows.parallel import ParallelFlow


def make_param(
    name: str,
    index: int,
    *,
    default: Any = NO_VAL,
    kind: str = ParamSpec.POSITIONAL_OR_KEYWORD,
) -> ParamSpec:
    return ParamSpec(
        name=name,
        index=index,
        kind=kind,
        type="Any",
        default=default,
    )


def value_param() -> ParamSpec:
    return make_param("value", 0)


def return_value(value: Any) -> Any:
    """Return the provided value."""
    return value


def make_raw_tool() -> Tool:
    return Tool(
        function=return_value,
        name="return_value",
        namespace="tests",
        description="Return the provided value.",
    )


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
        ignore_unhandled=True,
    )


def make_branch_component(
    *,
    name: str,
    output_key: str,
    transform: Callable[[Any], Any] | None = None,
    calls: list[dict[str, Any]] | None = None,
) -> StructuredInvokable:
    transform_fn = transform or (lambda value: value)

    def branch(value: Any) -> dict[str, Any]:
        if calls is not None:
            calls.append({"value": value})
        return {output_key: transform_fn(value)}

    return make_structured_component(
        branch,
        name=name,
        output_schema=[output_key],
    )


def make_two_branch_flow(
    *,
    branches: list[Workflow | StructuredInvokable] | None = None,
    parameters: type | list[str] | tuple[str, ...] | set[str] | list[ParamSpec] | None = None,
    output_names: list[str] | None = None,
    output_indices: list[int] | None = None,
    output_range: tuple[int, int] | None = None,
    input_shape: str = ParallelFlow.BROADCAST,
    filter_extraneous_inputs: bool | None = None,
) -> ParallelFlow:
    resolved_branches = branches or [
        make_branch_component(name="left_branch", output_key="left", transform=lambda value: value + 1),
        make_branch_component(name="right_branch", output_key="right", transform=lambda value: value * 2),
    ]
    resolved_output_names = output_names or [
        f"branch_{index}" for index in range(len(resolved_branches))
    ]

    return ParallelFlow(
        name="parallel_flow",
        description="Parallel test flow.",
        branches=resolved_branches,
        input_shape=input_shape,
        parameters=parameters if parameters is not None else ["value"],
        output_shape=ParallelFlow.NESTED,
        output_indices=output_indices,
        output_range=output_range,
        output_names=resolved_output_names,
        filter_extraneous_inputs=filter_extraneous_inputs,
    )


def make_flattened_flow(
    *,
    branches: list[Workflow | StructuredInvokable] | None = None,
    duplicate_key_policy: str = ParallelFlow.RAISE,
    output_indices: list[int] | None = None,
) -> ParallelFlow:
    resolved_branches = branches or [
        make_branch_component(name="a_branch", output_key="a", transform=lambda value: value + 1),
        make_branch_component(name="b_branch", output_key="b", transform=lambda value: value * 2),
    ]
    flow = make_two_branch_flow(
        branches=resolved_branches,
        output_names=[f"branch_{index}" for index in range(len(resolved_branches))],
    )
    flow.duplicate_key_policy = duplicate_key_policy
    flow.configure_output(
        output_indices=output_indices,
        output_shape=ParallelFlow.FLATTENED,
        output_names=None,
        duplicate_key_policy=duplicate_key_policy,
    )
    return flow


class EchoWorkflow(Workflow[WorkflowRunMetadata]):
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
        return WorkflowRunMetadata(kind="echo"), {"value": inputs["value"]}

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[WorkflowRunMetadata, Mapping[str, Any]]:
        self.async_run_inputs.append(dict(inputs))
        return WorkflowRunMetadata(kind="async_echo"), {"value": inputs["value"]}


def raising_branch(value: Any) -> dict[str, Any]:
    """Raise during branch execution."""
    raise RuntimeError("branch failed")


class TestParallelFlowConstruction:
    def test_constructor_rejects_non_list_branches(self) -> None:
        with pytest.raises(TypeError, match="branches must be"):
            ParallelFlow(
                name="bad_flow",
                description="Bad flow.",
                branches=(EchoWorkflow(),),  # type: ignore[arg-type]
                parameters=["value"],
                output_names=["only"],
            )

    def test_constructor_rejects_empty_branches(self) -> None:
        with pytest.raises(ValueError, match="branches must not be empty"):
            ParallelFlow(
                name="bad_flow",
                description="Bad flow.",
                branches=[],
                parameters=["value"],
                output_names=[],
            )

    def test_constructor_rejects_raw_tool_branch(self) -> None:
        with pytest.raises(TypeError, match="Workflow or StructuredInvokable"):
            ParallelFlow(
                name="bad_flow",
                description="Bad flow.",
                branches=[make_raw_tool()],  # type: ignore[list-item]
                parameters=["value"],
                output_names=["only"],
            )

    def test_constructor_preserves_workflow_branches(self) -> None:
        left = EchoWorkflow(name="left_workflow")
        right = EchoWorkflow(name="right_workflow")

        flow = make_two_branch_flow(
            branches=[left, right],
            output_names=["left", "right"],
        )

        assert flow.branches == (left, right)
        assert flow.branches[0] is left
        assert flow.branches[1] is right

    def test_constructor_wraps_structured_branches_in_basic_flow(self) -> None:
        left = make_branch_component(name="left_branch", output_key="left")
        right = make_branch_component(name="right_branch", output_key="right")

        flow = make_two_branch_flow(
            branches=[left, right],
            output_names=["left", "right"],
        )

        assert all(isinstance(branch, BasicFlow) for branch in flow.branches)
        assert flow.branches[0].component is left  # type: ignore[attr-defined]
        assert flow.branches[1].component is right  # type: ignore[attr-defined]

    def test_constructor_rejects_invalid_input_shape(self) -> None:
        with pytest.raises(ValueError, match="input_shape"):
            make_two_branch_flow(input_shape="weird")

    def test_broadcast_without_parameters_uses_first_branch_parameter_fallback(self) -> None:
        branches = [
            make_branch_component(name="left_branch", output_key="left"),
            make_branch_component(name="right_branch", output_key="right"),
        ]

        flow = ParallelFlow(
            name="parallel_flow",
            description="Parallel test flow.",
            branches=branches,
            output_names=["left", "right"],
        )

        assert [param.name for param in flow.parameters] == ["value"]
        assert flow.to_dict()["parameters_fallback_used"] is True

    def test_broadcast_with_explicit_parameters_uses_given_parameters(self) -> None:
        flow = make_two_branch_flow(parameters=["value"])

        assert [param.name for param in flow.parameters] == ["value"]
        assert flow.to_dict()["parameters_fallback_used"] is False

    def test_nested_requires_explicit_parameters(self) -> None:
        branches = [
            make_branch_component(name="left_branch", output_key="left"),
            make_branch_component(name="right_branch", output_key="right"),
        ]

        with pytest.raises(ValueError, match="parameters are required"):
            ParallelFlow(
                name="parallel_flow",
                description="Parallel test flow.",
                branches=branches,
                input_shape=ParallelFlow.NESTED,
                parameters=None,
                output_names=["left", "right"],
            )

    def test_nested_requires_parameter_count_equal_branch_count(self) -> None:
        with pytest.raises(ValueError, match=r"len\(parameters\)"):
            make_two_branch_flow(
                input_shape=ParallelFlow.NESTED,
                parameters=["only_one_payload"],
            )

    @pytest.mark.parametrize(
        "parameters",
        [
            [
                make_param("left_payload", 0, kind=ParamSpec.VAR_POSITIONAL),
                make_param("right_payload", 1, kind=ParamSpec.KEYWORD_ONLY),
            ],
            [
                make_param("left_payload", 0),
                make_param("right_payload", 1, kind=ParamSpec.VAR_KEYWORD),
            ],
        ],
    )
    def test_nested_rejects_variadic_parameters(
        self,
        parameters: list[ParamSpec],
    ) -> None:
        with pytest.raises(ValueError, match="does not permit"):
            make_two_branch_flow(
                input_shape=ParallelFlow.NESTED,
                parameters=parameters,
            )

    def test_constructor_inherits_first_branch_filter_flag_by_default(self) -> None:
        left = EchoWorkflow(name="left_workflow", filter_extraneous_inputs=False)
        right = EchoWorkflow(name="right_workflow", filter_extraneous_inputs=True)

        flow = make_two_branch_flow(
            branches=[left, right],
            output_names=["left", "right"],
            parameters=["value"],
        )

        assert flow.filter_extraneous_inputs is False

    def test_constructor_allows_filter_flag_override(self) -> None:
        left = EchoWorkflow(name="left_workflow", filter_extraneous_inputs=False)
        right = EchoWorkflow(name="right_workflow", filter_extraneous_inputs=True)

        flow = make_two_branch_flow(
            branches=[left, right],
            output_names=["left", "right"],
            parameters=["value"],
            filter_extraneous_inputs=True,
        )

        assert flow.filter_extraneous_inputs is True


class TestParallelFlowOutputConfiguration:
    def test_nested_output_requires_output_names(self) -> None:
        with pytest.raises(ValueError, match="output_names are required"):
            ParallelFlow(
                name="parallel_flow",
                description="Parallel test flow.",
                branches=[make_branch_component(name="left_branch", output_key="left")],
                parameters=["value"],
                output_shape=ParallelFlow.NESTED,
                output_names=None,
            )

    def test_nested_output_names_must_be_list(self) -> None:
        with pytest.raises(TypeError, match="output_names"):
            ParallelFlow(
                name="parallel_flow",
                description="Parallel test flow.",
                branches=[make_branch_component(name="left_branch", output_key="left")],
                parameters=["value"],
                output_shape=ParallelFlow.NESTED,
                output_names=("left",),  # type: ignore[arg-type]
            )

    def test_nested_output_names_length_must_match_projected_outputs(self) -> None:
        with pytest.raises(ValueError, match=r"len\(output_names\)"):
            make_two_branch_flow(output_names=["only_one_name"])

    @pytest.mark.parametrize("bad_name", ["", "   "])
    def test_nested_output_names_must_be_non_empty_strings(self, bad_name: str) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            make_two_branch_flow(output_names=[bad_name, "right"])

    @pytest.mark.parametrize("bad_name", ["bad-name", "1bad"])
    def test_nested_output_names_must_be_identifier_style(self, bad_name: str) -> None:
        with pytest.raises(ValueError, match="valid parameter-style name"):
            make_two_branch_flow(output_names=[bad_name, "right"])

    def test_nested_output_names_must_be_unique(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            make_two_branch_flow(output_names=["same", "same"])

    def test_output_indices_select_subset_in_order(self) -> None:
        branches = [
            make_branch_component(name="first_branch", output_key="first"),
            make_branch_component(name="second_branch", output_key="second"),
            make_branch_component(name="third_branch", output_key="third"),
        ]

        flow = make_two_branch_flow(
            branches=branches,
            output_indices=[2, 0],
            output_names=["third", "first"],
        )

        assert flow.output_indices == [2, 0]
        assert flow.output_names == ["third", "first"]

    def test_output_indices_resolve_negative_indices(self) -> None:
        branches = [
            make_branch_component(name="first_branch", output_key="first"),
            make_branch_component(name="second_branch", output_key="second"),
            make_branch_component(name="third_branch", output_key="third"),
        ]

        flow = make_two_branch_flow(
            branches=branches,
            output_indices=[-1, 0],
            output_names=["third", "first"],
        )

        assert flow.output_indices == [2, 0]

    def test_output_indices_reject_duplicates(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            make_two_branch_flow(output_indices=[0, 0], output_names=["a", "b"])

    def test_output_indices_reject_out_of_range(self) -> None:
        with pytest.raises(IndexError, match="out of range"):
            make_two_branch_flow(output_indices=[2], output_names=["missing"])

    def test_output_indices_reject_non_int_items(self) -> None:
        with pytest.raises(TypeError, match="items must be int"):
            make_two_branch_flow(output_indices=[0, "1"], output_names=["left", "right"])  # type: ignore[list-item]

    def test_output_range_selects_slice(self) -> None:
        branches = [
            make_branch_component(name="first_branch", output_key="first"),
            make_branch_component(name="second_branch", output_key="second"),
            make_branch_component(name="third_branch", output_key="third"),
        ]

        flow = make_two_branch_flow(
            branches=branches,
            output_range=(1, 3),
            output_names=["second", "third"],
        )

        assert flow.output_indices == [1, 2]

    def test_output_indices_and_range_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="either output_indices or output_range"):
            make_two_branch_flow(
                output_indices=[0],
                output_range=(0, 1),
                output_names=["left"],
            )

    def test_flattened_output_rejects_output_names(self) -> None:
        flow = make_two_branch_flow()

        with pytest.raises(ValueError, match="output_names must be None"):
            flow.configure_output(
                output_shape=ParallelFlow.FLATTENED,
                output_names=["bad"],
                duplicate_key_policy=ParallelFlow.RAISE,
            )

    @pytest.mark.parametrize(
        "policy",
        [ParallelFlow.RAISE, ParallelFlow.SKIP, ParallelFlow.UPDATE],
    )
    def test_duplicate_key_policy_accepts_valid_values(self, policy: str) -> None:
        flow = make_two_branch_flow()

        flow.duplicate_key_policy = policy

        assert flow.duplicate_key_policy == policy

    def test_duplicate_key_policy_rejects_invalid_value(self) -> None:
        flow = make_two_branch_flow()

        with pytest.raises(ValueError, match="duplicate_key_policy"):
            flow.duplicate_key_policy = "overwrite"

    @pytest.mark.xfail(
        strict=True,
        reason="Current ParallelFlow.__init__ does not pass duplicate_key_policy through to configure_output for flattened output.",
    )
    def test_flattened_output_can_be_configured_at_construction(self) -> None:
        ParallelFlow(
            name="parallel_flow",
            description="Parallel test flow.",
            branches=[
                make_branch_component(name="a_branch", output_key="a"),
                make_branch_component(name="b_branch", output_key="b"),
            ],
            parameters=["value"],
            output_shape=ParallelFlow.FLATTENED,
            output_names=None,
            duplicate_key_policy=ParallelFlow.RAISE,
        )


class TestParallelFlowBroadcastInvoke:
    def test_broadcast_invokes_every_branch_with_same_outer_inputs(self) -> None:
        left_calls: list[dict[str, Any]] = []
        right_calls: list[dict[str, Any]] = []
        flow = make_two_branch_flow(
            branches=[
                make_branch_component(
                    name="left_branch",
                    output_key="left",
                    transform=lambda value: value + 1,
                    calls=left_calls,
                ),
                make_branch_component(
                    name="right_branch",
                    output_key="right",
                    transform=lambda value: value * 2,
                    calls=right_calls,
                ),
            ],
            output_names=["left", "right"],
        )

        result = flow.invoke({"value": 3, "extra": "ignored"})

        assert result == {"left": {"left": 4}, "right": {"right": 6}}
        assert left_calls == [{"value": 3}]
        assert right_calls == [{"value": 3}]

    def test_broadcast_nested_output_returns_named_branch_results(self) -> None:
        flow = make_two_branch_flow(output_names=["left_result", "right_result"])

        result = flow.invoke({"value": 3})

        assert result == {
            "left_result": {"left": 4},
            "right_result": {"right": 6},
        }

    def test_broadcast_output_indices_project_subset_but_all_branches_execute(self) -> None:
        left_calls: list[dict[str, Any]] = []
        right_calls: list[dict[str, Any]] = []
        flow = make_two_branch_flow(
            branches=[
                make_branch_component(
                    name="left_branch",
                    output_key="left",
                    transform=lambda value: value + 1,
                    calls=left_calls,
                ),
                make_branch_component(
                    name="right_branch",
                    output_key="right",
                    transform=lambda value: value * 2,
                    calls=right_calls,
                ),
            ],
            output_indices=[1],
            output_names=["right_only"],
        )

        result = flow.invoke({"value": 3})

        assert result == {"right_only": {"right": 6}}
        assert left_calls == [{"value": 3}]
        assert right_calls == [{"value": 3}]
        assert [len(branch.checkpoints) for branch in flow.branches] == [1, 1]

    def test_broadcast_metadata_records_all_branch_records(self) -> None:
        flow = make_two_branch_flow(output_indices=[1], output_names=["right_only"])

        result = flow.invoke({"value": 3})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.kind == "parallel"
        assert len(metadata.branch_records) == 2

    def test_broadcast_metadata_output_count_matches_projected_count(self) -> None:
        flow = make_two_branch_flow(output_indices=[1], output_names=["right_only"])

        result = flow.invoke({"value": 3})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.output_count == 1

    def test_broadcast_output_topology_records_nested_projection(self) -> None:
        flow = make_two_branch_flow(output_indices=[1], output_names=["right_only"])

        result = flow.invoke({"value": 3})
        topology = flow.get_checkpoint(result.run_id).metadata.output_topology  # type: ignore[union-attr]

        assert topology.topology == OutputTopology.NESTED
        assert topology.indices == (1,)
        assert topology.names == ("right_only",)
        assert topology.duplicate_key_policy is None


class TestParallelFlowNestedInvoke:
    def test_nested_invokes_each_branch_with_its_named_payload(self) -> None:
        left_calls: list[dict[str, Any]] = []
        right_calls: list[dict[str, Any]] = []
        flow = make_two_branch_flow(
            branches=[
                make_branch_component(
                    name="left_branch",
                    output_key="left",
                    transform=lambda value: value + 1,
                    calls=left_calls,
                ),
                make_branch_component(
                    name="right_branch",
                    output_key="right",
                    transform=lambda value: value * 2,
                    calls=right_calls,
                ),
            ],
            input_shape=ParallelFlow.NESTED,
            parameters=["left_payload", "right_payload"],
            output_names=["left_result", "right_result"],
        )

        result = flow.invoke(
            {
                "left_payload": {"value": 3},
                "right_payload": {"value": 4},
            }
        )

        assert result == {
            "left_result": {"left": 4},
            "right_result": {"right": 8},
        }
        assert left_calls == [{"value": 3}]
        assert right_calls == [{"value": 4}]

    def test_nested_missing_required_payload_raises_validation_error_in_run(self) -> None:
        flow = make_two_branch_flow(
            input_shape=ParallelFlow.NESTED,
            parameters=["left_payload", "right_payload"],
            output_names=["left_result", "right_result"],
        )

        with pytest.raises(ValidationError, match="missing nested payload"):
            flow._run({"left_payload": {"value": 3}})

    def test_nested_missing_required_payload_public_invoke_wraps_execution_error(self) -> None:
        flow = make_two_branch_flow(
            input_shape=ParallelFlow.NESTED,
            parameters=["left_payload", "right_payload"],
            output_names=["left_result", "right_result"],
        )

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"left_payload": {"value": 3}})

    def test_nested_non_mapping_payload_raises_validation_error_in_run(self) -> None:
        flow = make_two_branch_flow(
            input_shape=ParallelFlow.NESTED,
            parameters=["left_payload", "right_payload"],
            output_names=["left_result", "right_result"],
        )

        with pytest.raises(ValidationError, match="must be a mapping"):
            flow._run(
                {
                    "left_payload": {"value": 3},
                    "right_payload": "not a mapping",
                }
            )

    def test_nested_default_payload_is_used_when_parameter_has_default(self) -> None:
        right_calls: list[dict[str, Any]] = []
        parameters = [
            make_param("left_payload", 0),
            make_param("right_payload", 1, default={"value": 10}),
        ]
        flow = make_two_branch_flow(
            branches=[
                make_branch_component(
                    name="left_branch",
                    output_key="left",
                    transform=lambda value: value + 1,
                ),
                make_branch_component(
                    name="right_branch",
                    output_key="right",
                    transform=lambda value: value * 2,
                    calls=right_calls,
                ),
            ],
            input_shape=ParallelFlow.NESTED,
            parameters=parameters,
            output_names=["left_result", "right_result"],
        )

        result = flow.invoke({"left_payload": {"value": 3}})

        assert result["right_result"] == {"right": 20}
        assert right_calls == [{"value": 10}]

    def test_nested_parent_checkpoint_records_outer_nested_inputs(self) -> None:
        flow = make_two_branch_flow(
            input_shape=ParallelFlow.NESTED,
            parameters=["left_payload", "right_payload"],
            output_names=["left_result", "right_result"],
        )

        result = flow.invoke(
            {
                "left_payload": {"value": 3},
                "right_payload": {"value": 4},
                "extra": "ignored",
            }
        )
        checkpoint = flow.get_checkpoint(result.run_id)

        assert checkpoint is not None
        assert checkpoint.inputs == {
            "left_payload": {"value": 3},
            "right_payload": {"value": 4},
        }


class TestParallelFlowFlattenedOutput:
    def test_flattened_output_merges_distinct_branch_keys(self) -> None:
        flow = make_flattened_flow()

        result = flow.invoke({"value": 3})

        assert result == {"a": 4, "b": 6}

    def test_flattened_output_respects_output_indices(self) -> None:
        branches = [
            make_branch_component(name="a_branch", output_key="a", transform=lambda value: value + 1),
            make_branch_component(name="b_branch", output_key="b", transform=lambda value: value * 2),
            make_branch_component(name="c_branch", output_key="c", transform=lambda value: value - 1),
        ]
        flow = make_flattened_flow(branches=branches, output_indices=[2, 0])

        result = flow.invoke({"value": 3})

        assert result == {"c": 2, "a": 4}

    def test_flattened_duplicate_policy_raise_rejects_duplicate_keys(self) -> None:
        flow = make_flattened_flow(
            branches=[
                make_branch_component(name="first_x_branch", output_key="x", transform=lambda value: 1),
                make_branch_component(name="second_x_branch", output_key="x", transform=lambda value: 2),
            ],
            duplicate_key_policy=ParallelFlow.RAISE,
        )

        with pytest.raises(ValidationError, match="duplicate flattened output key"):
            flow._run({"value": 3})

    def test_flattened_duplicate_policy_skip_keeps_first_value(self) -> None:
        flow = make_flattened_flow(
            branches=[
                make_branch_component(name="first_x_branch", output_key="x", transform=lambda value: 1),
                make_branch_component(name="second_x_branch", output_key="x", transform=lambda value: 2),
            ],
            duplicate_key_policy=ParallelFlow.SKIP,
        )

        result = flow.invoke({"value": 3})

        assert result == {"x": 1}

    def test_flattened_duplicate_policy_update_keeps_later_value(self) -> None:
        flow = make_flattened_flow(
            branches=[
                make_branch_component(name="first_x_branch", output_key="x", transform=lambda value: 1),
                make_branch_component(name="second_x_branch", output_key="x", transform=lambda value: 2),
            ],
            duplicate_key_policy=ParallelFlow.UPDATE,
        )

        result = flow.invoke({"value": 3})

        assert result == {"x": 2}

    def test_flattened_metadata_output_topology_records_duplicate_policy(self) -> None:
        flow = make_flattened_flow(duplicate_key_policy=ParallelFlow.SKIP)

        result = flow.invoke({"value": 3})
        topology = flow.get_checkpoint(result.run_id).metadata.output_topology  # type: ignore[union-attr]

        assert topology.topology == OutputTopology.FLATTENED
        assert topology.indices == (0, 1)
        assert topology.names is None
        assert topology.duplicate_key_policy == ParallelFlow.SKIP


class TestParallelFlowMetadata:
    def test_metadata_kind_is_parallel(self) -> None:
        flow = make_two_branch_flow()

        result = flow.invoke({"value": 3})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.kind == "parallel"

    def test_metadata_records_one_branch_record_per_branch(self) -> None:
        flow = make_two_branch_flow(output_indices=[1], output_names=["right_only"])

        result = flow.invoke({"value": 3})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert len(metadata.branch_records) == len(flow.branches)

    def test_branch_records_match_child_branches(self) -> None:
        flow = make_two_branch_flow()

        result = flow.invoke({"value": 3})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        for index, record in enumerate(metadata.branch_records):
            branch = flow.branches[index]
            assert record.slot == index
            assert record.instance_id == branch.instance_id
            assert record.full_name == branch.full_name
            assert record.run_id == branch.latest_run

    def test_metadata_output_count_matches_projection_count(self) -> None:
        flow = make_two_branch_flow(output_indices=[1], output_names=["right_only"])

        result = flow.invoke({"value": 3})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.output_count == 1

    def test_metadata_output_topology_matches_current_configuration(self) -> None:
        flow = make_two_branch_flow(output_indices=[1], output_names=["right_only"])

        result = flow.invoke({"value": 3})
        topology = flow.get_checkpoint(result.run_id).metadata.output_topology  # type: ignore[union-attr]

        assert topology == OutputTopology(
            topology=OutputTopology.NESTED,
            indices=(1,),
            names=("right_only",),
            duplicate_key_policy=None,
        )


class TestParallelFlowRetrieval:
    def test_get_branch_records_returns_none_for_unknown_run(self) -> None:
        flow = make_two_branch_flow()

        assert flow.get_branch_records("unknown") is None

    def test_get_branch_records_returns_metadata_records(self) -> None:
        flow = make_two_branch_flow()

        result = flow.invoke({"value": 3})
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert flow.get_branch_records(result.run_id) == metadata.branch_records

    def test_get_branch_results_returns_all_child_results(self) -> None:
        flow = make_two_branch_flow()

        result = flow.invoke({"value": 3})

        assert flow.get_branch_results(result.run_id) == [
            {"left": 4},
            {"right": 6},
        ]

    def test_get_branch_result_returns_one_child_result(self) -> None:
        flow = make_two_branch_flow()

        result = flow.invoke({"value": 3})

        assert flow.get_branch_result(result.run_id, 0) == {"left": 4}
        assert flow.get_branch_result(result.run_id, 1) == {"right": 6}

    def test_get_branch_result_supports_negative_index(self) -> None:
        flow = make_two_branch_flow()

        result = flow.invoke({"value": 3})

        assert flow.get_branch_result(result.run_id, -1) == {"right": 6}

    def test_get_branch_result_returns_none_for_unknown_parent_run(self) -> None:
        flow = make_two_branch_flow()

        assert flow.get_branch_result("unknown", 0) is None

    def test_get_branch_result_rejects_non_int_index(self) -> None:
        flow = make_two_branch_flow()

        with pytest.raises(TypeError, match="branch_index must be an int"):
            flow.get_branch_result("unknown", "0")  # type: ignore[arg-type]

    def test_get_branch_result_rejects_out_of_range_index(self) -> None:
        flow = make_two_branch_flow()

        with pytest.raises(IndexError, match="out of range"):
            flow.get_branch_result("unknown", 2)

    def test_get_branch_results_returns_none_for_missing_child_checkpoint(self) -> None:
        flow = make_two_branch_flow()

        result = flow.invoke({"value": 3})
        flow.branches[0].clear_memory()

        assert flow.get_branch_results(result.run_id) == [None, {"right": 6}]


class TestParallelFlowAsyncInvoke:
    def test_async_broadcast_invokes_all_branches(self) -> None:
        left_calls: list[dict[str, Any]] = []
        right_calls: list[dict[str, Any]] = []
        flow = make_two_branch_flow(
            branches=[
                make_branch_component(
                    name="left_branch",
                    output_key="left",
                    transform=lambda value: value + 1,
                    calls=left_calls,
                ),
                make_branch_component(
                    name="right_branch",
                    output_key="right",
                    transform=lambda value: value * 2,
                    calls=right_calls,
                ),
            ],
            output_names=["left", "right"],
        )

        result = asyncio.run(flow.async_invoke({"value": 3}))

        assert result == {"left": {"left": 4}, "right": {"right": 6}}
        assert left_calls == [{"value": 3}]
        assert right_calls == [{"value": 3}]

    def test_async_nested_routes_payloads_to_correct_branches(self) -> None:
        left_calls: list[dict[str, Any]] = []
        right_calls: list[dict[str, Any]] = []
        flow = make_two_branch_flow(
            branches=[
                make_branch_component(
                    name="left_branch",
                    output_key="left",
                    transform=lambda value: value + 1,
                    calls=left_calls,
                ),
                make_branch_component(
                    name="right_branch",
                    output_key="right",
                    transform=lambda value: value * 2,
                    calls=right_calls,
                ),
            ],
            input_shape=ParallelFlow.NESTED,
            parameters=["left_payload", "right_payload"],
            output_names=["left_result", "right_result"],
        )

        result = asyncio.run(
            flow.async_invoke(
                {
                    "left_payload": {"value": 3},
                    "right_payload": {"value": 4},
                }
            )
        )

        assert result == {
            "left_result": {"left": 4},
            "right_result": {"right": 8},
        }
        assert left_calls == [{"value": 3}]
        assert right_calls == [{"value": 4}]

    def test_async_metadata_records_all_branches(self) -> None:
        flow = make_two_branch_flow(output_indices=[1], output_names=["right_only"])

        result = asyncio.run(flow.async_invoke({"value": 3}))
        metadata = flow.get_checkpoint(result.run_id).metadata  # type: ignore[union-attr]

        assert metadata.kind == "parallel"
        assert len(metadata.branch_records) == 2
        assert metadata.output_count == 1

    def test_async_flattened_output_matches_sync_shape(self) -> None:
        flow = make_flattened_flow(duplicate_key_policy=ParallelFlow.UPDATE)

        result = asyncio.run(flow.async_invoke({"value": 3}))

        assert result == {"a": 4, "b": 6}

    def test_async_branch_failure_public_invoke_wraps_execution_error(self) -> None:
        flow = make_two_branch_flow(
            branches=[
                make_branch_component(name="good_branch", output_key="good"),
                make_structured_component(
                    raising_branch,
                    name="raising_branch",
                    output_schema=["never"],
                ),
            ],
            output_names=["good", "bad"],
        )

        with pytest.raises(ExecutionError, match="_async_run failed"):
            asyncio.run(flow.async_invoke({"value": 3}))


class TestParallelFlowValidationAndErrors:
    def test_run_raises_validation_error_if_branch_returns_non_flow_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_two_branch_flow()

        monkeypatch.setattr(flow.branches[1], "invoke", lambda inputs: {"bad": True})

        with pytest.raises(ValidationError, match="expected FlowResultDict"):
            flow._run({"value": 3})

    def test_public_invoke_wraps_bad_branch_result_as_execution_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_two_branch_flow()

        monkeypatch.setattr(flow.branches[1], "invoke", lambda inputs: {"bad": True})

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 3})

    def test_run_raises_runtime_error_if_branch_invocation_raises(self) -> None:
        flow = make_two_branch_flow(
            branches=[
                make_branch_component(name="good_branch", output_key="good"),
                make_structured_component(
                    raising_branch,
                    name="raising_branch",
                    output_schema=["never"],
                ),
            ],
            output_names=["good", "bad"],
        )

        with pytest.raises(RuntimeError, match="branch 1"):
            flow._run({"value": 3})

    def test_public_invoke_wraps_branch_runtime_failure_as_execution_error(self) -> None:
        flow = make_two_branch_flow(
            branches=[
                make_branch_component(name="good_branch", output_key="good"),
                make_structured_component(
                    raising_branch,
                    name="raising_branch",
                    output_schema=["never"],
                ),
            ],
            output_names=["good", "bad"],
        )

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 3})

    def test_async_run_raises_validation_error_if_branch_returns_non_flow_result(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_two_branch_flow()

        async def bad_async_invoke(self: Workflow[Any], inputs: Mapping[str, Any]) -> dict[str, bool]:
            return {"bad": True}

        monkeypatch.setattr(
            flow.branches[1],
            "async_invoke",
            MethodType(bad_async_invoke, flow.branches[1]),
        )

        with pytest.raises(ValidationError, match="expected FlowResultDict"):
            asyncio.run(flow._async_run({"value": 3}))

    def test_public_async_invoke_wraps_bad_branch_result_as_execution_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        flow = make_two_branch_flow()

        async def bad_async_invoke(self: Workflow[Any], inputs: Mapping[str, Any]) -> dict[str, bool]:
            return {"bad": True}

        monkeypatch.setattr(
            flow.branches[1],
            "async_invoke",
            MethodType(bad_async_invoke, flow.branches[1]),
        )

        with pytest.raises(ExecutionError, match="_async_run failed"):
            asyncio.run(flow.async_invoke({"value": 3}))

    def test_flattened_duplicate_key_raise_public_invoke_wraps_execution_error(self) -> None:
        flow = make_flattened_flow(
            branches=[
                make_branch_component(name="first_x_branch", output_key="x", transform=lambda value: 1),
                make_branch_component(name="second_x_branch", output_key="x", transform=lambda value: 2),
            ],
            duplicate_key_policy=ParallelFlow.RAISE,
        )

        with pytest.raises(ExecutionError, match="_run failed"):
            flow.invoke({"value": 3})


class TestParallelFlowSerialization:
    def test_to_dict_includes_branch_count_and_branch_snapshots(self) -> None:
        flow = make_two_branch_flow()

        data = flow.to_dict()

        assert data["type"] == "ParallelFlow"
        assert data["branch_count"] == 2
        assert len(data["branches"]) == 2

    def test_to_dict_includes_input_shape(self) -> None:
        flow = make_two_branch_flow(
            input_shape=ParallelFlow.NESTED,
            parameters=["left_payload", "right_payload"],
            output_names=["left", "right"],
        )

        data = flow.to_dict()

        assert data["input_shape"] == ParallelFlow.NESTED

    def test_to_dict_includes_parameters_fallback_used(self) -> None:
        branches = [
            make_branch_component(name="left_branch", output_key="left"),
            make_branch_component(name="right_branch", output_key="right"),
        ]
        flow = ParallelFlow(
            name="parallel_flow",
            description="Parallel test flow.",
            branches=branches,
            output_names=["left", "right"],
        )

        data = flow.to_dict()

        assert data["parameters_fallback_used"] is True

    def test_to_dict_includes_output_shape_indices_names_and_duplicate_policy(self) -> None:
        flow = make_two_branch_flow(output_indices=[1], output_names=["right_only"])

        data = flow.to_dict()

        assert data["output_shape"] == ParallelFlow.NESTED
        assert data["output_indices"] == [1]
        assert data["output_names"] == ["right_only"]
        assert data["duplicate_key_policy"] == ParallelFlow.RAISE

    def test_to_dict_reflects_reconfigured_output(self) -> None:
        flow = make_flattened_flow(duplicate_key_policy=ParallelFlow.SKIP)

        data = flow.to_dict()

        assert data["output_shape"] == ParallelFlow.FLATTENED
        assert data["output_indices"] == [0, 1]
        assert data["output_names"] is None
        assert data["duplicate_key_policy"] == ParallelFlow.SKIP

    def test_to_dict_after_run_includes_base_checkpoint_summary(self) -> None:
        flow = make_two_branch_flow()

        result = flow.invoke({"value": 3})
        data = flow.to_dict()

        assert data["checkpoint_count"] == 1
        assert data["runs"] == [result.run_id]
        assert data["branch_count"] == 2
