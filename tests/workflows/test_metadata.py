from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from typing import Any

import pytest

from atomic_agentic.core.sentinels import NO_VAL
from atomic_agentic.workflows.metadata import (
    BasicFlowRunMetadata,
    ChildRunRecord,
    IterationRecord,
    IterativeFlowRunMetadata,
    OutputTopology,
    ParallelFlowRunMetadata,
    RoutingFlowRunMetadata,
    SequentialFlowRunMetadata,
    WorkflowCheckpoint,
    WorkflowRunMetadata,
)


def child_record(slot: int = 0, *, run_id: str = "child_run") -> ChildRunRecord:
    return ChildRunRecord(
        slot=slot,
        instance_id=f"child_{slot}",
        full_name=f"Workflow.child_{slot}",
        run_id=run_id,
    )


def iteration_record(iteration: int = 0, *, decision: bool = False) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        body_run_id=f"body_{iteration}",
        judge_run_id=f"judge_{iteration}",
        judge_decision=decision,
    )


class TestMetadataDataclassConstruction:
    def test_workflow_run_metadata_stores_kind(self) -> None:
        metadata = WorkflowRunMetadata(kind="custom")

        assert metadata.kind == "custom"

    def test_child_run_record_stores_slot_identity_name_and_run_id(self) -> None:
        record = child_record(slot=2, run_id="run_2")

        assert record.slot == 2
        assert record.instance_id == "child_2"
        assert record.full_name == "Workflow.child_2"
        assert record.run_id == "run_2"

    def test_output_topology_stores_nested_shape(self) -> None:
        topology = OutputTopology(
            topology=OutputTopology.NESTED,
            indices=(0, 2),
            names=("left", "right"),
        )

        assert topology.topology == "nested"
        assert topology.indices == (0, 2)
        assert topology.names == ("left", "right")
        assert topology.duplicate_key_policy is None

    def test_output_topology_stores_flattened_shape(self) -> None:
        topology = OutputTopology(
            topology=OutputTopology.FLATTENED,
            indices=(1, 0),
            names=None,
            duplicate_key_policy="update",
        )

        assert topology.topology == "flattened"
        assert topology.indices == (1, 0)
        assert topology.names is None
        assert topology.duplicate_key_policy == "update"

    def test_iteration_record_stores_iteration_body_judge_and_decision(self) -> None:
        record = iteration_record(iteration=3, decision=True)

        assert record.iteration == 3
        assert record.body_run_id == "body_3"
        assert record.judge_run_id == "judge_3"
        assert record.judge_decision is True

    def test_basic_flow_run_metadata_defaults_kind_to_basic(self) -> None:
        metadata = BasicFlowRunMetadata(
            child_is_workflow=True,
            child_id="child",
            child_full_name="Workflow.child",
            child_run_id="run_1",
        )

        assert metadata.kind == "basic"

    def test_sequential_flow_run_metadata_defaults_kind_to_sequential(self) -> None:
        metadata = SequentialFlowRunMetadata(
            step_records=(child_record(),),
            return_child_index=0,
            return_child_run_id="child_run",
        )

        assert metadata.kind == "sequential"

    def test_routing_flow_run_metadata_defaults_kind_to_routing(self) -> None:
        metadata = RoutingFlowRunMetadata(
            router_run_id="router_run",
            router_instance_id="router",
            chosen_index=0,
            chosen_branch_record=child_record(),
        )

        assert metadata.kind == "routing"

    def test_iterative_flow_run_metadata_defaults_kind_to_iterative(self) -> None:
        metadata = IterativeFlowRunMetadata(
            iterations_completed=1,
            max_iterations=3,
            judge_approved_early=True,
            return_step_index=0,
            handoff_step_index=0,
            evaluate_step_index=0,
            iteration_records=(iteration_record(decision=True),),
        )

        assert metadata.kind == "iterative"

    def test_parallel_flow_run_metadata_defaults_kind_to_parallel(self) -> None:
        metadata = ParallelFlowRunMetadata(
            branch_records=(child_record(),),
            output_topology=OutputTopology(
                topology=OutputTopology.NESTED,
                indices=(0,),
                names=("only",),
            ),
            output_count=1,
        )

        assert metadata.kind == "parallel"


class TestMetadataImmutabilityAndSlots:
    def test_child_run_record_is_frozen(self) -> None:
        record = child_record()

        with pytest.raises(FrozenInstanceError):
            record.run_id = "new_run"  # type: ignore[misc]

    def test_iteration_record_is_frozen(self) -> None:
        record = iteration_record()

        with pytest.raises(FrozenInstanceError):
            record.judge_decision = True  # type: ignore[misc]

    def test_iterative_flow_run_metadata_is_frozen(self) -> None:
        metadata = IterativeFlowRunMetadata(
            iterations_completed=1,
            max_iterations=3,
            judge_approved_early=False,
            return_step_index=0,
            handoff_step_index=0,
            evaluate_step_index=0,
            iteration_records=(iteration_record(),),
        )

        with pytest.raises(FrozenInstanceError):
            metadata.iterations_completed = 2  # type: ignore[misc]

    @pytest.mark.parametrize(
        "record",
        [
            child_record(),
            iteration_record(),
            OutputTopology(topology="nested", indices=(0,), names=("only",)),
            BasicFlowRunMetadata(child_is_workflow=False, child_id="id", child_full_name="full"),
        ],
    )
    def test_metadata_records_do_not_have_instance_dict(self, record: Any) -> None:
        assert not hasattr(record, "__dict__")

    def test_workflow_checkpoint_is_frozen(self) -> None:
        checkpoint = WorkflowCheckpoint(
            run_id="run_1",
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            elapsed_s=0.1,
            inputs={"value": 1},
            result={"value": 2},
            metadata=WorkflowRunMetadata(kind="test"),
        )

        with pytest.raises(FrozenInstanceError):
            checkpoint.result = {"value": 3}  # type: ignore[misc]

    def test_workflow_checkpoint_does_not_have_instance_dict(self) -> None:
        checkpoint = WorkflowCheckpoint(
            run_id="run_1",
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            elapsed_s=0.1,
            inputs={"value": 1},
            result={"value": 2},
            metadata=WorkflowRunMetadata(kind="test"),
        )

        assert not hasattr(checkpoint, "__dict__")


class TestMetadataNoValDefaults:
    def test_basic_metadata_workflow_child_defaults_raw_result_to_no_val(self) -> None:
        metadata = BasicFlowRunMetadata(
            child_is_workflow=True,
            child_id="child",
            child_full_name="Workflow.child",
            child_run_id="run_1",
        )

        assert metadata.child_raw_result is NO_VAL
        assert metadata.has_child_raw_result is False
        assert metadata.child_raw_result_type == "Any"

    def test_basic_metadata_structured_child_can_store_raw_result(self) -> None:
        metadata = BasicFlowRunMetadata(
            child_is_workflow=False,
            child_id="child",
            child_full_name="Structured.child",
            child_raw_result={"raw": True},
            has_child_raw_result=True,
            child_raw_result_type="dict",
        )

        assert metadata.child_raw_result == {"raw": True}
        assert metadata.has_child_raw_result is True
        assert metadata.child_raw_result_type == "dict"

    def test_basic_metadata_default_child_run_id_is_no_val(self) -> None:
        metadata = BasicFlowRunMetadata(
            child_is_workflow=False,
            child_id="child",
            child_full_name="Structured.child",
        )

        assert metadata.child_run_id is NO_VAL

    def test_no_val_identity_is_preserved(self) -> None:
        metadata = BasicFlowRunMetadata(
            child_is_workflow=False,
            child_id="child",
            child_full_name="Structured.child",
        )

        assert metadata.child_run_id is NO_VAL
        assert metadata.child_raw_result is NO_VAL


class TestWorkflowCheckpointMetadataCarrier:
    def test_workflow_checkpoint_stores_inputs_result_and_metadata(self) -> None:
        metadata = WorkflowRunMetadata(kind="test")
        checkpoint = WorkflowCheckpoint(
            run_id="run_1",
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            elapsed_s=0.1,
            inputs={"value": 1},
            result={"value": 2},
            metadata=metadata,
        )

        assert checkpoint.inputs == {"value": 1}
        assert checkpoint.result == {"value": 2}
        assert checkpoint.metadata is metadata

    def test_workflow_checkpoint_is_generic_runtime_container(self) -> None:
        metadata = IterativeFlowRunMetadata(
            iterations_completed=1,
            max_iterations=1,
            judge_approved_early=True,
            return_step_index=0,
            handoff_step_index=0,
            evaluate_step_index=0,
            iteration_records=(iteration_record(decision=True),),
        )
        checkpoint: WorkflowCheckpoint[IterativeFlowRunMetadata] = WorkflowCheckpoint(
            run_id="run_1",
            started_at=datetime.now(timezone.utc),
            ended_at=datetime.now(timezone.utc),
            elapsed_s=0.1,
            inputs={"value": 1},
            result={"value": 2},
            metadata=metadata,
        )

        assert checkpoint.metadata.kind == "iterative"
        assert checkpoint.metadata.iterations_completed == 1

    def test_workflow_checkpoint_preserves_elapsed_fields(self) -> None:
        started_at = datetime.now(timezone.utc)
        ended_at = datetime.now(timezone.utc)
        checkpoint = WorkflowCheckpoint(
            run_id="run_1",
            started_at=started_at,
            ended_at=ended_at,
            elapsed_s=0.25,
            inputs={},
            result={},
            metadata=WorkflowRunMetadata(kind="test"),
        )

        assert checkpoint.started_at is started_at
        assert checkpoint.ended_at is ended_at
        assert checkpoint.elapsed_s == 0.25
