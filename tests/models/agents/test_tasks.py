from __future__ import annotations

from atomic_agentic.models.agents.tasks import (
    AgentTask,
    ToolAgentTask,
    PlanActTask,
    ReActTask,
    ReActStepMeta,
)
from atomic_agentic.constants.core import NO_VAL


class TestAgentTask:
    def test_required_fields_are_stored(self) -> None:
        task = AgentTask(turns=[], inputs={}, user_prompt="hi")

        assert task.turns == []
        assert task.inputs == {}
        assert task.user_prompt == "hi"

    def test_defaults(self) -> None:
        first = AgentTask(turns=[], inputs={}, user_prompt="hi")
        second = AgentTask(turns=[], inputs={}, user_prompt="hi")

        assert first.llm_records == []
        assert first.complete is False
        assert first.generated_response is NO_VAL

        # Guard against a shared default_factory bug.
        first.llm_records.append("marker")  # type: ignore[arg-type]
        assert second.llm_records == []

    def test_is_mutable(self) -> None:
        task = AgentTask(turns=[], inputs={}, user_prompt="hi")

        task.complete = True
        task.generated_response = "x"

        assert task.complete is True
        assert task.generated_response == "x"


class TestToolAgentTask:
    def test_inherits_agent_task_fields(self) -> None:
        task = ToolAgentTask(turns=[], inputs={}, user_prompt="hi")

        assert isinstance(task, AgentTask)

    def test_added_field_defaults(self) -> None:
        task = ToolAgentTask(turns=[], inputs={}, user_prompt="hi")

        assert task.messages == []
        assert task.cache_blackboard == []
        assert task.running_blackboard == []
        assert task.executed_steps == set()
        assert task.prepared_steps == []
        assert task.tool_calls_used == 0
        assert task.valid_cache_indices == frozenset()
        assert task.failed_cache_indices == frozenset()
        assert task.retries_used == 0

    def test_default_factories_are_independent_per_instance(self) -> None:
        first = ToolAgentTask(turns=[], inputs={}, user_prompt="hi")
        second = ToolAgentTask(turns=[], inputs={}, user_prompt="hi")

        first.executed_steps.add(0)
        first.prepared_steps.append(1)
        first.messages.append({"role": "user", "content": "hi"})

        assert second.executed_steps == set()
        assert second.prepared_steps == []
        assert second.messages == []


class TestPlanActTask:
    def test_inherits_tool_agent_task_fields(self) -> None:
        task = PlanActTask(turns=[], inputs={}, user_prompt="hi")

        assert isinstance(task, ToolAgentTask)

    def test_added_field_defaults(self) -> None:
        task = PlanActTask(turns=[], inputs={}, user_prompt="hi")

        assert task.batches == []
        assert task.batch_index == 0


class TestReActStepMeta:
    def test_defaults(self) -> None:
        meta = ReActStepMeta()

        assert meta.observable == 0
        assert meta.description == ""

    def test_construction_with_values(self) -> None:
        meta = ReActStepMeta(observable=3, description="did a thing")

        assert meta.observable == 3
        assert meta.description == "did a thing"


class TestReActTask:
    def test_inherits_tool_agent_task_fields(self) -> None:
        task = ReActTask(turns=[], inputs={}, user_prompt="hi")

        assert isinstance(task, ToolAgentTask)

    def test_added_field_defaults(self) -> None:
        task = ReActTask(turns=[], inputs={}, user_prompt="hi")

        assert task.next_step_index == 0
        assert task.step_meta == []

    def test_step_meta_holds_react_step_meta_instances(self) -> None:
        meta = ReActStepMeta(observable=1, description="d")
        task = ReActTask(turns=[], inputs={}, user_prompt="hi", step_meta=[meta])

        assert task.step_meta == [meta]
