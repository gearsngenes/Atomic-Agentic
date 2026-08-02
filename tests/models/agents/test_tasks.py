from __future__ import annotations

import pytest

from atomic_agentic.models.agents.tasks import (
    AgentTask,
    ToolAgentTask,
    PlanActTask,
    ReActTask,
    ReActStepMeta,
    ThinkingTask,
)
from atomic_agentic.models.agents.thought_models import AgentThought
from atomic_agentic.constants.core import NO_VAL


class TestAgentTask:
    def test_required_fields_are_stored(self) -> None:
        task = AgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")

        assert task.turns == []
        assert task.inputs == {}
        assert task.user_prompt == "hi"
        assert task.system_prompt_name == "role"

    def test_defaults(self) -> None:
        first = AgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")
        second = AgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")

        assert first.llm_records == []
        assert first.complete is False
        assert first.generated_response is NO_VAL
        assert first.historic_messages == []
        assert first.task_messages == []

        # Guard against a shared default_factory bug.
        first.llm_records.append("marker")  # type: ignore[arg-type]
        first.historic_messages.append({"role": "user", "content": "x"})
        first.task_messages.append({"role": "user", "content": "y"})
        assert second.llm_records == []
        assert second.historic_messages == []
        assert second.task_messages == []

    def test_is_mutable(self) -> None:
        task = AgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")

        task.complete = True
        task.generated_response = "x"

        assert task.complete is True
        assert task.generated_response == "x"


class TestAgentTaskSystemPromptName:
    """system_prompt_name is required (no default), but None is a
    legitimate value distinct from "not set" -- it means "render no system
    message at all"."""

    def test_required_no_default(self) -> None:
        with pytest.raises(TypeError):
            AgentTask(turns=[], inputs={}, user_prompt="hi")  # type: ignore[call-arg]

    def test_none_is_a_legal_explicit_value(self) -> None:
        task = AgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name=None)

        assert task.system_prompt_name is None

    def test_string_value_is_stored(self) -> None:
        task = AgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="plan_first")

        assert task.system_prompt_name == "plan_first"


class TestToolAgentTask:
    def test_inherits_agent_task_fields(self) -> None:
        task = ToolAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")

        assert isinstance(task, AgentTask)

    def test_added_field_defaults(self) -> None:
        task = ToolAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")

        assert task.running_blackboard == []
        assert task.executed_steps == set()
        assert task.prepared_steps == []
        assert task.tool_calls_used == 0
        assert task.valid_cache_indices == frozenset()
        assert task.failed_cache_indices == frozenset()
        assert task.retries_used == 0

    def test_no_cache_blackboard_field(self) -> None:
        # cache_blackboard was removed -- cache state lives only on the
        # owning agent's self._blackboard, never on the task.
        task = ToolAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")

        assert not hasattr(task, "cache_blackboard")

    def test_no_messages_field(self) -> None:
        # ToolAgentTask.messages was removed -- superseded by base
        # AgentTask's historic_messages/task_messages split.
        task = ToolAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")

        assert not hasattr(task, "messages")

    def test_default_factories_are_independent_per_instance(self) -> None:
        first = ToolAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")
        second = ToolAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")

        first.executed_steps.add(0)
        first.prepared_steps.append(1)
        first.task_messages.append({"role": "user", "content": "hi"})

        assert second.executed_steps == set()
        assert second.prepared_steps == []
        assert second.task_messages == []


class TestPlanActTask:
    def test_inherits_tool_agent_task_fields(self) -> None:
        task = PlanActTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="plan_first")

        assert isinstance(task, ToolAgentTask)

    def test_added_field_defaults(self) -> None:
        task = PlanActTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="plan_first")

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
        task = ReActTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="reason_then_act")

        assert isinstance(task, ToolAgentTask)

    def test_added_field_defaults(self) -> None:
        task = ReActTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="reason_then_act")

        assert task.next_step_index == 0
        assert task.step_meta == []

    def test_step_meta_holds_react_step_meta_instances(self) -> None:
        meta = ReActStepMeta(observable=1, description="d")
        task = ReActTask(
            turns=[], inputs={}, user_prompt="hi", system_prompt_name="reason_then_act", step_meta=[meta]
        )

        assert task.step_meta == [meta]


class TestThinkingTask:
    def test_inherits_agent_task_fields(self) -> None:
        task = ThinkingTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="thinking")

        assert isinstance(task, AgentTask)

    def test_added_field_defaults(self) -> None:
        task = ThinkingTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="thinking")

        assert task.thoughts == []

    def test_no_retries_or_rounds_used_fields(self) -> None:
        # No retry-budget field: the free-flowing category-marker parser
        # degrades unmarked text to a single OTHER thought rather than
        # failing, so there is no malformed-output case to retry. No
        # separate round counter either -- len(thoughts) is the round count.
        task = ThinkingTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="thinking")

        assert not hasattr(task, "retries_used")
        assert not hasattr(task, "rounds_used")

    def test_no_phase_field(self) -> None:
        # ThinkingTask deliberately has no phase field -- system_prompt_name
        # (base AgentTask) doubles as the phase discriminator: "role" means
        # the reply phase, anything else means a thinking round is active.
        task = ThinkingTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="thinking")

        assert not hasattr(task, "phase")

    def test_thoughts_holds_nested_rounds_of_agent_thought_instances(self) -> None:
        thought = AgentThought(category="OBSERVATION", content="obs")
        task = ThinkingTask(
            turns=[], inputs={}, user_prompt="hi", system_prompt_name="thinking", thoughts=[[thought]]
        )

        assert task.thoughts == [[thought]]

    def test_default_factories_are_independent_per_instance(self) -> None:
        first = ThinkingTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="thinking")
        second = ThinkingTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="thinking")

        first.thoughts.append([AgentThought(category="OTHER", content="x")])

        assert second.thoughts == []
