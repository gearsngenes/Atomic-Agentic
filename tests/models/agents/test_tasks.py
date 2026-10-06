from __future__ import annotations

import pytest

from atomic_agentic.agents.scriptact import ScriptActAgent
from atomic_agentic.models.agents.tasks import (
    AgentTask,
    ToolAgentTask,
    PlanActTask,
    ReActTask,
    ScriptActAgentTask,
    ThinkingTask,
)
from atomic_agentic.constants.core import NO_VAL
from ...fake_engines import FakeLLMEngine


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

        assert task.completed == []
        assert task.failed_statements == []
        assert task.cache == {}
        assert task.constant_values == {}
        assert task.regenerations_used == 0
        assert task.tool_calls_used == 0

    def test_tool_calls_used_is_derived_not_stored(self) -> None:
        # No stored counter field -- tool_calls_used is a @property computed
        # from completed/failed_statements via is_dispatched.
        assert "tool_calls_used" not in ToolAgentTask.__dataclass_fields__
        assert isinstance(ToolAgentTask.__dict__["tool_calls_used"], property)

    def test_no_blackboard_fields(self) -> None:
        # The pre-toolstatement-unification blackboard model (running_
        # blackboard/executed_steps/prepared_steps/valid_cache_indices/
        # failed_cache_indices/retries_used) was fully retired --
        # completed/failed_statements/cache/constant_values replace it.
        task = ToolAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")

        assert not hasattr(task, "running_blackboard")
        assert not hasattr(task, "executed_steps")
        assert not hasattr(task, "prepared_steps")
        assert not hasattr(task, "valid_cache_indices")
        assert not hasattr(task, "failed_cache_indices")
        assert not hasattr(task, "retries_used")

    def test_no_messages_field(self) -> None:
        # ToolAgentTask.messages was removed -- superseded by base
        # AgentTask's historic_messages/task_messages split.
        task = ToolAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")

        assert not hasattr(task, "messages")

    def test_default_factories_are_independent_per_instance(self) -> None:
        first = ToolAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")
        second = ToolAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="role")

        first.completed.append("marker")  # type: ignore[arg-type]
        first.cache["x"] = 1
        first.constant_values["K_X"] = 1
        first.failed_statements.append("marker")  # type: ignore[arg-type]
        first.task_messages.append({"role": "user", "content": "hi"})

        assert second.completed == []
        assert second.cache == {}
        assert second.constant_values == {}
        assert second.failed_statements == []
        assert second.task_messages == []


class TestPlanActTask:
    def test_inherits_tool_agent_task_fields(self) -> None:
        task = PlanActTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="plan_first")

        assert isinstance(task, ToolAgentTask)

    def test_added_field_defaults(self) -> None:
        task = PlanActTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="plan_first")

        assert task.pending == []
        assert task.resolved_args == []

    def test_no_batch_counter_field(self) -> None:
        # A single compile_batches call per invoke needs no cross-round
        # ToolStatement.batch_index uniqueness tracking -- unlike
        # ScriptActAgentTask, which can regenerate multiple times.
        task = PlanActTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="plan_first")

        assert not hasattr(task, "batch_counter")


class TestReActTask:
    def test_inherits_tool_agent_task_fields(self) -> None:
        task = ReActTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="reason_then_act")

        assert isinstance(task, ToolAgentTask)

    def test_added_field_defaults(self) -> None:
        task = ReActTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="reason_then_act")

        assert task.generated_step is None
        assert task.resolved_args is None
        assert task.last_call_failed is False

    def test_generated_step_holds_a_tool_statement(self) -> None:
        from atomic_agentic.models.agents.blackboard_models import ToolStatement

        step = ToolStatement(identifier=None, tool="add")
        task = ReActTask(
            turns=[], inputs={}, user_prompt="hi", system_prompt_name="reason_then_act",
            generated_step=step,
        )

        assert task.generated_step is step

    def test_no_step_meta_or_next_step_index_fields(self) -> None:
        # No fixed-size preallocated board/observability-decay window --
        # both dropped from the pre-rewrite shape, along with
        # next_step_index/step_meta. Every round renders a full snapshot of
        # completed/cache instead.
        task = ReActTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="reason_then_act")

        assert not hasattr(task, "next_step_index")
        assert not hasattr(task, "step_meta")


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

    def test_thoughts_holds_raw_values_directly(self) -> None:
        task = ThinkingTask(
            turns=[], inputs={}, user_prompt="hi", system_prompt_name="thinking",
            thoughts=["a raw thought", {"focus": "x"}],
        )

        assert task.thoughts == ["a raw thought", {"focus": "x"}]

    def test_default_factories_are_independent_per_instance(self) -> None:
        first = ThinkingTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="thinking")
        second = ThinkingTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="thinking")

        first.thoughts.append("x")

        assert second.thoughts == []


class TestScriptActAgentTask:
    def test_inherits_tool_agent_task_fields(self) -> None:
        task = ScriptActAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="planner")

        assert isinstance(task, ToolAgentTask)

    def test_added_field_defaults(self) -> None:
        task = ScriptActAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="planner")

        assert task.pending == []
        assert task.resolved_args == []
        assert task.needs_repair is False
        assert task.repair_rounds_used == 0
        assert task.repair_batch_start == 0
        assert task.batch_counter == 0
        # Inherited from ToolAgentTask, not redeclared here.
        assert task.completed == []
        assert task.failed_statements == []
        assert task.cache == {}
        assert task.constant_values == {}
        assert task.regenerations_used == 0
        assert task.tool_calls_used == 0

    def test_no_pre_repair_rework_fields(self) -> None:
        # planning_rounds_used/continue_planning/continuation_note were all
        # retired by the repair-on-failure rework (Pass 8) -- needs_repair/
        # repair_rounds_used/repair_batch_start replace them.
        task = ScriptActAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="planner")

        assert not hasattr(task, "planning_rounds_used")
        assert not hasattr(task, "continue_planning")
        assert not hasattr(task, "continuation_note")

    def test_default_factories_are_independent_per_instance(self) -> None:
        first = ScriptActAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="planner")
        second = ScriptActAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="planner")

        first.completed.append("marker")  # type: ignore[arg-type]
        first.cache["x"] = 1
        first.constant_values["K_X"] = 1
        first.failed_statements.append("marker")  # type: ignore[arg-type]
        first.pending.append(["marker"])  # type: ignore[list-item]

        assert second.completed == []
        assert second.cache == {}
        assert second.constant_values == {}
        assert second.failed_statements == []
        assert second.pending == []

    def test_needs_repair_round_trips(self) -> None:
        task = ScriptActAgentTask(turns=[], inputs={}, user_prompt="hi", system_prompt_name="planner")

        task.needs_repair = True

        assert task.needs_repair is True


class TestScriptActAgentTaskConstantValuesSeeding:
    """
    Integration-level: constant_values is populated by
    ScriptActAgent._initialize_task, not by bare ScriptActAgentTask
    construction. Cross-invocation/cross-round mutation-persistence
    behavior belongs to tests/agents/test_script.py's
    TestScriptActAgentMutationSafety -- these cases stay focused on the
    field itself getting populated correctly.
    """

    def test_constant_values_populated_with_the_right_keys_and_values(self) -> None:
        agent = ScriptActAgent(
            name="tests", namespace="tests", description="test",
            llm_engine=FakeLLMEngine(responses=[]),
        )
        agent.register_constant([1, 2, 3], alias="mylist")

        task = agent._initialize_task(turns=[], prompt="test", inputs={})

        constant_name = agent.get_constant("mylist").name
        assert task.constant_values == {constant_name: [1, 2, 3]}

    def test_mutable_constant_value_is_a_deep_copy_not_the_live_object(self) -> None:
        original = [1, 2, 3]
        agent = ScriptActAgent(
            name="tests", namespace="tests", description="test",
            llm_engine=FakeLLMEngine(responses=[]),
        )
        agent.register_constant(original, alias="mylist")

        task = agent._initialize_task(turns=[], prompt="test", inputs={})

        constant_name = agent.get_constant("mylist").name
        assert task.constant_values[constant_name] is not original
