from __future__ import annotations

import pytest

from atomic_agentic.models.agents.tasks import (
    AgentTask,
    ToolAgentTask,
    PlanActTask,
    ReActTask,
    ScratchpadTask,
    ScriptActAgentTask,
    ThinkingTask,
)
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.models.agents.context import ContextVariable


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
        task = AgentTask(turns=[], inputs={}, user_prompt="hi")

        task.complete = True
        task.generated_response = "x"

        assert task.complete is True
        assert task.generated_response == "x"


class TestThinkingComplete:
    """``thinking_complete`` replaced ``system_prompt_name``: True by default
    for a single-phase task, False for a thinking task."""

    def test_agent_task_defaults_to_complete(self) -> None:
        assert AgentTask(turns=[], inputs={}).thinking_complete is True

    def test_tool_task_defaults_to_complete(self) -> None:
        assert ToolAgentTask(turns=[], inputs={}).thinking_complete is True

    def test_thinking_task_defaults_to_not_complete(self) -> None:
        assert ThinkingTask(turns=[], inputs={}).thinking_complete is False

    def test_no_system_prompt_name_field(self) -> None:
        assert not hasattr(AgentTask(turns=[], inputs={}), "system_prompt_name")

    def test_is_mutable(self) -> None:
        task = ThinkingTask(turns=[], inputs={})
        task.thinking_complete = True
        assert task.thinking_complete is True


class TestToolAgentTask:
    def test_inherits_agent_task_fields(self) -> None:
        task = ToolAgentTask(turns=[], inputs={}, user_prompt="hi")

        assert isinstance(task, AgentTask)

    def test_added_field_defaults(self) -> None:
        task = ToolAgentTask(turns=[], inputs={}, user_prompt="hi")

        assert task.completed == []
        assert task.failed_statements == []
        assert not hasattr(task, "cache")
        assert not hasattr(task, "constant_values")
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
        # completed/failed_statements plus task.context replace it.
        task = ToolAgentTask(turns=[], inputs={}, user_prompt="hi")

        assert not hasattr(task, "running_blackboard")
        assert not hasattr(task, "executed_steps")
        assert not hasattr(task, "prepared_steps")
        assert not hasattr(task, "valid_cache_indices")
        assert not hasattr(task, "failed_cache_indices")
        assert not hasattr(task, "retries_used")

    def test_no_messages_field(self) -> None:
        # ToolAgentTask.messages was removed -- superseded by base
        # AgentTask's historic_messages/task_messages split.
        task = ToolAgentTask(turns=[], inputs={}, user_prompt="hi")

        assert not hasattr(task, "messages")

    def test_default_factories_are_independent_per_instance(self) -> None:
        first = ToolAgentTask(turns=[], inputs={}, user_prompt="hi")
        second = ToolAgentTask(turns=[], inputs={}, user_prompt="hi")

        first.completed.append("marker")  # type: ignore[arg-type]
        first.failed_statements.append("marker")  # type: ignore[arg-type]
        first.task_messages.append({"role": "user", "content": "hi"})

        assert second.completed == []
        assert second.failed_statements == []
        assert second.task_messages == []


class TestPlanActTask:
    def test_inherits_tool_agent_task_fields(self) -> None:
        task = PlanActTask(turns=[], inputs={}, user_prompt="hi")

        assert isinstance(task, ToolAgentTask)

    def test_added_field_defaults(self) -> None:
        task = PlanActTask(turns=[], inputs={}, user_prompt="hi")

        assert task.pending == []
        assert task.resolved_args == []
        assert not hasattr(task, "cache")
        assert not hasattr(task, "constant_values")

    def test_no_batch_counter_field(self) -> None:
        # A single compile_batches call per invoke needs no cross-round
        # ToolStatement.batch_index uniqueness tracking -- unlike
        # ScriptActAgentTask, which can regenerate multiple times.
        task = PlanActTask(turns=[], inputs={}, user_prompt="hi")

        assert not hasattr(task, "batch_counter")


class TestReActTask:
    def test_inherits_tool_agent_task_fields(self) -> None:
        task = ReActTask(turns=[], inputs={}, user_prompt="hi")

        assert isinstance(task, ToolAgentTask)

    def test_added_field_defaults(self) -> None:
        task = ReActTask(turns=[], inputs={}, user_prompt="hi")

        assert task.generated_step is None
        assert task.resolved_args is None
        assert task.last_call_failed is False
        assert not hasattr(task, "cache")
        assert not hasattr(task, "constant_values")

    def test_generated_step_holds_a_tool_statement(self) -> None:
        from atomic_agentic.models.agents.blackboard_models import ToolStatement

        step = ToolStatement(identifier=None, tool="add")
        task = ReActTask(
            turns=[], inputs={}, user_prompt="hi",
            generated_step=step,
        )

        assert task.generated_step is step

    def test_no_step_meta_or_next_step_index_fields(self) -> None:
        # No fixed-size preallocated board/observability-decay window --
        # both dropped from the pre-rewrite shape, along with
        # next_step_index/step_meta. Every round renders a full snapshot of
        # completed/cache instead.
        task = ReActTask(turns=[], inputs={}, user_prompt="hi")

        assert not hasattr(task, "next_step_index")
        assert not hasattr(task, "step_meta")


class TestThinkingTask:
    def test_inherits_agent_task_fields(self) -> None:
        task = ThinkingTask(turns=[], inputs={}, user_prompt="hi")

        assert isinstance(task, AgentTask)

    def test_added_field_defaults(self) -> None:
        task = ThinkingTask(turns=[], inputs={}, user_prompt="hi")

        assert task.thoughts == []

    def test_no_retries_or_rounds_used_fields(self) -> None:
        # No retry-budget field and no separate round counter: the base
        # keeps no budget or count (those, if any, belong to the strategy).
        task = ThinkingTask(turns=[], inputs={}, user_prompt="hi")

        assert not hasattr(task, "retries_used")
        assert not hasattr(task, "rounds_used")

    def test_no_thinking_rounds_field(self) -> None:
        task = ThinkingTask(turns=[], inputs={}, user_prompt="hi")

        assert not hasattr(task, "thinking_rounds")

    def test_no_phase_field(self) -> None:
        # The phase is the base ``thinking_complete`` flag, not a separate
        # ``phase`` field.
        task = ThinkingTask(turns=[], inputs={}, user_prompt="hi")

        assert not hasattr(task, "phase")

    def test_thoughts_holds_raw_values_directly(self) -> None:
        task = ThinkingTask(
            turns=[], inputs={}, user_prompt="hi",
            thoughts=["a raw thought", {"focus": "x"}, None, 3, [1, {"k": ["v"]}]],
        )

        assert task.thoughts == ["a raw thought", {"focus": "x"}, None, 3, [1, {"k": ["v"]}]]

    def test_default_factories_are_independent_per_instance(self) -> None:
        first = ThinkingTask(turns=[], inputs={}, user_prompt="hi")
        second = ThinkingTask(turns=[], inputs={}, user_prompt="hi")

        first.thoughts.append("x")

        assert second.thoughts == []


class TestScratchpadTask:
    def _make(self, **kwargs) -> ScratchpadTask:
        return ScratchpadTask(
            turns=[], inputs={}, user_prompt="hi", **kwargs
        )

    def test_is_a_thinking_task(self) -> None:
        assert isinstance(self._make(), ThinkingTask)

    def test_added_field_defaults_to_keep_thinking(self) -> None:
        task = self._make()

        assert task.more_thinking_required is True
        assert task.thoughts == []

    def test_flag_is_settable_and_independent_per_instance(self) -> None:
        first = self._make()
        second = self._make()

        first.more_thinking_required = False

        assert second.more_thinking_required is True

    def test_base_thinking_task_has_no_flag(self) -> None:
        task = ThinkingTask(turns=[], inputs={}, user_prompt="hi")

        assert not hasattr(task, "more_thinking_required")


class TestScriptActAgentTask:
    def test_inherits_tool_agent_task_fields(self) -> None:
        task = ScriptActAgentTask(turns=[], inputs={}, user_prompt="hi")

        assert isinstance(task, ToolAgentTask)

    def test_added_field_defaults(self) -> None:
        task = ScriptActAgentTask(turns=[], inputs={}, user_prompt="hi")

        assert task.pending == []
        assert task.resolved_args == []
        assert task.needs_repair is False
        assert task.repair_rounds_used == 0
        assert task.repair_batch_start == 0
        assert task.batch_counter == 0
        # Inherited from ToolAgentTask, not redeclared here.
        assert task.completed == []
        assert task.failed_statements == []
        assert not hasattr(task, "cache")
        assert not hasattr(task, "constant_values")
        assert task.regenerations_used == 0
        assert task.tool_calls_used == 0

    def test_no_pre_repair_rework_fields(self) -> None:
        # planning_rounds_used/continue_planning/continuation_note were all
        # retired by the repair-on-failure rework (Pass 8) -- needs_repair/
        # repair_rounds_used/repair_batch_start replace them.
        task = ScriptActAgentTask(turns=[], inputs={}, user_prompt="hi")

        assert not hasattr(task, "planning_rounds_used")
        assert not hasattr(task, "continue_planning")
        assert not hasattr(task, "continuation_note")

    def test_default_factories_are_independent_per_instance(self) -> None:
        first = ScriptActAgentTask(turns=[], inputs={}, user_prompt="hi")
        second = ScriptActAgentTask(turns=[], inputs={}, user_prompt="hi")

        first.completed.append("marker")  # type: ignore[arg-type]
        first.failed_statements.append("marker")  # type: ignore[arg-type]
        first.pending.append(["marker"])  # type: ignore[list-item]

        assert second.completed == []
        assert second.failed_statements == []
        assert second.pending == []

    def test_needs_repair_round_trips(self) -> None:
        task = ScriptActAgentTask(turns=[], inputs={}, user_prompt="hi")

        task.needs_repair = True

        assert task.needs_repair is True


class TestAgentTaskContextFields:
    def test_new_fields_default(self) -> None:
        task = AgentTask(turns=[], inputs={})

        assert task.user_prompt == ""
        assert task.conversation is None
        assert task.context == {}
        assert task.changed_context == set()
        assert task.pre_inputs == {}
        assert task.post_inputs == {}
        assert task.agentic_inputs == {}
        assert task.run_id is None
        assert task.started_at is None
        assert task.ended_at is None

    def test_new_mutable_fields_are_per_instance(self) -> None:
        first = AgentTask(turns=[], inputs={})
        second = AgentTask(turns=[], inputs={})

        first.context["x"] = ContextVariable(name="x", value=1)
        first.changed_context.add("x")
        first.pre_inputs["a"] = 1
        first.post_inputs["b"] = 2
        first.agentic_inputs["c"] = 3

        assert second.context == {}
        assert second.changed_context == set()
        assert second.pre_inputs == {}
        assert second.post_inputs == {}
        assert second.agentic_inputs == {}

    @pytest.mark.parametrize(
        "task_cls", [ToolAgentTask, ScriptActAgentTask, PlanActTask, ReActTask, ThinkingTask]
    )
    def test_subclass_tasks_inherit_new_fields(self, task_cls: type[AgentTask]) -> None:
        task = task_cls(turns=[], inputs={})

        assert task.user_prompt == ""
        assert task.context == {}
        assert task.conversation is None
