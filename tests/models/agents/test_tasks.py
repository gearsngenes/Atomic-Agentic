from __future__ import annotations

import pytest

from atomic_agentic.models.agents.blackboard_models import Statement, Step
from atomic_agentic.models.agents.tasks import (
    AgentTask,
    ToolAgentTask,
    PlanActTask,
    ReActTask,
    ScratchpadTask,
    ThinkingTask,
)
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.models.agents.context import ContextVariable


def _statement(kind: str = "assign", target: str | None = "x", *steps: Step, status: str = "ran") -> Statement:
    """A one-step statement: a tool step by default, ``status`` applied to every step."""
    from dataclasses import replace

    steps = steps or (Step(kind="tool", tool="add"),)
    return Statement(
        ordinal=0, source="", kind=kind, target=target,
        steps=tuple(replace(step, status=status) for step in steps),
    )


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
        # from completed/failed_statements via actions_used.
        assert "tool_calls_used" not in ToolAgentTask.__dataclass_fields__
        assert isinstance(ToolAgentTask.__dict__["tool_calls_used"], property)

    def test_tool_calls_used_counts_actions_over_both_lists(self) -> None:
        task = ToolAgentTask(turns=[], inputs={}, user_prompt="hi")
        composed = _statement(
            "assign", "x",
            Step(kind="tool", tool="inner", name="_SUB_0"), Step(kind="tool", tool="add"),
        )
        task.completed.append(composed)                                   # 2 tool steps
        task.completed.append(_statement("assign", "y", Step(kind="eval", args=())))  # call-free: 1
        task.completed.append(_statement("return", None, Step(kind="eval", args=())))  # return: 0
        task.failed_statements.append(_statement("call", None, status="failed"))  # a failed call counts: 1

        assert task.tool_calls_used == 4

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

        first.completed.append(_statement())
        first.failed_statements.append(_statement())
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
        assert task.needs_repair is False
        assert task.repair_rounds_used == 0
        assert task.repair_batch_start == 0
        # Inherited from ToolAgentTask, not redeclared here.
        assert task.completed == []
        assert task.failed_statements == []
        assert task.regenerations_used == 0
        assert task.tool_calls_used == 0
        assert not hasattr(task, "cache")
        assert not hasattr(task, "constant_values")

    def test_no_batch_or_resolved_args_fields(self) -> None:
        # Concurrency comes from plan_waves over statements, and resolved
        # arguments are never stored.
        task = PlanActTask(turns=[], inputs={}, user_prompt="hi")

        assert not hasattr(task, "batch_counter")
        assert not hasattr(task, "resolved_args")

    def test_no_pre_repair_rework_fields(self) -> None:
        # planning_rounds_used/continue_planning/continuation_note were all
        # retired by the repair-on-failure rework -- needs_repair/
        # repair_rounds_used/repair_batch_start replace them.
        task = PlanActTask(turns=[], inputs={}, user_prompt="hi")

        assert not hasattr(task, "planning_rounds_used")
        assert not hasattr(task, "continue_planning")
        assert not hasattr(task, "continuation_note")

    def test_default_factories_are_independent_per_instance(self) -> None:
        first = PlanActTask(turns=[], inputs={}, user_prompt="hi")
        second = PlanActTask(turns=[], inputs={}, user_prompt="hi")

        first.completed.append(_statement())
        first.failed_statements.append(_statement())
        first.pending.append(_statement(status="not_run"))

        assert second.completed == []
        assert second.failed_statements == []
        assert second.pending == []

    def test_needs_repair_round_trips(self) -> None:
        task = PlanActTask(turns=[], inputs={}, user_prompt="hi")

        task.needs_repair = True

        assert task.needs_repair is True


class TestReActTask:
    def test_inherits_tool_agent_task_fields(self) -> None:
        task = ReActTask(turns=[], inputs={}, user_prompt="hi")

        assert isinstance(task, ToolAgentTask)

    def test_added_field_defaults(self) -> None:
        task = ReActTask(turns=[], inputs={}, user_prompt="hi")

        assert task.statement is None
        assert task.failure_note == ""
        assert not hasattr(task, "cache")
        assert not hasattr(task, "constant_values")

    def test_statement_holds_a_statement(self) -> None:
        statement = _statement(status="not_run")
        task = ReActTask(turns=[], inputs={}, user_prompt="hi", statement=statement)

        assert task.statement is statement

    def test_no_json_only_fields(self) -> None:
        task = ReActTask(turns=[], inputs={}, user_prompt="hi")

        for removed in (
            "generated_step", "resolved_args", "last_call_failed", "pending", "source", "history",
        ):
            assert not hasattr(task, removed)

    def test_no_step_meta_or_next_step_index_fields(self) -> None:
        # No fixed-size preallocated board/observability-decay window.
        task = ReActTask(turns=[], inputs={}, user_prompt="hi")

        assert not hasattr(task, "next_step_index")
        assert not hasattr(task, "step_meta")

    def test_completed_is_independent_per_instance(self) -> None:
        first = ReActTask(turns=[], inputs={}, user_prompt="hi")
        second = ReActTask(turns=[], inputs={}, user_prompt="hi")

        first.completed.append(_statement())

        assert second.completed == []


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
        assert not hasattr(task, "started_at")
        assert not hasattr(task, "ended_at")

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
        "task_cls", [ToolAgentTask, PlanActTask, ReActTask, ThinkingTask]
    )
    def test_subclass_tasks_inherit_new_fields(self, task_cls: type[AgentTask]) -> None:
        task = task_cls(turns=[], inputs={})

        assert task.user_prompt == ""
        assert task.context == {}
        assert task.conversation is None
