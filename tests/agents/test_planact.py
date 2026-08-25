from __future__ import annotations

import pytest
import json
import asyncio

from .conftest import (
    FakeLLMEngine,
    make_planact_agent,
    make_react_agent,
    register_math_tools,
    react_step_json,
)

from atomic_agentic.agents.toolagent import return_tool
from atomic_agentic.agents.planact import PlanActAgent
from atomic_agentic.models.agents.blackboard_models import BlackboardSlot
from atomic_agentic.models.agents.tasks import PlanActTask
from atomic_agentic.exceptions import (
    ToolAgentError,
    ToolInvocationError,
)
from atomic_agentic.constants.core import NO_VAL


class TestPlanActAgent:
    def test_invokes_planned_tools_and_returns_value(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"step": 0, "tool": "Tool.tests.add", "args": {{"x": 2, "y": 3}}}},
                  {{"step": 1, "tool": "Tool.tests.multiply", "args": {{"x": "|STEP.0|", "y": 10}}}},
                  {{"step": 2, "tool": "{return_tool.full_name}", "args": {{"val": "|STEP.1|"}}}}
                ]
                """
            ]
        )

        result = agent.invoke({"prompt": "run plan"})

        assert result.result == 50

    def test_auto_appends_return_none_when_plan_has_no_return(self) -> None:
        agent = make_planact_agent(
            [
                """
                [
                  {"step": 0, "tool": "Tool.tests.add", "args": {"x": 2, "y": 3}}
                ]
                """
            ]
        )

        result = agent.invoke({"prompt": "run plan"})

        assert result.result is None

    def test_moves_return_step_to_end(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"step": 0, "tool": "{return_tool.full_name}", "args": {{"val": "|STEP.0|"}}}},
                  {{"step": 1, "tool": "Tool.tests.add", "args": {{"x": 1, "y": 2}}}}
                ]
                """
            ]
        )

        result = agent.invoke({"prompt": "run plan"})

        assert result.result == 3

    def test_executes_independent_steps_in_same_batch(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"step": 0, "tool": "Tool.tests.add", "args": {{"x": 1, "y": 2}}}},
                  {{"step": 1, "tool": "Tool.tests.multiply", "args": {{"x": 3, "y": 4}}}},
                  {{"step": 2, "tool": "{return_tool.full_name}", "args": {{"val": ["|STEP.0|", "|STEP.1|"]}}}}
                ]
                """
            ]
        )

        task = agent._initialize_task(turns=[], prompt="plan", inputs={})
        task = agent.think(task)
        task = agent.prepare(task)

        assert task.batches == [[0, 1], [2]]

    def test_rejects_multiple_return_steps(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"step": 0, "tool": "{return_tool.full_name}", "args": {{"val": 1}}}},
                  {{"step": 1, "tool": "{return_tool.full_name}", "args": {{"val": 2}}}}
                ]
                """
            ]
        )

        with pytest.raises(ToolAgentError, match="multiple return"):
            agent.invoke({"prompt": "run plan"})

    def test_rejects_plan_output_that_is_not_a_json_array(self) -> None:
        agent = make_planact_agent(['{"not": "a list"}'])

        with pytest.raises(ToolAgentError, match="plan must be a non-empty JSON array"):
            agent.invoke({"prompt": "run plan"})

    def test_rejects_plan_item_that_is_not_a_json_object(self) -> None:
        agent = make_planact_agent(["[1, 2, 3]"])

        with pytest.raises(ToolAgentError, match="must be a JSON object"):
            agent.invoke({"prompt": "run plan"})

    def test_rejects_unknown_tool_in_plan(self) -> None:
        agent = make_planact_agent(
            [
                """
                [
                  {"step": 0, "tool": "Tool.tests.missing", "args": {}}
                ]
                """
            ]
        )

        with pytest.raises(ToolAgentError, match="unknown tool"):
            agent.invoke({"prompt": "run plan"})

    def test_rejects_plan_exceeding_tool_calls_limit(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"step": 0, "tool": "Tool.tests.add", "args": {{"x": 1, "y": 2}}}},
                  {{"step": 1, "tool": "Tool.tests.multiply", "args": {{"x": 3, "y": 4}}}},
                  {{"step": 2, "tool": "{return_tool.full_name}", "args": {{"val": "|STEP.1|"}}}}
                ]
                """
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="tool_calls_limit"):
            agent.invoke({"prompt": "run plan"})

    def test_rejects_out_of_range_cache_reference(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"step": 0, "tool": "{return_tool.full_name}", "args": {{"val": "|CACHE.0|"}}}}
                ]
                """
            ]
        )

        with pytest.raises(ToolAgentError, match="cache indices that do not exist"):
            agent.invoke({"prompt": "run plan"})

    def test_rejects_future_step_dependency(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"step": 0, "tool": "Tool.tests.add", "args": {{"x": "|STEP.1|", "y": 2}}}},
                  {{"step": 1, "tool": "Tool.tests.multiply", "args": {{"x": 3, "y": 4}}}},
                  {{"step": 2, "tool": "{return_tool.full_name}", "args": {{"val": "|STEP.1|"}}}}
                ]
                """
            ]
        )

        with pytest.raises(ToolAgentError, match="invalid step dependencies"):
            agent.invoke({"prompt": "run plan"})

    def test_context_enabled_can_reference_cached_result_on_next_invoke(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"step": 0, "tool": "Tool.tests.add", "args": {{"x": 2, "y": 3}}}},
                  {{"step": 1, "tool": "{return_tool.full_name}", "args": {{"val": "|STEP.0|"}}}}
                ]
                """,
                f"""
                [
                  {{"step": 0, "tool": "{return_tool.full_name}", "args": {{"val": "|CACHE.0|"}}}}
                ]
                """,
            ],
            context_enabled=True,
        )

        assert agent.invoke({"prompt": "first"}).result == 5
        assert agent.invoke({"prompt": "second"}).result == 5

    def test_compile_batches_isolates_return_step(self) -> None:
        agent = make_planact_agent(["[]"])
        planned_slots = [
            BlackboardSlot(
                step=0,
                tool="Tool.tests.add",
                args={"x": 1, "y": 2},
                status="planned",
                step_dependencies=(),
            ),
            BlackboardSlot(
                step=1,
                tool="Tool.tests.multiply",
                args={"x": 3, "y": 4},
                status="planned",
                step_dependencies=(),
            ),
            BlackboardSlot(
                step=2,
                tool=return_tool.full_name,
                args={"val": "|STEP.1|"},
                status="planned",
                step_dependencies=(0, 1),
            ),
        ]

        batches = agent._compile_batches_from_deps(
            planned_slots=planned_slots,
            return_idx=2,
        )

        assert batches == [[0, 1], [2]]

    def test_think_generates_planned_slot_metadata(self) -> None:
        # think() generates + validates the whole plan onto
        # task.generated_plan; running_blackboard/batches don't exist yet --
        # those are prepare()'s job (compiled on its first call).
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"step": 0, "tool": "Tool.tests.add", "args": {{"x": 1, "y": 2}}}},
                  {{"step": 1, "tool": "Tool.tests.multiply", "args": {{"x": "|STEP.0|", "y": 4}}, "await": 0}},
                  {{"step": 2, "tool": "{return_tool.full_name}", "args": {{"val": "|STEP.1|"}}}}
                ]
                """
            ]
        )

        task = agent._initialize_task(turns=[], prompt="plan", inputs={})
        task = agent.think(task)
        plan = task.generated_plan

        assert plan[0].status == "planned"
        assert plan[0].step_dependencies == ()
        assert plan[1].status == "planned"
        assert plan[1].step_dependencies == (0,)
        assert plan[1].await_step == 0
        assert plan[2].tool == return_tool.full_name
        assert plan[2].step_dependencies == (0, 1)

    def test_prepare_marks_batch_zero_prepared(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"step": 0, "tool": "Tool.tests.add", "args": {{"x": 1, "y": 2}}}},
                  {{"step": 1, "tool": "{return_tool.full_name}", "args": {{"val": "|STEP.0|"}}}}
                ]
                """
            ]
        )
        task = agent._initialize_task(turns=[], prompt="plan", inputs={})
        task = agent.think(task)

        updated = agent.prepare(task)

        assert updated.prepared_steps == [0]
        assert updated.running_blackboard[0].status == "prepared"
        assert updated.running_blackboard[0].is_prepared() is True
        assert updated.running_blackboard[0].resolved_args == {"x": 1, "y": 2}

    def test_accepts_plan_missing_step_key_and_normalizes(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"tool": "Tool.tests.add", "args": {{"x": 1, "y": 2}}}},
                  {{"tool": "{return_tool.full_name}", "args": {{"val": "|STEP.0|"}}}}
                ]
                """
            ]
        )

        task = agent._initialize_task(turns=[], prompt="plan", inputs={})
        task = agent.think(task)
        plan = task.generated_plan

        assert [slot.step for slot in plan] == [0, 1]
        assert plan[0].tool == "Tool.tests.add"
        assert plan[1].tool == return_tool.full_name

    def test_accepts_non_sequential_plan_step_and_normalizes(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"step": 99, "tool": "Tool.tests.add", "args": {{"x": 1, "y": 2}}}},
                  {{"step": 42, "tool": "{return_tool.full_name}", "args": {{"val": "|STEP.0|"}}}}
                ]
                """
            ]
        )

        task = agent._initialize_task(turns=[], prompt="plan", inputs={})
        task = agent.think(task)
        plan = task.generated_plan

        assert [slot.step for slot in plan] == [0, 1]
        assert plan[0].tool == "Tool.tests.add"
        assert plan[1].tool == return_tool.full_name

    def test_rejects_await_on_return_step(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"step": 0, "tool": "{return_tool.full_name}", "args": {{"val": 1}}, "await": 0}}
                ]
                """
            ]
        )

        with pytest.raises(ToolAgentError, match="return step and must not include"):
            agent.invoke({"prompt": "run plan"})

    def test_async_invoke_executes_plan_and_returns_value(self) -> None:
        agent = make_planact_agent(
            [
                f"""
                [
                  {{"step": 0, "tool": "Tool.tests.add", "args": {{"x": 2, "y": 3}}}},
                  {{"step": 1, "tool": "Tool.tests.multiply", "args": {{"x": "|STEP.0|", "y": 10}}}},
                  {{"step": 2, "tool": "{return_tool.full_name}", "args": {{"val": "|STEP.1|"}}}}
                ]
                """
            ]
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run plan"}))

        assert result.result == 50

    def test_llm_record_system_prompt_name_is_plan_first(self) -> None:
        agent = make_planact_agent(
            [
                f'[{{"step": 0, "tool": "{return_tool.full_name}", "args": {{"val": 42}}}}]'
            ]
        )

        agent.invoke({"prompt": "run"})

        for rec in agent.records[-1].llm_records:
            assert rec.system_prompt_name == "plan_first"


class TestPlanActThinkPrepareHandoff:
    """New 1f coverage: think() generates+validates the plan once (no-op on
    later calls), storing it on task.generated_plan; prepare()'s first call
    compiles it into batches (task.batches), later calls just resolve/
    advance the cursor -- compilation itself never repeats."""

    def test_think_populates_generated_plan_and_is_no_op_after(self) -> None:
        agent = make_planact_agent(
            [f'[{{"tool": "{return_tool.full_name}", "args": {{"val": 1}}}}]']
        )
        task = agent._initialize_task(turns=[], prompt="run", inputs={})
        assert task.generated_plan is NO_VAL

        task = agent.think(task)
        first_plan = task.generated_plan
        assert first_plan is not NO_VAL

        task = agent.think(task)  # second call: no-op, same object
        assert task.generated_plan is first_plan

    def test_prepare_compiles_batches_only_on_first_call(self) -> None:
        agent = make_planact_agent(
            [
                json.dumps([
                    {"tool": "Tool.tests.add", "args": {"x": 1, "y": 2}},
                    {"tool": return_tool.full_name, "args": {"val": "|STEP.0|"}},
                ])
            ]
        )
        task = agent._initialize_task(turns=[], prompt="run", inputs={})
        task = agent.think(task)
        assert task.batches == []

        task = agent.prepare(task)
        first_batches = task.batches
        assert first_batches

        task = agent.act(task)
        task = agent.prepare(task)  # second batch -- compile not repeated
        assert task.batches is first_batches


class TestPlanActBlackboardInvariants:
    """New 1f coverage: across a multi-batch run, running_blackboard[i].step
    == i holds for every persisted slot and no slot is written twice --
    under both fail_fast settings."""

    def test_step_matches_index_and_no_double_write_fail_fast_true(self) -> None:
        agent = make_planact_agent(
            [
                json.dumps([
                    {"tool": "Tool.tests.add", "args": {"x": 1, "y": 2}},
                    {"tool": "Tool.tests.multiply", "args": {"x": "|STEP.0|", "y": 3}},
                    {"tool": return_tool.full_name, "args": {"val": "|STEP.1|"}},
                ])
            ]
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 9
        board = agent.blackboard
        assert len(board) == 3
        for i, slot in enumerate(board):
            assert slot.step == i
            assert slot.status == BlackboardSlot.EXECUTED

    def test_step_matches_index_and_no_double_write_fail_fast_false(self) -> None:
        agent = make_planact_agent(
            [
                json.dumps([
                    {"tool": "Tool.tests.fail_tool", "args": {}},
                    {"tool": "Tool.tests.add", "args": {"x": 1, "y": 2}},
                    {"tool": return_tool.full_name, "args": {"val": "|STEP.1|"}},
                ])
            ],
            fail_fast=False,
        )

        result = agent.invoke({"prompt": "run"})

        assert result.result == 3
        board = agent.blackboard
        assert len(board) == 3
        for i, slot in enumerate(board):
            assert slot.step == i
        assert board[0].status == BlackboardSlot.FAILED
        assert board[1].status == BlackboardSlot.EXECUTED
        assert board[2].status == BlackboardSlot.EXECUTED

    def test_double_prepare_without_act_is_unreachable_via_the_loop_and_breaks_by_hand(self) -> None:
        """Demonstrates why the base loop's fixed think -> prepare -> act
        ordering is what actually prevents double-prepare-without-act, not
        a guard inside prepare() itself (the old re-entry check was dropped
        in 1c as dead -- the base loop's external sequencing makes this call
        shape unreachable through normal invoke()/async_invoke() execution).
        A hand-rolled re-entry doesn't silently corrupt state -- it breaks
        loudly, because the second batch's return step depends on the first
        batch's results, which were never actually executed (act() never
        ran in between): _resolve_placeholders raises "not executed"."""
        agent = make_planact_agent(
            [
                json.dumps([
                    {"tool": "Tool.tests.add", "args": {"x": 1, "y": 2}},
                    {"tool": "Tool.tests.multiply", "args": {"x": 3, "y": 4}},
                    {"tool": return_tool.full_name, "args": {"val": "|STEP.1|"}},
                ])
            ]
        )
        task = agent._initialize_task(turns=[], prompt="run", inputs={})
        task = agent.think(task)

        task = agent.prepare(task)  # batch 0 prepared: steps [0, 1]
        first_batch_prepared = list(task.prepared_steps)
        assert first_batch_prepared == [0, 1]

        with pytest.raises(ToolAgentError, match="not executed"):
            agent.prepare(task)  # hand-rolled re-entry, no intervening act()


# ── TestPlanActGenerationRetry ─────────────────────────────────────────────────

class TestPlanActGenerationRetry:
    """Retry loop in _generate_plan/_agenerate_plan: budget, feedback, LLMRecord accumulation."""

    VALID_PLAN = json.dumps([
        {"step": 0, "tool": "Tool.tests.add", "args": {"x": 1, "y": 2}},
        {"step": 1, "tool": return_tool.full_name, "args": {"val": "|STEP.0|"}},
    ])

    VALID_PLAN_RETURN_ONLY = json.dumps([
        {"step": 0, "tool": return_tool.full_name, "args": {"val": 42}},
    ])

    INVALID_JSON = "this is not json at all"

    INVALID_PLAN_WRONG_TOOL = json.dumps([
        {"step": 0, "tool": "Tool.tests.nonexistent", "args": {}},
        {"step": 1, "tool": return_tool.full_name, "args": {"val": 0}},
    ])

    def test_zero_retries_raises_on_first_bad_json(self) -> None:
        agent = make_planact_agent([self.INVALID_JSON])
        with pytest.raises(ToolAgentError):
            agent.invoke({"prompt": "run"})

    def test_zero_retries_emits_one_llm_call_before_raise(self) -> None:
        engine = FakeLLMEngine([self.INVALID_JSON])
        agent = PlanActAgent(
            name="tests",
            namespace="tests",
            description=".",
            llm_engine=engine,
            generation_retries=0,
        )
        with pytest.raises(ToolAgentError):
            agent.invoke({"prompt": "run"})
        assert len(engine.calls) == 1

    def test_json_error_retry_succeeds_on_second_call(self) -> None:
        agent = make_planact_agent(
            [self.INVALID_JSON, self.VALID_PLAN],
            generation_retries=1,
        )
        result = agent.invoke({"prompt": "run"})
        assert result.result == 3

    def test_json_error_retry_stores_two_llm_records(self) -> None:
        engine = FakeLLMEngine([self.INVALID_JSON, self.VALID_PLAN_RETURN_ONLY])
        agent = PlanActAgent(
            name="tests",
            namespace="tests",
            description=".",
            llm_engine=engine,
            generation_retries=1,
        )
        agent.invoke({"prompt": "run"})
        llm_records = agent.records[-1].llm_records
        assert len(llm_records) == 2
        assert len(llm_records[0].messages) == 1
        # Self-contained convention: retry's messages = full task_messages so
        # far (1 original decompose-instruction + 2 injected feedback), not
        # just the 2 new ones.
        assert len(llm_records[1].messages) == 3

    def test_spec_error_retry_succeeds_on_second_call(self) -> None:
        agent = make_planact_agent(
            [self.INVALID_PLAN_WRONG_TOOL, self.VALID_PLAN],
            generation_retries=1,
        )
        result = agent.invoke({"prompt": "run"})
        assert result.result == 3

    def test_spec_error_retry_stores_two_llm_records(self) -> None:
        engine = FakeLLMEngine([self.INVALID_PLAN_WRONG_TOOL, self.VALID_PLAN_RETURN_ONLY])
        agent = PlanActAgent(
            name="tests",
            namespace="tests",
            description=".",
            llm_engine=engine,
            generation_retries=1,
        )
        register_math_tools(agent)  # type: ignore[arg-type]
        agent.invoke({"prompt": "run"})
        llm_records = agent.records[-1].llm_records
        assert len(llm_records) == 2
        assert len(llm_records[0].messages) == 1
        # Self-contained convention: retry's messages = full task_messages so
        # far (1 original decompose-instruction + 2 injected feedback), not
        # just the 2 new ones.
        assert len(llm_records[1].messages) == 3

    def test_budget_exhausted_raises_after_all_attempts(self) -> None:
        agent = make_planact_agent(
            [self.INVALID_JSON, self.INVALID_JSON],
            generation_retries=1,
        )
        with pytest.raises(ToolAgentError):
            agent.invoke({"prompt": "run"})

    def test_budget_exhausted_records_all_llm_calls_before_raise(self) -> None:
        engine = FakeLLMEngine([self.INVALID_JSON, self.INVALID_JSON])
        agent = PlanActAgent(
            name="tests",
            namespace="tests",
            description=".",
            llm_engine=engine,
            generation_retries=1,
        )
        with pytest.raises(ToolAgentError):
            agent.invoke({"prompt": "run"})
        assert len(engine.calls) == 2

    def test_json_feedback_appended_to_working_messages(self) -> None:
        engine = FakeLLMEngine([self.INVALID_JSON, self.VALID_PLAN_RETURN_ONLY])
        agent = PlanActAgent(
            name="tests",
            namespace="tests",
            description=".",
            llm_engine=engine,
            generation_retries=1,
        )
        agent.invoke({"prompt": "run"})
        assert len(engine.calls[1]) > len(engine.calls[0])
        last_msg = engine.calls[1][-1]
        assert last_msg["role"] == "user"
        assert "could not be parsed" in last_msg["content"]

    def test_spec_feedback_contains_reserialised_plan_not_resolved_args(self) -> None:
        engine = FakeLLMEngine([self.INVALID_PLAN_WRONG_TOOL, self.VALID_PLAN_RETURN_ONLY])
        agent = PlanActAgent(
            name="tests",
            namespace="tests",
            description=".",
            llm_engine=engine,
            generation_retries=1,
        )
        register_math_tools(agent)  # type: ignore[arg-type]
        agent.invoke({"prompt": "run"})
        last_msg = engine.calls[1][-1]
        assert last_msg["role"] == "user"
        assert "unknown tool" in last_msg["content"]
        assert "resolved_args" not in last_msg["content"]


class TestCascadeFailedPropagation:
    """
    Integration tests for cascade FAILED propagation in PlanActAgent.

    When a tool step fails (fail_fast=False), any later step whose args
    reference it via |STEP.N| placeholders is cascade-marked FAILED
    instead of raising. The return tool always raises when any of its
    arg dependencies failed.
    """

    def test_dependent_step_is_cascade_failed_not_raised(self) -> None:
        """Non-return step with a failed dep is cascade-marked FAILED; run succeeds."""
        agent = make_planact_agent(
            [
                json.dumps([
                    {"tool": "Tool.tests.fail_tool", "args": {}},
                    {"tool": "Tool.tests.add", "args": {"x": "|STEP.0|", "y": 2}},
                    {"tool": return_tool.full_name, "args": {"val": 99}},
                ])
            ],
            fail_fast=False,
        )

        result = agent.invoke({"prompt": "cascade test"})

        assert result.result == 99
        board = agent.blackboard
        assert board[0].status == BlackboardSlot.FAILED
        assert board[1].status == BlackboardSlot.FAILED
        assert "dependency" in str(board[1].error).lower() or "skipped" in str(board[1].error).lower()
        assert board[2].status == BlackboardSlot.EXECUTED

    def test_cascade_exception_records_includes_both_slots(self) -> None:
        """exception_records captures the execution failure and the cascade failure."""
        agent = make_planact_agent(
            [
                json.dumps([
                    {"tool": "Tool.tests.fail_tool", "args": {}},
                    {"tool": "Tool.tests.add", "args": {"x": "|STEP.0|", "y": 2}},
                    {"tool": return_tool.full_name, "args": {"val": 99}},
                ])
            ],
            fail_fast=False,
        )

        result = agent.invoke({"prompt": "cascade exception_records"})

        assert result.result == 99
        assert len(result.exception_records) == 2
        indices = [idx for idx, _ in result.exception_records]
        assert 0 in indices
        assert 1 in indices

    def test_return_step_with_failed_dep_raises(self) -> None:
        """Return step depending on a FAILED step raises ToolAgentError."""
        agent = make_planact_agent(
            [
                json.dumps([
                    {"tool": "Tool.tests.fail_tool", "args": {}},
                    {"tool": return_tool.full_name, "args": {"val": "|STEP.0|"}},
                ])
            ],
            fail_fast=False,
        )

        with pytest.raises(ToolAgentError, match="return step"):
            agent.invoke({"prompt": "return cascade raise"})

    def test_chain_cascade_return_raises(self) -> None:
        """Cascade chain: step 0 fails -> step 1 cascade FAILED -> return depends on step 1 -> raises."""
        agent = make_planact_agent(
            [
                json.dumps([
                    {"tool": "Tool.tests.fail_tool", "args": {}},
                    {"tool": "Tool.tests.add", "args": {"x": "|STEP.0|", "y": 2}},
                    {"tool": return_tool.full_name, "args": {"val": "|STEP.1|"}},
                ])
            ],
            fail_fast=False,
        )

        with pytest.raises(ToolAgentError, match="return step"):
            agent.invoke({"prompt": "chain cascade"})

    def test_cascade_does_not_affect_independent_steps(self) -> None:
        """Steps with no dep on the failed step execute normally; run succeeds."""
        agent = make_planact_agent(
            [
                json.dumps([
                    {"tool": "Tool.tests.fail_tool", "args": {}},
                    {"tool": "Tool.tests.add", "args": {"x": 3, "y": 4}},
                    {"tool": return_tool.full_name, "args": {"val": "|STEP.1|"}},
                ])
            ],
            fail_fast=False,
        )

        result = agent.invoke({"prompt": "independent step"})

        assert result.result == 7
        board = agent.blackboard
        assert board[0].status == BlackboardSlot.FAILED
        assert board[1].status == BlackboardSlot.EXECUTED
        assert board[2].status == BlackboardSlot.EXECUTED

    def test_fail_fast_true_does_not_cascade_raises_immediately(self) -> None:
        """With fail_fast=True (default), any failure raises ToolInvocationError immediately."""
        agent = make_planact_agent(
            [
                json.dumps([
                    {"tool": "Tool.tests.fail_tool", "args": {}},
                    {"tool": "Tool.tests.add", "args": {"x": "|STEP.0|", "y": 2}},
                    {"tool": return_tool.full_name, "args": {"val": 99}},
                ])
            ],
            fail_fast=True,
        )

        with pytest.raises(ToolInvocationError):
            agent.invoke({"prompt": "fail_fast=True raises"})


# ── TestFailedCacheRefValidation ───────────────────────────────────────────────

class TestFailedCacheRefValidation:
    """
    Tests for FAILED cache-ref detection in _validate_planned_slots (PlanAct)
    and _validate_generation_output (ReAct).

    Requires two-invoke sequences: first invoke leaves a FAILED slot in the
    persisted cache (fail_fast=False, context_enabled=True), then a second
    invoke's plan/step references that FAILED cache slot.
    """

    def test_validate_planned_slots_rejects_failed_cache_ref(self) -> None:
        """PlanAct: plan referencing a FAILED cache slot raises at validation time."""
        agent = make_planact_agent(
            [
                # First invoke: step 0 fails; return is independent.
                json.dumps([
                    {"tool": "Tool.tests.fail_tool", "args": {}},
                    {"tool": return_tool.full_name, "args": {"val": 1}},
                ]),
                # Second invoke: plan references |CACHE.0| which is FAILED.
                json.dumps([
                    {"tool": return_tool.full_name, "args": {"val": "|CACHE.0|"}},
                ]),
            ],
            fail_fast=False,
            context_enabled=True,
        )
        # First invoke succeeds (fail_fast=False, return is independent of failed step).
        result1 = agent.invoke({"prompt": "first run"})
        assert result1.result == 1
        # Second invoke: plan references a FAILED cache slot -> raises at validation.
        with pytest.raises(ToolAgentError, match="failed in this conversation"):
            agent.invoke({"prompt": "second run"})

    def test_validate_planned_slots_rejects_stale_step_ref_in_return(self) -> None:
        """PlanAct: return step using |STEP.N| where N >= plan length raises at validation.

        Reproduces the bug where _normalize_planned_slots overwrites return-slot
        step_dependencies to all-prior-steps, masking a stale global step index
        in the args that then blows up at _resolve_placeholders during execution.
        """
        # First invoke produces step 0 in the blackboard (global index 0).
        # Second invoke's return step mistakenly references |STEP.1| (index 1 in
        # a 1-step plan -- out of range; only |STEP.0| is valid).
        agent = make_planact_agent(
            [
                # First invoke: one real step + return.
                json.dumps([
                    {"tool": "Tool.tests.add", "args": {"x": 1, "y": 2}},
                    {"tool": return_tool.full_name, "args": {"val": "|STEP.0|"}},
                ]),
                # Second invoke: single return step uses stale index 1 (out of range).
                json.dumps([
                    {"tool": return_tool.full_name, "args": {"val": "|STEP.1|"}},
                ]),
            ],
            context_enabled=True,
        )
        result1 = agent.invoke({"prompt": "first run"})
        assert result1.result == 3
        # Second invoke: return step has |STEP.1| in a 1-slot plan (only index 0 valid).
        with pytest.raises(ToolAgentError, match="return step has invalid step references"):
            agent.invoke({"prompt": "second run"})

    def test_process_next_step_output_rejects_failed_cache_ref(self) -> None:
        """ReAct: a step referencing a FAILED cache slot raises at validation time."""
        agent = make_react_agent(
            [
                # First invoke: step 0 fails; step 1 is return with independent val.
                react_step_json(step=0, tool="Tool.tests.fail_tool", args={}),
                react_step_json(step=1, tool=return_tool.full_name, args={"val": 1}, duration=0),
                # Second invoke: step references |CACHE.0| (FAILED cache slot).
                react_step_json(step=0, tool="Tool.tests.add", args={"x": "|CACHE.0|", "y": 1}),
            ],
            tool_calls_limit=2,
            fail_fast=False,
            context_enabled=True,
        )
        result1 = agent.invoke({"prompt": "first run"})
        assert result1.result == 1
        with pytest.raises(ToolAgentError, match="failed in this conversation"):
            agent.invoke({"prompt": "second run"})


class TestAwaitFieldKeyInErrors:
    """B-5: _validate_tool_step_dict uses the LLM-facing 'await' key, not internal 'await_step'."""

    def test_invalid_await_type_error_uses_await_key(self) -> None:
        """_validate_tool_step_dict error for bad 'await' value names 'await', not 'await_step'."""
        agent = make_planact_agent([], context_enabled=False)
        # _validate_generation_output calls _validate_tool_step_dict; the await
        # type error fires before tool-existence lookup so no tools need to be
        # registered.
        plan = [{"tool": return_tool.full_name, "args": {"val": None}, "await": "not_an_int"}]
        task = PlanActTask(turns=[], inputs={}, user_prompt="run", system_prompt_name="tool_instructions")
        result = agent._validate_generation_output(json.dumps(plan), task=task)
        assert isinstance(result, str)
        assert "'await'" in result
        assert "'await_step'" not in result
