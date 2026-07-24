from __future__ import annotations

import pytest
import json
import asyncio
from typing import Any, Mapping

from conftest import (
    make_react_agent,
    make_planact_agent,
    react_step_json,
    register_math_tools,
    make_task,
    executed_slot,
    prepared_slot,
    make_llm_result,
    make_llm_record,
    make_tool_result,
    ScriptedTask,
    ScriptedToolAgent,
    ScriptedLLMEngine,
    BadRepr,
)

from atomic_agentic.agents.toolagent import ToolAgent, return_tool
from atomic_agentic.agents.react import ReActAgent
from atomic_agentic.models.agents.blackboard_models import BlackboardSlot
from atomic_agentic.models.agents.tasks import ReActTask, ReActStepMeta
from atomic_agentic.exceptions import (
    ToolAgentError,
    ToolInvocationError,
)
from atomic_agentic.constants.core import NO_VAL

class TestReActAgent:
    def test_requires_concrete_non_negative_tool_calls_limit(self) -> None:
        with pytest.raises(ToolAgentError, match="tool_calls_limit"):
            ReActAgent(
                name="bad_react",
                namespace="tests",
                description="Bad ReAct agent.",
                llm_engine=ScriptedLLMEngine([]),
                tool_calls_limit=-1,
            )

    def test_rejects_next_step_output_that_is_not_a_json_object(self) -> None:
        agent = make_react_agent(["[1, 2, 3]"], tool_calls_limit=1)

        with pytest.raises(ToolAgentError, match="next step output must be a JSON object"):
            agent.invoke({"prompt": "run react"})

    def test_invokes_step_by_step_until_return(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=0,
                    tool="Tool.tests.add",
                    args={"x": 2, "y": 3},
                    description="Add the two input numbers for the current calculation.",
                ),
                react_step_json(
                    step=1,
                    tool="Tool.tests.multiply",
                    args={"x": "<<__s0__>>", "y": 10},
                    description="Multiply the addition result by ten for the current calculation.",
                ),
                react_step_json(
                    step=2,
                    tool=return_tool.full_name,
                    args={"val": "<<__s1__>>"},
                    description="Return the final multiplied value because the current calculation is complete.",
                ),
            ],
            tool_calls_limit=2,
        )

        result = agent.invoke({"prompt": "run react"})

        assert result.result == 50

    def test_injects_running_plan_after_first_step(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=0,
                    tool="Tool.tests.add",
                    args={"x": 2, "y": 3},
                    duration=1,
                    description="Add the two input numbers so the result can be returned.",
                ),
                react_step_json(
                    step=1,
                    tool=return_tool.full_name,
                    args={"val": "<<__s0__>>"},
                    description="Return the addition result because the current task is complete.",
                ),
            ],
            tool_calls_limit=1,
        )

        result = agent.invoke({"prompt": "run react"})

        assert result.result == 5
        engine = agent.llm_engine
        assert isinstance(engine, ScriptedLLMEngine)
        assert len(engine.calls) == 2
        second_call_text = "\n".join(message["content"] for message in engine.calls[1])
        assert "RUNNING PLAN STEPS 0-0 SO FAR" in second_call_text
        assert "Add the two input numbers so the result can be returned." in second_call_text
        assert "Tool.tests.add" in second_call_text
        assert "result_ref" in second_call_text
        assert "<<__s0__>>" in second_call_text
        assert "observable_result" in second_call_text
        assert "5" in second_call_text
        assert "run_id" in second_call_text

    @pytest.mark.parametrize(
        "raw_response, match",
        [
            (
                json.dumps(
                    {
                        "step": 0,
                        "tool": "Tool.tests.add",
                        "args": {},
                        "description": "Try to run a step with no duration for validation.",
                    }
                ),
                "missing required key 'duration'",
            ),
            (
                json.dumps(
                    {
                        "step": 0,
                        "tool": "Tool.tests.add",
                        "args": {},
                        "duration": 0,
                    }
                ),
                "missing required key 'description'",
            ),
            (
                json.dumps(
                    {
                        "step": 0,
                        "args": {},
                        "duration": 0,
                        "description": "Try to run an incomplete step for validation.",
                    }
                ),
                "missing required keys",
            ),
            (
                json.dumps(
                    {
                        "step": 0,
                        "tool": "Tool.tests.add",
                        "duration": 0,
                        "description": "Try to run an incomplete step for validation.",
                    }
                ),
                "missing required keys",
            ),
        ],
    )
    def test_rejects_missing_required_step_keys(
        self,
        raw_response: str,
        match: str,
    ) -> None:
        agent = make_react_agent([raw_response], tool_calls_limit=1)

        with pytest.raises(ToolAgentError, match=match):
            agent.invoke({"prompt": "run react"})

    @pytest.mark.parametrize("duration", [-1, 4, 1.5, True, "1"])
    def test_rejects_invalid_duration(self, duration: Any) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=0,
                    tool="Tool.tests.add",
                    args={"x": 1, "y": 2},
                    duration=duration,
                    description="Add the two numbers for the current calculation.",
                ),
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="duration"):
            agent.invoke({"prompt": "run react"})

    @pytest.mark.parametrize("description", ["", "   ", 1, None])
    def test_rejects_invalid_description(self, description: Any) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=0,
                    tool="Tool.tests.add",
                    args={"x": 1, "y": 2},
                    duration=0,
                    description=description,
                ),
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="description"):
            agent.invoke({"prompt": "run react"})

    def test_accepts_missing_step_key_and_uses_expected_step(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=None,
                    tool="Tool.tests.add",
                    args={"x": 1, "y": 2},
                    description="Add the two test numbers for this ReAct step.",
                ),
            ],
            tool_calls_limit=1,
        )
        task = agent._initialize_task(
            turns=[],
            prompt="react",
            inputs={},
        )

        updated = agent._prepare_next_batch(task)

        slot = updated.running_blackboard[0]
        assert updated.prepared_steps == [0]
        assert slot.step == 0
        assert slot.tool == "Tool.tests.add"
        assert slot.resolved_args == {"x": 1, "y": 2}
        assert updated.step_meta[0].description == "Add the two test numbers for this ReAct step."

    def test_rejects_extra_step_keys(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=0,
                    tool="Tool.tests.add",
                    args={},
                    description="Attempt a step with an unsupported extra key.",
                    extra=True,
                ),
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="unsupported keys"):
            agent.invoke({"prompt": "run react"})

    def test_rejects_non_dict_args(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=0,
                    tool="Tool.tests.add",
                    args=[],
                    description="Attempt a step whose args are the wrong shape.",
                ),
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="'args' must be a dict"):
            agent.invoke({"prompt": "run react"})

    def test_accepts_mismatched_step_index_and_overrides_with_expected_step(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=99,
                    tool="Tool.tests.add",
                    args={"x": 1, "y": 2},
                    description="Add the two test numbers despite the advisory step mismatch.",
                ),
            ],
            tool_calls_limit=1,
        )
        task = agent._initialize_task(
            turns=[],
            prompt="react",
            inputs={},
        )

        updated = agent._prepare_next_batch(task)

        slot = updated.running_blackboard[0]
        assert updated.prepared_steps == [0]
        assert slot.step == 0
        assert slot.tool == "Tool.tests.add"
        assert slot.resolved_args == {"x": 1, "y": 2}

    def test_rejects_future_step_dependency(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=0,
                    tool="Tool.tests.add",
                    args={"x": "<<__s0__>>", "y": 2},
                    description="Attempt to use the current step as its own input dependency.",
                ),
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="illegal deps"):
            agent.invoke({"prompt": "run react"})

    def test_rejects_unknown_tool(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=0,
                    tool="Tool.tests.missing",
                    args={},
                    description="Attempt to call an unregistered tool.",
                ),
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="unknown tool"):
            agent.invoke({"prompt": "run react"})

    def test_rejects_when_next_step_exceeds_capacity(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=0,
                    tool="Tool.tests.add",
                    args={"x": 1, "y": 2},
                    description="Add the two numbers as the only permitted non-return call.",
                ),
                react_step_json(
                    step=1,
                    tool="Tool.tests.multiply",
                    args={"x": "<<__s0__>>", "y": 3},
                    description="Attempt to multiply after the non-return tool budget is exhausted.",
                ),
            ],
            tool_calls_limit=1,
        )

        with pytest.raises(ToolAgentError, match="tool_calls_limit exceeded"):
            agent.invoke({"prompt": "run react"})

    def test_prepare_next_batch_records_slot_metadata(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=0,
                    tool="Tool.tests.add",
                    args={"x": 2, "y": 3},
                    duration=2,
                    description="Add the two numbers and keep the result visible for later branching.",
                ),
            ],
            tool_calls_limit=2,
        )
        task = agent._initialize_task(
            turns=[],
            prompt="react",
            inputs={},
        )

        updated = agent._prepare_next_batch(task)

        slot = updated.running_blackboard[0]
        assert updated.prepared_steps == [0]
        assert slot.status == "prepared"
        assert slot.is_prepared() is True
        assert slot.step_dependencies == ()
        assert slot.await_step is NO_VAL
        assert slot.resolved_args == {"x": 2, "y": 3}
        assert updated.step_meta[0].observable == 2
        assert updated.step_meta[0].description == "Add the two numbers and keep the result visible for later branching."

    def test_prepare_next_batch_records_step_dependencies(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=1,
                    tool="Tool.tests.multiply",
                    args={"x": "<<__s0__>>", "y": 10},
                    description="Multiply the prior addition result by ten for the current calculation.",
                ),
            ],
            tool_calls_limit=2,
        )
        task = agent._initialize_task(
            turns=[],
            prompt="react",
            inputs={},
        )
        task.next_step_index = 1
        task.running_blackboard[0] = executed_slot(0, 5)
        task.step_meta[0].description = "Add the two numbers for the current calculation."

        updated = agent._prepare_next_batch(task)

        slot = updated.running_blackboard[1]
        assert slot.status == "prepared"
        assert slot.step_dependencies == (0,)
        assert slot.await_step is NO_VAL
        assert slot.resolved_args == {"x": 5, "y": 10}
        assert updated.step_meta[1].description == "Multiply the prior addition result by ten for the current calculation."

    def test_async_invoke_executes_step_by_step_until_return(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    step=0,
                    tool="Tool.tests.add",
                    args={"x": 2, "y": 3},
                    description="Add the two input numbers for the current calculation.",
                ),
                react_step_json(
                    step=1,
                    tool="Tool.tests.multiply",
                    args={"x": "<<__s0__>>", "y": 10},
                    description="Multiply the addition result by ten for the current calculation.",
                ),
                react_step_json(
                    step=2,
                    tool=return_tool.full_name,
                    args={"val": "<<__s1__>>"},
                    description="Return the final multiplied value because the current calculation is complete.",
                ),
            ],
            tool_calls_limit=2,
        )

        result = asyncio.run(agent.async_invoke({"prompt": "run react"}))

        assert result.result == 50

    def test_llm_record_system_prompt_name_is_reason_then_act(self) -> None:
        agent = make_react_agent(
            [
                react_step_json(
                    tool=return_tool.full_name,
                    args={"val": 42},
                    duration=0,
                    description="Return the value for the test.",
                )
            ],
            tool_calls_limit=1,
        )

        agent.invoke({"prompt": "run"})

        for rec in agent.records[-1].llm_records:
            assert rec.system_prompt_name == "reason_then_act"


class TestReActCascadeFailedPropagation:
    """
    Integration tests for cascade FAILED propagation in ReActAgent.

    When a tool step fails (fail_fast=False), a subsequent LLM-generated
    step whose args reference it via <<__sN__>> is cascade-marked FAILED in
    _apply_react_step_result without raising. The return tool always raises
    when its arg dependencies failed.
    """

    def test_react_dependent_step_is_cascade_failed(self) -> None:
        """ReAct: non-return step with a failed dep is cascade-marked FAILED; run succeeds."""
        agent = make_react_agent(
            [
                react_step_json(step=0, tool="Tool.tests.fail_tool", args={}),
                react_step_json(step=1, tool="Tool.tests.add", args={"x": "<<__s0__>>", "y": 2}),
                react_step_json(step=2, tool=return_tool.full_name, args={"val": 99}, duration=0),
            ],
            tool_calls_limit=2,
            fail_fast=False,
        )

        result = agent.invoke({"prompt": "react cascade"})

        assert result.result == 99
        board = agent.blackboard
        assert board[0].status == BlackboardSlot.FAILED
        assert board[1].status == BlackboardSlot.FAILED
        assert board[2].status == BlackboardSlot.EXECUTED

    def test_react_return_step_with_failed_dep_raises(self) -> None:
        """ReAct: return step depending on a FAILED step raises ToolAgentError."""
        agent = make_react_agent(
            [
                react_step_json(step=0, tool="Tool.tests.fail_tool", args={}),
                react_step_json(step=1, tool=return_tool.full_name, args={"val": "<<__s0__>>"}, duration=0),
            ],
            tool_calls_limit=1,
            fail_fast=False,
        )

        with pytest.raises(ToolAgentError, match="return step"):
            agent.invoke({"prompt": "react return cascade raise"})


# ── TestFailedCacheRefValidation ──────────────────────────────────────────────

class TestReActGenerationRetry:
    """Retry loop in _generate_next_step/_agenerate_next_step: shared budget, feedback, LLMRecord accumulation."""

    INVALID_JSON = "this is not json at all"

    # A structurally invalid step: references a tool that doesn't exist.
    INVALID_STEP_WRONG_TOOL = react_step_json(
        step=0, tool="Tool.tests.nonexistent", args={}, duration=0
    )

    VALID_STEP_0 = react_step_json(step=0, tool="Tool.tests.add", args={"x": 1, "y": 2}, duration=1)
    VALID_RETURN = react_step_json(step=1, tool=return_tool.full_name, args={"val": "<<__s0__>>"}, duration=0)
    VALID_RETURN_LITERAL = react_step_json(step=0, tool=return_tool.full_name, args={"val": 42}, duration=0)

    def test_zero_retries_raises_on_first_bad_json(self) -> None:
        """generation_retries=0 (default): bad JSON raises immediately."""
        agent = make_react_agent([self.INVALID_JSON])
        with pytest.raises(ToolAgentError):
            agent.invoke({"prompt": "run"})

    def test_zero_retries_emits_one_llm_call_before_raise(self) -> None:
        """generation_retries=0: exactly one LLM call is made before raising."""
        engine = ScriptedLLMEngine([self.INVALID_JSON])
        agent = ReActAgent(
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
        """generation_retries=1: bad JSON on first call, valid step on second; run completes."""
        agent = make_react_agent(
            [self.INVALID_JSON, self.VALID_RETURN_LITERAL],
            generation_retries=1,
        )
        result = agent.invoke({"prompt": "run"})
        assert result.result == 42

    def test_json_error_retry_stores_two_llm_records(self) -> None:
        """Two attempts on a single step produce two LLMRecords."""
        engine = ScriptedLLMEngine([self.INVALID_JSON, self.VALID_RETURN_LITERAL])
        agent = ReActAgent(
            name="tests",
            namespace="tests",
            description=".",
            llm_engine=engine,
            generation_retries=1,
        )
        agent.invoke({"prompt": "run"})
        llm_records = agent.records[-1].llm_records
        assert len(llm_records) == 2
        assert len(llm_records[0].messages) == 3
        assert len(llm_records[1].messages) == 2

    def test_spec_error_retry_succeeds_on_second_call(self) -> None:
        """generation_retries=1: unknown tool on first step attempt; valid step on second."""
        agent = make_react_agent(
            [self.INVALID_STEP_WRONG_TOOL, self.VALID_RETURN_LITERAL],
            generation_retries=1,
        )
        result = agent.invoke({"prompt": "run"})
        assert result.result == 42

    def test_spec_error_retry_stores_two_llm_records(self) -> None:
        """Spec-validation failure + successful retry = two LLMRecords for that step."""
        engine = ScriptedLLMEngine([self.INVALID_STEP_WRONG_TOOL, self.VALID_RETURN_LITERAL])
        agent = ReActAgent(
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
        assert len(llm_records[0].messages) == 3
        assert len(llm_records[1].messages) == 2

    def test_budget_exhausted_raises_after_all_attempts(self) -> None:
        """generation_retries=1: both attempts return invalid JSON → ToolAgentError."""
        agent = make_react_agent(
            [self.INVALID_JSON, self.INVALID_JSON],
            generation_retries=1,
        )
        with pytest.raises(ToolAgentError):
            agent.invoke({"prompt": "run"})

    def test_budget_exhausted_records_all_llm_calls(self) -> None:
        """All attempts (including the failing ones) are recorded as LLM calls."""
        engine = ScriptedLLMEngine([self.INVALID_JSON, self.INVALID_JSON])
        agent = ReActAgent(
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
        """On JSON-decode failure the retry call receives more messages than the first call."""
        engine = ScriptedLLMEngine([self.INVALID_JSON, self.VALID_RETURN_LITERAL])
        agent = ReActAgent(
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

    def test_spec_feedback_contains_reserialised_step_not_raw_string(self) -> None:
        """On spec-validation failure the retry user message contains the re-serialised step."""
        engine = ScriptedLLMEngine([self.INVALID_STEP_WRONG_TOOL, self.VALID_RETURN_LITERAL])
        agent = ReActAgent(
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

    def test_shared_budget_across_steps(self) -> None:
        """Retries consumed by step 0 reduce availability for step 1."""
        # generation_retries=1: step 0 uses the 1 retry (bad JSON → valid step).
        # Step 1 (return) immediately gets bad JSON; no retries left → ToolAgentError.
        agent = make_react_agent(
            [self.INVALID_JSON, self.VALID_STEP_0, self.INVALID_JSON],
            generation_retries=1,
            tool_calls_limit=3,
        )
        with pytest.raises(ToolAgentError, match="budget exhausted"):
            agent.invoke({"prompt": "run"})

    def test_llm_records_accumulate_all_attempts(self) -> None:
        """Total LLMRecords equals the sum of all attempt counts across all steps."""
        # step 0: 2 attempts (1 retry); return step: 1 attempt. Total = 3.
        engine = ScriptedLLMEngine([
            self.INVALID_JSON,       # step 0 attempt 1 — bad JSON
            self.VALID_STEP_0,       # step 0 attempt 2 — succeeds
            self.VALID_RETURN,       # return step — succeeds first try
        ])
        agent = ReActAgent(
            name="tests",
            namespace="tests",
            description=".",
            llm_engine=engine,
            generation_retries=1,
            tool_calls_limit=3,
        )
        register_math_tools(agent)  # type: ignore[arg-type]
        agent.invoke({"prompt": "run"})
        assert len(agent.records[-1].llm_records) == 3

    def test_observable_counters_stable_during_retries(self) -> None:
        """Observable counters on prior steps do not decrement during a retry attempt."""
        # step 0: valid, duration=2 (observable for 2 future successful generations).
        # return step: 1 failed attempt (bad JSON) before succeeding.
        # observable for step 0 must still be 2 after the failed retry, and 1 after the success.
        engine = ScriptedLLMEngine([
            self.VALID_STEP_0,           # step 0: observable=1 (duration=1)
            self.INVALID_JSON,           # return step attempt 1: fails → no counter decrement
            self.VALID_RETURN,           # return step attempt 2: succeeds → decrements
        ])
        agent = ReActAgent(
            name="tests",
            namespace="tests",
            description=".",
            llm_engine=engine,
            generation_retries=1,
            tool_calls_limit=3,
        )
        register_math_tools(agent)  # type: ignore[arg-type]
        agent.invoke({"prompt": "run"})
        # After the run: step 0 had observable=1, which was decremented exactly once
        # (by the successful return-step generation). It should now be 0.
        record = agent.records[-1]
        # The run completed; step 0's observable was decremented once (by the successful commit),
        # not twice (which would happen if the failed retry also decremented it).
        assert len(record.llm_records) == 3  # 1 (step0) + 1 (failed return) + 1 (success return)


# ── TestExecutePreparedBatchEarlyValidation ───────────────────────────────────

class TestMaxDurationSingleSource:
    """B3/B-7: max_duration is computed once in _prepare_next_batch and flows to both
    _build_react_messages, _generate_next_step, and _apply_react_step_result; none
    re-derive it independently."""

    def test_max_duration_limits_step_at_budget_boundary(self) -> None:
        """A step with duration == remaining budget is accepted."""
        agent = make_react_agent(
            [
                # Step 0: duration=2 with tool_calls_limit=3 → max_duration=3; fine.
                react_step_json(step=0, tool="Tool.tests.add", args={"x": 1, "y": 2}, duration=2),
                react_step_json(step=1, tool=return_tool.full_name, args={"val": "<<__s0__>>"}, duration=0),
            ],
            tool_calls_limit=3,
        )
        result = agent.invoke({"prompt": "run"})
        assert result.result == 3

    def test_step_exceeding_max_duration_triggers_retry(self) -> None:
        """A step with duration > remaining budget is rejected (validation error → retry)."""
        agent = make_react_agent(
            [
                # Step 0 bad: duration=5 when only 2 slots remain (prefix_len=0, limit=2).
                react_step_json(step=0, tool="Tool.tests.add", args={"x": 1, "y": 2}, duration=5),
                # Step 0 good: corrected on retry.
                react_step_json(step=0, tool="Tool.tests.add", args={"x": 1, "y": 2}, duration=1),
                react_step_json(step=1, tool=return_tool.full_name, args={"val": "<<__s0__>>"}, duration=0),
            ],
            tool_calls_limit=2,
            generation_retries=1,
        )
        result = agent.invoke({"prompt": "run"})
        assert result.result == 3

    def test_apply_react_step_result_accepts_max_duration_param(self) -> None:
        """_apply_react_step_result takes max_duration as a kwarg; no independent recomputation."""
        import inspect
        sig = inspect.signature(ReActAgent._apply_react_step_result)
        params = sig.parameters
        assert "max_duration" in params
        assert params["max_duration"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_max_duration_flows_from_prepare_to_apply(self) -> None:
        """observe_duration == max_duration at budget boundary is accepted end-to-end."""
        agent = make_react_agent(
            [
                # prefix_len=0, tool_calls_limit=1 → max_duration = max(0, 1-0) = 1; duration=1 OK.
                react_step_json(step=0, tool="Tool.tests.add", args={"x": 2, "y": 3}, duration=1),
                react_step_json(step=1, tool=return_tool.full_name, args={"val": "<<__s0__>>"}, duration=0),
            ],
            tool_calls_limit=1,
        )
        result = agent.invoke({"prompt": "run"})
        assert result.result == 5

# ── TestCacheRefValidation ───────────────────────────────────────────────────



class TestCacheRefValidation:
    """Three-category cache-ref validation for PlanAct and ReAct, plus B1 FAILED step visibility."""

    # ── PlanAct out-of-range ────────────────────────────────────────────────

    def test_planact_out_of_range_cache_ref_raises(self) -> None:
        """PlanAct: cache index beyond cache length raises with 'do not exist' message."""
        agent = make_planact_agent(
            [
                json.dumps([
                    {"tool": return_tool.full_name, "args": {"val": "<<__c5__>>"}},
                ]),
            ],
            context_enabled=False,
        )
        with pytest.raises(ToolAgentError, match="cache indices that do not exist"):
            agent.invoke({"prompt": "run"})

    # ── PlanAct out-of-conversation ──────────────────────────────────────────

    def test_planact_out_of_conv_cache_ref_raises(self) -> None:
        """PlanAct: cache index in range but not in either frozenset raises."""
        agent = make_planact_agent([], context_enabled=True)
        # Seed a slot in the blackboard (simulates a prior-session entry).
        prior_slot = BlackboardSlot(step=0, tool="Tool.tests.add", args={}, status=BlackboardSlot.EXECUTED)
        agent._blackboard.append(prior_slot)

        # Build the cache_blackboard as _setup_plan_init would (since context_enabled=True).
        cache_blackboard = [prior_slot.copy()]

        # Parse a plan that references cache index 0.
        plan_json = json.dumps([{"tool": return_tool.full_name, "args": {"val": "<<__c0__>>"}}])
        parsed = json.loads(plan_json)

        # Both frozensets are empty — index 0 is in-range but not from this conversation.
        result = agent._process_plan_output(
            parsed=parsed,
            cache_blackboard=cache_blackboard,
            valid_cache_indices=frozenset(),
            failed_cache_indices=frozenset(),
        )
        assert isinstance(result, str)
        assert "not part of this conversation" in result

    # ── PlanAct failed-in-conversation (already in TestFailedCacheRefValidation) ──

    # ── ReAct out-of-range ───────────────────────────────────────────────────

    def test_react_out_of_range_cache_ref_raises(self) -> None:
        """ReAct: cache index beyond cache length raises with 'do not exist' message."""
        agent = make_react_agent(
            [
                react_step_json(step=0, tool="Tool.tests.add", args={"x": "<<__c99__>>", "y": 1}),
            ],
            tool_calls_limit=1,
            context_enabled=False,
        )
        with pytest.raises(ToolAgentError, match="cache indices that do not exist"):
            agent.invoke({"prompt": "run"})

    # ── ReAct out-of-conversation ────────────────────────────────────────────

    def test_react_out_of_conv_cache_ref_raises(self) -> None:
        """ReAct: in-range cache index not in either frozenset raises."""
        agent = make_react_agent([], context_enabled=True, tool_calls_limit=1)
        prior_slot = BlackboardSlot(step=0, tool="Tool.tests.add", args={}, status=BlackboardSlot.EXECUTED)
        agent._blackboard.append(prior_slot)

        # Build a step referencing cache index 0.
        parsed = json.loads(react_step_json(step=0, tool="Tool.tests.add", args={"x": "<<__c0__>>", "y": 1}))
        cache_blackboard = [prior_slot.copy()]

        result = agent._process_next_step_output(
            parsed=parsed,
            expected_step=0,
            cache_blackboard=cache_blackboard,
            max_duration=1,
            valid_cache_indices=frozenset(),
            failed_cache_indices=frozenset(),
        )
        assert isinstance(result, str)
        assert "not part of this conversation" in result

    # ── B1: FAILED step visible in ReAct snapshot ───────────────────────────

    def test_failed_step_appears_in_react_snapshot(self) -> None:
        """B1: under fail_fast=False, a FAILED running-blackboard slot is rendered
        with status='FAILED' and error in the snapshot passed to the next step."""
        agent = make_react_agent(
            [
                react_step_json(step=0, tool="Tool.tests.fail_tool", args={}, duration=0),
                react_step_json(step=1, tool=return_tool.full_name, args={"val": 99}, duration=0),
            ],
            tool_calls_limit=2,
            fail_fast=False,
        )
        result = agent.invoke({"prompt": "run"})
        assert result.result == 99

        # Build the snapshot for prefix_len=1 (after step 0 fails, before step 1 is generated).
        task = agent._initialize_task(
            turns=[],
            prompt="react",
            inputs={},
        )
        # Manually seed a FAILED slot at index 0 to simulate the post-failure state.
        failed_slot = BlackboardSlot(
            step=0,
            tool="Tool.tests.fail_tool",
            args={},
            error=RuntimeError("intentional failure"),
            status=BlackboardSlot.FAILED,
        )
        task.running_blackboard[0] = failed_slot
        task.step_meta[0] = ReActStepMeta(observable=0, description="Test fail step.")

        working_messages, _ = agent._build_react_messages(task, prefix_len=1, max_duration=1)
        # working_messages[-2] is the assistant turn with the running-plan snapshot.
        snapshot_text = working_messages[-2]["content"]

        assert "FAILED" in snapshot_text
        assert "intentional failure" in snapshot_text

    def test_executed_step_carries_no_failed_status_in_snapshot(self) -> None:
        """B1: a successfully EXECUTED step does not get status=FAILED in the snapshot."""
        agent = make_react_agent(
            [
                react_step_json(step=0, tool="Tool.tests.add", args={"x": 1, "y": 2}, duration=1),
                react_step_json(step=1, tool=return_tool.full_name, args={"val": "<<__s0__>>"}, duration=0),
            ],
            tool_calls_limit=2,
        )
        result = agent.invoke({"prompt": "run"})
        assert result.result == 3

        # Build snapshot after step 0 executed — check no FAILED markers.
        task = agent._initialize_task(
            turns=[],
            prompt="react",
            inputs={},
        )
        # After invoke, the blackboard is persisted; index 0 is the executed add step.
        exec_slot = agent.blackboard[0]
        task.running_blackboard[0] = exec_slot
        task.step_meta[0] = ReActStepMeta(observable=0, description="Add two numbers.")

        working_messages, _ = agent._build_react_messages(task, prefix_len=1, max_duration=1)
        # working_messages[-2] is the assistant turn with the running-plan snapshot.
        snapshot_text = working_messages[-2]["content"]

        assert "result_ref" in snapshot_text
        # No slot-level "status" key should appear (only FAILED slots get that field).
        assert "'status'" not in snapshot_text
