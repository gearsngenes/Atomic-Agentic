from __future__ import annotations

import pytest

from conftest import (
    make_reactk_agent,
    subplan_step,
    subplan_json,
    ScriptedLLMEngine,
    make_tool_result,
)

from atomic_agentic.agents.toolagent import return_tool
from atomic_agentic.agents.reactk import ReActKAgent
from atomic_agentic.models.agents.blackboard_models import BlackboardSlot
from atomic_agentic.exceptions import ToolAgentError
from atomic_agentic.constants.core import NO_VAL


def _planned_slot(step: int, tool: str, args: dict, *, deps: tuple[int, ...] = (), await_step=NO_VAL) -> BlackboardSlot:
    return BlackboardSlot(
        step=step,
        tool=tool,
        args=args,
        resolved_args=NO_VAL,
        result=NO_VAL,
        error=NO_VAL,
        status=BlackboardSlot.PLANNED,
        step_dependencies=deps,
        await_step=await_step,
    )


class TestReActKAgentConstruction:
    def test_steps_per_round_must_be_positive_int(self) -> None:
        with pytest.raises(ToolAgentError, match="steps_per_round"):
            ReActKAgent(
                name="bad_reactk",
                namespace="tests",
                description="Bad ReActK agent.",
                llm_engine=ScriptedLLMEngine([]),
                steps_per_round=0,
            )

    def test_steps_per_round_is_mutable(self) -> None:
        agent = make_reactk_agent([], tool_calls_limit=3, steps_per_round=1)
        assert agent.steps_per_round == 1
        agent.steps_per_round = 4
        assert agent.steps_per_round == 4

    def test_requires_concrete_non_negative_tool_calls_limit(self) -> None:
        with pytest.raises(ToolAgentError, match="tool_calls_limit"):
            ReActKAgent(
                name="bad_reactk",
                namespace="tests",
                description="Bad ReActK agent.",
                llm_engine=ScriptedLLMEngine([]),
                tool_calls_limit=-1,
            )


class TestReActKAgentEndToEnd:
    def test_single_step_per_round_runs_to_completion(self) -> None:
        agent = make_reactk_agent(
            [
                subplan_json(subplan_step(
                    tool="Tool.tests.add", args={"x": 2, "y": 3},
                    description="Add the two input numbers for the current calculation.",
                )),
                subplan_json(subplan_step(
                    tool=return_tool.full_name, args={"val": "<<__s0__>>"},
                    description="Return the addition result because the current task is complete.",
                )),
            ],
            steps_per_round=1,
            tool_calls_limit=2,
        )

        result = agent.invoke({"prompt": "run reactk"})

        assert result.result == 5
        engine = agent.llm_engine
        assert isinstance(engine, ScriptedLLMEngine)
        assert len(engine.calls) == 2

        first_call_text = "\n".join(message["content"] for message in engine.calls[0])
        assert "at most 1 step(s) per round" in first_call_text
        assert "UP TO 1 step(s)" in first_call_text
        assert "duration must be an int from 0 to 2" in first_call_text

    def test_multi_step_subplan_batches_and_drains_in_one_round(self) -> None:
        agent = make_reactk_agent(
            [
                subplan_json(
                    subplan_step(
                        tool="Tool.tests.add", args={"x": 2, "y": 3}, duration=1,
                        description="Add the two input numbers for the current calculation.",
                    ),
                    subplan_step(
                        tool=return_tool.full_name, args={"val": "<<__s0__>>"},
                        description="Return the addition result because the current task is complete.",
                    ),
                ),
            ],
            steps_per_round=2,
            tool_calls_limit=2,
        )

        result = agent.invoke({"prompt": "run reactk"})

        assert result.result == 5
        engine = agent.llm_engine
        assert isinstance(engine, ScriptedLLMEngine)
        # Both steps came from a single generation round, compiled into two
        # sequential batches (return depends on the add step) and drained
        # with zero further LLM calls.
        assert len(engine.calls) == 1

    def test_absolute_indexing_shows_correct_running_plan_across_rounds(self) -> None:
        agent = make_reactk_agent(
            [
                subplan_json(subplan_step(
                    tool="Tool.tests.add", args={"x": 2, "y": 3},
                    description="Add the two input numbers for the current calculation.",
                )),
                subplan_json(subplan_step(
                    tool="Tool.tests.multiply", args={"x": "<<__s0__>>", "y": 10},
                    description="Multiply the addition result by ten for the current calculation.",
                )),
                subplan_json(subplan_step(
                    tool=return_tool.full_name, args={"val": "<<__s1__>>"},
                    description="Return the final multiplied value because the current task is complete.",
                )),
            ],
            steps_per_round=1,
            tool_calls_limit=2,
        )

        result = agent.invoke({"prompt": "run reactk"})

        assert result.result == 50
        engine = agent.llm_engine
        assert isinstance(engine, ScriptedLLMEngine)
        assert len(engine.calls) == 3

        second_call_text = "\n".join(message["content"] for message in engine.calls[1])
        assert "RUNNING PLAN STEPS 0-0 SO FAR" in second_call_text

        third_call_text = "\n".join(message["content"] for message in engine.calls[2])
        assert "RUNNING PLAN STEPS 0-1 SO FAR" in third_call_text
        assert "<<__s1__>>" in third_call_text

    def test_cascade_failure_skips_dependent_step_without_raising(self) -> None:
        agent = make_reactk_agent(
            [
                subplan_json(subplan_step(
                    tool="Tool.tests.fail_tool", args={},
                    description="Attempt a tool call that will fail for the current test.",
                )),
                subplan_json(subplan_step(
                    tool="Tool.tests.add", args={"x": "<<__s0__>>", "y": 1},
                    description="Try to build on the failed step's result.",
                )),
                subplan_json(subplan_step(
                    tool=return_tool.full_name, args={"val": None},
                    description="Return because the run cannot proceed further.",
                )),
            ],
            steps_per_round=1,
            tool_calls_limit=2,
            fail_fast=False,
        )

        result = agent.invoke({"prompt": "run reactk"})

        assert result.result is None

    def test_rejects_subplan_that_is_not_a_json_array(self) -> None:
        agent = make_reactk_agent(["{}"], steps_per_round=1, tool_calls_limit=2)

        with pytest.raises(ToolAgentError, match="non-empty JSON array"):
            agent.invoke({"prompt": "run reactk"})

    def test_rejects_return_step_with_nonzero_duration(self) -> None:
        agent = make_reactk_agent(
            [subplan_json(subplan_step(tool=return_tool.full_name, args={"val": None}, duration=1))],
            steps_per_round=1,
            tool_calls_limit=2,
        )

        with pytest.raises(ToolAgentError, match="return tool must use 'duration' 0"):
            agent.invoke({"prompt": "run reactk"})


class TestReActKAgentNormalization:
    def test_normalize_relocates_return_and_forces_full_dependency(self) -> None:
        agent = make_reactk_agent([], tool_calls_limit=5, steps_per_round=5)

        return_slot = _planned_slot(10, return_tool.full_name, {"val": None})
        add_slot = _planned_slot(11, "Tool.tests.add", {"x": 1, "y": 2})
        items = [(return_slot, 0, "return early"), (add_slot, 0, "add later")]

        normalized = agent._normalize_subplan_slots(items, cursor=10)

        assert not isinstance(normalized, str)
        assert normalized[0][0].tool == "Tool.tests.add"
        assert normalized[0][0].step == 10
        assert normalized[1][0].tool == return_tool.full_name
        assert normalized[1][0].step == 11
        # Forced full-history dependency: every absolute index before the
        # return's own final position, not just its immediate sibling.
        assert normalized[1][0].step_dependencies == tuple(range(11))
        assert normalized[1][0].await_step is NO_VAL

    def test_normalize_leaves_return_free_subplan_unchanged(self) -> None:
        agent = make_reactk_agent([], tool_calls_limit=5, steps_per_round=5)

        slot_a = _planned_slot(3, "Tool.tests.add", {"x": 1, "y": 2})
        slot_b = _planned_slot(4, "Tool.tests.multiply", {"x": "<<__s3__>>", "y": 5}, deps=(3,))
        items = [(slot_a, 0, "add"), (slot_b, 0, "multiply")]

        normalized = agent._normalize_subplan_slots(items, cursor=3)

        assert not isinstance(normalized, str)
        assert [slot.tool for slot, _, _ in normalized] == ["Tool.tests.add", "Tool.tests.multiply"]
        assert [slot.step for slot, _, _ in normalized] == [3, 4]

    def test_normalize_rejects_multiple_return_steps(self) -> None:
        agent = make_reactk_agent([], tool_calls_limit=5, steps_per_round=5)

        r1 = _planned_slot(0, return_tool.full_name, {"val": 1})
        r2 = _planned_slot(1, return_tool.full_name, {"val": 2})
        items = [(r1, 0, "a"), (r2, 0, "b")]

        result = agent._normalize_subplan_slots(items, cursor=0)

        assert isinstance(result, str)
        assert "multiple return steps" in result


class TestReActKAgentValidation:
    def test_validate_rejects_exceeding_steps_per_round(self) -> None:
        agent = make_reactk_agent([], tool_calls_limit=5, steps_per_round=1)

        slot_a = _planned_slot(0, "Tool.tests.add", {"x": 1, "y": 2})
        slot_b = _planned_slot(1, "Tool.tests.multiply", {"x": 1, "y": 2})
        items = [(slot_a, 0, "add"), (slot_b, 0, "multiply")]

        feedback = agent._validate_subplan_slots(
            items=items, cursor=0, round_cap=5,
            cache_blackboard=[], valid_cache_indices=frozenset(), failed_cache_indices=frozenset(),
        )

        assert feedback is not None
        assert "steps_per_round" in feedback

    def test_validate_rejects_exceeding_round_cap_budget(self) -> None:
        agent = make_reactk_agent([], tool_calls_limit=5, steps_per_round=5)

        slot_a = _planned_slot(0, "Tool.tests.add", {"x": 1, "y": 2})
        items = [(slot_a, 0, "add")]

        feedback = agent._validate_subplan_slots(
            items=items, cursor=0, round_cap=0,
            cache_blackboard=[], valid_cache_indices=frozenset(), failed_cache_indices=frozenset(),
        )

        assert feedback is not None
        assert "remaining tool-call budget" in feedback


class TestReActKAgentApplySubplanResult:
    def test_batch_compile_and_filter_handles_cross_round_dependency(self) -> None:
        agent = make_reactk_agent([], tool_calls_limit=5, steps_per_round=5)
        state = agent._initialize_run_state(
            turns=[], prompt="test task", inputs={},
            valid_cache_indices=frozenset(), failed_cache_indices=frozenset(),
        )

        slot0 = _planned_slot(0, "Tool.tests.add", {"x": 1, "y": 2})
        state = agent._apply_subplan_result(state, cursor=0, items=[(slot0, 0, "add")], llm_records=[])

        assert state.pending_batches == [[0]]
        assert state.next_step_index == 1

        # Simulate step 0 having drained/executed before round 2 generates.
        state.running_blackboard[0].status = BlackboardSlot.EXECUTED
        state.running_blackboard[0].result = make_tool_result(3)
        state.executed_steps.add(0)
        state.pending_batches = []

        slot1 = _planned_slot(1, "Tool.tests.multiply", {"x": "<<__s0__>>", "y": 10}, deps=(0,))
        state = agent._apply_subplan_result(state, cursor=1, items=[(slot1, 0, "multiply")], llm_records=[])

        # No KeyError despite slot1 depending on an earlier round's absolute
        # index, and the already-executed step 0 is filtered out of the new batch.
        assert state.pending_batches == [[1]]
        assert state.next_step_index == 2

    def test_observability_decrements_once_per_round_not_per_batch(self) -> None:
        agent = make_reactk_agent([], tool_calls_limit=5, steps_per_round=5)
        state = agent._initialize_run_state(
            turns=[], prompt="test task", inputs={},
            valid_cache_indices=frozenset(), failed_cache_indices=frozenset(),
        )

        slot0 = _planned_slot(0, "Tool.tests.add", {"x": 1, "y": 2})
        state = agent._apply_subplan_result(state, cursor=0, items=[(slot0, 2, "add")], llm_records=[])

        # The round that just set observable=2 must not decrement it itself.
        assert state.step_meta[0].observable == 2

        state.running_blackboard[0].status = BlackboardSlot.EXECUTED
        state.running_blackboard[0].result = make_tool_result(3)
        state.pending_batches = []

        slot1 = _planned_slot(1, "Tool.tests.multiply", {"x": "<<__s0__>>", "y": 10}, deps=(0,))
        state = agent._apply_subplan_result(state, cursor=1, items=[(slot1, 0, "multiply")], llm_records=[])

        # Exactly one further round decrements it exactly once.
        assert state.step_meta[0].observable == 1
