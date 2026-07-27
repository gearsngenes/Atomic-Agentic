from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from conftest import ScriptedLLMEngine

from atomic_agentic.agents.planask import PlanAskAgent
from atomic_agentic.exceptions import ThinkingAgentError
from atomic_agentic.models.results import ThinkingAgentResult


def thought_item(
    *,
    observation: str | None = None,
    question: str = "What is needed?",
    answer: str = "An answer.",
    **overrides: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {"observation": observation, "question": question, "answer": answer}
    item.update(overrides)
    return item


def batch_json(items: list[dict[str, Any]] | None = None) -> str:
    if items is None:
        items = [thought_item()]
    return json.dumps(items)


def make_agent(
    responses: list[str],
    *,
    max_thinking_rounds: int | None = 3,
    generation_retries: int = 0,
    **kwargs: Any,
) -> PlanAskAgent:
    return PlanAskAgent(
        name="tests",
        namespace="tests",
        description="PlanAskAgent under test.",
        llm_engine=ScriptedLLMEngine(responses),
        max_thinking_rounds=max_thinking_rounds,
        generation_retries=generation_retries,
        **kwargs,
    )


class TestConstruction:
    def test_registers_fixed_plan_ask_prompt(self) -> None:
        agent = make_agent([batch_json(), "reply"])
        assert "plan_ask" in agent._system_prompts
        assert agent._system_prompts["plan_ask"].description == "PlanAskAgent batch self-questioning prompt."

    def test_role_prompt_still_registered(self) -> None:
        agent = make_agent([batch_json(), "reply"])
        assert "role" in agent._system_prompts

    def test_no_new_constructor_parameters_beyond_thinking_agent(self) -> None:
        agent = make_agent(
            [batch_json(), "reply"],
            role_prompt="Be a helpful assistant.",
            role_description="Scoping a research task.",
            thoughts_window=None,
        )
        assert agent.role_description == "Scoping a research task."

    def test_max_thinking_rounds_none_accepted(self) -> None:
        # Unlike SelfAskAgent, None is permitted -- the batch is inherently
        # finite regardless of any cap.
        agent = make_agent([batch_json(), "reply"], max_thinking_rounds=None)
        assert agent._max_thinking_rounds is None


class TestBatchValidation:
    def test_not_a_list_triggers_retry_feedback(self) -> None:
        agent = make_agent(
            [json.dumps(thought_item()), batch_json(), "reply"], generation_retries=1
        )
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"

    def test_empty_list_triggers_retry_feedback(self) -> None:
        agent = make_agent([json.dumps([]), batch_json(), "reply"], generation_retries=1)
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"

    def test_list_exceeding_max_rounds_triggers_retry_feedback(self) -> None:
        too_many = batch_json([thought_item(question="q1"), thought_item(question="q2")])
        agent = make_agent(
            [too_many, batch_json(), "reply"], max_thinking_rounds=1, generation_retries=1
        )
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"

    def test_list_accepted_when_max_rounds_none_regardless_of_length(self) -> None:
        items = [thought_item(question=f"q{i}", answer=f"a{i}") for i in range(5)]
        agent = make_agent([batch_json(items), "reply"], max_thinking_rounds=None)
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"
        assert result.thoughts_end - result.thoughts_start == 5

    @pytest.mark.parametrize("missing_key", ["observation", "question", "answer"])
    def test_missing_required_key_in_item_triggers_retry_feedback(self, missing_key: str) -> None:
        bad_item = thought_item()
        del bad_item[missing_key]
        agent = make_agent(
            [batch_json([bad_item]), batch_json(), "reply"], generation_retries=1
        )
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"

    def test_unsupported_extra_key_in_item_triggers_retry_feedback(self) -> None:
        bad_item = thought_item(extra="not allowed")
        agent = make_agent(
            [batch_json([bad_item]), batch_json(), "reply"], generation_retries=1
        )
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"

    def test_observation_accepts_null(self) -> None:
        agent = make_agent([batch_json([thought_item(observation=None)]), "reply"])
        agent.invoke({"prompt": "q"})
        assert agent.get_thoughts()[0].observation is None

    def test_observation_accepts_string(self) -> None:
        agent = make_agent([batch_json([thought_item(observation="a note")]), "reply"])
        agent.invoke({"prompt": "q"})
        assert agent.get_thoughts()[0].observation == "a note"

    def test_observation_rejects_non_string_non_null(self) -> None:
        bad_item = thought_item(observation=123)
        agent = make_agent(
            [batch_json([bad_item]), batch_json(), "reply"], generation_retries=1
        )
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"

    @pytest.mark.parametrize("field", ["question", "answer"])
    def test_empty_string_field_in_item_triggers_retry_feedback(self, field: str) -> None:
        bad_item = thought_item(**{field: "   "})
        agent = make_agent(
            [batch_json([bad_item]), batch_json(), "reply"], generation_retries=1
        )
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"

    def test_item_not_a_mapping_triggers_retry_feedback(self) -> None:
        agent = make_agent(
            [json.dumps(["not an object"]), batch_json(), "reply"], generation_retries=1
        )
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"


class TestGenerationRetry:
    def test_json_decode_failure_retries_then_succeeds(self) -> None:
        agent = make_agent(["not json", batch_json(), "reply"], generation_retries=1)
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"

    def test_json_decode_exhaustion_raises_thinking_agent_error(self) -> None:
        agent = make_agent(["not json", "still not json"], generation_retries=0)
        with pytest.raises(ThinkingAgentError, match="generation retry budget exhausted"):
            agent.invoke({"prompt": "q"})

    def test_schema_validation_exhaustion_raises_thinking_agent_error(self) -> None:
        agent = make_agent([json.dumps([]), json.dumps([])], generation_retries=0)
        with pytest.raises(ThinkingAgentError, match="generation retry budget exhausted"):
            agent.invoke({"prompt": "q"})


class TestRoundsUsedAndTaskMessages:
    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_rounds_used_equals_length_of_validated_list(self, n: int) -> None:
        items = [thought_item(question=f"q{i}", answer=f"a{i}") for i in range(n)]
        agent = make_agent([batch_json(items), "reply"], max_thinking_rounds=10)
        result = agent.invoke({"prompt": "q"})
        assert result.thoughts_end - result.thoughts_start == n

    def test_task_messages_cleared_after_successful_batch(self) -> None:
        agent = make_agent([batch_json(), "reply"])
        task = agent._initialize_task(turns=[], prompt="q", inputs={"prompt": "q"})
        agent._think(task)
        assert task.task_messages == []


class TestUnconditionalPhaseTransition:
    def test_system_prompt_name_flips_to_role_after_single_batch(self) -> None:
        agent = make_agent([batch_json(), "reply"])
        task = agent._initialize_task(turns=[], prompt="q", inputs={"prompt": "q"})
        assert task.system_prompt_name == "plan_ask"

        agent._think(task)

        assert task.system_prompt_name == "role"

    def test_exactly_one_thinking_call_then_one_reply_call(self) -> None:
        engine = ScriptedLLMEngine([batch_json(), "reply"])
        agent = PlanAskAgent(
            name="tests", namespace="tests", description="d", llm_engine=engine, max_thinking_rounds=5,
        )
        agent.invoke({"prompt": "q"})
        assert len(engine.calls) == 2


class TestFullLifecycle:
    def test_invoke_returns_thinking_agent_result(self) -> None:
        agent = make_agent([batch_json(), "reply"])
        result = agent.invoke({"prompt": "q"})
        assert isinstance(result, ThinkingAgentResult)
        assert result.result == "reply"

    def test_async_invoke_thinks_then_replies(self) -> None:
        agent = make_agent([batch_json(), "reply"])
        result = asyncio.run(agent.async_invoke({"prompt": "q"}))
        assert isinstance(result, ThinkingAgentResult)
        assert result.result == "reply"

    def test_get_thoughts_returns_this_runs_thoughts_in_order(self) -> None:
        items = [
            thought_item(question="q1", answer="a1"),
            thought_item(question="q2", answer="a2"),
        ]
        agent = make_agent([batch_json(items), "reply"])
        result = agent.invoke({"prompt": "q"})
        thoughts = agent.get_thoughts(result.run_id)
        assert [t.question for t in thoughts] == ["q1", "q2"]
        assert [t.answer for t in thoughts] == ["a1", "a2"]
