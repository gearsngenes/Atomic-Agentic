from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from conftest import ScriptedLLMEngine

from atomic_agentic.agents.selfask import SelfAskAgent
from atomic_agentic.exceptions import ThinkingAgentError
from atomic_agentic.models.results import ThinkingAgentResult


def round_json(
    *,
    observation: str | None = None,
    question: str = "What is needed?",
    answer: str = "An answer.",
    keep_thinking: bool = False,
    **overrides: Any,
) -> str:
    payload: dict[str, Any] = {
        "observation": observation,
        "question": question,
        "answer": answer,
        "keep_thinking": keep_thinking,
    }
    payload.update(overrides)
    return json.dumps(payload)


def make_agent(
    responses: list[str],
    *,
    max_thinking_rounds: int = 3,
    generation_retries: int = 0,
    **kwargs: Any,
) -> SelfAskAgent:
    return SelfAskAgent(
        name="tests",
        namespace="tests",
        description="SelfAskAgent under test.",
        llm_engine=ScriptedLLMEngine(responses),
        max_thinking_rounds=max_thinking_rounds,
        generation_retries=generation_retries,
        **kwargs,
    )


class TestConstruction:
    def test_registers_fixed_self_ask_prompt(self) -> None:
        agent = make_agent([round_json(), "reply"])
        assert "self_ask" in agent._system_prompts
        assert agent._system_prompts["self_ask"].description == "SelfAskAgent self-questioning prompt."

    def test_role_prompt_still_registered(self) -> None:
        agent = make_agent([round_json(), "reply"])
        assert "role" in agent._system_prompts

    def test_no_new_constructor_parameters_beyond_thinking_agent(self) -> None:
        # role_prompt, role_description, max_thinking_rounds, generation_retries,
        # thoughts_window all come straight from ThinkingAgent.
        agent = make_agent(
            [round_json(), "reply"],
            role_prompt="Be a helpful assistant.",
            role_description="Answering trivia questions.",
            thoughts_window=None,
        )
        assert agent.role_description == "Answering trivia questions."

    def test_max_thinking_rounds_none_raises(self) -> None:
        # Unlike PlanAskAgent, SelfAskAgent's max_thinking_rounds is the
        # only backstop guaranteeing its live, iterative loop terminates --
        # None (unlimited) must never be permitted here.
        with pytest.raises(TypeError, match="max_thinking_rounds"):
            make_agent([round_json(), "reply"], max_thinking_rounds=None)  # type: ignore[arg-type]


class TestSelfAskRoundValidation:
    @pytest.mark.parametrize(
        "missing_key", ["observation", "question", "answer", "keep_thinking"]
    )
    def test_missing_required_key_triggers_retry_feedback(self, missing_key: str) -> None:
        bad = json.loads(round_json())
        del bad[missing_key]
        agent = make_agent([json.dumps(bad), round_json(), "reply"], generation_retries=1)
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"

    def test_unsupported_extra_key_triggers_retry_feedback(self) -> None:
        bad = json.loads(round_json())
        bad["extra"] = "not allowed"
        agent = make_agent([json.dumps(bad), round_json(), "reply"], generation_retries=1)
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"

    def test_observation_accepts_null(self) -> None:
        agent = make_agent([round_json(observation=None), "reply"])
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"
        assert agent.get_thoughts()[0].observation is None

    def test_observation_accepts_string(self) -> None:
        agent = make_agent([round_json(observation="a note"), "reply"])
        agent.invoke({"prompt": "q"})
        assert agent.get_thoughts()[0].observation == "a note"

    def test_observation_rejects_non_string_non_null(self) -> None:
        bad = json.loads(round_json())
        bad["observation"] = 123
        agent = make_agent([json.dumps(bad), round_json(), "reply"], generation_retries=1)
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"

    @pytest.mark.parametrize("field", ["question", "answer"])
    def test_empty_string_field_triggers_retry_feedback(self, field: str) -> None:
        bad = json.loads(round_json())
        bad[field] = "   "
        agent = make_agent([json.dumps(bad), round_json(), "reply"], generation_retries=1)
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"

    @pytest.mark.parametrize("value", [1, 0, "true", None])
    def test_keep_thinking_rejects_non_bool(self, value: Any) -> None:
        bad = json.loads(round_json())
        bad["keep_thinking"] = value
        agent = make_agent([json.dumps(bad), round_json(), "reply"], generation_retries=1)
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"


class TestGenerationRetry:
    def test_json_decode_failure_retries_then_succeeds(self) -> None:
        agent = make_agent(["not json", round_json(), "reply"], generation_retries=1)
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"

    def test_json_decode_exhaustion_raises_thinking_agent_error(self) -> None:
        agent = make_agent(["not json", "still not json"], generation_retries=0)
        with pytest.raises(ThinkingAgentError, match="generation retry budget exhausted"):
            agent.invoke({"prompt": "q"})

    def test_schema_validation_exhaustion_raises_thinking_agent_error(self) -> None:
        bad = json.loads(round_json())
        del bad["answer"]
        agent = make_agent([json.dumps(bad), json.dumps(bad)], generation_retries=0)
        with pytest.raises(ThinkingAgentError, match="generation retry budget exhausted"):
            agent.invoke({"prompt": "q"})

    def test_retries_used_increments_across_thinking_rounds(self) -> None:
        # Round 1 fails once then succeeds; round 2 fails once then succeeds --
        # retries_used is cumulative across the whole run (ThinkingTask-level).
        agent = make_agent(
            [
                "not json",
                round_json(keep_thinking=True),
                "not json",
                round_json(keep_thinking=False),
                "reply",
            ],
            generation_retries=2,
        )
        result = agent.invoke({"prompt": "q"})
        assert result.result == "reply"
        assert result.thoughts_end - result.thoughts_start == 2


class TestContinuation:
    def test_keep_thinking_false_stops_after_one_round(self) -> None:
        agent = make_agent([round_json(keep_thinking=False), "reply"], max_thinking_rounds=10)
        result = agent.invoke({"prompt": "q"})
        assert result.thoughts_end - result.thoughts_start == 1
        assert result.result == "reply"

    def test_max_thinking_rounds_forces_reply_without_raising(self) -> None:
        agent = make_agent(
            [
                round_json(keep_thinking=True),
                round_json(keep_thinking=True),
                "forced reply",
            ],
            max_thinking_rounds=2,
        )
        result = agent.invoke({"prompt": "q"})
        assert result.result == "forced reply"
        assert result.thoughts_end - result.thoughts_start == 2

    def test_multi_round_thoughts_persist_in_order(self) -> None:
        agent = make_agent(
            [
                round_json(question="q1", answer="a1", keep_thinking=True),
                round_json(question="q2", answer="a2", keep_thinking=False),
                "reply",
            ],
            max_thinking_rounds=5,
        )
        agent.invoke({"prompt": "q"})
        thoughts = agent.get_thoughts()
        assert [t.question for t in thoughts] == ["q1", "q2"]
        assert [t.answer for t in thoughts] == ["a1", "a2"]


class TestTaskMessagesClearing:
    def test_task_messages_cleared_after_successful_round(self) -> None:
        agent = make_agent([round_json(keep_thinking=False), "reply"])
        task = agent._initialize_task(turns=[], prompt="q", inputs={"prompt": "q"})
        agent._think(task)
        assert task.task_messages == []

    def test_next_round_snapshot_reflects_prior_thought(self) -> None:
        engine = ScriptedLLMEngine(
            [
                round_json(question="q1", answer="a1", keep_thinking=True),
                round_json(question="q2", answer="a2", keep_thinking=False),
                "reply",
            ]
        )
        agent = SelfAskAgent(
            name="tests", namespace="tests", description="d", llm_engine=engine, max_thinking_rounds=5,
        )
        agent.invoke({"prompt": "q"})
        second_round_messages = engine.calls[1]
        contents = [m["content"] for m in second_round_messages]
        assert any("a1" in c for c in contents)


class TestFullLifecycle:
    def test_invoke_returns_thinking_agent_result(self) -> None:
        agent = make_agent([round_json(), "reply"])
        result = agent.invoke({"prompt": "q"})
        assert isinstance(result, ThinkingAgentResult)
        assert result.result == "reply"

    def test_async_invoke_thinks_then_replies(self) -> None:
        agent = make_agent([round_json(), "reply"])
        result = asyncio.run(agent.async_invoke({"prompt": "q"}))
        assert isinstance(result, ThinkingAgentResult)
        assert result.result == "reply"

    def test_get_thoughts_returns_this_runs_thoughts(self) -> None:
        agent = make_agent([round_json(question="q1", answer="a1"), "reply"])
        result = agent.invoke({"prompt": "q"})
        thoughts = agent.get_thoughts(result.run_id)
        assert len(thoughts) == 1
        assert thoughts[0].question == "q1"
        assert thoughts[0].answer == "a1"

    def test_first_round_always_happens_even_with_trivial_prompt(self) -> None:
        # No zero-round shortcut: _initialize_task always seeds the self-ask
        # phase, so at least one LLM call for a thought always happens.
        engine = ScriptedLLMEngine([round_json(keep_thinking=False), "reply"])
        agent = SelfAskAgent(
            name="tests", namespace="tests", description="d", llm_engine=engine, max_thinking_rounds=1,
        )
        agent.invoke({"prompt": "q"})
        assert len(engine.calls) == 2
