from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from atomic_agentic.agents.basic import BasicAgent
from atomic_agentic.composites import Pipeline
from atomic_agentic.tools.base import Tool
from atomic_agentic.models.results import AtomicResult
from ..fake_engines import FakeLLMEngine, echo_latest_user


pytestmark = [pytest.mark.integration]

ROLE_PROMPT = "You are a deterministic integration-test writer."
TASK_PROMPT_START = "==== USER TASK PROMPT [START] ===="
TASK_PROMPT_END = "==== USER TASK PROMPT [END] ===="


def wrap(prompt: str) -> str:
    return f"{TASK_PROMPT_START}\n{prompt}\n{TASK_PROMPT_END}"


def build_prompt(topic: str, tone: str = "neutral") -> str:
    return f"Write about {topic} in a {tone} tone."


def package_response(result: str) -> dict[str, Any]:
    return {
        "final": result,
        "length": len(result),
        "was_postprocessed": True,
    }


def summarize_agent_result(
    final: str,
    length: int,
    was_postprocessed: bool,
) -> dict[str, Any]:
    return {
        "summary": f"{length}:{was_postprocessed}:{final}",
    }


def make_agent(
    *,
    engine: FakeLLMEngine | None = None,
    record_enabled: bool = False,
    records_window: int | None = None,
) -> BasicAgent:
    return BasicAgent(
        name="writer_agent",
        description="Deterministic writer integration-test agent.",
        llm_engine=engine or FakeLLMEngine(response_fn=echo_latest_user()),
        instructions=ROLE_PROMPT,
        record_enabled=record_enabled,
        records_window=records_window,
        pre_invoke=build_prompt,
        post_invoke=package_response,
        response_to_post="result",
    )


def make_agent_pipeline(
    *,
    engine: FakeLLMEngine | None = None,
    record_enabled: bool = False,
    records_window: int | None = None,
) -> tuple[FakeLLMEngine, BasicAgent, Pipeline]:
    resolved_engine = engine or FakeLLMEngine(response_fn=echo_latest_user())
    agent = make_agent(
        engine=resolved_engine,
        record_enabled=record_enabled,
        records_window=records_window,
    )
    pipe = Pipeline(
        name="writer_agent_pipeline",
        description="Pipeline wrapping an Agent.",
        steps=[agent],
    )
    return resolved_engine, agent, pipe


def make_summary_step() -> Tool:
    return Tool(
        function=summarize_agent_result,
        name="summarize_agent_result",
        description="Summarize the agent result.",
    )


class TestAgentBasicPipeline:
    def test_agent_can_be_wrapped_in_pipeline(self) -> None:
        engine, _agent, pipe = make_agent_pipeline()

        result = pipe.invoke({"topic": "pytest", "tone": "strict"})

        expected_prompt = "Write about pytest in a strict tone."
        expected_message = wrap(expected_prompt)
        expected_raw = f"ECHO: {expected_message}"

        assert isinstance(result, AtomicResult)
        assert result.result == {
            "final": expected_raw,
            "length": len(expected_raw),
            "was_postprocessed": True,
        }
        assert result.invoker_id == pipe.full_name
        assert engine.calls == [
            [
                {"role": "system", "content": ROLE_PROMPT},
                {"role": "user", "content": expected_message},
            ]
        ]

    def test_pipeline_result_is_the_agents_raw_response(self) -> None:
        _engine, agent, pipe = make_agent_pipeline()

        result = pipe.invoke({"topic": "agents", "tone": "concise"})

        direct_agent = make_agent()
        direct = direct_agent.invoke({"topic": "agents", "tone": "concise"})
        assert result.result == direct.result
        # This agent is constructed with record_enabled=False (the
        # make_agent_pipeline default), so nothing is stored.
        assert agent.get_conversation().records == []

    def test_agent_can_feed_pipeline_step(self) -> None:
        agent = make_agent()
        summary_step = make_summary_step()
        pipe = Pipeline(
            name="agent_summary_pipeline",
            description="Agent result feeding a downstream summary step.",
            steps=[agent, summary_step],
        )

        result = pipe.invoke({"topic": "composition", "tone": "direct"})

        expected_prompt = "Write about composition in a direct tone."
        expected_raw = f"ECHO: {wrap(expected_prompt)}"
        expected_second = {
            "summary": f"{len(expected_raw)}:True:{expected_raw}",
        }

        assert isinstance(result, AtomicResult)
        assert result.result == expected_second

    def test_wrapped_record_enabled_agent_preserves_history_across_pipeline_invokes(
        self,
    ) -> None:
        engine, agent, pipe = make_agent_pipeline(
            record_enabled=True,
        )

        first = pipe.invoke({"topic": "first topic", "tone": "plain"})
        second = pipe.invoke({"topic": "second topic", "tone": "plain"})

        first_prompt = "Write about first topic in a plain tone."
        second_prompt = "Write about second topic in a plain tone."
        assert first.result["final"] == f"ECHO: {wrap(first_prompt)}"
        assert second.result["final"] == f"ECHO: {wrap(second_prompt)}"
        assert len(engine.calls) == 2

        second_call_messages = engine.calls[1]
        assert [message["role"] for message in second_call_messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert second_call_messages[1]["content"] == first_prompt
        assert second_call_messages[2]["content"] == f"ECHO: {wrap(first_prompt)}"
        assert second_call_messages[3]["content"] == wrap(second_prompt)

    def test_async_agent_pipeline(self) -> None:
        engine, _agent, pipe = make_agent_pipeline()

        result = asyncio.run(
            pipe.async_invoke({"topic": "async pytest", "tone": "strict"})
        )

        expected_prompt = "Write about async pytest in a strict tone."
        expected_raw = f"ECHO: {wrap(expected_prompt)}"

        assert isinstance(result, AtomicResult)
        assert result.result == {
            "final": expected_raw,
            "length": len(expected_raw),
            "was_postprocessed": True,
        }
        assert len(engine.calls) == 1

    def test_composed_agent_pipeline_to_dict_exposes_agent_and_engine_snapshot(
        self,
    ) -> None:
        _engine, agent, pipe = make_agent_pipeline(
            record_enabled=True,
            records_window=1,
        )

        pipe.invoke({"topic": "serialization", "tone": "careful"})
        data = pipe.to_dict()

        assert data["type"] == "Pipeline"
        assert data["name"] == "writer_agent_pipeline"
        assert data["step_count"] == 1

        agent_snapshot = data["steps"][0]["invokable"]
        assert agent_snapshot["type"] == "BasicAgent"
        assert agent_snapshot["name"] == agent.name
        assert agent_snapshot["instructions"]["template"] == ROLE_PROMPT
        assert agent_snapshot["record_enabled"] is True
        assert agent_snapshot["records_window"] == 1
        assert agent_snapshot["pre_invoke"]["name"] == f"{agent.name}_pre_invoke"
        assert agent_snapshot["post_invoke"]["name"] == f"{agent.name}_post_invoke"
        assert agent_snapshot["llm"]["type"] == "FakeLLMEngine"
        assert "secret" not in str(data).lower()
        assert "api_key" not in str(data).lower()
