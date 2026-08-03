from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from atomic_agentic.agents.basic import BasicAgent
from atomic_agentic.tools.base import Tool
from atomic_agentic.core.Invokable import StructuredInvokable
from atomic_agentic.models.results.workflows import BasicFlowResult, SequentialFlowResult
from atomic_agentic.workflows.basic import BasicFlow
from atomic_agentic.workflows.sequential import SequentialFlow
from fake_engines import FakeLLMEngine, echo_latest_user


pytestmark = [pytest.mark.integration]

ROLE_PROMPT = "You are a deterministic integration-test writer."


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
    context_enabled: bool = False,
    records_window: int | None = None,
) -> BasicAgent:
    return BasicAgent(
        name="writer_agent",
        namespace="integration",
        description="Deterministic writer integration-test agent.",
        llm_engine=engine or FakeLLMEngine(response_fn=echo_latest_user()),
        role_prompt=ROLE_PROMPT,
        context_enabled=context_enabled,
        records_window=records_window,
        pre_invoke=build_prompt,
        post_invoke=package_response,
    )


def make_structured_agent(
    *,
    engine: FakeLLMEngine | None = None,
    context_enabled: bool = False,
    records_window: int | None = None,
) -> tuple[FakeLLMEngine, BasicAgent, StructuredInvokable]:
    resolved_engine = engine or FakeLLMEngine(response_fn=echo_latest_user())
    agent = make_agent(
        engine=resolved_engine,
        context_enabled=context_enabled,
        records_window=records_window,
    )
    structured_agent = StructuredInvokable(
        component=agent,
        name="structured_writer_agent",
        description="Structured writer agent integration wrapper.",
        output_schema=["final", "length", "was_postprocessed"],
    )
    return resolved_engine, agent, structured_agent


def make_basic_agent_flow(
    *,
    engine: FakeLLMEngine | None = None,
    context_enabled: bool = False,
    records_window: int | None = None,
) -> tuple[FakeLLMEngine, BasicAgent, StructuredInvokable, BasicFlow]:
    resolved_engine, agent, structured_agent = make_structured_agent(
        engine=engine,
        context_enabled=context_enabled,
        records_window=records_window,
    )
    flow = BasicFlow(
        component=structured_agent,
        name="writer_agent_basic_flow",
        description="BasicFlow wrapping a structured Agent.",
    )
    return resolved_engine, agent, structured_agent, flow


def make_summary_step() -> StructuredInvokable:
    tool = Tool(
        function=summarize_agent_result,
        name="summarize_agent_result",
        namespace="integration",
        description="Summarize the structured agent result.",
    )
    return StructuredInvokable(
        component=tool,
        name="structured_summary_step",
        description="Structured summary step.",
        output_schema=["summary"],
    )


class TestAgentStructuredBasicPipeline:
    def test_agent_can_be_structured_and_wrapped_in_basic_flow(self) -> None:
        engine, _agent, _structured_agent, flow = make_basic_agent_flow()

        result = flow.invoke({"topic": "pytest", "tone": "strict"})

        expected_prompt = "Write about pytest in a strict tone."
        expected_raw = f"ECHO: {expected_prompt}"

        assert isinstance(result, BasicFlowResult)
        assert result.result == {
            "final": expected_raw,
            "length": len(expected_raw),
            "was_postprocessed": True,
        }
        assert result.run_id == flow.checkpoints[-1].result.run_id
        assert len(flow.checkpoints) == 1
        assert engine.calls == [
            [
                {"role": "system", "content": ROLE_PROMPT},
                {"role": "user", "content": expected_prompt},
            ]
        ]

    def test_basic_flow_checkpoint_records_agent_raw_result(self) -> None:
        _engine, agent, structured_agent, flow = make_basic_agent_flow()

        result = flow.invoke({"topic": "agents", "tone": "concise"})
        checkpoint = flow.get_checkpoint(result.run_id)
        assert checkpoint is not None

        assert checkpoint.result.child_id == structured_agent.instance_id
        assert checkpoint.result.child_type == "StructuredInvokable"
        # Records are always stored regardless of context_enabled.
        assert len(agent.records) == 1

    def test_structured_agent_can_feed_sequential_flow_step(self) -> None:
        _engine, _agent, structured_agent = make_structured_agent()
        summary_step = make_summary_step()
        flow = SequentialFlow(
            name="agent_summary_sequence",
            namespace="integration",
            description="Agent result feeding a downstream summary step.",
            steps=[structured_agent, summary_step],
        )

        result = flow.invoke({"topic": "composition", "tone": "direct"})

        expected_prompt = "Write about composition in a direct tone."
        expected_raw = f"ECHO: {expected_prompt}"
        expected_first = {
            "final": expected_raw,
            "length": len(expected_raw),
            "was_postprocessed": True,
        }
        expected_second = {
            "summary": f"{len(expected_raw)}:True:{expected_raw}",
        }

        assert isinstance(result, SequentialFlowResult)
        assert result.result == expected_second
        step_results = flow.get_step_results(result.run_id)
        assert [r.result for r in step_results] == [
            expected_first,
            expected_second,
        ]
        assert flow.get_step_result(result.run_id, 0).result == expected_first
        assert flow.get_step_result(result.run_id, 1).result == expected_second

    def test_wrapped_context_enabled_agent_preserves_history_across_flow_invokes(
        self,
    ) -> None:
        engine, agent, _structured_agent, flow = make_basic_agent_flow(
            context_enabled=True,
        )

        first = flow.invoke({"topic": "first topic", "tone": "plain"})
        second = flow.invoke({"topic": "second topic", "tone": "plain"})

        assert first.result["final"] == "ECHO: Write about first topic in a plain tone."
        assert second.result["final"] == "ECHO: Write about second topic in a plain tone."
        assert len(flow.checkpoints) == 2
        assert len(engine.calls) == 2

        second_call_messages = engine.calls[1]
        assert [message["role"] for message in second_call_messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]
        assert second_call_messages[1]["content"] == "Write about first topic in a plain tone."
        assert second_call_messages[2]["content"] == "ECHO: Write about first topic in a plain tone."
        assert second_call_messages[3]["content"] == "Write about second topic in a plain tone."

    def test_async_structured_agent_basic_flow_pipeline_records_checkpoint(self) -> None:
        engine, _agent, _structured_agent, flow = make_basic_agent_flow()

        result = asyncio.run(
            flow.async_invoke({"topic": "async pytest", "tone": "strict"})
        )

        expected_prompt = "Write about async pytest in a strict tone."
        expected_raw = f"ECHO: {expected_prompt}"

        assert isinstance(result, BasicFlowResult)
        assert result.result == {
            "final": expected_raw,
            "length": len(expected_raw),
            "was_postprocessed": True,
        }
        assert len(flow.checkpoints) == 1
        assert flow.checkpoints[0].result.run_id == result.run_id
        assert len(engine.calls) == 1

    def test_composed_agent_pipeline_to_dict_exposes_agent_and_engine_snapshot(
        self,
    ) -> None:
        _engine, agent, structured_agent, flow = make_basic_agent_flow(
            context_enabled=True,
            records_window=1,
        )

        result = flow.invoke({"topic": "serialization", "tone": "careful"})
        data = flow.to_dict()

        assert data["type"] == "BasicFlow"
        assert data["name"] == "writer_agent_basic_flow"
        assert data["checkpoint_count"] == 1
        assert data["runs"] == [result.run_id]

        structured_snapshot = data["component"]
        assert structured_snapshot["type"] == "StructuredInvokable"
        assert structured_snapshot["name"] == structured_agent.name

        agent_snapshot = structured_snapshot["component"]
        assert agent_snapshot["type"] == "BasicAgent"
        assert agent_snapshot["name"] == agent.name
        assert agent_snapshot["role_prompt"] == ROLE_PROMPT
        assert agent_snapshot["context_enabled"] is True
        assert agent_snapshot["records_window"] == 1
        assert agent_snapshot["pre_invoke"]["name"] == "pre_invoke"
        assert agent_snapshot["post_invoke"]["name"] == "post_invoke"
        assert agent_snapshot["llm"]["type"] == "FakeLLMEngine"
        assert "secret" not in str(data).lower()
        assert "api_key" not in str(data).lower()