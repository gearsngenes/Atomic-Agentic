from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from atomic_agentic.agents.base import Agent
from atomic_agentic.engines.LLMEngines import LLMEngine
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.StructuredInvokable import StructuredInvokable
from atomic_agentic.workflows.base import FlowResultDict
from atomic_agentic.workflows.basic import BasicFlow
from atomic_agentic.workflows.sequential import SequentialFlow


pytestmark = [pytest.mark.integration]

ROLE_PROMPT = "You are a deterministic integration-test writer."


class StatefulEchoLLMEngine(LLMEngine):
    """Deterministic LLMEngine used for integration tests.

    This engine records every normalized message batch and returns the latest
    user message with a stable prefix. It exercises the real Agent -> Engine
    contract without using a network provider.
    """

    def __init__(self, *, prefix: str = "ECHO", **kwargs: Any) -> None:
        super().__init__(
            name="stateful_echo_engine",
            description="Stateful echo engine for integration tests.",
            **kwargs,
        )
        self.prefix = prefix
        self.calls: list[list[dict[str, str]]] = []

    def _build_provider_payload(
        self,
        messages: list[dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        copied_messages = [dict(message) for message in messages]
        self.calls.append(copied_messages)

        latest_user = next(
            (
                message["content"]
                for message in reversed(copied_messages)
                if message["role"] == "user"
            ),
            "",
        )

        return {
            "latest_user": latest_user,
            "message_count": len(copied_messages),
        }

    def _call_provider(self, payload: Any) -> Any:
        return payload

    def _extract_text(self, response: Any) -> str:
        return f"{self.prefix}: {response['latest_user']}"

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        return {"path": path}

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        return None


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
    engine: StatefulEchoLLMEngine | None = None,
    context_enabled: bool = False,
    history_window: int | None = None,
) -> Agent:
    return Agent(
        name="writer_agent",
        description="Deterministic writer integration-test agent.",
        llm_engine=engine or StatefulEchoLLMEngine(),
        role_prompt=ROLE_PROMPT,
        context_enabled=context_enabled,
        history_window=history_window,
        pre_invoke=build_prompt,
        post_invoke=package_response,
    )


def make_structured_agent(
    *,
    engine: StatefulEchoLLMEngine | None = None,
    context_enabled: bool = False,
    history_window: int | None = None,
) -> tuple[StatefulEchoLLMEngine, Agent, StructuredInvokable]:
    resolved_engine = engine or StatefulEchoLLMEngine()
    agent = make_agent(
        engine=resolved_engine,
        context_enabled=context_enabled,
        history_window=history_window,
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
    engine: StatefulEchoLLMEngine | None = None,
    context_enabled: bool = False,
    history_window: int | None = None,
) -> tuple[StatefulEchoLLMEngine, Agent, StructuredInvokable, BasicFlow]:
    resolved_engine, agent, structured_agent = make_structured_agent(
        engine=engine,
        context_enabled=context_enabled,
        history_window=history_window,
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

        assert isinstance(result, FlowResultDict)
        assert result == {
            "final": expected_raw,
            "length": len(expected_raw),
            "was_postprocessed": True,
        }
        assert result.run_id == flow.latest_run
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

        metadata = checkpoint.metadata

        assert metadata.kind == "basic"
        assert metadata.child_is_workflow is False
        assert metadata.child_id == structured_agent.instance_id
        assert metadata.child_full_name == structured_agent.full_name
        assert metadata.has_child_raw_result is True
        assert metadata.child_raw_result == dict(result)
        assert metadata.child_raw_result_type == "dict"
        assert agent.history == []

    def test_structured_agent_can_feed_sequential_flow_step(self) -> None:
        _engine, _agent, structured_agent = make_structured_agent()
        summary_step = make_summary_step()
        flow = SequentialFlow(
            name="agent_summary_sequence",
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

        assert isinstance(result, FlowResultDict)
        assert result == expected_second
        assert flow.get_step_results(result.run_id) == [
            expected_first,
            expected_second,
        ]
        assert flow.get_step_result(result.run_id, 0) == expected_first
        assert flow.get_step_result(result.run_id, 1) == expected_second

    def test_wrapped_context_enabled_agent_preserves_history_across_flow_invokes(
        self,
    ) -> None:
        engine, agent, _structured_agent, flow = make_basic_agent_flow(
            context_enabled=True,
        )

        first = flow.invoke({"topic": "first topic", "tone": "plain"})
        second = flow.invoke({"topic": "second topic", "tone": "plain"})

        assert first["final"] == "ECHO: Write about first topic in a plain tone."
        assert second["final"] == "ECHO: Write about second topic in a plain tone."
        assert len(flow.checkpoints) == 2
        assert len(engine.calls) == 2

        assert [message["role"] for message in agent.history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]

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

        assert isinstance(result, FlowResultDict)
        assert result == {
            "final": expected_raw,
            "length": len(expected_raw),
            "was_postprocessed": True,
        }
        assert len(flow.checkpoints) == 1
        assert flow.checkpoints[0].run_id == result.run_id
        assert len(engine.calls) == 1

    def test_composed_agent_pipeline_to_dict_exposes_agent_and_engine_snapshot(
        self,
    ) -> None:
        _engine, agent, structured_agent, flow = make_basic_agent_flow(
            context_enabled=True,
            history_window=1,
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
        assert agent_snapshot["type"] == "Agent"
        assert agent_snapshot["name"] == agent.name
        assert agent_snapshot["role_prompt"] == ROLE_PROMPT
        assert agent_snapshot["context_enabled"] is True
        assert agent_snapshot["history_window"] == 1
        assert agent_snapshot["pre_invoke"]["name"] == "pre_invoke"
        assert agent_snapshot["post_invoke"]["name"] == "post_invoke"
        assert agent_snapshot["llm"]["type"] == "StatefulEchoLLMEngine"
        assert "secret" not in str(data).lower()
        assert "api_key" not in str(data).lower()