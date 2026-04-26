from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from atomic_agentic.agents.base import Agent
from atomic_agentic.engines.LLMEngines import OpenAIEngine
from atomic_agentic.workflows.StructuredInvokable import StructuredInvokable
from atomic_agentic.workflows.base import FlowResultDict
from atomic_agentic.workflows.basic import BasicFlow


pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.network,
    pytest.mark.slow,
]

ROLE_PROMPT = "You are a terse integration-test assistant."


def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _live_tests_enabled() -> bool:
    _load_env()
    return os.getenv("AA_RUN_LIVE_LLM_TESTS") == "1"


def _skip_if_live_tests_disabled() -> None:
    if not _live_tests_enabled():
        pytest.skip("Set AA_RUN_LIVE_LLM_TESTS=1 to run live LLM integration tests.")


def build_prompt(topic: str, tone: str = "short") -> str:
    return (
        f"Write one short sentence about {topic}. "
        f"Use a {tone} tone. Do not include markdown."
    )


def package_response(result: str) -> dict[str, Any]:
    return {
        "final": result,
        "length": len(result),
        "was_postprocessed": True,
    }


def _openai_engine() -> OpenAIEngine:
    _skip_if_live_tests_disabled()

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set.")

    try:
        return OpenAIEngine(
            model=os.getenv("AA_TEST_OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            timeout_seconds=60,
            max_retries=0,
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))


def make_live_openai_agent() -> Agent:
    return Agent(
        name="live_openai_writer_agent",
        description="Live OpenAI writer agent for integration tests.",
        llm_engine=_openai_engine(),
        role_prompt=ROLE_PROMPT,
        context_enabled=False,
        pre_invoke=build_prompt,
        post_invoke=package_response,
    )


def make_live_openai_basic_flow() -> tuple[Agent, StructuredInvokable, BasicFlow]:
    agent = make_live_openai_agent()
    structured_agent = StructuredInvokable(
        component=agent,
        name="structured_live_openai_writer_agent",
        description="Structured live OpenAI writer agent.",
        output_schema=["final", "length", "was_postprocessed"],
    )
    flow = BasicFlow(
        component=structured_agent,
        name="live_openai_writer_basic_flow",
        description="BasicFlow wrapping a structured live OpenAI Agent.",
    )
    return agent, structured_agent, flow


def _assert_live_structured_result(result: Any) -> None:
    assert isinstance(result, FlowResultDict)
    assert set(result.keys()) == {"final", "length", "was_postprocessed"}
    assert isinstance(result["final"], str)
    assert result["final"].strip()
    assert isinstance(result["length"], int)
    assert result["length"] > 0
    assert result["was_postprocessed"] is True


class TestLiveAgentStructuredBasicPipeline:
    def test_live_openai_agent_can_be_structured_and_wrapped_in_basic_flow(
        self,
    ) -> None:
        _agent, _structured_agent, flow = make_live_openai_basic_flow()

        result = flow.invoke({"topic": "pytest integration tests", "tone": "clear"})

        _assert_live_structured_result(result)
        assert result.run_id == flow.latest_run
        assert len(flow.checkpoints) == 1

        checkpoint = flow.get_checkpoint(result.run_id)
        assert checkpoint is not None
        assert checkpoint.result == dict(result)
        assert checkpoint.metadata.kind == "basic"
        assert checkpoint.metadata.child_is_workflow is False
        assert checkpoint.metadata.has_child_raw_result is True
        assert checkpoint.metadata.child_raw_result == dict(result)
        assert checkpoint.metadata.child_raw_result_type == "dict"

    def test_live_openai_structured_agent_basic_flow_async_pipeline(
        self,
    ) -> None:
        _agent, _structured_agent, flow = make_live_openai_basic_flow()

        result = asyncio.run(
            flow.async_invoke({"topic": "async pytest integration tests", "tone": "brief"})
        )

        _assert_live_structured_result(result)
        assert result.run_id == flow.latest_run
        assert len(flow.checkpoints) == 1

        checkpoint = flow.get_checkpoint(result.run_id)
        assert checkpoint is not None
        assert checkpoint.result == dict(result)
        assert checkpoint.metadata.kind == "basic"
        assert checkpoint.metadata.child_is_workflow is False

    def test_live_openai_composed_pipeline_to_dict_does_not_expose_secrets(
        self,
    ) -> None:
        agent, structured_agent, flow = make_live_openai_basic_flow()

        result = flow.invoke({"topic": "safe serialization", "tone": "plain"})
        data = flow.to_dict()

        assert data["type"] == "BasicFlow"
        assert data["name"] == "live_openai_writer_basic_flow"
        assert data["checkpoint_count"] == 1
        assert data["runs"] == [result.run_id]

        structured_snapshot = data["component"]
        assert structured_snapshot["type"] == "StructuredInvokable"
        assert structured_snapshot["name"] == structured_agent.name

        agent_snapshot = structured_snapshot["component"]
        assert agent_snapshot["type"] == "Agent"
        assert agent_snapshot["name"] == agent.name
        assert agent_snapshot["llm"]["type"] == "OpenAIEngine"
        assert "api_key" not in str(data).lower()
        assert "secret" not in str(data).lower()