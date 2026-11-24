from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from atomic_agentic.agents import BasicAgent
from atomic_agentic.composites import Pipeline
from atomic_agentic.llm import OpenAIEngine
from atomic_agentic.models.results import AtomicResult


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
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))


def make_live_openai_agent() -> BasicAgent:
    return BasicAgent(
        name="live_openai_writer_agent",
        description="Live OpenAI writer agent for integration tests.",
        llm_engine=_openai_engine(),
        instructions=ROLE_PROMPT,
        record_enabled=False,
        pre_invoke=build_prompt,
        post_invoke=package_response,
        response_to_post="result",
    )


def make_live_openai_pipeline() -> tuple[BasicAgent, Pipeline]:
    agent = make_live_openai_agent()
    pipe = Pipeline(
        name="live_openai_writer_pipeline",
        description="Pipeline wrapping a live OpenAI Agent.",
        steps=[agent],
    )
    return agent, pipe


def _assert_live_agent_result(result: AtomicResult) -> None:
    assert isinstance(result, AtomicResult)
    assert set(result.result.keys()) == {"final", "length", "was_postprocessed"}
    assert isinstance(result.result["final"], str)
    assert result.result["final"].strip()
    assert isinstance(result.result["length"], int)
    assert result.result["length"] > 0
    assert result.result["was_postprocessed"] is True


class TestLiveAgentBasicPipeline:
    def test_live_openai_agent_can_be_wrapped_in_pipeline(
        self,
    ) -> None:
        _agent, pipe = make_live_openai_pipeline()

        result = pipe.invoke({"topic": "pytest integration tests", "tone": "clear"})

        _assert_live_agent_result(result)

    def test_live_openai_agent_pipeline_async(
        self,
    ) -> None:
        _agent, pipe = make_live_openai_pipeline()

        result = asyncio.run(
            pipe.async_invoke({"topic": "async pytest integration tests", "tone": "brief"})
        )

        _assert_live_agent_result(result)

    def test_live_openai_composed_pipeline_to_dict_does_not_expose_secrets(
        self,
    ) -> None:
        agent, pipe = make_live_openai_pipeline()

        pipe.invoke({"topic": "safe serialization", "tone": "plain"})
        data = pipe.to_dict()

        assert data["type"] == "Pipeline"
        assert data["name"] == "live_openai_writer_pipeline"
        assert data["step_count"] == 1

        agent_snapshot = data["steps"][0]["invokable"]
        assert agent_snapshot["type"] == "BasicAgent"
        assert agent_snapshot["name"] == agent.name
        assert agent_snapshot["llm"]["type"] == "OpenAIEngine"
        assert "api_key" not in str(data).lower()
        assert "secret" not in str(data).lower()
