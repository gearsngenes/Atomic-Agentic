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
from atomic_agentic.composites import Graph, Pipeline
from atomic_agentic.llm import OpenAIEngine


pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.network,
    pytest.mark.slow,
]

APPROVED = "<<APPROVED>>"
MAX_DEPTH = 8


def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _live_tests_enabled() -> bool:
    _load_env()
    return os.getenv("AA_RUN_LIVE_LLM_TESTS") == "1"


def _skip_if_live_tests_disabled() -> None:
    if not _live_tests_enabled():
        pytest.skip("Set AA_RUN_LIVE_LLM_TESTS=1 to run live LLM integration tests.")


def _openai_engine() -> OpenAIEngine:
    _skip_if_live_tests_disabled()

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set.")

    try:
        engine = OpenAIEngine(
            model=os.getenv("AA_TEST_OPENAI_MODEL", "gpt-4o-mini"),
            timeout=60,
        )
        engine.update_runtime_parameters(temperature=0)
        return engine
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _agent(name: str, instructions: str, pre: Any, post: Any, engine: OpenAIEngine) -> BasicAgent:
    return BasicAgent(
        name=name,
        description=instructions,
        llm_engine=engine,
        instructions=instructions,
        record_enabled=False,
        pre_invoke=pre,
        post_invoke=post,
        response_to_post="result",
    )


def make_live_pipeline() -> Pipeline:
    engine = _openai_engine()

    def outliner_pre(topic: str) -> str:
        return f"Write a three-bullet outline for a two-sentence note about: {topic}"

    def writer_pre(outline: str, revision_notes: str = "") -> str:
        if revision_notes.strip():
            return f"Revise the note using these notes:\n{revision_notes}\nOutline:\n{outline}"
        return f"Write a two-sentence note following this outline:\n{outline}"

    def critic_pre(draft: str) -> str:
        return f"Review this note:\n{draft}"

    outliner = _agent(
        "live_outliner", "You outline tersely.", outliner_pre,
        lambda result: {"outline": result}, engine,
    )
    writer = _agent(
        "live_writer", "You write terse notes.", writer_pre,
        lambda result: {"draft": result}, engine,
    )
    critic = _agent(
        "live_critic",
        f"You are a lenient editor. If the note is acceptable reply ONLY with {APPROVED}.",
        critic_pre,
        lambda result: {"approved": APPROVED in result, "revision_notes": result},
        engine,
    )

    loop = Graph(
        "live_write_critique_loop",
        "Write and critique until approved.",
        returns="draft",
        max_traversal_depth=MAX_DEPTH,
    )
    loop.add_sequence({"write": writer, "critique": critic}, repeat_until={"approved": True})
    loop.set_start("write")
    return Pipeline("live_outline_write_critique", "Live outline, write, critique.", [outliner, loop])


def _run_or_skip_on_cap(run: Any) -> Any:
    try:
        return run()
    except RuntimeError as exc:
        if "max_traversal_depth" in str(exc):
            pytest.skip(f"Live critic never approved within the depth cap: {exc}")
        raise


class TestLiveAgentComposites:
    def test_live_outline_write_critique_pipeline(self) -> None:
        pipe = make_live_pipeline()

        result = _run_or_skip_on_cap(lambda: pipe.invoke({"topic": "pytest integration tests"}))

        assert isinstance(result.result, str)
        assert result.result.strip()

    def test_live_outline_write_critique_async(self) -> None:
        pipe = make_live_pipeline()

        result = _run_or_skip_on_cap(
            lambda: asyncio.run(pipe.async_invoke({"topic": "async pytest integration tests"}))
        )

        assert isinstance(result.result, str)
        assert result.result.strip()
