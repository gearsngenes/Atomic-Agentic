from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from atomic_agentic.agents import BasicAgent
from atomic_agentic.composites import Graph, Pipeline, Router
from atomic_agentic.models.results import GraphResult, RouterResult
from atomic_agentic.tools.base import Tool

from ..fake_engines import FakeLLMEngine


pytestmark = [pytest.mark.integration]

APPROVED = "<<APPROVED>>"
OUTLINE = "OUTLINE: intro, body, close"


def last_user(messages: list[dict[str, str]]) -> str:
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


def make_agent(name: str, engine: FakeLLMEngine, pre: Any, post: Any = None) -> BasicAgent:
    return BasicAgent(
        name=name,
        description=f"Scripted {name} agent.",
        llm_engine=engine,
        instructions=f"You are the {name}.",
        record_enabled=False,
        pre_invoke=pre,
        post_invoke=post,
        response_to_post="result" if post is not None else (),
    )


def outliner_pre(topic: str) -> str:
    return f"Outline an essay about {topic}."


def outliner_post(result: str) -> dict[str, str]:
    return {"outline": result}


def writer_pre(outline: str, revision_notes: str = "") -> str:
    if revision_notes.strip():
        return f"Revise using these notes:\n{revision_notes}\nFollow this outline:\n{outline}"
    return f"Write a first draft.\nFollow this outline:\n{outline}"


def writer_post(result: str) -> dict[str, str]:
    return {"draft": result}


def critic_pre(draft: str) -> str:
    return f"Critique this draft:\n{draft}"


def critic_post(result: str) -> dict[str, Any]:
    return {"approved": APPROVED in result, "revision_notes": result}


@dataclass
class Rig:
    pipeline: Pipeline
    loop: Graph
    outliner_engine: FakeLLMEngine
    writer_engine: FakeLLMEngine
    critic_engine: FakeLLMEngine


def build_rig(*, approve_on_call: int | None = 2, max_traversal_depth: int = 12) -> Rig:
    """Outliner, then a write/critique loop; the critic approves on its Nth call."""
    outliner_engine = FakeLLMEngine(response_fn=lambda messages: OUTLINE)

    drafts = {"n": 0}

    def write(messages: list[dict[str, str]]) -> str:
        drafts["n"] += 1
        seen = [ln for ln in last_user(messages).splitlines() if ln.startswith("OUTLINE:")]
        return f"DRAFT {drafts['n']}: {seen[0]}"

    reviews = {"n": 0}

    def review(messages: list[dict[str, str]]) -> str:
        reviews["n"] += 1
        return APPROVED if reviews["n"] == approve_on_call else f"NOTES {reviews['n']}: add detail"

    writer_engine = FakeLLMEngine(response_fn=write)
    critic_engine = FakeLLMEngine(response_fn=review)

    outliner = make_agent("outliner", outliner_engine, outliner_pre, outliner_post)
    writer = make_agent("writer", writer_engine, writer_pre, writer_post)
    critic = make_agent("critic", critic_engine, critic_pre, critic_post)

    loop = Graph("write_critique_loop", "Write and critique.", returns="draft",
                 max_traversal_depth=max_traversal_depth)
    loop.add_sequence({"write": writer, "critique": critic}, repeat_until={"approved": True})
    loop.set_start("write")
    pipeline = Pipeline("outline_write_critique", "Outline, then write and critique.", [outliner, loop])
    return Rig(pipeline, loop, outliner_engine, writer_engine, critic_engine)


class TestAgentComposites:
    def test_outline_write_critique_returns_approved_draft(self) -> None:
        rig = build_rig()

        result = rig.pipeline.invoke({"topic": "mosasaurs"})

        assert result.result == f"DRAFT 2: {OUTLINE}"
        assert rig.writer_engine.call_count == 2
        first, second = (last_user(call) for call in rig.writer_engine.calls)
        assert OUTLINE in first
        assert "Revise" not in first
        assert "NOTES 1: add detail" in second
        assert OUTLINE in second

    def test_loop_trace_shows_two_iterations(self) -> None:
        rig = build_rig()

        result = rig.loop.invoke({"outline": OUTLINE})

        assert isinstance(result, GraphResult)
        assert result.trace == (("write",), ("critique",), ("write",), ("critique",))
        assert result.result == f"DRAFT 2: {OUTLINE}"

    def test_never_approving_critic_hits_depth_cap(self) -> None:
        rig = build_rig(approve_on_call=None, max_traversal_depth=4)

        with pytest.raises(RuntimeError, match="max_traversal_depth"):
            rig.pipeline.invoke({"topic": "mosasaurs"})

    def test_async_pipeline_matches_sync(self) -> None:
        sync_result = build_rig().pipeline.invoke({"topic": "mosasaurs"})
        async_rig = build_rig()

        async_result = asyncio.run(async_rig.pipeline.async_invoke({"topic": "mosasaurs"}))

        assert async_result.result == sync_result.result
        assert async_rig.writer_engine.call_count == 2

    def test_agent_as_router_selector_and_branch(self) -> None:
        selector_engine = FakeLLMEngine(
            response_fn=lambda m: "billing" if "invoice" in last_user(m) else "general"
        )
        reply_engine = FakeLLMEngine(response_fn=lambda m: "AGENT REPLY")
        selector = make_agent(
            "triager", selector_engine, lambda request: f"Classify: {request}",
            lambda result: result.strip(),
        )
        reply = make_agent("replier", reply_engine, lambda request: f"Reply to: {request}")
        billing = Tool(
            function=lambda request: f"BILLING TOOL: {request}",
            name="billing_tool",
            description="Plain billing branch.",
        )
        router = Router("triage", "Agent selects.", selector, {"billing": billing, "general": reply})

        billed = router.invoke({"request": "my invoice"})
        general = router.invoke({"request": "hello"})

        assert isinstance(billed, RouterResult)
        assert (billed.route, billed.branch, billed.result) == (
            "billing", "billing_tool", "BILLING TOOL: my invoice",
        )
        assert (general.route, general.branch, general.result) == ("general", "replier", "AGENT REPLY")
        assert reply_engine.call_count == 1
