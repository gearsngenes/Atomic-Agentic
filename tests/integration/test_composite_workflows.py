from __future__ import annotations

import asyncio
from typing import Any

import pytest

from atomic_agentic.tools.base import Tool
from atomic_agentic.models.results.workflows import (
    IterativeFlowResult,
    ParallelFlowResult,
    RoutingFlowResult,
    SequentialFlowResult,
)
from atomic_agentic.workflows.iterative import IterativeFlow
from atomic_agentic.workflows.parallel import ParallelFlow
from atomic_agentic.workflows.routing import RoutingFlow
from atomic_agentic.workflows.sequential import SequentialFlow


pytestmark = [pytest.mark.integration]


def make_step(function: Any, *, name: str) -> Tool:
    return Tool(
        function=function,
        name=name,
        description=f"Integration test tool: {name}.",
    )


# --------------------------------------------------------------------------- #
# Shared routing helpers
# --------------------------------------------------------------------------- #
def route_by_mode(mode: str, value: int = 0, text: str = "") -> int:
    return 0 if mode == "text" else 1


def uppercase_text(text: str) -> dict[str, Any]:
    return {
        "branch": "text",
        "payload": text.upper(),
    }


def double_route_number(value: int) -> dict[str, Any]:
    return {
        "branch": "number",
        "payload": value * 2,
    }


def finalize_routed_result(branch: str, payload: Any) -> dict[str, str]:
    return {
        "final": f"{branch}:{payload}",
    }


def make_text_branch() -> Tool:
    return make_step(uppercase_text, name="uppercase_text")


def make_number_branch() -> Tool:
    return make_step(double_route_number, name="double_route_number")


def make_mode_router() -> Tool:
    return Tool(
        function=route_by_mode,
        name="route_by_mode",
        description="Route text mode to branch 0 and all other modes to branch 1.",
    )


# --------------------------------------------------------------------------- #
# Sequential + Parallel helpers
# --------------------------------------------------------------------------- #
def normalize_topic(topic: str) -> dict[str, str]:
    return {
        "topic": " ".join(topic.strip().lower().split()),
    }


def title_branch(topic: str) -> dict[str, str]:
    return {
        "title": topic.title(),
    }


def keywords_branch(topic: str) -> dict[str, list[str]]:
    return {
        "keywords": topic.split(),
    }


def summary_branch(topic: str) -> dict[str, str]:
    return {
        "summary": f"Summary for {topic}",
    }


def merge_report(
    title_branch: dict[str, str],
    keywords_branch: dict[str, list[str]],
    summary_branch: dict[str, str],
) -> dict[str, dict[str, Any]]:
    return {
        "report": {
            **title_branch,
            **keywords_branch,
            **summary_branch,
        }
    }


def make_parallel_enrichment_flow() -> ParallelFlow:
    return ParallelFlow(
        name="parallel_enrichment",
        description="Enrich a normalized topic in parallel.",
        branches=[
            make_step(title_branch, name="title_branch"),
            make_step(keywords_branch, name="keywords_branch"),
            make_step(summary_branch, name="summary_branch"),
        ],
        result_mode=ParallelFlow.DICT,
        selected_branches=[0, 1, 2],
        result_keys=["title_branch", "keywords_branch", "summary_branch"],
    )


def make_sequential_parallel_report_flow() -> SequentialFlow:
    return SequentialFlow(
        name="sequential_parallel_report",
        description="Normalize a topic, enrich it in parallel, then merge a report.",
        steps=[
            make_step(normalize_topic, name="normalize_topic"),
            make_parallel_enrichment_flow(),
            make_step(merge_report, name="merge_report"),
        ],
    )


# --------------------------------------------------------------------------- #
# Routing -> Sequential branch helpers
# --------------------------------------------------------------------------- #
def extract_text_payload(text: str) -> dict[str, str]:
    return {
        "text_payload": text.strip().upper(),
    }


def label_text(text_payload: str) -> dict[str, str]:
    return {
        "text_result": f"text:{text_payload}",
    }


def increment_number(value: int) -> dict[str, int]:
    return {
        "number": value + 1,
    }


def square_number(number: int) -> dict[str, int]:
    return {
        "number_result": number * number,
    }


def make_text_sequence_branch() -> SequentialFlow:
    return SequentialFlow(
        name="text_sequence_branch",
        description="Text processing sequence branch.",
        steps=[
            make_step(extract_text_payload, name="extract_text_payload"),
            make_step(label_text, name="label_text"),
        ],
    )


def make_number_sequence_branch() -> SequentialFlow:
    return SequentialFlow(
        name="number_sequence_branch",
        description="Number processing sequence branch.",
        steps=[
            make_step(increment_number, name="increment_number"),
            make_step(square_number, name="square_number"),
        ],
    )


def make_routing_between_sequences_flow() -> RoutingFlow:
    return RoutingFlow(
        name="routing_between_sequences",
        description="Route to either a text sequence or number sequence.",
        router=make_mode_router(),
        branches=[
            make_text_sequence_branch(),
            make_number_sequence_branch(),
        ],
    )


# --------------------------------------------------------------------------- #
# Parallel -> Sequential branch helpers
# --------------------------------------------------------------------------- #
def extract_title(article: str) -> dict[str, str]:
    first_word = article.strip().split()[0]
    return {
        "title": first_word.title(),
    }


def score_title(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "title_score": len(title),
    }


def extract_tags(article: str) -> dict[str, list[str]]:
    return {
        "tags": article.strip().lower().split()[:2],
    }


def score_tags(tags: list[str]) -> dict[str, Any]:
    return {
        "tags": tags,
        "tag_score": len(tags),
    }


def make_title_sequence_branch() -> SequentialFlow:
    return SequentialFlow(
        name="title_sequence_branch",
        description="Extract and score an article title.",
        steps=[
            make_step(extract_title, name="extract_title"),
            make_step(score_title, name="score_title"),
        ],
    )


def make_tag_sequence_branch() -> SequentialFlow:
    return SequentialFlow(
        name="tag_sequence_branch",
        description="Extract and score article tags.",
        steps=[
            make_step(extract_tags, name="extract_tags"),
            make_step(score_tags, name="score_tags"),
        ],
    )


def make_parallel_sequence_branches_flow() -> ParallelFlow:
    return ParallelFlow(
        name="parallel_sequence_branches",
        description="Run two sequential analysis branches in parallel.",
        branches=[
            make_title_sequence_branch(),
            make_tag_sequence_branch(),
        ],
        result_mode=ParallelFlow.DICT,
        selected_branches=[0, 1],
        result_keys=["title_branch", "tag_branch"],
    )


# --------------------------------------------------------------------------- #
# Sequential + Iterative helpers
# --------------------------------------------------------------------------- #
def initialize_score(score: int) -> dict[str, int]:
    return {
        "score": score,
    }


def improve_score(score: int) -> dict[str, int]:
    return {"score": score + 1}


def approve_score_at_three(score: int) -> bool:
    return score >= 3


def finalize_score(score: int) -> dict[str, Any]:
    return {
        "final_score": score,
        "status": "approved",
    }


def make_iterative_score_flow() -> IterativeFlow:
    judge = Tool(
        function=approve_score_at_three,
        name="approve_score_at_three",
        description="Approve once score is at least 3.",
    )

    return IterativeFlow(
        name="iterative_score_refinement",
        description="Improve a score until the judge approves it.",
        body_steps=[
            make_step(improve_score, name="improve_score")
        ],
        max_iterations=5,
        checkers=[(judge, True)],
        result_setting_indices=[0],
        handoff_index=0,
    )


def make_sequential_iterative_score_flow() -> SequentialFlow:
    return SequentialFlow(
        name="sequential_iterative_score",
        description="Initialize, iteratively improve, then finalize a score.",
        steps=[
            make_step(initialize_score, name="initialize_score"),
            make_iterative_score_flow(),
            make_step(finalize_score, name="finalize_score"),
        ],
    )


class TestCompositeWorkflows:
    def test_sequential_routes_then_finalizes_selected_branch(self) -> None:
        routing_flow = RoutingFlow(
            name="mode_router_flow",
            description="Route normalized input to a mode-specific branch.",
            router=make_mode_router(),
            branches=[
                make_text_branch(),
                make_number_branch(),
            ],
        )
        flow = SequentialFlow(
            name="sequential_routing_finalizer",
            description="Route an input and finalize the selected branch result.",
            steps=[
                routing_flow,
                make_step(finalize_routed_result, name="finalize_routed_result"),
            ],
        )

        result = flow.invoke({"mode": "text", "text": "hello world", "value": 4})

        assert isinstance(result, SequentialFlowResult)
        assert result.result == {"final": "text:HELLO WORLD"}

        assert [r.result for r in result.trace] == [
            {"branch": "text", "payload": "HELLO WORLD"},
            {"final": "text:HELLO WORLD"},
        ]

        routing_result = result.trace[0]
        assert isinstance(routing_result, RoutingFlowResult)
        assert routing_result.result == {"branch": "text", "payload": "HELLO WORLD"}
        assert routing_result.selected_branch == 0
        assert len(routing_result.trace) == 2

    def test_sequential_parallel_enrichment_then_merge(self) -> None:
        flow = make_sequential_parallel_report_flow()

        result = flow.invoke({"topic": "  Atomic   Agents  "})

        expected_report = {
            "title": "Atomic Agents",
            "keywords": ["atomic", "agents"],
            "summary": "Summary for atomic agents",
        }
        expected_result = {"report": expected_report}
        expected_parallel_result = {
            "title_branch": {"title": "Atomic Agents"},
            "keywords_branch": {"keywords": ["atomic", "agents"]},
            "summary_branch": {"summary": "Summary for atomic agents"},
        }

        assert isinstance(result, SequentialFlowResult)
        assert result.result == expected_result

        assert result.trace[0].result == {"topic": "atomic agents"}
        assert result.trace[1].result == expected_parallel_result
        assert result.trace[2].result == result.result

        parallel_result = result.trace[1]
        assert isinstance(parallel_result, ParallelFlowResult)
        assert len(parallel_result.trace) == 3

    def test_routing_can_select_sequential_branch(self) -> None:
        flow = make_routing_between_sequences_flow()

        result = flow.invoke({"mode": "number", "value": 4, "text": "ignored"})

        assert isinstance(result, RoutingFlowResult)
        assert result.result == {"number_result": 25}
        assert result.selected_branch == 1

        text_sequence = flow.branches[0]
        number_sequence = flow.branches[1]
        assert isinstance(text_sequence, SequentialFlow)
        assert isinstance(number_sequence, SequentialFlow)

        selected_result = result.trace[1]
        assert isinstance(selected_result, SequentialFlowResult)
        assert [r.result for r in selected_result.trace] == [
            {"number": 5},
            {"number_result": 25},
        ]

    def test_parallel_can_run_sequential_branches(self) -> None:
        flow = make_parallel_sequence_branches_flow()

        result = flow.invoke(
            {"article": "Atomic Agentic makes workflows composable"}
        )

        assert isinstance(result, ParallelFlowResult)
        assert result.result == {
            "title_branch": {"title": "Atomic", "title_score": 6},
            "tag_branch": {"tags": ["atomic", "agentic"], "tag_score": 2},
        }

        title_sequence = flow.branches[0]
        tag_sequence = flow.branches[1]
        assert isinstance(title_sequence, SequentialFlow)
        assert isinstance(tag_sequence, SequentialFlow)

        assert len(result.trace) == 2
        title_result, tag_result = result.trace
        assert [r.result for r in title_result.trace] == [
            {"title": "Atomic"},
            {"title": "Atomic", "title_score": 6},
        ]
        assert [r.result for r in tag_result.trace] == [
            {"tags": ["atomic", "agentic"]},
            {"tags": ["atomic", "agentic"], "tag_score": 2},
        ]

    def test_sequential_iterative_refinement_then_finalize(self) -> None:
        flow = make_sequential_iterative_score_flow()

        result = flow.invoke({"score": 0})

        assert isinstance(result, SequentialFlowResult)
        assert result.result == {
            "final_score": 3,
            "status": "approved",
        }
        assert [r.result for r in result.trace] == [
            {"score": 0},
            {"score": 3},
            {"final_score": 3, "status": "approved"},
        ]

        iterative_result = result.trace[1]
        assert isinstance(iterative_result, IterativeFlowResult)
        assert iterative_result.exited_early is True
        assert iterative_result.iterations_completed == 3
        assert iterative_result.max_iterations == 5
        assert iterative_result.iterations_completed < iterative_result.max_iterations
        assert len(iterative_result.trace) == 6

        judge_results = [iterative_result.trace[i].result for i in range(1, 6, 2)]
        assert judge_results == [False, False, True]


class TestCompositeWorkflowAsync:
    def test_async_sequential_parallel_enrichment_then_merge(self) -> None:
        flow = make_sequential_parallel_report_flow()

        result = asyncio.run(
            flow.async_invoke({"topic": "  Async   Atomic Agents  "})
        )

        expected_report = {
            "title": "Async Atomic Agents",
            "keywords": ["async", "atomic", "agents"],
            "summary": "Summary for async atomic agents",
        }
        expected_result = {"report": expected_report}
        expected_parallel_result = {
            "title_branch": {"title": "Async Atomic Agents"},
            "keywords_branch": {"keywords": ["async", "atomic", "agents"]},
            "summary_branch": {"summary": "Summary for async atomic agents"},
        }

        assert isinstance(result, SequentialFlowResult)
        assert result.result == expected_result

        assert result.trace[0].result == {"topic": "async atomic agents"}
        assert result.trace[1].result == expected_parallel_result
        assert result.trace[2].result == result.result

        parallel_result = result.trace[1]
        assert isinstance(parallel_result, ParallelFlowResult)
        assert len(parallel_result.trace) == 3
