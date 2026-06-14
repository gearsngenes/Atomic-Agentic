from __future__ import annotations

import asyncio
from typing import Any

import pytest

from atomic_agentic.tools.base import Tool
from atomic_agentic.core.Invokable import StructuredInvokable
from atomic_agentic.results.workflows import (
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


def structured_component(
    function: Any,
    *,
    name: str,
    output_schema: list[str],
) -> StructuredInvokable:
    tool = Tool(
        function=function,
        name=name,
        namespace="integration",
        description=f"Integration test tool: {name}.",
    )
    return StructuredInvokable(
        component=tool,
        name=f"structured_{name}",
        description=f"Structured integration component: {name}.",
        output_schema=output_schema,
        ignore_unhandled=True,
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


def make_text_branch() -> StructuredInvokable:
    return structured_component(
        uppercase_text,
        name="uppercase_text",
        output_schema=["branch", "payload"],
    )


def make_number_branch() -> StructuredInvokable:
    return structured_component(
        double_route_number,
        name="double_route_number",
        output_schema=["branch", "payload"],
    )


def make_mode_router() -> Tool:
    return Tool(
        function=route_by_mode,
        name="route_by_mode",
        namespace="integration",
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
            structured_component(
                title_branch,
                name="title_branch",
                output_schema=["title"],
            ),
            structured_component(
                keywords_branch,
                name="keywords_branch",
                output_schema=["keywords"],
            ),
            structured_component(
                summary_branch,
                name="summary_branch",
                output_schema=["summary"],
            ),
        ],
        parameters=["topic"],
        output_type=ParallelFlow.DICT,
        output_names=["title_branch", "keywords_branch", "summary_branch"],
    )


def make_sequential_parallel_report_flow() -> SequentialFlow:
    return SequentialFlow(
        name="sequential_parallel_report",
        description="Normalize a topic, enrich it in parallel, then merge a report.",
        steps=[
            structured_component(
                normalize_topic,
                name="normalize_topic",
                output_schema=["topic"],
            ),
            make_parallel_enrichment_flow(),
            structured_component(
                merge_report,
                name="merge_report",
                output_schema=["report"],
            ),
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
            structured_component(
                extract_text_payload,
                name="extract_text_payload",
                output_schema=["text_payload"],
            ),
            structured_component(
                label_text,
                name="label_text",
                output_schema=["text_result"],
            ),
        ],
    )


def make_number_sequence_branch() -> SequentialFlow:
    return SequentialFlow(
        name="number_sequence_branch",
        description="Number processing sequence branch.",
        steps=[
            structured_component(
                increment_number,
                name="increment_number",
                output_schema=["number"],
            ),
            structured_component(
                square_number,
                name="square_number",
                output_schema=["number_result"],
            ),
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
            structured_component(
                extract_title,
                name="extract_title",
                output_schema=["title"],
            ),
            structured_component(
                score_title,
                name="score_title",
                output_schema=["title", "title_score"],
            ),
        ],
    )


def make_tag_sequence_branch() -> SequentialFlow:
    return SequentialFlow(
        name="tag_sequence_branch",
        description="Extract and score article tags.",
        steps=[
            structured_component(
                extract_tags,
                name="extract_tags",
                output_schema=["tags"],
            ),
            structured_component(
                score_tags,
                name="score_tags",
                output_schema=["tags", "tag_score"],
            ),
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
        parameters=["article"],
        output_type=ParallelFlow.DICT,
        output_names=["title_branch", "tag_branch"],
    )


# --------------------------------------------------------------------------- #
# Sequential + Iterative helpers
# --------------------------------------------------------------------------- #
def initialize_score(score: int) -> dict[str, int]:
    return {
        "score": score,
    }


def improve_score(score: int) -> int:
    return score + 1


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
        namespace="integration",
        description="Approve once score is at least 3.",
    )

    return IterativeFlow(
        name="iterative_score_refinement",
        description="Improve a score until the judge approves it.",
        body_steps=[
            structured_component(
                improve_score,
                name="improve_score",
                output_schema=["score"],
            )
        ],
        judge=judge,
        approval_value=True,
        max_iterations=5,
        return_index=0,
        handoff_index=0,
        evaluate_index=0,
    )


def make_sequential_iterative_score_flow() -> SequentialFlow:
    return SequentialFlow(
        name="sequential_iterative_score",
        description="Initialize, iteratively improve, then finalize a score.",
        steps=[
            structured_component(
                initialize_score,
                name="initialize_score",
                output_schema=["score"],
            ),
            make_iterative_score_flow(),
            structured_component(
                finalize_score,
                name="finalize_score",
                output_schema=["final_score", "status"],
            ),
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
                structured_component(
                    finalize_routed_result,
                    name="finalize_routed_result",
                    output_schema=["final"],
                ),
            ],
        )

        result = flow.invoke({"mode": "text", "text": "hello world", "value": 4})

        assert isinstance(result, SequentialFlowResult)
        assert result.result == {"final": "text:HELLO WORLD"}

        step_results = flow.get_step_results(result.run_id)
        assert [r.result for r in step_results] == [
            {"branch": "text", "payload": "HELLO WORLD"},
            {"final": "text:HELLO WORLD"},
        ]

        routing_step_result = flow.get_step_result(result.run_id, 0)
        assert routing_step_result.result == {"branch": "text", "payload": "HELLO WORLD"}

        routing_checkpoint = routing_flow.checkpoints[0]
        assert isinstance(routing_checkpoint.result, RoutingFlowResult)
        assert routing_checkpoint.result.selected_key == 0
        assert len(routing_flow.branches[0].checkpoints) == 1
        assert len(routing_flow.branches[1].checkpoints) == 0

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

        step_results = flow.get_step_results(result.run_id)
        assert step_results[0].result == {"topic": "atomic agents"}
        assert step_results[1].result == expected_parallel_result
        assert step_results[2].result == result.result

        parallel_flow = flow.steps[1]
        assert isinstance(parallel_flow, ParallelFlow)
        parallel_result = parallel_flow.checkpoints[0].result
        assert isinstance(parallel_result, ParallelFlowResult)
        assert len(parallel_result.branch_runs) == 3

    def test_routing_can_select_sequential_branch(self) -> None:
        flow = make_routing_between_sequences_flow()

        result = flow.invoke({"mode": "number", "value": 4, "text": "ignored"})

        assert isinstance(result, RoutingFlowResult)
        assert result.result == {"number_result": 25}

        checkpoint = flow.get_checkpoint(result.run_id)
        assert checkpoint is not None
        assert checkpoint.result.selected_key == 1

        text_sequence = flow.branches[0]
        number_sequence = flow.branches[1]
        assert isinstance(text_sequence, SequentialFlow)
        assert isinstance(number_sequence, SequentialFlow)

        assert len(text_sequence.checkpoints) == 0
        assert len(number_sequence.checkpoints) == 1
        assert checkpoint.result.chosen_branch_run == number_sequence.checkpoints[-1].result.run_id

        selected_branch_results = number_sequence.get_step_results(
            number_sequence.checkpoints[-1].result.run_id
        )
        assert [r.result for r in selected_branch_results] == [
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

        assert len(title_sequence.checkpoints) == 1
        assert len(tag_sequence.checkpoints) == 1
        title_step_results = title_sequence.get_step_results(
            title_sequence.checkpoints[-1].result.run_id
        )
        assert [r.result for r in title_step_results] == [
            {"title": "Atomic"},
            {"title": "Atomic", "title_score": 6},
        ]
        tag_step_results = tag_sequence.get_step_results(
            tag_sequence.checkpoints[-1].result.run_id
        )
        assert [r.result for r in tag_step_results] == [
            {"tags": ["atomic", "agentic"]},
            {"tags": ["atomic", "agentic"], "tag_score": 2},
        ]

        checkpoint = flow.get_checkpoint(result.run_id)
        assert checkpoint is not None
        assert len(checkpoint.result.branch_runs) == 2

    def test_sequential_iterative_refinement_then_finalize(self) -> None:
        flow = make_sequential_iterative_score_flow()

        result = flow.invoke({"score": 0})

        assert isinstance(result, SequentialFlowResult)
        assert result.result == {
            "final_score": 3,
            "status": "approved",
        }
        step_results = flow.get_step_results(result.run_id)
        assert [r.result for r in step_results] == [
            {"score": 0},
            {"score": 3},
            {"final_score": 3, "status": "approved"},
        ]

        iterative_flow = flow.steps[1]
        assert isinstance(iterative_flow, IterativeFlow)
        iterative_result = iterative_flow.checkpoints[0].result
        assert isinstance(iterative_result, IterativeFlowResult)
        assert len(iterative_result.iteration_runs) == 3
        assert iterative_result.max_iterations == 5
        assert len(iterative_result.iteration_runs) < iterative_result.max_iterations

        pairs = iterative_flow.get_iteration_results(iterative_result.run_id)
        assert [judge_result.result for _, judge_result in pairs] == [
            False,
            False,
            True,
        ]


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

        step_results = flow.get_step_results(result.run_id)
        assert step_results[0].result == {"topic": "async atomic agents"}
        assert step_results[1].result == expected_parallel_result
        assert step_results[2].result == result.result

        parallel_flow = flow.steps[1]
        assert isinstance(parallel_flow, ParallelFlow)
        assert len(parallel_flow.checkpoints) == 1
        parallel_result = parallel_flow.checkpoints[0].result
        assert isinstance(parallel_result, ParallelFlowResult)
        assert len(parallel_result.branch_runs) == 3
