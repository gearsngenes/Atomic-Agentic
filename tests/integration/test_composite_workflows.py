from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from atomic_agentic.composites import Graph, Pipeline, Router
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results import GraphResult, RouterResult
from atomic_agentic.tools.base import Tool


pytestmark = [pytest.mark.integration]


def make_step(function: Callable[..., Any], *, name: str) -> Tool:
    return Tool(function=function, name=name, description=f"Integration step {name}.")


def kw_params(*names: str) -> list[ParamSpec]:
    return [
        ParamSpec(name=n, index=i, kind=ParamSpec.KEYWORD_ONLY, type=("Any",))
        for i, n in enumerate(names)
    ]


# ---------------------------------------------------------------------
# Router-then-finalize
# ---------------------------------------------------------------------
def kind_of(value: Any) -> str:
    return "number" if isinstance(value, (int, float)) else "text"


def handle_text(value: str) -> dict[str, Any]:
    return {"kind": "text", "payload": value.upper()}


def handle_number(value: float) -> dict[str, Any]:
    return {"kind": "number", "payload": value * 2}


def package(kind: str, payload: Any) -> dict[str, Any]:
    return {"summary": f"{kind}:{payload}"}


def make_router() -> Router:
    return Router(
        "kind_router",
        "Route by value type.",
        selector=make_step(kind_of, name="kind_of"),
        routes={
            "text": make_step(handle_text, name="handle_text"),
            "number": make_step(handle_number, name="handle_number"),
        },
    )


def make_route_pipeline() -> Pipeline:
    return Pipeline(
        "route_then_package",
        "Router, then package the result.",
        [make_router(), make_step(package, name="package")],
    )


# ---------------------------------------------------------------------
# Normalize, parallel enrichment, merge
# ---------------------------------------------------------------------
def normalize(text: str) -> dict[str, str]:
    return {"text": " ".join(text.split())}


def make_title(text: str) -> dict[str, str]:
    return {"title": text.split(".")[0].title()}


def make_keywords(text: str) -> dict[str, list[str]]:
    return {"keywords": sorted({w.strip(".,").lower() for w in text.split() if len(w) > 4})}


def make_summary(text: str) -> dict[str, str]:
    return {"summary": text[:20]}


def merge_report(title: str, keywords: list[str], summary: str) -> dict[str, str]:
    return {"report": f"{title} | {','.join(keywords)} | {summary}"}


def make_enrichment_graph() -> Graph:
    graph = Graph("enrich", "Parallel enrichment then merge.", returns="report")
    graph.add_node("merge_report", make_step(merge_report, name="merge_report"))
    graph.add_parallel(
        {
            "title": make_step(make_title, name="title"),
            "keywords": make_step(make_keywords, name="keywords"),
            "summary": make_step(make_summary, name="summary"),
        },
        fan_in="merge_report",
    )
    graph.set_start(["title", "keywords", "summary"])
    return graph


def make_enrichment_pipeline() -> Pipeline:
    return Pipeline(
        "normalize_enrich",
        "Normalize, then enrich in parallel.",
        [make_step(normalize, name="normalize"), make_enrichment_graph()],
    )


class TestCompositeWorkflows:
    def test_pipeline_routes_then_finalizes_selected_branch(self) -> None:
        pipe = make_route_pipeline()

        assert pipe.invoke({"value": "abc"}).result == {"summary": "text:ABC"}
        assert pipe.invoke({"value": 21}).result == {"summary": "number:42"}

        direct = make_router().invoke({"value": 21})
        assert isinstance(direct, RouterResult)
        assert direct.route == "number"
        assert direct.branch == "handle_number"
        assert direct.result == {"kind": "number", "payload": 42}

    def test_pipeline_parallel_enrichment_then_merge(self) -> None:
        pipe = make_enrichment_pipeline()

        result = pipe.invoke({"text": "Hello   World.  Pipelines are neat."})

        assert result.result == "Hello World | hello,neat,pipelines,world | Hello World. Pipelin"

    def test_router_selects_pipeline_branch(self) -> None:
        shout = Pipeline(
            "shout",
            "Strip then upper.",
            [make_step(lambda text: {"text": text.strip()}, name="strip"),
             make_step(lambda text: text.upper(), name="upper")],
        )
        reverse = Pipeline(
            "reverse",
            "Strip then reverse.",
            [make_step(lambda text: {"text": text.strip()}, name="strip2"),
             make_step(lambda text: text[::-1], name="reverse_text")],
        )
        router = Router(
            "mode_router",
            "Pick a pipeline by mode.",
            selector=make_step(lambda mode: mode, name="read_mode"),
            routes={"shout": shout, "reverse": reverse},
            parameters=kw_params("mode", "text"),
        )

        loud = router.invoke({"mode": "shout", "text": "  abc "})
        back = router.invoke({"mode": "reverse", "text": "  abc "})

        assert (loud.result, loud.branch, loud.route) == ("ABC", "shout", "shout")
        assert (back.result, back.branch, back.route) == ("cba", "reverse", "reverse")

    def test_graph_parallel_nodes_are_pipelines(self) -> None:
        short = Pipeline(
            "short_chain",
            "Two steps.",
            [make_step(lambda n: {"m": n + 1}, name="s1"),
             make_step(lambda m: {"short_result": m * 2}, name="s2")],
        )
        long = Pipeline(
            "long_chain",
            "Three steps.",
            [make_step(lambda n: {"a": n + 1}, name="l1"),
             make_step(lambda a: {"b": a * 3}, name="l2"),
             make_step(lambda b: {"long_result": b - 1}, name="l3")],
        )
        graph = Graph("two_chains", "Pipelines as parallel nodes.", {"short": short, "long": long},
                      start=["short", "long"])

        result = graph.invoke({"n": 4})

        assert isinstance(result, GraphResult)
        assert result.trace == (("short", "long"),)
        assert result.result["short_result"] == 10
        assert result.result["long_result"] == 14

    def test_pipeline_loop_then_finalize(self) -> None:
        def init(limit: int) -> dict[str, int]:
            return {"score": 0}

        def bump(score: int) -> dict[str, int]:
            return {"score": score + 1}

        def review(score: int, limit: int) -> dict[str, bool]:
            return {"approved": score >= limit}

        def finalize(score: int, approved: bool) -> dict[str, Any]:
            return {"final_score": score, "approved": approved}

        def build(limit_depth: int) -> Pipeline:
            loop = Graph("loop", "Bump until approved.", max_traversal_depth=limit_depth)
            loop.add_sequence(
                {"bump": make_step(bump, name="bump"), "review": make_step(review, name="review")},
                repeat_until={"approved": True},
            )
            loop.set_start("bump")
            return Pipeline(
                "init_loop_finalize",
                "Init, loop, finalize.",
                [make_step(init, name="init"), loop, make_step(finalize, name="finalize")],
            )

        assert build(10).invoke({"limit": 3}).result == {"final_score": 3, "approved": True}

        with pytest.raises(RuntimeError, match="max_traversal_depth"):
            build(6).invoke({"limit": 99})


class TestCompositeWorkflowsAsync:
    def test_async_pipeline_routes_then_finalizes_selected_branch(self) -> None:
        pipe = make_route_pipeline()

        text = asyncio.run(pipe.async_invoke({"value": "abc"}))
        number = asyncio.run(pipe.async_invoke({"value": 21}))

        assert text.result == {"summary": "text:ABC"}
        assert number.result == {"summary": "number:42"}

    def test_async_pipeline_parallel_enrichment_then_merge(self) -> None:
        pipe = make_enrichment_pipeline()

        result = asyncio.run(pipe.async_invoke({"text": "Hello   World.  Pipelines are neat."}))

        assert result.result == pipe.invoke({"text": "Hello   World.  Pipelines are neat."}).result
