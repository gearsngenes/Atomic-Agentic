"""
01_Graph.py

Graph basics with plain functions (no LLM, no keys).

Demonstrates:
- nodes as functions: unbound (a returned dict merges whole into state) and
  bound to a key with (function, "key")
- fixed edges, a shared start (fan-out) and a fan-in node
- two independent routers on one node (loop-or-finalize, escalate once stuck)
- a same-superstep write collision resolved by collision_handling_order
- `returns` as None, a str and a list, switched with set_returns
- `trace` (node names per superstep) and the max_traversal_depth safety net

Run from the repo root:
    python examples/Composite_Examples/01_Graph.py
"""

from __future__ import annotations

from atomic_agentic.composites import Graph

MAX_REVISIONS = 3


# ---------------------------------------------------------------------
# Node functions: each reads the state keys it names and returns a dict
# ---------------------------------------------------------------------
def draft(topic: str, revision: int = 0) -> dict:
    revision += 1
    return {"story": f"Draft {revision} about {topic}", "revision": revision}


def fact_check(revision: int, strictness: int) -> dict:
    return {"facts_ok": revision >= strictness, "checked_by": "fact_check"}


def style_check(revision: int, strictness: int) -> dict:
    return {"style_ok": revision >= strictness, "checked_by": "style_check"}


def triage(facts_ok: bool, style_ok: bool) -> dict:
    return {"approved": facts_ok and style_ok}


def finalize(story: str) -> str:
    return f"PUBLISHED: {story}"


def escalate(story: str) -> str:
    return f"ESCALATED to an editor: {story}"


# ---------------------------------------------------------------------
# Routers read the merged state and return a node name or None
# ---------------------------------------------------------------------
def loop_or_finalize(approved: bool, revision: int) -> str | None:
    if approved:
        return "finalize"
    return None if revision >= MAX_REVISIONS else "draft"


def escalate_if_stuck(approved: bool, revision: int) -> str | None:
    return "escalate" if not approved and revision >= MAX_REVISIONS else None


def build_review_graph() -> Graph:
    return Graph(
        name="story_review",
        description="Draft, check in parallel, triage, then publish or escalate.",
        nodes={
            "draft": draft,
            "fact_check": fact_check,
            "style_check": style_check,
            "triage": triage,
            # a str binding stores the whole (non-mapping) response under a key
            "finalize": (finalize, "outcome"),
            "escalate": (escalate, "outcome"),
        },
        edges=[
            # one source with two targets: both checkers run in the same superstep
            ("draft", "fact_check"),
            ("draft", "style_check"),
            # two targets into one node: triage waits for both and runs once
            ("fact_check", "triage"),
            ("style_check", "triage"),
            # two independent routers on one node
            ("triage", loop_or_finalize),
            ("triage", escalate_if_stuck),
        ],
        start="draft",
        # both checkers write "checked_by"; style_check wins the collision
        collision_handling_order=["style_check"],
        returns="outcome",
    )


def main() -> None:
    graph = build_review_graph()

    print("\n=== 1. Approving run (strictness 2) ===")
    result = graph.invoke({"topic": "mosasaurs", "strictness": 2})
    print("result:", result.result)
    print("trace :")
    for step, names in enumerate(result.trace, start=1):
        print(f"  superstep {step}: {names}")

    print("\n=== 2. Never-approving run (strictness 99) ===")
    result = graph.invoke({"topic": "mosasaurs", "strictness": 99})
    print("result:", result.result)
    print("trace :", [names for names in result.trace])

    print("\n=== 3. Collision order ===")
    graph.set_returns("checked_by")
    print("collision_handling_order =", graph.collision_handling_order)
    print("checked_by              =", graph.invoke({"topic": "t", "strictness": 1}).result)
    graph.set_collision_handling_order(None)
    print("collision_handling_order =", graph.collision_handling_order)
    print("checked_by              =", graph.invoke({"topic": "t", "strictness": 1}).result)

    print("\n=== 4. returns: None, str, list ===")
    inputs = {"topic": "mosasaurs", "strictness": 1}
    graph.set_returns(None)
    print("None :", graph.invoke(inputs).result)
    graph.set_returns("outcome")
    print("str  :", graph.invoke(inputs).result)
    graph.set_returns(["outcome", "revision"])
    print("list :", graph.invoke(inputs).result)

    print("\n=== 5. max_traversal_depth ===")
    ping_pong = Graph(
        name="ping_pong",
        description="Two nodes that point at each other forever.",
        nodes={"ping": lambda: {"hit": "ping"}, "pong": lambda: {"hit": "pong"}},
        edges=[("ping", "pong"), ("pong", "ping")],
        start="ping",
        max_traversal_depth=5,
    )
    try:
        ping_pong.invoke({})
    except RuntimeError as exc:
        print("RuntimeError:", exc)


if __name__ == "__main__":
    main()
