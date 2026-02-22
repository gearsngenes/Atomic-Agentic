"""
Atomic Agentic + LangGraph Integration Researcher Example
"""

from __future__ import annotations

from typing import Any, Mapping, TypedDict

from dotenv import load_dotenv

try:
    from langgraph.graph import StateGraph, END
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This example requires 'langgraph'. Install with: pip install langgraph") from exc

from researcher_agents import writer, critic
from researcher_tools import research_tool, iterator_tool, judge

load_dotenv()

APPROVAL_FLAG = "<<APPROVED>>"
MAX_ITERS = 3


# ---------------------------------------------------------------------
# 1) Define the state schema
# ---------------------------------------------------------------------
class ResearchState(TypedDict, total=False):
    query: str
    sources: list[str]
    draft: str
    revision_notes: str
    approved: bool
    iteration: int

# ---------------------------------------------------------------------
# 2) Define the graph structure
# ---------------------------------------------------------------------
def build_graph():
    # intantiate the graph
    graph = StateGraph(ResearchState)

    # add nodes (Atomic Tools & Agents)
    graph.add_node("research", research_tool.invoke)
    graph.add_node("writer", writer.invoke)
    graph.add_node("critic", critic.invoke)
    graph.add_node("judge", judge.invoke)
    graph.add_node("iterate", iterator_tool.invoke)

    # add edges (define flow)
    graph.set_entry_point("research")
    graph.add_edge("research", "writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", "judge")
    graph.add_edge("judge", "iterate")

    # Define conditional routing after iteration based on approval flag or max iterations
    def route_after_iter(state: ResearchState) -> Any:
        if state.get("approved") or state.get("iteration") >= MAX_ITERS:
            return END
        return "writer"

    # add conditional edges after iteration
    graph.add_conditional_edges(
        "iterate",
        route_after_iter,
        {"writer": "writer", END: END},
    )

    return graph

# ---------------------------------------------------------------------
# 2) Define main
# ---------------------------------------------------------------------
def main():
    # Seed state for the graph execution
    seed: ResearchState = {
        "query": "What are the main benefits and risks of CRISPR gene editing in medicine?",
        "iteration": 0,
        "revision_notes": "",
        "draft": "",
        "sources": [],
    }

    # Build and compile the graph
    graph = build_graph()
    app = graph.compile()

    # Invoke the graph with the seed state
    final_state = app.invoke(seed)

    print("\n================ FINAL DRAFT (LANGGRAPH) ================\n")
    print(final_state.get("draft", "(No draft returned)"))

# ---------------------------------------------------------------------
# 3) Execute main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
