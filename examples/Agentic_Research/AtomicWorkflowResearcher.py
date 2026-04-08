"""
Atomic-only example:
- Research Tool (Tavily SDK): query -> sources[list[str]] (clean formatted passages + URLs)
- MakerCheckerFlow: writer (maker) + critic (checker) + judge (early stop on <<APPROVED>>)
- SequentialFlow: research -> makerchecker
- Prints the final APA-style report

Requirements:
- pip install tavily-python python-dotenv
- TAVILY_API_KEY in environment or .env
- OPENAI_API_KEY in environment or .env  (for OpenAIEngine)
"""

from __future__ import annotations

from researcher_agents import writer, critic
from researcher_tools import research_tool, judge
from atomic_agentic.workflows import SequentialFlow
from atomic_agentic.workflows import IterativeFlow
from atomic_agentic import StructuredInvokable

MAX_REVISIONS = 3

# ---------------------------------------------------------------------
# 5) MakerCheckerFlow and SequentialFlow wiring
# ---------------------------------------------------------------------
maker_checker = IterativeFlow(
    name="research_report_makerchecker",
    description="Iteratively refine an APA report with early-stop approval.",
    body_steps=[
        StructuredInvokable(writer, output_schema=["draft"]),
        StructuredInvokable(critic, output_schema=["revision_notes"]),],
    judge=judge,
    return_index=0,
    max_iterations=MAX_REVISIONS,
)

# ------------------------------------------------------
# SequentialFlow to chain research -> makerchecker
# ------------------------------------------------------
flow = SequentialFlow(
    name="atomic_researcher_flow",
    description="Atomic workflow chaining Tavily research with iterative maker-checker refinement.",
    steps=[StructuredInvokable(research_tool, output_schema=["query", "sources"]), maker_checker],
)

def main() -> None:
    inputs = {"query": "What is the latest discovery/news regarding mosasaurs, and their placement within the clade toxicofera?"}
    final = flow.invoke(inputs)

    print("\n================ FINAL DRAFT (ATOMIC ONLY) ================\n")
    print(final["draft"])


if __name__ == "__main__":
    main()
