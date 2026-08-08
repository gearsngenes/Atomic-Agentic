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
from atomic_agentic.workflows import IterativeFlow, CheckerSpec

MAX_REVISIONS = 3

# ---------------------------------------------------------------------
# 5) MakerCheckerFlow and SequentialFlow wiring
# ---------------------------------------------------------------------
maker_checker = IterativeFlow(
    name="research_report_makerchecker",
    namespace="research",
    description="Iteratively refine an APA report with early-stop approval.",
    body_steps=[writer, critic],
    max_iterations=MAX_REVISIONS,
    result_setting_indices=[0],
    # handoff_index defaults to the last body step (critic) -- unchanged.
)
# judge_approved (researcher_tools.py) returns {"approved": bool}, not a
# bare bool -- approval_value must match that mapping shape exactly.
maker_checker.add_checker(index=1, judge=judge, approval_value={"approved": True}) 

# ------------------------------------------------------
# SequentialFlow to chain research -> makerchecker
# ------------------------------------------------------
flow = SequentialFlow(
    name="atomic_researcher_flow",
    namespace="research",
    description="Atomic workflow chaining Tavily research with iterative maker-checker refinement.",
    steps=[research_tool, maker_checker],
)

def main() -> None:
    inputs = {"query": "What is the latest discovery/news regarding mosasaurs, and their placement within the clade toxicofera?"}
    final = flow.invoke(inputs)

    print("\n================ FINAL DRAFT (ATOMIC ONLY) ================\n")
    print(final.result["draft"])


if __name__ == "__main__":
    main()
