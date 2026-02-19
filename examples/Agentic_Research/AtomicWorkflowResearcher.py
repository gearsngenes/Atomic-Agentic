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
from atomic_agentic.workflows import SequentialFlow, BundlingPolicy, MappingPolicy
from atomic_agentic.workflows.composites import MakerCheckerFlow

MAX_REVISIONS = 3

# ---------------------------------------------------------------------
# 5) MakerCheckerFlow and SequentialFlow wiring
# ---------------------------------------------------------------------
maker_checker = MakerCheckerFlow(
    name="research_report_makerchecker",
    description="Iteratively refine an APA report with early-stop approval.",
    maker=writer,
    checker=critic,
    judge=judge,
    max_revisions=MAX_REVISIONS,
    output_schema=["final_draft"],
    mapping_policy=MappingPolicy.MATCH_FIRST_LENIENT,
    bundling_policy=BundlingPolicy.UNBUNDLE,
)


def main() -> None:
    flow = SequentialFlow(
        name="research_report_builder_atomic_only",
        description="Research with Tavily, then draft+refine an APA report via MakerCheckerFlow.",
        steps=[research_tool, maker_checker],
        output_schema=["final_draft"],
        bundling_policy=BundlingPolicy.UNBUNDLE,
        mapping_policy=MappingPolicy.MATCH_FIRST_LENIENT,
    )

    inputs = {"query": "What are the main benefits and risks of CRISPR gene editing in medicine?"}
    final = flow.invoke(inputs)

    print("\n================ FINAL DRAFT (ATOMIC ONLY) ================\n")
    import json
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
