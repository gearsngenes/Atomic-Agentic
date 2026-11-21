"""
Atomic-only example:
- Research Tool (Tavily SDK): query -> sources[list[str]] (clean formatted passages + URLs)
- Graph loop: writer -> critic -> judge, repeated until the judge sees <<APPROVED>>
  (add_sequence with repeat_until; max_traversal_depth caps a critic that never approves)
- Pipeline: research -> writer/critic loop
- Prints the final APA-style report

Requirements:
- pip install tavily-python python-dotenv
- TAVILY_API_KEY in environment or .env
- OPENAI_API_KEY in environment or .env  (for OpenAIEngine)
"""

from __future__ import annotations

from atomic_agentic.composites import Graph, Pipeline

from researcher_agents import writer, critic
from researcher_tools import research_tool, judge

MAX_REVISIONS = 3

# ---------------------------------------------------------------------
# Writer/critic loop: the critic returns revision notes, the judge turns them
# into {"approved": bool}, and the sequence repeats until approved.
# ---------------------------------------------------------------------
loop = Graph(
    name="research_report_loop",
    description="Iteratively refine an APA report until the critic approves.",
    returns="draft",
    max_traversal_depth=MAX_REVISIONS * 3,
)
loop.add_sequence(
    {"write": writer, "critique": critic, "judge": judge},
    repeat_until={"approved": True},
)
loop.set_start("write")

# ------------------------------------------------------
# Pipeline: research -> loop (the research dict seeds the loop's state)
# ------------------------------------------------------
pipeline = Pipeline(
    name="atomic_researcher_pipeline",
    description="Atomic pipeline chaining Tavily research with an iterative writer/critic loop.",
    steps=[research_tool, loop],
)


def main() -> None:
    inputs = {"query": "What is the latest discovery/news regarding mosasaurs, and their placement within the clade toxicofera?"}
    final = pipeline.invoke(inputs)

    print("\n================ FINAL DRAFT (ATOMIC ONLY) ================\n")
    print(final.result)


if __name__ == "__main__":
    main()
