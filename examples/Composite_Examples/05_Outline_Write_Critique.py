"""
05_Outline_Write_Critique.py

A Pipeline whose first step is an outliner agent and whose second step is a
Graph writer/critic loop.

Demonstrates:
- agents as composite steps; pre_invoke/post_invoke shape each agent's inputs
  and outputs into state keys
- Graph.add_sequence(..., repeat_until={"approved": True}): write -> critique,
  looping back to write until the critic approves
- the outliner's {"outline": ...} seeding the loop, and the approved draft
  being the pipeline's final result
- max_traversal_depth as the cap on a critic that never approves

Requirements:
- OPENAI_API_KEY in the environment or a .env file (loaded below)

Run from the repo root:
    python examples/Composite_Examples/05_Outline_Write_Critique.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from atomic_agentic.agents import BasicAgent
from atomic_agentic.composites import Graph, Pipeline
from atomic_agentic.llm import OpenAIEngine

load_dotenv()

APPROVED = "<<APPROVED>>"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "output_markdowns" / "outline_write_critique_draft.md"

# Objective, checkable criteria: the writer aims at them and the critic approves
# as soon as they are all met, so the loop converges instead of chasing polish.
CRITERIA = (
    "1. exactly three paragraphs, in plain prose (no headings, no bullet points)\n"
    "2. covers every point of the outline, in its order\n"
    "3. under 250 words in total\n"
    "4. no claims that are clearly false"
)


# ---------------------------------------------------------------------
# Agents: each post_invoke returns a dict that becomes state keys
# ---------------------------------------------------------------------
def outliner_pre(topic: str) -> str:
    return f"Write a short bullet outline (3 to 5 points) for a 3-paragraph essay about: {topic}"


def outliner_post(result: str) -> dict[str, str]:
    return {"outline": result}


def writer_pre(outline: str, revision_notes: str = "") -> str:
    if revision_notes.strip():
        return (
            "Revise your essay. Fix every failed criterion below and change nothing else.\n\n"
            f"Failed criteria:\n{revision_notes}\n\n"
            f"Outline:\n{outline}\n\nCriteria:\n{CRITERIA}"
        )
    return f"Write the essay.\n\nOutline:\n{outline}\n\nCriteria:\n{CRITERIA}"


def writer_post(result: str) -> dict[str, str]:
    return {"draft": result}


def critic_pre(draft: str, outline: str) -> str:
    return (
        f"Check this essay against the criteria.\n\nCriteria:\n{CRITERIA}\n\n"
        f"Outline:\n{outline}\n\nEssay:\n{draft}"
    )


def critic_post(result: str) -> dict[str, Any]:
    return {"approved": APPROVED in result, "revision_notes": result}


def make_agent(name: str, instructions: str, pre: Any, post: Any, engine: OpenAIEngine) -> BasicAgent:
    return BasicAgent(
        name=name,
        description=instructions,
        llm_engine=engine,
        instructions=instructions,
        record_enabled=False,
        pre_invoke=pre,
        post_invoke=post,
        response_to_post="result",
    )


def build_pipeline(max_traversal_depth: int = 12) -> Pipeline:
    engine = OpenAIEngine(model="gpt-4o-mini")
    outliner = make_agent("outliner", "You outline essays concisely.", outliner_pre, outliner_post, engine)
    writer = make_agent(
        "writer",
        "You write clear, compact essays that meet every stated criterion. Return only the essay text.",
        writer_pre,
        writer_post,
        engine,
    )
    critic = make_agent(
        "critic",
        "You are a pragmatic editor who checks an essay against stated criteria only. "
        "Do not hold out for style, polish or personal preference. "
        f"If every criterion is met, reply with ONLY {APPROVED} and nothing else. "
        "Otherwise reply with one line per failed criterion, naming the criterion and the specific fix.",
        critic_pre,
        critic_post,
        engine,
    )

    # write -> critique, then back to write until the critic approves
    loop = Graph(
        "write_critique_loop",
        "Write and critique until approved.",
        returns="draft",
        max_traversal_depth=max_traversal_depth,
    )
    loop.add_sequence({"write": writer, "critique": critic}, repeat_until={"approved": True})
    loop.set_start("write")

    # the outliner's {"outline": ...} seeds the loop's state
    return Pipeline(
        "outline_write_critique",
        "Outline an essay, then write and critique it until approved.",
        [outliner, loop],
    )


def main() -> None:
    pipeline = build_pipeline()
    result = pipeline.invoke({"topic": "Why mosasaurs were not dinosaurs"})

    print("\n================ APPROVED DRAFT ================\n")
    print(result.result)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(result.result, encoding="utf-8")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
