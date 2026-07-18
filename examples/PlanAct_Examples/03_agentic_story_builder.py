from dotenv import load_dotenv
from pathlib import Path
import logging

from atomic_agentic.agents import BasicAgent, PlanActAgent
from atomic_agentic.llm import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

llm_engine = OpenAIEngine(model="gpt-4o-mini")

OUTLINER_PROMPT = """
You are the *Story Outliner*.
Input: story_idea (one sentence).
Output: **JSON only** with keys:
  working_title, premise,
  characters [ {{name, motivation, conflict}} … ],
  scenes     [ {{title, purpose}} … ]
""".strip()

WRITER_PROMPT = """
You are the *Story Writer*.
Required arg: outline_json (from Outliner, though only for the first draft).
Afterwards, you may instead get revision notes from the reviewer, which
you will use to apply changes to your last draft with.

Return ONLY markdown for the story draft.
Break the story up into sections, where logical, with ## headings.
Max 1000 words. Never include the outline or revision notes verbatim.
""".strip()

REVIEWER_PROMPT = """
You are the *Reviewer* / test audience.
Input: draft_md (markdown).
Output: bullet-point critique ONLY (max 8 bullets). No rewriting.
""".strip()

outliner = BasicAgent(
    name="StoryOutliner",
    namespace="examples",
    description="Generate a structured outline from a one-sentence idea.",
    llm_engine=llm_engine,
    role_prompt=OUTLINER_PROMPT,
)

def writer_pre(outline: str | None = None, revision_notes: str | None = None) -> str:
    if outline:
        return f"Here is the story outline to use for your first draft: {outline}"
    elif revision_notes:
        return f"Here are the revision notes to apply to your last draft: {revision_notes}"
    else:
        raise ValueError("Either outline or revision_notes must be provided.")

writer = BasicAgent(
    name="StoryWriter",
    namespace="examples",
    description="Writes drafts based on the outline or reviewer notes.",
    llm_engine=llm_engine,
    role_prompt=WRITER_PROMPT,
    context_enabled=True,
    pre_invoke=writer_pre,
)

def reviewer_pre(draft: str) -> str:
    return f"Review & edit the below draft:\n```\n{draft}\n```"

reviewer = BasicAgent(
    name="DraftReviewer",
    namespace="examples",
    description="Reviews drafts and provides revision notes.",
    llm_engine=llm_engine,
    role_prompt=REVIEWER_PROMPT,
    context_enabled=True,
    pre_invoke=reviewer_pre,
)

orch = PlanActAgent(
    name="StoryPlanner",
    namespace="examples",
    description="Plan-once agent that orchestrates outliner/writer/reviewer.",
    llm_engine=llm_engine,
    context_enabled=False,
    tool_calls_limit=None,
)

# Register agents-as-tools and capture their full tool ids for deterministic prompting
outliner_tool = orch.register(outliner)
writer_tool = orch.register(writer)
reviewer_tool = orch.register(reviewer)

if __name__ == "__main__":
    idea = input("\nStory idea: ").strip()
    loops_raw = input("How many review/revision cycles? ").strip()
    loops = int(loops_raw) if loops_raw else 1
    if loops <= 0:
        raise ValueError("loops must be > 0")

    # Enforce a tight tool-call budget for this run:
    # outliner (1) + initial write (1) + loops * (reviewer + writer) (2 * loops) + return (1)
    orch.tool_calls_limit = 2 * loops + 3

    task_prompt = (
        f"TASK: Write a story based on the following idea: {idea!r}\n"
        "Use the outliner to generate a structured outline, then write a first draft. "
        f"Then for {loops} cycles, have the reviewer critique the draft and the writer apply the notes."
    )

    print("\n⇢ Planning + execution …")
    final_draft_md = orch.invoke({"prompt": task_prompt}).result

    print("\n========== FINAL DRAFT ==========\n")
    print(final_draft_md)

    out_dir = Path("examples/output_markdowns")
    out_dir.mkdir(exist_ok=True)
    filepath = out_dir / "planact_story.md"
    filepath.write_text(final_draft_md, encoding="utf-8")
    print(f"\n✓ Story saved to: {filepath.resolve()}")
