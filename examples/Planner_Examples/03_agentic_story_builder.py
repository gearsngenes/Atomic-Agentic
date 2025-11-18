from dotenv import load_dotenv
import logging
from atomic_agentic.Agents import Agent
from atomic_agentic.ToolAgents import PlannerAgent
from atomic_agentic.LLMEngines import OpenAIEngine

load_dotenv()

logging.getLogger().setLevel(level=logging.INFO)

llm_engine = OpenAIEngine(model="gpt-4o-mini")

OUTLINER_PROMPT = """
You are the *Story Outliner*.
Input: story_idea (one sentence).
Output: **JSON only** with keys:
  working_title, premise,
  characters [ {name, motivation, conflict} … ],
  scenes     [ {title, purpose} … ]
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
Output: bullet-point critique ONLY (max 8 bullets).  No rewriting.
""".strip()

outliner = Agent(
    "StoryOutliner",
    description="Generate a structured outline from a one-sentence idea.",
    llm_engine=llm_engine,
    role_prompt=OUTLINER_PROMPT
)
writer = Agent(
    "StoryWriter",
    description="Writes drafts based on the outline or reviewer notes.",
    llm_engine=llm_engine,
    role_prompt=WRITER_PROMPT,
    context_enabled=True
)
reviewer = Agent(
    "DraftReviewer",
    description="Reviews drafts and provides revision notes.",
    llm_engine=llm_engine,
    role_prompt=REVIEWER_PROMPT,
    context_enabled=True
)

orch = PlannerAgent(
    name="StoryPlanner",
    description="Planner that orchestrates outliner, writer, reviewer to produce a polished draft.",
    llm_engine=llm_engine
)

orch.register(outliner)
orch.register(reviewer)
orch.register(writer)

if __name__ == "__main__":
    idea = input("\nStory idea: ").strip()
    loops = int(input("How many review/revision cycles? "))

    task_prompt = (
        "Follow the below instructions to create a story:\n"
        f"Create an outline for a full story based on the following user idea: “{idea}”.\n"
        f"Then write the first draft using the generated outline.\n"
        f"Send the writer's draft to a reviewer for revision.\n"
        f"Send the reviewer's revision notes back to the writer.\n"
        f"Repeat the review & rewrite process to update the draft {loops} times.\n"
        f"Return the final draft once it's prepared."
    )

    print("\n⇢ Planning + execution …")
    final_draft_md = orch.invoke({"prompt": task_prompt})

    print("\n========== FINAL DRAFT ==========\n")
    print(final_draft_md)

    out_dir = Path("examples/output_markdowns")
    out_dir.mkdir(exist_ok=True)
    filepath = out_dir / "planner_story.md"
    filepath.write_text(final_draft_md, encoding="utf-8")
    print(f"\n✓ Story saved to: {filepath.resolve()}")
