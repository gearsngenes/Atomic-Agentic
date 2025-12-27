from dotenv import load_dotenv
from pathlib import Path
import logging

from atomic_agentic.agents import Agent
from atomic_agentic.agents.tool_agents import PlanActAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

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
Output: bullet-point critique ONLY (max 8 bullets). No rewriting.
""".strip()

outliner = Agent(
    "StoryOutliner",
    description="Generate a structured outline from a one-sentence idea.",
    llm_engine=llm_engine,
    role_prompt=OUTLINER_PROMPT,
)

writer = Agent(
    "StoryWriter",
    description="Writes drafts based on the outline or reviewer notes.",
    llm_engine=llm_engine,
    role_prompt=WRITER_PROMPT,
    context_enabled=True,
)

reviewer = Agent(
    "DraftReviewer",
    description="Reviews drafts and provides revision notes.",
    llm_engine=llm_engine,
    role_prompt=REVIEWER_PROMPT,
    context_enabled=True,
)

orch = PlanActAgent(
    name="StoryPlanner",
    description="Plan-once agent that orchestrates outliner/writer/reviewer.",
    llm_engine=llm_engine,
    run_concurrent=False,   # dependencies dominate here; sequential is clearer
    context_enabled=False,
)

# Register agents-as-tools and capture their full tool ids for deterministic prompting
outliner_tool = orch.register(outliner)[0]
writer_tool = orch.register(writer)[0]
reviewer_tool = orch.register(reviewer)[0]

if __name__ == "__main__":
    idea = input("\nStory idea: ").strip()
    loops_raw = input("How many review/revision cycles? ").strip()
    loops = int(loops_raw) if loops_raw else 1
    if loops < 0:
        raise ValueError("loops must be >= 0")

    # Enforce a tight tool-call budget for this run:
    # outliner (1) + writer draft (1) + loops*(reviewer+writer) (2*loops)
    orch.tool_calls_limit = 2 * loops + 2

    task_prompt = (
        "Follow the below instructions to create a story.\n\n"
        f"1) Call {outliner_tool} with a prompt that includes this user idea:\n"
        f"   {idea!r}\n"
        "   (This should return outline JSON.)\n\n"
        f"2) Call {writer_tool} with a prompt that includes the outline JSON from step 0.\n"
        "   (This should return a first-draft markdown story.)\n\n"
        f"3) For {loops} cycle(s):\n"
        f"   - Call {reviewer_tool} with a prompt that includes the latest draft markdown.\n"
        f"   - Call {writer_tool} with a prompt that includes the reviewer critique and asks for a revised draft.\n\n"
        "Return the final draft markdown."
    )

    print("\n⇢ Planning + execution …")
    final_draft_md = orch.invoke({"prompt": task_prompt})

    print("\n========== FINAL DRAFT ==========\n")
    print(final_draft_md)

    out_dir = Path("examples/output_markdowns")
    out_dir.mkdir(exist_ok=True)
    filepath = out_dir / "planact_story.md"
    filepath.write_text(final_draft_md, encoding="utf-8")
    print(f"\n✓ Story saved to: {filepath.resolve()}")
