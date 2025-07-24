import sys, os
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import logging

logging.basicConfig(level=logging.INFO)
# ───────────────────────────  local imports  ────────────────────
from modules.Agents import Agent
from modules.PlannerAgents import AgenticPlannerAgent
from modules.LLMEngines import *

# define a global llm engine to give to each of our agents
llm_engine = OpenAIEngine(model = "gpt-4o-mini")

# ───────────────────────────  ROLE PROMPTS  ─────────────────────
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

# ───────────────────────────  WORKER AGENTS  ────────────────────
outliner = Agent("StoryOutliner", llm_engine, OUTLINER_PROMPT)
writer   = Agent("StoryWriter",   llm_engine, WRITER_PROMPT, context_enabled=True)
reviewer = Agent("DraftReviewer", llm_engine, REVIEWER_PROMPT, context_enabled=True)

# ───────────────────────────  ORCHESTRATOR  ─────────────────────
orch = AgenticPlannerAgent(name = "StoryPlanner", llm_engine=llm_engine)

orch.register(tool = outliner,
                    description = "Fleshes out a full outline from a brief idea description.")
orch.register(tool = reviewer,
                    description = "Reviews story drafts, provides revision notes back to the writer.")
# writer & reviewer are exposed as ordinary tools (not agent-tools)
orch.register(tool = writer,
                    description = "Writes a draft based on the story outline, plus any additional context (i.e. revision notes)")

# ─────────────────────────────  MAIN  ───────────────────────────
if __name__ == "__main__":
    idea  = input("\nStory idea: ").strip()
    loops = int(input("How many review/revision cycles? "))

    task_prompt = (
        f"Write a full story based on: “{idea}”.\n"
        f"Run {loops} write/review cycle(s).\n"
        f"For the write-review steps, after creating revision notes for the latest draft, "
        f"send the latest revision notes back to the writer to rewrite the story:\n\n"
        f"Revision Notes:\n<revision notes here>'\n\n"
        f"Return the final draft once it's prepared."
    )

    # The orchestrator handles both planning *and* execution.
    print("\n⇢ Planning + execution …")
    final_draft_md = orch.invoke(task_prompt)

    print("\n========== FINAL DRAFT ==========\n")
    print(final_draft_md)

    # ───────────── save markdown file ─────────────
    out_dir = Path("examples/output_markdowns")
    out_dir.mkdir(exist_ok=True)
    filename  = "planner_story.md"
    filepath  = out_dir / filename
    filepath.write_text(final_draft_md, encoding="utf-8")

    print(f"\n✓ Story saved to: {filepath.resolve()}")
