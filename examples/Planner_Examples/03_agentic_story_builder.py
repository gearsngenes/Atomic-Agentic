import sys, os
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import logging
# ───────────────────────────  local imports  ────────────────────
from modules.Agents import Agent
from modules.ToolAgents import PlannerAgent
from modules.LLMEngines import *

logging.getLogger().setLevel(level=logging.INFO) # to enable us to see the agentic planning steps

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
outliner = Agent("StoryOutliner", description = "Should be called when given a new story idea. Fleshes it out into a full, structured outline.", llm_engine=llm_engine, role_prompt= OUTLINER_PROMPT)
writer   = Agent("StoryWriter", description = "Writes a draft based on the story outline, plus any additional context (i.e. revision notes)", llm_engine= llm_engine, role_prompt=WRITER_PROMPT, context_enabled=True)
reviewer = Agent("DraftReviewer", description = "Reviews story drafts, provides revision notes back to the writer.", llm_engine=llm_engine, role_prompt=REVIEWER_PROMPT, context_enabled=True)

# ───────────────────────────  ORCHESTRATOR  ─────────────────────
orch = PlannerAgent(name = "StoryPlanner",
                    description = "A stroy building planner that utilizes a story-outliner, writer, and reviewer to construct polished story drafts",
                    llm_engine = llm_engine)

orch.register(outliner)
orch.register(reviewer)
orch.register(writer)

# ─────────────────────────────  MAIN  ───────────────────────────
if __name__ == "__main__":
    idea  = input("\nStory idea: ").strip()
    loops = int(input("How many review/revision cycles? "))

    task_prompt = (
        f"1) Outline a full story based on: “{idea}”.\n"
        f"2) Then write the first draft using the outline\n"
        f"3) Send the story draft to a reviewer\n"
        f"4) Send ONLY the reviewer's revision notes back to the writer to rewrite the draft.\n"
        f"5) Write {loops} total draft(s) using this process.\n"
        f"6) Return the final draft once it's prepared."
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
