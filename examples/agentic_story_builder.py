import sys
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import re, datetime
import logging

logging.basicConfig(level=logging.INFO)
# ───────────────────────────  local imports  ────────────────────
from modules.Agents import Agent, PlannerAgent
from modules.Plugins import ConsolePlugin
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
Required arg: outline_json (from Outliner).
Optional args: prior_draft, revision_notes (may be empty strings).

Return ONLY markdown for the story draft.
Use scene titles as ## headings.
Max 1 500 words.  Never include the outline or revision notes verbatim.
""".strip()

REVIEWER_PROMPT = """
You are the *Reviewer* / test audience.
Input: draft_md (markdown).
Output: bullet-point critique ONLY (max 8 bullets).  No rewriting.
""".strip()

# ───────────────────────────  WORKER AGENTS  ────────────────────
outliner = Agent("StoryOutliner", OUTLINER_PROMPT)
writer   = Agent("StoryWriter", WRITER_PROMPT)
reviewer = Agent("DraftReviewer", REVIEWER_PROMPT)

# ───────────────────────────  ORCHESTRATOR  ─────────────────────
orch = PlannerAgent(name = "StoryPlanner")
orch.register_agent(outliner,
                    description="Flesh out a full outline from a brief idea description.")
orch.register_agent(reviewer,
                    description="Reviews story drafts, provides revision notes back to the writer.")
# writer & reviewer are exposed as ordinary tools (not agent-tools)
orch.register_agent(writer,
                   "Writes a draft based on the story outline, plus any additional context (i.e. revision notes, prior drafts)")

# register the print method from ConsolePlugin
orch.register_plugin(ConsolePlugin())

# ───────────────────────────  HELPERS  ──────────────────────────
def slugify(text: str, maxlen: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:maxlen] or "story"

# ─────────────────────────────  MAIN  ───────────────────────────
if __name__ == "__main__":
    idea  = input("\nStory idea: ").strip()
    loops = int(input("How many review/revision cycles? "))

    task_prompt = (
        f"Write a full story based on: “{idea}”.\n"
        f"Run {loops} write/review cycle(s).\n"
        f"For the write-review steps, after creating revision notes for the latest draft, "
        f"send the original outline, the reviewer's latest revision notes, and the "
        f"writer's latest draft as a single string formatted like so back to the writer to rewrite the story:\n\n"
        f"Outline:\n<outline here>\n\nPrior Draft:\n<prior draft here>\n\nRevision Notes:\n<revision notes here>'\n\n"
        f"Print and return the final draft once it's prepared."
    )

    # The orchestrator handles both planning *and* execution.
    print("\n⇢ Planning + execution …")
    final_draft_md = orch.invoke(task_prompt)

    print("\n========== FINAL DRAFT ==========\n")
    print(final_draft_md)

    # ───────────── save markdown file ─────────────
    out_dir = Path("examples/output_markdowns")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename  = f"{slugify(idea)}-{timestamp}.md"
    filepath  = out_dir / filename
    filepath.write_text(final_draft_md, encoding="utf-8")

    print(f"\n✓ Story saved to: {filepath.resolve()}")
