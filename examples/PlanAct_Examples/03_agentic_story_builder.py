from pathlib import Path
import logging

from atomic_agentic.agents import BasicAgent, PlanActAgent
from atomic_agentic.llm import OpenAIEngine

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)

# Sub-agents (outliner/writer/reviewer) stay on their own fixed engine,
# separate from `llm_engine` -- switching the provider under test (the
# PlanActAgent orchestrator below) should never also silently switch what
# these delegation targets run on.
sub_agent_llm = OpenAIEngine(model="gpt-4o-mini")

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
Write a coherent, engaging story with purposeful scenes, believable
characters, controlled pacing, clear and vivid prose, and an earned ending.
When revising, preserve what works and maintain continuity. Avoid distracting
repetition, clichés, and unnecessary exposition.

Return ONLY markdown for the story draft.
Break the story up into sections, where logical, with ## headings.
Max 1000 words. Never include the outline or revision notes verbatim.
""".strip()

REVIEWER_PROMPT = """
You are the *Reviewer* / test audience.
Input: draft_md (markdown).
Assess coherence and payoff, character motivation and change, scene purpose
and pacing, prose and dialogue, and emotional impact and resolution. Judge
only the draft; don't assume an unseen brief or outline.

Prioritize useful strengths to preserve and the biggest issues to fix. Tie
each suggested revision to a specific moment, its reader impact, and a concise
action. Don't force criticism when something works.

Output: bullet-point critique ONLY (max 8 bullets). No rewriting.
""".strip()

# Standing orchestration rules, lifted out of the per-invocation task prompt
# and into tool_instructions -- the process itself (outliner-first, then
# alternate reviewer/writer, always return the writer's latest draft) is now
# reusable agent behavior, not one-off task text repeated on every call.
ORCHESTRATION_INSTRUCTIONS = """
You use the provided tools to perform a multi-step story-building process.
Always call the outliner first, exactly once, before any drafting begins.
After the outline is ready, call the writer for the first draft using the
outline. For every subsequent draft, alternate reviewer -> writer: send the
latest draft to the reviewer, then pass ONLY the reviewer's notes to the
writer. NEVER pass the outline, story idea, or draft itself to the writer
again. Always finish by returning the writer's latest draft verbatim -- never
the outline, reviewer's critique, or a summary of the process.

So in otherwords:
Step 1:
outliner (story_idea) -> outline

Step 2:
writer (outline = outline) -> latest_draft

Step 3:
reviewer (draft = latest_draft) -> feedback
writer (revision_notes = feedback) -> latest_draft

Repeat step 3's review/rewrite EXACTLY {loops} TIMES.
...

Step N:
return the latest updated_draft as the final answer.
""".strip()

outliner = BasicAgent(
    name="StoryOutliner",
    namespace="examples",
    description="Generate a structured outline from a one-sentence idea.",
    llm_engine=sub_agent_llm,
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
    description="Writes a first draft from an outline, then revises using only reviewer notes.",
    llm_engine=sub_agent_llm,
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
    llm_engine=sub_agent_llm,
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
    tool_instructions=ORCHESTRATION_INSTRUCTIONS,
)

# Register agents-as-tools -- each reachable by its own bare agent name
# (outliner.name/writer.name/reviewer.name) since no alias is given.
orch.register_tool(outliner)
orch.register_tool(writer)
orch.register_tool(reviewer)

if __name__ == "__main__":
    idea = input("\nStory idea: ").strip()
    loops_raw = input("How many review/rewrite cycles? ").strip()
    loops = int(loops_raw) if loops_raw else 1
    if loops <= 0:
        raise ValueError("loops must be > 0")

    # Enforce a tight tool-call budget for this run:
    # outliner (1) + initial write (1) + 2 * loops
    orch.tool_calls_limit = 2 * loops + 2

    task_prompt = (
        f"TASK: Write a story based on the following idea: {idea!r}\n"
    )

    print("\n⇢ Planning + execution …")
    final_draft_md = orch.invoke({"prompt": task_prompt, "loops": loops}).result

    print("\n========== FINAL DRAFT ==========\n")
    print(final_draft_md)

    out_dir = Path("examples/output_markdowns")
    out_dir.mkdir(exist_ok=True)
    filepath = out_dir / "planact_story.md"
    filepath.write_text(final_draft_md, encoding="utf-8")
    print(f"\n✓ Story saved to: {filepath.resolve()}")
