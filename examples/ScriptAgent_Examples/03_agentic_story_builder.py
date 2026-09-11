"""03_agentic_story_builder.py

Mirrors PlanAct_Examples/03_agentic_story_builder.py, rebuilt on ScriptAgent.

Three BasicAgents (StoryOutliner/StoryWriter/DraftReviewer) are registered
as tools verbatim -- nothing about them is ScriptAgent-specific. Where this
pairs interestingly with 02_async_planner_test.py: 02 has five void calls
with zero data dependency, so only the developer-level
`tool_concurrency_limit` knob can force ordering there. Here, every step's
input is literally the previous step's output (writer(outline=...),
reviewer(draft_md=...), writer(revision_notes=...), ...) -- real data
dependencies, which compile_batches already sequences correctly for free,
no concurrency knob needed.

Budget note: unlike PlanAct's JSON "return" step (a counted step),
ScriptAgent's `return <expr>` is a language terminal, not a tool call --
it costs 0 against tool_calls_limit. So the budget here is
`2*loops + 2` (outline + first draft + loops*(review + write)), one less
than PlanAct's `2*loops + 3`.

Task prompt is deliberately terse (tool names + loop count only, no
explanation of *why* the calls chain) -- there's no ordering ambiguity to
hand-hold here the way there was in 02, since a data dependency isn't
optional/discoverable, it's just present in the args or not.
"""
from pathlib import Path
import logging

from atomic_agentic.agents import BasicAgent, ScriptAgent

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)

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
    description="Writes drafts based on the outline or reviewer notes (exclusive, do NOT send both).",
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

orch = ScriptAgent(
    name="StoryPlanner",
    namespace="examples",
    description="One-shot agent that orchestrates outliner/writer/reviewer.",
    llm_engine=llm_engine,
    context_enabled=True,
    planning_rounds_limit=1,
)

# Registered under each agent's own bare name -- no id capture needed, the
# task prompt below refers to them by the exact same names ScriptAgent
# shows the LLM in its own tool list.
orch.register_tool(outliner)
orch.register_tool(writer)
orch.register_tool(reviewer)

if __name__ == "__main__":
    idea = input("\nStory idea: ").strip()
    loops_raw = input("How many review/revision cycles? ").strip()
    loops = int(loops_raw) if loops_raw else 1
    if loops <= 0:
        raise ValueError("loops must be > 0")

    # outline (1) + initial write (1) + loops * (reviewer + writer) (2 * loops)
    # -- no separate budget slot for `return`, unlike PlanAct's counted step.
    orch.tool_calls_limit = 2 * loops + 2

    task_prompt = (
        f"TASK: Write a story based on the following idea: {idea!r}\n"
        "Create a structured outline, then write a first draft. "
        f"Then for {loops} cycles, review and critique the draft then forward the notes to rewrite it."
    )

    print("\n⇢ Planning + execution …")
    final_draft_md = orch.invoke({"prompt": task_prompt}).result

    print("\n========== FINAL DRAFT ==========\n")
    print(final_draft_md)

    record = orch.get_conversation()[-1]
    print("\n=== EXECUTED SCRIPT ===")
    print(record.render_as_code())

    out_dir = Path("examples/output_markdowns")
    out_dir.mkdir(exist_ok=True)
    filepath = out_dir / "script_agent_story.md"
    filepath.write_text(final_draft_md, encoding="utf-8")
    print(f"\n✓ Story saved to: {filepath.resolve()}")
