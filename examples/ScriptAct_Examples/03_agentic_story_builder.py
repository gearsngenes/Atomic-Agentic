"""03_agentic_story_builder.py

Mirrors PlanAct_Examples/03_agentic_story_builder.py, rebuilt on ScriptActAgent.

Three BasicAgents (StoryOutliner/StoryWriter/DraftReviewer) are registered
as tools verbatim -- nothing about them is ScriptActAgent-specific. Where this
pairs interestingly with 02_async_planner_test.py: 02 has five void calls
with zero data dependency, so only the developer-level
`tool_concurrency_limit` knob can force ordering there. Here, every step's
input is literally the previous step's output (writer(outline=...),
reviewer(draft_md=...), writer(revision_notes=...), ...) -- real data
dependencies, which compile_batches already sequences correctly for free,
no concurrency knob needed.

Budget note: unlike PlanAct's JSON "return" step (a counted step),
ScriptActAgent's `return <expr>` is a language terminal, not a tool call --
it costs 0 against tool_calls_limit. So the budget here is
`2*loops + 2` (outline + first draft + loops*(review + write)), one less
than PlanAct's `2*loops + 3`.

Mirrors PlanAct_Examples/03_agentic_story_builder.py's split exactly: the
task prompt (`planner_prestep`) states only the idea -- no tool names, no
loop count, no explanation of *why* the calls chain (there's no ordering
ambiguity to hand-hold here the way there was in 02, since a data
dependency isn't optional/discoverable, it's just present in the args or
not). Every process mechanic -- outliner-first, alternate reviewer/writer,
always return the writer's latest draft, and the loop count itself via
`{loops}` -- lives in `ORCHESTRATION_INSTRUCTIONS` instead, standing
orchestration behavior rather than one-off task text repeated on every
call. `tool_instructions` renders `{loops}` against this invocation's own
`task.inputs`, never baked into the task prompt.
"""
from pathlib import Path
import logging

from atomic_agentic.agents import BasicAgent, ScriptActAgent
from atomic_agentic.llm import OpenAIEngine

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

So in other words:
Step 1:
outliner(story_idea) -> outline

Step 2:
writer(outline=outline) -> latest_draft

Step 3:
reviewer(draft=latest_draft) -> feedback
writer(revision_notes=feedback) -> latest_draft

Repeat step 3's review/rewrite EXACTLY {loops} TIMES.

Step N:
return latest_draft as the final answer, verbatim.
""".strip()

sub_agent_llm = OpenAIEngine(model="gpt-4o-mini")

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

def planner_prestep(story_idea: str) -> str:
    return (
        f"Write a story based on the following idea: {story_idea!r}\n"
        "Return the final draft as output."
    )

orch = ScriptActAgent(
    name="StoryPlanner",
    namespace="examples",
    description="One-shot agent that orchestrates outliner/writer/reviewer.",
    llm_engine=llm_engine,
    context_enabled=True,
    replanning_limit=1,
    pre_invoke=planner_prestep,
    tool_instructions=ORCHESTRATION_INSTRUCTIONS,
)

# Registered under each agent's own bare name -- no id capture needed, the
# task prompt below refers to them by the exact same names ScriptActAgent
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

    orch.tool_calls_limit = 3 * loops

    print("\n⇢ Planning + execution …")
    final_draft_md = str(orch.invoke({"story_idea": idea, "loops": loops}).result)

    print("\n========== FINAL DRAFT ==========\n")
    print(final_draft_md)

    record = orch.get_conversation()[-1]
    print("\n=== EXECUTED SCRIPT ===")
    print(record.render_as_code())

    out_dir = Path("examples/output_markdowns")
    out_dir.mkdir(exist_ok=True)
    filepath = out_dir / "scriptact_agent_story.md"
    filepath.write_text(final_draft_md, encoding="utf-8")
    print(f"\n✓ Story saved to: {filepath.resolve()}")
