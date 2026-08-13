"""
04_IterativeFlow.py

Beginner-friendly IterativeFlow example.
Demonstrates an agentic writer/critic loop with a checker-gated approval
judge, using LLM agents and clear schema-driven steps. IterativeFlow owns
its loop body directly (no nested SequentialFlow); the approval judge is
registered as a checker at the critic step, and the writer's draft is the
result-setting step.
"""

from __future__ import annotations
from pathlib import Path
from pprint import pprint
from dotenv import load_dotenv
from atomic_agentic.agents import BasicAgent
from atomic_agentic.llm import OpenAIEngine
from atomic_agentic.tools import toolify
from atomic_agentic import StructuredInvokable
from atomic_agentic.workflows import IterativeFlow

load_dotenv()


# ---------------------------------------------------------------------
# LLM Engine
# ---------------------------------------------------------------------
engine = OpenAIEngine(model="gpt-4o-mini")


# ---------------------------------------------------------------------
# Writer agent: generates or revises a story draft
# ---------------------------------------------------------------------
def writer_pre(*, prompt: str = "", revision_notes: str = "") -> str:
    prompt = prompt.strip()
    revision_notes = revision_notes.strip()
    if revision_notes and prompt:
        return (
            "You are revising a story draft.\n\n"
            f"Original assignment:\n{prompt}\n\n"
            f"Revision notes:\n{revision_notes}\n\n"
            "Produce a fully revised version of the story. "
            "Do not explain your changes. Return only the story text."
        )
    if revision_notes:
        return (
            "You are revising a story draft.\n\n"
            f"Revision notes:\n{revision_notes}\n\n"
            "Produce a revised version of the story. "
            "Do not explain your changes. Return only the story text."
        )
    return (
        "You are writing an original short story.\n\n"
        f"Assignment:\n{prompt}\n\n"
        "Write a vivid, polished short story that is imaginative but coherent. "
        "Make it feel complete, with a clear beginning, middle, and end. "
        "Return only the story text."
    )


writer_agent = BasicAgent(
    name="story_writer",
    namespace="examples",
    description="Writes and revises short stories from a request and revision notes.",
    llm_engine=engine,
    role_prompt=(
        "You are a strong short-fiction writer with a gift for clear storytelling, "
        "memorable imagery, emotional momentum, and satisfying endings. "
        "When given a fresh story request, write a complete short story rather than an outline. "
        "When given revision notes, apply them faithfully while preserving strengths that already work. "
        "Aim for prose that is readable, lively, and specific. Avoid generic filler, repetition, "
        "or detached commentary about the writing process.\n\n"
        "Your stories should feel intentional: establish the premise quickly, develop conflict or tension, "
        "and land on an ending that feels earned. When revising, do not merely patch sentences. "
        "Actually improve characterization, clarity, pacing, imagery, and narrative payoff wherever needed. "
        "Return only the revised story text and never include bullet points, explanations, or editor notes."
    ),
    pre_invoke=writer_pre,
    context_enabled=True,
)

writer = StructuredInvokable(
    component=writer_agent,
    name="story_writer_step",
    description="Structured story-writing step that emits a draft field.",
    output_schema=["draft"],
)


# ---------------------------------------------------------------------
# Critic agent: reviews a draft and returns revision notes or approval
# ---------------------------------------------------------------------
def critic_pre(*, draft: str) -> str:
    return (
        "Review the following short story draft.\n\n"
        f"{draft}\n\n"
        "Evaluate it for clarity, pacing, emotional payoff, originality, prose quality, "
        "and whether the ending feels earned. If the story is strong enough to approve, "
        "respond with exactly <<APPROVED>> and nothing else. Otherwise, provide concrete "
        "revision notes that the writer can directly apply."
    )


critic_agent = BasicAgent(
    name="story_critic",
    namespace="examples",
    description="Critiques a story draft and either approves it or gives revision notes.",
    llm_engine=engine,
    role_prompt=(
        "You are a demanding but constructive fiction editor. "
        "Your job is to inspect a story draft and decide whether it is publication-worthy for its intended scope. "
        "You care about coherence, specificity, pacing, atmosphere, character motivation, and the strength of the ending. "
        "You are not impressed by empty flourish. You want the piece to feel alive, purposeful, and complete.\n\n"
        "When the draft is already strong, approve it decisively by replying with exactly <<APPROVED>>. "
        "Do not add any extra words in that case. When the draft is not yet good enough, give concise but specific "
        "revision notes that focus on the highest-value changes. The notes should be directly actionable by a writer "
        "and should not meander into abstract literary theory."
    ),
    pre_invoke=critic_pre,
    context_enabled=True,
)

critic = StructuredInvokable(
    component=critic_agent,
    name="story_critic_step",
    description="Structured story-critic step that emits revision notes.",
    output_schema=["revision_notes"],
)


# ---------------------------------------------------------------------
# Approval judge: returns True if the critic approved the draft
# ---------------------------------------------------------------------
def approval_judge(*, revision_notes: str) -> bool:
    return revision_notes.strip() == "<<APPROVED>>"

judge = toolify(
    approval_judge,
    name="approval_judge",
    namespace="workflow",
    description="Return True when the critic has approved the story draft.",
)


# ---------------------------------------------------------------------
# IterativeFlow: writer/critic loop with approval judge
# ---------------------------------------------------------------------
flow = IterativeFlow(
    name="story_writer_checker_flow",
    namespace="examples",
    description="Iterative writer/critic loop with a checker-gated approval judge.",
    body_steps=[writer, critic],
    max_iterations=5,
    checkers=[None, (judge, True)],  # critic's notes checked for approval
    result_setting_indices=[0],  # outer result is the writer's draft
    handoff_index=1,             # critic notes become next writer input
)


if __name__ == "__main__":
    inputs = {
        "prompt": (
            "Write a short story about Robin Hood in space. "
            "Keep it adventurous, emotionally grounded, and a little playful."
        )
    }

    final = flow.invoke(inputs)

    output_dir = Path("examples/output_markdowns")
    output_dir.mkdir(exist_ok=True)
    draft_path = output_dir / "iterflow_final_draft.md"
    checkpoints_path = output_dir / "iterflow_checkpoints.txt"

    draft_path.write_text(final.result["draft"], encoding="utf-8")

    with open(checkpoints_path, "w", encoding="utf-8") as f:
        f.write("=== OUTER ITERATIVE RESULT ===\n")
        pprint(final, stream=f)
        f.write(f"\nouter run_id: {final.run_id}\n")

        # The checker sits at the body's last step (critic, index 1), so no
        # step is ever skipped by an early exit here -- every completed
        # iteration contributes exactly len(body_steps) + (1 per registered
        # checker) trace entries (writer, critic, judge), in order.
        # flow.checkers is a fixed-length, position-indexed list (one slot
        # per body step, None where nothing's registered) -- the registered
        # count is a sum over non-None slots, not a bare len(). approval_value
        # is read from the live flow (checkers are mutable post-construction),
        # not the result -- the result only records which step index, if
        # any, actually triggered the early exit.
        stride = len(flow.body_steps) + sum(1 for c in flow.checkers if c is not None)
        approval_value = flow.checkers[1][1]

        f.write("\n=== ITERATION-BY-ITERATION RESULTS ===\n")
        for i in range(final.iterations_completed):
            writer_result, critic_result, judge_result = final.trace[i * stride : (i + 1) * stride]
            approved = judge_result.result == approval_value
            f.write(f"\n-- iteration {i} -- judge result: {judge_result.result!r} (approved={approved})\n")
            pprint(writer_result.result, stream=f)
            pprint(critic_result.result, stream=f)

        f.write("\n=== RUN SUMMARY ===\n")
        f.write(f"exited_early: {final.exited_early}\n")
        f.write(f"triggering_step: {final.triggering_step!r}\n")
        f.write(f"iterations_completed: {final.iterations_completed}\n")
        f.write(f"result_setting_indices: {final.result_setting_indices}\n")
        f.write(f"handoff_index: {final.handoff_index}\n")
        f.write(f"max_iterations: {final.max_iterations}\n")

    print(f"Final draft written to: {draft_path}")
    print(f"Checkpoints written to: {checkpoints_path}")
