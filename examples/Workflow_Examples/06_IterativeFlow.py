
"""
06_IterativeFlow.py

Beginner-friendly IterativeFlow example.
Demonstrates an agentic writer/critic loop with an approval judge, using LLM agents and clear schema-driven steps.
"""

from __future__ import annotations
from pprint import pprint
from dotenv import load_dotenv
from atomic_agentic.agents import Agent
from atomic_agentic.engines.LLMEngines import OpenAIEngine
from atomic_agentic.tools import toolify
from atomic_agentic.workflows.StructuredInvokable import StructuredInvokable
from atomic_agentic.workflows import IterativeFlow

load_dotenv()



# ---------------------------------------------------------------------
# LLM Engine
# ---------------------------------------------------------------------
engine = OpenAIEngine(model="gpt-5-mini")



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



writer_agent = Agent(
    name="story_writer",
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



critic_agent = Agent(
    name="story_critic",
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
    filter_extraneous_inputs=True,
)



# ---------------------------------------------------------------------
# IterativeFlow: writer/critic loop with approval judge
# ---------------------------------------------------------------------
flow = IterativeFlow(
    name="story_writer_checker_flow",
    description="Iterative writer/critic loop with approval judge.",
    body_steps=[writer, critic],
    judge=judge,
    max_iterations=5,
    return_index=0,    # outer result is the writer's draft
    handoff_index=1,   # critic notes become next writer input
    evaluate_index=1,  # critic notes are also sent to judge
)

inputs = {
    "prompt": (
        "Write a short story about Robin Hood in space. "
        "Keep it adventurous, emotionally grounded, and a little playful."
    )
}

final = flow.invoke(inputs)

print("\n=== FINAL STORY ===\n")
print(final["draft"])

print("\n=== OUTER ITERATIVE RESULT ===")
pprint(dict(final))
print(f"outer run_id: {final.run_id}")

checkpoint = flow.get_checkpoint(final.run_id)
print("\n=== CHECKPOINT ===")
pprint(checkpoint)