"""
06_MakerCheckerFlow.py

Correct, invariant-safe Maker–Checker–Judge example for Atomic-Agentic.
"""

from typing import Mapping
from dotenv import load_dotenv

from atomic_agentic.agents import Agent
from atomic_agentic.engines.LLMEngines import OpenAIEngine
from atomic_agentic.workflows.composites import MakerCheckerFlow
from atomic_agentic.tools import toolify
from atomic_agentic.workflows import BundlingPolicy

load_dotenv()

# ─────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────

engine = OpenAIEngine(model="gpt-4o-mini")


# ─────────────────────────────────────────────────────────────
# Writer (Maker)
# Inputs: prompt, revision_notes
# Output: draft (string)
# ─────────────────────────────────────────────────────────────

def writer_pre(*, prompt: str = "", revision_notes: str = "") -> str:
    if revision_notes:
        return (
            f"Prompt:\n{prompt}\n\n"
            f"Revision notes:\n{revision_notes}\n\n"
            "Rewrite accordingly."
        )
    return prompt


writer = Agent(
    name="writer",
    description="Writes and revises text based on feedback.",
    llm_engine=engine,
    role_prompt="You are a concise professional writer.",
    pre_invoke=writer_pre,
)


# ─────────────────────────────────────────────────────────────
# Checker
# Input: draft
# Output: revision_notes
# ─────────────────────────────────────────────────────────────

def checker_pre(*, draft: str) -> str:
    return (
        "Review the following draft.\n\n"
        f"{draft}\n\n"
    )


def checker_post(*, result: str) -> Mapping[str, str]:
    return {"revision_notes": result.strip()}


checker = Agent(
    name="checker",
    description="Reviews drafts and suggests improvements or approves.",
    llm_engine=engine,
    role_prompt=(
        "You are a strict but fair editor. You inspect drafts by a writer.\n\n"
        "If the draft given is excellent, reply ONLY with:\n"
        "<<Approved>>\n\n"
        "Otherwise, give clear revision instructions."),
    pre_invoke=checker_pre,
    post_invoke=checker_post,
)


# ─────────────────────────────────────────────────────────────
# Judge Tool
# Inputs MUST MATCH maker inputs: prompt + revision_notes
# Output: bool
# ─────────────────────────────────────────────────────────────

def judge_revision(*, revision_notes: str) -> bool:
    # prompt is intentionally unused; included to satisfy schema invariant
    return "<<Approved>>" in revision_notes


judge = toolify(
    judge_revision,
    name="approval_judge",
    description="Returns True if checker approved the draft",
    filter_extraneous_inputs=True,  # critical for schema invariance in this flow
)


# ─────────────────────────────────────────────────────────────
# Maker–Checker Flow
# ─────────────────────────────────────────────────────────────

flow = MakerCheckerFlow(
    name="writer_checker_flow",
    description="Writer–Checker loop with early-stop judge",
    maker=writer,
    checker=checker,
    judge=judge,
    max_revisions=5,
    output_schema=["final_draft"],
    bundling_policy=BundlingPolicy.UNBUNDLE,
)


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────

inputs = {"prompt": "Write a short story about Robin Hood... IIIINNN SPAAAAACE!!!!."}

final = flow.invoke(inputs)

ckpt = flow.checkpoints[-1]
meta = ckpt.metadata

print("\n=== FINAL DRAFT ===\n")
print(final["final_draft"])

print("\n=== METADATA ===")
for k, v in meta.items():
    print(f"{k}: {v}")
