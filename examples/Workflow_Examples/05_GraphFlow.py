"""
05_GraphFlow.py

Beginner-friendly GraphFlow example.
Demonstrates an iterative editorial pipeline: a drafter writes a short
story, two independent reviewers (fact-checker, style-editor) critique it
concurrently in the same superstep (fan-out/fan-in), and a triage node
carries two independent routers -- one that loops back to the drafter for
revision or finalizes, and one that escalates to a human once revisions
stall. The loop-back edge is GraphFlow's headline new capability: no
Sequential/Parallel/Routing/Iterative combination can express "route back
to an earlier node based on a runtime decision" the way a real graph can.

Also demonstrates a real same-superstep state-write collision: both
reviewers write a shared "checked_by" key, resolved deterministically via
GraphFlow's priority/tiebreak mechanism rather than left to chance.

Each LLM-backed node is a plain BasicAgent using pre_invoke/post_invoke to
build its prompt and shape its dict output directly -- no toolify wrapper
around an agent.invoke() call. post_invoke's return value becomes
AgentResult.result verbatim, and its extra parameters beyond `result` are
auto-grafted onto the agent's own schema, so e.g. drafter_post can read
revision_count straight from GraphFlow's accumulated state and return the
incremented dict itself. toolify is used only for triage and the two
routers, which are genuinely plain deterministic Python with no LLM call.
context_enabled=False throughout: GraphFlow's accumulated state is the one
channel data flows through here, not each agent's own turn history.
"""

from __future__ import annotations
from pathlib import Path
from pprint import pprint
from dotenv import load_dotenv
from atomic_agentic.agents import BasicAgent
from atomic_agentic.llm import OpenAIEngine
from atomic_agentic.tools import toolify
from atomic_agentic.workflows import GraphFlow

load_dotenv()


# ---------------------------------------------------------------------
# LLM Engine
# ---------------------------------------------------------------------
engine = OpenAIEngine(model="gpt-5-mini")


# ---------------------------------------------------------------------
# Drafter node: writes or revises the story, tracks revision_count itself
# ---------------------------------------------------------------------
def drafter_pre(*, topic: str="", revision_notes: str = "") -> str:
    if revision_notes:
        return (
            f"Revision notes from editors:\n{revision_notes}\n\n"
            "Produce a fully revised version of the story that addresses these notes. "
            "Return only the story text."
        )
    return (
        f"Assignment: {topic}\n\nWrite a vivid, complete short story with a clear "
        "beginning, middle, and end. Return only the story text."
    )


def drafter_post(*, result: str, revision_count: int = 0) -> dict:
    return {"draft_text": result, "revision_count": revision_count + 1}


drafter = BasicAgent(
    name="drafter",
    namespace="examples",
    description="Drafts or revises the story and tracks revision_count.",
    llm_engine=engine,
    role_prompt=(
        "You are a strong short-fiction writer with a gift for clear storytelling, "
        "memorable imagery, and satisfying endings. Return only the story text -- "
        "no commentary, no bullet points, no explanations."
    ),
    pre_invoke=drafter_pre,
    post_invoke=drafter_post,
    context_enabled=True,
)


# ---------------------------------------------------------------------
# fact_checker / style_editor: concurrent reviewers, fan out from drafter
# ---------------------------------------------------------------------
def fact_pre(*, draft_text: str) -> str:
    return f"Review this story for internal consistency:\n\n{draft_text}"


def fact_post(*, result: str) -> dict:
    passed = result.strip() == "APPROVED"
    return {
        "fact_pass": passed,
        "fact_issues": [] if passed else [result.strip()],
        "checked_by": "fact_checker",
    }


fact_checker = BasicAgent(
    name="fact_checker",
    namespace="examples",
    description="Checks the draft's internal consistency.",
    llm_engine=engine,
    role_prompt=(
        "You are a meticulous continuity checker for fiction. You care about internal "
        "consistency only -- timeline, character details, and established rules of the "
        "story's own world -- never real-world fact-checking. If the draft is internally "
        "consistent, respond with exactly APPROVED and nothing else. Otherwise, list the "
        "specific inconsistencies concisely."
    ),
    pre_invoke=fact_pre,
    post_invoke=fact_post,
    context_enabled=False,
)


def style_pre(*, draft_text: str) -> str:
    return f"Review this story's prose style and pacing:\n\n{draft_text}"


def style_post(*, result: str) -> dict:
    passed = result.strip() == "APPROVED"
    return {
        "style_pass": passed,
        "style_issues": [] if passed else [result.strip()],
        "checked_by": "style_editor",
    }


style_editor = BasicAgent(
    name="style_editor",
    namespace="examples",
    description="Checks the draft's prose style and pacing.",
    llm_engine=engine,
    role_prompt=(
        "You are a discerning prose editor. You care about pacing, voice, imagery, and "
        "whether the ending feels earned. If the draft is publication-ready, respond with "
        "exactly APPROVED and nothing else. Otherwise, give concise, concrete style notes."
    ),
    pre_invoke=style_pre,
    post_invoke=style_post,
    context_enabled=False,
)


# ---------------------------------------------------------------------
# triage: fan-in point; combines both reviews into one revision_notes
# string. "checked_by" is a deliberate collision -- both reviewers write
# it, and GraphFlow's priorities resolve it (style_editor wins below).
# Passed through here so the winning value is directly observable in the
# trace, rather than only living in internal accumulated state. Plain
# deterministic Python, no LLM call -- toolify is the right tool here.
# ---------------------------------------------------------------------
def triage_fn(
    *,
    fact_pass: bool,
    style_pass: bool,
    fact_issues: list,
    style_issues: list,
    revision_count: int,
    checked_by: str,
) -> dict:
    return {
        "passed": fact_pass and style_pass,
        "revision_notes": "\n".join(fact_issues + style_issues),
        "revision_count": revision_count,
        "checked_by": checked_by,
    }


triage = toolify(
    triage_fn,
    name="triage",
    namespace="examples",
    description="Combines both reviews and decides whether the draft is ready.",
)


# ---------------------------------------------------------------------
# Two independent routers on triage: one decides loop-back vs finalize,
# the other independently decides whether to escalate once revisions
# stall. Additive, not exclusive -- exactly GraphFlow's multi-router
# contract. router_a defers (returns None) past the threshold so the two
# don't fire contradictory targets in the same round.
# ---------------------------------------------------------------------
REVISION_THRESHOLD = 3


def loop_or_finalize(*, passed: bool, revision_count: int) -> str | None:
    if passed:
        return "finalize"
    if revision_count >= REVISION_THRESHOLD:
        return None
    return "drafter"


def escalate_if_stuck(*, passed: bool, revision_count: int) -> str | None:
    if not passed and revision_count >= REVISION_THRESHOLD:
        return "human_escalation"
    return None


loop_or_finalize_router = toolify(
    loop_or_finalize,
    name="loop_or_finalize_router",
    namespace="examples",
    description="Loops back to the drafter, or proceeds to finalize once approved.",
)

escalation_router = toolify(
    escalate_if_stuck,
    name="escalation_router",
    namespace="examples",
    description="Escalates to a human once revisions stop converging.",
)


# ---------------------------------------------------------------------
# finalize / human_escalation: leaf nodes (no outgoing edges declared,
# automatically dead ends). Both write the same output keys so the outer
# result is populated no matter which leaf a run actually reaches.
# ---------------------------------------------------------------------
def finalize_fn(*, draft_text: str, revision_count: int) -> dict:
    return {"final_story": draft_text, "total_revisions": revision_count, "outcome": "finalized"}


finalize = toolify(
    finalize_fn,
    name="finalize",
    namespace="examples",
    description="Packages the approved draft as the final output.",
)


def human_escalation_fn(
    *, draft_text: str, fact_issues: list, style_issues: list, revision_count: int
) -> dict:
    return {
        "final_story": draft_text,
        "total_revisions": revision_count,
        "outcome": "escalated",
        "outstanding_fact_issues": fact_issues,
        "outstanding_style_issues": style_issues,
    }


human_escalation = toolify(
    human_escalation_fn,
    name="human_escalation",
    namespace="examples",
    description="Hands a stuck draft off for human review.",
)


# ---------------------------------------------------------------------
# GraphFlow: drafter -> {fact_checker, style_editor} -> triage
#            -> (drafter | finalize) and independently -> human_escalation
# ---------------------------------------------------------------------
flow = GraphFlow(
    name="editorial_pipeline",
    namespace="examples",
    description=(
        "Drafts a story, reviews it from two angles concurrently, and either loops "
        "back for revision, finalizes, or escalates to a human."
    ),
    nodes={
        "drafter": drafter,
        "fact_checker": fact_checker,
        "style_editor": style_editor,
        "triage": triage,
        "finalize": finalize,
        "human_escalation": human_escalation,
    },
    edges=[
        ("drafter", "fact_checker"),
        ("drafter", "style_editor"),
        ("fact_checker", "triage"),
        ("style_editor", "triage"),
        ("triage", loop_or_finalize_router),
        ("triage", escalation_router),
    ],
    start="drafter",
    priorities={"style_editor": 2},  # style_editor's "checked_by" wins ties with fact_checker
    max_edge_traversals=15,  # safety cap; a normal run needs ~3 supersteps per revision cycle
    result_mode=GraphFlow.DICT,
    return_keys=["final_story", "total_revisions", "outcome"],
)


if __name__ == "__main__":
    inputs = {"topic": "A lighthouse keeper who befriends a shipwrecked robot."}

    final = flow.invoke(inputs)

    output_dir = Path("examples/output_markdowns")
    story_path = output_dir / "graphflow_final_story.md"
    trace_path = output_dir / "graphflow_trace.txt"

    story_path.write_text(final.result["final_story"], encoding="utf-8")

    with open(trace_path, "w", encoding="utf-8") as f:
        f.write("=== OUTER GRAPH RESULT ===\n")
        f.write(f"outcome: {final.result['outcome']}\n")
        f.write(f"total_revisions: {final.result['total_revisions']}\n")
        f.write(f"termination_reason: {final.termination_reason}\n")
        f.write(f"run_id: {final.run_id}\n")

        f.write("\n=== SUPERSTEP-BY-SUPERSTEP TRAVERSAL (edge_traversals) ===\n")
        for i, round_nodes in enumerate(final.edge_traversals):
            f.write(f"superstep {i}: {round_nodes}\n")

        # Every triage firing, found via invoker_id rather than trace
        # indexing -- robust to how many revision cycles actually ran.
        f.write("\n=== TRIAGE DECISIONS (collision resolution + routing input) ===\n")
        triage_entries = [e for e in final.trace if e.invoker_id == triage.full_name]
        for i, entry in enumerate(triage_entries):
            f.write(f"\n-- triage firing {i} -- checked_by resolved to: {entry.result['checked_by']!r}\n")
            pprint(entry.result, stream=f)

        f.write("\n=== FULL RESULT ===\n")
        pprint(final, stream=f)

    print(f"Final story written to: {story_path}")
    print(f"Trace written to: {trace_path}")
    print(f"outcome: {final.result['outcome']} after {final.result['total_revisions']} revision(s)")
