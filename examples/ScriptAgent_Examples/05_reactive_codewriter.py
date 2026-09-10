"""05_orchestrating_agents.py

Mirrors ReAct_Examples/02_orchestrating_agents.py (a code builder + reviewer
loop), redesigned to make the approve/revise decision structural instead of
free-form text judgment.

ReAct's original has the ORCHESTRATOR LLM eyeball the reviewer's raw prose
and decide for itself whether it says "Approved" -- free-form judgment at
exactly the spot most worth making reliable. Here, CodeReviewer's own
post_invoke does that check deterministically in Python (exact match against
a designated sentinel, not a substring check -- substring risks a false
positive on prose like "not approved because..."), returning a plain
{"approved": bool, "feedback": str | None} dict instead of raw prose.

This doesn't remove ScriptAgent's need for judgment -- the grammar still has
no `if`/`elif`/`else`, so the orchestrator still needs a `# PAUSE` after each
review to decide stop-vs-continue. What it removes is ambiguity at that
judgment point and ALL string-handling from the orchestrator's own generated
code: it never inspects or parses the review itself, just forwards the whole
dict as one identifier (`CodeWriter(feedback=review)`) to the next builder
call. What changes is what the CONTINUATION-round model sees: instead of
free-form critique prose it has to interpret, the rendered snapshot shows a
literal `{'approved': True, 'feedback': None}` it can pattern-match directly
-- a real reduction in hallucination risk exactly where ReAct's version was
weakest.

This is the same scenario Pass 2.6's live field-test already validated (a
genuinely reactive writer/reviewer loop resolved via checkpoint
continuation) -- this example productizes that proven mechanism, with the
dict-handoff layered on top as a real improvement, not just a restyle.

Standing risk worth remembering here specifically: a silent if-cutoff
(grammar violation with no `# PAUSE` note) renders ONLY the work-so-far
snapshot on continuation, never the original task text -- a model fixated on
satisfying the reviewer could lose a requirement from the original task the
code snapshot alone doesn't imply.
"""
import json
import logging
from pathlib import Path

from atomic_agentic.agents import BasicAgent, ScriptAgent

from shared_engine import llm_engine, OpenAIEngine

logging.basicConfig(level=logging.INFO)

sub_agent_llm = OpenAIEngine(model="gpt-4o-mini")

# ──────────────────────────  CODE WRITER  ───────────────────────────


def writer_pre(task: str | None = None, feedback: dict | None = None) -> str:
    if feedback is not None:
        notes = feedback.get("feedback")
        return (
            "Read and internalize the following feedback on your last draft, "
            f"then use your best judgement to revise it: {notes}\n\n"
            "Please provide the updated code."
        )
    elif task is not None:
        return f"Implement code so that it meets the user's request:\n{task}"
    else:
        raise ValueError("Either task or feedback must be provided.")


writer = BasicAgent(
    name="CodeWriter",
    namespace="examples",
    description=(
        "Generates Python code per user request OR revises its latest draft from "
        "structured feedback. Pass ONLY `task` for a first draft, or ONLY "
        "`feedback` (the reviewer's dict) to revise the last draft."
    ),
    llm_engine=sub_agent_llm,
    role_prompt=(
        "You are a senior software engineer who writes Python code for requested tasks.\n"
        "Return ONLY the code, with no explanations."
    ),
    context_enabled=True,
    records_window=10,
    pre_invoke=writer_pre,
)

# ──────────────────────────  CODE REVIEWER  ─────────────────────────

APPROVED_FLAG = "APPROVED"


def reviewer_pre(draft_code: str) -> str:
    return f"Please review and provide feedback for the following code: ```python\n{draft_code}\n```"


def reviewer_post(critique: str) -> dict:
    if critique.strip().upper() == APPROVED_FLAG:
        return {"approved": True, "feedback": None}
    return {"approved": False, "feedback": critique}


reviewer = BasicAgent(
    name="CodeReviewer",
    namespace="examples",
    description="Reviews draft code and returns a structured {approved, feedback} verdict.",
    llm_engine=sub_agent_llm,
    role_prompt=(
        "You are an expert Python code analyst. Thoroughly and brutally evaluate the code for "
        "accuracy, readability, and overall design optimization. If the code has no critical or "
        f"necessary revisions left, reply with EXACTLY the single word {APPROVED_FLAG} and nothing "
        "else. Otherwise, return ONLY revision critiques that you deem critical or necessary before "
        "handing this off to a professional developer, focusing on:\n"
        "- Syntax or semantic errors in the code (high priority fixes)\n"
        "- Redundant or duplicate code that could be refactored into reusable chunks\n"
        "- Overly complex or irrelevant/unused code that isn't needed for the task\n"
    ),
    context_enabled=True,
    records_window=10,
    pre_invoke=reviewer_pre,
    post_invoke=reviewer_post,
)

# ──────────────────────────  ORCHESTRATOR  ──────────────────────────

orchestrator = ScriptAgent(
    name="CodeOrchestrator",
    namespace="examples",
    description="Orchestrates a write/review loop between CodeWriter and CodeReviewer.",
    llm_engine=llm_engine,
    context_enabled=True,
    tool_calls_limit=10,
    planning_rounds_limit=6,
    response_preview_limit=50,
    generation_retries=None,
)
orchestrator.register_tool(writer)
orchestrator.register_tool(reviewer)

if __name__ == "__main__":
    task = (
        "Write a Python module that scaffolds an agentic AI design with clean OOP and "
        "provider-agnostic LLM backends (e.g., Bedrock, OpenAI, llama-cpp-python).\n\n"
        "Process:\n"
        "1) Write a draft of the code.\n"
        "2) Review the code and provide feedback.\n"
        "3) Pause after each review to check if the reviewer has approved."
        "If not approved, use the whole feedback report to write a new draft.\n"
        "4) Repeat untill approved and RETURN THE FINAL CODE DRAFT"
    )

    result = orchestrator.invoke({"prompt": task}).result
    record = orchestrator.get_conversation()[-1]

    out_dir = Path("examples/output_markdowns")
    out_dir.mkdir(exist_ok=True)

    filepath = out_dir / "ScriptAgent_Code.py"
    filepath.write_text(result, encoding="utf-8")
    print(f"\n>> Final draft code saved to: {filepath.resolve()}")

    filepath = out_dir / "ScriptAgent_Script.txt"
    filepath.write_text(record.render_as_code(), encoding="utf-8")
    print(f"\n>> Executed script saved to: {filepath.resolve()}")

    filepath = out_dir / "ScriptAgent_Record.json"
    filepath.write_text(json.dumps(record.to_dict(), indent=2, default=str), encoding="utf-8")
    print(f"\n>> Serialized record saved to: {filepath.resolve()}")
