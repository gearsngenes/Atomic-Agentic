"""02_orchestrating_agents.py

ReActAgent iteratively orchestrating two BasicAgent sub-agents (a code
builder and a code reviewer) through a build -> review -> revise loop --
deciding one tool call per round rather than a fixed upfront sequence, so
it can stop as soon as the reviewer's grade clears the bar instead of
always running a fixed number of cycles.

Live-tested against gpt-4o-mini. The structured dict return
({"approval_status": ..., "feedback": ...}) isn't the problem it first
looked like -- the actual root cause of unreliable "$name" forwarding was
that `AgenticOrchestrator` had no `response_preview_limit` set, so the full
draft code and full review dict were always fully visible in "Cached
values" every round, giving the model no real incentive to reference
instead of retype. Setting `response_preview_limit` below closed that gap
completely; the structured return stays exactly as it was.

"approval_status" is deliberately the first key in CodeReviewer's schema,
before "feedback": OpenAI's structured output fills object fields in
declared order, so this also means the model commits to a grade before
writing the critique prose -- and, since previews now truncate hard, a
partially-visible preview shows the grade first rather than being eaten
entirely by critique text.
"""
import json
import logging
from pathlib import Path
from pprint import pformat

from atomic_agentic.agents import BasicAgent, ReActAgent
from atomic_agentic.llm import OpenAIEngine

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)

# Sub-agents (builder/reviewer) stay on their own fixed engine, separate
# from `llm_engine` -- switching the provider under test (the ReActAgent
# orchestrator below) should never also silently switch what these
# delegation targets run on.
sub_agent_llm = OpenAIEngine(model="gpt-4o-mini")


def builder_prestep(task_description: str | None = None, *, review_feedback: dict | None = None) -> str:
    if review_feedback is not None:
        return (
            "Read and internalize the following review feedback on your last draft, then use "
            f"your best judgement to rebuild it: {json.dumps(review_feedback)}\n\n"
            "Provide the updated code."
        )
    elif task_description:
        return f"Implement code so that it meets the user's request:\n{task_description}"
    else:
        raise ValueError("Either task_description or review_feedback must be provided.")


builder = BasicAgent(
    name="CodeBuilderAgent",
    namespace="examples",
    description="""
    Returns: code string based on the task or review feedback provided.
    First draft: give "task_description" positionally ("name": null).
    Revision: give the reviewer's entire result dict as the KEYWORD argument
    "review_feedback" (that is, "name": "review_feedback") -- "review_feedback" comes
    after a "*" in this tool's signature, so it can ONLY be filled by name, never by a
    positional argument. Reference the reviewer's result whole, by its
    "$name" -- never retype or rewrite its content.
    """,
    llm_engine=sub_agent_llm,
    role_prompt=(
        "You are a senior software engineer who writes Python code for requested tasks.\n"
        "Return ONLY the code, with no explanations."
    ),
    context_enabled=True,
    pre_invoke=builder_prestep,
    records_window=10,
)


def reviewer_prestep(draft_code: str) -> str:
    return f"Please review and provide feedback for the following code: ```python\n{draft_code}\n```"


reviewer = BasicAgent(
    name="CodeReviewer",
    namespace="examples",
    description="""
    Returns a dict: {"approval_status": "Rebuild"|"Acceptable"|"Outstanding", "feedback": <critique string>}.
    On "Rebuild", pass this whole result straight back to CodeBuilderAgent's "review_feedback"
    keyword argument for the next revision, by "$name" -- never read it apart or retype it.
    "Acceptable" and "Outstanding" both mean: stop and return the latest draft, no rebuild.
    """,
    llm_engine=sub_agent_llm,
    role_prompt=(
        "You are an expert Python code analyst. Thoroughly and brutally evaluate the code for "
        "accuracy, readability, and overall design optimization. Focus on:\n"
        "- Syntax or semantic errors in the code (high priority fixes)\n"
        "- Redundant or duplicate code that could be refactored into reusable chunks\n"
        "- Overly complex or irrelevant/unused code that isn't needed for the task\n\n"
        "Grade the code on a three-tier scale, then write your critique:\n"
        "- 'Rebuild': a real issue above remains -- not ready to ship.\n"
        "- 'Acceptable': no blocking issues -- ready to ship, though not polished or exceptional.\n"
        "- 'Outstanding': clean and idiomatic, free of the issues above -- nothing left to fix.\n"
        "In 'feedback', return ONLY the revision critiques that justify the grade -- no rewriting."
    ),
    context_enabled=True,
    pre_invoke=reviewer_prestep,
    records_window=10,
    response_schema={
        "type": "object",
        "properties": {
            "approval_status": {"type": "string", "enum": ["Rebuild", "Acceptable", "Outstanding"]},
            "feedback": {"type": "string"},
        },
        "required": ["approval_status", "feedback"],
        "additionalProperties": False,
    }
)

# Standing orchestration rules, lifted out of the per-invocation task prompt
# and into tool_instructions -- the builder/reviewer interaction contract
# (strict alternation, how to branch on the three-tier grade, when to stop,
# what to return) is reusable agent behavior, not one-off task text
# repeated on every call. Keeps `task` below down to just the content request.
ORCHESTRATION_INSTRUCTIONS = """
You use the provided tools to perform a multi-step code-building process.

Step 1:
CodeBuilderAgent(task_description = user's task) -> latest_draft

Step 2:
CodeReviewer(draft_code = latest_draft) -> review_result
# {{"approval_status": "Rebuild"|"Acceptable"|"Outstanding", "feedback": <critique text>}}

Step 3, branch on review_result["approval_status"]:
- "Rebuild": your very next call must be CodeBuilderAgent, passing review_result as the
  KEYWORD argument "review_feedback" -- reference it by its bound "$name", never retyped,
  never read apart into pieces. Then repeat from Step 2.
- "Acceptable" or "Outstanding": stop now -- both mean the code is ready to ship. There is
  no difference in what you do next, only in how good the result turned out.

Strictly alternate: builder, reviewer, builder, reviewer, ... -- never call CodeReviewer
twice in a row, and never call CodeBuilderAgent twice in a row.

Step N:
Return CodeBuilderAgent's FINAL latest_draft once review_result["approval_status"] is
"Acceptable" or "Outstanding". If you are ever forced to stop before that (the tool-call
budget runs out), still return the most recent builder draft -- never return null or nothing.

ONLY approval_status decides whether to continue the loop -- never judge the code yourself.
""".strip()

orchestrator = ReActAgent(
    name="AgenticOrchestrator",
    namespace="examples",
    description="Orchestrates calls between the code builder and the code reviewer.",
    llm_engine=llm_engine,
    tool_calls_limit=10,
    context_enabled=True,
    tool_instructions=ORCHESTRATION_INSTRUCTIONS,
    response_preview_limit=10,
)

# Register both agents as tools -- each reachable by its own bare name
# (builder.name / reviewer.name) since no alias is given.
orchestrator.register_tool(builder)
orchestrator.register_tool(reviewer)

task = (
    "Write a Python module that scaffolds an agentic AI design with clean OOP and provider-agnostic "
    "LLM backends (e.g., Bedrock, OpenAI, llama-cpp-python)."
)

result = orchestrator.invoke({"prompt": task}).result
record = orchestrator.get_conversation()[-1]
serialized_record = json.dumps(record.to_dict(), indent=2)

out_dir = Path("examples/output_markdowns")
out_dir.mkdir(exist_ok=True)

if isinstance(result, str):
    filepath = out_dir / "ReAct_Code.py"
    filepath.write_text(result, encoding="utf-8")
    print(f"\n>> Final draft code saved to: {filepath.resolve()}")
else:
    print(f"\n>> WARNING: orchestrator did not return code (got {result!r}) -- nothing to save.")

filepath = out_dir / "ReAct_statements.txt"
filepath.write_text(record.render_as_code(), encoding="utf-8")
print(f">> Executed calls saved to: {filepath.resolve()}")

filepath = out_dir / "ReAct_Record.json"
filepath.write_text(serialized_record, encoding="utf-8")
print(f">> Serialized record saved to: {filepath.resolve()}")
