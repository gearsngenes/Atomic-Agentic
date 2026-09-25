"""02_orchestrating_agents.py

ReActAgent iteratively orchestrating two BasicAgent sub-agents (a code
builder and a code reviewer) through a build -> review -> revise loop --
deciding one tool call per round rather than a fixed upfront sequence, so
it can stop as soon as the reviewer approves instead of always running a
fixed number of cycles.
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


def builder_prestep(task_prompt: str | None = None, *, feedback: dict | None = None) -> str:
    if feedback is not None:
        return (
            "Read and internalize the following feedback on your last draft, then use your "
            f"best judgement to re-build it: {json.dumps(feedback)}\n\n"
            "Provide the updated code."
        )
    elif task_prompt:
        return f"Implement code so that it meets the user's request:\n{task_prompt}"
    else:
        raise ValueError("Either task_prompt or feedback must be provided.")


builder = BasicAgent(
    name="CodeBuilderAgent",
    namespace="examples",
    description="""
    Returns: code string based on the task or feedback provided.
    First draft: give "task_prompt" positionally ("name": null).
    Revision: give the reviewer's entire result dict as the KEYWORD argument
    "feedback" (that is, "name": "feedback") -- "feedback" comes after a "*"
    in this tool's signature, so it can ONLY be filled by name, never by a
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
    Returns a dict: {"feedback": <critique string>, "approval_status": "Approved" or "Rebuild"}.
    Pass this whole result straight back to CodeBuilderAgent's "feedback" keyword
    argument for the next revision, by "$name" -- never read it apart or retype it.
    """,
    llm_engine=sub_agent_llm,
    role_prompt=(
        "You are an expert Python code analyst. Thoroughly and brutally evaluate the code for "
        "accuracy, readability, and overall design optimization. Return ONLY revision critiques "
        "that you deem critical or necessary for the code to be ready to hand off to a "
        "professional developer. Focus on:\n"
        "- Syntax or semantic errors in the code (high priority fixes)\n"
        "- Redundant or duplicate code that could be refactored into reusable chunks\n"
        "- Overly complex or irrelevant/unused code that isn't needed for the task\n\n"
        "If the code is clean enough and lacks these issues to a significant degree, set "
        "the approval status to 'Approved'. Otherwise, set it to 'Rebuild'."
    ),
    context_enabled=True,
    pre_invoke=reviewer_prestep,
    records_window=10,
    response_schema={
        "type": "object",
        "properties": {
            "feedback": {"type": "string"},
            "approval_status": {"type": "string", "enum": ["Approved", "Rebuild"]},
        },
        "required": ["feedback", "approval_status"],
        "additionalProperties": False,
    }
)

orchestrator = ReActAgent(
    name="AgenticOrchestrator",
    namespace="examples",
    description="Orchestrates calls between the code builder and the code reviewer.",
    llm_engine=llm_engine,
    records_window=10,
    tool_calls_limit=16,
    context_enabled=True,
)

# Register both agents as tools -- each reachable by its own bare name
# (builder.name / reviewer.name) since no alias is given.
orchestrator.register_tool(builder)
orchestrator.register_tool(reviewer)

task = (
    "Write a Python module that scaffolds an agentic AI design with clean OOP and provider-agnostic "
    "LLM backends (e.g., Bedrock, OpenAI, llama-cpp-python).\n\n"
    "Strictly alternate: builder, reviewer, builder, reviewer, ... -- never call the reviewer twice "
    "in a row, and never call the builder twice in a row.\n\n"
    "Decide immediately after every reviewer call, before doing anything else: check its "
    "approval_status. If it is 'Approved', stop now and return the most recent builder draft as "
    "the final answer. Otherwise, your very next call must be the builder, passing the reviewer's "
    "entire result as the builder's 'feedback' keyword argument -- reference it by its bound name, "
    "never rewritten or retyped.\n\n"
    "Do NOT use the entire number of tool calls you have -- stop the moment the reviewer approves. "
    "If you are ever forced to stop before approval (the tool-call budget runs out), still return "
    "the most recent builder draft as the final answer -- never return null or nothing."
)

result = orchestrator.invoke({"prompt": task}).result
record = orchestrator.get_conversation()[-1].to_dict()
serialized_record = json.dumps(record, indent=2)

out_dir = Path("examples/output_markdowns")
out_dir.mkdir(exist_ok=True)

if isinstance(result, str):
    filepath = out_dir / "ReAct_Code.py"
    filepath.write_text(result, encoding="utf-8")
    print(f"\n>> Final draft code saved to: {filepath.resolve()}")
else:
    print(f"\n>> WARNING: orchestrator did not return code (got {result!r}) -- nothing to save.")

filepath = out_dir / "ReAct_statements.txt"
filepath.write_text(pformat(record["statements"]), encoding="utf-8")
print(f">> Executed calls saved to: {filepath.resolve()}")

filepath = out_dir / "ReAct_Record.json"
filepath.write_text(serialized_record, encoding="utf-8")
print(f">> Serialized record saved to: {filepath.resolve()}")
