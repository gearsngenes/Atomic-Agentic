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


def builder_prestep(task: str | None = None, revision_notes: str | None = None) -> str:
    if revision_notes:
        return (
            "Read and internalize the following feedback on your last draft, then use your "
            f"best judgement to re-build it: {revision_notes}\n\n"
            "Please provide the updated code."
        )
    elif task:
        return f"Implement code so that it meets the user's request:\n{task}"
    else:
        raise ValueError("Either task or revision_notes must be provided.")


builder = BasicAgent(
    name="CodeBuilderAgent",
    namespace="examples",
    description="""
    Returns: code string based on the task or revision notes provided.
    If it is the first draft, it will receive the TASK only. If it is given 
    revision notes, it will receive the REVISION_NOTES only.
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
    Returns: revision notes string based on the latest draft code from the code builder.
    If the reviewer determines that the code is ready, it will return ONLY the string 
    "Approved". Otherwise, it will be sent BACK to the code builder to rebuild a new draft.
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
        "If the code is clean enough and lacks these issues to a significant degree, reply "
        "ONLY with 'Approved'."
    ),
    context_enabled=True,
    pre_invoke=reviewer_prestep,
    records_window=10,
)

orchestrator = ReActAgent(
    name="AgenticOrchestrator",
    namespace="examples",
    description="Orchestrates calls between the code builder and the code reviewer.",
    llm_engine=llm_engine,
    records_window=10,
    tool_calls_limit=10,
    context_enabled=True,
    response_preview_limit=100,
)

# Register both agents as tools -- each reachable by its own bare name
# (builder.name / reviewer.name) since no alias is given.
orchestrator.register_tool(builder)
orchestrator.register_tool(reviewer)

task = (
    "Write a Python module that scaffolds an agentic AI design with clean OOP and provider-agnostic "
    "LLM backends (e.g., Bedrock, OpenAI, llama-cpp-python).\n\n"
    "Process:\n"
    "1) Code builder writers the first draft code.\n"
    "2) Send the draft code to the reviewer tool to critique it.\n"
    "3) Check if the revision-notes from the reviewer return 'Approved'. If so, stop and go to step 6.\n"
    "4) Otherwise, send the revision notes back to the builder and iterate.\n\n"
    "5) Repeat steps 2-4 until the code reviewer returns 'Approved'.\n\n"
    "6) Return the CODE BUILDER's latest draft after approval.\n\n"
    "Note: Do NOT try to use the ENTIRE number of tool calls you have -- consider this as you "
    "approach your limit."
)

result = orchestrator.invoke({"prompt": task}).result
record = orchestrator.get_conversation()[-1].to_dict()
serialized_record = json.dumps(record, indent=2)

out_dir = Path("examples/output_markdowns")
out_dir.mkdir(exist_ok=True)

filepath = out_dir / "ReAct_Code.py"
filepath.write_text(result, encoding="utf-8")
print(f"\n>> Final draft code saved to: {filepath.resolve()}")

filepath = out_dir / "ReAct_statements.txt"
filepath.write_text(pformat(record["statements"]), encoding="utf-8")
print(f">> Executed calls saved to: {filepath.resolve()}")

filepath = out_dir / "ReAct_Record.json"
filepath.write_text(serialized_record, encoding="utf-8")
print(f">> Serialized record saved to: {filepath.resolve()}")
