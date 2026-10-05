"""02_orchestrating_agents.py

ReActAgent iteratively orchestrating two BasicAgent sub-agents (a code
builder and a code reviewer) through a build -> review -> revise loop --
deciding one tool call per round rather than a fixed upfront sequence, so
it can stop as soon as the reviewer's grade clears the bar instead of
always running a fixed number of cycles.

`AgenticOrchestrator` sets `response_preview_limit` below so draft code and
review dicts truncate in "Cached values" each round -- without it, the
full values are always visible, leaving the model no real incentive to
reference an earlier result by "$name" instead of retyping it.

"approval_status" is deliberately the first key in CodeReviewer's schema,
before "feedback": OpenAI's structured output fills object fields in
declared order, so this also means the model commits to a grade before
writing the critique prose -- and, since previews truncate hard, a
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
sub_agent_llm = OpenAIEngine(model="gpt-5-mini")


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
    Returns complete Python module source, joined with "#- path/to/module.py" headers.
    """,
    llm_engine=sub_agent_llm,
    role_prompt=(
        "You are a senior software engineer. Produce complete, cohesive Python module(s) "
        "that fulfill the task, with consistent names, imports, and interfaces. Avoid "
        "stubs or omitted implementation unless explicitly requested.\n"
        "Return ONLY source code: no explanation, prose, or Markdown fences. For multiple "
        "modules, output each module in full, preceded by its own standalone header in "
        "the form '#- path/to/module.py' (including the first module). Put each module's "
        "source after its header, then begin the next module with the next header."
    ),
    context_enabled=True,
    pre_invoke=builder_prestep,
    records_window=10,
)


def reviewer_prestep(draft_code: str, criteria: str) -> str:
    return (
        "Review the project against the following criteria:\n"
        f"{criteria}\n\n"
        f"Project source:\n```python\n{draft_code}\n```"
    )


reviewer = BasicAgent(
    name="CodeReviewer",
    namespace="examples",
    description="""
    Inputs: draft_code (project source) and criteria (string acceptance criteria).
    Returns a dict: {"approval_status": "Rebuild"|"Acceptable"|"Outstanding", "feedback": <critique string>}.
    On "Rebuild", pass this whole result straight back to CodeBuilderAgent's "review_feedback"
    keyword argument for the next revision -- never read it apart or retype it.
    "Acceptable" and "Outstanding" both mean: stop and return the latest draft, no rebuild.
    """,
    llm_engine=sub_agent_llm,
    role_prompt=(
        "You are an expert Python project reviewer. Inputs: draft_code (the project source) "
        "and criteria (a string of acceptance requirements). Judge the project primarily "
        "against every stated criterion; identify unmet or ambiguous requirements, and check "
        "correctness, integration across modules, readability, and design. Focus on:\n"
        "- Syntax or semantic errors in the code (high priority fixes)\n"
        "- Redundant or duplicate code that could be refactored into reusable chunks\n"
        "- Overly complex or irrelevant/unused code that isn't needed for the task\n\n"
        "- Overly simple or underdeveloped code that doesn't fully meet the task requirements\n"
        "Grade the code on a three-tier scale, then write your critique:\n"
        "- 'Rebuild': one or more criteria are unmet, or a significant issue remains -- not ready to ship.\n"
        "- 'Acceptable': no blocking issues -- ready to ship, though not polished or exceptional.\n"
        "- 'Outstanding': clean and idiomatic, free of the issues above -- nothing left to fix.\n"
        "In 'feedback', return ONLY specific revision critiques that justify the grade -- no rewriting."
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

RULES:
1. NEVER call CodeBuilderAgent or CodeReviewer twice in a row -- strictly alternate: builder, reviewer, builder, reviewer, ...
2. Call CodeReviewer with the latest draft as "draft_code" and a "criteria" string that captures all material requirements and constraints from the user's original task. Keep the criteria consistent across review rounds.
3. If "approval_status" is "Acceptable" or "Outstanding", return the latest CodeBuilderAgent draft.
4. Otherwise, call CodeBuilderAgent with the reviewer's entire result dictionary as the "review_feedback" keyword-bound argument.

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
    "Write a Python joined list of modules that scaffold an agentic AI design with clean OOP and provider-agnostic "
    "LLM backends (e.g., Bedrock, OpenAI, llama-cpp-python). It should have minimally five tiers:\n"
    "1) a base LLM interface class with abstract methods for sending prompts and receiving responses,"
    "including structured schema output\n"
    "2) a concrete implementation of that interface for at least one provider (e.g., OpenAI, bedrock, anthropic, etc.)\n"
    "3) a higher-level agent class that uses the LLM interface to perform tasks, with methods for planning, acting, and reviewing\n"
    "4) an agent class capable of orchestrating multiple sub-agents AND registerable tools (python methods) to perform complex multi-step tasks, "
    "including delegation and result aggregation\n"
    "5) a main module that accurately demonstrates the usage of the above classes, including a sample multi-step task that requires both planning and review\n"
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
