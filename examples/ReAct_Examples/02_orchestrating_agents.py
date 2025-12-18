import logging

from dotenv import load_dotenv

from atomic_agentic.Agents import Agent, ReActAgent
from atomic_agentic.LLMEngines import OpenAIEngine
from atomic_agentic.Tools import Tool

load_dotenv()
logging.basicConfig(level=logging.INFO)

llm = OpenAIEngine(model="gpt-5-mini")

def builder_prestep(task: str | None, revision_notes: str | None) -> str:
    if revision_notes:
        return f"Please rebuild code for the following task: {task}\n\nRevision Notes from earlier: {revision_notes}\n\nPlease provide the updated code."
    elif task:
        return f"Task: {task}\n\nPlease implement the initial code."
    else:
        raise ValueError("Either task or revision_notes must be provided.")

builder = Agent(
    name="CodeBuilderAgent",
    description="Generates Python code per user request OR revises its earlier drafts from revision notes (provide one or the other).",
    llm_engine=llm,
    role_prompt=(
        "You are a senior software engineer who writes Python code for requested tasks.\n"
        "Return ONLY the code, with no explanations."
    ),
    context_enabled=True,
    pre_invoke=builder_prestep,
    history_window=10,
)


def reviewer_prestep(draft_code: str) -> str:
    return f"Please review the following code for the task: {draft_code}\n\nReturn revision suggestions or 'Approved' if no changes are needed."

reviewer = Agent(
    name="CodeReviewer",
    description="Reviews code from the builder and returns revision suggestions or 'Approved'.",
    llm_engine=llm,
    role_prompt=(
        "You are an expert Python performance analyst. Thoroughly and brutally evaluate the code for "
        "accuracy, readability, and SOLID design. Return ONLY revision suggestions. If no changes are "
        "needed, return 'Approved'."
    ),
    context_enabled=True,
    pre_invoke=reviewer_prestep,
    history_window=10,
)

orchestrator = ReActAgent(
    name="AgenticOrchestrator",
    description="Orchestrates calls between the code builder and the code reviewer.",
    llm_engine=llm,
    history_window=10,
    tool_calls_limit=25,
    context_enabled=False,
    preview_limit=2_500,
)

# Register both agents as tools.
orchestrator.register(builder)
orchestrator.register(reviewer)

task = (
    "Write a Python module that scaffolds an agentic AI design with clean OOP and provider-agnostic "
    "LLM backends (e.g., Bedrock, OpenAI, llama-cpp-python).\n\n"
    "Process:\n"
    "1) Call the builder tool to draft code.\n"
    "2) Call the reviewer tool to critique it.\n"
    "3) If the reviewer returns 'Approved', stop and return the latest code.\n"
    "4) Otherwise, call the builder again with the review feedback and iterate.\n\n"
    "IMPORTANT:\n"
    "- When calling tools, your args object MUST match the tool's argument schema shown in AVAILABLE TOOLS.\n"
    "- For these AgentTool tools (default Agents), call them with: {\"prompt\": \"...\"} (NOT wrapped in 'inputs').\n"
)

result = orchestrator.invoke({"prompt": task})

print("\n=== Final Result ===\n")
print(result)
