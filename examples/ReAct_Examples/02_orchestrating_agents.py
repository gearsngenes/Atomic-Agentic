import logging

from dotenv import load_dotenv

from atomic_agentic.agents import Agent
from atomic_agentic.agents import ReActAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

llm = OpenAIEngine(model="gpt-5-mini")

def builder_prestep(task: str | None = None, revision_notes: str | None = None) -> str:
    if revision_notes:
        return f"""
    Read and internalize the following feedback from a code, and once you have, use your best judgement
    from the revision notes to re-build the last code you : {revision_notes}\n\n
    Please provide the updated code."""
    elif task:
        return f"Implement code so that it meets the user's request:\n{task}"
    else:
        raise ValueError("Either task or revision_notes must be provided.")

builder = Agent(
    name="CodeBuilderAgent",
    description="""
    Generates Python code per user request OR revises its latest drafts from revision notes. If this is
    the first draft of a code, provide ONLY the task. If you are sending feedback to revise the latest
    draft, send ONLY the revision notes.""",
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
    return f"Please review and provide feedback for the following code: ```python\n{draft_code}\n```"

reviewer = Agent(
    name="CodeReviewer",
    description="Reviews draft code from the builder and returns revision suggestions or 'Approved'.",
    llm_engine=llm,
    role_prompt=(
        "You are an expert Python code analyst. Thoroughly and brutally evaluate the code for "
        "accuracy, readability, and overall design optimization. Return ONLY revision critiques that "
        "you deem are critical or necessary for the the code to be ready to hand off to a professional "
        "developer for future use. Such critiques should focus on:\n"
        "- Syntax or semantic errors in the code (high priority fixes)\n"
        "- Redundant or duplicate code that could be refactored into reusable chunks\n"
        "- Overly complex or irrelevant/unused code that isn't needed for the codebuilder's task\n\n"
        "If none of these types of significant errors are present in a major way, "
        "return a single 'Approved' flag."
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
    tool_calls_limit=10,
    context_enabled=False,
    preview_limit=100,
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
    "4) Otherwise, call the builder again and send ONLY the revision notes feedback and iterate.\n\n"
    "Note: Do NOT try use the ENTIRE # of tool call limits you have. consider this if you start "
    "approaching your limit"
)

result = orchestrator.invoke({"prompt": task})

print("\n=== Final Result ===\n")
print(result)
