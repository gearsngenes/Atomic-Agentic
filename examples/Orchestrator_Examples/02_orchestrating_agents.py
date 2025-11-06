import os, sys, logging
from pathlib import Path

# Setting the repo root on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.LLMEngines import OpenAIEngine
from modules.ToolAgents import OrchestratorAgent
from modules.Agents import Agent

logging.getLogger().setLevel(logging.INFO)

# 1) LLM engine
llm = OpenAIEngine(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 2) Helper agents (schema-first; default pre-invoke expects {"prompt": str})
builder = Agent(
    name="CodeBuilderAgent",
    description="Generates Python code per request/revision.",
    llm_engine=llm,
    role_prompt=(
        "You are a senior software engineer who writes Python code for requested tasks.\n"
        "Return ONLY the code, with no explanations."
    ),
    context_enabled=True,
    history_window=10,
)

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
    history_window=10,
)

# 3) Orchestrator (sequential single-step loop)
orchestrator = OrchestratorAgent(
    name="AgenticOrchestrator",
    description="Orchestrates calls between the code builder and the code reviewer.",
    llm_engine=llm,
    history_window=10,
    max_steps=20,
    max_failures=5,
)

# 4) Register both agents as tools
orchestrator.register(builder)
orchestrator.register(reviewer)

# 5) Dynamic task (schema-first: mapping with 'prompt')
task = (
    "Write a Python class that scaffolds an agentic AI design with clean OOP and provider-agnostic "
    "LLM backends (e.g., Bedrock, OpenAI, llama-cpp-python). First, have the CodeBuilderAgent draft "
    "the implementation; then have CodeReviewer review and propose improvements; apply the improvements "
    "by calling the builder again; repeat until the reviewer returns 'Approved'; finally, return the "
    "latest approved code."
)

result = orchestrator.invoke({"prompt": task})

print("\n=== Final Result ===\n")
print(result)
