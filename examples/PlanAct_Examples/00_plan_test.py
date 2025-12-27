import os
from dotenv import load_dotenv

from atomic_agentic.agents.tool_agents import PlanActAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()

# LLM engine
llm_engine = OpenAIEngine(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# PlanAct agent (ReWOO-style: one plan LLM call, then execute tools)
agent = PlanActAgent(
    name="Test-PlanAct",
    description="Testing one-shot planning + execution over local python tools.",
    llm_engine=llm_engine,
    context_enabled=True,    # True => persists blackboard across runs
    run_concurrent=False,     # True => run dependency waves concurrently
)

# Simple local tools (docstrings become tool descriptions if you omit description=...)
def tool_1(seed: int) -> str:
    """Processes seed input and passes it to the next tool."""
    print("Tool 1 executed")
    return f"1) Result from tool_1. Seed was: {seed}"


def tool_2(t1_result: str) -> str:
    """Processes the result from tool_1."""
    print("Tool 2 executed")
    return t1_result + "\n2) Result from tool_2"


def tool_3(t2_result: str) -> str:
    """Finalizes the result based on tool_2 output."""
    print("Tool 3 executed")
    return t2_result + "\n3) Result from tool_3"


# Register tools (callables are wrapped into Tool instances via toolify)
t1_full = agent.register(tool_1, namespace="local")[0]
t2_full = agent.register(tool_2, namespace="local")[0]
t3_full = agent.register(tool_3, namespace="local")[0]

seed = 32

# Use the fully-qualified tool ids to make the example deterministic.
task_prompt = (
    f"Call the tools 1, 2, and 3 in sequential order, with the initial seed = {seed}"
)

final = agent.invoke({"prompt": task_prompt})

print("\nFinal Result:\n", final)

# Optional: inspect executed steps (even if context_enabled=False, the view is useful for debugging)
print("\nBlackboard (resolved args):\n", agent.blackboard_dumps(raw_results=True))
