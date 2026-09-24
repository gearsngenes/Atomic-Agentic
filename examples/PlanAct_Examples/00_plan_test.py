from atomic_agentic.agents import PlanActAgent

from shared_engine import llm_engine

# PlanAct agent (ReWOO-style: one plan LLM call, then execute tools)
agent = PlanActAgent(
    name="Test_PlanAct",
    namespace="examples",
    description="Testing one-shot planning + execution over local python tools.",
    llm_engine=llm_engine,
    context_enabled=True,    # True => persists conversation history across runs
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


# Register tools (callables are toolified under agent.name as namespace;
# each is reachable by its own bare name -- tool_1/tool_2/tool_3 -- since no
# alias is given).
agent.register_tools([tool_1, tool_2, tool_3])

seed = 32

task_prompt = (
    f"Call the tools tool_1, tool_2, and tool_3 in sequential order, with the initial seed = {seed}"
)

final = agent.invoke({"prompt": task_prompt})

print("\nFinal Result:\n", final.result)

# Optional: inspect the executed plan (even if context_enabled=False, the view is useful for debugging)
from pprint import pprint
print("\nExecuted calls:\n")
pprint(agent.get_conversation()[-1].statements)
