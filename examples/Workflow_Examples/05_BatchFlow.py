import sys, logging, json
from pathlib import Path
from typing import Any

# Project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.ToolAgents import PlannerAgent
from modules.Workflows import BatchFlow
from modules.Tools import Tool
from modules.Plugins import MathPlugin

logging.getLogger().setLevel(logging.INFO)

# -------------------------
# Define agents & tools
# -------------------------
LLM = OpenAIEngine(model="gpt-4o")

# a paleontology expert (expects {"prompt": ...} when invoked via AgentFlow)
agent1 = Agent(
    name="Paleontologist",
    description="An expert in paleontology",
    role_prompt="You are a paleontology expert. Answer the user's questions about dinosaurs with detailed and accurate information.",
    llm_engine=LLM,
    context_enabled=True,
)

# a mathematician planner agent (expects {"prompt": ...})
agent2 = PlannerAgent(
    name="Mathematician",
    description="A mathematician who excels in calculations",
    llm_engine=LLM,
    context_enabled=True,
    is_async=True,
)
agent2.register(MathPlugin)

# a joke-telling agent (expects {"prompt": ...})
agent3 = Agent(
    name="Jokester",
    description="Tells jokes",
    role_prompt="You are a jokester who tells short, friendly jokes.",
    llm_engine=LLM,
)

# a string length-comparing tool (expects {"str1": ..., "str2": ...})
def compare_lengths(str1: str, str2: str) -> str:
    if len(str1) == len(str2):
        return f"Lengths of '{str1}' and '{str2}' are equal"
    longer = len(str1) > len(str2)
    if longer:
        return f"'{str1}' is longer than '{str2}'"
    return f"'{str2}' is longer than '{str1}'"

tool1 = Tool(
    "Compare_Lengths",
    compare_lengths,
    description="Compares the lengths of two strings and returns an analysis of which is longer",
)

# -------------------------
# Build BatchFlow
# -------------------------
# Example A: unwrap_outputs=True with custom output labels (stable via internal label map)
batch_unwrapped = BatchFlow(
    name="BatchFlowExampleUnwrapped",
    description="Runs branches in parallel and returns a flat dict keyed by custom labels",
    branches=[agent1, agent2, agent3, tool1],
    unwrap_outputs=True,
    labels=["paleo", "math", "joke", "comp"],  # must match number of branches; unique strings
)

# Example B: unwrap_outputs=False (default) with a single output_key
batch_wrapped = BatchFlow(
    name="BatchFlowExampleWrapped",
    description="Runs branches in parallel and returns all results under a single key",
    branches=[agent1, agent2, agent3, tool1],
    unwrap_outputs=False,
    output_key="batch_results",  # defaults to WF_RESULT if not provided
)

# -------------------------
# Prepare inputs (DICT-ONLY!)
# Each branch receives a dict under its own name:
#   - Agents (AgentFlow) expect {"prompt": "..."}
#   - Tools (ToolFlow) expect their own parameter names
# -------------------------
inputs: dict[str, Any] = {
    "Paleontologist": {"prompt": "Tell me about Tyrannosaurus rex."},
    "Mathematician": {"prompt": "Compute 2**16 divided by three."},
    "Jokester": {"prompt": "Tell me a one-liner about computers."},
    "Compare_Lengths": {"str1": "dogs", "str2": "dolphins"},
}

def pretty(title: str, obj: Any):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=2))

if __name__ == "__main__":
    # --- Unwrapped mode: result has one key per branch label (paleo/math/joke/comp)
    res_unwrapped = batch_unwrapped.invoke(inputs)
    pretty("UNWRAPPED RESULT", res_unwrapped)
    # Example shape:
    # {
    #   "paleo": {...},   # result dict from Paleontologist
    #   "math":  {...},   # result dict from Mathematician
    #   "joke":  {...},   # result dict from Jokester
    #   "comp":  {...}    # result dict from Compare_Lengths
    # }

    # --- Wrapped mode: result is { "batch_results": { <branch_name>: <branch_result_dict>, ... } }
    res_wrapped = batch_wrapped.invoke(inputs)
    pretty("WRAPPED RESULT", res_wrapped)
    # Example shape:
    # {
    #   "batch_results": {
    #     "Paleontologist": {...},
    #     "Mathematician":  {...},
    #     "Jokester":       {...},
    #     "Compare_Lengths": {...}
    #   }
    # }