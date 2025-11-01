import sys, logging, json
from pathlib import Path
from typing import Any

# Project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Tools import Tool
from modules.Workflows import ScatterFlow

logging.getLogger().setLevel(logging.INFO)

# -------------------------
# Define simple tools that accept the *same* broadcast schema: {"prompt": ...}
# Each returns a *dict* (required by ScatterFlow).
# -------------------------
def word_count_tool(prompt: str) -> dict:
    words = (prompt or "").split()
    return {"word_count": len(words)}

def is_question_tool(prompt: str) -> dict:
    s = (prompt or "").strip()
    return {"is_question": s.endswith("?")}

def keyword_flags_tool(prompt: str) -> dict:
    s = (prompt or "").lower()
    flags = {
        "mentions_ai": "ai" in s,
        "mentions_math": any(k in s for k in ["math", "algebra", "geometry", "calculus"]),
        "mentions_dino": any(k in s for k in ["dinosaur", "dinosaurs", "t. rex", "trex", "tyrannosaurus"]),
    }
    return flags  # dict of disjoint keys

tool_word_count = Tool(
    "WordCount",
    word_count_tool,
    description="Counts words in the prompt",
)

tool_is_question = Tool(
    "IsQuestion",
    is_question_tool,
    description="Detects if the prompt ends with a question mark",
)

tool_keyword_flags = Tool(
    "KeywordFlags",
    keyword_flags_tool,
    description="Sets boolean flags for a few simple keywords",
)

# -------------------------
# Optional: an Agent that also accepts {"prompt": ...} and returns a dict.
# (Depending on your AgentFlow defaults, this typically yields a dict result.)
# -------------------------
LLM = OpenAIEngine(model="gpt-4o-mini")

jokester = Agent(
    name="Jokester",
    description="Tells short, friendly jokes.",
    role_prompt="""
    You are a jokester who responds concisely as a JSON object with a 'joke' field. 
    Do not add any additional ```json``` style fencing around your response""",
    llm_engine=LLM,
    context_enabled=False,
)

# -------------------------
# Build ScatterFlow (broadcast fan-out)
#   • input_schema is immutable and shared by all branches
#   • branches must accept the same input_schema (set-equivalent)
# -------------------------

# Example A: flatten=False (default) → results grouped by branch name
scatter_grouped = ScatterFlow(
    name="ScatterFlowGrouped",
    description="Broadcasts one input to all branches; returns a dict keyed by branch names.",
    input_schema=["prompt"],
    branches=[jokester, tool_word_count, tool_is_question, tool_keyword_flags],
    output_schema=["batch_results"],   # packaged under this key by base invoke()
    bundle_all=True,
    flatten=False,                     # grouped by branch name
)

# Example B: flatten=True → merge branch dicts into a single flat dict (with collision checks)
# Avoid collisions: each branch should produce distinct keys.
scatter_flat = ScatterFlow(
    name="ScatterFlowFlat",
    description="Broadcasts one input to all branches; returns a single flattened dict of results.",
    input_schema=["prompt"],
    branches=[tool_word_count, tool_is_question, tool_keyword_flags, jokester],  # tools produce disjoint keys
    output_schema=["batch_results"],   # wrapped under this key by base invoke()
    bundle_all=True,
    flatten=True,                      # flatten branch dicts into one dict
)

# -------------------------
# Prepare the *broadcast* input (DICT-ONLY!)
# Same dict is sent to every branch because this is a broadcast flow.
# -------------------------
broadcast_inputs: dict[str, Any] = {
    "prompt": "Tell me a quick computer joke. Also, how many words are in this sentence?"
}

def pretty(title: str, obj: Any):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=2))

if __name__ == "__main__":
    # --- Grouped result: { "batch_results": { <branch>: <dict>, ... } }
    res_grouped = scatter_grouped.invoke(broadcast_inputs)
    pretty("SCATTER (grouped by branch name)", res_grouped)

    # --- Flattened result: { "batch_results": { <key>: <value>, ... } }
    #     NOTE: if any branch returns a single-key dict keyed by WF_RESULT/JUDGE_RESULT,
    #     ScatterFlow will insert it as { <branch_name>: <raw_result_dict> } instead of flattening it.
    res_flat = scatter_flat.invoke(broadcast_inputs)
    pretty("SCATTER (flattened single dict)", res_flat)
