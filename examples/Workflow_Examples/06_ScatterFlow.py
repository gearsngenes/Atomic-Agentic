# examples/Workflow_Examples/06_ScatterFlow.py

import sys, logging, json
from pathlib import Path
from typing import Any

# Project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Tools import Tool
from modules.Workflows import ScatterFlow

logging.getLogger().setLevel(logging.INFO)  # or DEBUG

# ------------------------------------------------------------
# Shared broadcast schema: {"prompt": <str>}
# Each branch returns a dict (recommended for flattening).
# ------------------------------------------------------------
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
    return flags  # disjoint keys; good for flattening

tool_word_count = Tool(
    name = "WordCount",
    func = word_count_tool,
    description="Counts words in the prompt",
)

tool_is_question = Tool(
    name = "IsQuestion",
    func = is_question_tool,
    description="Detects if the prompt ends with a question mark",
)

tool_keyword_flags = Tool(
    name = "KeywordFlags",
    func = keyword_flags_tool,
    description="Sets boolean flags for a few simple keywords",
)

# ------------------------------------------------------------
# Agent that accepts {"prompt": ...} and returns a dict.
# New framework note:
#   Agents expose arguments_map via their *pre_invoke* Tool.
#   We add a tiny identity pre-tool so the Agent’s schema is {"prompt"}.
# ------------------------------------------------------------
def _prompt_identity(prompt: str) -> dict:
    # Pass-through pre-invoke that normalizes the broadcast input
    return prompt

pre_prompt = Tool(
    name = "PromptInput",
    func = _prompt_identity,
    description="Normalizes user prompt into {'prompt': <str>}",
)

LLM = OpenAIEngine(model="gpt-4o-mini")

jokester = Agent(
    name="Jokester",
    description="Tells short, friendly jokes; returns a JSON object with a 'joke' field.",
    role_prompt="""
    You are a jokester who responds concisely as a JSON object with a 'joke' field.
    Do not add any ```json``` fencing around your response.
    """,
    llm_engine=LLM,
    context_enabled=False,
    pre_invoke=pre_prompt,  # <-- ensures Agent's arguments_map matches the tools'
)

# ------------------------------------------------------------
# Build ScatterFlow (broadcast fan-out)
#   • input_schema is derived dynamically from the first branch's arguments_map
#   • all branches must share the same input schema (enforced by ScatterFlow)
#   • name_collision_policy defaults to "fail_fast"
# ------------------------------------------------------------

# Example A: flatten=False (group results by branch name)
scatter_grouped = ScatterFlow(
    name="ScatterFlowGrouped",
    description="Broadcasts one input to all branches; returns a dict keyed by branch names.",
    branches=[jokester, tool_word_count, tool_is_question, tool_keyword_flags],
    output_schema=["batch_results"],  # packaged under this key by base invoke()
    bundle_all=True,
    flatten=False,                    # grouped by branch name
    # name_collision_policy="fail_fast",  # optional: "skip" or "replace"
)

# Example B: flatten=True → merge branch dicts into a single flat dict
# Collision policy is right-biased (later branches win).
scatter_flat = ScatterFlow(
    name="ScatterFlowFlat",
    description="Broadcasts one input to all branches; returns a single flattened dict of results.",
    branches=[tool_word_count, tool_is_question, tool_keyword_flags, jokester],
    output_schema=["word_count", "is_question", "kw_flags", "joke"],  # wrapped under this key by base invoke()
    bundle_all=False,
    flatten=True,                     # flatten branch dicts into one dict
)

# ------------------------------------------------------------
# Prepare the *broadcast* input (DICT-ONLY!)
# Same dict is sent to every branch because this is a broadcast flow.
# ------------------------------------------------------------
broadcast_inputs: dict[str, Any] = {
    "prompt": "Tell me a quick computer joke. Also, how many words are in this sentence?"
}

def pretty(title: str, obj: Any):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=2))

if __name__ == "__main__":
    # --- Grouped result: { "batch_results": { <branch>: <result>, ... } }
    res_grouped = scatter_grouped.invoke(broadcast_inputs)
    pretty("SCATTER (grouped by branch name)", res_grouped)

    # --- Flattened result: { "batch_results": { <k>: <v>, ... } }
    # Flattening rules in new framework:
    #   • If a branch returns {WF_RESULT: x} or {JUDGE_RESULT: x}, it is unwrapped and stored as {<branch_name>: x}
    #   • If a branch returns a plain mapping, its pairs are merged (right-biased) into the flat dict
    #   • If a branch returns a non-mapping, it is stored as {<branch_name>: value}
    res_flat = scatter_flat.invoke(broadcast_inputs)
    pretty("SCATTER (flattened single dict)", res_flat)
