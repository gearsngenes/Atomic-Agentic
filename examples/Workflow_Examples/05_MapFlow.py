import sys, logging, json
from pathlib import Path
from typing import Any

# Project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Tools import Tool
from modules.Workflows import MapFlow

logging.getLogger().setLevel(logging.INFO)

# -------------------------
# Branch tools (each returns a dict)
# -------------------------
def to_upper(text: str) -> dict:
    return {"upper": (text or "").upper()}

def word_stats(text: str) -> dict:
    s = (text or "")
    return {"chars": len(s), "words": len(s.split())}

def extract_hashtags(text: str) -> dict:
    s = (text or "")
    tags = [t[1:] for t in s.split() if t.startswith("#") and len(t) > 1]
    return {"hashtags": tags}

tool_upper = Tool(
    "Uppercase",
    to_upper,
    description="Uppercases the provided text.",
)

tool_stats = Tool(
    "WordStats",
    word_stats,
    description="Counts characters and words in the provided text.",
)

tool_tags = Tool(
    "Hashtags",
    extract_hashtags,
    description="Extracts hashtags (without the #) from the provided text.",
)

# -------------------------
# Build MapFlow (tailored fan-out)
#   • input_schema is always the ordered list of current branch names
#   • each branch receives inputs[branch.name] (a dict) if present
#   • missing branches: result=None (flatten=False) or ignored (flatten=True)
# -------------------------

# Grouped results: { "map_results": { "Uppercase": {...} | None, ... } }
map_grouped = MapFlow(
    name="MapFlowGrouped",
    description="Tailored fan-out; returns dict keyed by branch names with dict or None.",
    branches=[tool_upper, tool_stats, tool_tags],
    output_schema=["map_results"],
    bundle_all=True,
    flatten=False,  # keep per-branch buckets
)

# Flattened results: { "map_results": { <merged keys> } }
#   - single-key envelopes using WF_RESULT/JUDGE_RESULT would be inserted as {branch: {...}} (none here)
map_flat = MapFlow(
    name="MapFlowFlat",
    description="Tailored fan-out; returns a single flattened dict merged from branches.",
    branches=[tool_upper, tool_stats, tool_tags],
    output_schema=["map_results"],
    bundle_all=True,
    flatten=True,   # merge keys across branches; collisions raise
)

def pretty(title: str, obj: Any):
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=2))

if __name__ == "__main__":
    # MapFlow input payloads MUST be dicts addressed by branch name.
    # Each addressed payload must itself be a dict that matches the branch's expected input.
    # Our tools each expect {"text": "..."} as their input dict (ToolFlow maps this to text: str).

    # 1) Partial addressing (missing one branch)
    inputs_partial = {
        "Uppercase": {"text": "hello, mapflow."},
        "WordStats": {"text": "how many words are in this sentence?"}
        # "Hashtags" key is intentionally missing
    }

    grouped_partial = map_grouped.invoke(inputs_partial)
    pretty("MAP (grouped) — partial addressing (missing 'Hashtags')", grouped_partial)

    flat_partial = map_flat.invoke(inputs_partial)
    pretty("MAP (flattened) — partial addressing (missing 'Hashtags')", flat_partial)

    # 2) Full addressing (all branches present)
    inputs_full = {
        "Uppercase": {"text": "scatter & map flows are siblings."},
        "WordStats": {"text": "short and sweet."},
        "Hashtags": {"text": "mixing #AI with #python and #dev"}
    }

    grouped_full = map_grouped.invoke(inputs_full)
    pretty("MAP (grouped) — full addressing", grouped_full)

    flat_full = map_flat.invoke(inputs_full)
    pretty("MAP (flattened) — full addressing", flat_full)

    # 3) Edge: unexpected key (should be rejected by base validation)
    # Uncomment to observe validation error:
    # bad_inputs = {"NotABranch": {"text": "oops"}}
    # map_grouped.invoke(bad_inputs)
