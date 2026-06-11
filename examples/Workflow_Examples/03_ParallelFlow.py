"""
03_ParallelFlow.py

Beginner-friendly ParallelFlow example.
Demonstrates running multiple analysis branches on the same input in
parallel, projecting results as a dict (output_type="dict") and as a list
(output_type="list", with a reordered subset of branches).
"""

from __future__ import annotations
from pprint import pprint
from atomic_agentic.tools import Tool
from atomic_agentic.workflows import ParallelFlow

# ──────────────────────────────────────────────────────────────
# Branch functions: each returns a mapping with clear keys
# ──────────────────────────────────────────────────────────────
def word_stats(text: str) -> dict[str, int]:
    """Return word and unique word counts."""
    words = [token for token in text.split() if token.strip()]
    unique_words = {token.lower().strip(".,!?;:()[]{}\"'") for token in words}
    return {
        "word_count": len(words),
        "unique_word_count": len({w for w in unique_words if w}),
    }

def line_stats(text: str) -> dict[str, int]:
    """Return line and non-empty line counts."""
    lines = text.splitlines()
    non_empty = [line for line in lines if line.strip()]
    return {
        "line_count": len(lines),
        "non_empty_line_count": len(non_empty),
    }

def text_flags(text: str) -> dict[str, bool]:
    """Return simple boolean flags about the text."""
    stripped = text.strip()
    return {
        "has_question_mark": "?" in text,
        "is_long_text": len(stripped) > 120,
        "starts_with_heading": stripped.startswith("#"),
    }

# ──────────────────────────────────────────────────────────────
# Wrap each function as a Tool — branches no longer need to be
# Mapping-shaped via StructuredInvokable; ParallelFlow does no
# Mapping validation on branch results.
# ──────────────────────────────────────────────────────────────
word_stats_tool = Tool(
    function=word_stats,
    name="word_stats",
    namespace="examples",
    description="Compute word and unique word counts.",
)

line_stats_tool = Tool(
    function=line_stats,
    name="line_stats",
    namespace="examples",
    description="Compute line and non-empty line counts.",
)

text_flags_tool = Tool(
    function=text_flags,
    name="text_flags",
    namespace="examples",
    description="Compute simple boolean flags for text.",
)

# ──────────────────────────────────────────────────────────────
# Build the ParallelFlow: dict output, all branches, named keys
# ──────────────────────────────────────────────────────────────
parallel = ParallelFlow(
    name="text_parallel_analysis",
    description="Run several text-analysis branches in parallel.",
    branches=[
        word_stats_tool,
        line_stats_tool,
        text_flags_tool,
    ],
    parameters=["text"],
    output_type=ParallelFlow.DICT,
    output_names=["words", "lines", "flags"],
)

sample_text = """# Release Notes\n\nParallel workflows should be easy to reason about.\nCan we preserve branch identity without making downstream composition ugly?\n\nThis short sample exists only to exercise the flow.\n"""

# ──────────────────────────────────────────────────────────────
# Run the parallel flow and display results
# ──────────────────────────────────────────────────────────────
result = parallel.invoke({"text": sample_text})

print("\n=== DICT RESULT ===")
pprint(result.result)
print("\nrun_id:", result.run_id)
print("branch_runs:", result.branch_runs)
print("output_indices:", result.output_indices)

print("\n=== CHILD BRANCH RESULTS RESOLVED FROM PARENT RUN ===")
for i, branch_result in enumerate(parallel.get_branch_results(result.run_id)):
    print(f"Branch {i}: run_id={branch_result.run_id}, result={branch_result.result}")

# ──────────────────────────────────────────────────────────────
# A second flow: list output, reordered subset of branches
# (output projection is fixed at construction, so this is a
# separate ParallelFlow instance rather than a reconfiguration)
# ──────────────────────────────────────────────────────────────
parallel_subset = ParallelFlow(
    name="text_parallel_subset",
    description="Project only the text-flags and word-stats branches, as a list.",
    branches=[
        word_stats_tool,
        line_stats_tool,
        text_flags_tool,
    ],
    parameters=["text"],
    output_type=ParallelFlow.LIST,
    output_indices=[2, 0],
)

list_result = parallel_subset.invoke({"text": sample_text})

print("\n=== LIST RESULT (branches 2 then 0) ===")
pprint(list_result.result)
print("\nrun_id:", list_result.run_id)
print("output_indices:", list_result.output_indices)

print("\n=== FLOW SNAPSHOT ===")
pprint(parallel.to_dict())
