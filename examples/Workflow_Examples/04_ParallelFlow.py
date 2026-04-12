"""
04_ParallelFlow.py

Beginner-friendly ParallelFlow example.
Demonstrates running multiple analysis branches on the same input in parallel, with clear output structure and metadata inspection.
"""

from __future__ import annotations
from pprint import pprint
from atomic_agentic.tools import Tool
from atomic_agentic.workflows.StructuredInvokable import StructuredInvokable
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
# Wrap each function as a Tool, then as a StructuredInvokable branch
# ──────────────────────────────────────────────────────────────
word_stats_branch = StructuredInvokable(
    component=Tool(
        function=word_stats,
        name="word_stats",
        namespace="examples",
        description="Compute word and unique word counts.",
    ),
    name="word_stats_branch",
    description="Structured branch for word stats.",
    output_schema=["word_count", "unique_word_count"],
)

line_stats_branch = StructuredInvokable(
    component=Tool(
        function=line_stats,
        name="line_stats",
        namespace="examples",
        description="Compute line and non-empty line counts.",
    ),
    name="line_stats_branch",
    description="Structured branch for line stats.",
    output_schema=["line_count", "non_empty_line_count"],
)

text_flags_branch = StructuredInvokable(
    component=Tool(
        function=text_flags,
        name="text_flags",
        namespace="examples",
        description="Compute simple boolean flags for text.",
    ),
    name="text_flags_branch",
    description="Structured branch for text flags.",
    output_schema=["has_question_mark", "is_long_text", "starts_with_heading"],
)

# ──────────────────────────────────────────────────────────────
# Build the ParallelFlow
# ──────────────────────────────────────────────────────────────
parallel = ParallelFlow(
    name="text_parallel_analysis",
    description="Run several text-analysis branches in parallel.",
    branches=[
        word_stats_branch,
        line_stats_branch,
        text_flags_branch,
    ],
    input_shape=ParallelFlow.BROADCAST,
    parameters=["text"],
    output_shape=ParallelFlow.NESTED,
    output_indices=[0, 1, 2],
    output_names=["words", "lines", "flags"],
    duplicate_key_policy=ParallelFlow.RAISE,
)

sample_text = """# Release Notes\n\nParallel workflows should be easy to reason about.\nCan we preserve branch identity without making downstream composition ugly?\n\nThis short sample exists only to exercise the flow.\n"""

# ──────────────────────────────────────────────────────────────
# Run the parallel flow and display results
# ──────────────────────────────────────────────────────────────
result = parallel.invoke({"text": sample_text})

print("\n=== NESTED RESULT ===")
pprint(dict(result))
print("\nrun_id:", result.run_id)

print("\n=== PARENT CHECKPOINT METADATA ===")
checkpoint = parallel.get_checkpoint(result.run_id)
if checkpoint is not None:
    pprint(checkpoint.metadata)

print("\n=== CHILD BRANCH RESULTS RESOLVED FROM PARENT RUN ===")
pprint(parallel.get_branch_results(result.run_id))

# ──────────────────────────────────────────────────────────────
# Reconfigure output: flatten and select only two branches
# ──────────────────────────────────────────────────────────────
parallel.configure_output(
    output_indices=[2, 0],
    output_shape=ParallelFlow.FLATTENED,
    output_names=None,
    duplicate_key_policy=ParallelFlow.RAISE,
)

flattened_result = parallel.invoke({"text": sample_text})

print("\n=== FLATTENED RESULT (branches 2 then 0) ===")
pprint(dict(flattened_result))
print("\nrun_id:", flattened_result.run_id)

print("\n=== UPDATED FLOW SNAPSHOT ===")
pprint(parallel.to_dict())
