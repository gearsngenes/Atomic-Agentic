"""
05_SequentialFlow.py

SequentialFlow example using tool steps created via `toolify`.

Key point:
- toolify(...) returns a single Tool instance.
  Append each Tool individually to the tools list.
"""

from __future__ import annotations

from typing import Any, Dict, List

from atomic_agentic.tools import toolify
from atomic_agentic.workflows import SequentialFlow


# ───────────────────────────────────────────────────────────────────────────────
# Plain callables (toolified into Tools)
# ───────────────────────────────────────────────────────────────────────────────

def add_and_carry(x: int, y: int, factor: int = 10) -> Dict[str, int]:
    return x+y, factor


def multiply(value: int, factor: int) -> Dict[str, int]:
    return value * factor


def to_message(value: int) -> Dict[str, str]:
    return f"Final computed value = {value}"


def main() -> None:
    # toolify returns a single Tool — append it individually.
    tools: List[Any] = []
    tools.append(toolify(add_and_carry, name="add_and_carry", description="Add x+y and carry factor forward"))
    tools.append(toolify(multiply, name="multiply", description="Multiply value by factor"))
    tools.append(toolify(to_message, name="to_message", description="Format value as a message"))

    flow = SequentialFlow(
        name="demo_sequential",
        description="Demo SequentialFlow: add -> multiply -> format",
        steps=tools,  # <-- list of Tools (AtomicInvokable), not list-of-lists
    )

    inputs: Dict[str, Any] = {"x": 2, "y": 3}  # factor uses default 10
    final_result = flow.invoke(inputs)

    # Pull metadata from the flow's latest checkpoint
    flow_ckpt = flow.checkpoints[-1]
    meta = flow_ckpt.metadata

    print("\n=== Final packaged result ===")
    print(final_result)

    print("\n=== SequentialFlow metadata ===")
    print(meta)

    # Your SequentialFlow uses "midwork_checkpoints"
    step_ckpt_indices = meta.get("midwork_checkpoints", [])
    print("\n=== midwork_checkpoints indices ===")
    print(step_ckpt_indices)

    # Use the indices to retrieve each step wrapper's checkpoint record
    print("\n=== Per-step checkpoint lookup ===")
    for i, (step_wrapper, ckpt_idx) in enumerate(zip(flow.steps, step_ckpt_indices)):
        step_ckpt = step_wrapper.checkpoints[ckpt_idx]
        step_name = getattr(step_wrapper.component, "name", f"step_{i}")
        print(f"\nStep {i}: {step_name}")
        print(f"  checkpoint index: {ckpt_idx}")
        print(f"  inputs:  {step_ckpt.inputs}")
        print(f"  outputs: {step_ckpt.packaged_output}")
        print(f"  meta:    {step_ckpt.metadata}")


if __name__ == "__main__":
    main()
