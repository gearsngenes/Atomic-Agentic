
"""
03_SequentialFlow.py

Beginner-friendly SequentialFlow example.
Each step is a StructuredInvokable (schema-wrapped Tool), ensuring clear data flow and output contracts.
"""

from __future__ import annotations


from typing import Any, Dict, List
from pprint import pprint

from atomic_agentic.tools import toolify
from atomic_agentic.workflows import SequentialFlow
from atomic_agentic import StructuredInvokable


# ───────────────────────────────────────────────────────────────────────────────
# Plain callables (toolified into Tools)
# ───────────────────────────────────────────────────────────────────────────────


def add_and_carry(x: int, y: int, factor: int = 10) -> tuple[int, int]:
    """Step 1: Add x + y, carry forward 'factor'"""
    return x + y, factor

def multiply(value: int, factor: int) -> int:
    """Step 2: Multiply value by factor"""
    return value * factor

def to_message(value: int) -> str:
    """Step 3: Format value as a message"""
    return f"Final computed value = {value}"



def main() -> None:
    # Wrap each function as a Tool, then as a StructuredInvokable with explicit output schema
    step1 = StructuredInvokable(
        component=add_and_carry,
        name="add_and_carry",
        description="Add x+y and carry factor forward",
        output_schema=["value", "factor"]
    )
    step2 = StructuredInvokable(
        component=multiply,
        name="multiply",
        description="Multiply value by factor",
        output_schema=["value"]
    )
    step3 = StructuredInvokable(
        component=to_message,
        name="to_message",
        description="Format value as a message",
        output_schema=["message"]
    )

    flow = SequentialFlow(
        name="demo_sequential",
        description="Demo SequentialFlow: add -> multiply -> format",
        steps=[step1, step2, step3],
    )

    # Only x and y are required; factor uses default 10
    inputs = {"x": 2, "y": 3}
    final_result = flow.invoke(inputs)

    print("\n=== Final packaged result ===")
    pprint(dict(final_result))

    print("\n=== SequentialFlow run_id ===")
    print(final_result.run_id)

    print("\n=== Checkpoints ===")
    for i, step in enumerate(flow.steps):
        print(f"\nStep {i}: {step.component.name}")
        for ckpt in step.checkpoints:
            print(f"  run_id: {ckpt.run_id}")
            print(f"  inputs: {ckpt.inputs}")
            print(f"  result: {dict(ckpt.result)}")

    print("\nAll steps and outputs complete.")


if __name__ == "__main__":
    main()
