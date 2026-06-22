"""
02_SequentialFlow.py

Beginner-friendly SequentialFlow example.
Step 1 wraps a tuple-returning function in StructuredInvokable to adapt it
into a mapping-shaped result. Steps 2 and 3 are plain Tools whose functions
already return dicts with the keys the next step (or the caller) expects.
"""

from __future__ import annotations

from pprint import pprint

from atomic_agentic.workflows import SequentialFlow
from atomic_agentic import StructuredInvokable
from atomic_agentic.tools import Tool


def add_and_carry(x: int, y: int, factor: int = 10) -> tuple[int, int]:
    """Step 1: Add x + y, carry forward 'factor'"""
    return {"value": x + y, "factor": factor}

def multiply(value: int, factor: int) -> dict:
    """Step 2: Multiply value by factor, return as a dict"""
    return {"value": value * factor}

def to_message(value: int) -> dict:
    """Step 3: Format value as a message dict"""
    return {"message": f"Final computed value = {value}"}


def main() -> None:
    step1 = Tool(add_and_carry)
    step2 = Tool(multiply)
    step3 = Tool(to_message)

    flow = SequentialFlow(
        name="demo_sequential",
        namespace="examples",
        description="Demo SequentialFlow: add -> multiply -> format",
        steps=[step1, step2, step3],
    )

    # Only x and y are required; factor uses default 10
    inputs = {"x": 2, "y": 3}
    final_result = flow.invoke(inputs)

    print("\n=== Final result ===")
    pprint(final_result.result)

    print("\n=== SequentialFlow run_id ===")
    print(final_result.run_id)

    print("\n=== step_runs / return_index ===")
    print("step_runs:", final_result.step_runs)
    print("return_index:", final_result.return_index)

    print("\n=== Per-step results (get_step_results) ===")
    for i, step_result in enumerate(flow.get_step_results(final_result.run_id)):
        print(f"Step {i}: run_id={step_result.run_id}, result={step_result.result}")

    print("\n=== Checkpoints ===")
    for i, step in enumerate(flow.steps):
        print(f"\nStep {i}: {step.component.name}")
        for ckpt in step.checkpoints:
            print(f"  run_id: {ckpt.result.run_id}")
            print(f"  inputs: {ckpt.inputs}")
            print(f"  result: {ckpt.result.result}")

    print("\nAll steps and outputs complete.")


if __name__ == "__main__":
    main()
