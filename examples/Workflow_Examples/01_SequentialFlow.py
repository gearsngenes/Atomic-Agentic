"""
01_SequentialFlow.py

Beginner-friendly SequentialFlow example.
Step 1 wraps a tuple-returning function in StructuredInvokable to adapt it
into a mapping-shaped result. Steps 2 and 3 are plain Tools whose functions
already return dicts with the keys the next step (or the caller) expects.
"""

from __future__ import annotations

from pprint import pprint

from atomic_agentic.workflows import SequentialFlow
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

    print("\n=== return_index ===")
    print("return_index:", final_result.return_index)

    print("\n=== Per-step results (trace) ===")
    # trace holds every step's own AtomicResult, in step order, whenever
    # include_trace is enabled (the default). trace[return_index] is the
    # same result object that produced final_result.result.
    for i, step_result in enumerate(final_result.trace):
        print(f"Step {i}: run_id={step_result.run_id}, result={step_result.result}")

    print("\n=== include_trace=False: no per-step visibility ===")
    lean_flow = SequentialFlow(
        name="demo_sequential_lean",
        namespace="examples",
        description="Same pipeline, tracing disabled.",
        steps=[step1, step2, step3],
        include_trace=False,
    )
    lean_result = lean_flow.invoke(inputs)
    print("trace:", lean_result.trace)  # None -- no per-step run ids at all

    print("\nAll steps and outputs complete.")


if __name__ == "__main__":
    main()
