"""
BasicFlow Example: Beginner-Friendly Introduction
------------------------------------------------
This example demonstrates wrapping a simple Python function as a Tool, then
using BasicFlow to expose it as a workflow-shaped node.
"""

from pprint import pprint

from atomic_agentic.tools import Tool
from atomic_agentic.workflows import BasicFlow


def square_plus_one(x: int) -> int:
    """Return x^2 + 1 as a raw integer."""
    return (x * x) + 1


def main() -> None:
    print("\n=== BasicFlow: Beginner Example ===")

    flow = BasicFlow(
        component=Tool(square_plus_one),
        name="square_flow",
        description="Workflow wrapper around a simple tool.",
    )

    # --- Workflow Calls ---
    print("\n--- Workflow Calls ---")
    results = []
    for x in (3, 4, 5):
        flow_result = flow.invoke({"x": x})
        results.append(flow_result)
        print(f"x={x} -> result={flow_result.result}, run_id={flow_result.run_id}")

    # --- Checkpoints ---
    print("\n--- Checkpoints ---")
    print("checkpoint count:", len(flow.checkpoints))

    # lookup by positional index
    first_checkpoint = flow.get_checkpoint(0)
    print("\nfirst checkpoint (by index 0):")
    pprint(first_checkpoint)

    # lookup by WorkflowResult identity
    second_checkpoint = flow.get_checkpoint(results[1])
    print("\nsecond checkpoint (by result identity):")
    pprint(second_checkpoint)

    # lookup by run_id
    third_checkpoint = flow.get_checkpoint(results[2].run_id)
    print("\nthird checkpoint (by run_id):")
    pprint(third_checkpoint)

    # --- Serialization Snapshot ---
    print("\n--- Serialization Snapshot ---")
    print("\nflow.to_dict():")
    pprint(flow.to_dict())

    print("\nAll checks passed if no errors above.")


if __name__ == "__main__":
    main()
