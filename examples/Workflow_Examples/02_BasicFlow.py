"""
This file has been replaced by 01_BasicFlow.py, which provides a clear, beginner-friendly BasicFlow example.
Please see 01_BasicFlow.py for the updated workflow demonstration.
"""

# examples/Workflow_Examples/01_BasicFlow.py
"""
BasicFlow Example: Beginner-Friendly Introduction
------------------------------------------------
This example demonstrates how to wrap a simple Python function as a Tool, then use StructuredInvokable and BasicFlow to create a clear, schema-driven workflow.
"""

from pprint import pprint

from atomic_agentic import StructuredInvokable
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.basic import BasicFlow


def square_plus_one(x: int) -> int:
    """Return x^2 + 1 as a raw integer."""
    return (x * x) + 1


def main() -> None:
    print("\n=== BasicFlow: Beginner Example ===")


    flow = BasicFlow(
        component=StructuredInvokable(
            component=Tool(square_plus_one),
            output_schema=["value"],  # output will be {"value": ...}
        ),
        name="square_flow",
        description="Workflow wrapper around a structured single-value tool.",
    )


    # --- Workflow Call ---
    print("\n--- Workflow Call ---")
    flow_result = flow.invoke({"x": 3})
    print("flow mapping:", dict(flow_result))
    print("flow run_id:", flow_result.run_id)

    # --- Checkpoints and Serialization ---
    print("\n--- Checkpoints ---")
    latest_checkpoint = flow.get_checkpoint(flow.latest_run)
    pprint(latest_checkpoint)

    print("\n--- Serialization Snapshot ---")
    print("\nflow.to_dict():")
    pprint(flow.to_dict())

    print("\nAll checks passed if no errors above.")


if __name__ == "__main__":
    main()
