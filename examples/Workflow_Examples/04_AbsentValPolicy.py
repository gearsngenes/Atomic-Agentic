from __future__ import annotations

from typing import Any, Mapping

from dotenv import load_dotenv

from atomic_agentic.workflows import AbsentValPolicy
from atomic_agentic.tools import Tool
from atomic_agentic.workflows.workflows import ToolFlow, BundlingPolicy, MappingPolicy


# -----------------------------------------------------------------------------
# Env (safe even if you don't need any keys for this script)
# -----------------------------------------------------------------------------
load_dotenv()


# -----------------------------------------------------------------------------
# Shared output schema (expects 3 fields)
# -----------------------------------------------------------------------------
OUTPUT_SCHEMA = ["a", "b", "c"]
INPUTS: Mapping[str, Any] = {"x": 7}


# -----------------------------------------------------------------------------
# Tools that intentionally under-provide outputs
# -----------------------------------------------------------------------------
def tool_partial_mapping(*, x: int) -> dict[str, Any]:
    # Missing "b" and "c"
    return {"a": x}


def tool_short_sequence(*, x: int) -> tuple[int, int]:
    # Only 2 values for a 3-field schema
    return (x, x + 1)


def tool_scalar(*, x: int) -> int:
    # Scalar for a 3-field schema
    return x


TOOLS: list[tuple[str, Tool]] = [
    ("mapping missing keys", Tool(tool_partial_mapping, name="tool_partial_mapping")),
    ("short sequence", Tool(tool_short_sequence, name="tool_short_sequence")),
    ("scalar", Tool(tool_scalar, name="tool_scalar")),
]


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_once(*, label: str, flow: ToolFlow, inputs: Mapping[str, Any]) -> None:
    print(f"\n--- {label} ---")
    print("inputs:", dict(inputs))
    try:
        result = flow.invoke(dict(inputs))
        print("result:", result)
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")


def exercise_tool(tool_label: str, tool: Tool) -> None:
    print("\n" + "=" * 78)
    print(f"TOOLFLOW CASE: {tool_label}")
    print(f"Tool: {tool.full_name}")
    print(f"Expected output_schema: {OUTPUT_SCHEMA}")
    print("=" * 78)

    for policy in (AbsentValPolicy.RAISE, AbsentValPolicy.FILL, AbsentValPolicy.DROP):
        flow = ToolFlow(
            tool=tool,
            output_schema=OUTPUT_SCHEMA,
            bundling_policy=BundlingPolicy.UNBUNDLE,
            mapping_policy=MappingPolicy.STRICT,
            absent_val_policy=policy,
        )

        # Only relevant for AbsentValPolicy.FILL, but harmless for others.
        flow.default_absent_val = "<ABSENT>"

        run_once(
            label=f"absent_val_policy={policy}",
            flow=flow,
            inputs=INPUTS,
        )


def main() -> None:
    print("=== AbsentValPolicy smoke test (ToolFlow only) ===")
    print("This intentionally creates missing outputs vs output_schema=['a','b','c'].\n")
    for label, tool in TOOLS:
        exercise_tool(label, tool)

    print("\n=== Done ===")
    print("RAISE -> throws if any field is still NO_VAL after packaging/validation")
    print("FILL  -> missing fields replaced with default_absent_val")
    print("DROP  -> missing fields removed (order preserved)")


if __name__ == "__main__":
    main()
