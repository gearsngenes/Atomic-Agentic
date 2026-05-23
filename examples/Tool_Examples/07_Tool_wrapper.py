"""
Beginner-friendly demo: using Tool to wrap plain functions and AtomicInvokable objects.

This demonstrates the v1.4-style Tool behavior:

1. Tool can wrap a plain Python callable.
2. Tool can also wrap another AtomicInvokable, such as an existing Tool.
3. Tool.wraps_invokable tells you whether the underlying function is an AtomicInvokable.
4. The wrapped-invokable Tool preserves the wrapped invokable's declared schema.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from atomic_agentic.tools import Tool
from atomic_agentic.core.Exceptions import ToolInvocationError

logging.basicConfig(level=logging.INFO)


# --- 1) Define plain Python functions ---

def add(a: int, b: int = 0) -> int:
    """Add two integers."""
    return a + b


async def async_emphasize(text: str, *, times: int = 1) -> str:
    """Repeat text with emphasis asynchronously."""
    await asyncio.sleep(0)
    return " ".join([text.upper() + "!"] * times)


# --- 2) Wrap plain functions as normal Tools ---

base_add_tool = Tool(
    add,
    name="add",
    namespace="math",
    description="Add two integers. Args: a:int, b:int=0. Returns: int.",
)

base_emphasize_tool = Tool(
    async_emphasize,
    name="emphasize",
    namespace="text",
    description="Uppercase and repeat text asynchronously.",
)


# --- 3) Wrap existing Tools as invokable-backed Tools ---

wrapped_add_tool = Tool(
    base_add_tool,
    name="wrapped_add",
    namespace="wrapped",
    description="Tool wrapping another Tool through the AtomicInvokable callable path.",
)

wrapped_emphasize_tool = Tool(
    base_emphasize_tool,
    name="wrapped_emphasize",
    namespace="wrapped",
    description="Tool wrapping an async Tool through the AtomicInvokable async path.",
)


# --- 4) Utility helpers for printing results ---

def show_plan(tool: Tool) -> None:
    print(f"\n-- {tool.full_name} call plan --")
    print("wraps_invokable:", tool.wraps_invokable)
    print("signature:", tool.signature)
    print("parameters:")

    for param in tool.parameters:
        default_str = (
            "(no default)"
            if param.default.__class__.__name__ == "NO_VAL"
            else f"default={param.default!r}"
        )
        print(f"  {param.name}: {param.kind}, type={param.type}, {default_str}")

    print("metadata:")
    metadata = tool.to_dict()
    print("  namespace:", metadata.get("namespace"))
    print("  wraps_invokable:", metadata.get("wraps_invokable"))
    print("  module:", metadata.get("module"))
    print("  qualname:", metadata.get("qualname"))


def run_case(label: str, tool: Tool, inputs: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print("inputs:", inputs)

    try:
        result = tool.invoke(inputs)
        print("OK:", result)
    except (ToolInvocationError, TypeError, ValueError) as exc:
        print("ERR:", exc)


async def run_async_case(label: str, tool: Tool, inputs: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    print("inputs:", inputs)

    try:
        result = await tool.async_invoke(inputs)
        print("OK:", result)
    except (ToolInvocationError, TypeError, ValueError) as exc:
        print("ERR:", exc)


# --- 5) Demo cases ---

if __name__ == "__main__":
    # Inspect plain callable-backed tools.
    show_plan(base_add_tool)
    show_plan(base_emphasize_tool)

    # Inspect invokable-backed tools.
    show_plan(wrapped_add_tool)
    show_plan(wrapped_emphasize_tool)

    # Plain Tool behavior.
    run_case(
        "plain Tool: add",
        base_add_tool,
        {"a": 2, "b": 3},
    )

    # Tool wrapping another Tool.
    run_case(
        "wrapped Tool: add",
        wrapped_add_tool,
        {"a": 10, "b": 5},
    )

    # Defaults still flow through the wrapped schema.
    run_case(
        "wrapped Tool: add with default b",
        wrapped_add_tool,
        {"a": 7},
    )

    # Async callable-backed Tool.
    asyncio.run(
        run_async_case(
            "plain async Tool: emphasize",
            base_emphasize_tool,
            {"text": "hello", "times": 2},
        )
    )

    # Invokable-backed async Tool.
    # This should use wrapped_emphasize_tool.async_execute() -> base_emphasize_tool.async_call(...).
    asyncio.run(
        run_async_case(
            "wrapped async Tool: emphasize",
            wrapped_emphasize_tool,
            {"text": "atomic", "times": 3},
        )
    )

    # Common mistake: unknown key with strict filtering disabled.
    strict_wrapped_add = Tool(
        base_add_tool,
        name="strict_wrapped_add",
        namespace="wrapped",
        description="Strict wrapper around add.",
        filter_extraneous_inputs=False,
    )

    run_case(
        "wrapped Tool: unknown key",
        strict_wrapped_add,
        {"a": 1, "b": 2, "extra": 99},
    )