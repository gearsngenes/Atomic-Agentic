"""
Beginner-friendly demo: using Tool to wrap simple Python functions.
"""
from typing import Annotated
from atomic_agentic.tools import Tool
from atomic_agentic.exceptions import ToolInvocationError
import logging

logging.basicConfig(level = logging.INFO)

# --- 1) Define plain Python functions with Annotated parameter descriptions ---

def add(
    a: Annotated[int, "First integer to add"],
    b: Annotated[int, "Second integer to add"] = 0,
) -> int:
    """Add two integers."""
    return a + b

def greet(
    name: Annotated[str, "Name of the person to greet"],
    *,
    excited: Annotated[bool, "If True, appends '!!!' instead of '.'"] = False,
) -> str:
    """Greet a person by name."""
    base = f"Hello, {name}"
    return base + "!!!" if excited else base + "."

def summarize(
    text: Annotated[str, "Text to summarize"],
    max_chars: Annotated[int, "Maximum character count for the output"] = 60,
) -> str:
    """Return a summary clamped to max_chars characters."""
    s = text.strip()
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


# --- 2) Wrap as Tools (descriptions are now concise — param docs live on the params) ---

t_add = Tool(
    add,
    name="add",
    description="Add two integers.",
    namespace="local",
)

t_greet = Tool(
    greet,
    name="greet",
    description="Greet a person by name.",
    namespace="local",
)

t_sum = Tool(
    summarize,
    name="summarize",
    description="Summarize text clamped to a character limit.",
    namespace="local",
)


# --- 3) Utility helpers for printing results ---

def show_plan(tool: Tool) -> None:
    meta = tool.to_dict()
    print(f"\n-- {tool.name} call plan --")
    print("signature:", tool.signature)
    print("parameters:")
    for param in tool.parameters:
        default_str = "(no default)" if param.default.__class__.__name__ == "NO_VAL" else f"default={param.default}"
        desc_str = f"  # {param.description}" if param.description else ""
        print(f"  {param.name}: {param.kind}, type={param.type}, {default_str}{desc_str}")

def run_case(label: str, tool: Tool, inputs: dict) -> None:
    print(f"\n=== {label} ===")
    print("inputs:", inputs)
    try:
        result = tool.invoke(inputs)
        print("OK:", result.result)
    except (ToolInvocationError, TypeError, ValueError) as e:
        print("ERR:", e)


# --- 4) Beginner-friendly usage examples ---

if __name__ == "__main__":
    # Inspect the binding plans (nice for learning how names map)
    show_plan(t_add)
    show_plan(t_greet)
    show_plan(t_sum)

    # Happy paths
    run_case("add: both args", t_add, {"a": 2, "b": 3})           # -> 5
    run_case("add: with default b", t_add, {"a": 7})               # -> 7

    run_case("greet: basic", t_greet, {"name": "Ava"})             # -> "Hello, Ava."
    run_case("greet: excited", t_greet, {"name": "Ava", "excited": True})  # -> "Hello, Ava!!!"

    long_text = "Atomic-Agentic helps you orchestrate agents, tools, and workflows cleanly."
    run_case("summarize: clamp", t_sum, {"text": long_text, "max_chars": 40})

    # Common mistakes (to see helpful errors)
    run_case("greet: missing required 'name'", t_greet, {"excited": True})
    run_case("add: unknown key silently filtered", t_add, {"a": 1, "b": 2, "extra": 99})  # -> 3
