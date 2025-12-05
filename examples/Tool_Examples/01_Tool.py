"""
Beginner-friendly demo: using Tool to wrap simple Python functions.

Adjust the import path to your project layout if needed:
from modules.Tools import Tool, ToolInvocationError
"""
from atomic_agentic.Tools import Tool, ToolInvocationError
import logging

logging.basicConfig(level = logging.INFO)

# --- 1) Define plain Python functions (the Tool wraps these) ---

def add(a: int, b: int = 0) -> int:
    """Add two integers (b defaults to 0)."""
    return a + b

def greet(name: str, *, excited: bool = False) -> str:
    """Keyword-only example: 'excited' must be passed by name."""
    base = f"Hello, {name}"
    return base + "!!!" if excited else base + "."

def summarize(text: str, max_chars: int = 60) -> str:
    """Return a short summary clamped to max_chars."""
    s = text.strip()
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


# --- 2) Wrap them as Tools (description should describe the FUNCTION, not dict plumbing) ---

t_add = Tool(
    add,
    name="add",
    description="Add two integers. Args: a:int (required), b:int=0. Returns: int.",
    source="local",
)

t_greet = Tool(
    greet,
    name="greet",
    description="Greet a person. Args: name:str (required), excited:bool=False (keyword-only). Returns: str.",
    source="local",
)

t_sum = Tool(
    summarize,
    name="summarize",
    description="Summarize text. Args: text:str (required), max_chars:int (required). Returns: str.",
    source="local",
)


# --- 3) Utility helpers for printing results ---

def show_plan(tool: Tool) -> None:
    meta = tool.to_dict()
    print(f"\n-- {tool.name} call plan --")
    print("signature:", meta["signature"])
    print("argument map:", meta["arguments_map"])

def run_case(label: str, tool: Tool, inputs: dict) -> None:
    print(f"\n=== {label} ===")
    print("inputs:", inputs)
    try:
        result = tool.invoke(inputs)
        print("OK:", result)
    except ToolInvocationError as e:
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
    run_case("add: unknown key error", t_add, {"a": 1, "b": 2, "extra": 99})
