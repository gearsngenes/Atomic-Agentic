# modules/Plugins.py
"""
Plugins (Tool Lists)
====================

This module exposes *prebuilt lists of Tools* that you can register on a ToolAgent
(e.g., PlanActAgent / IterActAgent) via `batch_register(...)`.

Example
-------
>>> from atomic_agentic.Plugins import MATH_TOOLS, CONSOLE_TOOLS, PARSER_TOOLS
>>> agent.batch_register(MATH_TOOLS)
>>> agent.batch_register(CONSOLE_TOOLS)
>>> agent.batch_register(PARSER_TOOLS)

Design
------
- No custom Plugin classes. Each “plugin” is just a `List[Tool]`.
- Functions are defined with clear, named parameters so Tool schemas are explicit.
- Tool names are unique within a list to avoid registration collisions.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence
import json
import logging
import math
import re

# Import the Tool primitive directly (avoids pulling optional MCP deps from .Tools)
from .Primitives import Tool

__all__ = ["MATH_TOOLS", "CONSOLE_TOOLS", "PARSER_TOOLS"]

# ────────────────────────── Math Tools ──────────────────────────

def add(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> float:
    return a / b if b != 0 else float("inf")

def power(a: float, b: float) -> float:
    return a**b

def sqrt(x: float) -> float:
    if x < 0:
        raise ValueError("sqrt: x must be non-negative")
    return math.sqrt(x)

def mean(nums: Sequence[float]) -> float:
    return (sum(nums) / len(nums)) if nums else 0.0

def max_value(nums: Sequence[float]) -> float:
    return max(nums)

def min_value(nums: Sequence[float]) -> float:
    return min(nums)

def sin(x: float) -> float:
    return math.sin(x)

def cos(x: float) -> float:
    return math.cos(x)

def tan(x: float) -> float:
    return math.tan(x)

def cot(x: float) -> float:
    t = math.tan(x)
    return (1.0 / t) if t != 0 else float("inf")


MATH_TOOLS: List[Tool] = [
    Tool(function=add, name="add", namespace="Math", description="Return a + b."),
    Tool(function=subtract, name="subtract", namespace="Math", description="Return a - b."),
    Tool(function=multiply, name="multiply", namespace="Math", description="Return a * b."),
    Tool(function=divide, name="divide", namespace="Math", description="Return a / b (inf if b == 0)."),
    Tool(function=power, name="power", namespace="Math", description="Return a ** b."),
    Tool(function=sqrt, name="sqrt", namespace="Math", description="Return sqrt(x); x must be >= 0."),
    Tool(function=mean, name="mean", namespace="Math", description="Return arithmetic mean of a list of numbers."),
    Tool(function=max_value, name="max_value", namespace="Math", description="Return the maximum of a list of numbers."),
    Tool(function=min_value, name="min_value", namespace="Math", description="Return the minimum of a list of numbers."),
    Tool(function=sin, name="sin", namespace="Math", description="Return sin(x) (x in radians)."),
    Tool(function=cos, name="cos", namespace="Math", description="Return cos(x) (x in radians)."),
    Tool(function=tan, name="tan", namespace="Math", description="Return tan(x) (x in radians)."),
    Tool(function=cot, name="cot", namespace="Math", description="Return cot(x) (x in radians; inf at tan(x)=0)."),
]

# ───────────────────────── Console Tools ─────────────────────────

def print_value(value: Any) -> None:
    print(value)

def user_input(prompt: str) -> str:
    return input(prompt)

def basic_config(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))

def log_message(message: str, level: str = "INFO") -> None:
    logging.log(getattr(logging, level.upper(), logging.INFO), message)


CONSOLE_TOOLS: List[Tool] = [
    Tool(function=print_value, name="print", namespace="Console", description="Print any value to the console."),
    Tool(function=user_input, name="user_input", namespace="Console", description="Prompt user for input and return the entered string."),
    Tool(function=basic_config, name="basic_config", namespace="Console", description="Configure root logging (level: DEBUG|INFO|WARNING|ERROR|CRITICAL)."),
    Tool(function=log_message, name="log", namespace="Console", description="Log a message at the specified level."),
]

# ───────────────────────── Parser Tools ─────────────────────────

def json_loads(s: str) -> Any:
    return json.loads(s)

def to_str(x: Any) -> str:
    return str(x)

def split_string(s: str, sep: Optional[str] = None) -> List[str]:
    return s.split(sep)

def safe_eval(s: str) -> Any:
    # Evaluate simple Python literal expressions with no builtins.
    return eval(s, {"__builtins__": None}, {})

def extract_json_string(s: str) -> Optional[str]:
    """Try to extract a JSON object/array substring from text.

    Returns the first {...} or [...] block if found; otherwise None.
    """
    m_obj = re.search(r"({.*})", s, re.DOTALL)
    if m_obj:
        return m_obj.group(1)
    m_arr = re.search(r"(\[.*\])", s, re.DOTALL)
    if m_arr:
        return m_arr.group(1)
    return None

def join_strings(lst: Sequence[str], sep: str = "") -> str:
    return sep.join(lst)


PARSER_TOOLS: List[Tool] = [
    Tool(function=json_loads, name="json_loads", namespace="Parser", description="Parse JSON string to Python value."),
    Tool(function=to_str, name="to_str", namespace="Parser", description="Convert any value to string."),
    Tool(function=split_string, name="split", namespace="Parser", description="Split string by separator into list of strings."),
    Tool(function=safe_eval, name="safe_eval", namespace="Parser", description="Evaluate a simple Python literal safely (no builtins)."),
    Tool(function=extract_json_string, name="extract_json_string", namespace="Parser", description="Extract first JSON object/array substring from text."),
    Tool(function=join_strings, name="join", namespace="Parser", description="Join list of strings with a separator."),
]
