# modules/Plugins.py
"""
Plugins (Tool Lists)
====================

This module exposes *prebuilt lists of Tools* that you can register on a
ToolAgent/PlannerAgent via `batch_register(...)`.

Example
-------
>>> from modules.Plugins import MATH_TOOLS, CONSOLE_TOOLS, PARSER_TOOLS
>>> planner.batch_register(MATH_TOOLS)
>>> planner.batch_register(CONSOLE_TOOLS)
>>> planner.batch_register(PARSER_TOOLS)

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

from .Tools import Tool

__all__ = ["MATH_TOOLS", "CONSOLE_TOOLS", "PARSER_TOOLS"]

# ────────────────────────── Math Tools ──────────────────────────

def add(a: float, b: float) -> float: return a + b
def subtract(a: float, b: float) -> float: return a - b
def multiply(a: float, b: float) -> float: return a * b
def divide(a: float, b: float) -> float:
    return a / b if b != 0 else float("inf")
def power(a: float, b: float) -> float: return a ** b
def sqrt(x: float) -> float:
    if x < 0:
        raise ValueError("sqrt: x must be non-negative")
    return math.sqrt(x)
def mean(nums: Sequence[float]) -> float:
    return (sum(nums) / len(nums)) if nums else 0.0
def max_value(nums: Sequence[float]) -> float: return max(nums)
def min_value(nums: Sequence[float]) -> float: return min(nums)
def sin(x: float) -> float: return math.sin(x)
def cos(x: float) -> float: return math.cos(x)
def tan(x: float) -> float: return math.tan(x)
def cot(x: float) -> float:
    t = math.tan(x)
    return (1.0 / t) if t != 0 else float("inf")

MATH_TOOLS: List[Tool] = [
    Tool(func=add,        name="add",        description="Return a + b."),
    Tool(func=subtract,   name="subtract",   description="Return a - b."),
    Tool(func=multiply,   name="multiply",   description="Return a * b."),
    Tool(func=divide,     name="divide",     description="Return a / b (inf if b == 0)."),
    Tool(func=power,      name="power",      description="Return a ** b."),
    Tool(func=sqrt,       name="sqrt",       description="Return sqrt(x); x must be >= 0."),
    Tool(func=mean,       name="mean",       description="Return arithmetic mean of a list of numbers."),
    Tool(func=max_value,  name="max_value",  description="Return the maximum of a list of numbers."),
    Tool(func=min_value,  name="min_value",  description="Return the minimum of a list of numbers."),
    Tool(func=sin,        name="sin",        description="Return sin(x) (x in radians)."),
    Tool(func=cos,        name="cos",        description="Return cos(x) (x in radians)."),
    Tool(func=tan,        name="tan",        description="Return tan(x) (x in radians)."),
    Tool(func=cot,        name="cot",        description="Return cot(x) (x in radians; inf at tan(x)=0)."),
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
    Tool(func=print_value,  name="print",        description="Print any value to the console."),
    Tool(func=user_input,   name="user_input",   description="Prompt user for input and return the entered string."),
    Tool(func=basic_config, name="basic_config", description="Configure root logging (level: DEBUG|INFO|WARNING|ERROR|CRITICAL)."),
    Tool(func=log_message,  name="log",          description="Log a message at the specified level."),
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
    """
    Try to extract a JSON object/array substring from text.
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
    Tool(func=json_loads,       name="json_loads",          description="Parse JSON string to Python value."),
    Tool(func=to_str,           name="to_str",              description="Convert any value to string."),
    Tool(func=split_string,     name="split",               description="Split string by separator into list of strings."),
    Tool(func=safe_eval,        name="safe_eval",           description="Evaluate a simple Python literal safely (no builtins)."),
    Tool(func=extract_json_string, name="extract_json_string", description="Extract first JSON object/array substring from text."),
    Tool(func=join_strings,     name="join",                description="Join list of strings with a separator."),
]
