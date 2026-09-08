# tools/prebuilt.py
"""
Prebuilt Tool Collections
=========================

This module exposes *prebuilt lists of Tools* that you can register on a ToolAgent
(e.g., PlanActAgent / ReActAgent) via `batch_register(...)`.

Example
-------
>>> from atomic_agentic.tools.prebuilt import MATH_TOOLS, CONSOLE_TOOLS, PARSER_TOOLS
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
import ast
import json
import logging
import math
import re

# Import the Tool primitive directly (avoids pulling optional MCP deps from .Tools)
from .base import Tool

__all__ = ["BASIC_MATH_TOOLS",
           "EXPONENT_TOOLS",
           "TRIG_TOOLS",
           "STAT_TOOLS",
           "CONSOLE_TOOLS",
           "PARSER_TOOLS"]

# ────────────────────────── Basic Math Tools ──────────────────────────

# Basic Arithmetic
def add(a: float, b: float) -> float:
    return a + b
def subtract(a: float, b: float) -> float:
    return a - b
def multiply(a: float, b: float) -> float:
    return a * b
def divide(a: float, b: float) -> float:
    return a / b if b != 0 else float("inf")

BASIC_MATH_TOOLS: List[Tool] = [
    Tool(function=add, name="add", namespace="Basic_Math", description="Return sum of two numbers a + b."),
    Tool(function=subtract, name="subtract", namespace="Basic_Math", description="Return difference of two numbers a - b."),
    Tool(function=multiply, name="multiply", namespace="Basic_Math", description="Return product of two numbers a * b."),
    Tool(function=divide, name="divide", namespace="Basic_Math", description="Return quotient of two numbers a / b (inf if b == 0)."),
]

# Exponentiation and Roots
def power(a: float, b: float) -> float:
    return a**b
def sqrt(x: float) -> float:
    if x < 0:
        raise ValueError("sqrt: x must be non-negative")
    return math.sqrt(x)
def log(x: float) -> float:
    return math.log(x)

EXPONENT_TOOLS: List[Tool] = [
    Tool(function=power, name="power", namespace="Exponents", description="Return power of two numbers a ** b."),
    Tool(function=log, name="log", namespace="Exponents", description="Return natural logarithm of a number x; x must be > 0."),
    Tool(function=sqrt, name="sqrt", namespace="Exponents", description="Return square root of a number x; x must be >= 0."),
]

# Statistics
def mean(nums: Sequence[float]) -> float:
    return (sum(nums) / len(nums)) if nums else 0.0
def max_value(nums: Sequence[float]) -> float:
    return max(nums)
def min_value(nums: Sequence[float]) -> float:
    return min(nums)

STAT_TOOLS: List[Tool] = [
    Tool(function=mean, name="mean", namespace="Stats", description="Return arithmetic mean of a sequence of numbers."),
    Tool(function=max_value, name="max_value", namespace="Stats", description="Return the maximum of a sequence of numbers."),
    Tool(function=min_value, name="min_value", namespace="Stats", description="Return the minimum of a sequence of numbers."),
]

# Trigonometry
def sin(x: float) -> float:
    return math.sin(x)
def cos(x: float) -> float:
    return math.cos(x)
def tan(x: float) -> float:
    return math.tan(x)
def cot(x: float) -> float:
    t = math.tan(x)
    return (1.0 / t) if t != 0 else float("inf")
def asin(x: float) -> float:
    return math.asin(x)
def acos(x: float) -> float:
    return math.acos(x)
def atan(x: float) -> float:
    return math.atan(x)
def acot(x: float) -> float:
    t = math.tan(x)
    return (1.0 / t) if t != 0 else float("inf")
def sinh(x: float) -> float:
    return math.sinh(x)
def cosh(x: float) -> float:
    return math.cosh(x)
def tanh(x: float) -> float:
    return math.tanh(x)
def coth(x: float) -> float:
    t = math.tanh(x)
    return (1.0 / t) if t != 0 else float("inf")
def asinh(x: float) -> float:
    return math.asinh(x)
def acosh(x: float) -> float:
    if x < 1:
        raise ValueError("acosh: x must be >= 1")
    return math.acosh(x)
def atanh(x: float) -> float:
    return math.atanh(x)
def acoth(x: float) -> float:
    t = math.tanh(x)
    return (1.0 / t) if t != 0 else float("inf")

TRIG_TOOLS: List[Tool] = [
    Tool(function=sin, name="sin", namespace="Trig", description="Return sin(x) (x in radians)."),
    Tool(function=cos, name="cos", namespace="Trig", description="Return cos(x) (x in radians)."),
    Tool(function=tan, name="tan", namespace="Trig", description="Return tan(x) (x in radians)."),
    Tool(function=cot, name="cot", namespace="Trig", description="Return cot(x) (x in radians; inf at tan(x)=0)."),
    Tool(function=asin, name="asin", namespace="Trig", description="Return arcsin(x) (result in radians)."),
    Tool(function=acos, name="acos", namespace="Trig", description="Return arccos(x) (result in radians)."),
    Tool(function=atan, name="atan", namespace="Trig", description="Return arctan(x) (result in radians)."),
    Tool(function=acot, name="acot", namespace="Trig", description="Return arccot(x) (result in radians; inf at tan(x)=0)."),
    Tool(function=sinh, name="sinh", namespace="Trig", description="Return sinh(x) (hyperbolic sine)."),
    Tool(function=cosh, name="cosh", namespace="Trig", description="Return cosh(x) (hyperbolic cosine)."),
    Tool(function=tanh, name="tanh", namespace="Trig", description="Return tanh(x) (hyperbolic tangent)."),
    Tool(function=coth, name="coth", namespace="Trig", description="Return coth(x) (hyperbolic cotangent; inf at tanh(x)=0)."),
    Tool(function=asinh, name="asinh", namespace="Trig", description="Return arcsinh(x) (inverse hyperbolic sine)."),
    Tool(function=acosh, name="acosh", namespace="Trig", description="Return arccosh(x) (inverse hyperbolic cosine; x must be >= 1)."),
    Tool(function=atanh, name="atanh", namespace="Trig", description="Return arctanh(x) (inverse hyperbolic tangent)."),
    Tool(function=acoth, name="acoth", namespace="Trig", description="Return arccoth(x) (inverse hyperbolic cotangent; inf at tanh(x)=0)."),
]

# ───────────────────────── Console Tools ─────────────────────────

def print_tool(*objects) -> None:
    print(*objects)

def user_input(prompt: str) -> str:
    return input(prompt)

def basic_config(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))

def log_message(message: str, level: str = "INFO") -> None:
    logging.log(getattr(logging, level.upper(), logging.INFO), message)


CONSOLE_TOOLS: List[Tool] = [
    Tool(function=print_tool, name="print", namespace="Console", description="Print any value or collection of values to the console."),
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
    # Parse a literal Python value (str/num/tuple/list/dict/set/bool/None) --
    # no names, calls, or operators, unlike eval() with __builtins__ stripped.
    return ast.literal_eval(s)
def dump_json_string(x: Any) -> str:
    return json.dumps(x)
def join_strings(lst: Sequence[str], sep: str = "") -> str:
    return sep.join(lst)
def regex_match(pattern: str, s: str) -> bool:
    return re.search(pattern, s) is not None
def regex_replace(pattern: str, s: str, repl: str) -> str:
    return re.sub(pattern, repl, s)

PARSER_TOOLS: List[Tool] = [
    Tool(function=json_loads, name="json_loads", namespace="Parser", description="Parse JSON string to Python value."),
    Tool(function=to_str, name="to_str", namespace="Parser", description="Convert any value to string."),
    Tool(function=split_string, name="split", namespace="Parser", description="Split string by separator into list of strings."),
    Tool(function=safe_eval, name="safe_eval", namespace="Parser", description="Evaluate a simple Python literal safely (no builtins)."),
    Tool(function=join_strings, name="join", namespace="Parser", description="Join list of strings with a separator."),
    Tool(function=regex_match, name="regex_match", namespace="Parser", description="Check if a string matches a regex pattern."),
    Tool(function=regex_replace, name="regex_replace", namespace="Parser", description="Replace occurrences of a regex pattern in a string."),
]

# ─────────────────────────  Conditional Tools  ─────────────────────────
def contains_substring(s: str, substr: str) -> bool:
    return substr in s
def is_in(x: Any, lst: Sequence[Any]) -> bool:
    return x in lst
def has_key(d: dict, key: Any) -> bool:
    return key in d
def if_else_select(condition: bool, true_val: Any, false_val: Any) -> Any:
    return true_val if condition else false_val
def get_from_dict(d: dict, key: Any, default: Optional[Any] = None) -> Any:
    return d.get(key, default)
def get_from_seq(seq: Sequence[Any], index: int, default: Optional[Any] = None) -> Any:
    try:
        return seq[index]
    except IndexError:
        return default
def len_of(seq: Sequence[Any]) -> int:
    return len(seq)
