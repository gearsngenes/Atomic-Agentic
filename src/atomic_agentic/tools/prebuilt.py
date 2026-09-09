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
    """Return the sum of two numbers a and b."""
    return a + b
def subtract(a: float, b: float) -> float:
    """Return the difference of two numbers a and b."""
    return a - b
def multiply(a: float, b: float) -> float:
    """Return the product of two numbers a and b."""
    return a * b
def divide(a: float, b: float) -> float:
    """Return the quotient of two numbers a and b (inf if b == 0)."""
    return a / b if b != 0 else float("inf")

BASIC_MATH_TOOLS: List[Tool] = [
    Tool(function=add, name="add", namespace="Basic_Math"),
    Tool(function=subtract, name="subtract", namespace="Basic_Math"),
    Tool(function=multiply, name="multiply", namespace="Basic_Math"),
    Tool(function=divide, name="divide", namespace="Basic_Math"),
]

# Exponentiation and Roots
def power(a: float, b: float) -> float:
    """Return a raised to the power of b."""
    return a**b
def sqrt(x: float) -> float:
    """Calls math.sqrt(x)"""
    return math.sqrt(x)
def log(x: float) -> float:
    """Calls math.log(x)."""
    return math.log(x)

EXPONENT_TOOLS: List[Tool] = [
    Tool(function=power, name="power", namespace="Exponents"),
    Tool(function=log, name="log", namespace="Exponents"),
    Tool(function=sqrt, name="sqrt", namespace="Exponents"),
]

# Statistics
def mean(nums: Sequence[float]) -> float:
    """Return the arithmetic mean of a sequence of numbers."""
    return (sum(nums) / len(nums)) if nums else 0.0
def max_value(nums: Sequence[float]) -> float:
    """Return the maximum value in a sequence of numbers."""
    return max(nums)
def min_value(nums: Sequence[float]) -> float:
    """Return the minimum value in a sequence of numbers."""
    return min(nums)

STAT_TOOLS: List[Tool] = [
    Tool(function=mean, namespace="Stats"),
    Tool(function=max_value, namespace="Stats"),
    Tool(function=min_value, namespace="Stats"),
]

# Trigonometry
def sin(x: float) -> float:
    """Return the sine of x (x in radians)."""
    return math.sin(x)
def cos(x: float) -> float:
    """Return the cosine of x (x in radians)."""
    return math.cos(x)
def tan(x: float) -> float:
    """Return the tangent of x (x in radians)."""
    return math.tan(x)
def cot(x: float) -> float:
    """Return the cotangent of x (x in radians; inf at tan(x)=0)."""
    t = math.tan(x)
    return (1.0 / t) if t != 0 else float("inf")
def asin(x: float) -> float:
    """Return the arcsine of x (x in [-1, 1])."""
    return math.asin(x)
def acos(x: float) -> float:
    """Return the arccosine of x (x in [-1, 1])."""
    return math.acos(x)
def atan(x: float) -> float:
    """Return the arctangent of x (x in [-inf, inf])."""
    return math.atan(x)
def acot(x: float) -> float:
    """Return the arccotangent of x (x in [-inf, inf]; inf at tan(x)=0)."""
    t = math.tan(x)
    return (1.0 / t) if t != 0 else float("inf")
def sinh(x: float) -> float:
    """Return the hyperbolic sine of x."""
    return math.sinh(x)
def cosh(x: float) -> float:
    """Return the hyperbolic cosine of x."""
    return math.cosh(x)
def tanh(x: float) -> float:
    """Return the hyperbolic tangent of x."""
    return math.tanh(x)
def coth(x: float) -> float:
    """Return the hyperbolic cotangent of x (inf at tanh(x)=0)."""
    t = math.tanh(x)
    return (1.0 / t) if t != 0 else float("inf")
def asinh(x: float) -> float:
    """Return the inverse hyperbolic sine of x."""
    return math.asinh(x)
def acosh(x: float) -> float:
    """Return the inverse hyperbolic cosine of x."""
    if x < 1:
        raise ValueError("acosh: x must be >= 1")
    return math.acosh(x)
def atanh(x: float) -> float:
    """Return the inverse hyperbolic tangent of x."""
    return math.atanh(x)
def acoth(x: float) -> float:
    """Return the inverse hyperbolic cotangent of x (inf at tanh(x)=0)."""
    t = math.tanh(x)
    return (1.0 / t) if t != 0 else float("inf")

TRIG_TOOLS: List[Tool] = [
    Tool(function=sin, namespace="Trig"),
    Tool(function=cos, namespace="Trig"),
    Tool(function=tan, namespace="Trig"),
    Tool(function=cot, namespace="Trig"),
    Tool(function=asin, namespace="Trig"),
    Tool(function=acos, namespace="Trig"),
    Tool(function=atan, namespace="Trig"),
    Tool(function=acot, namespace="Trig"),
    Tool(function=sinh, namespace="Trig"),
    Tool(function=cosh, namespace="Trig"),
    Tool(function=tanh, namespace="Trig"),
    Tool(function=coth, namespace="Trig"),
    Tool(function=asinh, namespace="Trig"),
    Tool(function=acosh, namespace="Trig"),
    Tool(function=atanh, namespace="Trig"),
    Tool(function=acoth, namespace="Trig"),
]

# ───────────────────────── Console Tools ─────────────────────────

def print_tool(*objects) -> None:
    """Calls print(*objects)."""
    print(*objects)

def user_input(prompt: str) -> str:
    """Prompt user for input text and return the entered string."""
    return input(prompt)

def basic_config(level: str = "INFO") -> None:
    """Configure root logging (level: DEBUG|INFO|WARNING|ERROR|CRITICAL)."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))

def log_message(message: str, level: str) -> None:
    """Log a message at the specified level."""
    logging.log(getattr(logging, level.upper(), logging.INFO), message)

def log_info(message: str) -> None:
    """Log a message at INFO level."""
    logging.info(message)

def log_warning(message: str) -> None:
    """Log a message at WARNING level."""
    logging.warning(message)

def log_error(message: str) -> None:
    """Log a message at ERROR level."""
    logging.error(message)

def log_critical(message: str) -> None:
    """Log a message at CRITICAL level."""
    logging.critical(message)

def log_debug(message: str) -> None:
    """Log a message at DEBUG level."""
    logging.debug(message)

def log_trace(message: str) -> None:
    """Log a message at TRACE level."""
    logging.log(logging.TRACE, message)

CONSOLE_TOOLS: List[Tool] = [
    Tool(function=print_tool, namespace="Console"),
    Tool(function=user_input, namespace="Console"),
    Tool(function=basic_config, namespace="Console"),
    Tool(function=log_message, namespace="Console"),
    Tool(function=log_info, namespace="Console"),
    Tool(function=log_warning, namespace="Console"),
    Tool(function=log_error, namespace="Console"),
    Tool(function=log_critical, namespace="Console"),
    Tool(function=log_debug, namespace="Console"),
    Tool(function=log_trace, namespace="Console"),
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
def is_in(x: Any, y: Any) -> bool:
    """returns boolean value of 'x in y'"""
    return x in y
def if_else_select(condition: bool, true_val: Any, false_val: Any) -> Any:
    return true_val if condition else false_val

# ─────────────────────────  Collection Tools  ─────────────────────────
def has_key(d: dict, key: Any) -> bool:
    """returns boolean value of 'key in d'"""
    return key in d
def get_from_dict(d: dict, key: Any, default: Optional[Any] = None) -> Any:
    """calls d.get(key, default)"""
    return d.get(key, default)
def get_keys(d: dict) -> List[Any]:
    """Returns list(d.keys())"""
    return list(d.keys())
def get_from_seq(seq: Sequence[Any], index: int) -> Any:
    """Returns seq[index]"""
    return seq[index]
def len_of(seq: Sequence[Any]) -> int:
    """Returns len(seq)"""
    return len(seq)
def append_to_list(lst: List[Any], item: Any) -> None:
    """calls lst.append(item)"""
    lst.append(item)
def update_dict(source: dict, updates: dict) -> None:
    """Calls source.update(updates)"""
    source.update(updates)
def get_range(start: int, stop: int, step: int = 1) -> List[int]:
    """Return a list of integers from start to stop-1."""
    return list(range(start, stop, step))
def sort_list(lst: List[Any], reverse: bool = False) -> List[Any]:
    """Return sorted(lst, reverse=reverse)."""
    return sorted(lst, reverse=reverse)
def slice_seq(seq: Sequence[Any], start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None) -> Sequence[Any]:
    """Return seq[start:stop:step]."""
    return seq[start:stop:step]

COLLECTION_TOOLS: List[Tool] = [
    Tool(function=has_key, name="has_key", namespace="Collection"),
    Tool(function=get_from_dict, name="get_from_dict", namespace="Collection"),
    Tool(function=get_keys, name="get_keys", namespace="Collection"),
    Tool(function=get_from_seq, name="get_from_seq", namespace="Collection"),
    Tool(function=len_of, name="len_of", namespace="Collection"),    
    Tool(function=append_to_list, name="append_to_list", namespace="Collection"),
    Tool(function=update_dict, name="update_dict", namespace="Collection"),
    Tool(function=get_range, name="get_range", namespace="Collection"),
    Tool(function=sort_list, name="sort_list", namespace="Collection"),
    Tool(function=slice_seq, name="slice_seq", namespace="Collection"),
]