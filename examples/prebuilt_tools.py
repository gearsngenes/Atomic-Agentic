# prebuilt_tools.py
"""
Prebuilt Tool Collections (example-space reference, not shipped in src/)
==========================================================================

Single shared module at `examples/` root, referenced by every subfolder
that registers math/console tools (`ReAct_Examples/`, `PlanAct_Examples/`,
`ScriptAct_Examples/`, `Tool_Examples/`) -- one source of truth instead of
a per-folder copy. `PARSER_TOOLS`/`COLLECTION_TOOLS` were dropped outright
(confirmed unused anywhere in the repo, not just this package) rather than
carried forward unused.

Import convention (deliberately NOT the zero-shim `shared_engine.py`
pattern -- this module lives one directory higher, at `examples/` itself,
not beside the importing script): each consumer adds `examples/` to
`sys.path` before importing, e.g.::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from prebuilt_tools import BASIC_MATH_TOOLS

Plain functions, not `Tool`-wrapped: `register_tool`/`register_tools`
already normalize a bare callable via `toolify(name=fn.__name__,
description=description or fn.__doc__, namespace=self.name)` -- so a
docstring alone supplies the description, and no `Tool(...)` construction
is needed here. The one behavior difference from a pre-wrapped `Tool`:
namespace comes from the *registering agent's own name*, not a per-category
label like `"Basic_Math"`/`"Trig"` -- fine for these examples since no two
functions below share a bare name, so there's no collision risk either way.
`Tool_Examples/a2a_sdk_atomic_host_server.py` is the one consumer that
needs real `Tool` instances (publishes skills directly via
`A2AtomicExecutor`, which requires `AtomicInvokable`, not a bare callable)
-- it calls `toolify(fn, namespace=...)` itself at the call site rather
than this module pre-wrapping everything for that one special case.

Example
-------
>>> from prebuilt_tools import BASIC_MATH_TOOLS, CONSOLE_TOOLS
>>> agent.register_tools(BASIC_MATH_TOOLS)
>>> agent.register_tools(CONSOLE_TOOLS)
"""

from __future__ import annotations

from typing import Any, Callable, List, Sequence
import logging
import math

__all__ = ["BASIC_MATH_TOOLS", "EXPONENT_TOOLS", "TRIG_TOOLS", "STAT_TOOLS", "CONSOLE_TOOLS"]

# ────────────────────────── Basic Math Tools ──────────────────────────

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

BASIC_MATH_TOOLS: List[Callable[..., Any]] = [add, subtract, multiply, divide]

# ────────────────────────── Exponentiation and Roots ──────────────────────────

def power(a: float, b: float) -> float:
    """Return a raised to the power of b."""
    return a**b
def sqrt(x: float) -> float:
    """Calls math.sqrt(x)"""
    return math.sqrt(x)
def log(x: float) -> float:
    """Calls math.log(x)."""
    return math.log(x)

EXPONENT_TOOLS: List[Callable[..., Any]] = [power, log, sqrt]

# ──────────────────────────────── Statistics ────────────────────────────────

def mean(nums: Sequence[float]) -> float:
    """Return the arithmetic mean of a sequence of numbers."""
    return (sum(nums) / len(nums)) if nums else 0.0
def max_value(nums: Sequence[float]) -> float:
    """Return the maximum value in a sequence of numbers."""
    return max(nums)
def min_value(nums: Sequence[float]) -> float:
    """Return the minimum value in a sequence of numbers."""
    return min(nums)

STAT_TOOLS: List[Callable[..., Any]] = [mean, max_value, min_value]

# ───────────────────────────── Trigonometry ─────────────────────────────

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

TRIG_TOOLS: List[Callable[..., Any]] = [
    sin, cos, tan, cot, asin, acos, atan, acot,
    sinh, cosh, tanh, coth, asinh, acosh, atanh, acoth,
]

# ───────────────────────────── Console Tools ─────────────────────────────

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

CONSOLE_TOOLS: List[Callable[..., Any]] = [
    print_tool, user_input, basic_config, log_message,
    log_info, log_warning, log_error, log_critical, log_debug,
]
