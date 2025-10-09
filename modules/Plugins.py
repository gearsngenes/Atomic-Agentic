"""
Plugins
=======

Lightweight, static plugin catalog exposing callable utilities as *tools*.

This module defines a `Plugin` **TypedDict** and three concrete plugin specs:
`MathPlugin`, `ConsolePlugin`, and `ParserPlugin`. Each plugin describes a
name and a `method_map`, where every entry contains:

    {
        "<method_name>": {
            "callable": <python callable>,
            "description": "<one-line purpose and arg semantics>"
        },
        ...
    }

Design notes
------------
- **No runtime side effects** beyond defining these dicts.
- Callables are intentionally implemented as small lambdas for brevity.
- Descriptions are concise and optimized for LLM prompts / tool menus.
- This file only improves documentation and readability. **No functional changes.**
"""

import sys
from pathlib import Path
# Setting the root (kept as-is for compatibility with existing imports)
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Any, TypedDict
import re
import json
import logging

class Plugin(TypedDict):
    """
    Structured specification for a plugin that can be "toolified".

    Fields
    ------
    name : str
        Human-readable namespace for the plugin. Used as the `<source>` segment
        of fully-qualified tool keys, e.g., `plugin.<name>.<method>`.

    method_map : dict[str, dict[str, Any]]
        Mapping from method name → metadata dict with keys:
          - "callable": a Python callable implementing the method.
          - "description": short, user-facing help text for prompts/menus.
    """
    name: str
    method_map: dict[str, dict[str, Any]]


# ──────────────────────────────────────────────────────────────────────────────
# MathPlugin
# Basic arithmetic and trig helpers.
# ──────────────────────────────────────────────────────────────────────────────
MathPlugin: Plugin = {
    "name": "MathPlugin",
    "method_map": {
        "add": {
            "callable": lambda a, b: a + b,
            "description": "Takes in two numbers 'a' and 'b' and returns their sum."
        },
        "subtract": {
            "callable": lambda a, b: a - b,
            "description": "Takes in two numbers 'a' and 'b' and returns their difference."
        },
        "multiply": {
            "callable": lambda a, b: a * b,
            "description": "Takes in two numbers 'a' and 'b' and returns their product."
        },
        "divide": {
            "callable": lambda a, b: a / b if b != 0 else float('inf'),
            "description": "Takes in two numbers 'a' and 'b' and returns their quotient."
        },
        "power": {
            "callable": lambda a, b: a ** b,
            "description": "Takes in two numbers 'a' and 'b' and returns 'a' raised to the power of 'b'."
        },
        "sqrt": {
            "callable": lambda x: x ** 0.5,
            "description": "Takes in a number 'x' and returns its square root. 'x' must be non-negative."
        },
        "mean": {
            "callable": lambda nums: sum(nums) / len(nums) if nums else 0,
            "description": "Takes in a list of numbers and returns their arithmetic mean (average)."
        },
        "max": {
            "callable": lambda nums: max(nums),
            "description": "Takes in a list of numbers and returns the maximum value."
        },
        "min": {
            "callable": lambda nums: min(nums),
            "description": "Takes in a list of numbers and returns the minimum value."
        },
        "sin": {
            "callable": lambda x: __import__('math').sin(x),
            "description": "Takes in an angle in radians and returns its sine."
        },
        "cos": {
            "callable": lambda x: __import__('math').cos(x),
            "description": "Takes in an angle in radians and returns its cosine."
        },
        "tan": {
            "callable": lambda x: __import__('math').tan(x),
            "description": "Takes in an angle in radians and returns its tangent."
        },
        "cot": {
            "callable": lambda x: 1 / __import__('math').tan(x) if __import__('math').tan(x) != 0 else float('inf'),
            "description": "Takes in an angle in radians and returns its cotangent."
        },
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# ConsolePlugin
# Minimal console I/O and logging helpers.
# ──────────────────────────────────────────────────────────────────────────────
ConsolePlugin: Plugin = {
    "name": "ConsolePlugin",
    "method_map": {
        "print": {
            "callable": lambda value: print(value),
            "description": "Args: value (any). Returns: None. Prints any specified value or output to the console. Value can be any type."
        },
        "user_input": {
            "callable": lambda prompt: input(prompt),
            "description": "Args: prompt (str). Returns: str. Gets user input from the console using the given prompt string."
        },
        "basic_config":{
            "callable": lambda level="INFO": logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO)),
            "description": "Args: level (str). Returns: None. Configures basic logging config with the specified log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
        },
        "log": {
            "callable": lambda message, level="INFO": logging.log(getattr(logging, level.upper(), logging.INFO), message),
            "description": "Args: message (str), level (str). Returns: None. Logs a message at the specified log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
        },
    }
}

# ──────────────────────────────────────────────────────────────────────────────
# ParserPlugin
# Simple parsing and string utilities (JSON, split/join, eval, etc.).
# Note: duplicate "split" key preserved intentionally (original behavior).
# ──────────────────────────────────────────────────────────────────────────────
ParserPlugin: Plugin = {
    "name": "ParserPlugin",
    "method_map": {
        "json_loads": {
            "callable": lambda s: json.loads(s),
            "description": "Parses a JSON-formatted string with the json.loads() method and returns a valid Python object."
        },
        "to_str": {
            "callable": lambda x: str(x),
            "description": "Args: x (any). Returns: str. Converts a value to a string."
        },
        "split": {
            "callable": lambda s, sep=None: s.split(sep),
            "description": "Args: s (str), sep (str|None). Returns: list of str. Splits a string by the given separator."
        },
        "safe_eval": {
            "callable": lambda s: eval(s, {"__builtins__": None}, {}),
            "description": "Args: s (str). Returns: any. Safely evaluates a string as a Python literal (use with caution)."
        },
        "extract_json_string": {
            "callable": lambda s: re.search(r'({.*})', s, re.DOTALL).group(1) if re.search(r'({.*})', s, re.DOTALL) else (re.search(r'(\[.*\])', s, re.DOTALL).group(1) if re.search(r'(\[.*\])', s, re.DOTALL) else None),
            "description": "Extracts a potential json-string from text, cleaning off any excess characters or text surrounding it before passing it to 'json_loads'."
        },
        "split": {
            "callable": lambda s, sep=None: s.split(sep),
            "description": "Takes in a string and returns a list of strings that is made by splitting it at every instance of a specified separator."
        },
        "join": {
            "callable": lambda lst, sep="": sep.join(lst),
            "description": "Args: lst (list of str), sep (str). Returns: str. Joins a list of strings into a single string with the given separator."
        },
    }
}
