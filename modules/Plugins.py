import sys
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from abc import ABC
from typing import Dict, Any, List
import re
import json
import logging

class Plugin(ABC):
    def __init__(self):
        self._plugin_description: str = ""
        self._method_map: Dict[str, Dict[str, Any]] = {}

    def plugin_description(self) -> str:
        return self._plugin_description

    def method_map(self) -> Dict[str, Dict[str, Any]]:
        # Return a copy to prevent external mutation
        return self._method_map.copy()

    def get_methods(self, method_names: List[str]) -> Dict[str, Dict[str, Any]]:
        missing = [name for name in method_names if name not in self._method_map]
        if missing:
            raise ValueError(f"Methods not found in plugin: {missing}")
        return {name: self._method_map[name] for name in method_names}

class MathPlugin(Plugin):
    def __init__(self):
        super().__init__()
        self._plugin_description = (
            "Provides basic mathematical operations: addition, subtraction, multiplication, division, exponents, and more. "
            "Best suited for tasks involving arithmetic, algebra, and numeric computation."
        )
        self._method_map = {
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

class ConsolePlugin(Plugin):
    def __init__(self):
        super().__init__()
        self._plugin_description = (
            "Provides console I/O methods: print, log, and user input. "
            "Best suited for tasks involving terminal output, logging, and user interaction."
        )
        self._method_map = {
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

class ParserPlugin(Plugin):
    def __init__(self):
        super().__init__()
        self._plugin_description = (
            "Focuses on parsing and extracting values from string inputs, converting them to standard Python primitives "
            "(int, float, bool, list, dict, tuple, etc). Useful for text parsing, data cleaning, and structured extraction."
        )
        self._method_map = {
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

class PythonPlugin(Plugin):
    def __init__(self):
        super().__init__()
        self._plugin_description = (
            "Provides Python-specific tools for code execution, expression evaluation, type inspection, "
            "module import, and dynamic function generation/execution. Useful for automation and meta-programming."
        )
        self._method_map = {
            "exec_code": {
                "callable": lambda code: exec(code),
                "description": "Args: code (str). Returns: dict or error string. Executes arbitrary Python code presented as a string."
            },
            "get_type": {
                "callable": lambda x: type(x).__name__,
                "description": "Args: x (any). Returns: str. Returns the type name of an object."
            },
        }
