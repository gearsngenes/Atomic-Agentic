# Run the below command once you've implemented the server:
# mcpo --port 8000 -- python sample_mcp_server.py

from typing import Annotated
from pydantic import Field
try:
    from mcp.server.fastmcp import FastMCP as MCPServer  # mcp < 2
except ImportError:
    from mcp.server.mcpserver import MCPServer  # mcp >= 2
import ast
import math

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Call, ast.Attribute, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
    ast.Mod, ast.USub, ast.UAdd,
)


def _compile_math_expr(source: str):
    """Compile ``source`` only if it is plain arithmetic in ``x`` plus ``math.<name>(...)``."""
    tree = ast.parse(source, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsupported syntax in function: {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed in function.")
        if isinstance(node, ast.Name) and node.id not in ("x", "math"):
            raise ValueError(f"Unknown name in function: {node.id!r}")
        if isinstance(node, ast.Attribute) and not (
            isinstance(node.value, ast.Name) and node.value.id == "math"
            and not node.attr.startswith("_")
        ):
            raise ValueError("Only math.<name> attribute access is allowed in function.")
    return compile(tree, "<function>", "eval")

mcp = MCPServer(name="Demo Mathematics Server")

@mcp.tool()
def mul(
    a: Annotated[float, Field(description="First factor")],
    b: Annotated[float, Field(description="Second factor")],
) -> float:
    """Multiply two numbers and return the product."""
    return a * b

@mcp.tool()
def power(
    base: Annotated[float, Field(description="The base number")],
    exponent: Annotated[float, Field(description="The exponent to raise base to")],
) -> float:
    """Raise base to the power of exponent."""
    return base ** exponent

@mcp.tool()
def factorial(
    n: Annotated[int, Field(description="Non-negative integer to compute the factorial of")],
) -> int:
    """Calculate the factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    return math.factorial(n)

@mcp.tool()
def derivative(
    func: Annotated[str, Field(description="Stringified function using 'x' as the variable, e.g. 'x**2 + 2*x + 1'")],
    x: Annotated[float, Field(description="Point at which to evaluate the derivative")],
) -> float:
    """Numerically approximate the derivative of func at point x."""
    h = 1e-12
    code = _compile_math_expr(func)
    def f(x_val):
        return eval(code, {"__builtins__": {}, "x": x_val, "math": math})
    return (f(x + h) - f(x)) / h

if __name__ == "__main__":
    MODE = "streamable-http" #"stdio" # Streamable HTTP for standard MCP servers, stdio for mcpo-style servers
    mcp.run(transport=MODE)
