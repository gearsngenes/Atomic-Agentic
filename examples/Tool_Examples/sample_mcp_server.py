# Run the below command once you've implemented the server:
# mcpo --port 8000 -- python sample_mcp_server.py

from mcp.server.fastmcp import FastMCP
import math

mcp = FastMCP(name="Demo Mathematics Server")

@mcp.tool()
def multiply(a: float, b: float) -> list[int]:
    """Multiply two numbers a, b, and return the product in a single-item list."""
    return [a * b]

@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise 'base' to the power of 'exponent'."""
    return base ** exponent

from typing import TypedDict
class FactorialOut(TypedDict):
    n: int
    value: int

@mcp.tool()
def factorial(n: int) -> FactorialOut:
    import math
    return {"n": n, "value": math.factorial(n)}


@mcp.tool()
def derivative(func: str, x: float) -> float:
    """
    Description: Calculates (or closely approximates) the derivative of a function 'func' at point 'x'

    Args:
        func: str — A stringified representation of the function, using 'x' as the variable.
        x: float — The specific point at which to evaluate the derivative.

    Returns:
        float — The numerical approximation of the derivative at x.
    """
    h = 1e-12
    def f(x_val):
        return eval(func, {"x": x_val, "math": math})
    return (f(x + h) - f(x)) / h

if __name__ == "__main__":
    MODE = "streamable-http" #"stdio" # Streamable HTTP for standard MCP servers, stdio for mcpo-style servers
    mcp.run(transport=MODE)
