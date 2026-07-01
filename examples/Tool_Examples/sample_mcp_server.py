# Run the below command once you've implemented the server:
# mcpo --port 8000 -- python sample_mcp_server.py

from typing import Annotated, TypedDict
from pydantic import Field
from mcp.server.fastmcp import FastMCP
import math

mcp = FastMCP(name="Demo Mathematics Server")

@mcp.tool()
def multiply(
    a: Annotated[float, Field(description="First factor")],
    b: Annotated[float, Field(description="Second factor")],
) -> list[int]:
    """Multiply two numbers and return the product in a single-item list."""
    return [a * b]

@mcp.tool()
def power(
    base: Annotated[float, Field(description="The base number")],
    exponent: Annotated[float, Field(description="The exponent to raise base to")],
) -> float:
    """Raise base to the power of exponent."""
    return base ** exponent

class FactorialOut(TypedDict):
    n: int
    value: int

@mcp.tool()
def factorial(
    n: Annotated[int, Field(description="Non-negative integer to compute the factorial of")],
) -> FactorialOut:
    """Return n and its factorial as a structured result."""
    return {"n": n, "value": math.factorial(n)}

@mcp.tool()
def derivative(
    func: Annotated[str, Field(description="Stringified function using 'x' as the variable, e.g. 'x**2 + 2*x + 1'")],
    x: Annotated[float, Field(description="Point at which to evaluate the derivative")],
) -> float:
    """Numerically approximate the derivative of func at point x."""
    h = 1e-12
    def f(x_val):
        return eval(func, {"x": x_val, "math": math})
    return (f(x + h) - f(x)) / h

if __name__ == "__main__":
    MODE = "streamable-http" #"stdio" # Streamable HTTP for standard MCP servers, stdio for mcpo-style servers
    mcp.run(transport=MODE)
