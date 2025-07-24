# Run the below command once you've implemented the server:
# mcpo --port 8000 -- python sample_mcp_server.py

from mcp.server.fastmcp import FastMCP
import math

mcp = FastMCP(name="Demo Mathematics Server")

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
    mcp.run(transport="stdio")
