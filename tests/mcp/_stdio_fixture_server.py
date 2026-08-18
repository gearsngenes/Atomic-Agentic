"""Minimal real MCP stdio server, spawned as a subprocess by
test_client_hub_real_stdio.py. Not itself a pytest module -- run directly
via `sys.executable <this file>` as a persistent MCPClientHub's stdio
transport, exercising the real anyio-based mcp SDK client/server plumbing
(no mocks) that test_client_hub.py's monkeypatched fakes cannot reach.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="AA Test Fixture Stdio Server")


@mcp.tool()
def echo(value: str) -> str:
    """Return value unchanged."""
    return value


if __name__ == "__main__":
    mcp.run(transport="stdio")
