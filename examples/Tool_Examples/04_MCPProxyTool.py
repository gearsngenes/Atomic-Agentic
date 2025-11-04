# examples/Tool_Examples/04_MCPProxyTool.py
"""
Iterate all tools on a Streamable-HTTP MCP server, wrap each with MCPProxyTool,
and invoke them with example or synthesized inputs.

Server: ensure it runs Streamable-HTTP and is mounted at /mcp (or provide a full URL with path).
"""

import sys, asyncio
from pathlib import Path
from typing import Any, Mapping

# repo root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from urllib.parse import urlparse, urlunparse
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from modules.ToolAdapters import MCPProxyTool
from modules.Tools import ToolInvocationError


SERVER_URL  = "http://127.0.0.1:8000"   # we'll normalize to /mcp if path missing
SERVER_NAME = "Mathematics_Server"
HEADERS     = None  # e.g., {"Authorization": "Bearer ..."} if your server needs auth

EXAMPLE_INPUTS: dict[str, dict[str, Any]] = {
    "mul":        {"a": 3, "b": 4},
    "multiply":   {"a": 3, "b": 4},
    "power":      {"base": 2, "exponent": 8},
    "factorial":  {"n": 5},
    "derivative": {"func": "x**2 + 2*x + 1", "x": 3.0},
}

def ensure_streamable_http_url(url: str) -> str:
    parts = urlparse(url)
    if not parts.path or parts.path == "/":
        parts = parts._replace(path="/mcp")
    return urlunparse(parts)

async def _list_remote_tools(url: str) -> list[str]:
    # streamablehttp_client(...) may yield a transport object or a tuple
    async with streamablehttp_client(url=url, headers=HEADERS) as transport:
        # robust extraction of (read, write)
        read = getattr(transport, "read", None)
        write = getattr(transport, "write", None)
        if read is None or write is None:
            read, write = transport[0], transport[1]  # tuple form
        async with ClientSession(read, write) as session:
            # REQUIRED handshake before any request.  (initialize → list_tools)  :contentReference[oaicite:10]{index=10}
            await session.initialize()
            tools_resp = await session.list_tools()  # discover available tools  :contentReference[oaicite:11]{index=11}
            tool_objs = getattr(tools_resp, "tools", tools_resp)
            return [t.name for t in tool_objs]

def _show_plan(proxy: MCPProxyTool) -> None:
    meta = proxy.to_dict()
    print(f"\n-- {proxy.full_name} --")
    print("from:", proxy.source)
    print("signature:", meta["signature"])
    print("required: ", sorted(meta["required_names"]))
    print("params:   ", meta["p_or_kw_names"])

def _invoke(proxy: MCPProxyTool, inputs: Mapping[str, Any]) -> None:
    print("inputs:", inputs)
    try:
        result = proxy.invoke(dict(inputs))
        print("result:", result)
    except ToolInvocationError as e:
        print("invoke error:", e)

def main() -> None:
    resolved = ensure_streamable_http_url(SERVER_URL)
    print("Connecting to:", resolved)
    tool_names = asyncio.run(_list_remote_tools(resolved))
    print("Discovered tools:", tool_names)

    for name in tool_names:
        proxy = MCPProxyTool(tool_name=name, server_name=SERVER_NAME, server_url=resolved, headers=HEADERS)
        _show_plan(proxy)

        inputs = EXAMPLE_INPUTS.get(name)
        _invoke(proxy, inputs)

if __name__ == "__main__":
    main()
