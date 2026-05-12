# examples/Tool_Examples/04_MCPProxyTool.py
"""
Iterate all tools on a Streamable-HTTP MCP server, wrap each with MCPProxyTool,
and invoke them with example or synthesized inputs.

Server: ensure it runs Streamable-HTTP and is mounted at /mcp (or provide a full URL with path).
"""
from typing import Any, Mapping
import json
from atomic_agentic.tools.mcp import MCPProxyTool
from atomic_agentic.mcp import MCPClientHub
from atomic_agentic.core.Exceptions import ToolInvocationError


SERVER_URL  = "http://127.0.0.1:8000/mcp"   # we'll normalize to /mcp if path missing
SERVER_NAME = "Mathematics_Server"
HEADERS     = None  # e.g., {"Authorization": "Bearer ..."} if your server needs auth

EXAMPLE_INPUTS: dict[str, dict[str, Any]] = {
    "mul":        {"a": 3, "b": 4},
    "multiply":   {"a": 3, "b": 4},
    "power":      {"base": 2, "exponent": 8},
    "factorial":  {"n": 5},
    "derivative": {"func": "x**2 + 2*x + 1", "x": 3.0},
}

client = MCPClientHub(
    endpoint=SERVER_URL,
    transport_mode="streamable_http",
    headers=HEADERS)

def _show_plan(proxy: MCPProxyTool) -> None:
    print(f"\n-- {proxy.full_name} --")
    print("from:", proxy.namespace)
    print("signature:", proxy.signature)
    print("parameters:")
    for param in proxy.parameters:
        default_str = "(no default)" if param.default.__class__.__name__ == "NO_VAL" else f"default={param.default}"
        print(f"  {param.name}: {param.kind}, type={param.type}, {default_str}")

def _invoke(proxy: MCPProxyTool, inputs: Mapping[str, Any]) -> None:
    print("inputs:", inputs)
    try:
        result = proxy.invoke(dict(inputs))
        print("result:", result, type(result))
    except ToolInvocationError as e:
        print("invoke error:", e)

def main() -> None:
    print("Connecting to:", SERVER_URL)
    tool_names = client.list_tools()
    print("Discovered tools:", tool_names.keys())

    for name in tool_names:
        proxy = MCPProxyTool(remote_name=name, client_hub=client)
        _show_plan(proxy)

        inputs = EXAMPLE_INPUTS.get(name)
        _invoke(proxy, inputs)

if __name__ == "__main__":
    main()
