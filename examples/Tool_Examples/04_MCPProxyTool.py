# examples/Tool_Examples/04_MCPProxyTool.py
"""
Iterate all tools on an MCP server, wrap each with MCPProxyTool, and invoke
them with example or synthesized inputs.

Supports all three MCPClientHub transports via TRANSPORT_MODE below:
- "stdio": this script spawns `sample_mcp_server.py` itself as a subprocess
  -- no separate launch needed. Just set sample_mcp_server.py's MODE = "stdio"
  and run this script.
- "sse" / "streamable_http": start `sample_mcp_server.py` yourself first
  (set its MODE to "sse" or "streamable-http" to match), then run this
  script separately.
"""
from pathlib import Path
from typing import Any, Literal, Mapping
import json
import sys
from atomic_agentic.tools import MCPProxyTool
from atomic_agentic.mcp import MCPClientHub
from atomic_agentic.exceptions import ToolInvocationError


TRANSPORT_MODE: Literal["stdio", "sse", "streamable_http"] = "stdio"
SERVER_NAME = "Mathematics_Server"
HEADERS     = None  # e.g., {"Authorization": "Bearer ..."} if your server needs auth

# One planned MCPClientHub configuration per transport -- edit these directly
# to try out read_timeout_seconds/client_kwargs/session_kwargs. Only
# CLIENT_CONFIGS[TRANSPORT_MODE] is actually used to build `client` below.
CLIENT_CONFIGS: dict[str, dict[str, Any]] = {
    "stdio": dict(
        transport_mode="stdio",
        command=sys.executable,
        args=[str(Path(__file__).parent / "sample_mcp_server.py")],
        client_kwargs={"cwd": str(Path(__file__).parent)},
        read_timeout_seconds=30.0,
    ),
    "sse": dict(
        transport_mode="sse",
        endpoint="http://127.0.0.1:8000/sse",
        headers=HEADERS,
        client_kwargs={"timeout": 5, "sse_read_timeout": 300},
        read_timeout_seconds=30.0,
    ),
    "streamable_http": dict(
        transport_mode="streamable_http",
        endpoint="http://127.0.0.1:8000/mcp",
        headers=HEADERS,
        client_kwargs={"terminate_on_close": True},
        read_timeout_seconds=30.0,
    ),
}

EXAMPLE_INPUTS: dict[str, dict[str, Any]] = {
    "mul":        {"a": 3, "b": 4},
    "multiply":   {"a": 3, "b": 4},
    "power":      {"base": 2, "exponent": 8},
    "factorial":  {"n": 5},
    "derivative": {"func": "x**2 + 2*x + 1", "x": 3.0},
}

client = MCPClientHub(**CLIENT_CONFIGS[TRANSPORT_MODE])

def _show_plan(proxy: MCPProxyTool) -> None:
    print(f"\n-- {proxy.full_name} --")
    print("from:", proxy.namespace)
    print("description:", proxy.description)
    print("signature:", proxy.signature)
    print("parameters:")
    for param in proxy.parameters:
        default_str = "(no default)" if param.default.__class__.__name__ == "NO_VAL" else f"default={param.default}"
        desc_str = f"  # {param.description}" if param.description else ""
        print(f"  {param.name}: {param.kind}, type={param.type}, {default_str}{desc_str}")
    print("structured (to_dict):")
    print(json.dumps(proxy.to_dict(), indent=2, default=str))

def _invoke(proxy: MCPProxyTool, inputs: Mapping[str, Any]) -> None:
    print("inputs:", inputs)
    try:
        result = proxy.invoke(dict(inputs))
        print("result:", result.result, type(result))
    except ToolInvocationError as e:
        print("invoke error:", e)

def main() -> None:
    print(f"Connecting via {TRANSPORT_MODE}:", client.endpoint or client.command)
    tool_names = client.list_tools()
    print("Discovered tools:", tool_names.keys())

    for name in tool_names:
        proxy = MCPProxyTool(remote_name=name, namespace = SERVER_NAME, client_hub=client)
        _show_plan(proxy)

        inputs = EXAMPLE_INPUTS.get(name)
        _invoke(proxy, inputs)

if __name__ == "__main__":
    main()
