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
from atomic_agentic.constants.core import NO_VAL


TRANSPORT_MODE: Literal["stdio", "sse", "streamable_http"] = "streamable_http"
SERVER_NAME = "Mathematics_Server"
HEADERS     = None  # e.g., {"Authorization": "Bearer ..."} if your server needs auth

# One planned MCPClientHub configuration per transport -- edit these directly
# to try out read_timeout_seconds/client_kwargs/session_kwargs. Only
# CLIENT_CONFIGS[TRANSPORT_MODE] is actually used to build `client` below.
# persistent defaults to False (one-shot per call). Add persistent=True to a
# config (and use `with MCPClientHub(...) as client:` or
# call client.close() when done) to hold one connection open across every
# tool call this script makes instead of reopening per call.
CLIENT_CONFIGS: dict[str, dict[str, Any]] = {
    "stdio": dict(
        location=sys.executable,
        transport_mode="stdio",
        args=[str(Path(__file__).parent / "sample_mcp_server.py")],
        client_kwargs={"cwd": str(Path(__file__).parent)},
        read_timeout_seconds=30.0,
    ),
    "sse": dict(
        location="http://127.0.0.1:8000/sse",
        transport_mode="sse",
        headers=HEADERS,
        client_kwargs={"timeout": 5, "sse_read_timeout": 300},
        read_timeout_seconds=30.0,
    ),
    "streamable_http": dict(
        location="http://127.0.0.1:8000/mcp",
        transport_mode="streamable_http",
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

def _show_plan(proxy: MCPProxyTool) -> None:
    print(f"\n-- {proxy.full_name} --")
    print("description:", proxy.description)
    print("signature:", proxy.signature)
    print("parameters:")
    for param in proxy.parameters:
        default_str = "(no default)" if param.default is NO_VAL else f"default={param.default}"
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
    with MCPClientHub(**CLIENT_CONFIGS[TRANSPORT_MODE]) as client:
        print(f"Connected via {TRANSPORT_MODE}:", client.location)
        tool_names = list(client.list_tools())
        print("Discovered tools:", tool_names)

        for name in tool_names:
            proxy = MCPProxyTool(client, name)
            _show_plan(proxy)

            inputs = EXAMPLE_INPUTS.get(name)
            if inputs is None:
                print(f"(no example inputs for {name!r}; skipping invocation)")
                continue
            _invoke(proxy, inputs)

if __name__ == "__main__":
    main()
