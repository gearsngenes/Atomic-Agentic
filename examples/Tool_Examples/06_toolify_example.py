from __future__ import annotations

import json
from typing import Any, Mapping

from dotenv import load_dotenv

from atomic_agentic.tools import Tool
from atomic_agentic.tools.Toolify import toolify
from atomic_agentic.agents import Agent
from atomic_agentic.engines.LLMEngines import OpenAIEngine
from atomic_agentic.core.Exceptions import ToolInvocationError
from atomic_agentic.mcp.MCPClientHub import MCPClientHub

load_dotenv()


# ---------- A callable we'll wrap via toolify(callable, ...) ----------


def add_scale(a: int, b: int, scale: float = 1.0) -> float:
    """Return (a + b) * scale."""
    return (a + b) * scale


# ---------- A Tool we pre-wrap manually (then pass to toolify) ----------


def summarize(text: str, limit: int = 40) -> str:
    """Truncate text to `limit` chars with ellipsis."""
    s = str(text)
    return s if len(s) <= limit else s[: max(0, limit - 1)] + "…"


pre_wrapped_tool = Tool(
    function=summarize,
    name="summarize",
    namespace="local_demo",
    description="Truncate text to a character limit (default limit: 40).",
)


# ---------- Agent that uses a pre_invoke Tool (drives AdapterTool schema) ----------


def to_prompt(topic: str, style: str, *, audience: str = "general") -> str:
    """Compose a small prompt string from structured inputs."""
    return f"Write about '{topic}' in a {style} style for {audience} readers."


pre_invoke_tool = Tool(
    function=to_prompt,
    name="to_prompt",
    namespace="local_demo",
    description="Compose a prompt from {topic, style, audience?}. Returns a string prompt.",
)

agent = Agent(
    name="Writer",
    description="Concise writing assistant.",
    llm_engine=OpenAIEngine("gpt-4o-mini"),
    role_prompt="You are a concise writing assistant.",
    pre_invoke=pre_invoke_tool,
)


# ---------- Remote demo targets ----------
# MCP: run your sample MCP server separately and expose streamable HTTP at /mcp.
# A2A: run trivia_host_server.py separately; it binds on localhost:6000.

MCP_ENDPOINT = "http://127.0.0.1:8000/mcp"
MCP_NAMESPACE = "demo_mcp"
MCP_HEADERS: Mapping[str, str] | None = None

A2A_TRIVIA_ENDPOINT = "http://localhost:6000"
A2A_HEADERS: Mapping[str, str] | None = None


# ---------- Helpers ----------


def _jsonable_params(t: Tool) -> list[dict[str, Any]]:
    """Convert ParamSpec-like items into JSON-safe dicts for display."""
    params: list[dict[str, Any]] = []
    for spec in t.parameters:
        if hasattr(spec, "to_dict"):
            item = spec.to_dict()
        else:
            item = dict(spec)

        if "default" in item:
            try:
                json.dumps(item["default"])
            except TypeError:
                item["default"] = repr(item["default"])

        params.append(item)
    return params


def show_plan(t: Tool) -> None:
    """Pretty-print a human-readable overview of a Tool."""
    print("\n" + "=" * 72)
    print(f"Tool       : {t.full_name}")
    print(f"Namespace  : {t.namespace}")
    print(f"Description: {t.description}")
    print(f"Signature  : {t.signature}")
    print(f"Return type: {t.return_type}")
    print("Parameters :")
    print(json.dumps(_jsonable_params(t), indent=2))


def invoke_with_inputs(t: Tool, inputs: Mapping[str, Any] | None = None) -> None:
    """Invoke a Tool with given inputs and show a readable result."""
    payload = dict(inputs or {})

    print("\nInputs:")
    if payload:
        for k, v in payload.items():
            print(f"  {k} = {v!r}")
    else:
        print("  <empty mapping>")

    try:
        result = t.invoke(payload)
        print("Result:")
        print(f"  {result!r}")
    except ToolInvocationError as exc:
        print("Invocation error:")
        print(f"  {exc}")
    except Exception as exc:
        print("Error:")
        print(f"  {exc}")


# ---------- Demo flow ----------


def main() -> None:
    # 1) Agent -> AdapterTool
    print("\n[1] Agent -> AdapterTool via toolify(agent)")
    agent_tool = toolify(agent)
    show_plan(agent_tool)
    invoke_with_inputs(
        agent_tool,
        {"topic": "unit testing", "style": "concise", "audience": "beginners"},
    )

    # 2) Callable -> Tool
    print("\n[2] Callable -> Tool via toolify(add_scale, ...)")
    callable_tool = toolify(
        add_scale,
        name="add_scale",
        description="Compute (a + b) * scale.",
        namespace="local_demo",
    )
    show_plan(callable_tool)
    invoke_with_inputs(callable_tool, {"a": 2, "b": 3, "scale": 10})

    # 3) Tool -> same instance, updated in place when overrides are provided
    print("\n[3] Tool -> same Tool instance, with optional metadata updates")
    updated_tool = toolify(
        pre_wrapped_tool,
        namespace="local_demo_updated",
        description="Truncate text to a character limit with ellipsis.",
    )
    show_plan(updated_tool)
    invoke_with_inputs(
        updated_tool,
        {
            "text": "Atomic-Agentic is awesome for strict schema tools!",
            "limit": 28,
        },
    )

    # 4) MCP client hub + remote_name -> MCPProxyTool
    print("\n[4] MCPClientHub + remote_name -> MCPProxyTool")
    print("[MCP] Connecting to:", MCP_ENDPOINT)

    try:
        mcp_hub = MCPClientHub(
            transport_mode="streamable_http",
            endpoint=MCP_ENDPOINT,
            headers=MCP_HEADERS,
        )

        remote_tools = mcp_hub.list_tools()
        remote_names = list(remote_tools.keys())
        print(f"[MCP] Discovered {len(remote_names)} tools: {remote_names}")

        example_inputs_by_remote_name: dict[str, dict[str, Any]] = {
            "mul": {"a": 3, "b": 4},
            "multiply": {"a": 3, "b": 4},
            "power": {"base": 2, "exponent": 8},
            "factorial": {"n": 5},
            "derivative": {"func": "2.71828**x + 2*x + 1", "x": 3.0},
        }

        if remote_names:
            remote_name = remote_names[0]
            print(f"[MCP] Registering remote MCP tool {remote_name!r}...")

            mcp_tool = toolify(
                mcp_hub,
                remote_name=remote_name,
                namespace=MCP_NAMESPACE,
            )
            show_plan(mcp_tool)

            inputs = example_inputs_by_remote_name.get(remote_name)
            if not inputs:
                print("(no canned inputs for this MCP tool; invoking with empty input mapping)")
            invoke_with_inputs(mcp_tool, inputs)

    except Exception as exc:
        print("[MCP] Skipping MCP demo due to error:", exc)

    # 5) A2A endpoint -> A2AProxyTool (Trivia host)
    print("\n[5] A2A endpoint -> A2AProxyTool via toolify(component=None, a2a_endpoint=...)")
    print("[A2A] Start trivia_host_server.py first. Connecting to:", A2A_TRIVIA_ENDPOINT)

    try:
        a2a_tool = toolify(
            None,
            namespace="demo_a2a",
            a2a_endpoint=A2A_TRIVIA_ENDPOINT,
            a2a_headers=A2A_HEADERS,
        )
        show_plan(a2a_tool)
        invoke_with_inputs(
            a2a_tool,
            {
                "prompt": (
                    "Give me two concise trivia facts about Saturn."
                )
            },
        )
    except Exception as exc:
        print("[A2A] Skipping A2A demo due to error:", exc)


if __name__ == "__main__":
    main()