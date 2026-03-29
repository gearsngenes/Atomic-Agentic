from __future__ import annotations

import json
from typing import Any, Mapping

from dotenv import load_dotenv

from atomic_agentic.a2a.PyA2AtomicClient import PyA2AtomicClient
from atomic_agentic.agents import Agent
from atomic_agentic.core.Exceptions import ToolInvocationError
from atomic_agentic.core.sentinels import NO_VAL
from atomic_agentic.engines.LLMEngines import OpenAIEngine
from atomic_agentic.mcp.MCPClientHub import MCPClientHub
from atomic_agentic.tools.Toolify import batch_toolify, toolify
from atomic_agentic.tools.base import Tool

load_dotenv()


def add_scale(a: int, b: int, scale: float = 1.0) -> float:
    """Return (a + b) * scale."""
    return (a + b) * scale


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

MCP_ENDPOINT = "http://127.0.0.1:8000/mcp"
MCP_NAMESPACE = "demo_mcp"
MCP_HEADERS: Mapping[str, str] | None = None

A2A_MATH_URL = "http://localhost:7000"
A2A_MATH_REMOTE_NAME = "MathPlannerAgent"
A2A_HEADERS: Mapping[str, str] | None = None


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _jsonable_params(t: Tool) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    for spec in t.parameters:
        item = spec.to_dict() if hasattr(spec, "to_dict") else dict(spec)
        if "default" in item and item["default"] is not NO_VAL:
            item["default"] = _jsonable(item["default"])
        params.append(item)
    return params


def show_plan(t: Tool) -> None:
    print("\n" + "=" * 72)
    print(f"Tool       : {t.full_name}")
    print(f"Namespace  : {t.namespace}")
    print(f"Description: {t.description}")
    print(f"Signature  : {t.signature}")
    print(f"Return type: {t.return_type}")
    print("Parameters :")
    print(json.dumps(_jsonable_params(t), indent=2))


def invoke_with_inputs(t: Tool, inputs: Mapping[str, Any] | None = None) -> None:
    payload = dict(inputs or {})

    print("\nInputs:")
    if payload:
        for k, v in payload.items():
            print(f"  {k} = {_jsonable(v)!r}")
    else:
        print("  <empty mapping>")

    try:
        result = t.invoke(payload)
        print("Result:")
        print(f"  {_jsonable(result)!r}")
    except ToolInvocationError as exc:
        print("Invocation error:")
        print(f"  {exc}")
    except Exception as exc:
        print("Error:")
        print(f"  {exc}")


def main() -> None:
    print("\n[1] Agent -> AdapterTool via toolify(agent)")
    agent_tool = toolify(agent)
    show_plan(agent_tool)
    invoke_with_inputs(
        agent_tool,
        {"topic": "unit testing", "style": "concise", "audience": "beginners"},
    )

    print("\n[2] Callable -> Tool via toolify(add_scale, ...)")
    callable_tool = toolify(
        add_scale,
        name="add_scale",
        description="Compute (a + b) * scale.",
        namespace="local_demo",
    )
    show_plan(callable_tool)
    invoke_with_inputs(callable_tool, {"a": 2, "b": 3, "scale": 10})

    print("\n[3] Tool -> same Tool instance, updated in place when overrides are provided")
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
            mcp_tool = toolify(
                mcp_hub,
                remote_name=remote_name,
                namespace=MCP_NAMESPACE,
            )
            show_plan(mcp_tool)
            invoke_with_inputs(mcp_tool, example_inputs_by_remote_name.get(remote_name))

    except Exception as exc:
        print("[MCP] Skipping MCP demo due to error:", exc)

    print("\n[5] PyA2AtomicClient + remote_name -> PyA2AtomicTool via toolify(client, ...)")
    print("[A2A] Connecting to:", A2A_MATH_URL)
    try:
        a2a_client = PyA2AtomicClient(
            url=A2A_MATH_URL,
            headers=A2A_HEADERS,
        )
        discovered = a2a_client.list_invokables()
        print(f"[A2A] Discovered {len(discovered)} invokables: {list(discovered.keys())}")

        a2a_tool = toolify(
            a2a_client,
            remote_name=A2A_MATH_REMOTE_NAME,
            namespace="demo_a2a",
        )
        show_plan(a2a_tool)
        invoke_with_inputs(
            a2a_tool,
            {
                "prompt": "Give the sum of 20 and negative six all divided by seven.",
            },
        )

    except Exception as exc:
        print("[A2A] Skipping A2A demo due to error:", exc)

    print("\n[6] batch_toolify([PyA2AtomicClient]) -> one proxy per remote invokable")
    try:
        a2a_client = PyA2AtomicClient(
            url=A2A_MATH_URL,
            headers=A2A_HEADERS,
        )
        a2a_tools = batch_toolify(
            [a2a_client],
            batch_namespace="demo_a2a_batch",
        )
        print(f"[A2A] batch_toolify produced {len(a2a_tools)} tool(s):")
        for t in a2a_tools:
            print(f"  - {t.full_name}")

    except Exception as exc:
        print("[A2A] Skipping batch A2A demo due to error:", exc)


if __name__ == "__main__":
    main()
