"""
Toolify demo: converting components into single Tool instances.

Toolify patterns:
- Agent           -> toolify(agent)                      -> AgentTool (single)
- Callable        -> toolify(callable, ...)              -> Tool (single)
- Tool (wrapped)  -> toolify(tool)                       -> Tool (single, passthrough)
- MCP URL        -> toolify(url, name=..., remote_protocol='mcp') -> MCPProxyTool (single)
- A2A URL        -> toolify(url, remote_protocol='a2a') -> A2AProxyTool (single, auto-discovers)

Note: toolify() returns a SINGLE Tool instance, not a list.

Server for MCP path (run separately):
    # Ensure your sample server is Streamable-HTTP mounted at /mcp
    # e.g., in another shell:
    #   python sample_mcp_server.py
"""

from typing import Any, Mapping, MutableMapping

from dotenv import load_dotenv

from atomic_agentic.tools.Toolify import toolify
from atomic_agentic import NO_VAL
from atomic_agentic.tools import Tool
from atomic_agentic.agents import Agent
from atomic_agentic.engines.LLMEngines import OpenAIEngine
from atomic_agentic.core.Exceptions import ToolInvocationError

load_dotenv()  # take environment variables from .env file (if exists)


# ---------- A callable we'll wrap via toolify(callable, ...) ----------


def add_scale(a: int, b: int, scale: float = 1.0) -> float:
    """Return (a + b) * scale."""
    return (a + b) * scale


# ---------- A Tool we pre-wrap manually (then pass to toolify) ----------


def summarize(text: str, limit: int = 40) -> str:
    """Truncate text to 'limit' chars with ellipsis."""
    s = str(text)
    return s if len(s) <= limit else s[: max(0, limit - 1)] + "…"


# New Tool API: Tool(function, name, namespace, description)
pre_wrapped_tool = Tool(
    summarize,
    "summarize",
    "local.demo",
    "Truncate text to a character limit (limit default: 40).",
)


# ---------- Agent that uses a pre_invoke Tool (drives AgentTool schema) ----------


def to_prompt(topic: str, style: str, *, audience: str = "general") -> str:
    """Compose a small prompt string from structured inputs."""
    return f"Write about '{topic}' in a {style} style for {audience} readers."


pre_invoke_tool = Tool(
    to_prompt,
    "to_prompt",
    "local.demo",
    "Compose prompt from {topic, style, audience?}. Returns a string prompt.",
)

agent = Agent(
    name="Writer",
    description="Concise writing assistant.",
    llm_engine=OpenAIEngine("gpt-4o-mini"),
    role_prompt="You are a concise writing assistant.",
    pre_invoke=pre_invoke_tool,
)


# ---------- MCP server URL (normalized to /mcp if path is empty) ----------

SERVER_URL = "http://127.0.0.1:8000"  # we'll normalize to /mcp if needed
MCP_NAMESPACE = "demo_mcp"
MCP_HEADERS: MutableMapping[str, str] | None = None  # presence of 'headers' is required

# ---------- Helpers ----------


def show_plan(t: Tool) -> None:
    """Pretty-print a human-readable overview of a Tool."""
    print("\n" + "=" * 72)
    print(f"Tool       : {t.full_name}")
    ns = t.namespace
    print(f"Namespace  : {ns}")

    desc = t.description
    desc = desc.strip()
    print(f"Description: {desc or '<no description>'}")

    signature = t.signature
    print(f"Signature  : {signature}")
    import json
    print("Parameters :", json.dumps(t.parameters, indent=2))

def invoke_with_inputs(t: Tool, inputs: Mapping[str, Any]) -> None:
    """Invoke a Tool with given inputs and show a readable result."""
    print("\nInputs:")
    for k, v in inputs.items():
        print(f"  {k} = {v!r}")

    try:
        result = t.invoke(dict(inputs))
        print("Result:")
        print(f"  {result!r}")
    except ToolInvocationError as e:
        print("Invocation error:")
        print(f"  {e}")


# ---------- Demo flow ----------


def main() -> None:
    # 1) Agent -> AgentTool
    print("\n[1] Agent -> AgentTool via toolify(agent)")
    agent_tool = toolify(agent)
    show_plan(agent_tool)
    invoke_with_inputs(
        agent_tool,
        {"topic": "unit testing", "style": "concise", "audience": "beginners"},
    )

    # 2) Callable -> Tool (requires name & description)
    print("\n[2] Callable -> Tool via toolify(add_scale, ...)")
    callable_tool = toolify(
        add_scale,
        name="add_scale",
        description="Compute (a + b) * scale.",
        namespace="local.demo",
    )
    show_plan(callable_tool)
    invoke_with_inputs(callable_tool, {"a": 2, "b": 3, "scale": 10})

    # 3) Pre-wrapped Tool -> passthrough
    print("\n[3] Pre-wrapped Tool -> passthrough via toolify(pre_wrapped_tool)")
    passthrough_tool = toolify(pre_wrapped_tool)
    show_plan(passthrough_tool)
    invoke_with_inputs(
        passthrough_tool,
        {
            "text": "Atomic-Agentic is awesome for strict schema tools!",
            "limit": 28,
        },
    )

    # 4) MCP URL + name -> MCPProxyTool (single tool from server)
    print("\n[4] MCP URL + name -> single MCPProxyTool via toolify(url, name=..., remote_protocol='mcp', ...)")
    print("[MCP] Connecting to:", SERVER_URL)
    try:
        # Provide canned examples for common demo names; synthesize otherwise
        EXAMPLE_INPUTS: dict[str, dict[str, Any]] = {
            "mul": {"a": 3, "b": 4},
            "multiply": {"a": 3, "b": 4},
            "power": {"base": 2, "exponent": 8},
            "factorial": {"n": 5},
            "derivative": {"func": "2.71828**x + 2*x + 1", "x": 3.0},
        }

        # First, discover available tools on the server
        from atomic_agentic.tools import list_mcp_tools
        tool_names = list_mcp_tools(SERVER_URL, headers=MCP_HEADERS)
        print(f"[MCP] Discovered {len(tool_names)} tools: {list(tool_names.keys())}")

        # Show a single example tool from the server
        if tool_names:
            example_name = list(tool_names.keys())[0]
            print(f"\n[4a] Registering MCP tool '{example_name}'...")
            mcp_tool = toolify(
                SERVER_URL,
                name=example_name,
                namespace=MCP_NAMESPACE,
                remote_protocol="mcp",
                headers=MCP_HEADERS,
            )
            show_plan(mcp_tool)
            inputs = EXAMPLE_INPUTS.get(mcp_tool.name)
            if not inputs:
                print("(no required params detected; calling with empty inputs)")
            invoke_with_inputs(mcp_tool, inputs)

    except Exception as e:
        print("[MCP] Skipping MCP demo due to error:", e)


if __name__ == "__main__":
    main()
