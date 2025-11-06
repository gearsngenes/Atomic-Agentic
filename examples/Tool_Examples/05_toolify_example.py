"""
ToolFactory demo:
- Agent  -> toolify(agent)          -> [AgentTool]
- Callable -> toolify(callable, ...) -> [Tool]
- Tool (pre-wrapped) -> toolify(tool) -> [Tool]
- MCP URL -> toolify(url, server_name=...) -> [MCPProxyTool, ...]

Server for MCP path (run separately):
    # Ensure your sample server is Streamable-HTTP mounted at /mcp
    # e.g., in another shell:
    #   python sample_mcp_server.py
"""

import sys
from pathlib import Path
from typing import Any, Mapping

# Make repo root importable (aligns with other examples)
sys.path.append(str(Path(__file__).resolve().parents[2]))

from urllib.parse import urlparse, urlunparse
from modules.ToolAdapters import toolify
from modules.Tools import Tool, ToolInvocationError
from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine


# ---------- A callable we'll wrap via toolify(callable, ...) ----------

def add_scale(a: int, b: int, scale: float = 1.0) -> float:
    """Return (a + b) * scale."""
    return (a + b) * scale


# ---------- A Tool we pre-wrap manually (then pass to toolify) ----------

def summarize(text: str, limit: int = 40) -> str:
    """Truncate text to 'limit' chars with ellipsis."""
    s = str(text)
    return s if len(s) <= limit else s[: max(0, limit - 1)] + "…"

pre_wrapped_tool = Tool(
    func=summarize,
    name="summarize",
    description="Truncate text to a character limit (limit default: 40).",
    type="python",
    source="local",
)


# ---------- Agent that uses a pre_invoke Tool (drives AgentTool schema) ----------

def to_prompt(topic: str, style: str, *, audience: str = "general") -> str:
    """Compose a small prompt string from structured inputs."""
    return f"Write about '{topic}' in a {style} style for {audience} readers."

pre_invoke_tool = Tool(
    func=to_prompt,
    name="to_prompt",
    description="Compose prompt from {topic, style, audience?}. Returns: str.",
    type="python",
    source="local",
)

agent = Agent(
    name="Writer",
    description="Concise writing assistant.",
    llm_engine=OpenAIEngine("gpt-4o-mini"),
    role_prompt="You are a concise writing assistant.",
    pre_invoke=pre_invoke_tool,
)


# ---------- MCP server URL (normalized to /mcp if path is empty) ----------

SERVER_URL = "http://127.0.0.1:8000"   # we'll normalize to /mcp if needed
SERVER_NAME = "Demo Mathematics Server"

def normalize_streamable_http(url: str) -> str:
    parts = urlparse(url)
    if not parts.path or parts.path == "/":
        parts = parts._replace(path="/mcp")
    return urlunparse(parts)


# ---------- Helpers ----------

def show_plan(t: Tool) -> None:
    meta = t.to_dict()
    print(f"\n-- {t.full_name} --")
    print("type/source:", t.type, "/", t.source)
    print("signature:", meta["signature"])
    print("required: ", sorted(meta["required_names"]))
    print("params:   ", meta["p_or_kw_names"])

def invoke_with_inputs(t: Tool, inputs: Mapping[str, Any]) -> None:
    print("inputs:", inputs)
    try:
        result = t.invoke(dict(inputs))
        print("result:", result)
    except ToolInvocationError as e:
        print("invoke error:", e)


def synthesize_required_inputs(t: Tool) -> dict:
    """Create minimal inputs for required params using annotations."""
    meta = t.to_dict()
    required = set(meta["required_names"])
    out: dict[str, Any] = {}
    for p in meta["p_or_kw_names"]:
        if p in required:
            ann = t.arguments_map[p]["ann"]
            if ann is float:
                out[p] = 0.0
            elif ann is int:
                out[p] = 0
            elif ann is bool:
                out[p] = False
            elif ann is list or getattr(getattr(ann, "__origin__", None), "__name__", "") == "list":
                out[p] = []
            else:
                out[p] = ""
    return out


# ---------- Demo flow ----------

def main() -> None:
    # 1) Agent -> AgentTool
    agent_tools = toolify(agent)
    agent_tool = agent_tools[0]
    show_plan(agent_tool)
    invoke_with_inputs(agent_tool, {"topic": "unit testing", "style": "concise", "audience": "beginners"})

    # 2) Callable -> Tool (requires name & description)
    callable_tools = toolify(add_scale, name="add_scale", description="(a + b) * scale.")
    callable_tool = callable_tools[0]
    show_plan(callable_tool)
    invoke_with_inputs(callable_tool, {"a": 2, "b": 3, "scale": 10})

    # 3) Pre-wrapped Tool -> passthrough
    passthrough_tools = toolify(pre_wrapped_tool)
    passthrough_tool = passthrough_tools[0]
    show_plan(passthrough_tool)
    invoke_with_inputs(passthrough_tool, {"text": "Atomic-Agentic is awesome for strict schema tools!", "limit": 28})

    # 4) MCP URL -> list of MCPProxyTools
    resolved = normalize_streamable_http(SERVER_URL)
    print("\n[MCP] Connecting to:", resolved)
    try:
        mcp_tools = toolify(resolved, server_name=SERVER_NAME)
        print(f"[MCP] Discovered {len(mcp_tools)} tools")
        # Provide canned examples for common demo names; synthesize otherwise
        EXAMPLE_INPUTS = {
            "mul":        {"a": 3, "b": 4},
            "multiply":   {"a": 3, "b": 4},
            "power":      {"base": 2, "exponent": 8},
            "factorial":  {"n": 5},
            "derivative": {"func": "2.71828**x + 2*x + 1", "x": 3.0},
        }
        for t in mcp_tools:
            show_plan(t)
            inputs = EXAMPLE_INPUTS.get(t.name, None) or synthesize_required_inputs(t)
            if inputs == {}:
                print("(no required params detected; calling with empty inputs)")
            invoke_with_inputs(t, inputs)
    except Exception as e:
        print("[MCP] Skipping MCP demo due to error:", e)


if __name__ == "__main__":
    main()
