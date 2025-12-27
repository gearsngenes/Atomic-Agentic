"""
AgentTool demo: wrapping an Agent as a Tool with a schema taken from the Agent's pre_invoke Tool.

Adjust the import path to your project layout if needed:
from atomic_agentic.Agents import AgentTool
"""
from dotenv import load_dotenv
from atomic_agentic.tools import Tool
from atomic_agentic.tools.invokable import AgentTool
from atomic_agentic.core.Exceptions import ToolInvocationError
from atomic_agentic.agents.toolagents import Agent
from atomic_agentic.engines.LLMEngines import OpenAIEngine
import json

load_dotenv()  # take environment variables from .env file (if exists)

import logging

logging.basicConfig(level = logging.INFO)

# --- 2) Define a richer pre-invoke Tool (schema mirrors desired Agent inputs) ---

def to_prompt(topic: str, style: str, *, audience: str = "general") -> str:
    """Compose a natural-language prompt from structured inputs."""
    return f"Write about '{topic}' in a {style} style for {audience} readers."

# --- 3) Build the Agent and wrap it as an AgentTool ---

agent = Agent(
    name="Writer",
    description="Helpful writing assistant.",
    llm_engine=OpenAIEngine(model="gpt-4o-mini"),
    role_prompt="You are a concise writing assistant.",
    pre_invoke=to_prompt,  # <-- schema source for AgentTool
)

agent_tool = AgentTool(agent)  # type="agent", source=agent.name, name="invoke"


# --- 4) Utility helpers for inspection & runs (same vibe as 01_Tool.py) ---

def show_plan(tool: Tool) -> None:
    meta = tool.to_dict()
    print(f"\n-- {tool.full_name} call plan --")
    print("signature:", meta["signature"])
    print("argument map:", json.dumps(tool.arguments_map, indent = 2))

def run_case(label: str, tool: Tool, inputs: dict) -> None:
    print(f"\n=== {label} ===")
    print("inputs:", inputs)
    try:
        result = tool.invoke(inputs)
        print("OK:", result)
    except ToolInvocationError as e:
        print("ERR:", e)


# --- 5) Demo cases ---

if __name__ == "__main__":
    # Inspect the binding plan mirrored from agent.pre_invoke
    show_plan(agent_tool)

    # Happy paths
    run_case(
        "invoke: minimal (required only)",
        agent_tool,
        {"topic": "unit testing", "style": "pragmatic"},
    )
    run_case(
        "invoke: with kw-only audience",
        agent_tool,
        {"topic": "memory safety", "style": "tutorial", "audience": "beginners"},
    )

    # Common mistakes to see strict errors
    run_case(
        "invoke: missing required 'topic'",
        agent_tool,
        {"style": "formal", "audience": "execs"},
    )
    run_case(
        "invoke: unknown key",
        agent_tool,
        {"topic": "refactoring", "style": "guide", "extra": 123},
    )
