"""
Base Tool demo: wrapping an Agent directly as a Tool.

This mirrors the deprecated AdapterTool from v1.x example, but uses the v1.4-style base Tool support
for AtomicInvokable objects.
"""

from dotenv import load_dotenv
from atomic_agentic.tools import Tool
from atomic_agentic.core.Exceptions import ToolInvocationError, AgentInvocationError
from atomic_agentic.agents import Agent
from atomic_agentic.engines.LLMEngines import OpenAIEngine
import logging

load_dotenv()  # take environment variables from .env file (if exists)

logging.basicConfig(level=logging.INFO)


# --- 1) Define a richer pre-invoke Tool/callable
# This schema becomes the Agent's input schema, which base Tool will reuse
# when wrapping the Agent as an AtomicInvokable.

def to_prompt(topic: str, style: str, *, audience: str = "general") -> str:
    """Compose a natural-language prompt from structured inputs."""
    return f"Write about '{topic}' in a {style} style for {audience} readers."


# --- 2) Build the Agent ---

agent = Agent(
    name="Writer",
    description="Helpful writing assistant.",
    llm_engine=OpenAIEngine(model="gpt-4o-mini"),
    role_prompt="You are a concise writing assistant.",
    pre_invoke=to_prompt,
)


# --- 3) Wrap the Agent directly with base Tool ---

agent_tool = Tool(
    function=agent,
    name="writer_tool",
    namespace="agent_wrapped",
    description="Base Tool wrapping the Writer Agent directly.",
)

assert agent_tool.wraps_invokable is True


# --- 4) Utility helpers for inspection & runs ---

def show_plan(tool: Tool) -> None:
    meta = tool.to_dict()

    print(f"\n-- {tool.full_name} call plan --")
    print("wraps_invokable:", tool.wraps_invokable)
    print("signature:", tool.signature)
    print("parameters:")

    for param in tool.parameters:
        default_str = (
            "(no default)"
            if param.default.__class__.__name__ == "NO_VAL"
            else f"default={param.default}"
        )
        print(f"  {param.name}: {param.kind}, type={param.type}, {default_str}")

    print("metadata:")
    print("  namespace:", meta.get("namespace"))
    print("  wraps_invokable:", meta.get("wraps_invokable"))
    print("  has invokable_function:", "invokable_function" in meta)


def run_case(label: str, tool: Tool, inputs: dict) -> None:
    print(f"\n=== {label} ===")
    print("inputs:", inputs)

    try:
        result = tool.invoke(inputs)
        print("OK:", result)
    except (ToolInvocationError, ValueError, TypeError, AgentInvocationError) as e:
        print("ERR:", e)


# --- 5) Demo cases ---

if __name__ == "__main__":
    # Inspect the binding plan mirrored from the Agent's declared schema.
    show_plan(agent_tool)

    # Happy paths
    run_case(
        "base Tool wrapped Agent: minimal required inputs",
        agent_tool,
        {"topic": "unit testing", "style": "pragmatic"},
    )

    run_case(
        "base Tool wrapped Agent: with kw-only audience",
        agent_tool,
        {
            "topic": "memory safety",
            "style": "tutorial",
            "audience": "beginners",
        },
    )

    # Common mistakes to see strict errors.
    # This version disables filtering so unknown keys surface as binding errors.
    agent_tool_strict = Tool(
        function=agent,
        name="strict_writer_tool",
        namespace="agent_wrapped",
        description="Strict base Tool wrapping the Writer Agent directly.",
        filter_extraneous_inputs=False,
    )

    run_case(
        "base Tool wrapped Agent: missing required 'topic'",
        agent_tool_strict,
        {"style": "formal", "audience": "execs"},
    )

    run_case(
        "base Tool wrapped Agent: unknown key",
        agent_tool_strict,
        {"topic": "refactoring", "style": "guide", "extra": 123},
    )