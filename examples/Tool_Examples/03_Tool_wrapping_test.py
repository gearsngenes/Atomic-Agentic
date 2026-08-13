"""
Base Tool demo: wrapping an Agent directly as a Tool.

``Tool`` treats any ``AtomicInvokable`` as a naturally well-behaved plain
callable — no special wrapper type, no explicit "this is an invokable" flag.
Schema comes straight from the wrapped Agent's own declared ``parameters``/
``return_type`` (via ``extract_io``); name/description default to the
Agent's own ``.name``/``.description`` (kept in sync as ``__name__``/
``__doc__`` by the base class). Invoking the resulting Tool calls the Agent
through its own ``__call__``, which returns the unwrapped text response
directly — Tool re-wraps that into its own ``ToolResult`` envelope.
"""

from dotenv import load_dotenv
from atomic_agentic.tools import Tool
from atomic_agentic.exceptions import ToolInvocationError, AgentInvocationError
from atomic_agentic.agents import BasicAgent
from atomic_agentic.llm import OpenAIEngine
from atomic_agentic.constants.core import NO_VAL
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

agent = BasicAgent(
    name="Writer",
    namespace="agent_wrapped",
    description="Helpful writing assistant.",
    llm_engine=OpenAIEngine(model="gpt-4o-mini"),
    role_prompt="You are a concise writing assistant.",
    pre_invoke=to_prompt,
)


# --- 3) Wrap the Agent directly with base Tool ---

agent_tool = Tool(
    function=agent,
    name="writer_tool",
    description="Base Tool wrapping the Writer Agent directly.",
)

assert agent_tool.function is agent  # delegates by reference, doesn't copy/mutate the Agent


# --- 4) Utility helpers for inspection & runs ---

def show_plan(tool: Tool) -> None:
    meta = tool.to_dict()

    print(f"\n-- {tool.full_name} call plan --")
    print("wraps:", type(tool.function).__name__)
    print("signature:", tool.signature)
    print("parameters:")

    for param in tool.parameters:
        default_str = (
            "(no default)"
            if param.default is NO_VAL
            else f"default={param.default}"
        )
        print(f"  {param.name}: {param.kind}, type={param.type}, {default_str}")

    print("metadata:")
    print("  namespace:", meta.get("namespace"))
    print("  module:", meta.get("module"))
    print("  qualname:", meta.get("qualname"))


def run_case(label: str, tool: Tool, inputs: dict) -> None:
    print(f"\n=== {label} ===")
    print("inputs:", inputs)

    try:
        result = tool.invoke(inputs)
        print("OK:", result.result)
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

    # Common mistakes.
    run_case(
        "base Tool wrapped Agent: missing required 'topic'",
        agent_tool,
        {"style": "formal", "audience": "execs"},
    )

    run_case(
        "base Tool wrapped Agent: unknown key is silently filtered",
        agent_tool,
        {"topic": "refactoring", "style": "guide", "extra": 123},
    )
