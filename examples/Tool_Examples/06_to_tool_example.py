from __future__ import annotations

import json
from typing import Any, Mapping

from dotenv import load_dotenv

from atomic_agentic.a2a import A2AClientHub
from atomic_agentic.agents import BasicAgent
from atomic_agentic.exceptions import ToolInvocationError
from atomic_agentic.constants.a2a_sdk import TRANSPORT_JSON_RPC
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.llm import OpenAIEngine
from atomic_agentic.mcp import MCPClientHub
from atomic_agentic.adapters import to_tools, to_tool
from atomic_agentic.tools import Tool

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
    description="Truncate text to a character limit (default limit: 40).",
)


def to_prompt(topic: str, style: str, *, audience: str = "general") -> str:
    """Compose a small prompt string from structured inputs."""
    return f"Write about '{topic}' in a {style} style for {audience} readers."


pre_invoke_tool = Tool(
    function=to_prompt,
    name="to_prompt",
    description="Compose a prompt from {topic, style, audience?}. Returns a string prompt.",
)

agent = BasicAgent(
    name="Writer",
    description="Concise writing assistant.",
    llm_engine=OpenAIEngine("gpt-4o-mini"),
    instructions="You are a concise writing assistant.",
    pre_invoke=pre_invoke_tool,
)


MCP_ENDPOINT = "http://127.0.0.1:8000/mcp"
MCP_HEADERS: Mapping[str, str] | None = None

# a2a-sdk-backed targets -- see a2a_sdk_atomic_host_server.py (invokable,
# publishes Atomic skills) and a2a_sdk_foreign_host_server.py (plain a2a-sdk
# agent, no skills -- generic mode only) in this same directory.
A2A_SDK_ATOMIC_URL = "http://127.0.0.1:9000"
A2A_SDK_FOREIGN_URL = "http://127.0.0.1:9001"


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
        print(f"  {_jsonable(result.result)!r}")
    except ToolInvocationError as exc:
        print("Invocation error:")
        print(f"  {exc}")
    except Exception as exc:
        print("Error:")
        print(f"  {exc}")


def main() -> None:
    print("\n[1] Agent -> Tool via to_tool(agent) — delegates by reference")
    agent_tool = to_tool(agent)
    show_plan(agent_tool)
    invoke_with_inputs(
        agent_tool,
        {"topic": "unit testing", "style": "concise", "audience": "beginners"},
    )

    print("\n[2] Callable -> Tool via to_tool(add_scale, ...)")
    callable_tool = to_tool(
        add_scale,
        name="add_scale",
        description="Compute (a + b) * scale.",
    )
    show_plan(callable_tool)
    invoke_with_inputs(callable_tool, {"a": 2, "b": 3, "scale": 10})

    print("\n[3] Tool -> new Tool instance wrapping the same function, with overrides applied")
    updated_tool = to_tool(
        pre_wrapped_tool,
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
            persistent=False,
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
            mcp_tool = to_tool(
                mcp_hub,
                remote_name=remote_name,
            )
            show_plan(mcp_tool)
            invoke_with_inputs(mcp_tool, example_inputs_by_remote_name.get(remote_name))

    except Exception as exc:
        print("[MCP] Skipping MCP demo due to error:", exc)

    print("\n[5] A2AClientHub + remote_name -> A2AProxyTool (skill mode) via to_tool(hub, remote_name=...)")
    print("[A2A-sdk] Connecting to:", A2A_SDK_ATOMIC_URL)
    try:
        atomic_hub = A2AClientHub(A2A_SDK_ATOMIC_URL, TRANSPORT_JSON_RPC, persistent=False)
        skills = atomic_hub.get_atomic_skills()
        print(f"[A2A-sdk] Discovered {len(skills)} Atomic skill(s): {sorted(skills.keys())}")

        if skills:
            skill_id = "add" if "add" in skills else next(iter(skills))
            skill_tool = to_tool(atomic_hub, remote_name=skill_id)
            show_plan(skill_tool)
            invoke_with_inputs(skill_tool, {"a": 12, "b": 5})

    except Exception as exc:
        print("[A2A-sdk] Skipping skill-mode demo due to error:", exc)

    print("\n[6] A2AClientHub, remote_name omitted -> A2AProxyTool (generic mode) via to_tool(hub)")
    print("[A2A-sdk] Connecting to:", A2A_SDK_FOREIGN_URL)
    try:
        foreign_hub = A2AClientHub(A2A_SDK_FOREIGN_URL, TRANSPORT_JSON_RPC, persistent=False)
        print(f"[A2A-sdk] {foreign_hub.agent_card.name!r} publishes "
              f"{len(foreign_hub.get_atomic_skills())} Atomic skill(s) -- not invokable, generic mode only")

        generic_tool = to_tool(foreign_hub)
        show_plan(generic_tool)
        invoke_with_inputs(
            generic_tool,
            {
                "parts": [
                    {"text": "hello there", "data": None, "raw_b64": None, "url": None, "filename": None, "media_type": None},
                ],
            },
        )

    except Exception as exc:
        print("[A2A-sdk] Skipping generic-mode demo due to error:", exc)

    print("\n[7] to_tools([A2AClientHub]) -> one proxy per Atomic skill + one trailing generic proxy")
    try:
        atomic_hub = A2AClientHub(A2A_SDK_ATOMIC_URL, TRANSPORT_JSON_RPC, persistent=False)
        a2a_sdk_tools = to_tools(
            [atomic_hub],
        )
        print(f"[A2A-sdk] to_tools produced {len(a2a_sdk_tools)} tool(s):")
        for t in a2a_sdk_tools:
            print(f"  - {t.full_name} (skill_id={t.skill_id!r})")

    except Exception as exc:
        print("[A2A-sdk] Skipping batch A2A-sdk demo due to error:", exc)

    print("\n[8] @Tool.of -> decorator form of the Tool constructor; the name is bound to a Tool")

    @Tool.of
    def shout(text: str) -> str:
        """Upper-case the text."""
        return text.upper()

    @Tool.of(name="join_words", description="Join words with a separator.")
    def join(words: list[str], sep: str = " ") -> str:
        return sep.join(words)

    # Note: a method loses `self` binding (Tool is not a descriptor); decorate
    # module-level functions, or bind the method first.
    for t in (shout, join):
        print(f"  - {t.name}: {t.description}")
    print("  shout ->", shout.invoke({"text": "hello"}).result)
    print("  join_words ->", join.invoke({"words": ["a", "b"], "sep": "-"}).result)


if __name__ == "__main__":
    main()
