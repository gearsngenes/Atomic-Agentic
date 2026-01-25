"""
A2A Proxy Client
----------------
One script to test ANY running A2A-hosted agent/tool-agent via A2AgentTool.

Usage examples:
  python a2a_proxy_client.py trivia
  python a2a_proxy_client.py math
  python a2a_proxy_client.py inter
  python a2a_proxy_client.py --url http://localhost:6000 --prompt "Tell me something about octopuses."
"""

from typing import Any, Dict

from atomic_agentic.tools import A2AProxyTool


DEFAULT_TARGETS: Dict[str, Dict[str, Any]] = {
    "trivia": {
        "url": "http://localhost:6000",
        "inputs":{"prompt": "Give me one interesting trivia fact about octopuses."},
    },
    "math": {
        "url": "http://localhost:7000",
        "inputs": {"prompt": "Compute (12.5 * 3) + sqrt(81) and return only the final number."},
    },
    "inter": {
        "url": "http://localhost:8000",
        "inputs":{"prompt": (
            "Use your remote agents to: "
            "1) get one fun trivia fact about honeybees, "
            "2) compute 18^2 - 10, "
            "then return a dictionary with keys 'fact' and 'math'."
        )},
    },
}


def show_plan(tool: A2AProxyTool) -> None:
    print(f"\n-- {tool.full_name} call plan --")
    print("signature:", tool.signature)
    print("return_type:", tool.return_type)
    print("parameters:")
    for param in tool.parameters:
        default_str = "(no default)" if param.default.__class__.__name__ == "NO_VAL" else f"default={param.default}"
        print(f"  {param.name}: {param.kind}, type={param.type}, {default_str}")


def main() -> None:
    target = "math"
    url = DEFAULT_TARGETS[target]["url"]
    inputs = DEFAULT_TARGETS[target]["inputs"]

    tool = A2AProxyTool(url=url)

    show_plan(tool)

    print("\n=== INVOKE ===")
    result = tool.invoke(inputs)
    print(result, "(type:", type(result), ")")


if __name__ == "__main__":
    main()
