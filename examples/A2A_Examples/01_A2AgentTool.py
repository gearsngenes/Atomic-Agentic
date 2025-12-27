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

from atomic_agentic.tools.a2a import A2AgentTool


DEFAULT_TARGETS: Dict[str, Dict[str, Any]] = {
    "trivia": {
        "url": "http://localhost:6000",
        "prompt": "Give me one interesting trivia fact about octopuses.",
    },
    "math": {
        "url": "http://localhost:7000",
        "prompt": "Compute (12.5 * 3) + sqrt(81) and return only the final number.",
    },
    "inter": {
        "url": "http://localhost:8000",
        "prompt": (
            "Use your remote agents to: "
            "1) get one fun trivia fact about honeybees, "
            "2) compute 18^2 - 10, "
            "then return a dictionary with keys 'fact' and 'math'."
        ),
    },
}


def main() -> None:
    target = "math"
    url = DEFAULT_TARGETS[target]["url"]
    inputs = {"prompt": DEFAULT_TARGETS[target]["prompt"]}

    tool = A2AgentTool(url=url)

    print("\n=== A2A TOOL METADATA ===")
    print("full_name :", tool.full_name)
    print("return_type:", tool.return_type)
    print("arguments_map:", tool.arguments_map)

    print("\n=== INVOKE ===")
    result = tool.invoke(inputs)
    print(result, "(type:", type(result), ")")


if __name__ == "__main__":
    main()
