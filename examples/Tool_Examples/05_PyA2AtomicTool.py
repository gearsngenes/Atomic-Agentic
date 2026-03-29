from __future__ import annotations

import argparse
import json
from typing import Any, Mapping

from atomic_agentic.a2a.PyA2AtomicClient import PyA2AtomicClient
from atomic_agentic.core.sentinels import NO_VAL
from atomic_agentic.tools.a2a import PyA2AtomicTool


DEFAULT_TARGETS: dict[str, dict[str, Any]] = {
    "trivia": {
        "url": "http://localhost:6000",
        "remote_name": "TriviaAgent",
        "inputs": {
            "prompt": "Give me one interesting trivia fact about octopuses.",
        },
    },
    "math": {
        "url": "http://localhost:7000",
        "remote_name": "MathPlannerAgent",
        "inputs": {
            "prompt": "Compute (12.5 * 3) + sqrt(81) and return only the final number.",
        },
    },
    "inter": {
        "url": "http://localhost:8000",
        "remote_name": "InterAgentPlanner",
        "inputs": {
            "prompt": (
                "Use your remote agents to: "
                "1) get one fun trivia fact about honeybees, "
                "2) compute 18^2 - 10, "
                "then return a dictionary with keys 'fact' and 'math'."
            ),
        },
    },
}


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def show_discovery(client: PyA2AtomicClient) -> dict[str, dict[str, Any]]:
    print(f"\n=== HOST DISCOVERY @ {client.url} ===")
    print("agent_card.name       :", getattr(client.agent_card, "name", None))
    print("agent_card.description:", getattr(client.agent_card, "description", None))

    invokables = client.list_invokables()
    if not invokables:
        print("No invokables discovered.")
        return invokables

    print("discovered invokables :")
    for remote_name, meta in invokables.items():
        print(f"  - {remote_name}: {meta.get('description')}")
    return invokables


def show_plan(tool: PyA2AtomicTool) -> None:
    print(f"\n=== TOOL PROXY ===")
    print("full_name   :", tool.full_name)
    print("remote_name :", tool.remote_name)
    print("namespace   :", tool.namespace)
    print("description :", tool.description)
    print("signature   :", tool.signature)
    print("return_type :", tool.return_type)
    print("parameters  :")
    for param in tool.parameters:
        default_text = "(no default)" if param.default is NO_VAL else f"default={_jsonable(param.default)!r}"
        print(
            f"  - {param.name}: kind={param.kind}, type={param.type}, {default_text}"
        )


def resolve_remote_name(
    discovered: Mapping[str, dict[str, Any]],
    configured_remote_name: str | None,
    explicit_remote_name: str | None,
) -> str:
    if explicit_remote_name:
        remote_name = explicit_remote_name.strip()
    elif configured_remote_name:
        remote_name = configured_remote_name.strip()
    elif len(discovered) == 1:
        remote_name = next(iter(discovered.keys()))
    else:
        raise ValueError(
            "remote_name was not provided and this host exposes multiple invokables."
        )

    if remote_name not in discovered:
        raise KeyError(
            f"Requested remote_name {remote_name!r} was not found. "
            f"Available: {sorted(discovered.keys())!r}"
        )

    return remote_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a PyA2AtomicHost and proxy one remote invokable."
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="math",
        choices=sorted(DEFAULT_TARGETS.keys()),
        help="Named demo target.",
    )
    parser.add_argument("--url", help="Override the target URL.")
    parser.add_argument(
        "--remote-name",
        help="Override the target remote invokable name.",
    )
    args = parser.parse_args()

    target_cfg = DEFAULT_TARGETS[args.target]
    url = args.url or target_cfg["url"]

    client = PyA2AtomicClient(url=url)
    discovered = show_discovery(client)

    remote_name = resolve_remote_name(
        discovered=discovered,
        configured_remote_name=target_cfg.get("remote_name"),
        explicit_remote_name=args.remote_name,
    )

    tool = PyA2AtomicTool(
        remote_name=remote_name,
        client=client,
    )
    show_plan(tool)

    inputs = dict(target_cfg.get("inputs", {}))
    print("\n=== INVOKE ===")
    print("inputs:", json.dumps(inputs, indent=2))
    result = tool.invoke(inputs)
    print("result:", _jsonable(result))
    print("type  :", type(result).__name__)


if __name__ == "__main__":
    main()
