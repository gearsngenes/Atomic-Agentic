# examples/Tool_Examples/06_A2AProxyTool.py
"""
A2AProxyTool over the a2a-sdk-backed A2AClientHub -- skill (invokable) mode
vs. generic (send-parts) mode, against two separately-run servers:

- a2a_sdk_atomic_host_server.py (port 9000): an A2AtomicExecutor host that
  publishes Atomic-Agentic's param-schema extension. Reachable through
  A2AProxyTool's skill mode (skill_id=<tool name>) with a fully typed,
  auto-derived signature -- no re-derivation needed on the client side.
- a2a_sdk_foreign_host_server.py (port 9001): a plain a2a-sdk agent with no
  Atomic-Agentic extension at all. Not invokable -- there is no skill_id to
  bind to -- so it is only reachable through generic mode's fixed
  (parts, metadata) signature.

Start both servers first, each in its own terminal:
    python a2a_sdk_atomic_host_server.py
    python a2a_sdk_foreign_host_server.py
then run this script separately.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from atomic_agentic.a2a import A2AClientHub
from atomic_agentic.constants.a2a_sdk import ATOMIC_RESULT_KEY, SKILL_ROUTING_KEY, TRANSPORT_JSON_RPC
from atomic_agentic import NO_VAL
from atomic_agentic.exceptions import ToolDefinitionError, ToolInvocationError
from atomic_agentic.tools import A2AProxyTool

ATOMIC_URL = "http://127.0.0.1:9000"
FOREIGN_URL = "http://127.0.0.1:9001"


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def show_plan(tool: A2AProxyTool) -> None:
    print(f"\n-- {tool.full_name} --")
    print("skill_id      :", tool.skill_id)
    print("transport_mode:", tool.transport_mode)
    print("base_url      :", tool.base_url)
    print("description   :", tool.description)
    print("signature     :", tool.signature)
    print("return_type   :", tool.return_type)
    print("parameters    :")
    for param in tool.parameters:
        default_text = "(no default)" if param.default is NO_VAL else f"default={_jsonable(param.default)!r}"
        print(f"  - {param.name}: kind={param.kind}, type={param.type}, {default_text}")


def invoke(tool: A2AProxyTool, inputs: Mapping[str, Any]) -> None:
    print("inputs:", json.dumps(dict(inputs), indent=2, default=str))
    try:
        result = tool.invoke(dict(inputs))
        print("result:", _jsonable(result.result))
    except (ToolDefinitionError, ToolInvocationError) as exc:
        print(f"{type(exc).__name__}:", exc)


def main() -> None:
    # ------------------------------------------------------------------ #
    # [1] Skill mode -- A2AtomicExecutor host, one proxy per discovered skill
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("[1] Skill mode against the Atomic host")
    atomic_hub = A2AClientHub(ATOMIC_URL, TRANSPORT_JSON_RPC, persistent=False)

    discovered = atomic_hub.get_atomic_skills()
    print(f"discovered {len(discovered)} Atomic skill(s): {sorted(discovered.keys())}")

    add_tool = A2AProxyTool(atomic_hub, skill_id="add")
    show_plan(add_tool)
    invoke(add_tool, {"a": 12, "b": 5})

    mean_tool = A2AProxyTool(atomic_hub, skill_id="mean")
    show_plan(mean_tool)
    invoke(mean_tool, {"nums": [4, 9, 16, 25]})

    print("\n[1b] Unknown skill_id -- caught at construction, not invocation")
    try:
        A2AProxyTool(atomic_hub, skill_id="does_not_exist")
    except ToolDefinitionError as exc:
        print("ToolDefinitionError:", exc)

    # ------------------------------------------------------------------ #
    # [2] Generic mode -- a plain, non-Atomic a2a-sdk agent (not invokable)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("[2] Generic mode against the foreign host (no skill_id to bind to)")
    foreign_hub = A2AClientHub(FOREIGN_URL, TRANSPORT_JSON_RPC, persistent=False)

    print(f"foreign host publishes {len(foreign_hub.get_atomic_skills())} Atomic skill(s) -- not invokable")

    send_parts_tool = A2AProxyTool(foreign_hub)
    show_plan(send_parts_tool)
    invoke(
        send_parts_tool,
        {
            "parts": [
                {"text": "hello there", "data": None, "raw_b64": None, "url": None, "filename": None, "media_type": None},
                {"text": None, "data": {"a": 1, "b": 2}, "raw_b64": None, "url": None, "filename": None, "media_type": None},
            ],
            "metadata": {"session": "demo-1"},
        },
    )

    # ------------------------------------------------------------------ #
    # [3] Generic mode against the Atomic host -- the relationship between
    #     the two modes: skill mode is a convenience wrapper over exactly
    #     this same generic call, with routing metadata and result
    #     unwrapping done for you.
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print("[3] Generic mode against the Atomic host -- manual skill routing")
    manual_tool = A2AProxyTool(atomic_hub)

    print("\n[3a] Missing routing metadata -- the executor has no skill to dispatch to")
    invoke(
        manual_tool,
        {
            "parts": [
                {"text": None, "data": {"a": 12, "b": 5}, "raw_b64": None, "url": None, "filename": None, "media_type": None},
            ],
            "metadata": None,
        },
    )

    print(f"\n[3b] {SKILL_ROUTING_KEY!r} metadata set by hand -- reproduces skill mode's 'add' call above")
    invoke(
        manual_tool,
        {
            "parts": [
                {"text": None, "data": {"a": 12, "b": 5}, "raw_b64": None, "url": None, "filename": None, "media_type": None},
            ],
            "metadata": {SKILL_ROUTING_KEY: "add"},
        },
    )
    print(
        f"(note: skill mode already unwraps the {ATOMIC_RESULT_KEY!r} key seen above -- "
        "generic mode leaves that to the caller)"
    )

    atomic_hub.close()
    foreign_hub.close()


if __name__ == "__main__":
    main()
