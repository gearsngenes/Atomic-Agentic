from __future__ import annotations

import logging

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from starlette.applications import Starlette

from atomic_agentic.a2a import A2AtomicExecutor
from atomic_agentic.constants.a2a_sdk import TRANSPORT_JSON_RPC
from atomic_agentic.adapters import to_tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prebuilt_tools import BASIC_MATH_TOOLS, EXPONENT_TOOLS, STAT_TOOLS

HOST = "127.0.0.1"
PORT = 9000
BASE_URL = f"http://{HOST}:{PORT}"

# Plain functions are converted with to_tool on the fly here -- A2AtomicExecutor requires
# real AtomicInvokable instances (calls .to_agent_card() directly, no
# to_tool step of its own), unlike Agent.register_tools which normalizes a
# bare callable itself. A subset of the math tools is exposed, picked for
# parameter variety: two floats (add, multiply), one float (sqrt), and a
# sequence of floats (mean). They live in three separate lists, so all three
# are searched.
SKILL_NAMES = {"add", "multiply", "sqrt", "mean"}


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    skills = [
        to_tool(fn)
        for fn in (*BASIC_MATH_TOOLS, *EXPONENT_TOOLS, *STAT_TOOLS)
        if fn.__name__ in SKILL_NAMES
    ]
    executor = A2AtomicExecutor(skills)

    card = executor.to_agent_card(
        BASE_URL,
        name="a2a_sdk_atomic_host",
        description=(
            "A2AtomicExecutor host exposing math Tools as Atomic skills. "
            "Reachable through A2AProxyTool's skill mode (name=<skill name>) "
            "as well as its generic mode (name=None), provided the caller "
            "supplies the skill_id routing metadata by hand."
        ),
        transport_mode=TRANSPORT_JSON_RPC,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(request_handler, rpc_url="/"),
        ]
    )

    print(f"Serving Atomic skills {sorted(SKILL_NAMES)!r} at {BASE_URL}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
