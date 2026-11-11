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
from atomic_agentic.tools import toolify

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prebuilt_tools import BASIC_MATH_TOOLS

HOST = "127.0.0.1"
PORT = 9000
BASE_URL = f"http://{HOST}:{PORT}"

# Plain functions are toolified on the fly here -- A2AtomicExecutor requires
# real AtomicInvokable instances (calls .to_agent_card() directly, no
# toolify step of its own), unlike Agent.register_tools which normalizes a
# bare callable itself. A subset of BASIC_MATH_TOOLS is exposed, picked for
# parameter variety: two floats (add), two floats (multiply). NOTE: "sqrt"
# and "mean" below are pre-existing names in SKILL_NAMES that were never
# actually in BASIC_MATH_TOOLS (sqrt/mean live in EXPONENT_TOOLS/STAT_TOOLS
# instead) -- a latent no-op in the filter below, left as found, not fixed
# here.
SKILL_NAMES = {"add", "multiply", "sqrt", "mean"}


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    skills = [
        toolify(fn, namespace="Basic_Math")
        for fn in BASIC_MATH_TOOLS
        if fn.__name__ in SKILL_NAMES
    ]
    executor = A2AtomicExecutor(skills)

    card = executor.to_agent_card(
        BASE_URL,
        name="a2a_sdk_atomic_host",
        description=(
            "A2AtomicExecutor host exposing math Tools as Atomic skills. "
            "Reachable through A2AProxyTool's skill mode (skill_id=<tool name>) "
            "as well as its generic mode (skill_id=None), provided the caller "
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
