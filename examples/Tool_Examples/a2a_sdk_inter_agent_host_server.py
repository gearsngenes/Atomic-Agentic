"""
An A2AtomicExecutor host exposing a PlanActAgent, "InterAgentPlanner", on
port 8000. The planner's tools are two A2AClientHubs, one per remote agent, so
each remote agent's Atomic skill becomes a tool it can plan with.

Start the backends first, each in its own terminal, in this order:
    python a2a_sdk_trivia_host_server.py
    python a2a_sdk_math_host_server.py
    python a2a_sdk_inter_agent_host_server.py
Each hub resolves its agent card when it is constructed, so both backends must
already be serving when this file starts.
"""
from __future__ import annotations

import logging

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from dotenv import load_dotenv
from starlette.applications import Starlette

from atomic_agentic.a2a import A2AClientHub, A2AtomicExecutor
from atomic_agentic.agents import PlanActAgent
from atomic_agentic.constants.a2a_sdk import TRANSPORT_JSON_RPC
from atomic_agentic.llm import OpenAIEngine

load_dotenv()

HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"

TRIVIA_URL = "http://127.0.0.1:6000"
MATH_URL = "http://127.0.0.1:7000"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    llm = OpenAIEngine(model="gpt-4o-mini")

    planner = PlanActAgent(
        name="InterAgentPlanner",
        description="A planner that orchestrates remote trivia and math agents through A2AProxyTool proxies.",
        llm_engine=llm,
        record_enabled=False,
        tool_calls_limit=16,
    )

    trivia_hub = A2AClientHub(TRIVIA_URL, TRANSPORT_JSON_RPC, persistent=False)
    math_hub = A2AClientHub(MATH_URL, TRANSPORT_JSON_RPC, persistent=False)

    planner.register_tools([trivia_hub], name_collision_policy="raise")
    planner.register_tools([math_hub], name_collision_policy="raise")

    executor = A2AtomicExecutor([planner])
    card = executor.to_agent_card(
        BASE_URL,
        name="inter_agent_host",
        description="A2AtomicExecutor host exposing the InterAgentPlanner invokable.",
        version="1.0.0",
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

    print(f"Serving the InterAgentPlanner skill at {BASE_URL}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
