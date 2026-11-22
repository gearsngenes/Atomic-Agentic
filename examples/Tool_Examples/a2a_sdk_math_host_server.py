"""
An A2AtomicExecutor host exposing one PlanActAgent as an Atomic skill,
"MathPlannerAgent", on port 7000. The agent solves problems with the local
math tools.

Start it first, in its own terminal:
    python a2a_sdk_math_host_server.py
then reach it from another script (see 06_A2AProxyTool.py for the pattern):
    hub = A2AClientHub("http://127.0.0.1:7000", TRANSPORT_JSON_RPC, persistent=False)
    tool = toolify(hub, remote_name="MathPlannerAgent")
Also the math backend for a2a_sdk_inter_agent_host_server.py.
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

from atomic_agentic.a2a import A2AtomicExecutor
from atomic_agentic.agents import PlanActAgent
from atomic_agentic.constants.a2a_sdk import TRANSPORT_JSON_RPC
from atomic_agentic.llm import OpenAIEngine

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prebuilt_tools import BASIC_MATH_TOOLS

load_dotenv()

HOST = "127.0.0.1"
PORT = 7000
BASE_URL = f"http://{HOST}:{PORT}"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    llm = OpenAIEngine(model="gpt-4o-mini")

    math_agent = PlanActAgent(
        name="MathPlannerAgent",
        description="A tool-using planner that solves problems with the local math tools.",
        llm_engine=llm,
        record_enabled=False,
        tool_calls_limit=12,
    )
    math_agent.register_tools(BASIC_MATH_TOOLS, name_collision_policy="raise")

    executor = A2AtomicExecutor([math_agent])
    card = executor.to_agent_card(
        BASE_URL,
        name="math_host",
        description="A2AtomicExecutor host exposing the MathPlannerAgent invokable.",
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

    print(f"Serving the MathPlannerAgent skill at {BASE_URL}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
