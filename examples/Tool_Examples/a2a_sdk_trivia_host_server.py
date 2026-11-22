"""
An A2AtomicExecutor host exposing one BasicAgent as an Atomic skill,
"TriviaAgent", on port 6000.

Start it first, in its own terminal:
    python a2a_sdk_trivia_host_server.py
then reach it from another script (see 05_A2AProxyTool.py for the pattern):
    hub = A2AClientHub("http://127.0.0.1:6000", TRANSPORT_JSON_RPC, persistent=False)
    tool = to_tool(hub, remote_name="TriviaAgent")
Also the trivia backend for a2a_sdk_inter_agent_host_server.py.
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
from atomic_agentic.agents import BasicAgent
from atomic_agentic.constants.a2a_sdk import TRANSPORT_JSON_RPC
from atomic_agentic.llm import OpenAIEngine

load_dotenv()

HOST = "127.0.0.1"
PORT = 6000
BASE_URL = f"http://{HOST}:{PORT}"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    llm = OpenAIEngine(model="gpt-4o-mini")

    trivia_agent = BasicAgent(
        name="TriviaAgent",
        description="A concise trivia agent that returns one interesting fact about the user's topic.",
        llm_engine=llm,
        instructions=(
            "You are a concise trivia specialist. "
            "When given a prompt or topic, return one accurate and interesting trivia fact "
            "in 2-4 sentences."
        ),
        record_enabled=False,
    )

    executor = A2AtomicExecutor([trivia_agent])
    card = executor.to_agent_card(
        BASE_URL,
        name="trivia_host",
        description="A2AtomicExecutor host exposing the TriviaAgent invokable.",
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

    print(f"Serving the TriviaAgent skill at {BASE_URL}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
