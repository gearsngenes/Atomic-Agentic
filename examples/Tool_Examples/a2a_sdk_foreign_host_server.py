from __future__ import annotations

import logging

import uvicorn
from a2a.helpers import new_data_part, new_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Part, Role
from google.protobuf.json_format import MessageToDict
from starlette.applications import Starlette

from atomic_agentic.constants.a2a_sdk import TRANSPORT_JSON_RPC

HOST = "127.0.0.1"
PORT = 9001
BASE_URL = f"http://{HOST}:{PORT}"


class ShoutBackExecutor(AgentExecutor):
    """
    A plain, non-Atomic a2a-sdk agent -- its AgentCard carries no
    Atomic-Agentic param-schema extension, so it is only reachable through
    A2AProxyTool's generic mode (skill_id=None), never skill mode.

    Transforms whatever Parts it receives instead of a pure echo, so a
    generic-mode caller can see its input actually got processed remotely:
    text Parts come back upper-cased, data Parts come back annotated with
    their own sorted key list, and raw/url Parts pass through unchanged
    (nothing meaningful to transform). Any inbound message metadata is
    echoed back as one extra data Part, so generic mode's metadata
    passthrough is directly visible in the reply.
    """

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        reply_parts: list[Part] = []
        for part in context.message.parts:
            if part.HasField("text"):
                reply_parts.append(new_text_part(part.text.upper()))
            elif part.HasField("data"):
                payload = MessageToDict(part.data)
                keys = sorted(payload.keys()) if isinstance(payload, dict) else []
                reply_parts.append(new_data_part({"received": payload, "keys": keys}))
            else:
                reply_parts.append(part)

        if context.metadata:
            reply_parts.append(new_data_part({"received_metadata": dict(context.metadata)}))

        await event_queue.enqueue_event(new_message(reply_parts, role=Role.ROLE_AGENT))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    card = AgentCard(
        name="a2a_sdk_foreign_host",
        description=(
            "A plain a2a-sdk agent with no Atomic-Agentic extension -- "
            "reachable only through A2AProxyTool's generic mode. Upper-cases "
            "text Parts and annotates data Parts with their key list."
        ),
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="shout_back",
                name="shout_back",
                description="Transforms whatever Parts it receives.",
            )
        ],
        capabilities=AgentCapabilities(),
        supported_interfaces=[AgentInterface(url=BASE_URL, protocol_binding=TRANSPORT_JSON_RPC)],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=ShoutBackExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(request_handler, rpc_url="/"),
        ]
    )

    print(f"Serving foreign (non-Atomic) A2A agent at {BASE_URL}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
