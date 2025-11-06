# modules/A2Agents.py
from __future__ import annotations

"""
A2Agents
========

Schema-driven A2A adapters:

- A2AProxyAgent: client-side proxy that forwards a single `payload` mapping to a
  remote A2A agent using a function call and returns the RAW Message. The server
  responds with FunctionResponseContent so callers can safely read:
      raw.content.response["result"]

- A2AServerAgent: server-side adapter that exposes a local `seed` Agent over
  python-a2a. It dynamically defines a decorated A2A server class per-instance,
  but DELAYS instantiation until `run()` so we can pass a fully-qualified URL
  into the Agent Card (prevents "URL is required for A2A agent card" errors).

Message-level pattern (no task-level mixing)
--------------------------------------------
We use `handle_message(...)` on the server and return `Message(FunctionResponseContent(...))`
for function calls. This mirrors python-a2a's function-calling example.
"""

from typing import Any, Mapping, Dict, Optional

# Project-local imports
from .Agents import Agent
from .Tools import Tool

# python-a2a imports
from python_a2a import (
    A2AClient, A2AServer, run_server, agent,
    Message, MessageRole,
    FunctionCallContent, FunctionResponseContent, FunctionParameter,
    TextContent
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A2A_RESULT_KEY = "__py_A2A_result__"  # server uses this key inside FunctionResponseContent.response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_mapping(name: str, obj: Any) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise TypeError(f"{name} expects a Mapping[str, Any]")
    return obj  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------

class A2AProxyAgent(Agent):
    """
    A2AProxyAgent
    -------------
    Client-side proxy that forwards dict inputs to a remote A2A agent via a
    single function call: name="invoke", parameters=[("payload", <dict>)].

    Contract
    --------
    - .invoke(inputs: Mapping) -> Message
      Sends a FunctionCallContent and returns the RAW response Message.
      The server replies with FunctionResponseContent so your code can do:
          raw.content.response["result"]  # str or structured (server-defined)

    - Convenience:
        .invoke_response(inputs) -> Dict[str, Any] | None
        Returns the `content.response` dict if present, else None.

    Notes
    -----
    - We deliberately bypass the base Agent's pre-invoke tool here; the proxy's
      job is transport only.
    """

    def __init__(self, url: str, name: str = "A2AProxy", description: str = "Proxy to remote A2A agent"):
        # We still call Agent.__init__ to keep consistent metadata, but the engine
        # is never used by this proxy.
        super().__init__(name=name, description=description, llm_engine=None, role_prompt="", context_enabled=False)
        # Expect a FULL endpoint (e.g., "http://127.0.0.1:5000/a2a").
        self._client = A2AClient(url)

    # We override attach/detach to NO-OP, since the proxy doesn't hold files or context.
    def attach(self, *file_paths: str) -> None:  # type: ignore[override]
        return

    def detach(self, *file_paths: str) -> None:  # type: ignore[override]
        return

    def invoke(self, inputs: Mapping[str, Any]) -> Message:  # type: ignore[override]
        _ensure_mapping("A2AProxyAgent.invoke", inputs)

        # Typed FunctionParameter avoids `'dict' object has no attribute 'name'`.
        call = FunctionCallContent(
            name="invoke",
            parameters=[FunctionParameter(name="payload", value=dict(inputs))]
        )
        msg = Message(content=call, role=MessageRole.USER)
        resp = self._client.send_message(msg).content.response[A2A_RESULT_KEY]
        return resp

    # Optional discovery helpers (schema mirroring)
    def fetch_arguments_map(self) -> Optional[Dict[str, Any]]:
        call = FunctionCallContent(
            name="arguments_map",
            parameters=[]
        )
        msg = Message(content=call, role=MessageRole.USER)
        resp = self._client.send_message(msg)
        if getattr(resp.content, "type", None) == "function_response":
            return resp.content.response  # type: ignore[return-value]
        return None

    def fetch_agent_meta(self) -> Optional[Dict[str, Any]]:
        call = FunctionCallContent(name="agent_meta", parameters=[])
        msg = Message(content=call, role=MessageRole.USER)
        resp = self._client.send_message(msg)
        if getattr(resp.content, "type", None) == "function_response":
            return resp.content.response  # type: ignore[return-value]
        return None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class A2AServerAgent:
    """
    A2AServerAgent
    --------------
    Wraps a local `seed: Agent` as a python-a2a server using the message-level
    function-calling pattern. We define a dynamic subclass of `A2AServer` with the
    `@agent(...)` decorator, but we **instantiate it in `run()`** with the final
    `url`, to satisfy the Agent Card requirement.

    Exposed function names:
      - "invoke":         payload: Mapping[str, Any]  -> {result: <seed.invoke(...)>}
      - "arguments_map":  no params                  -> {arguments_map: <seed.pre_invoke.to_dict()["arguments_map"]>}
      - "agent_meta":     no params                  -> {name, description}
    """

    def __init__(
        self,
        seed: Agent,
        *,
        name: str | None = None,
        description: str | None = None,
        version: str = "1.0.0",
        host: str = "localhost",  # optional explicit card URL
        port: int = 5000,
    ):
        if not isinstance(seed, Agent):
            raise TypeError("A2AServerAgent requires a seed Agent.")
        self.seed = seed

        self._name = name
        self._description = description
        self._version = version
        self._host = host  # if provided, used verbatim in Agent Card
        self._port = port

        # runtime fields
        self._server_cls = None
        self._server = None
        self._scheme: str = "http"

        outer = self

        @agent(name=name,
               description=description,
               version=version,
               url=f"http://{self._host}:{self._port}"
               )
        class _Server(A2AServer):
            """Dynamic per-instance server. Instantiated in run(url=...)."""

            def handle_message(self, message: Message) -> Message:
                content = getattr(message, "content", None)
                ctype = getattr(content, "type", None)

                # Text: brief help
                if ctype == "text":
                    return Message(
                        content=TextContent(
                            text="Call as function_call: name in {'invoke','arguments_map','agent_meta'}."
                        ),
                        role=MessageRole.AGENT,
                        parent_message_id=message.message_id,
                        conversation_id=message.conversation_id
                    )

                # Function call dispatch
                if ctype == "function_call":
                    fn = content.name
                    params = {p.name: p.value for p in (content.parameters or [])}

                    try:
                        if fn == "invoke":
                            payload = params.get("payload", {})
                            _ensure_mapping("invoke.payload", payload)
                            result = outer.seed.invoke(payload)  # Agent returns str by contract
                            return Message(
                                content=FunctionResponseContent(
                                    name="invoke",
                                    response={A2A_RESULT_KEY: result}
                                ),
                                role=MessageRole.AGENT,
                                parent_message_id=message.message_id,
                                conversation_id=message.conversation_id
                            )

                        if fn == "arguments_map":
                            pre = outer.seed.pre_invoke
                            if not isinstance(pre, Tool):
                                raise ValueError("seed.pre_invoke is not a Tool")
                            argmap = pre.to_dict().get("arguments_map", {})
                            return Message(
                                content=FunctionResponseContent(
                                    name="arguments_map",
                                    response={"arguments_map": argmap}
                                ),
                                role=MessageRole.AGENT,
                                parent_message_id=message.message_id,
                                conversation_id=message.conversation_id
                            )

                        if fn == "agent_meta":
                            meta = {"name": outer.seed.name, "description": outer.seed.description}
                            return Message(
                                content=FunctionResponseContent(
                                    name="agent_meta",
                                    response=meta
                                ),
                                role=MessageRole.AGENT,
                                parent_message_id=message.message_id,
                                conversation_id=message.conversation_id
                            )

                        # Unknown function
                        return Message(
                            content=FunctionResponseContent(
                                name=fn,
                                response={"error": f"Unknown function '{fn}'."}
                            ),
                            role=MessageRole.AGENT,
                            parent_message_id=message.message_id,
                            conversation_id=message.conversation_id
                        )

                    except Exception as e:
                        return Message(
                            content=FunctionResponseContent(
                                name=fn,
                                response={"error": f"{type(e).__name__}: {e}"}
                            ),
                            role=MessageRole.AGENT,
                            parent_message_id=message.message_id,
                            conversation_id=message.conversation_id
                        )

                # Fallback
                return Message(
                    content=TextContent(text="Unsupported content type."),
                    role=MessageRole.AGENT,
                    parent_message_id=message.message_id,
                    conversation_id=message.conversation_id
                )

        self._server = _Server(url = f"http://{self._host}:{self._port}")

    # -----------------------------
    # Run / URL composition
    # -----------------------------
    def _compose_public_url(self, *, public_host: str, port: int, scheme: str = "http") -> str:
        # a2a clients hit "<base>/a2a"; the card URL itself should be the base origin
        return f"{scheme}://{public_host}:{port}"

    def run(
        self,
        *,
        debug: bool = False,
    ) -> None:
        """
        Start the server and publish a valid Agent Card URL.

        - If `public_url` was passed to __init__, we use it verbatim.
        - Otherwise we compose `scheme://public_host:port`. If `public_host` is
          not provided, default "127.0.0.1".
        - We instantiate the dynamic A2A server *here* with `url=...`, avoiding
          the "URL is required for A2A agent card" exception.
        """
        # Run HTTP listener (client uses "<final_url>/a2a")
        run_server(self._server, host=self._host, port=self._port, debug=debug)
