# modules/A2Agents.py
from __future__ import annotations
import logging
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
_logger = logging.getLogger(__name__)
from typing import Any, Mapping, Dict, Optional

# Project-local imports
from .Agents import Agent
from .LLMEngines import LLMEngine
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

    def __init__(self, url: str,
                 name: str|None = None,
                 description: str|None = None):
        # We still call Agent.__init__ to keep consistent metadata, but the engine
        # is never used by this proxy.
        # Expect a FULL endpoint (e.g., "http://127.0.0.1:5000/a2a").
        self._client = A2AClient(url)
        self._url = url
        agent_card = self._client.get_agent_card()
        super().__init__(name=name if name else agent_card.name,
                         description=description if description else agent_card.description,
                         llm_engine=None,
                         role_prompt="You are a proxy Agent forwarding calls over A2A.",
                         context_enabled=False)

    @property
    def url(self) -> str:
        return self._url
    @url.setter
    def url(self, val: str) -> None:
        self._url = val
        self._client = A2AClient(val)

    # We override attach/detach to NO-OP, since the proxy doesn't hold files or context.
    def attach(self, path: str) -> bool:  # type: ignore[override]
        return False

    def detach(self, path: str) -> bool:  # type: ignore[override]
        return False

    def _invoke(self, inputs: Mapping[str, Any]) -> Any:
        pass
    
    def invoke(self, inputs: Mapping[str, Any]) -> Any:  # type: ignore[override]
        _ensure_mapping("A2AProxyAgent.invoke", inputs)
        _logger.info(f"[A2AProxyAgent - {self.name}].invoke: forwarding payload to {self.url}")

        # Typed FunctionParameter avoids `'dict' object has no attribute 'name'`.
        call = FunctionCallContent(
            name="invoke",
            parameters=[FunctionParameter(name="payload", value=dict(inputs))]
        )
        msg = Message(content=call, role=MessageRole.USER)
        resp = self._client.send_message(msg).content.response
        result = resp
        if isinstance(resp, Mapping):
            result = resp.get(A2A_RESULT_KEY, resp)
        _logger.info(f"[A2AProxyAgent - {self.name}].invoke: received result of type {type(result)}")
        return result
    
    @property
    def role_prompt(self)->str:
        return self._role_prompt
    @property
    def llm_engine(self)->LLMEngine:
        return self._llm_engine
    @property
    def context_enabled(self)->bool:
        return self._context_enabled
    @property
    def arguments_map(self):
        call = FunctionCallContent(name="arguments_map", parameters=[])
        msg = Message(content=call, role=MessageRole.USER)
        resp = self._client.send_message(msg)
        if getattr(resp.content, "type", None) == "function_response":
            return resp.content.response[A2A_RESULT_KEY]  # type: ignore[return-value]
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

class A2AServerAgent(Agent):
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
        self._seed = seed
        super().__init__(name = name if name else seed.name,
                         description = description if description else seed.description,
                         llm_engine = seed.llm_engine,
                         role_prompt = seed.role_prompt,
                         context_enabled=seed.context_enabled,
                         pre_invoke=seed.pre_invoke,
                         history_window=seed.history_window)
        self._version = version
        self._host = host  # if provided, used verbatim in Agent Card
        self._port = port
        self._arguments_map = seed.arguments_map

        # runtime fields
        self._server = None
        
        outer = self

        @agent(name=name if name else seed.name,
               description=description if description else seed.description,
               version=version,
               url=f"http://{self._host}:{self._port}"
               )
        class _Server(A2AServer):
            """Dynamic per-instance server. Instantiated in run(url=...)."""

            def handle_message(self, message: Message) -> Message:
                content = message.content
                ctype = content.type

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
                            result = outer._seed.invoke(payload)  # Agent returns str by contract
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
                            # argmap = outer._seed.pre_invoke.to_dict().get("arguments_map", {})
                            return Message(
                                content=FunctionResponseContent(
                                    name="arguments_map",
                                    response={A2A_RESULT_KEY: outer._seed.arguments_map}
                                ),
                                role=MessageRole.AGENT,
                                parent_message_id=message.message_id,
                                conversation_id=message.conversation_id
                            )

                        if fn == "agent_meta":
                            meta = {"name": outer._seed.name, "description": outer._seed.description}
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
    @property
    def seed(self)->Agent:
        return self._seed
    @seed.setter
    def seed(self, val:Agent)->None:
        self._seed = val
    @property
    def role_prompt(self)->str:
        return self._seed.role_prompt
    @property
    def llm_engine(self)->LLMEngine:
        return self._seed.llm_engine
    @property
    def context_enabled(self):
        return self._seed.context_enabled
    @property
    def history_window(self)->int:
        return self._seed.history_window
    
    def attach(self, path: str) -> bool:  # type: ignore[override]
        return self._seed.attach(path)
    
    def detach(self, path: str) -> bool:  # type: ignore[override]
        return self._seed.detach(path)

    def run(self,*,debug: bool = False,) -> None:
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
