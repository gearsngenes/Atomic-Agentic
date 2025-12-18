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
from collections import OrderedDict

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
                 description: str|None = None,
                 headers: Any|None = None):
        # We still call Agent.__init__ to keep consistent metadata, but the engine
        # is never used by this proxy.
        # Expect a FULL endpoint (e.g., "http://127.0.0.1:5000/a2a").
        self._client = A2AClient(url, headers = headers)
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
        raise NotImplementedError("A2AProxyAgent._invoke is not implemented")
    
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
    def to_dict(self)-> OrderedDict[str, Any]:
        dict_data = super().to_dict()
        dict_data.update(OrderedDict(
            url = self._url,
        ))
