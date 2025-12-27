from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
import json
import logging
import re
import string
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from python_a2a import (
    A2AServer, run_server, agent,
    Message, MessageRole, FunctionResponseContent,
    TextContent
)

from ..agents import Agent

A2A_RESULT_KEY = "__py_A2A_result__"

# ───────────────────────────────────────────────────────────────────────────────
# A2AgentHost wrapper class
# ───────────────────────────────────────────────────────────────────────────────
class A2AgentHost:
    """
    A2AgentHost
    -----------
    Wraps a local :class:`~atomic_agentic.Primitives.Agent` as a python-a2a server
    using the message-level function-calling pattern.

    Exposed function names:
      - "invoke":          payload: Mapping[str, Any] -> {__py_A2A_result__: <agent.invoke(payload)>}
      - "agent_metadata":  no params                  -> {arguments_map: <agent.arguments_map>, return_type: <agent.post_invoke.return_type>}
    """

    def __init__(
        self,
        seed_agent: Agent,
        version: str = "1.0.0",
        host: str = "localhost",
        port: int = 5000,
    ) -> None:
        if not isinstance(seed_agent, Agent):
            raise TypeError("A2AgentHost requires a seed Agent.")
        self._seed_agent = seed_agent
        self._version = version
        self._host = host
        self._port = port

        outer = self

        @agent(
            name=seed_agent.name,
            description=seed_agent.description,
            version=version,
            url=f"http://{host}:{port}",
        )
        class _Server(A2AServer):
            """Per-instance A2A server wrapper around a local Agent."""

            def handle_message(self, message: Message) -> Message:
                content = message.content
                ctype = content.type

                # Text: brief help
                if ctype == "text":
                    return Message(
                        content=TextContent(
                            text="Call as function_call: name in {'invoke','agent_metadata'}."
                        ),
                        role=MessageRole.AGENT,
                        parent_message_id=message.message_id,
                        conversation_id=message.conversation_id,
                    )

                # Function call dispatch
                if ctype == "function_call":
                    fn = content.name
                    params = {p.name: p.value for p in (content.parameters or [])}

                    try:
                        if fn == "invoke":
                            payload = params.get("payload", {})
                            if not isinstance(payload, Mapping):
                                raise TypeError("invoke expects 'payload' to be a mapping")
                            result = outer._seed_agent.invoke(payload)  # returns Any
                            return Message(
                                content=FunctionResponseContent(
                                    name="invoke",
                                    response={A2A_RESULT_KEY: result},
                                ),
                                role=MessageRole.AGENT,
                                parent_message_id=message.message_id,
                                conversation_id=message.conversation_id,
                            )

                        if fn == "agent_metadata":
                            meta = {
                                "arguments_map": outer._seed_agent.arguments_map,
                                "return_type": outer._seed_agent.post_invoke.return_type,
                            }
                            return Message(
                                content=FunctionResponseContent(
                                    name="agent_metadata",
                                    response=meta,
                                ),
                                role=MessageRole.AGENT,
                                parent_message_id=message.message_id,
                                conversation_id=message.conversation_id,
                            )

                        # Unknown function
                        return Message(
                            content=FunctionResponseContent(
                                name=fn,
                                response={"error": f"Unknown function '{fn}'."},
                            ),
                            role=MessageRole.AGENT,
                            parent_message_id=message.message_id,
                            conversation_id=message.conversation_id,
                        )

                    except Exception as e:
                        return Message(
                            content=FunctionResponseContent(
                                name=fn,
                                response={"error": f"{type(e).__name__}: {e}"},
                            ),
                            role=MessageRole.AGENT,
                            parent_message_id=message.message_id,
                            conversation_id=message.conversation_id,
                        )

                # Fallback
                return Message(
                    content=TextContent(text="Unsupported content type."),
                    role=MessageRole.AGENT,
                    parent_message_id=message.message_id,
                    conversation_id=message.conversation_id,
                )

        self._server = _Server(url=f"http://{self._host}:{self._port}")

    @property
    def seed_agent(self) -> Agent:
        """The wrapped Agent instance."""
        return self._seed_agent

    @property
    def host(self) -> str:
        """The host address the server will bind to."""
        return self._host

    @property
    def port(self) -> int:
        """The port number the server will listen on."""
        return self._port

    def run(self, *, debug: bool = False) -> None:
        """Start the underlying A2A server."""
        run_server(self._server, host=self._host, port=self._port, debug=debug)
