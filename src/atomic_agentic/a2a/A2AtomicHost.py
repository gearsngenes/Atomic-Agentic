from __future__ import annotations

from typing import Mapping
from python_a2a import (
    A2AServer, run_server, agent,
    Message, MessageRole, FunctionResponseContent,
    TextContent
)

from ..core.Invokable import AtomicInvokable

A2A_RESULT_KEY = "__py_A2A_result__"

# ───────────────────────────────────────────────────────────────────────────────
# A2AtomicHost wrapper class
# ───────────────────────────────────────────────────────────────────────────────
class A2AtomicHost:
    """
    A2AtomicHost
    -----------
    Wraps a local AtomicInvokable as a python-a2a server
    using the message-level function-calling pattern.

    Exposed function names:
      - "invoke":          payload: Mapping[str, Any] -> {__py_A2A_result__: <agent.invoke(payload)>}
      - "agent_metadata":  no params                  -> {arguments_map: <agent.arguments_map>, return_type: <agent.post_invoke.return_type>}
    """

    def __init__(
        self,
        component: AtomicInvokable,
        version: str = "1.0.0",
        host: str = "localhost",
        port: int = 5000,
    ) -> None:
        if not isinstance(component, AtomicInvokable):
            raise TypeError("A2AtomicHost requires a seed Agent.")
        self._component = component
        self._version = version
        self._host = host
        self._port = port

        outer = self

        @agent(
            name=component.name,
            description=component.description,
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
                            result = outer._component.invoke(payload)  # returns Any
                            return Message(
                                content=FunctionResponseContent(
                                    name="invoke",
                                    response={A2A_RESULT_KEY: result},
                                ),
                                role=MessageRole.AGENT,
                                parent_message_id=message.message_id,
                                conversation_id=message.conversation_id,
                            )

                        if fn == "invokable_metadata":
                            params_list = [spec.to_dict() for spec in outer._component.parameters]
                            ret_type = outer._component.return_type
                            meta = {
                                "parameters": params_list,
                                "return_type": ret_type,
                            }
                            return Message(
                                content=FunctionResponseContent(
                                    name="invokable_metadata",
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
    def component(self) -> AtomicInvokable:
        """The wrapped Agent instance."""
        return self._component

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
