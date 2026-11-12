from __future__ import annotations

import logging
from typing import Any, Mapping

from python_a2a import (
    A2AServer,
    AgentCard,
    Message,
    MessageRole,
    FunctionResponseContent,
    TextContent,
    run_server,
)

from ..core.Invokable import AtomicInvokable
from ..constants.python_a2a import (
    GET_INVOKABLE_METADATA_FUNCTION,
    LIST_INVOKABLES_FUNCTION,
    PYA2A_RESULT_KEY,
)
from ..constants.core import IDENTIFIER_PATTERN

__all__ = ["PyA2AtomicHost"]

logger = logging.getLogger(__name__)


class PyA2AtomicHost(A2AServer):
    """
    Host a registry of local AtomicInvokables as a single python-a2a server.

    Remote function-call contract
    -----------------------------
    Reserved function names:
      - "list_invokables"
          Return a mapping of invokable name -> invokable metadata.
      - "get_invokable_metadata"
          Input: {"name": <str>}
          Return metadata for one invokable.

    All other function names are treated as a registered invokable name.
    The parsed function-call parameters are passed directly to:
        invokable.invoke(inputs=<parsed parameter dict>)

    Direct invocation success payload:
      { "__py_a2a_result__": <invokable's AtomicResult.result payload> }

      `invoke()` returns an AtomicResult-family envelope; only its `.result`
      (the caller-facing payload) crosses the wire under this key. Envelope
      fields (run_id, started_at/ended_at, elapsed_s, invoker_id) are not
      included.

    Error payload:
      {
        "error": <message>,
        "error_type": <exception class name>,
        "function_name": <requested function name>,
      }
    """

    _RESERVED_REMOTE_NAMES: frozenset[str] = frozenset(
        {LIST_INVOKABLES_FUNCTION, GET_INVOKABLE_METADATA_FUNCTION}
    )

    def __init__(
        self,
        invokables: list[AtomicInvokable] | Mapping[str, AtomicInvokable],
        *,
        name: str,
        description: str,
        version: str = "1.0.0",
        host: str = "localhost",
        port: int = 5000,
        url: str | None = None,
        capabilities: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("description must be a non-empty string.")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("version must be a non-empty string.")
        if not isinstance(host, str) or not host.strip():
            raise ValueError("host must be a non-empty string.")
        if not isinstance(port, int) or port <= 0:
            raise ValueError("port must be an int > 0.")

        normalized_host = host.strip()
        normalized_port = port
        normalized_url = (url.strip() if isinstance(url, str) and url.strip() else f"http://{normalized_host}:{normalized_port}")
        normalized_capabilities = dict(capabilities) if capabilities is not None else None

        self._host: str = normalized_host
        self._port: int = normalized_port
        self._url: str = normalized_url
        self._version: str = version.strip()

        self._invokables: dict[str, AtomicInvokable] = self._normalize_invokables(invokables)

        agent_card = self._build_agent_card(
            name=name.strip(),
            description=description.strip(),
            version=self._version,
            url=self._url,
            capabilities=normalized_capabilities,
        )
        super().__init__(agent_card=agent_card)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return self._url

    @property
    def version(self) -> str:
        return self._version

    @property
    def invokable_names(self) -> list[str]:
        """
        Registry keys (remote names) currently registered — equal to each
        invokable's `.name` only when no override was given at registration.
        """
        return list(self._invokables.keys())

    # ------------------------------------------------------------------ #
    # Local registry management
    # ------------------------------------------------------------------ #
    def register(self, invokable: AtomicInvokable, remote_name: str | None = None) -> str:
        """
        Register `invokable` under `remote_name` (registry-local alias, never
        stored on the invokable itself). `remote_name=None` defaults to
        `invokable.name`. Returns the effective registry key.
        """
        if not isinstance(invokable, AtomicInvokable):
            raise TypeError(
                f"register expects an AtomicInvokable, got {type(invokable).__name__!r}."
            )

        if remote_name is None:
            key = invokable.name
            self._validate_remote_name(key, check_identifier=False)
        else:
            if not isinstance(remote_name, str) or not remote_name.strip():
                raise ValueError("remote_name must be a non-empty string.")
            key = remote_name.strip()
            self._validate_remote_name(key, check_identifier=True)

        if key in self._invokables:
            raise ValueError(f"Duplicate invokable name {key!r} is not allowed.")

        self._invokables[key] = invokable
        logger.debug("%s registered invokable %s", type(self).__name__, key)
        return key

    def remove(self, name: str) -> bool:
        """
        Remove the invokable registered under `name` (registry key / remote
        name). Returns whether an entry was actually removed.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        removed = self._invokables.pop(name.strip(), None) is not None
        logger.debug("%s removed invokable %s -> %s", type(self).__name__, name, removed)
        return removed

    def clear_invokables(self) -> None:
        self._invokables.clear()
        logger.debug("%s cleared all invokables", type(self).__name__)

    # ------------------------------------------------------------------ #
    # Server runner
    # ------------------------------------------------------------------ #
    def run_server(self, *, debug: bool = False) -> None:
        """
        Run this host through python-a2a's server runner.
        """
        run_server(self, host=self._host, port=self._port, debug=debug)

    # ------------------------------------------------------------------ #
    # A2A message handling
    # ------------------------------------------------------------------ #
    def handle_message(self, message: Message) -> Message:
        content = message.content
        content_type = content.type

        if content_type == "text":
            return self._make_text_response(
                text=(
                    "This host expects function calls. Reserved functions: "
                    f"{LIST_INVOKABLES_FUNCTION!r}, "
                    f"{GET_INVOKABLE_METADATA_FUNCTION!r}. "
                    "Any other function name is treated as a registered AtomicInvokable name."
                ),
                message=message,
            )

        if content_type != "function_call":
            return self._make_text_response(
                text="Unsupported content type. Use text or function_call.",
                message=message,
            )

        function_name = (content.name or "").strip()

        try:
            if not function_name:
                raise ValueError("Function name must be a non-empty string.")

            params = {}
            for param in content.parameters or []:
                param_name = (param.name or "").strip()
                if not param_name:
                    raise ValueError("All input parameters must have non-empty parameter names")
                if param_name in params:
                    raise ValueError(f"Duplicate parameter name {param!r} found in parameters.")
                params[param_name] = param.value

            if function_name == LIST_INVOKABLES_FUNCTION:
                payload = self._list_invokables_payload()

            elif function_name == GET_INVOKABLE_METADATA_FUNCTION:
                raw_name = params.get("name")
                if not isinstance(raw_name, str) or not raw_name.strip():
                    raise ValueError(
                        f"{GET_INVOKABLE_METADATA_FUNCTION!r} requires a non-empty 'name' parameter."
                    )
                payload = self._get_invokable_metadata_payload(raw_name.strip())

            else:
                result = self._invoke_registered_invokable(function_name, params)
                payload = {PYA2A_RESULT_KEY: result}

            return self._make_function_response(
                function_name=function_name,
                payload=payload,
                message=message,
            )

        except Exception as exc:
            logger.debug(
                "%s.%s failed for function %r: %s",
                type(self).__name__,
                self.agent_card.name,
                function_name,
                exc,
                exc_info=True,
            )
            return self._make_function_response(
                function_name=function_name or "unknown_function",
                payload={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "function_name": function_name or "unknown_function",
                },
                message=message,
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_agent_card(
        self,
        *,
        name: str,
        description: str,
        version: str,
        url: str,
        capabilities: Mapping[str, Any] | None,
    ) -> AgentCard:
        kwargs: dict[str, Any] = {
            "name": name,
            "description": description,
            "url": url,
            "version": version,
        }
        if capabilities is not None:
            kwargs["capabilities"] = dict(capabilities)
        return AgentCard(**kwargs)

    def _normalize_invokables(
        self,
        invokables: list[AtomicInvokable] | Mapping[str, AtomicInvokable],
    ) -> dict[str, AtomicInvokable]:
        """
        Normalize either construction form into a `remote_name -> invokable`
        registry. Mapping keys are authoritative remote names (identifier-
        validated); list items default their key to `.name` (already
        identifier-valid at the invokable's own construction time, so only
        the reserved-name check applies).
        """
        registry: dict[str, AtomicInvokable] = {}

        if isinstance(invokables, Mapping):
            for remote_name, invokable in invokables.items():
                if not isinstance(invokable, AtomicInvokable):
                    raise TypeError(
                        f"invokables[{remote_name!r}] must be an AtomicInvokable, "
                        f"got {type(invokable).__name__!r}."
                    )
                self._validate_remote_name(remote_name, check_identifier=True)
                registry[remote_name] = invokable
            return registry

        if isinstance(invokables, list):
            for index, invokable in enumerate(invokables):
                if not isinstance(invokable, AtomicInvokable):
                    raise TypeError(
                        f"invokables[{index}] must be an AtomicInvokable, "
                        f"got {type(invokable).__name__!r}."
                    )

                key = invokable.name
                self._validate_remote_name(key, check_identifier=False)
                if key in registry:
                    raise ValueError(f"Duplicate invokable name {key!r} is not allowed.")

                registry[key] = invokable

            return registry

        raise TypeError("invokables must be a list[AtomicInvokable] or Mapping[str, AtomicInvokable].")

    @staticmethod
    def _validate_remote_name(remote_name: str, *, check_identifier: bool) -> None:
        """
        Shared registry-key validation: always rejects reserved wire function
        names; additionally rejects non-identifier strings when
        `check_identifier` is True (dict-form keys and explicit `register()`
        overrides — list-form `.name` defaults are already identifier-valid).
        """
        if remote_name in PyA2AtomicHost._RESERVED_REMOTE_NAMES:
            raise ValueError(
                f"remote_name {remote_name!r} collides with a reserved wire function name."
            )
        if check_identifier and not IDENTIFIER_PATTERN.fullmatch(remote_name):
            raise ValueError(f"remote_name {remote_name!r} is not a valid identifier.")

    def _list_invokables_payload(self) -> dict[str, dict[str, Any]]:
        return {
            name: self._get_invokable_metadata_payload(name)
            for name in self._invokables.keys()
        }

    def _get_invokable_metadata_payload(self, name: str) -> dict[str, Any]:
        invokable = self._invokables.get(name)
        if invokable is None:
            raise KeyError(f"Unknown invokable {name!r}.")
        return {
            "name": name,
            # Raw description, not the rendered getter - the receiving proxy
            # derives its own parameter-bullet block from the separately-sent
            # `parameters` list, so sending rendered text here would double it.
            "description": invokable._description,
            "parameters": [spec.to_dict() for spec in invokable.parameters],
            "return_type": invokable.return_type,
            "invokable_type": type(invokable).__name__,
            "extra_description": invokable._extra_description(),
        }

    def _invoke_registered_invokable(
        self,
        name: str,
        inputs: Mapping[str, Any],
    ) -> Any:
        invokable = self._invokables.get(name)
        if invokable is None:
            raise KeyError(f"Unknown invokable {name!r}.")
        return invokable.invoke(dict(inputs)).result

    def _make_function_response(
        self,
        *,
        function_name: str,
        payload: Mapping[str, Any],
        message: Message,
    ) -> Message:
        return Message(
            content=FunctionResponseContent(
                name=function_name,
                response=dict(payload),
            ),
            role=MessageRole.AGENT,
            parent_message_id=message.message_id,
            conversation_id=message.conversation_id,
        )

    def _make_text_response(
        self,
        *,
        text: str,
        message: Message,
    ) -> Message:
        return Message(
            content=TextContent(text=text),
            role=MessageRole.AGENT,
            parent_message_id=message.message_id,
            conversation_id=message.conversation_id,
        )

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "name": self.agent_card.name,
            "description": self.agent_card.description,
            "version": self._version,
            "host": self._host,
            "port": self._port,
            "url": self._url,
            "invokable_names": self.invokable_names,
            "invokable_count": len(self._invokables),
        }