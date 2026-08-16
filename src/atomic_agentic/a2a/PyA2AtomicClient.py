from __future__ import annotations

import asyncio
from typing import Any, Mapping

from python_a2a import (
    A2AClient,
    FunctionCallContent,
    FunctionParameter,
    Message,
    MessageRole,
)

from ..constants.core import HeaderValue
from ..exceptions import PyA2AtomicConnectionError, RemoteInvocationError
from ..utils.core import normalize_headers
from ..constants.python_a2a import (
    GET_INVOKABLE_METADATA_FUNCTION,
    LIST_INVOKABLES_FUNCTION,
    PYA2A_RESULT_KEY,
)

__all__ = ["PyA2AtomicClient"]


class PyA2AtomicClient:
    """
    Thin transport/client adapter for PyA2AtomicHost-compatible endpoints.

    Public contract
    ---------------
    - list_invokables() -> dict[str, dict[str, Any]]
    - get_invokable_metadata(remote_name) -> dict[str, Any]
    - call_invokable(remote_name, inputs) -> Any

    Agent card fetching
    --------------------
    Construction builds the underlying A2AClient, which attempts to fetch
    the remote agent card internally. If that fetch fails, python_a2a
    substitutes a stub "Unknown Agent" card rather than raising (this is
    python_a2a's own behavior). Genuine failures surface via refresh() or
    on the first real remote operation (list_invokables/
    get_invokable_metadata/call_invokable).
    """

    def __init__(
        self,
        url: str,
        headers: Mapping[str, HeaderValue] | None = None,
        timeout: float = 600,
        google_a2a_compatible: bool = False,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string.")

        resolved_url = url.strip()
        self._headers: Mapping[str, str] | None = normalize_headers(headers)

        self._client = A2AClient(
            resolved_url,
            headers=dict(self._headers) if self._headers is not None else None,
            timeout=float(timeout),
            google_a2a_compatible=google_a2a_compatible,
        )
        # get_agent_card() is a plain attribute read and cannot raise; a
        # failed internal fetch inside A2AClient.__init__ already produced
        # a stub card by this point (see class docstring).
        self._agent_card: Any = self._client.get_agent_card()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def url(self) -> str:
        """Live endpoint URL. python_a2a's A2AClient rewrites its own
        endpoint_url after a successful call (trying URL variants like
        `/a2a` or `/tasks/send`), so this reads the client's current value
        rather than the originally-supplied construction argument."""
        return self._client.endpoint_url

    @property
    def headers(self) -> Mapping[str, str] | None:
        return self._headers

    @headers.setter
    def headers(self, value: Mapping[str, HeaderValue] | None) -> None:
        self._headers = normalize_headers(value)
        self._client.headers = dict(self._headers) if self._headers is not None else {}

    @property
    def timeout(self) -> float:
        return self._client.timeout

    @property
    def google_a2a_compatible(self) -> bool:
        return self._client.is_using_google_a2a_format()

    def refresh(
        self,
        headers: Mapping[str, HeaderValue] | None = None,
        timeout: float | None = None,
        google_a2a_compatible: bool | None = None,
    ) -> None:
        """
        Update stored client configuration and re-fetch the agent card.

        Mutates the existing A2AClient in place (headers/timeout are plain
        public attributes; google_a2a_compatible has a public setter) — no
        new client instance is created. Raises if headers, timeout, and
        google_a2a_compatible are all None, since a no-op refresh() is a
        caller bug. If the agent-card re-fetch fails, all mutations are
        rolled back before raising, so a failed refresh() leaves the client
        exactly as it was.
        """
        if headers is None and timeout is None and google_a2a_compatible is None:
            raise ValueError(
                "refresh() requires at least one of headers, timeout, or "
                "google_a2a_compatible; none were provided."
            )

        previous_headers = self._headers
        previous_client_headers = self._client.headers
        previous_timeout = self._client.timeout
        previous_google_a2a_compatible = self._client.is_using_google_a2a_format()

        if headers is not None:
            self._headers = normalize_headers(headers)
            self._client.headers = dict(self._headers) if self._headers is not None else {}
        if timeout is not None:
            self._client.timeout = float(timeout)
        if google_a2a_compatible is not None:
            self._client.use_google_a2a_format(google_a2a_compatible)

        try:
            self._agent_card = self._client._fetch_agent_card()
        except Exception as exc:
            self._headers = previous_headers
            self._client.headers = previous_client_headers
            self._client.timeout = previous_timeout
            self._client.use_google_a2a_format(previous_google_a2a_compatible)
            raise PyA2AtomicConnectionError(
                f"{type(self).__name__}: failed to refresh agent card from {self.url!r}: {exc}"
            ) from exc

    @classmethod
    async def async_create(
        cls,
        url: str,
        headers: Mapping[str, HeaderValue] | None = None,
        timeout: float = 600,
        google_a2a_compatible: bool = False,
    ) -> "PyA2AtomicClient":
        """Non-blocking construction. __init__ performs a blocking HTTP
        fetch internally via A2AClient's own constructor, so this runs it
        in a worker thread rather than stalling the event loop."""
        return await asyncio.to_thread(
            cls,
            url,
            headers=headers,
            timeout=timeout,
            google_a2a_compatible=google_a2a_compatible,
        )

    @property
    def agent_card(self) -> Any:
        return self._agent_card

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def list_invokables(self) -> dict[str, dict[str, Any]]:
        payload = self._send_function_call(
            function_name=LIST_INVOKABLES_FUNCTION,
            parameters={},
        )
        self._raise_if_error_payload(
            payload=payload,
            function_name=LIST_INVOKABLES_FUNCTION,
        )

        result: dict[str, dict[str, Any]] = {}
        for name, meta in payload.items():
            if not isinstance(name, str):
                raise PyA2AtomicConnectionError(
                    "list_invokables returned a non-string invokable name."
                )
            if not isinstance(meta, Mapping):
                raise PyA2AtomicConnectionError(
                    f"list_invokables returned non-mapping metadata for {name!r}."
                )
            result[name] = dict(meta)

        return result

    def get_invokable_metadata(self, remote_name: str) -> dict[str, Any]:
        resolved_remote_name = str(remote_name).strip()
        if not resolved_remote_name:
            raise ValueError("remote_name must be a non-empty string.")

        payload = self._send_function_call(
            function_name=GET_INVOKABLE_METADATA_FUNCTION,
            parameters={"name": resolved_remote_name},
        )
        self._raise_if_error_payload(
            payload=payload,
            function_name=GET_INVOKABLE_METADATA_FUNCTION,
        )

        required_keys = {
            "name",
            "description",
            "parameters",
            "return_type",
            "invokable_type",
        }
        missing = required_keys - set(payload.keys())
        if missing:
            raise PyA2AtomicConnectionError(
                f"get_invokable_metadata missing required key(s): {sorted(missing)!r}."
            )

        return dict(payload)

    def call_invokable(
        self,
        remote_name: str,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        resolved_remote_name = str(remote_name).strip()
        if not resolved_remote_name:
            raise ValueError("remote_name must be a non-empty string.")
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping.")

        payload = self._send_function_call(
            function_name=resolved_remote_name,
            parameters=dict(inputs),
        )
        return dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "url": self.url,
            "has_headers": self._headers is not None,
            "header_keys": sorted(self._headers.keys()) if self._headers is not None else [],
            "timeout": self.timeout,
            "google_a2a_compatible": self.google_a2a_compatible,
            "agent_name": getattr(self._agent_card, "name", None),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _send_function_call(
        self,
        *,
        function_name: str,
        parameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(function_name, str) or not function_name.strip():
            raise ValueError("function_name must be a non-empty string.")
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping.")

        call = FunctionCallContent(
            name=function_name.strip(),
            parameters=[
                FunctionParameter(name=str(name), value=value)
                for name, value in parameters.items()
            ],
        )
        message = Message(content=call, role=MessageRole.USER)

        try:
            response = self._client.send_message(message)
        except Exception as exc:  # pragma: no cover
            raise PyA2AtomicConnectionError(
                f"{type(self).__name__}: send_message failed for function "
                f"{function_name!r}: {exc}"
            ) from exc

        content = getattr(response, "content", None)
        if getattr(content, "type", None) != "function_response":
            raise PyA2AtomicConnectionError(
                f"{type(self).__name__}: expected function_response for "
                f"{function_name!r}, got {getattr(content, 'type', None)!r}."
            )

        payload = getattr(content, "response", None)
        if not isinstance(payload, Mapping):
            raise PyA2AtomicConnectionError(
                f"{type(self).__name__}: function_response payload for "
                f"{function_name!r} must be a mapping."
            )

        return dict(payload)

    def _raise_if_error_payload(
        self,
        *,
        payload: Mapping[str, Any],
        function_name: str,
    ) -> None:
        raw_error = payload.get("error")
        if raw_error is None:
            return

        error_type = payload.get("error_type")
        if not isinstance(error_type, str) or not error_type.strip():
            error_type = "RemoteError"

        raise RemoteInvocationError(
            str(raw_error),
            error_type=error_type,
            function_name=function_name,
        )