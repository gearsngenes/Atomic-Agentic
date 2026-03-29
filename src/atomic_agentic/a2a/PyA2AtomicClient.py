from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from python_a2a import (
    A2AClient,
    FunctionCallContent,
    FunctionParameter,
    Message,
    MessageRole,
)

from .PyA2AtomicHost import PyA2AtomicHost, PYA2A_RESULT_KEY

__all__ = ["PyA2AtomicClient"]


class PyA2AtomicClient:
    """
    Thin transport/client adapter for PyA2AtomicHost-compatible endpoints.

    Public contract
    ---------------
    - list_invokables() -> dict[str, dict[str, Any]]
    - get_invokable_metadata(remote_name) -> dict[str, Any]
    - call_invokable(remote_name, inputs) -> Any

    The client eagerly fetches the remote agent card on construction so that
    connection failures surface immediately.
    """

    @staticmethod
    def _normalize_headers(
        value: Mapping[str, str] | None,
    ) -> Mapping[str, str] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("headers must be a mapping of strings to strings.")

        normalized: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise ValueError("headers must be a mapping of strings to strings.")
            normalized[key] = item

        return MappingProxyType(normalized)

    def __init__(
        self,
        url: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("url must be a non-empty string.")

        self._url: str = url.strip()
        self._headers: Mapping[str, str] | None = self._normalize_headers(headers)
        self._client: A2AClient
        self._agent_card: Any

        self._rebuild_client()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def url(self) -> str:
        return self._url

    @property
    def headers(self) -> Mapping[str, str] | None:
        return self._headers

    @headers.setter
    def headers(self, value: Mapping[str, str] | None) -> None:
        self._headers = self._normalize_headers(value)
        self._rebuild_client()

    @property
    def agent_card(self) -> Any:
        return self._agent_card

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def list_invokables(self) -> dict[str, dict[str, Any]]:
        payload = self._send_function_call(
            function_name=PyA2AtomicHost.LIST_INVOKABLES_FUNCTION,
            parameters={},
        )
        self._raise_if_error_payload(
            payload=payload,
            function_name=PyA2AtomicHost.LIST_INVOKABLES_FUNCTION,
        )

        result: dict[str, dict[str, Any]] = {}
        for name, meta in payload.items():
            if not isinstance(name, str):
                raise RuntimeError("list_invokables returned a non-string invokable name.")
            if not isinstance(meta, Mapping):
                raise RuntimeError(
                    f"list_invokables returned non-mapping metadata for {name!r}."
                )
            result[name] = dict(meta)

        return result

    def get_invokable_metadata(self, remote_name: str) -> dict[str, Any]:
        resolved_remote_name = str(remote_name).strip()
        if not resolved_remote_name:
            raise ValueError("remote_name must be a non-empty string.")

        payload = self._send_function_call(
            function_name=PyA2AtomicHost.GET_INVOKABLE_METADATA_FUNCTION,
            parameters={"name": resolved_remote_name},
        )
        self._raise_if_error_payload(
            payload=payload,
            function_name=PyA2AtomicHost.GET_INVOKABLE_METADATA_FUNCTION,
        )

        required_keys = {
            "name",
            "description",
            "parameters",
            "return_type",
            "filter_extraneous_inputs",
            "invokable_type",
        }
        missing = required_keys - set(payload.keys())
        if missing:
            raise RuntimeError(
                f"get_invokable_metadata missing required key(s): {sorted(missing)!r}."
            )

        return dict(payload)

    def call_invokable(
        self,
        remote_name: str,
        inputs: Mapping[str, Any],
    ) -> Any:
        resolved_remote_name = str(remote_name).strip()
        if not resolved_remote_name:
            raise ValueError("remote_name must be a non-empty string.")
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping.")

        payload = self._send_function_call(
            function_name=resolved_remote_name,
            parameters=dict(inputs),
        )
        self._raise_if_error_payload(
            payload=payload,
            function_name=resolved_remote_name,
        )

        if PYA2A_RESULT_KEY not in payload:
            raise RuntimeError(
                f"Direct invokable call {resolved_remote_name!r} did not return "
                f"required result key {PYA2A_RESULT_KEY!r}."
            )

        return payload[PYA2A_RESULT_KEY]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "url": self._url,
            "has_headers": self._headers is not None,
            "header_keys": sorted(self._headers.keys()) if self._headers is not None else [],
            "agent_name": getattr(self._agent_card, "name", None),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _rebuild_client(self) -> None:
        headers = dict(self._headers) if self._headers is not None else None
        self._client = A2AClient(self._url, headers=headers)

        try:
            self._agent_card = self._client.get_agent_card()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"{type(self).__name__}: failed to fetch agent card from {self._url!r}: {exc}"
            ) from exc

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
            raise RuntimeError(
                f"{type(self).__name__}: send_message failed for function "
                f"{function_name!r}: {exc}"
            ) from exc

        content = getattr(response, "content", None)
        if getattr(content, "type", None) != "function_response":
            raise RuntimeError(
                f"{type(self).__name__}: expected function_response for "
                f"{function_name!r}, got {getattr(content, 'type', None)!r}."
            )

        payload = getattr(content, "response", None)
        if not isinstance(payload, Mapping):
            raise RuntimeError(
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

        raise RuntimeError(
            f"{type(self).__name__}: remote call {function_name!r} failed "
            f"with {error_type}: {raw_error}"
        )