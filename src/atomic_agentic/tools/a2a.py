from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, Mapping, Optional

from ..core.Exceptions import ToolDefinitionError, ToolInvocationError
from ..core.Parameters import ParamSpec
from ..core.sentinels import NO_VAL
from .base import Tool
from ..a2a.PyA2AtomicClient import PyA2AtomicClient


__all__ = ["PyA2AtomicTool"]


class PyA2AtomicTool(Tool):
    """
    Proxy one remote PyA2AtomicHost invokable as a normal AA Tool.

    Construction paths
    ------------------
    1) Pass an existing PyA2AtomicClient plus a required remote_name.
    2) Pass raw transport config (url, optional headers) plus a required remote_name.

    Local AA-facing identity
    ------------------------
    - name: user-supplied name, otherwise remote_name
    - namespace: user-supplied namespace, otherwise remote agent card name, otherwise "pya2a"
    - description: user-supplied description, otherwise remote metadata description,
      otherwise remote agent card description, otherwise a stub

    Refresh semantics
    -----------------
    Refresh only re-fetches remote metadata, updates the call binding, and rebuilds
    the schema. It does NOT automatically mutate the local AA-facing name, namespace,
    or description after construction.
    """

    def __init__(
        self,
        remote_name: str,
        name: str | None = None,
        namespace: str | None = None,
        description: str | None = None,
        *,
        client: PyA2AtomicClient | None = None,
        url: str | None = None,
        headers: Mapping[str, str] | None = None,
        filter_extraneous_inputs: bool = True,
    ) -> None:
        resolved_remote_name = str(remote_name).strip()
        if not resolved_remote_name:
            raise ToolDefinitionError("remote_name must be a non-empty string.")

        if client is not None:
            if not isinstance(client, PyA2AtomicClient):
                raise TypeError(
                    f"client must be a PyA2AtomicClient, got {type(client)!r}."
                )
            if url is not None or headers is not None:
                raise ValueError(
                    "Pass either client or raw transport settings (url/headers), not both."
                )
            resolved_client = client
        else:
            if not isinstance(url, str) or not url.strip():
                raise ValueError(
                    "url is required when client is not provided and must be a non-empty string."
                )
            resolved_client = PyA2AtomicClient(url=url, headers=headers)

        self._client: PyA2AtomicClient = resolved_client
        self._remote_name: str = resolved_remote_name
        self._remote_metadata: dict[str, Any] = self._client.get_invokable_metadata(
            self._remote_name
        )

        agent_card = self._client.agent_card

        resolved_name = str(name or "").strip() or self._remote_name

        resolved_namespace = str(namespace or "").strip()
        if not resolved_namespace:
            resolved_namespace = str(getattr(agent_card, "name", "") or "").strip() or "pya2a"

        explicit_description = str(description or "").strip()
        remote_description = str(self._remote_metadata.get("description") or "").strip()
        host_description = str(getattr(agent_card, "description", "") or "").strip()
        resolved_description = (
            explicit_description
            or remote_description
            or host_description
            or f"PyA2Atomic tool '{resolved_name}'"
        )

        function = self._client.call_invokable

        super().__init__(
            function=function,
            name=resolved_name,
            namespace=resolved_namespace,
            description=resolved_description,
            filter_extraneous_inputs=filter_extraneous_inputs,
        )

    # ------------------------------------------------------------------ #
    # Proxy properties
    # ------------------------------------------------------------------ #
    @property
    def client(self) -> PyA2AtomicClient:
        return self._client

    @property
    def remote_name(self) -> str:
        return self._remote_name

    @property
    def url(self) -> str:
        return self._client.url

    @property
    def headers(self) -> Mapping[str, str] | None:
        return self._client.headers

    @headers.setter
    def headers(self, value: Mapping[str, str] | None) -> None:
        self._client.headers = value
        self.refresh()

    @property
    def agent_card(self) -> Any:
        return self._client.agent_card

    @property
    def remote_metadata(self) -> dict[str, Any]:
        return dict(self._remote_metadata)

    # ------------------------------------------------------------------ #
    # Signature Building (Template Method)
    # ------------------------------------------------------------------ #
    def _build_tool_signature(self) -> tuple[list[ParamSpec], str]:
        parameters_raw = self._remote_metadata.get("parameters")
        return_type = self._remote_metadata.get("return_type")

        if not isinstance(parameters_raw, list):
            raise ToolDefinitionError(
                f"{self.full_name}: remote metadata 'parameters' must be a list."
            )
        if not isinstance(return_type, str):
            raise ToolDefinitionError(
                f"{self.full_name}: remote metadata 'return_type' must be a str."
            )

        parameters: list[ParamSpec] = []
        for index, item in enumerate(parameters_raw):
            if not isinstance(item, Mapping):
                raise ToolDefinitionError(
                    f"{self.full_name}: parameters[{index}] must be a mapping."
                )
            parameters.append(ParamSpec.from_dict(dict(item)))

        return parameters, return_type

    # ------------------------------------------------------------------ #
    # Tool helpers
    # ------------------------------------------------------------------ #
    def _get_mod_qual(
        self,
        function: Callable[..., Any],
    ) -> tuple[Optional[str], Optional[str]]:
        return PyA2AtomicClient.call_invokable.__module__, PyA2AtomicClient.call_invokable.__qualname__

    def to_arg_kwarg(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        return tuple(), dict(inputs)

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if args:
            raise ToolInvocationError(
                f"{self.full_name}: PyA2Atomic tools do not accept positional arguments; got {args!r}."
            )
        return self._function(self._remote_name, inputs=kwargs)

    async def async_execute(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        if args:
            raise ToolInvocationError(
                f"{self.full_name}: PyA2Atomic tools do not accept positional arguments; got {args!r}."
            )

        try:
            return await asyncio.to_thread(
                self._function,
                self._remote_name,
                inputs=kwargs,
            )
        except Exception as exc:
            raise ToolInvocationError(
                f"{self.full_name}: async invocation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def refresh(self, headers: Any = NO_VAL) -> None:
        """
        Re-fetch remote metadata and rebuild the local binding.

        Refresh does not rewrite the Tool's local name, namespace, or description.
        It only updates remote metadata, callable binding, parameters, and return type.
        """
        if headers is not NO_VAL:
            self._client.headers = headers

        self._remote_metadata = self._client.get_invokable_metadata(self._remote_name)

        self._function = self._client.call_invokable
        self._module, self._qualname = self._get_mod_qual(self._function)

        parameters, return_type = self._build_tool_signature()
        self._parameters = parameters
        self._return_type = return_type

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "remote_name": self._remote_name,
                "client": self._client.to_dict(),
            }
        )
        return data