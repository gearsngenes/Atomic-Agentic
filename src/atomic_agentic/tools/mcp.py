from __future__ import annotations

import functools
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
)

from ..core.Exceptions import ToolDefinitionError, ToolInvocationError
from ..core.Parameters import ParamSpec
from ..core.sentinels import NO_VAL
from ..mcp.MCPClientHub import MCPClientHub
from .base import Tool

__all__ = ["MCPProxyTool"]

class MCPProxyTool(Tool):
    """
    Proxy a single remote MCP tool as a normal AA Tool.

    The proxy owns the AA-facing identity (`name`, `namespace`, `description`)
    and a remote MCP binding (`remote_name`) backed by one `MCPClientHub`.
    """

    def __init__(
        self,
        remote_name: str,
        name: str | None = None,
        namespace: str | None = None,
        description: str = "",
        client_hub: MCPClientHub | None = None,
        transport_mode: Literal["stdio", "sse", "streamable_http"] | None = None,
        endpoint: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        headers: Mapping[str, str] | None = None,
        filter_extraneous_inputs: bool = True,
    ) -> None:
        resolved_remote_name = str(remote_name).strip()
        if not resolved_remote_name:
            raise ToolDefinitionError("remote_name must be a non-empty string.")

        if client_hub is not None:
            if not isinstance(client_hub, MCPClientHub):
                raise TypeError(
                    f"client_hub must be an MCPClientHub, got {type(client_hub)!r}."
                )
            if any(
                value is not None
                for value in (transport_mode, endpoint, command, args, headers)
            ):
                raise ValueError(
                    "Pass either client_hub or raw transport settings, not both."
                )
            client = client_hub
        else:
            if transport_mode is None:
                raise ValueError(
                    "transport_mode is required when client_hub is not provided."
                )
            client = MCPClientHub(
                transport_mode=transport_mode,
                endpoint=endpoint,
                command=command,
                args=args,
                headers=headers,
            )

        self._client_hub: MCPClientHub = client
        self._remote_name: str = resolved_remote_name

        all_tools = self.client_hub.list_tools()
        if self.remote_name not in all_tools:
            raise ToolDefinitionError(
                f"MCPProxyTool: remote tool {self.remote_name!r} not found for "
                f"{self.client_hub.transport_mode!r} transport."
            )

        self._mcpdata: Dict[str, Any] = dict(all_tools[self.remote_name])

        resolved_name = str(name or self.remote_name).strip()
        if not resolved_name:
            raise ToolDefinitionError("name must resolve to a non-empty string.")

        explicit_description = str(description or "").strip()
        remote_description = str(self._mcpdata.get("description") or "").strip()
        resolved_description = (
            explicit_description
            or remote_description
            or f"MCP proxy tool '{resolved_name}'"
        )

        function = functools.partial(self.client_hub.call_tool, self.remote_name)

        super().__init__(
            function=function,
            name=resolved_name,
            namespace=namespace or "mcp",
            description=resolved_description,
            filter_extraneous_inputs=filter_extraneous_inputs,
        )

    @property
    def client_hub(self) -> MCPClientHub:
        return self._client_hub

    @property
    def remote_name(self) -> str:
        return self._remote_name

    @remote_name.setter
    def remote_name(self, value: str) -> None:
        new_remote_name = str(value).strip()
        if not new_remote_name:
            raise ToolDefinitionError("remote_name must be a non-empty string.")
        if new_remote_name == self._remote_name:
            return

        prior_remote_name = self._remote_name
        self._remote_name = new_remote_name
        try:
            self.refresh()
        except Exception:
            self._remote_name = prior_remote_name
            raise

    @property
    def transport_mode(self) -> Literal["stdio", "sse", "streamable_http"]:
        return self.client_hub.transport_mode

    @property
    def endpoint(self) -> str | None:
        return self.client_hub.endpoint

    @property
    def command(self) -> str | None:
        return self.client_hub.command

    @property
    def args(self) -> tuple[str, ...] | None:
        return self.client_hub.args

    @property
    def headers(self) -> Mapping[str, str] | None:
        return self.client_hub.headers

    @property
    def mcpdata(self) -> Dict[str, Any]:
        return dict(self._mcpdata)

    @property
    def raw_metadata(self) -> Dict[str, Any]:
        raw = self._mcpdata.get("raw_metadata")
        return dict(raw) if isinstance(raw, Mapping) else {}

    def _build_tool_signature(self) -> tuple[list[ParamSpec], str]:
        parameters = self._mcpdata.get("parameters")
        return_type = self._mcpdata.get("return_type")

        if not isinstance(parameters, list) or not all(
            isinstance(param, ParamSpec) for param in parameters
        ):
            raise TypeError(
                f"{type(self).__name__}: MCP metadata parameters must be list[ParamSpec]."
            )
        if not isinstance(return_type, str):
            raise TypeError(
                f"{type(self).__name__}: MCP metadata return_type must be str."
            )

        return list(parameters), return_type

    def _get_mod_qual(
        self,
        function: Callable[..., Any],
    ) -> tuple[Optional[str], Optional[str]]:
        # MCP-backed tools do not point at a stable remote import path.
        return MCPClientHub.call_tool.__module__, MCPClientHub.call_tool.__qualname__

    def to_arg_kwarg(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        # MCP calls are always dict-first.
        return tuple(), dict(inputs)

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        if args:
            raise ToolInvocationError(
                f"{self.full_name}: MCP tools do not accept positional arguments; got {args!r}."
            )

        raw_result = self._function(inputs=kwargs)

        if not isinstance(raw_result, Mapping):
            raise ToolInvocationError(
                f"{self.full_name}: MCP client hub returned a non-mapping result envelope."
            )

        is_error = raw_result.get("isError")
        if is_error is None:
            is_error = False

        if bool(is_error):
            self._raise_mcp_tool_error(raw_result)

        return self._extract_proxy_result(raw_result)

    def _extract_proxy_result(self, raw_result: Mapping[str, Any]) -> Any:
        mode = self._mcpdata.get("extraction_mode")
        if mode not in {"extract_result", "structured_content", "content_blocks"}:
            raise ToolInvocationError(
                f"{self.full_name}: invalid MCP extraction_mode {mode!r}."
            )

        if mode == "extract_result":
            structured = raw_result.get("structuredContent")
            if structured is None:
                raise ToolInvocationError(
                    f"{self.full_name}: structuredContent is required for extract_result mode."
                )
            if not isinstance(structured, Mapping):
                raise ToolInvocationError(
                    f"{self.full_name}: structuredContent must be mapping-like in extract_result mode."
                )
            if "result" not in structured:
                raise ToolInvocationError(
                    f"{self.full_name}: structuredContent is missing required 'result' key."
                )
            return structured["result"]

        if mode == "structured_content":
            structured = raw_result.get("structuredContent")
            if structured is None:
                raise ToolInvocationError(
                    f"{self.full_name}: structuredContent is required for structured_content mode."
                )
            return structured

        if "content" not in raw_result:
            raise ToolInvocationError(
                f"{self.full_name}: MCP client hub result is missing 'content'."
            )
        return raw_result["content"]

    def _raise_mcp_tool_error(self, raw_result: Mapping[str, Any]) -> None:
        # Prefer the structured error payload when present. Otherwise surface the
        # content blocks, and finally the whole envelope.
        if raw_result.get("structuredContent") is not None:
            payload = raw_result["structuredContent"]
        elif raw_result.get("content") is not None:
            payload = raw_result["content"]
        else:
            payload = dict(raw_result)

        raise ToolInvocationError(
            f"{self.full_name}: remote MCP tool reported isError=True. Payload: {payload!r}"
        )

    def refresh(self, headers: Any = NO_VAL) -> None:
        """
        Re-fetch remote metadata and rebuild the local MCP binding.

        Refresh semantics intentionally prefer fresh remote description data.
        If the remote description is missing, fall back to a stub based on the
        AA-facing local tool name.
        """
        if headers is not NO_VAL:
            self.client_hub.headers = headers

        all_tools = self.client_hub.list_tools()
        if self.remote_name not in all_tools:
            raise ToolDefinitionError(
                f"{self.full_name}: remote tool {self.remote_name!r} not found during refresh."
            )

        self._mcpdata = dict(all_tools[self.remote_name])

        remote_description = str(self._mcpdata.get("description") or "").strip()
        self.description = remote_description or f"MCP proxy tool '{self.name}'"

        self._function = functools.partial(self.client_hub.call_tool, self.remote_name)
        self._module, self._qualname = self._get_mod_qual(self._function)
        parameters, return_type = self._build_tool_signature()
        self._parameters = parameters
        self._return_type = return_type

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update(
            {
                "remote_name": self.remote_name,
                "raw_metadata": self.raw_metadata,
                "client_hub": self.client_hub.to_dict(),
            }
        )
        return data