from __future__ import annotations

import functools
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Mapping,
    Optional,
)

from ..exceptions import ToolInvocationError
from ..models.parameters import ParamSpec
from ..mcp.mcp_client_hub import MCPClientHub
from ..models.results.tools import MCPToolResult
from .base import Tool

__all__ = ["MCPProxyTool"]

class MCPProxyTool(Tool):
    """
    Proxy a single remote MCP tool as a normal AA Tool.

    Built from an `MCPClientHub` and the remote tool's name; the tool's
    metadata is read from the hub's stored catalog (`hub.get_tool(name)`, which
    raises `MCPToolError` for an unknown name) with no checks of its own and no
    network call. The Tool's own
    name is `<hub.name>_<name>` (the raw name when the hub is unnamed); the raw
    remote name is kept as `remote_name` and is what calls use. The description
    comes from the remote tool and may be replaced through the `description`
    setter. Only hub-based construction is supported; a different identity
    comes from wrapping the proxy with `to_tool(proxy, name=..., description=...)`.
    """

    result_cls: ClassVar[type[MCPToolResult]] = MCPToolResult

    @classmethod
    def of(cls, fn=None, /, *, name=None, description=None):
        """Not available: a proxy tool is built from a hub, not a function."""
        raise TypeError(
            "MCPProxyTool.of is not supported; build it from an MCPClientHub "
            "(or use to_tool)."
        )

    def __init__(self, client_hub: MCPClientHub, name: str) -> None:
        # State is set before Tool.__init__ runs (it reads _mcpdata). The
        # remote metadata comes from the hub's stored catalog; an unknown
        # name fails in get_tool.
        self._client_hub = client_hub
        self._remote_name = name
        self._mcpdata: Dict[str, Any] = dict(client_hub.get_tool(name))
        tool_name = f"{client_hub.name}_{name}" if client_hub.name is not None else name
        description = (
            str(self._mcpdata.get("description") or "").strip()
            or f"MCP proxy tool '{tool_name}'"
        )
        super().__init__(
            function=functools.partial(client_hub.call_tool, name),
            name=tool_name,
            description=description,
        )

    @property
    def client_hub(self) -> MCPClientHub:
        return self._client_hub

    @property
    def remote_name(self) -> str:
        return self._remote_name

    @property
    def mcpdata(self) -> Dict[str, Any]:
        return dict(self._mcpdata)

    @property
    def raw_metadata(self) -> Dict[str, Any]:
        raw = self._mcpdata.get("raw_metadata")
        return dict(raw) if isinstance(raw, Mapping) else {}

    def _build_tool_signature(self) -> tuple[list[ParamSpec], tuple[str, ...]]:
        parameters = self._mcpdata.get("parameters")
        return_type = self._mcpdata.get("return_type")

        if not isinstance(parameters, list) or not all(
            isinstance(param, ParamSpec) for param in parameters
        ):
            raise TypeError(
                f"{type(self).__name__}: MCP metadata parameters must be list[ParamSpec]."
            )
        if not isinstance(return_type, (str, list, tuple)):
            raise TypeError(
                f"{type(self).__name__}: MCP metadata return_type must be str, list or tuple."
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

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if args:
            raise ToolInvocationError(
                f"{self.full_name}: MCP tools do not accept positional arguments; got {args!r}."
            )

        try:
            raw_result = self._function(inputs=kwargs)
        except ToolInvocationError:
            raise
        except Exception as e:
            raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e

        if not isinstance(raw_result, Mapping):
            raise ToolInvocationError(
                f"{self.full_name}: MCP client hub returned a non-mapping result envelope."
            )

        is_error = raw_result.get("isError")
        if is_error is None:
            is_error = False

        if bool(is_error):
            self._raise_mcp_tool_error(raw_result)

        return {
            "transport_mode": self.client_hub.transport_mode,
            "remote_name": self.remote_name,
            "location": self.client_hub.location,
            "result": self._extract_proxy_result(raw_result),
        }

    async def async_execute(
        self,
        args: tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        if args:
            raise ToolInvocationError(
                f"{self.full_name}: MCP tools do not accept positional arguments; got {args!r}."
            )

        try:
            raw_result = await self.client_hub.async_call_tool(self.remote_name, kwargs)
        except ToolInvocationError:
            raise
        except Exception as exc:
            raise ToolInvocationError(
                f"{self.full_name}: async MCP invocation failed: {exc}"
            ) from exc

        if not isinstance(raw_result, Mapping):
            raise ToolInvocationError(
                f"{self.full_name}: MCP client hub returned a non-mapping result envelope."
            )

        is_error = raw_result.get("isError")
        if is_error is None:
            is_error = False

        if bool(is_error):
            self._raise_mcp_tool_error(raw_result)

        return {
            "transport_mode": self.client_hub.transport_mode,
            "remote_name": self.remote_name,
            "location": self.client_hub.location,
            "result": self._extract_proxy_result(raw_result),
        }

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