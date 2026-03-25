from __future__ import annotations

from contextlib import AsyncExitStack
from types import MappingProxyType
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Literal,
    Mapping,
)

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from .utils import (
    _run_coro_sync,
    _build_mcp_tool_metadata,
    _normalize_mcp_call_result,
    T,
    ) 

__all__ = ["MCPClientHub"]

class MCPClientHub:
    """
    Stateless MCP transport/session hub.

    Public methods are synchronous, but each operation internally opens,
    initializes, uses, and closes an MCP session inside a single coroutine.

    Immutable transport identity:
    - transport_mode
    - endpoint
    - command
    - args

    Mutable request state:
    - headers
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
        transport_mode: Literal["stdio", "sse", "streamable_http"],
        endpoint: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        mode = str(transport_mode).strip()
        if mode not in {"stdio", "sse", "streamable_http"}:
            raise ValueError(
                "transport_mode must be one of: 'stdio', 'sse', 'streamable_http'."
            )

        normalized_endpoint: str | None = None
        if endpoint is not None:
            if not isinstance(endpoint, str):
                raise ValueError("endpoint must be a string when provided.")
            normalized_endpoint = endpoint.strip() or None

        normalized_command: str | None = None
        if command is not None:
            if not isinstance(command, str):
                raise ValueError("command must be a string when provided.")
            normalized_command = command.strip() or None

        normalized_args: tuple[str, ...] | None = None
        if args is not None:
            if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
                raise ValueError("args must be a list of strings when provided.")
            normalized_args = tuple(args)

        normalized_headers = self._normalize_headers(headers)

        if mode == "stdio" and not normalized_command:
            raise ValueError("stdio transport requires a non-empty command string.")
        if mode in {"sse", "streamable_http"} and not normalized_endpoint:
            raise ValueError(f"{mode} transport requires a non-empty endpoint string.")

        self._transport_mode: Literal["stdio", "sse", "streamable_http"] = mode
        self._endpoint: str | None = normalized_endpoint
        self._command: str | None = normalized_command
        self._args: tuple[str, ...] | None = normalized_args
        self._headers: Mapping[str, str] | None = normalized_headers

    @property
    def transport_mode(self) -> Literal["stdio", "sse", "streamable_http"]:
        return self._transport_mode

    @property
    def endpoint(self) -> str | None:
        return self._endpoint

    @property
    def command(self) -> str | None:
        return self._command

    @property
    def args(self) -> tuple[str, ...] | None:
        return self._args

    @property
    def headers(self) -> Mapping[str, str] | None:
        return self._headers

    @headers.setter
    def headers(self, value: Mapping[str, str] | None) -> None:
        self._headers = self._normalize_headers(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transport_mode": self.transport_mode,
            "endpoint": self.endpoint,
            "command": self.command,
            "args": list(self.args) if self.args is not None else None,
            "has_headers": self.headers is not None,
            "header_keys": sorted(self.headers.keys()) if self.headers is not None else [],
        }

    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        return _run_coro_sync(self._alist_tools())

    def call_tool(
        self,
        remote_name: str,
        inputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        resolved_remote_name = str(remote_name).strip()
        if not resolved_remote_name:
            raise RuntimeError("remote_name must be a non-empty string.")
        if not isinstance(inputs, Mapping):
            raise RuntimeError("inputs must be a mapping.")

        return _run_coro_sync(self._acall_tool(resolved_remote_name, inputs))

    def _unpack_transport_streams(self, transport: Any) -> tuple[Any, Any]:
        if isinstance(transport, tuple):
            if len(transport) == 2:
                read_stream, write_stream = transport
                return read_stream, write_stream
            if len(transport) == 3:
                read_stream, write_stream, _ = transport
                return read_stream, write_stream

        raise RuntimeError(
            f"Unexpected transport stream shape for {self.transport_mode}: {type(transport)!r}"
        )

    async def _awith_session(
        self,
        operation: Callable[[ClientSession], Awaitable[T]],
    ) -> T:
        try:
            async with AsyncExitStack() as stack:
                if self.transport_mode == "stdio":
                    if self.command is None:
                        raise RuntimeError("stdio transport requires command.")

                    server_params = StdioServerParameters(
                        command=self.command,
                        args=list(self.args or ()),
                    )
                    client_context = stdio_client(server_params)

                elif self.transport_mode == "sse":
                    if self.endpoint is None:
                        raise RuntimeError("sse transport requires endpoint.")

                    headers_dict = dict(self.headers) if self.headers is not None else None
                    client_context = sse_client(url=self.endpoint, headers=headers_dict)

                else:
                    if self.endpoint is None:
                        raise RuntimeError("streamable_http transport requires endpoint.")

                    if self.headers is not None:
                        http_client = await stack.enter_async_context(
                            httpx.AsyncClient(
                                headers=dict(self.headers),
                                follow_redirects=True,
                            )
                        )
                        client_context = streamable_http_client(
                            url=self.endpoint,
                            http_client=http_client,
                        )
                    else:
                        client_context = streamable_http_client(url=self.endpoint)

                transport = await stack.enter_async_context(client_context)
                read_stream, write_stream = self._unpack_transport_streams(transport)

                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                return await operation(session)

        except Exception as exc:
            raise RuntimeError(
                f"Failed MCP operation ({self.transport_mode}): {exc}"
            ) from exc

    async def _alist_tools(self) -> Dict[str, Dict[str, Any]]:
        async def _op(session: ClientSession) -> Dict[str, Dict[str, Any]]:
            try:
                tools_result = await session.list_tools()
            except Exception as exc:
                raise RuntimeError(f"Failed to list MCP tools: {exc}") from exc

            raw_tools = getattr(tools_result, "tools", tools_result)
            result: dict[str, dict[str, Any]] = {}

            if raw_tools is None:
                return result

            for raw_tool in raw_tools:
                result[raw_tool.name] = _build_mcp_tool_metadata(raw_tool)

            return result

        return await self._awith_session(_op)

    async def _acall_tool(
        self,
        remote_name: str,
        inputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        async def _op(session: ClientSession) -> Dict[str, Any]:
            try:
                raw_result = await session.call_tool(
                    remote_name,
                    arguments=dict(inputs),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to call MCP tool '{remote_name}': {exc}"
                ) from exc

            return _normalize_mcp_call_result(raw_result)

        return await self._awith_session(_op)
