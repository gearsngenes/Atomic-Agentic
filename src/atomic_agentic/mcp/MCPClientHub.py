from __future__ import annotations

from contextlib import AsyncExitStack
from datetime import timedelta
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

from ..constants.core import HeaderValue, T
from ..exceptions import MCPConnectionError, MCPToolError
from ..utils.core import run_coro_sync, normalize_headers
from ..utils.mcp import (
    _build_mcp_tool_metadata,
    _normalize_mcp_call_result,
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

    Mutable request/session configuration:
    - headers (direct setter, or via refresh())
    - client_kwargs, session_kwargs (via refresh() only)

    read_timeout_seconds is fixed at construction: no setter, and not
    accepted by refresh().
    """

    def __init__(
        self,
        transport_mode: Literal["stdio", "sse", "streamable_http"],
        endpoint: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        headers: Mapping[str, HeaderValue] | None = None,
        read_timeout_seconds: float | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        session_kwargs: Mapping[str, Any] | None = None,
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

        normalized_headers = normalize_headers(headers)

        if mode == "stdio" and not normalized_command:
            raise ValueError("stdio transport requires a non-empty command string.")
        if mode in {"sse", "streamable_http"} and not normalized_endpoint:
            raise ValueError(f"{mode} transport requires a non-empty endpoint string.")

        normalized_read_timeout_seconds: float | None = (
            float(read_timeout_seconds) if read_timeout_seconds is not None else None
        )
        normalized_client_kwargs = self._normalize_kwargs_mapping(
            client_kwargs, param_name="client_kwargs"
        )
        normalized_session_kwargs = self._normalize_kwargs_mapping(
            session_kwargs, param_name="session_kwargs"
        )

        self._validate_streamable_http_collision(
            transport_mode=mode,
            headers=normalized_headers,
            client_kwargs=normalized_client_kwargs,
        )

        self._transport_mode: Literal["stdio", "sse", "streamable_http"] = mode
        self._endpoint: str | None = normalized_endpoint
        self._command: str | None = normalized_command
        self._args: tuple[str, ...] | None = normalized_args
        self._headers: Mapping[str, str] | None = normalized_headers
        self._read_timeout_seconds: float | None = normalized_read_timeout_seconds
        self._client_kwargs: Dict[str, Any] | None = normalized_client_kwargs
        self._session_kwargs: Dict[str, Any] | None = normalized_session_kwargs

    @staticmethod
    def _normalize_kwargs_mapping(
        value: Mapping[str, Any] | None,
        *,
        param_name: str,
    ) -> Dict[str, Any] | None:
        """Validate an escape-hatch kwargs bucket is a mapping (or None)."""
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError(f"{param_name} must be a mapping when provided.")
        return dict(value)

    @staticmethod
    def _validate_streamable_http_collision(
        *,
        transport_mode: str,
        headers: Mapping[str, str] | None,
        client_kwargs: Mapping[str, Any] | None,
    ) -> None:
        """Raise if headers and client_kwargs['http_client'] both target the
        same auth surface under transport_mode="streamable_http". Takes
        explicit candidate values (not self.*) so callers can validate
        before committing a mutation."""
        if (
            transport_mode == "streamable_http"
            and headers is not None
            and client_kwargs is not None
            and "http_client" in client_kwargs
        ):
            raise ValueError(
                "Cannot set both headers and client_kwargs['http_client'] "
                "under transport_mode='streamable_http'; both configure the "
                "same underlying auth surface."
            )

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
    def headers(self, value: Mapping[str, HeaderValue] | None) -> None:
        self._headers = normalize_headers(value)

    @property
    def read_timeout_seconds(self) -> float | None:
        return self._read_timeout_seconds

    @property
    def client_kwargs(self) -> Mapping[str, Any] | None:
        """Read-only view; mutate via refresh(client_kwargs=...)."""
        return MappingProxyType(self._client_kwargs) if self._client_kwargs is not None else None

    @property
    def session_kwargs(self) -> Mapping[str, Any] | None:
        """Read-only view; mutate via refresh(session_kwargs=...)."""
        return MappingProxyType(self._session_kwargs) if self._session_kwargs is not None else None

    def refresh(
        self,
        headers: Mapping[str, HeaderValue] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        session_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Update stored request/session configuration.

        MCP sessions open and close per invocation — there is no persistent
        connection to refresh, only stored config consumed by the next call.
        Each provided bucket wholesale-replaces its stored value (never
        merged). Raises if none of headers/client_kwargs/session_kwargs are
        provided, since a no-op refresh() is a caller bug. Validation runs
        against the resolved candidate values before anything is committed,
        so a failed refresh() leaves stored config untouched.
        """
        if headers is None and client_kwargs is None and session_kwargs is None:
            raise ValueError(
                "refresh() requires at least one of headers, client_kwargs, "
                "or session_kwargs; none were provided."
            )

        resolved_headers = (
            normalize_headers(headers) if headers is not None else self._headers
        )
        resolved_client_kwargs = (
            self._normalize_kwargs_mapping(client_kwargs, param_name="client_kwargs")
            if client_kwargs is not None
            else self._client_kwargs
        )
        resolved_session_kwargs = (
            self._normalize_kwargs_mapping(session_kwargs, param_name="session_kwargs")
            if session_kwargs is not None
            else self._session_kwargs
        )

        self._validate_streamable_http_collision(
            transport_mode=self._transport_mode,
            headers=resolved_headers,
            client_kwargs=resolved_client_kwargs,
        )

        self._headers = resolved_headers
        self._client_kwargs = resolved_client_kwargs
        self._session_kwargs = resolved_session_kwargs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transport_mode": self.transport_mode,
            "endpoint": self.endpoint,
            "command": self.command,
            "args": list(self.args) if self.args is not None else None,
            "has_headers": self.headers is not None,
            "header_keys": sorted(self.headers.keys()) if self.headers is not None else [],
            "read_timeout_seconds": self.read_timeout_seconds,
            "has_client_kwargs": self._client_kwargs is not None,
            "client_kwargs_keys": (
                sorted(self._client_kwargs.keys()) if self._client_kwargs is not None else []
            ),
            "has_session_kwargs": self._session_kwargs is not None,
            "session_kwargs_keys": (
                sorted(self._session_kwargs.keys()) if self._session_kwargs is not None else []
            ),
        }

    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """Sync wrapper over async_list_tools()."""
        return run_coro_sync(self.async_list_tools())

    def call_tool(
        self,
        remote_name: str,
        inputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Sync wrapper over async_call_tool(), which owns input validation."""
        return run_coro_sync(self.async_call_tool(remote_name, inputs))

    def _unpack_transport_streams(self, transport: Any) -> tuple[Any, Any]:
        if isinstance(transport, tuple):
            if len(transport) == 2:
                read_stream, write_stream = transport
                return read_stream, write_stream
            if len(transport) == 3:
                read_stream, write_stream, _ = transport
                return read_stream, write_stream

        raise MCPConnectionError(
            f"Unexpected transport stream shape for {self.transport_mode}: {type(transport)!r}"
        )

    async def _awith_session(
        self,
        operation: Callable[[ClientSession], Awaitable[T]],
    ) -> T:
        try:
            async with AsyncExitStack() as stack:
                transport_kwargs = dict(self._client_kwargs) if self._client_kwargs is not None else {}

                if self.transport_mode == "stdio":
                    server_params = StdioServerParameters(
                        command=self.command,
                        args=list(self.args or ()),
                        **transport_kwargs,
                    )
                    client_context = stdio_client(server_params)

                elif self.transport_mode == "sse":
                    headers_dict = dict(self.headers) if self.headers is not None else None
                    client_context = sse_client(
                        url=self.endpoint,
                        headers=headers_dict,
                        **transport_kwargs,
                    )

                else:
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
                            **transport_kwargs,
                        )
                    else:
                        client_context = streamable_http_client(
                            url=self.endpoint,
                            **transport_kwargs,
                        )

                transport = await stack.enter_async_context(client_context)
                read_stream, write_stream = self._unpack_transport_streams(transport)

                session_kwargs = dict(self._session_kwargs) if self._session_kwargs is not None else {}
                resolved_timeout = (
                    timedelta(seconds=self._read_timeout_seconds)
                    if self._read_timeout_seconds is not None
                    else None
                )
                session = await stack.enter_async_context(
                    ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=resolved_timeout,
                        **session_kwargs,
                    )
                )
                await session.initialize()
                return await operation(session)

        except (MCPConnectionError, MCPToolError):
            # Already typed by `operation` (list/call_tool) — propagate as-is
            # rather than re-wrapping into a generic connection error.
            raise
        except Exception as exc:
            raise MCPConnectionError(
                f"Failed MCP operation ({self.transport_mode}): {exc}"
            ) from exc

    async def async_list_tools(self) -> Dict[str, Dict[str, Any]]:
        """Native async implementation — public so async callers can await
        it directly instead of paying for run_coro_sync's thread bridging."""
        async def _op(session: ClientSession) -> Dict[str, Dict[str, Any]]:
            try:
                tools_result = await session.list_tools()
            except Exception as exc:
                raise MCPConnectionError(f"Failed to list MCP tools: {exc}") from exc

            raw_tools = getattr(tools_result, "tools", tools_result)
            result: dict[str, dict[str, Any]] = {}

            if raw_tools is None:
                return result

            for raw_tool in raw_tools:
                result[raw_tool.name] = _build_mcp_tool_metadata(raw_tool)

            return result

        return await self._awith_session(_op)

    async def async_call_tool(
        self,
        remote_name: str,
        inputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Native async implementation — public so async callers can await
        it directly instead of paying for run_coro_sync's thread bridging.
        Owns input validation; `call_tool` no longer duplicates it."""
        resolved_remote_name = str(remote_name).strip()
        if not resolved_remote_name:
            raise ValueError("remote_name must be a non-empty string.")
        if not isinstance(inputs, Mapping):
            raise ValueError("inputs must be a mapping.")

        async def _op(session: ClientSession) -> Dict[str, Any]:
            try:
                raw_result = await session.call_tool(
                    resolved_remote_name,
                    arguments=dict(inputs),
                )
            except Exception as exc:
                raise MCPToolError(
                    f"Failed to call MCP tool '{resolved_remote_name}': {exc}"
                ) from exc

            return _normalize_mcp_call_result(raw_result)

        return await self._awith_session(_op)
