from __future__ import annotations

import asyncio
import atexit
import logging
import threading
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
from ..utils.core import (
    normalize_headers,
    run_coro_async,
    run_coro_sync,
    start_background_loop,
    stop_background_loop,
)
from ..utils.mcp import (
    _build_mcp_tool_metadata,
    _normalize_mcp_call_result,
)


__all__ = ["MCPClientHub"]

logger = logging.getLogger(__name__)


class MCPClientHub:
    """
    MCP transport/session hub, with an explicit connection-lifetime choice.

    persistent=False: every list_tools()/call_tool() opens a fresh transport
    + ClientSession, uses it once, and tears it down -- unchanged from this
    class's original behavior.

    persistent=True: the transport + ClientSession are opened once, at
    construction, and held open across every subsequent call via one
    dedicated background event loop (utils/core.py's start_background_loop),
    for the identical cross-event-loop-safety reason A2AClientHub's own
    persistent gRPC channel needs one: ClientSession and every MCP transport
    (stdio/sse/streamable_http) are anyio task-group-owning resources whose
    live background tasks are pinned to the loop that created them.
    close()/async_close() are real, idempotent teardown; a persistent hub
    that's never explicitly closed still gets torn down automatically when
    the process exits normally (atexit-registered at construction), so a
    forgotten close() doesn't orphan the underlying stdio subprocess.

    Immutable transport identity: transport_mode, endpoint, command, args,
    persistent. Mutable request/session configuration: headers (direct
    setter, or via refresh()), client_kwargs/session_kwargs (via refresh()
    only). read_timeout_seconds is fixed at construction.
    """

    def __init__(
        self,
        transport_mode: Literal["stdio", "sse", "streamable_http"],
        persistent: bool,
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

        if not isinstance(persistent, bool):
            raise TypeError("persistent must be a bool.")

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
        self._persistent: bool = persistent

        # Persistent-connection state -- None/absent unless persistent=True.
        # No self._exit_stack field: the AsyncExitStack that opens the
        # transport/session lives entirely inside _run_persistent_session's
        # own local scope, since it must be entered *and* exited by that one
        # task (see _run_persistent_session's docstring). self._close_event/
        # self._holder_task are the external handles used to ask that task
        # to close and to know when it actually has.
        self._bg_loop: asyncio.AbstractEventLoop | None = None
        self._bg_thread: threading.Thread | None = None
        self._close_event: asyncio.Event | None = None
        self._holder_task: "asyncio.Task[None] | None" = None
        self._session: ClientSession | None = None
        # Created unconditionally regardless of mode -- modern asyncio.Lock
        # is not loop-bound at construction, so this is safe to create here
        # even though it may later be awaited from the background loop's
        # own thread. Guards close()/refresh() against racing each other.
        self._lock: asyncio.Lock = asyncio.Lock()

        if persistent:
            self._bg_loop, self._bg_thread = start_background_loop()
            try:
                self._session, self._close_event, self._holder_task = run_coro_sync(
                    self._spawn_holder(
                        self._headers, self._client_kwargs, self._session_kwargs
                    ),
                    loop=self._bg_loop,
                )
            except Exception:
                stop_background_loop(self._bg_loop, self._bg_thread)
                self._bg_loop = None
                self._bg_thread = None
                raise
            # Self-contained safety net: if the process exits normally
            # without an explicit close(), still tear down the held
            # connection (and any spawned stdio subprocess) rather than
            # orphaning it. Unregistered by close() itself.
            atexit.register(self.close)

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

    @property
    def persistent(self) -> bool:
        return self._persistent

    @property
    def is_connected(self) -> bool:
        """
        True iff this hub currently holds a live, open session.

        Always False for a non-persistent hub (nothing is held between
        calls) and False for a persistent hub after close().
        """
        return self._session is not None

    def refresh(
        self,
        headers: Mapping[str, HeaderValue] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        session_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Update stored request/session configuration.

        Each provided bucket wholesale-replaces its stored value (never
        merged). Raises if none of headers/client_kwargs/session_kwargs are
        provided, since a no-op refresh() is a caller bug. Validation runs
        against the resolved candidate values before anything is committed,
        so a failed refresh() leaves stored config untouched.

        Non-persistent hubs (and persistent hubs after close()): pure local
        mutation, no I/O -- there is no live connection to touch.

        Persistent, connected hubs: sse/streamable_http reconnect
        transparently. stdio fully restarts the underlying server process
        (a warning is logged when this happens) -- there is no cheaper way
        to apply new stdio launch config to an already-running process.
        """
        if headers is None and client_kwargs is None and session_kwargs is None:
            raise ValueError(
                "refresh() requires at least one of headers, client_kwargs, "
                "or session_kwargs; none were provided."
            )

        if self._bg_loop is None:
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
            return

        run_coro_sync(
            self._do_refresh(headers, client_kwargs, session_kwargs), loop=self._bg_loop
        )

    async def _do_refresh(
        self,
        headers: Mapping[str, HeaderValue] | None,
        client_kwargs: Mapping[str, Any] | None,
        session_kwargs: Mapping[str, Any] | None,
    ) -> None:
        """Persistent-mode refresh core. Guarded by self._lock so a
        concurrent refresh() or close() can't interleave inconsistently.
        Builds and validates the new connection before committing it --
        a failure here leaves stored config and the live session untouched."""
        async with self._lock:
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

            if self._transport_mode == "stdio":
                logger.warning(
                    "Refreshing a persistent stdio MCPClientHub restarts the "
                    "underlying server process; any process-local state it "
                    "held is discarded."
                )

            new_session, new_close_event, new_holder_task = await self._spawn_holder(
                resolved_headers, resolved_client_kwargs, resolved_session_kwargs
            )

            old_close_event, old_holder_task = self._close_event, self._holder_task
            self._session = new_session
            self._close_event = new_close_event
            self._holder_task = new_holder_task
            self._headers = resolved_headers
            self._client_kwargs = resolved_client_kwargs
            self._session_kwargs = resolved_session_kwargs

            # Retire the old connection through its own holder task -- the
            # same task-affinity requirement close() has to respect applies
            # here too (see _run_persistent_session's docstring).
            old_close_event.set()
            try:
                await old_holder_task
            except Exception:
                # Best-effort; the refresh itself already succeeded, a
                # failure closing the now-discarded old connection shouldn't
                # surface as a refresh() failure.
                pass

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
            "persistent": self.persistent,
            "is_connected": self.is_connected,
        }

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

    async def _connect_session(
        self,
        headers: Mapping[str, str] | None,
        client_kwargs: Mapping[str, Any] | None,
        session_kwargs: Mapping[str, Any] | None,
    ) -> tuple[AsyncExitStack, ClientSession]:
        """
        Build one fresh (AsyncExitStack, ClientSession) pair, without
        closing it. Takes headers/client_kwargs/session_kwargs explicitly
        (rather than always reading self.*) so refresh() can build-and-
        validate a candidate connection before committing it.

        The AsyncExitStack is entered manually (not via `async with`) so it
        can outlive this coroutine call -- callers that want a fresh,
        immediately-torn-down connection are responsible for closing it
        themselves (see _do_operation's non-persistent path).
        """
        transport_kwargs = dict(client_kwargs) if client_kwargs is not None else {}

        stack = AsyncExitStack()
        await stack.__aenter__()

        try:
            if self.transport_mode == "stdio":
                server_params = StdioServerParameters(
                    command=self.command,
                    args=list(self.args or ()),
                    **transport_kwargs,
                )
                client_context = stdio_client(server_params)

            elif self.transport_mode == "sse":
                headers_dict = dict(headers) if headers is not None else None
                client_context = sse_client(
                    url=self.endpoint,
                    headers=headers_dict,
                    **transport_kwargs,
                )

            else:
                if headers is not None:
                    http_client = await stack.enter_async_context(
                        httpx.AsyncClient(
                            headers=dict(headers),
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

            resolved_session_kwargs = dict(session_kwargs) if session_kwargs is not None else {}
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
                    **resolved_session_kwargs,
                )
            )
            await session.initialize()
            return stack, session

        except (MCPConnectionError, MCPToolError):
            # AsyncExitStack's manual/unscoped form does not auto-unwind
            # already-entered contexts when a later enter_async_context call
            # raises -- unwind explicitly before propagating.
            await stack.aclose()
            raise
        except Exception as exc:
            await stack.aclose()
            raise MCPConnectionError(
                f"Failed to establish MCP session ({self.transport_mode}): {exc}"
            ) from exc

    async def _run_persistent_session(
        self,
        headers: Mapping[str, str] | None,
        client_kwargs: Mapping[str, Any] | None,
        session_kwargs: Mapping[str, Any] | None,
        ready: asyncio.Event,
        close_event: asyncio.Event,
        result_box: Dict[str, Any],
    ) -> None:
        """
        Owns one persistent session's entire open-through-close lifecycle
        within a single asyncio Task.

        ClientSession and every MCP transport (stdio/sse/streamable_http)
        are built on anyio task groups, whose cancel scopes may only be
        exited by the same Task that entered them -- confirmed by direct
        reproduction: dispatching "open" (at construction) and "close" (at
        a later close()/refresh() call) as two independently-dispatched
        coroutines via run_coroutine_threadsafe, each becoming its own Task,
        raised "Attempted to exit cancel scope in a different task than it
        was entered in" the moment the second one tried to close what the
        first one opened. This is stricter than the same-*loop* requirement
        A2AClientHub's gRPC/httpx resources have -- those aren't anyio
        task-group-owning, so A2AClientHub's close() dispatching as its own
        separate task is safe there but not transferable here.

        Stays alive (parked on close_event.wait()) after signaling ready,
        so the *same* task that opened the connection is still the one
        running when it's later asked to close it.
        """
        try:
            stack, session = await self._connect_session(headers, client_kwargs, session_kwargs)
        except Exception as exc:
            result_box["error"] = exc
            ready.set()
            return

        result_box["session"] = session
        ready.set()
        await close_event.wait()
        await stack.aclose()

    async def _spawn_holder(
        self,
        headers: Mapping[str, str] | None,
        client_kwargs: Mapping[str, Any] | None,
        session_kwargs: Mapping[str, Any] | None,
    ) -> tuple[ClientSession, asyncio.Event, "asyncio.Task[None]"]:
        """
        Must be called from a coroutine already running on self._bg_loop
        (dispatched there via run_coro_sync/run_coro_async, or awaited
        directly from another coroutine already running on it, e.g.
        _do_refresh).

        Starts a new _run_persistent_session task and waits for it to
        either connect successfully or fail. On success, returns
        (session, close_event, task): the caller commits `session` as the
        live one, and holds onto close_event/task to signal and await a
        later close (see close()/_do_close() and _do_refresh()). On
        failure, raises the already-typed MCPConnectionError directly --
        the task has already returned by that point, nothing further to
        signal or await.
        """
        close_event = asyncio.Event()
        ready = asyncio.Event()
        result_box: Dict[str, Any] = {}

        task = asyncio.create_task(
            self._run_persistent_session(
                headers, client_kwargs, session_kwargs, ready, close_event, result_box
            )
        )
        await ready.wait()

        if "error" in result_box:
            raise result_box["error"]

        return result_box["session"], close_event, task

    async def _do_operation(
        self,
        operation: Callable[[ClientSession], Awaitable[T]],
    ) -> T:
        """
        Run operation against the live, held-open session (persistent mode,
        currently connected), or against a fresh session that's opened,
        used, and closed within this one call (non-persistent mode, or a
        persistent hub after close()).

        A typed MCPConnectionError/MCPToolError raised by operation is
        already meaningful and passes through unwrapped; any other
        exception is wrapped as MCPConnectionError -- the same blanket
        safety net the old _awith_session provided around every operation
        it ran, preserved here even though list_tools()/call_tool()'s own
        `_op` closures already wrap their specific failure points too.
        """
        if self._session is not None:
            return await self._run_operation(operation, self._session)

        stack, session = await self._connect_session(
            self._headers, self._client_kwargs, self._session_kwargs
        )
        try:
            return await self._run_operation(operation, session)
        finally:
            await stack.aclose()

    async def _run_operation(
        self,
        operation: Callable[[ClientSession], Awaitable[T]],
        session: ClientSession,
    ) -> T:
        try:
            return await operation(session)
        except (MCPConnectionError, MCPToolError):
            raise
        except Exception as exc:
            raise MCPConnectionError(
                f"Failed MCP operation ({self.transport_mode}): {exc}"
            ) from exc

    def close(self) -> None:
        """
        Real teardown for persistent mode; a no-op for non-persistent
        (nothing held between calls). Idempotent -- self._bg_loop is None
        both before construction ever ran in persistent mode and after this
        has already run once, so a second call is a safe no-op.
        """
        if self._bg_loop is None:
            return
        run_coro_sync(self._do_close(), loop=self._bg_loop)
        stop_background_loop(self._bg_loop, self._bg_thread)
        atexit.unregister(self.close)
        self._bg_loop = None
        self._bg_thread = None
        self._close_event = None
        self._holder_task = None
        self._session = None

    async def async_close(self) -> None:
        """Async counterpart to close()."""
        if self._bg_loop is None:
            return
        await run_coro_async(self._do_close(), loop=self._bg_loop)
        stop_background_loop(self._bg_loop, self._bg_thread)
        atexit.unregister(self.close)
        self._bg_loop = None
        self._bg_thread = None
        self._close_event = None
        self._holder_task = None
        self._session = None

    async def _do_close(self) -> None:
        """
        Guarded by self._lock so close() can't race an in-flight refresh().
        Signals the holder task to proceed past its close_event.wait() and
        awaits it -- the actual stack.aclose() runs inside that task, the
        same one that originally opened it (see _run_persistent_session's
        docstring for why that matters).
        """
        async with self._lock:
            self._close_event.set()
            await self._holder_task

    def __enter__(self) -> "MCPClientHub":
        """No-op: persistent mode is already connected by construction;
        non-persistent mode has nothing to open. Returns self either way."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Calls close() unconditionally; a no-op for non-persistent hubs
        via close()'s own guard. No async counterpart -- matches
        A2AClientHub's own precedent of cutting async context-manager
        surface (__aenter__/__aexit__/async_close as a dunder pair) as
        unneeded."""
        self.close()

    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """Sync entry point -- dispatches onto the background loop when
        persistent and connected, else a fresh temporary loop per call."""
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

        return run_coro_sync(self._do_operation(_op), loop=self._bg_loop)

    def call_tool(
        self,
        remote_name: str,
        inputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Sync entry point -- dispatches onto the background loop when
        persistent and connected, else a fresh temporary loop per call."""
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

        return run_coro_sync(self._do_operation(_op), loop=self._bg_loop)

    async def async_list_tools(self) -> Dict[str, Dict[str, Any]]:
        """Native async implementation -- public so async callers can await
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

        if self._bg_loop is not None:
            return await run_coro_async(self._do_operation(_op), loop=self._bg_loop)
        return await self._do_operation(_op)

    async def async_call_tool(
        self,
        remote_name: str,
        inputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Native async implementation -- public so async callers can await
        it directly instead of paying for run_coro_sync's thread bridging.
        Owns input validation; call_tool no longer duplicates it."""
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

        if self._bg_loop is not None:
            return await run_coro_async(self._do_operation(_op), loop=self._bg_loop)
        return await self._do_operation(_op)
