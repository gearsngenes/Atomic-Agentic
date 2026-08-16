from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import grpc
import httpx
from a2a.client import Client, ClientConfig, create_client
from a2a.client.errors import A2AClientError, A2AClientTimeoutError, AgentCardResolutionError
from a2a.helpers import new_message
from a2a.types import AgentCard, Part, Role, SendMessageRequest, StreamResponse, TaskState

from ..constants.a2a_sdk import TERMINAL_FAILURE_STATES, TRANSPORT_GRPC, VALID_TRANSPORT_MODES
from ..constants.core import HeaderValue
from ..exceptions import A2AProxyError
from ..utils.core import normalize_headers, run_coro_async, run_coro_sync, start_background_loop, stop_background_loop

__all__ = ["A2AClientHub"]


class A2AClientHub:
    """
    Connectivity to one remote A2A server, over the caller's choice of
    JSON-RPC, REST (HTTP+JSON), or gRPC -- and the caller's choice of
    connection lifetime via the required `persistent` knob.

    Purely mechanical: accepts and returns literal a2a-sdk Part objects with
    zero Python-native type dispatch (Pass 3's A2AProxyTool owns that).

    persistent=True: one connection is opened at construction and held for
    the instance's whole lifetime. Owns one background (loop, thread) pair
    via start_background_loop(); every async operation dispatches onto that
    same loop via run_coro_sync(..., loop=self._bg_loop) / run_coro_async(...,
    loop=self._bg_loop) -- required because a grpc.aio.Channel breaks when
    reused across independently-created event loops (confirmed by direct
    reproduction), and this is what keeps a persistent gRPC channel bound to
    one stable loop for its whole life. close() is the real, necessary
    teardown; an unclosed persistent hub leaks a live background thread +
    loop + connection for the process's remaining life (the thread is a
    daemon, so it won't hang process exit). close() is idempotent, and after
    it runs this hub transparently behaves like a non-persistent one for any
    further calls (self._bg_loop is None is the single source of truth,
    checked the same way by send_parts/async_send_parts/_do_refresh).

    persistent=False: no connection is held between calls (httpx.AsyncClient
    and grpc.aio.Channel are both terminal once closed -- there is no
    keep-the-instance-reopen-it path in the underlying libraries). Each
    send_parts call opens, uses, and closes its own connection within one
    run_coro_sync call, safe because nothing outlives that one call.
    agent_card is still resolved and cached once at construction regardless
    of mode.

    transport_mode/base_url/persistent are immutable identity, fixed at
    construction -- constructing a new instance is the only way to change
    any of them. headers/timeout are mutable only via refresh().
    """

    def __init__(
        self,
        base_url: str,
        transport_mode: str,
        persistent: bool,
        headers: Mapping[str, HeaderValue] | None = None,
        timeout: float = 600,
        relative_card_path: str | None = None,
    ) -> None:
        """
        Validate identity/config, then build the underlying connection.

        1. Validate base_url is non-empty.
        2. Validate transport_mode against VALID_TRANSPORT_MODES.
        3. Validate persistent is a bool.
        4-8. Normalize and store headers/timeout/relative_card_path, the
           frozen identity triple (base_url/transport_mode/persistent), and
           an unconditional refresh lock.
        9. persistent=True: start_background_loop(), connect on it, roll the
           loop back if connection fails. persistent=False: self._bg_loop/
           _bg_thread/_client are all None; a plain run_coro_sync
           connect/resolve/close cycle resolves only the agent card.
        """
        resolved_base_url = str(base_url).strip()
        if not resolved_base_url:
            raise ValueError("base_url must be a non-empty string.")

        mode = str(transport_mode).strip()
        if mode not in VALID_TRANSPORT_MODES:
            raise ValueError(
                f"transport_mode must be one of {VALID_TRANSPORT_MODES!r}; got {transport_mode!r}."
            )

        if not isinstance(persistent, bool):
            raise TypeError("persistent must be a bool.")

        self._headers: Mapping[str, str] | None = normalize_headers(headers)
        self._timeout: float = float(timeout)
        self._relative_card_path: str | None = relative_card_path
        self._base_url: str = resolved_base_url
        self._transport_mode: str = mode
        self._persistent: bool = persistent
        self._refresh_lock: asyncio.Lock = asyncio.Lock()

        if persistent:
            self._bg_loop, self._bg_thread = start_background_loop()
            try:
                self._client, self._agent_card = run_coro_sync(
                    self._connect(self._headers, self._timeout), loop=self._bg_loop
                )
            except Exception:
                stop_background_loop(self._bg_loop, self._bg_thread)
                raise
        else:
            self._bg_loop = None
            self._bg_thread = None
            self._client = None
            self._agent_card = run_coro_sync(self._resolve_card(self._headers, self._timeout))

    @classmethod
    async def async_create(
        cls,
        base_url: str,
        transport_mode: str,
        persistent: bool,
        headers: Mapping[str, HeaderValue] | None = None,
        timeout: float = 600,
        relative_card_path: str | None = None,
    ) -> "A2AClientHub":
        """
        Non-blocking construction. __init__ performs blocking async-bridging
        internally (run_coro_sync), so this runs it in a worker thread rather
        than stalling the caller's event loop.
        """
        return await asyncio.to_thread(
            cls,
            base_url,
            transport_mode,
            persistent,
            headers=headers,
            timeout=timeout,
            relative_card_path=relative_card_path,
        )

    async def _connect(
        self,
        headers: Mapping[str, str] | None,
        timeout: float,
    ) -> tuple[Client, AgentCard]:
        """
        Build one fresh Client and resolve its AgentCard.

        Takes headers/timeout explicitly (rather than always reading
        self._headers/self._timeout) so refresh() can build-and-validate a
        candidate connection before committing it.
        """
        httpx_client = httpx.AsyncClient(
            headers=dict(headers) if headers is not None else None,
            timeout=timeout,
        )
        grpc_channel_factory = (
            self._build_grpc_channel_factory() if self._transport_mode == TRANSPORT_GRPC else None
        )
        config = ClientConfig(
            streaming=False,
            polling=False,
            httpx_client=httpx_client,
            supported_protocol_bindings=[self._transport_mode],
            grpc_channel_factory=grpc_channel_factory,
        )
        try:
            client = await create_client(
                self._base_url,
                client_config=config,
                relative_card_path=self._relative_card_path,
            )
        except (
            A2AClientError,
            A2AClientTimeoutError,
            AgentCardResolutionError,
            httpx.HTTPError,
            ValueError,
        ) as exc:
            raise A2AProxyError(
                f"{type(self).__name__}: failed to connect to {self._base_url!r} "
                f"via {self._transport_mode}: {exc}"
            ) from exc

        # create_client(agent=base_url) is used (not a pre-resolved
        # AgentCard) specifically so each transport resolves its card
        # through its own native path with zero AA-side custom
        # per-transport logic; reading the cached result off the private
        # attribute afterward is the only custom part.
        return client, client._card

    async def _resolve_card(
        self,
        headers: Mapping[str, str] | None,
        timeout: float,
    ) -> AgentCard:
        """
        One-shot connect -> resolve -> close, used by non-persistent
        construction to determine the agent card. Kept as a single
        coroutine (one run_coro_sync call) rather than a separate connect
        call followed by a separate close call -- closing a connection
        under a different event loop than the one that opened it breaks
        httpx's underlying transport teardown (confirmed by direct
        reproduction), the same cross-loop hazard that already forced
        persistent mode onto one dedicated background loop.
        """
        client, card = await self._connect(headers, timeout)
        await client.close()
        return card

    def _build_grpc_channel_factory(self) -> Callable[[str], "grpc.aio.Channel"]:
        """
        Build this hub's gRPC channel factory (a2a-sdk ships none).

        Insecure-vs-TLS is inferred once from self._base_url's scheme (bare
        host:port or grpc:// -> insecure; grpcs:///https:// -> system-trust
        TLS) -- not a per-call decision.
        """
        parsed = urlsplit(self._base_url)
        is_secure = parsed.scheme in {"grpcs", "https"}

        def _factory(url: str) -> "grpc.aio.Channel":
            # Strip any scheme prefix the resolved AgentCard's matching gRPC
            # AgentInterface URL may carry -- GrpcTransport.create() calls
            # this factory with whatever URL string that interface
            # declares, which may itself be scheme-less.
            target = urlsplit(url).netloc or url
            if is_secure:
                return grpc.aio.secure_channel(target, grpc.ssl_channel_credentials())
            return grpc.aio.insecure_channel(target)

        return _factory

    @property
    def transport_mode(self) -> str:
        return self._transport_mode

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def persistent(self) -> bool:
        return self._persistent

    @property
    def headers(self) -> Mapping[str, str] | None:
        return self._headers

    @property
    def timeout(self) -> float:
        return self._timeout

    @property
    def agent_card(self) -> AgentCard:
        return self._agent_card

    def refresh(
        self,
        headers: Mapping[str, HeaderValue] | None = None,
        timeout: float | None = None,
    ) -> None:
        """
        Update stored headers/timeout and re-resolve the agent card.

        transport_mode/base_url/persistent are not refreshable -- immutable
        identity; construct a new A2AClientHub to change them. Raises
        ValueError if called with neither argument (a no-op refresh is a
        caller bug).
        """
        if headers is None and timeout is None:
            raise ValueError(
                "refresh() requires at least one of headers or timeout; neither was provided."
            )
        run_coro_sync(self._do_refresh(headers, timeout), loop=self._bg_loop)

    async def _do_refresh(
        self,
        headers: Mapping[str, HeaderValue] | None,
        timeout: float | None,
    ) -> None:
        """
        Core refresh logic, shared by both modes -- guarded by
        self._refresh_lock so two concurrent refresh() calls can't interleave
        inconsistently. Branches on self._bg_loop is not None (not
        self._persistent directly) so a closed persistent hub's refresh()
        degrades the same way send_parts does, instead of touching a client
        that no longer exists.
        """
        async with self._refresh_lock:
            resolved_headers = normalize_headers(headers) if headers is not None else self._headers
            resolved_timeout = float(timeout) if timeout is not None else self._timeout

            new_client, new_card = await self._connect(resolved_headers, resolved_timeout)

            if self._bg_loop is not None:
                old_client = self._client
                self._client = new_client
                try:
                    await old_client.close()
                except Exception:
                    # Best-effort; the refresh itself already succeeded, a
                    # failure closing the now-discarded old client shouldn't
                    # surface as a refresh() failure.
                    pass
            else:
                await new_client.close()

            self._agent_card, self._headers, self._timeout = new_card, resolved_headers, resolved_timeout

    def close(self) -> None:
        """
        Real teardown for persistent mode; a no-op for non-persistent
        (nothing held between calls). Idempotent -- self._bg_loop is None
        both before construction ever ran in persistent mode and after this
        has already run once, so a second call is a safe no-op.
        """
        if self._bg_loop is None:
            return
        run_coro_sync(self._client.close(), loop=self._bg_loop)
        stop_background_loop(self._bg_loop, self._bg_thread)
        self._bg_loop = None
        self._bg_thread = None
        self._client = None

    def send_parts(self, parts: Sequence[Part]) -> tuple[Part, ...]:
        """
        Dispatches onto the background loop if one is currently held
        (persistent mode, not yet closed); else runs a fresh open/send/close
        cycle -- safe because nothing outlives that one call.
        """
        return run_coro_sync(self._do_send_parts(parts, self._client), loop=self._bg_loop)

    async def async_send_parts(self, parts: Sequence[Part]) -> tuple[Part, ...]:
        """Async counterpart to send_parts."""
        if self._bg_loop is not None:
            return await run_coro_async(self._do_send_parts(parts, self._client), loop=self._bg_loop)
        return await self._do_send_parts(parts, None)

    async def _do_send_parts(self, parts: Sequence[Part], client: Client | None) -> tuple[Part, ...]:
        """
        Core send/receive logic. client=None (always true for non-persistent
        mode, and true for a persistent mode already torn down by close())
        means this connects first and closes after; an explicit live client
        (persistent mode's self._client) means only send, leaving the held
        connection open.

        1. Validate parts. 2. Connect if client is None. 3-4. Build and send
        an outbound Message/SendMessageRequest, consuming exactly one
        StreamResponse (streaming=False guarantees this). 5-7. Unwrap the
        response into a Part tuple, raising A2AProxyError for any
        non-terminal-completed outcome. 8. Close the connection if it was
        opened just for this call.
        """
        if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)) or not parts:
            raise ValueError("parts must be a non-empty sequence of a2a-sdk Part objects.")
        if not all(isinstance(part, Part) for part in parts):
            raise TypeError("every item in parts must be an a2a-sdk Part instance.")

        owns_client = client is None
        if owns_client:
            client, _ = await self._connect(self._headers, self._timeout)

        try:
            # new_message()'s own default role is ROLE_AGENT (a
            # server-reply-oriented default); ROLE_USER must be passed
            # explicitly here since this is the client's outbound message.
            message = new_message(list(parts), role=Role.ROLE_USER)
            request = SendMessageRequest(message=message)

            try:
                stream_response: StreamResponse | None = None
                async for item in client.send_message(request):
                    stream_response = item
                    break  # streaming=False guarantees exactly one item.
            except (A2AClientError, A2AClientTimeoutError, httpx.HTTPError, ValueError) as exc:
                raise A2AProxyError(f"{type(self).__name__}: send_parts failed: {exc}") from exc

            if stream_response is None:
                raise A2AProxyError(f"{type(self).__name__}: send_parts received no response.")

            if stream_response.HasField("message"):
                return tuple(stream_response.message.parts)

            if stream_response.HasField("task"):
                task = stream_response.task
                state = task.status.state
                if state == TaskState.TASK_STATE_COMPLETED:
                    if task.status.HasField("message"):
                        return tuple(task.status.message.parts)
                    return tuple(part for artifact in task.artifacts for part in artifact.parts)
                if state in TERMINAL_FAILURE_STATES:
                    raise A2AProxyError(
                        f"{type(self).__name__}: remote task ended in state {TaskState.Name(state)!r}"
                    )
                raise A2AProxyError(
                    f"{type(self).__name__}: remote task returned unsupported/non-terminal state "
                    f"{TaskState.Name(state)!r} (submitted/working shouldn't occur under blocking "
                    f"config; input-required/auth-required are explicitly out of scope for this track)"
                )

            raise A2AProxyError(f"{type(self).__name__}: response had neither task nor message")
        finally:
            if owns_client:
                await client.close()

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "transport_mode": self.transport_mode,
            "persistent": self.persistent,
            "has_headers": self._headers is not None,
            "header_keys": sorted(self._headers.keys()) if self._headers is not None else [],
            "timeout": self.timeout,
            "agent_name": getattr(self._agent_card, "name", None),
        }