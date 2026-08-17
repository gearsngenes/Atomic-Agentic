from __future__ import annotations

import asyncio
import importlib
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import grpc
import httpx
import pytest
from a2a.helpers import (
    new_data_part,
    new_message,
    new_raw_part,
    new_task_from_user_message,
    new_text_part,
    new_url_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler, GrpcHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.routes.rest_routes import create_rest_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Role,
    a2a_pb2_grpc,
)
from google.protobuf.json_format import MessageToDict
from starlette.applications import Starlette

from atomic_agentic.a2a.A2AClientHub import A2AClientHub

# a2a/__init__.py's own `from .A2AClientHub import A2AClientHub` shadows the
# package's A2AClientHub *submodule* attribute with the *class* -- so a plain
# `import atomic_agentic.a2a.A2AClientHub as X` resolves X to the class, not
# the module, and `X.create_client` (below) doesn't exist. importlib.import_module
# bypasses that shadowing by going straight through sys.modules -- same fix
# already applied in tests/llm/test_base.py for the identical scenario.
a2a_client_hub_module = importlib.import_module("atomic_agentic.a2a.A2AClientHub")
from atomic_agentic.constants.a2a_sdk import TRANSPORT_GRPC, TRANSPORT_JSON_RPC, TRANSPORT_REST
from atomic_agentic.exceptions import A2AProxyError
from atomic_agentic.utils.core import run_coro_sync, start_background_loop, stop_background_loop


# --------------------------------------------------------------------------- #
# Fixture A2A server: a plain a2a-sdk AgentExecutor (not AA-specific) that
# echoes back whatever Parts it receives, exposed over all three transports.
# A DataPart shaped {"trigger": "fail"} makes it fail the task instead, for
# TestFailureModes.
# --------------------------------------------------------------------------- #


class _EchoAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        parts = list(context.message.parts)
        should_fail = any(
            part.WhichOneof("content") == "data"
            and MessageToDict(part.data) == {"trigger": "fail"}
            for part in parts
        )
        should_be_slow = any(
            part.WhichOneof("content") == "data"
            and MessageToDict(part.data) == {"trigger": "slow"}
            for part in parts
        )

        if should_fail:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
            updater = TaskUpdater(event_queue, task.id, task.context_id)
            await updater.failed()
            return

        if should_be_slow:
            # Deterministically exceeds any sane client timeout, unlike
            # racing a real (fast, in-process, loopback) round-trip against
            # a millisecond-scale clock -- see test_timeout_raises_a2a_proxy_error.
            await asyncio.sleep(2)

        reply = new_message(parts, role=Role.ROLE_AGENT)
        await event_queue.enqueue_event(reply)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class FixtureServer:
    http_url: str
    grpc_target: str
    rpc_url: str


@pytest.fixture(scope="module")
def fixture_server():
    """Stands up one Echo AgentExecutor over JSON-RPC + REST (one Starlette
    app, one http port) and gRPC (a separate port), all backed by the same
    request handler. Runs on a dedicated background loop for the whole
    module's test session."""
    http_port = _free_port()
    grpc_port = _free_port()
    http_url = f"http://127.0.0.1:{http_port}"
    rpc_url = "/rpc"
    grpc_target = f"127.0.0.1:{grpc_port}"

    card = AgentCard(
        name="EchoFixtureAgent",
        description="Echoes back whatever Parts it receives.",
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(id="echo", name="echo", description="Echoes parts back.")],
        capabilities=AgentCapabilities(),
        supported_interfaces=[
            AgentInterface(url=f"{http_url}{rpc_url}", protocol_binding=TRANSPORT_JSON_RPC),
            AgentInterface(url=http_url, protocol_binding=TRANSPORT_REST),
            AgentInterface(url=grpc_target, protocol_binding=TRANSPORT_GRPC),
        ],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=_EchoAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    app = Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(request_handler, rpc_url=rpc_url),
            *create_rest_routes(request_handler),
        ]
    )

    import uvicorn

    uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=http_port, log_level="critical")
    uvicorn_server = uvicorn.Server(uvicorn_config)

    loop, thread = start_background_loop()

    # grpc.aio.server() must be constructed inside the same coroutine that
    # gets dispatched onto this loop -- constructing it outside (in this
    # fixture function's own calling context) and only dispatching .start()
    # onto the loop binds the server to whatever loop was "current" at
    # construction time, not this loop, hitting the identical
    # cross-event-loop hazard start_background_loop exists to eliminate
    # (confirmed by direct reproduction during critique).
    grpc_server_box: list[grpc.aio.Server] = []

    async def _startup() -> None:
        grpc_server = grpc.aio.server()
        a2a_pb2_grpc.add_A2AServiceServicer_to_server(GrpcHandler(request_handler), grpc_server)
        grpc_server.add_insecure_port(grpc_target)
        await grpc_server.start()
        grpc_server_box.append(grpc_server)
        asyncio.ensure_future(uvicorn_server.serve())

    run_coro_sync(_startup(), loop=loop)
    grpc_server = grpc_server_box[0]

    # Poll the agent card endpoint until the http server is actually
    # accepting connections (uvicorn.serve() was fired-and-forgot above).
    deadline = time.monotonic() + 10
    with httpx.Client() as probe:
        while True:
            try:
                response = probe.get(f"{http_url}/.well-known/agent-card.json", timeout=1)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("fixture A2A server did not become ready in time.")
            time.sleep(0.1)

    yield FixtureServer(http_url=http_url, grpc_target=grpc_target, rpc_url=rpc_url)

    async def _shutdown() -> None:
        # request_handler.aclose() drains DefaultRequestHandler's
        # ActiveTaskRegistry -- the SDK-sanctioned shutdown hook (its own
        # docstring: "so a server shutdown leaves no pending asyncio.Task",
        # "intended to be wired into an ASGI lifespan/on_shutdown hook").
        # Without it, every request's ActiveTask (producer+consumer tasks)
        # is left orphaned; with ~50+ requests across this fixture's full
        # parametrized test matrix, that accumulation is what hung the whole
        # pytest session at final teardown -- confirmed directly this pass.
        await request_handler.aclose()
        uvicorn_server.should_exit = True
        await asyncio.sleep(0.3)
        await grpc_server.stop(grace=1)

    run_coro_sync(_shutdown(), loop=loop)
    stop_background_loop(loop, thread)


@pytest.fixture(params=[True, False], ids=["persistent", "non_persistent"])
def persistent(request: pytest.FixtureRequest) -> bool:
    return request.param


@pytest.fixture(params=[TRANSPORT_JSON_RPC, TRANSPORT_REST, TRANSPORT_GRPC])
def transport_mode(request: pytest.FixtureRequest) -> str:
    return request.param


def _make_hub(fixture_server: FixtureServer, transport_mode: str, persistent: bool) -> A2AClientHub:
    return A2AClientHub(fixture_server.http_url, transport_mode, persistent)


class TestConstruction:
    def test_construction_succeeds_per_transport_and_mode(
        self, fixture_server: FixtureServer, transport_mode: str, persistent: bool
    ) -> None:
        hub = _make_hub(fixture_server, transport_mode, persistent)
        try:
            assert hub.transport_mode == transport_mode
            assert hub.base_url == fixture_server.http_url
            assert hub.persistent is persistent
            assert hub.agent_card.name == "EchoFixtureAgent"
        finally:
            hub.close()

    def test_invalid_transport_mode_raises(self, fixture_server: FixtureServer) -> None:
        with pytest.raises(ValueError, match="transport_mode"):
            A2AClientHub(fixture_server.http_url, "bogus", True)

    def test_unreachable_base_url_raises_a2a_proxy_error(self) -> None:
        with pytest.raises(A2AProxyError):
            A2AClientHub("http://127.0.0.1:1", TRANSPORT_JSON_RPC, True)

        with pytest.raises(A2AProxyError):
            A2AClientHub("http://127.0.0.1:1", TRANSPORT_JSON_RPC, False)

    def test_async_create_parity(self, fixture_server: FixtureServer) -> None:
        async def _build() -> A2AClientHub:
            return await A2AClientHub.async_create(
                fixture_server.http_url, TRANSPORT_JSON_RPC, True
            )

        hub = asyncio.run(_build())
        try:
            assert hub.transport_mode == TRANSPORT_JSON_RPC
        finally:
            hub.close()


class TestSendParts:
    def test_round_trips_each_part_kind(
        self, fixture_server: FixtureServer, transport_mode: str, persistent: bool
    ) -> None:
        hub = _make_hub(fixture_server, transport_mode, persistent)
        try:
            for part in (
                new_text_part("hello"),
                new_data_part({"a": 1}),
                new_raw_part(b"raw-bytes", media_type="application/octet-stream"),
                new_url_part("https://example.com/file"),
            ):
                result = hub.send_parts([part])
                assert len(result) == 1
                assert result[0].WhichOneof("content") == part.WhichOneof("content")
        finally:
            hub.close()

    def test_multi_part_round_trip(self, fixture_server: FixtureServer, persistent: bool) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            parts = [new_text_part("one"), new_text_part("two")]
            result = hub.send_parts(parts)
            assert len(result) == 2
        finally:
            hub.close()

    def test_async_send_parts(self, fixture_server: FixtureServer, persistent: bool) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)

        async def _call() -> tuple:
            return await hub.async_send_parts([new_text_part("async hello")])

        try:
            result = asyncio.run(_call())
            assert len(result) == 1
        finally:
            hub.close()

    def test_empty_parts_raises(self, fixture_server: FixtureServer, persistent: bool) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            with pytest.raises(ValueError):
                hub.send_parts([])
        finally:
            hub.close()


class TestNonPersistentRebuild:
    def test_send_parts_reconnects_every_call(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            original_create_client = a2a_client_hub_module.create_client
            with patch.object(
                a2a_client_hub_module, "create_client", wraps=original_create_client
            ) as spy:
                hub.send_parts([new_text_part("one")])
                hub.send_parts([new_text_part("two")])
                assert spy.call_count == 2
        finally:
            hub.close()

    def test_client_attribute_is_never_held_between_calls(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            assert hub._client is None
            hub.send_parts([new_text_part("x")])
            assert hub._client is None
        finally:
            hub.close()


class TestConcurrency:
    def test_concurrent_threads_against_one_persistent_hub(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, True)
        try:
            results: list[Any] = [None] * 5

            def worker(index: int) -> None:
                results[index] = hub.send_parts([new_text_part(f"call-{index}")])

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert all(r is not None and len(r) == 1 for r in results)
        finally:
            hub.close()

    def test_concurrent_async_callers_on_separate_loops(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, True)
        try:
            outcomes: list[Any] = [None] * 5

            async def call(index: int) -> None:
                outcomes[index] = await hub.async_send_parts([new_text_part(f"acall-{index}")])

            def run_in_own_loop(index: int) -> None:
                asyncio.run(call(index))

            threads = [threading.Thread(target=run_in_own_loop, args=(i,)) for i in range(5)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            assert all(o is not None and len(o) == 1 for o in outcomes)
        finally:
            hub.close()


class TestClose:
    def test_persistent_close_stops_background_thread(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, True)
        thread = hub._bg_thread

        hub.close()

        assert not thread.is_alive()

    def test_persistent_close_is_idempotent(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, True)
        hub.close()
        hub.close()  # does not raise

    def test_send_parts_after_close_falls_back_to_fresh_connect(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, True)
        hub.close()

        # Documented use-after-close behavior: a closed persistent hub
        # degrades to non-persistent per-call reconnects rather than
        # raising -- self._bg_loop is None is the single source of truth
        # every dispatch site checks.
        result = hub.send_parts([new_text_part("after-close")])
        assert len(result) == 1

    def test_non_persistent_close_is_a_harmless_no_op(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        hub.close()
        # still usable -- non-persistent mode holds no live connection to close.
        hub.send_parts([new_text_part("still works")])
        hub.close()


class TestRefresh:
    def test_refresh_updates_headers_and_timeout(
        self, fixture_server: FixtureServer, persistent: bool
    ) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            hub.refresh(headers={"X-Test": "1"}, timeout=123.0)
            assert hub.headers == {"X-Test": "1"}
            assert hub.timeout == 123.0
            assert hub.agent_card.name == "EchoFixtureAgent"
        finally:
            hub.close()

    def test_refresh_rolls_back_on_failure(self, fixture_server: FixtureServer, persistent: bool) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            original_headers = hub.headers
            original_timeout = hub.timeout
            original_base_url = hub._base_url

            # Point the hub at an unreachable base_url so the refresh's own
            # _connect() call fails before anything is committed, then
            # restore it -- base_url itself is not meant to be mutated
            # outside this reproduction of a real transport failure.
            hub._base_url = "http://127.0.0.1:1"
            try:
                with pytest.raises(A2AProxyError):
                    hub.refresh(headers={"X-Broken": "1"})
            finally:
                hub._base_url = original_base_url

            assert hub.headers == original_headers
            assert hub.timeout == original_timeout
        finally:
            hub.close()

    def test_refresh_with_no_arguments_raises(self, fixture_server: FixtureServer, persistent: bool) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            with pytest.raises(ValueError):
                hub.refresh()
        finally:
            hub.close()

    def test_persistent_concurrent_refresh_does_not_corrupt_state(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, True)
        try:
            def worker(index: int) -> None:
                hub.refresh(timeout=100.0 + index)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # No corruption: client/card pairing is still usable afterward.
            hub.send_parts([new_text_part("after-concurrent-refresh")])
        finally:
            hub.close()

    def test_refresh_after_close_falls_back_to_fresh_connect(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, True)
        hub.close()

        hub.refresh(timeout=222.0)
        assert hub.timeout == 222.0


class TestToDict:
    def test_shape_and_no_raw_header_values(self, fixture_server: FixtureServer, persistent: bool) -> None:
        hub = A2AClientHub(
            fixture_server.http_url, TRANSPORT_JSON_RPC, persistent, headers={"Authorization": "secret"}
        )
        try:
            data = hub.to_dict()
            assert data["transport_mode"] == TRANSPORT_JSON_RPC
            assert data["base_url"] == fixture_server.http_url
            assert data["persistent"] is persistent
            assert data["has_headers"] is True
            assert data["header_keys"] == ["Authorization"]
            assert "secret" not in str(data)
            assert data["agent_name"] == "EchoFixtureAgent"
        finally:
            hub.close()


class TestAtomicSkillDetection:
    """The fixture server here is a plain, non-AA-aware AgentExecutor that
    never publishes PARAM_SCHEMA_EXT_URI -- exactly the "extension absent"
    case get_atomic_skills() is scoped to degrade gracefully on. Detection
    against a real A2AtomicExecutor-backed server (extension present, real
    skills) is covered end-to-end in test_atomic_executor.py, which owns its
    own atomic-aware fixture server."""

    def test_get_atomic_skills_is_empty_when_extension_absent(
        self, fixture_server: FixtureServer, persistent: bool
    ) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            assert hub.get_atomic_skills() == {}
        finally:
            hub.close()

    def test_refresh_rebuilds_atomic_skills_without_error(
        self, fixture_server: FixtureServer, persistent: bool
    ) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            hub.refresh(timeout=123.0)
            assert hub.get_atomic_skills() == {}
        finally:
            hub.close()

    def test_get_atomic_skills_returns_a_copy(
        self, fixture_server: FixtureServer, persistent: bool
    ) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            skills = hub.get_atomic_skills()
            skills["injected"] = None  # type: ignore[assignment]
            assert hub.get_atomic_skills() == {}
        finally:
            hub.close()


class TestFailureModes:
    def test_server_side_task_failure_raises_a2a_proxy_error(
        self, fixture_server: FixtureServer, persistent: bool
    ) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            with pytest.raises(A2AProxyError):
                hub.send_parts([new_data_part({"trigger": "fail"})])
        finally:
            hub.close()

    def test_timeout_raises_a2a_proxy_error(self, fixture_server: FixtureServer) -> None:
        # Deterministic, not a clock race: the server sleeps 2s on this
        # trigger (see _EchoAgentExecutor), so a 0.5s client timeout is
        # guaranteed to be exceeded regardless of machine speed/load --
        # unlike the previous hub._timeout = 0.001 approach, which raced a
        # real (fast, in-process, loopback) round-trip against a
        # millisecond-scale clock and was genuinely flaky (confirmed: failed
        # ~1/3 of runs).
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            hub._timeout = 0.5
            with pytest.raises(A2AProxyError):
                hub.send_parts([new_data_part({"trigger": "slow"})])
        finally:
            hub._timeout = 600
            hub.close()
