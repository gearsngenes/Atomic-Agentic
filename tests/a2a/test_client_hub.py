from __future__ import annotations

import asyncio
import importlib
import inspect
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

from atomic_agentic.a2a.a2a_client_hub import A2AClientHub

a2a_client_hub_module = importlib.import_module("atomic_agentic.a2a.a2a_client_hub")
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
    return A2AClientHub(fixture_server.http_url, transport_mode, persistent=persistent)


class TestConstruction:
    def test_construction_succeeds_per_transport_and_mode(
        self, fixture_server: FixtureServer, transport_mode: str, persistent: bool
    ) -> None:
        hub = _make_hub(fixture_server, transport_mode, persistent)
        try:
            assert hub.transport_mode == transport_mode
            assert hub.location == fixture_server.http_url
            assert hub.persistent is persistent
            assert hub.agent_card.name == "EchoFixtureAgent"
        finally:
            hub.close()

    def test_invalid_transport_mode_raises(self, fixture_server: FixtureServer) -> None:
        with pytest.raises(ValueError, match="transport_mode"):
            A2AClientHub(fixture_server.http_url, "bogus", persistent=True)

    def test_unreachable_location_raises_a2a_proxy_error(self) -> None:
        with pytest.raises(A2AProxyError):
            A2AClientHub("http://127.0.0.1:1", TRANSPORT_JSON_RPC, persistent=True)

        with pytest.raises(A2AProxyError):
            A2AClientHub("http://127.0.0.1:1", TRANSPORT_JSON_RPC)

    def test_async_create_parity(self, fixture_server: FixtureServer) -> None:
        async def _build() -> A2AClientHub:
            return await A2AClientHub.async_create(
                fixture_server.http_url, TRANSPORT_JSON_RPC, persistent=True
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
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC)
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
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC)
        try:
            assert hub._client is None
            hub.send_parts([new_text_part("x")])
            assert hub._client is None
        finally:
            hub.close()


class TestConcurrency:
    def test_concurrent_threads_against_one_persistent_hub(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, persistent=True)
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
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, persistent=True)
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
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, persistent=True)
        thread = hub._bg_thread

        hub.close()

        assert not thread.is_alive()

    def test_persistent_close_is_idempotent(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, persistent=True)
        hub.close()
        hub.close()  # does not raise

    def test_send_parts_after_close_falls_back_to_fresh_connect(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, persistent=True)
        hub.close()

        # Documented use-after-close behavior: a closed persistent hub
        # degrades to non-persistent per-call reconnects rather than
        # raising -- self._bg_loop is None is the single source of truth
        # every dispatch site checks.
        result = hub.send_parts([new_text_part("after-close")])
        assert len(result) == 1

    def test_non_persistent_close_is_a_harmless_no_op(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC)
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
            hub.refresh(headers={"X-Test": "1"}, timeout_seconds=123.0)
            assert hub.headers == {"X-Test": "1"}
            assert hub.timeout_seconds == 123.0
            assert hub.agent_card.name == "EchoFixtureAgent"
        finally:
            hub.close()

    def test_refresh_rolls_back_on_failure(self, fixture_server: FixtureServer, persistent: bool) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            original_headers = hub.headers
            original_timeout = hub.timeout_seconds
            original_location = hub._location

            # Point the hub at an unreachable location so the refresh's own
            # _connect() call fails before anything is committed, then
            # restore it -- location itself is not meant to be mutated
            # outside this reproduction of a real transport failure.
            hub._location = "http://127.0.0.1:1"
            try:
                with pytest.raises(A2AProxyError):
                    hub.refresh(headers={"X-Broken": "1"})
            finally:
                hub._location = original_location

            assert hub.headers == original_headers
            assert hub.timeout_seconds == original_timeout
        finally:
            hub.close()

    def test_bare_refresh_re_resolves_card_and_rebuilds_skills(
        self, fixture_server: FixtureServer, persistent: bool
    ) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            connects: list[int] = []
            builds: list[int] = []
            real_connect = hub._connect
            real_build = A2AClientHub._build_atomic_skills

            async def counting_connect(*args: Any, **kwargs: Any) -> Any:
                connects.append(1)
                return await real_connect(*args, **kwargs)

            def counting_build(*args: Any, **kwargs: Any) -> Any:
                builds.append(1)
                return real_build(*args, **kwargs)

            hub._connect = counting_connect  # type: ignore[method-assign]
            hub._build_atomic_skills = counting_build  # type: ignore[method-assign]

            hub.refresh()

            assert connects == [1]
            assert builds == [1]
            assert hub.list_skills() == {}
        finally:
            hub.close()

    def test_persistent_concurrent_refresh_does_not_corrupt_state(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, persistent=True)
        try:
            def worker(index: int) -> None:
                hub.refresh(timeout_seconds=100.0 + index)

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
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, persistent=True)
        hub.close()

        hub.refresh(timeout_seconds=222.0)
        assert hub.timeout_seconds == 222.0


class TestToDict:
    def test_shape_and_no_raw_header_values(self, fixture_server: FixtureServer, persistent: bool) -> None:
        hub = A2AClientHub(
            fixture_server.http_url,
            TRANSPORT_JSON_RPC,
            persistent=persistent,
            headers={"Authorization": "secret"},
        )
        try:
            data = hub.to_dict()
            assert data["transport_mode"] == TRANSPORT_JSON_RPC
            assert data["location"] == fixture_server.http_url
            assert "base_url" not in data
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
    case list_skills() is scoped to degrade gracefully on. Detection
    against a real A2AtomicExecutor-backed server (extension present, real
    skills) is covered end-to-end in test_atomic_executor.py, which owns its
    own atomic-aware fixture server."""

    def test_list_skills_is_empty_when_extension_absent(
        self, fixture_server: FixtureServer, persistent: bool
    ) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            assert hub.list_skills() == {}
        finally:
            hub.close()

    def test_refresh_rebuilds_atomic_skills_without_error(
        self, fixture_server: FixtureServer, persistent: bool
    ) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            hub.refresh(timeout_seconds=123.0)
            assert hub.list_skills() == {}
        finally:
            hub.close()

    def test_get_skill_hit_and_miss(
        self, fixture_server: FixtureServer, persistent: bool
    ) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            sentinel = object()
            hub._atomic_skills = {"add": sentinel}  # type: ignore[dict-item]

            assert hub.get_skill("add") is sentinel
            with pytest.raises(A2AProxyError, match="'nope'") as excinfo:
                hub.get_skill("nope")
            assert "['add']" in str(excinfo.value)
        finally:
            hub.close()

    def test_list_skills_returns_a_copy(
        self, fixture_server: FixtureServer, persistent: bool
    ) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            skills = hub.list_skills()
            skills["injected"] = None  # type: ignore[assignment]
            assert hub.list_skills() == {}
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
        # unlike the previous hub._timeout_seconds = 0.001 approach, which raced a
        # real (fast, in-process, loopback) round-trip against a
        # millisecond-scale clock and was genuinely flaky (confirmed: failed
        # ~1/3 of runs).
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC)
        try:
            hub._timeout_seconds = 0.5
            with pytest.raises(A2AProxyError):
                hub.send_parts([new_data_part({"trigger": "slow"})])
        finally:
            hub._timeout_seconds = 600
            hub.close()


class TestConstructionContract:
    def test_two_required_arguments_and_defaults(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC)
        try:
            assert hub.persistent is False
            assert hub.name is None
            assert hub.timeout_seconds == 600
            assert hub.client_kwargs is None
            assert hub.headers is None
        finally:
            hub.close()

    def test_options_are_keyword_only(self, fixture_server: FixtureServer) -> None:
        with pytest.raises(TypeError):
            A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, True)  # type: ignore[misc]

    @pytest.mark.parametrize("bad", [None, 5, ["x"]])
    def test_non_str_location_raises_value_error(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="location must be a string"):
            A2AClientHub(bad, TRANSPORT_JSON_RPC)

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_location_raises_value_error(self, blank: str) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            A2AClientHub(blank, TRANSPORT_JSON_RPC)

    def test_non_bool_persistent_raises_type_error(self, fixture_server: FixtureServer) -> None:
        with pytest.raises(TypeError, match="persistent"):
            A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, persistent="yes")  # type: ignore[arg-type]

    def test_location_is_stripped(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(f"  {fixture_server.http_url}  ", TRANSPORT_JSON_RPC)
        try:
            assert hub.location == fixture_server.http_url
        finally:
            hub.close()

    def test_removed_views_are_gone_and_skills_verb_is_list_skills(
        self, fixture_server: FixtureServer
    ) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC)
        try:
            assert hub.location == fixture_server.http_url
            assert not hasattr(hub, "base_url")
            assert not hasattr(hub, "get_atomic_skills")
            assert isinstance(hub.list_skills(), dict)
        finally:
            hub.close()

    def test_base_url_is_no_longer_a_constructor_argument(self, fixture_server: FixtureServer) -> None:
        with pytest.raises(TypeError):
            A2AClientHub(base_url=fixture_server.http_url, transport_mode=TRANSPORT_JSON_RPC)  # type: ignore[call-arg]


class TestName:
    def test_default_none(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC)
        try:
            assert hub.name is None
        finally:
            hub.close()

    def test_valid_identifier_stored_and_in_to_dict(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, name="echo_agent")
        try:
            assert hub.name == "echo_agent"
            assert hub.to_dict()["name"] == "echo_agent"
        finally:
            hub.close()

    @pytest.mark.parametrize("bad", [5, ["x"], b"x"])
    def test_non_str_raises_type_error(self, bad: Any) -> None:
        with pytest.raises(TypeError, match="name"):
            A2AClientHub("http://127.0.0.1:1", TRANSPORT_JSON_RPC, name=bad)

    @pytest.mark.parametrize("bad", ["", " ", " a", "a ", "a\n", "a-b", "1a"])
    def test_non_identifier_raises_value_error(self, bad: str) -> None:
        with pytest.raises(ValueError, match="name"):
            A2AClientHub("http://127.0.0.1:1", TRANSPORT_JSON_RPC, name=bad)

    def test_name_has_no_setter(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, name="echo")
        try:
            with pytest.raises(AttributeError):
                hub.name = "other"  # type: ignore[misc]
        finally:
            hub.close()


class TestTimeoutSeconds:
    def test_constructor_property_and_refresh(self, fixture_server: FixtureServer, persistent: bool) -> None:
        hub = A2AClientHub(
            fixture_server.http_url, TRANSPORT_JSON_RPC, persistent=persistent, timeout_seconds=7
        )
        try:
            assert hub.timeout_seconds == 7.0
            hub.refresh(timeout_seconds=9)
            assert hub.timeout_seconds == 9.0
        finally:
            hub.close()

    def test_old_timeout_names_are_gone(self, fixture_server: FixtureServer) -> None:
        with pytest.raises(TypeError):
            A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, timeout=5)  # type: ignore[call-arg]

        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC)
        try:
            assert not hasattr(hub, "timeout")
            with pytest.raises(TypeError):
                hub.refresh(timeout=5)  # type: ignore[call-arg]
        finally:
            hub.close()


class TestClientKwargs:
    def test_forwarded_to_httpx_client(self, fixture_server: FixtureServer, persistent: bool) -> None:
        with patch.object(
            a2a_client_hub_module.httpx, "AsyncClient", wraps=httpx.AsyncClient
        ) as spy:
            hub = A2AClientHub(
                fixture_server.http_url,
                TRANSPORT_JSON_RPC,
                persistent=persistent,
                client_kwargs={"follow_redirects": True},
            )
            try:
                assert spy.call_args.kwargs["follow_redirects"] is True
                assert spy.call_args.kwargs["timeout"] == 600.0
            finally:
                hub.close()

    def test_forwarded_on_non_persistent_send(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(
            fixture_server.http_url, TRANSPORT_JSON_RPC, client_kwargs={"follow_redirects": True}
        )
        try:
            with patch.object(
                a2a_client_hub_module.httpx, "AsyncClient", wraps=httpx.AsyncClient
            ) as spy:
                hub.send_parts([new_text_part("hi")])
                assert spy.call_args.kwargs["follow_redirects"] is True
        finally:
            hub.close()

    @pytest.mark.parametrize("key", ["headers", "timeout"])
    def test_headers_or_timeout_key_raises(self, key: str) -> None:
        with pytest.raises(ValueError, match=key):
            A2AClientHub("http://127.0.0.1:1", TRANSPORT_JSON_RPC, client_kwargs={key: 1})

    def test_non_mapping_raises(self) -> None:
        with pytest.raises(ValueError, match="client_kwargs must be a mapping"):
            A2AClientHub("http://127.0.0.1:1", TRANSPORT_JSON_RPC, client_kwargs=["x"])  # type: ignore[arg-type]

    def test_property_is_read_only_view(self, fixture_server: FixtureServer) -> None:
        source = {"follow_redirects": True}
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, client_kwargs=source)
        try:
            assert dict(hub.client_kwargs) == {"follow_redirects": True}
            with pytest.raises(TypeError):
                hub.client_kwargs["verify"] = False  # type: ignore[index]
            source["verify"] = False
            assert dict(hub.client_kwargs) == {"follow_redirects": True}
        finally:
            hub.close()

    def test_refresh_replaces_client_kwargs(self, fixture_server: FixtureServer, persistent: bool) -> None:
        hub = A2AClientHub(
            fixture_server.http_url,
            TRANSPORT_JSON_RPC,
            persistent=persistent,
            client_kwargs={"follow_redirects": True},
        )
        try:
            hub.refresh(client_kwargs={"trust_env": False})
            assert dict(hub.client_kwargs) == {"trust_env": False}
            assert len(hub.send_parts([new_text_part("after-refresh")])) == 1
        finally:
            hub.close()

    def test_refresh_with_bad_client_kwargs_rolls_back(
        self, fixture_server: FixtureServer, persistent: bool
    ) -> None:
        hub = A2AClientHub(
            fixture_server.http_url,
            TRANSPORT_JSON_RPC,
            persistent=persistent,
            headers={"X-Keep": "1"},
            client_kwargs={"follow_redirects": True},
        )
        try:
            original_client = hub._client
            original_card = hub.agent_card
            original_skills = hub.list_skills()

            with pytest.raises(ValueError, match="client_kwargs may not set"):
                hub.refresh(client_kwargs={"headers": {"X": "1"}})
            with pytest.raises(TypeError):
                hub.refresh(client_kwargs={"not_an_httpx_option": 1})

            assert dict(hub.client_kwargs) == {"follow_redirects": True}
            assert hub.headers == {"X-Keep": "1"}
            assert hub.timeout_seconds == 600.0
            assert hub._client is original_client
            assert hub.agent_card == original_card
            assert hub.list_skills() == original_skills
            assert len(hub.send_parts([new_text_part("still-works")])) == 1
        finally:
            hub.close()


class TestRefreshContract:
    def test_bare_refresh_is_valid(self, fixture_server: FixtureServer, persistent: bool) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            hub.refresh()
            assert len(hub.send_parts([new_text_part("after-bare-refresh")])) == 1
        finally:
            hub.close()


class TestAsyncCreateOptions:
    def test_async_create_forwards_options(self, fixture_server: FixtureServer, persistent: bool) -> None:
        async def _build() -> A2AClientHub:
            return await A2AClientHub.async_create(
                fixture_server.http_url,
                TRANSPORT_JSON_RPC,
                name="echo",
                persistent=persistent,
                headers={"X-Test": "1"},
                timeout_seconds=11,
                client_kwargs={"follow_redirects": True},
            )

        hub = asyncio.run(_build())
        try:
            assert hub.name == "echo"
            assert hub.persistent is persistent
            assert hub.headers == {"X-Test": "1"}
            assert hub.timeout_seconds == 11.0
            assert dict(hub.client_kwargs) == {"follow_redirects": True}
        finally:
            hub.close()


class TestToDictExtended:
    def test_adds_name_location_client_kwargs_info_and_renames_timeout(
        self, fixture_server: FixtureServer
    ) -> None:
        hub = A2AClientHub(
            fixture_server.http_url,
            TRANSPORT_JSON_RPC,
            name="echo",
            timeout_seconds=7,
            client_kwargs={"verify": False, "follow_redirects": True},
        )
        try:
            data = hub.to_dict()
            assert data["name"] == "echo"
            assert data["location"] == fixture_server.http_url
            assert "base_url" not in data
            assert data["timeout_seconds"] == 7.0
            assert "timeout" not in data
            assert data["has_client_kwargs"] is True
            assert data["client_kwargs_keys"] == ["follow_redirects", "verify"]
        finally:
            hub.close()

    def test_no_client_kwargs_reports_empty(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC)
        try:
            data = hub.to_dict()
            assert data["name"] is None
            assert data["has_client_kwargs"] is False
            assert data["client_kwargs_keys"] == []
        finally:
            hub.close()


_SECONDARY_CONSTRUCTORS = [
    ("jsonrpc", TRANSPORT_JSON_RPC),
    ("rest", TRANSPORT_REST),
    ("grpc", TRANSPORT_GRPC),
]


class TestSecondaryConstructors:
    @pytest.mark.parametrize(("method", "transport"), _SECONDARY_CONSTRUCTORS)
    def test_equivalent_to_general_form(
        self, fixture_server: FixtureServer, method: str, transport: str, persistent: bool
    ) -> None:
        options: dict[str, Any] = {
            "name": "remote",
            "persistent": persistent,
            "headers": {"X-Test": "1"},
            "timeout_seconds": 7,
            "client_kwargs": {"follow_redirects": True},
        }

        built = getattr(A2AClientHub, method)(fixture_server.http_url, **options)
        general = A2AClientHub(fixture_server.http_url, transport, **options)
        try:
            assert built.location == general.location == fixture_server.http_url
            assert built.transport_mode == general.transport_mode == transport
            assert built.name == general.name == "remote"
            assert built.persistent is general.persistent is persistent
            assert built.headers == general.headers == {"X-Test": "1"}
            assert built.timeout_seconds == general.timeout_seconds == 7.0
            assert built.client_kwargs == general.client_kwargs
            assert built.include_names == general.include_names
            assert built.exclude_names == general.exclude_names
            assert built.to_dict() == general.to_dict()
        finally:
            built.close()
            general.close()

    def test_classmethods_return_the_subclass(self, fixture_server: FixtureServer) -> None:
        class SubHub(A2AClientHub):
            pass

        for method in ("jsonrpc", "rest", "grpc"):
            hub = getattr(SubHub, method)(fixture_server.http_url)
            try:
                assert type(hub) is SubHub
            finally:
                hub.close()

    @pytest.mark.parametrize("method", ["jsonrpc", "rest", "grpc"])
    def test_transport_mode_is_not_an_option(
        self, fixture_server: FixtureServer, method: str
    ) -> None:
        with pytest.raises(TypeError):
            getattr(A2AClientHub, method)(
                fixture_server.http_url, transport_mode=TRANSPORT_JSON_RPC
            )

    @pytest.mark.parametrize("method", ["jsonrpc", "rest", "grpc"])
    def test_signature_parity_with_init(self, method: str) -> None:
        init_params = dict(inspect.signature(A2AClientHub.__init__).parameters)
        del init_params["self"]
        params = dict(inspect.signature(getattr(A2AClientHub, method)).parameters)

        assert set(params) == (set(init_params) - {"location", "transport_mode"}) | {"url"}
        for pname in set(params) - {"url"}:
            assert params[pname].default == init_params[pname].default
            assert params[pname].kind == init_params[pname].kind
            assert params[pname].kind is inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.parametrize(("method", "transport"), _SECONDARY_CONSTRUCTORS)
    def test_default_persistent_false_and_name_none(
        self, fixture_server: FixtureServer, method: str, transport: str
    ) -> None:
        hub = getattr(A2AClientHub, method)(fixture_server.http_url)
        try:
            assert hub.persistent is False
            assert hub.name is None
            assert hub.transport_mode == transport
        finally:
            hub.close()


# --------------------------------------------------------------------------- #
# Stage 9 Pass 6: close-always-clears, atexit, `with`, leak fixes, remote_name
# --------------------------------------------------------------------------- #


class _RecordingClient:
    """Stand-in for an a2a Client that records close() and can raise from it."""

    def __init__(self, real: Any = None, error: Exception | None = None) -> None:
        self.real = real
        self.error = error
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1
        if self.real is not None:
            await self.real.close()
        if self.error is not None:
            raise self.error


class TestCloseAlwaysClears:
    def test_close_whose_client_close_raises_still_resets_state(
        self, fixture_server: FixtureServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stops: list[Any] = []
        real_stop = a2a_client_hub_module.stop_background_loop

        def recording_stop(loop: Any, thread: Any) -> None:
            stops.append((loop, thread))
            real_stop(loop, thread)

        monkeypatch.setattr(a2a_client_hub_module, "stop_background_loop", recording_stop)

        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, persistent=True)
        loop, thread = hub._bg_loop, hub._bg_thread
        hub._client = _RecordingClient(real=hub._client, error=RuntimeError("close boom"))

        with pytest.raises(RuntimeError, match="close boom"):
            hub.close()

        assert stops == [(loop, thread)]
        assert hub._bg_loop is None
        assert hub._bg_thread is None
        assert hub._client is None

        hub.close()  # no-op
        assert stops == [(loop, thread)]


class TestAtexit:
    def test_persistent_registers_and_close_unregisters(
        self, fixture_server: FixtureServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registered: list[Any] = []
        unregistered: list[Any] = []
        monkeypatch.setattr(a2a_client_hub_module.atexit, "register", registered.append)
        monkeypatch.setattr(a2a_client_hub_module.atexit, "unregister", unregistered.append)

        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, persistent=True)
        assert registered == [hub.close]

        hub.close()
        assert unregistered == [hub.close]

    def test_non_persistent_registers_nothing(
        self, fixture_server: FixtureServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registered: list[Any] = []
        monkeypatch.setattr(a2a_client_hub_module.atexit, "register", registered.append)

        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC)
        hub.close()

        assert registered == []

    def test_failed_persistent_construction_leaves_no_loop_and_registers_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registered: list[Any] = []
        monkeypatch.setattr(a2a_client_hub_module.atexit, "register", registered.append)

        hub = A2AClientHub.__new__(A2AClientHub)
        with pytest.raises(A2AProxyError):
            hub.__init__("http://127.0.0.1:1", TRANSPORT_JSON_RPC, persistent=True)

        assert hub._bg_loop is None
        assert hub._bg_thread is None
        assert registered == []


class TestContextManager:
    def test_with_closes_on_exit(self, fixture_server: FixtureServer) -> None:
        with A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, persistent=True) as hub:
            assert hub._bg_loop is not None
            thread = hub._bg_thread

        assert hub._bg_loop is None
        assert not thread.is_alive()

    def test_with_closes_on_exception(self, fixture_server: FixtureServer) -> None:
        with pytest.raises(RuntimeError, match="body boom"):
            with A2AClientHub(
                fixture_server.http_url, TRANSPORT_JSON_RPC, persistent=True
            ) as hub:
                raise RuntimeError("body boom")

        assert hub._bg_loop is None

    def test_enter_returns_self_for_non_persistent(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC)
        with hub as entered:
            assert entered is hub


class TestLeakFixes:
    def test_connect_failure_closes_httpx_client(
        self, fixture_server: FixtureServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC)
        created: list[Any] = []
        base = httpx.AsyncClient

        class RecordingAsyncClient(base):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.aclose_calls = 0
                created.append(self)

            async def aclose(self) -> None:
                self.aclose_calls += 1
                await super().aclose()

        async def failing_create_client(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("create boom")

        monkeypatch.setattr(a2a_client_hub_module.httpx, "AsyncClient", RecordingAsyncClient)
        monkeypatch.setattr(a2a_client_hub_module, "create_client", failing_create_client)

        with pytest.raises(A2AProxyError, match="create boom"):
            run_coro_sync(hub._connect(None, 5.0, None))
        assert [c.aclose_calls for c in created] == [1]

        async def unexpected_create_client(*args: Any, **kwargs: Any) -> Any:
            raise KeyError("unexpected")

        monkeypatch.setattr(a2a_client_hub_module, "create_client", unexpected_create_client)

        with pytest.raises(KeyError):
            run_coro_sync(hub._connect(None, 5.0, None))
        assert [c.aclose_calls for c in created] == [1, 1]

    def test_refresh_closes_new_client_when_skill_build_raises(
        self, fixture_server: FixtureServer, persistent: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hub = _make_hub(fixture_server, TRANSPORT_JSON_RPC, persistent)
        try:
            old_client = hub._client
            old_card = hub._agent_card
            old_headers = hub.headers
            old_skills = hub._atomic_skills
            new_client = _RecordingClient()

            async def fake_connect(*args: Any, **kwargs: Any) -> Any:
                return new_client, old_card

            def failing_build(*args: Any, **kwargs: Any) -> Any:
                raise ValueError("skills boom")

            monkeypatch.setattr(hub, "_connect", fake_connect)
            monkeypatch.setattr(hub, "_build_atomic_skills", failing_build)

            with pytest.raises(ValueError, match="skills boom"):
                hub.refresh(headers={"X-New": "1"})

            assert new_client.closed == 1
            assert hub._client is old_client
            assert hub._agent_card is old_card
            assert hub.headers == old_headers
            assert hub._atomic_skills is old_skills
        finally:
            monkeypatch.undo()
            hub.close()


class TestCallAtomicSkillRemoteName:
    def test_keyword_remote_name_routes_and_blank_raises(
        self, fixture_server: FixtureServer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC)
        sent: list[Any] = []

        def fake_send_parts(parts: Any, metadata: Any = None) -> Any:
            sent.append(metadata)
            return (new_data_part({"__atomic_result__": 5}),)

        monkeypatch.setattr(hub, "send_parts", fake_send_parts)

        assert hub.call_atomic_skill(remote_name="add", inputs={"a": 1}) == 5
        assert sent == [{"skill_id": "add"}]

        with pytest.raises(ValueError, match="remote_name must be a non-empty string."):
            hub.call_atomic_skill(remote_name="  ", inputs={})

        async def fake_async_send_parts(parts: Any, metadata: Any = None) -> Any:
            return (new_data_part({"__atomic_result__": 6}),)

        monkeypatch.setattr(hub, "async_send_parts", fake_async_send_parts)
        assert asyncio.run(hub.async_call_atomic_skill(remote_name="add", inputs={})) == 6
