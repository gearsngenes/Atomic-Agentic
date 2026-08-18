from __future__ import annotations

import asyncio
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping

import httpx
import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueueLegacy
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import TaskState, TaskStatusUpdateEvent
from google.protobuf.json_format import MessageToDict
from starlette.applications import Starlette

from atomic_agentic.a2a.A2AClientHub import A2AClientHub
from atomic_agentic.a2a.A2AtomicExecutor import A2AtomicExecutor
from atomic_agentic.constants.a2a_sdk import PARAM_SCHEMA_EXT_URI, TRANSPORT_JSON_RPC
from atomic_agentic.core.Invokable import AtomicInvokable
from atomic_agentic.exceptions import A2AProxyError
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results.atomic import AtomicResult
from atomic_agentic.utils.core import run_coro_sync, start_background_loop, stop_background_loop


# --------------------------------------------------------------------------- #
# Fixture AtomicInvokables
# --------------------------------------------------------------------------- #


def make_param(name: str, index: int) -> ParamSpec:
    return ParamSpec(name=name, index=index, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="Any")


class AddInvokable(AtomicInvokable):
    def invoke(self, inputs: Mapping[str, Any]) -> AtomicResult:
        started_at = datetime.now(timezone.utc)
        result = inputs["a"] + inputs["b"]
        ended_at = datetime.now(timezone.utc)
        return self.make_result(result, started_at, ended_at)


class FailingInvokable(AtomicInvokable):
    def invoke(self, inputs: Mapping[str, Any]) -> AtomicResult:
        raise RuntimeError("deliberate failure")


def make_add_invokable(name: str = "add") -> AddInvokable:
    return AddInvokable(
        name=name,
        namespace="tests",
        description="Adds two numbers.",
        parameters=[make_param("a", 0), make_param("b", 1)],
        return_type="int",
    )


def make_failing_invokable(name: str = "boom") -> FailingInvokable:
    return FailingInvokable(
        name=name, namespace="tests", description="Always raises.", parameters=[], return_type="Any"
    )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_list_form_keys_by_name(self) -> None:
        executor = A2AtomicExecutor([make_add_invokable("add"), make_add_invokable("add2")])
        assert set(executor._invokables) == {"add", "add2"}

    def test_mapping_form_uses_explicit_aliases(self) -> None:
        executor = A2AtomicExecutor({"alias_name": make_add_invokable("add")})
        assert set(executor._invokables) == {"alias_name"}

    def test_list_form_duplicate_name_raises(self) -> None:
        with pytest.raises(ValueError):
            A2AtomicExecutor([make_add_invokable("dup"), make_add_invokable("dup")])

    def test_mapping_form_bad_identifier_key_raises(self) -> None:
        with pytest.raises(ValueError):
            A2AtomicExecutor({"not a valid identifier": make_add_invokable()})

    def test_non_invokable_item_raises(self) -> None:
        with pytest.raises(TypeError):
            A2AtomicExecutor([object()])  # type: ignore[list-item]

    def test_wrong_invokables_type_raises(self) -> None:
        with pytest.raises(TypeError):
            A2AtomicExecutor("not a list or mapping")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# to_agent_card()
# --------------------------------------------------------------------------- #


class TestToAgentCard:
    def test_builds_one_skill_per_invokable_from_shared_metadata(self) -> None:
        executor = A2AtomicExecutor([make_add_invokable("add")])
        card = executor.to_agent_card(
            "http://127.0.0.1:9",
            name="TestAgent",
            description="Test.",
            version="0.1.0",
            transport_mode=TRANSPORT_JSON_RPC,
        )
        assert card.name == "TestAgent"
        assert card.version == "0.1.0"
        assert len(card.skills) == 1
        assert card.skills[0].id == "add"
        assert card.skills[0].name == "add"
        assert card.skills[0].description == "Adds two numbers."

    def test_supported_interface_matches_transport_mode(self) -> None:
        executor = A2AtomicExecutor([make_add_invokable()])
        card = executor.to_agent_card(
            "http://127.0.0.1:9", name="A", description="D", transport_mode=TRANSPORT_JSON_RPC
        )
        assert len(card.supported_interfaces) == 1
        assert card.supported_interfaces[0].protocol_binding == TRANSPORT_JSON_RPC
        assert card.supported_interfaces[0].url == "http://127.0.0.1:9"

    def test_publishes_param_schema_extension_with_full_metadata(self) -> None:
        executor = A2AtomicExecutor([make_add_invokable("add")])
        card = executor.to_agent_card(
            "http://127.0.0.1:9", name="A", description="D", transport_mode=TRANSPORT_JSON_RPC
        )

        extensions = list(card.capabilities.extensions)
        assert len(extensions) == 1
        ext = extensions[0]
        assert ext.uri == PARAM_SCHEMA_EXT_URI
        assert ext.required is False

        raw = MessageToDict(ext.params)
        assert set(raw.keys()) == {"add"}
        assert raw["add"]["remote_name"] == "add"
        assert raw["add"]["description"] == "Adds two numbers."
        assert len(raw["add"]["params"]) == 2
        assert raw["add"]["return_type"] == "int"

    def test_builds_fresh_card_every_call_no_stale_caching(self) -> None:
        executor = A2AtomicExecutor([make_add_invokable()])
        card1 = executor.to_agent_card(
            "http://127.0.0.1:9", name="First", description="D", transport_mode=TRANSPORT_JSON_RPC
        )
        card2 = executor.to_agent_card(
            "http://127.0.0.1:9", name="Second", description="D", transport_mode=TRANSPORT_JSON_RPC
        )
        assert card1.name == "First"
        assert card2.name == "Second"

    def test_invalid_transport_mode_raises(self) -> None:
        executor = A2AtomicExecutor([make_add_invokable()])
        with pytest.raises(ValueError):
            executor.to_agent_card(
                "http://127.0.0.1:9", name="A", description="D", transport_mode="bogus"
            )

    def test_empty_base_url_raises(self) -> None:
        executor = A2AtomicExecutor([make_add_invokable()])
        with pytest.raises(ValueError):
            executor.to_agent_card("", name="A", description="D", transport_mode=TRANSPORT_JSON_RPC)


# --------------------------------------------------------------------------- #
# cancel() -- unit-level, no live server needed
# --------------------------------------------------------------------------- #


class TestCancel:
    def test_publishes_task_state_canceled(self) -> None:
        executor = A2AtomicExecutor([make_add_invokable()])
        queue = EventQueueLegacy()
        context = RequestContext(call_context=SimpleNamespace(), task_id="t1", context_id="c1")

        asyncio.run(executor.cancel(context, queue))

        event = asyncio.run(queue.dequeue_event())
        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.TASK_STATE_CANCELED


# --------------------------------------------------------------------------- #
# End-to-end dispatch via a real running server + A2AClientHub
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class FixtureServer:
    http_url: str


@pytest.fixture(scope="module")
def fixture_server():
    """Real A2AtomicExecutor server, JSON-RPC only (transport-layer
    correctness across all three transports is already Pass 1's job; this
    pass only needs to prove routing/dispatch). JSON-RPC is mounted at the
    site root so one shared base_url works for both the agent-card route and
    the RPC route without a caller-chosen rpc_url suffix.

    Runs on a dedicated background loop via start_background_loop()/
    stop_background_loop() -- the same mechanism test_client_hub.py's own
    fixture_server uses -- rather than a bare threading.Thread(target=
    uvicorn_server.run). A bare-thread uvicorn.Server.run() spins up its own
    fresh event loop per thread and stopping it via should_exit doesn't
    guarantee in-flight ActiveTask producer/consumer tasks get cancelled
    before that loop closes; on Windows this left orphaned asyncio Tasks
    that hung the whole interpreter at process exit (observed directly
    during this pass's own test run, not theoretical)."""
    port = _free_port()
    http_url = f"http://127.0.0.1:{port}"

    executor = A2AtomicExecutor(
        {"add": make_add_invokable("add"), "boom": make_failing_invokable("boom")}
    )
    card = executor.to_agent_card(
        http_url, name="AtomicFixtureAgent", description="Fixture.", transport_mode=TRANSPORT_JSON_RPC
    )

    request_handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card
    )
    app = Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(request_handler, rpc_url="/"),
        ]
    )

    import uvicorn

    uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical")
    uvicorn_server = uvicorn.Server(uvicorn_config)

    loop, thread = start_background_loop()
    run_coro_sync(_start_uvicorn(uvicorn_server), loop=loop)

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

    yield FixtureServer(http_url=http_url)

    async def _shutdown() -> None:
        # request_handler.aclose() drains DefaultRequestHandler's
        # ActiveTaskRegistry -- confirmed via its own docstring as the
        # SDK-sanctioned shutdown hook ("so a server shutdown leaves no
        # pending asyncio.Task", "intended to be wired into an ASGI
        # lifespan/on_shutdown hook"). Skipping this is exactly what left
        # every request's ActiveTask (producer+consumer tasks) orphaned,
        # which is what hung the whole pytest session at final teardown --
        # confirmed directly during this pass's own verification.
        await request_handler.aclose()
        uvicorn_server.should_exit = True
        await asyncio.sleep(0.3)

    run_coro_sync(_shutdown(), loop=loop)
    stop_background_loop(loop, thread)


async def _start_uvicorn(uvicorn_server: "uvicorn.Server") -> None:
    asyncio.ensure_future(uvicorn_server.serve())


class TestEndToEndDispatch:
    def test_successful_call_round_trips_result(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            assert hub.call_atomic_skill("add", {"a": 2, "b": 3}) == 5
        finally:
            hub.close()

    def test_unknown_skill_raises(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            with pytest.raises(A2AProxyError):
                hub.call_atomic_skill("does_not_exist", {})
        finally:
            hub.close()

    def test_invokable_exception_raises(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            with pytest.raises(A2AProxyError):
                hub.call_atomic_skill("boom", {})
        finally:
            hub.close()

    def test_hub_detects_published_atomic_skills(self, fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            skills = hub.get_atomic_skills()
            assert set(skills) == {"add", "boom"}
            assert skills["add"].description == "Adds two numbers."
            assert len(skills["add"].params) == 2
        finally:
            hub.close()
