from __future__ import annotations

import asyncio
import base64
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import httpx
import pytest
from a2a.helpers import new_message, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import Part, Role
from google.protobuf.json_format import MessageToDict

from atomic_agentic.a2a.A2AClientHub import A2AClientHub
from atomic_agentic.a2a.A2AtomicExecutor import A2AtomicExecutor
from atomic_agentic.constants.a2a_sdk import TRANSPORT_JSON_RPC
from atomic_agentic.core.Invokable import AtomicInvokable
from atomic_agentic.exceptions import ToolDefinitionError, ToolInvocationError
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results.atomic import AtomicResult
from atomic_agentic.models.results.tools import A2AProxyToolResult
from atomic_agentic.tools.a2a_sdk import A2AProxyTool
from atomic_agentic.utils.a2a import _dict_to_part, _part_to_dict
from atomic_agentic.utils.core import run_coro_sync, start_background_loop, stop_background_loop


# --------------------------------------------------------------------------- #
# _dict_to_part / _part_to_dict -- unit level, no server needed
# --------------------------------------------------------------------------- #


class TestDictToPart:
    def test_text(self) -> None:
        part = _dict_to_part({"text": "hello"})
        assert part.text == "hello"

    def test_data(self) -> None:
        part = _dict_to_part({"data": {"a": 1}})
        assert MessageToDict(part.data) == {"a": 1}

    def test_raw_b64_from_str(self) -> None:
        raw = b"binary-payload"
        encoded = base64.b64encode(raw).decode("ascii")
        part = _dict_to_part({"raw_b64": encoded})
        assert part.raw == raw

    def test_raw_b64_from_bytes(self) -> None:
        raw = b"binary-payload"
        part = _dict_to_part({"raw_b64": raw})
        assert part.raw == raw

    def test_url(self) -> None:
        part = _dict_to_part({"url": "https://example.com"})
        assert part.url == "https://example.com"

    def test_filename_and_media_type_on_raw(self) -> None:
        part = _dict_to_part(
            {"raw_b64": b"x", "filename": "chart.png", "media_type": "image/png"}
        )
        assert part.filename == "chart.png"
        assert part.media_type == "image/png"

    def test_zero_content_keys_raises(self) -> None:
        with pytest.raises(ValueError):
            _dict_to_part({})

    def test_multiple_content_keys_raises(self) -> None:
        with pytest.raises(ValueError):
            _dict_to_part({"text": "a", "url": "https://example.com"})

    def test_filename_on_text_raises(self) -> None:
        with pytest.raises(ValueError):
            _dict_to_part({"text": "a", "filename": "x.txt"})

    def test_filename_on_data_raises(self) -> None:
        with pytest.raises(ValueError):
            _dict_to_part({"data": {}, "filename": "x.json"})

    def test_invalid_base64_raises(self) -> None:
        with pytest.raises(ValueError):
            _dict_to_part({"raw_b64": "not valid base64!!"})

    def test_wrong_type_for_text_raises(self) -> None:
        with pytest.raises(TypeError):
            _dict_to_part({"text": 123})

    def test_non_mapping_raises(self) -> None:
        with pytest.raises(TypeError):
            _dict_to_part(["not", "a", "mapping"])  # type: ignore[arg-type]


class TestPartToDict:
    def test_text_round_trips(self) -> None:
        part = _dict_to_part({"text": "hello"})
        d = _part_to_dict(part)
        assert d == {
            "text": "hello",
            "data": None,
            "raw_b64": None,
            "url": None,
            "filename": None,
            "media_type": None,
        }

    def test_raw_round_trips_to_b64_str(self) -> None:
        raw = b"binary-payload"
        part = _dict_to_part({"raw_b64": raw, "filename": "f.bin", "media_type": "application/octet-stream"})
        d = _part_to_dict(part)
        assert d["raw_b64"] == base64.b64encode(raw).decode("ascii")
        assert isinstance(d["raw_b64"], str)
        assert d["filename"] == "f.bin"
        assert d["media_type"] == "application/octet-stream"

    def test_data_round_trips(self) -> None:
        part = _dict_to_part({"data": {"a": 1}})
        d = _part_to_dict(part)
        assert d["data"] == {"a": 1}

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError):
            _part_to_dict("not a part")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Fixture servers
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class AddInvokable(AtomicInvokable):
    def invoke(self, inputs: Mapping[str, Any]) -> AtomicResult:
        started_at = datetime.now(timezone.utc)
        result = inputs["a"] + inputs["b"]
        ended_at = datetime.now(timezone.utc)
        return self.make_result(result, started_at, ended_at)


class FailingInvokable(AtomicInvokable):
    def invoke(self, inputs: Mapping[str, Any]) -> AtomicResult:
        raise RuntimeError("deliberate failure")


def _make_param(name: str, index: int) -> ParamSpec:
    return ParamSpec(name=name, index=index, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="Any")


def _make_add_invokable() -> AddInvokable:
    return AddInvokable(
        name="add",
        namespace="tests",
        description="Adds two numbers.",
        parameters=[_make_param("a", 0), _make_param("b", 1)],
        return_type="int",
    )


def _make_failing_invokable() -> FailingInvokable:
    return FailingInvokable(name="boom", namespace="tests", description="Always raises.", parameters=[], return_type="Any")


async def _start_uvicorn(uvicorn_server: "uvicorn.Server") -> None:
    asyncio.ensure_future(uvicorn_server.serve())


async def _wait_ready(http_url: str) -> None:
    deadline = time.monotonic() + 10
    async with httpx.AsyncClient() as probe:
        while True:
            try:
                response = await probe.get(f"{http_url}/.well-known/agent-card.json", timeout=1)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("fixture A2A server did not become ready in time.")
            await asyncio.sleep(0.1)


@dataclass
class FixtureServer:
    http_url: str


@pytest.fixture(scope="module")
def atomic_fixture_server():
    """Real A2AtomicExecutor server (skill mode target) -- mirrors
    test_atomic_executor.py's own fixture exactly."""
    port = _free_port()
    http_url = f"http://127.0.0.1:{port}"

    executor = A2AtomicExecutor({"add": _make_add_invokable(), "boom": _make_failing_invokable()})
    card = executor.to_agent_card(
        http_url, name="ProxyToolAtomicFixture", description="Fixture.", transport_mode=TRANSPORT_JSON_RPC
    )
    request_handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card)
    _run_server(request_handler, card, http_url, port)

    loop, thread = start_background_loop()
    run_coro_sync(_start_uvicorn(_uvicorn_server_box[-1]), loop=loop)
    run_coro_sync(_wait_ready(http_url), loop=loop)

    yield FixtureServer(http_url=http_url)

    run_coro_sync(_shutdown_server(request_handler, _uvicorn_server_box[-1]), loop=loop)
    stop_background_loop(loop, thread)


class _EchoAgentExecutor(AgentExecutor):
    """Generic mode target -- echoes back whatever Parts it receives, and
    also echoes context.metadata back as one extra data Part, so metadata
    passthrough is directly verifiable."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        from a2a.helpers import new_data_part

        parts: list[Part] = list(context.message.parts)
        if context.metadata:
            parts = [*parts, new_data_part({"received_metadata": dict(context.metadata)})]
        reply = new_message(parts, role=Role.ROLE_AGENT)
        await event_queue.enqueue_event(reply)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


_uvicorn_server_box: list[Any] = []


def _run_server(request_handler: DefaultRequestHandler, card, http_url: str, port: int) -> None:
    from starlette.applications import Starlette
    import uvicorn

    app = Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(request_handler, rpc_url="/"),
        ]
    )
    uvicorn_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical")
    _uvicorn_server_box.append(uvicorn.Server(uvicorn_config))


async def _shutdown_server(request_handler: DefaultRequestHandler, uvicorn_server) -> None:
    await request_handler.aclose()
    uvicorn_server.should_exit = True
    await asyncio.sleep(0.3)


@pytest.fixture(scope="module")
def echo_fixture_server():
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

    port = _free_port()
    http_url = f"http://127.0.0.1:{port}"
    card = AgentCard(
        name="ProxyToolEchoFixture",
        description="Echoes back whatever Parts it receives.",
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(id="echo", name="echo", description="Echoes parts back.")],
        capabilities=AgentCapabilities(),
        supported_interfaces=[AgentInterface(url=http_url, protocol_binding=TRANSPORT_JSON_RPC)],
    )
    request_handler = DefaultRequestHandler(
        agent_executor=_EchoAgentExecutor(), task_store=InMemoryTaskStore(), agent_card=card
    )
    _run_server(request_handler, card, http_url, port)

    loop, thread = start_background_loop()
    run_coro_sync(_start_uvicorn(_uvicorn_server_box[-1]), loop=loop)
    run_coro_sync(_wait_ready(http_url), loop=loop)

    yield FixtureServer(http_url=http_url)

    run_coro_sync(_shutdown_server(request_handler, _uvicorn_server_box[-1]), loop=loop)
    stop_background_loop(loop, thread)


# --------------------------------------------------------------------------- #
# Skill mode
# --------------------------------------------------------------------------- #


class TestSkillMode:
    def test_construction_defaults_name_namespace(self, atomic_fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(atomic_fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            tool = A2AProxyTool(hub, skill_id="add")
            assert tool.name == "add"
            assert tool.namespace == "a2a"
            assert tool.skill_id == "add"
            assert [p.name for p in tool.parameters] == ["a", "b"]
            assert tool.return_type == "int"
        finally:
            hub.close()

    def test_unknown_skill_id_raises(self, atomic_fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(atomic_fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            with pytest.raises(ToolDefinitionError):
                A2AProxyTool(hub, skill_id="does_not_exist")
        finally:
            hub.close()

    def test_sync_invocation(self, atomic_fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(atomic_fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            tool = A2AProxyTool(hub, skill_id="add")
            result = tool.invoke({"a": 2, "b": 3})
            assert result.result == 5
            assert isinstance(result, A2AProxyToolResult)
            assert result.skill_id == "add"
            assert result.transport_mode == TRANSPORT_JSON_RPC
        finally:
            hub.close()

    def test_async_invocation(self, atomic_fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(atomic_fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            result = asyncio.run(A2AProxyTool(hub, skill_id="add").async_invoke({"a": 4, "b": 5}))
            assert result.result == 9
        finally:
            hub.close()

    def test_remote_failure_raises_tool_invocation_error(self, atomic_fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(atomic_fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            tool = A2AProxyTool(hub, skill_id="boom")
            with pytest.raises(ToolInvocationError):
                tool.invoke({})
        finally:
            hub.close()

    def test_to_dict(self, atomic_fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(atomic_fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            data = A2AProxyTool(hub, skill_id="add").to_dict()
            assert data["skill_id"] == "add"
            assert "client_hub" in data
        finally:
            hub.close()


# --------------------------------------------------------------------------- #
# Generic mode
# --------------------------------------------------------------------------- #


class TestGenericMode:
    def test_construction_defaults_name_namespace(self, echo_fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(echo_fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            tool = A2AProxyTool(hub)
            assert tool.name == "send_parts"
            assert tool.namespace == "a2a"
            assert tool.skill_id is None
            assert {p.name for p in tool.parameters} == {"parts", "metadata"}
            assert tool.return_type == "list"
        finally:
            hub.close()

    def test_sync_round_trip_mixed_kinds(self, echo_fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(echo_fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            tool = A2AProxyTool(hub)
            raw = b"payload"
            result = tool.invoke(
                {
                    "parts": [
                        {"text": "hello"},
                        {"data": {"a": 1}},
                        {"raw_b64": raw, "filename": "f.bin"},
                        {"url": "https://example.com"},
                    ]
                }
            )
            assert isinstance(result, A2AProxyToolResult)
            assert result.skill_id is None
            values = result.result
            assert values[0]["text"] == "hello"
            assert values[1]["data"] == {"a": 1}
            assert values[2]["raw_b64"] == base64.b64encode(raw).decode("ascii")
            assert values[2]["filename"] == "f.bin"
            assert values[3]["url"] == "https://example.com"
        finally:
            hub.close()

    def test_async_round_trip(self, echo_fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(echo_fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            tool = A2AProxyTool(hub)
            result = asyncio.run(tool.async_invoke({"parts": [{"text": "async hi"}]}))
            assert result.result[0]["text"] == "async hi"
        finally:
            hub.close()

    def test_metadata_passthrough(self, echo_fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(echo_fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            tool = A2AProxyTool(hub)
            result = tool.invoke({"parts": [{"text": "hi"}], "metadata": {"session": "abc"}})
            echoed = next(v for v in result.result if v["data"] is not None)
            assert echoed["data"] == {"received_metadata": {"session": "abc"}}
        finally:
            hub.close()

    def test_missing_parts_raises(self, echo_fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(echo_fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            with pytest.raises(ToolInvocationError):
                A2AProxyTool(hub).invoke({})
        finally:
            hub.close()

    def test_malformed_part_dict_raises(self, echo_fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(echo_fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            with pytest.raises(ToolInvocationError):
                A2AProxyTool(hub).invoke({"parts": [{}]})
        finally:
            hub.close()

    def test_explicit_name_overrides_default(self, echo_fixture_server: FixtureServer) -> None:
        hub = A2AClientHub(echo_fixture_server.http_url, TRANSPORT_JSON_RPC, False)
        try:
            tool = A2AProxyTool(hub, name="custom_send", namespace="custom_ns")
            assert tool.name == "custom_send"
            assert tool.namespace == "custom_ns"
        finally:
            hub.close()
