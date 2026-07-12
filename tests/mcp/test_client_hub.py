from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import pytest

from atomic_agentic.exceptions import MCPConnectionError, MCPToolError
from atomic_agentic.mcp.MCPClientHub import MCPClientHub

client_hub_module = importlib.import_module("atomic_agentic.mcp.MCPClientHub")


class FakeSession:
    def __init__(self) -> None:
        self.called_tools: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> Any:
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="search",
                    description="Search documents.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                    outputSchema={
                        "type": "object",
                        "properties": {
                            "result": {"type": "string"},
                        },
                    },
                )
            ]
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.called_tools.append((name, arguments))
        return SimpleNamespace(
            content=["content"],
            structuredContent={"result": arguments},
            isError=False,
        )


class FailingListSession(FakeSession):
    async def list_tools(self) -> Any:
        raise ValueError("boom")


class FailingCallSession(FakeSession):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        raise ValueError("boom")


class FakeHub(MCPClientHub):
    def __init__(self, session: FakeSession, **kwargs: Any) -> None:
        self.fake_session = session
        super().__init__(**kwargs)

    async def _awith_session(
        self,
        operation: Callable[[FakeSession], Awaitable[Any]],
    ) -> Any:
        return await operation(self.fake_session)


class TestMCPClientHubConstruction:
    def test_valid_stdio_construction(self) -> None:
        hub = MCPClientHub("stdio", command="python", args=["server.py"])

        assert hub.transport_mode == "stdio"
        assert hub.command == "python"
        assert hub.args == ("server.py",)
        assert hub.endpoint is None

    def test_valid_sse_construction(self) -> None:
        hub = MCPClientHub("sse", endpoint="http://localhost:8000/sse")

        assert hub.transport_mode == "sse"
        assert hub.endpoint == "http://localhost:8000/sse"

    def test_valid_streamable_http_construction(self) -> None:
        hub = MCPClientHub("streamable_http", endpoint="http://localhost:8000/mcp")

        assert hub.transport_mode == "streamable_http"
        assert hub.endpoint == "http://localhost:8000/mcp"

    @pytest.mark.parametrize("mode", ["bad", "", "websocket"])
    def test_invalid_transport_mode_raises(self, mode: str) -> None:
        with pytest.raises(ValueError, match="transport_mode"):
            MCPClientHub(mode)  # type: ignore[arg-type]

    def test_stdio_requires_command(self) -> None:
        with pytest.raises(ValueError, match="stdio transport requires"):
            MCPClientHub("stdio")

    @pytest.mark.parametrize("mode", ["sse", "streamable_http"])
    def test_http_modes_require_endpoint(self, mode: str) -> None:
        with pytest.raises(ValueError, match="requires a non-empty endpoint"):
            MCPClientHub(mode)  # type: ignore[arg-type]

    def test_args_must_be_list_of_strings(self) -> None:
        with pytest.raises(ValueError, match="args"):
            MCPClientHub("stdio", command="python", args=("server.py",))  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="args"):
            MCPClientHub("stdio", command="python", args=["server.py", 1])  # type: ignore[list-item]


class TestMCPClientHubHeaders:
    def test_scalar_header_values_are_normalized(self) -> None:
        hub = MCPClientHub(
            "sse",
            endpoint="http://localhost:8000/sse",
            headers={
                "X-Int": 1,
                "X-Float": 1.5,
                "X-Bool": True,
                "X-Bytes": b"abc",
                "X-ByteArray": bytearray(b"xyz"),
            },  # type: ignore[arg-type]
        )

        assert hub.headers == {
            "X-Int": "1",
            "X-Float": "1.5",
            "X-Bool": "true",
            "X-Bytes": "abc",
            "X-ByteArray": "xyz",
        }
    def test_headers_are_normalized_and_hidden_in_to_dict(self) -> None:
        hub = MCPClientHub(
            "sse",
            endpoint="http://localhost:8000/sse",
            headers={"Authorization": "Bearer secret", "X-Test": "yes"},
        )

        assert hub.headers == {"Authorization": "Bearer secret", "X-Test": "yes"}

        data = hub.to_dict()
        assert data["has_headers"] is True
        assert data["header_keys"] == ["Authorization", "X-Test"]
        assert "Bearer secret" not in str(data)

    def test_headers_are_immutable_mapping(self) -> None:
        hub = MCPClientHub(
            "sse",
            endpoint="http://localhost:8000/sse",
            headers={"Authorization": "Bearer secret"},
        )

        with pytest.raises(TypeError):
            hub.headers["Authorization"] = "changed"  # type: ignore[index]

    def test_headers_setter_revalidates(self) -> None:
        hub = MCPClientHub("sse", endpoint="http://localhost:8000/sse")

        hub.headers = {"X-Test": "yes"}

        assert hub.headers == {"X-Test": "yes"}

    @pytest.mark.parametrize(
        ("headers", "match"),
        [
            ("bad", "headers must be a mapping"),
            ({1: "ok"}, "header names must be strings"),
            ({"bad": None}, "must not be None"),
            ({"bad": {"nested": "value"}}, "must not be a mapping"),
            ({"bad": ["value"]}, "must not be a collection"),
            ({"bad": object()}, "must be str, int, float, bool, bytes, or bytearray"),
            ({"bad": b"\xff"}, "ASCII-decodable"),
            ({"bad\n": "value"}, "forbidden character"),
            ({"bad": "line\nbreak"}, "forbidden character"),
        ],
    )
    def test_invalid_headers_raise(self, headers: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            MCPClientHub(
                "sse",
                endpoint="http://localhost:8000/sse",
                headers=headers,  # type: ignore[arg-type]
            )


class TestMCPClientHubLocalHelpers:
    def test_to_dict_shape(self) -> None:
        hub = MCPClientHub(
            "stdio",
            command="python",
            args=["server.py"],
            headers={"X-Test": "yes"},
        )

        assert hub.to_dict() == {
            "transport_mode": "stdio",
            "endpoint": None,
            "command": "python",
            "args": ["server.py"],
            "has_headers": True,
            "header_keys": ["X-Test"],
        }

    def test_unpack_transport_streams_accepts_two_tuple(self) -> None:
        hub = MCPClientHub("stdio", command="python")

        assert hub._unpack_transport_streams(("read", "write")) == ("read", "write")

    def test_unpack_transport_streams_accepts_three_tuple(self) -> None:
        hub = MCPClientHub("stdio", command="python")

        assert hub._unpack_transport_streams(("read", "write", "session")) == (
            "read",
            "write",
        )

    def test_unpack_transport_streams_rejects_bad_shape(self) -> None:
        hub = MCPClientHub("stdio", command="python")

        with pytest.raises(MCPConnectionError, match="Unexpected transport stream shape"):
            hub._unpack_transport_streams(("only-one",))


class TestMCPClientHubOperationsWithoutRealServer:
    def test_list_tools_uses_fake_session(self) -> None:
        session = FakeSession()
        hub = FakeHub(
            session,
            transport_mode="stdio",
            command="python",
        )

        tools = hub.list_tools()

        assert list(tools) == ["search"]
        assert tools["search"]["description"] == "Search documents."
        assert tools["search"]["return_type"] == "str"

    def test_call_tool_uses_fake_session(self) -> None:
        session = FakeSession()
        hub = FakeHub(
            session,
            transport_mode="stdio",
            command="python",
        )

        result = hub.call_tool("search", {"query": "hello"})

        assert session.called_tools == [("search", {"query": "hello"})]
        assert result == {
            "content": ["content"],
            "structuredContent": {"result": {"query": "hello"}},
            "isError": False,
        }

    def test_call_tool_rejects_blank_remote_name(self) -> None:
        hub = FakeHub(FakeSession(), transport_mode="stdio", command="python")

        with pytest.raises(ValueError, match="remote_name"):
            hub.call_tool("   ", {"query": "hello"})

    def test_call_tool_rejects_non_mapping_inputs(self) -> None:
        hub = FakeHub(FakeSession(), transport_mode="stdio", command="python")

        with pytest.raises(ValueError, match="inputs"):
            hub.call_tool("search", ["not", "mapping"])  # type: ignore[arg-type]

    def test_list_tools_wraps_session_failure_as_connection_error(self) -> None:
        hub = FakeHub(FailingListSession(), transport_mode="stdio", command="python")

        with pytest.raises(MCPConnectionError, match="Failed to list MCP tools"):
            hub.list_tools()

    def test_call_tool_wraps_session_failure_as_tool_error(self) -> None:
        hub = FakeHub(FailingCallSession(), transport_mode="stdio", command="python")

        with pytest.raises(MCPToolError, match="Failed to call MCP tool 'search'"):
            hub.call_tool("search", {"query": "hello"})


class TestMCPClientHubRefresh:
    def test_refresh_without_headers_is_noop(self) -> None:
        hub = MCPClientHub(
            "sse",
            endpoint="http://localhost:8000/sse",
            headers={"X-Old": "yes"},
        )

        hub.refresh()

        assert hub.headers == {"X-Old": "yes"}

    def test_refresh_with_headers_updates_stored_headers(self) -> None:
        hub = MCPClientHub("sse", endpoint="http://localhost:8000/sse")

        hub.refresh(headers={"X-New": "yes"})

        assert hub.headers == {"X-New": "yes"}

    def test_refresh_header_validation_applies(self) -> None:
        hub = MCPClientHub("sse", endpoint="http://localhost:8000/sse")

        with pytest.raises(ValueError, match="headers must be a mapping"):
            hub.refresh(headers="bad")  # type: ignore[arg-type]


class FakeAsyncCM:
    """Minimal async context manager yielding a fixed value."""

    def __init__(self, value: Any) -> None:
        self._value = value

    async def __aenter__(self) -> Any:
        return self._value

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


class FakeRealSession:
    """Stands in for `mcp.ClientSession` — just enough surface for
    `_awith_session` to reach `operation(session)`."""

    async def initialize(self) -> None:
        return None

    async def __aenter__(self) -> "FakeRealSession":
        return self

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


class TestAwithSessionErrorWrapping:
    """
    Exercises the *real* (non-`FakeHub`-overridden) `_awith_session` by
    monkeypatching `stdio_client`/`ClientSession` at module scope so setup
    trivially succeeds without a real subprocess/server. Proves the
    double-wrap fix: exceptions already typed by `operation` pass through
    unwrapped; anything else gets wrapped as `MCPConnectionError`.
    """

    def _make_hub(self, monkeypatch: pytest.MonkeyPatch) -> MCPClientHub:
        monkeypatch.setattr(
            client_hub_module,
            "stdio_client",
            lambda server_params: FakeAsyncCM(("read", "write")),
        )
        monkeypatch.setattr(
            client_hub_module, "ClientSession", lambda *a, **k: FakeRealSession()
        )
        return MCPClientHub("stdio", command="python")

    def test_typed_exception_from_operation_passes_through_unwrapped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hub = self._make_hub(monkeypatch)

        async def op(session: Any) -> Any:
            raise MCPToolError("Failed to call MCP tool 'x': boom")

        with pytest.raises(MCPToolError, match=r"^Failed to call MCP tool 'x': boom$"):
            asyncio.run(hub._awith_session(op))

    def test_unexpected_exception_from_operation_is_wrapped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hub = self._make_hub(monkeypatch)

        async def op(session: Any) -> Any:
            raise ValueError("boom")

        with pytest.raises(MCPConnectionError, match="Failed MCP operation"):
            asyncio.run(hub._awith_session(op))
