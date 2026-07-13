from __future__ import annotations

import asyncio
import importlib
from datetime import timedelta
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


class TestMCPClientHubConstructionExtras:
    def test_defaults_are_none(self) -> None:
        hub = MCPClientHub("stdio", command="python")

        assert hub.read_timeout_seconds is None
        assert hub.client_kwargs is None
        assert hub.session_kwargs is None

    def test_read_timeout_seconds_is_coerced_to_float(self) -> None:
        hub = MCPClientHub("stdio", command="python", read_timeout_seconds=5)

        assert hub.read_timeout_seconds == 5.0

    def test_client_kwargs_and_session_kwargs_are_stored(self) -> None:
        hub = MCPClientHub(
            "stdio",
            command="python",
            client_kwargs={"cwd": "/tmp"},
            session_kwargs={"client_info": None},
        )

        assert hub.client_kwargs == {"cwd": "/tmp"}
        assert hub.session_kwargs == {"client_info": None}

    @pytest.mark.parametrize("param_name", ["client_kwargs", "session_kwargs"])
    def test_non_mapping_kwargs_raise(self, param_name: str) -> None:
        with pytest.raises(ValueError, match=f"{param_name} must be a mapping"):
            MCPClientHub("stdio", command="python", **{param_name: "bad"})  # type: ignore[arg-type]

    @pytest.mark.parametrize("param_name", ["client_kwargs", "session_kwargs"])
    def test_kwargs_properties_are_immutable_mapping(self, param_name: str) -> None:
        hub = MCPClientHub("stdio", command="python", **{param_name: {"a": 1}})

        with pytest.raises(TypeError):
            getattr(hub, param_name)["a"] = 2  # type: ignore[index]

    def test_streamable_http_headers_and_http_client_collide(self) -> None:
        with pytest.raises(ValueError, match="Cannot set both headers"):
            MCPClientHub(
                "streamable_http",
                endpoint="http://localhost:8000/mcp",
                headers={"Authorization": "Bearer x"},
                client_kwargs={"http_client": object()},
            )

    def test_streamable_http_headers_alone_is_fine(self) -> None:
        hub = MCPClientHub(
            "streamable_http",
            endpoint="http://localhost:8000/mcp",
            headers={"Authorization": "Bearer x"},
        )

        assert hub.headers == {"Authorization": "Bearer x"}

    def test_streamable_http_client_kwargs_without_http_client_is_fine(self) -> None:
        hub = MCPClientHub(
            "streamable_http",
            endpoint="http://localhost:8000/mcp",
            headers={"Authorization": "Bearer x"},
            client_kwargs={"terminate_on_close": False},
        )

        assert hub.client_kwargs == {"terminate_on_close": False}

    def test_collision_check_does_not_apply_to_sse(self) -> None:
        sentinel = object()
        hub = MCPClientHub(
            "sse",
            endpoint="http://localhost:8000/sse",
            headers={"Authorization": "Bearer x"},
            client_kwargs={"http_client": sentinel},
        )

        assert hub.headers == {"Authorization": "Bearer x"}
        assert hub.client_kwargs["http_client"] is sentinel


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
            "read_timeout_seconds": None,
            "has_client_kwargs": False,
            "client_kwargs_keys": [],
            "has_session_kwargs": False,
            "session_kwargs_keys": [],
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
    def test_refresh_with_nothing_provided_raises(self) -> None:
        hub = MCPClientHub(
            "sse",
            endpoint="http://localhost:8000/sse",
            headers={"X-Old": "yes"},
        )

        with pytest.raises(ValueError, match="requires at least one of"):
            hub.refresh()

    def test_refresh_with_headers_updates_stored_headers(self) -> None:
        hub = MCPClientHub("sse", endpoint="http://localhost:8000/sse")

        hub.refresh(headers={"X-New": "yes"})

        assert hub.headers == {"X-New": "yes"}

    def test_refresh_header_validation_applies(self) -> None:
        hub = MCPClientHub("sse", endpoint="http://localhost:8000/sse")

        with pytest.raises(ValueError, match="headers must be a mapping"):
            hub.refresh(headers="bad")  # type: ignore[arg-type]

    def test_refresh_client_kwargs_alone_updates_only_client_kwargs(self) -> None:
        hub = MCPClientHub(
            "sse",
            endpoint="http://localhost:8000/sse",
            headers={"X-Old": "yes"},
            session_kwargs={"client_info": None},
        )

        hub.refresh(client_kwargs={"timeout": 10})

        assert hub.client_kwargs == {"timeout": 10}
        assert hub.headers == {"X-Old": "yes"}
        assert hub.session_kwargs == {"client_info": None}

    def test_refresh_session_kwargs_alone_updates_only_session_kwargs(self) -> None:
        hub = MCPClientHub(
            "sse",
            endpoint="http://localhost:8000/sse",
            headers={"X-Old": "yes"},
            client_kwargs={"timeout": 10},
        )

        hub.refresh(session_kwargs={"client_info": None})

        assert hub.session_kwargs == {"client_info": None}
        assert hub.headers == {"X-Old": "yes"}
        assert hub.client_kwargs == {"timeout": 10}

    def test_refresh_replaces_client_kwargs_wholesale_not_merged(self) -> None:
        hub = MCPClientHub("sse", endpoint="http://localhost:8000/sse")

        hub.refresh(client_kwargs={"a": 1})
        hub.refresh(client_kwargs={"b": 2})

        assert hub.client_kwargs == {"b": 2}

    def test_refresh_client_kwargs_validation_applies(self) -> None:
        hub = MCPClientHub("sse", endpoint="http://localhost:8000/sse")

        with pytest.raises(ValueError, match="client_kwargs must be a mapping"):
            hub.refresh(client_kwargs="bad")  # type: ignore[arg-type]

    def test_refresh_revalidates_streamable_http_collision(self) -> None:
        hub = MCPClientHub(
            "streamable_http",
            endpoint="http://localhost:8000/mcp",
            headers={"Authorization": "Bearer x"},
        )

        with pytest.raises(ValueError, match="Cannot set both headers"):
            hub.refresh(client_kwargs={"http_client": object()})

    def test_refresh_collision_failure_does_not_mutate_stored_state(self) -> None:
        """A refresh() call that fails validation must leave headers/
        client_kwargs/session_kwargs exactly as they were — not partially
        applied despite the raise."""
        hub = MCPClientHub(
            "streamable_http",
            endpoint="http://localhost:8000/mcp",
            headers={"Authorization": "Bearer x"},
            session_kwargs={"client_info": None},
        )

        with pytest.raises(ValueError, match="Cannot set both headers"):
            hub.refresh(
                client_kwargs={"http_client": object()},
                session_kwargs={"client_info": "changed"},
            )

        assert hub.headers == {"Authorization": "Bearer x"}
        assert hub.client_kwargs is None
        assert hub.session_kwargs == {"client_info": None}


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


class RecordingCall:
    """Captures every (args, kwargs) call and delegates to a return-value factory."""

    def __init__(self, return_value_factory: Callable[..., Any]) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._return_value_factory = return_value_factory

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self._return_value_factory(*args, **kwargs)


async def _identity_op(session: Any) -> Any:
    return session


class TestAwithSessionKwargsForwarding:
    """
    Proves client_kwargs/session_kwargs/read_timeout_seconds actually reach
    the underlying transport constructors and ClientSession, by
    monkeypatching those symbols with recording fakes (same style as
    `TestAwithSessionErrorWrapping._make_hub`).
    """

    def test_stdio_forwards_client_kwargs_to_server_params(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        recorder = RecordingCall(lambda **kwargs: SimpleNamespace(**kwargs))
        monkeypatch.setattr(client_hub_module, "StdioServerParameters", recorder)
        monkeypatch.setattr(
            client_hub_module,
            "stdio_client",
            lambda server: FakeAsyncCM(("read", "write")),
        )
        monkeypatch.setattr(
            client_hub_module, "ClientSession", lambda *a, **k: FakeRealSession()
        )

        hub = MCPClientHub("stdio", command="python", client_kwargs={"cwd": "/tmp"})
        asyncio.run(hub._awith_session(_identity_op))

        _, kwargs = recorder.calls[0]
        assert kwargs["command"] == "python"
        assert kwargs["cwd"] == "/tmp"

    def test_sse_forwards_client_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = RecordingCall(lambda **kwargs: FakeAsyncCM(("read", "write")))
        monkeypatch.setattr(client_hub_module, "sse_client", recorder)
        monkeypatch.setattr(
            client_hub_module, "ClientSession", lambda *a, **k: FakeRealSession()
        )

        hub = MCPClientHub(
            "sse",
            endpoint="http://localhost:8000/sse",
            client_kwargs={"timeout": 10},
        )
        asyncio.run(hub._awith_session(_identity_op))

        _, kwargs = recorder.calls[0]
        assert kwargs["url"] == "http://localhost:8000/sse"
        assert kwargs["timeout"] == 10

    def test_streamable_http_forwards_client_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        recorder = RecordingCall(lambda **kwargs: FakeAsyncCM(("read", "write")))
        monkeypatch.setattr(client_hub_module, "streamable_http_client", recorder)
        monkeypatch.setattr(
            client_hub_module, "ClientSession", lambda *a, **k: FakeRealSession()
        )

        hub = MCPClientHub(
            "streamable_http",
            endpoint="http://localhost:8000/mcp",
            client_kwargs={"terminate_on_close": False},
        )
        asyncio.run(hub._awith_session(_identity_op))

        _, kwargs = recorder.calls[0]
        assert kwargs["url"] == "http://localhost:8000/mcp"
        assert kwargs["terminate_on_close"] is False

    def test_client_session_forwards_read_timeout_and_session_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            client_hub_module,
            "stdio_client",
            lambda server: FakeAsyncCM(("read", "write")),
        )
        recorder = RecordingCall(lambda *a, **k: FakeRealSession())
        monkeypatch.setattr(client_hub_module, "ClientSession", recorder)

        hub = MCPClientHub(
            "stdio",
            command="python",
            read_timeout_seconds=12.5,
            session_kwargs={"client_info": None},
        )
        asyncio.run(hub._awith_session(_identity_op))

        _, kwargs = recorder.calls[0]
        assert kwargs["read_timeout_seconds"] == timedelta(seconds=12.5)
        assert kwargs["client_info"] is None

    def test_foreign_client_kwargs_key_is_wrapped_as_connection_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_sse_client(
            url: str,
            headers: dict[str, Any] | None = None,
            timeout: float = 5,
            sse_read_timeout: float = 300,
        ) -> FakeAsyncCM:
            return FakeAsyncCM(("read", "write"))

        monkeypatch.setattr(client_hub_module, "sse_client", fake_sse_client)
        monkeypatch.setattr(
            client_hub_module, "ClientSession", lambda *a, **k: FakeRealSession()
        )

        hub = MCPClientHub(
            "sse",
            endpoint="http://localhost:8000/sse",
            client_kwargs={"env": {"X": "1"}},
        )

        with pytest.raises(MCPConnectionError, match="Failed MCP operation"):
            asyncio.run(hub._awith_session(_identity_op))
