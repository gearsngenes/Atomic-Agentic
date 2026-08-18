from __future__ import annotations

import asyncio
import importlib
import sys
import threading
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import pytest

from atomic_agentic.exceptions import MCPConnectionError, MCPToolError
from atomic_agentic.mcp.MCPClientHub import MCPClientHub
from atomic_agentic.utils.core import start_background_loop, stop_background_loop

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
    """persistent=False always -- __init__ would otherwise try to eagerly
    connect via the real (unmocked) _connect_session when persistent=True,
    since only _do_operation is overridden here, not connection setup."""

    def __init__(self, session: FakeSession, **kwargs: Any) -> None:
        self.fake_session = session
        super().__init__(**kwargs)

    async def _do_operation(
        self,
        operation: Callable[[FakeSession], Awaitable[Any]],
    ) -> Any:
        return await operation(self.fake_session)


class TestMCPClientHubConstruction:
    def test_valid_stdio_construction(self) -> None:
        hub = MCPClientHub("stdio", persistent=False, command="python", args=["server.py"])

        assert hub.transport_mode == "stdio"
        assert hub.command == "python"
        assert hub.args == ("server.py",)
        assert hub.endpoint is None
        assert hub.persistent is False

    def test_valid_sse_construction(self) -> None:
        hub = MCPClientHub("sse", persistent=False, endpoint="http://localhost:8000/sse")

        assert hub.transport_mode == "sse"
        assert hub.endpoint == "http://localhost:8000/sse"

    def test_valid_streamable_http_construction(self) -> None:
        hub = MCPClientHub(
            "streamable_http", persistent=False, endpoint="http://localhost:8000/mcp"
        )

        assert hub.transport_mode == "streamable_http"
        assert hub.endpoint == "http://localhost:8000/mcp"

    @pytest.mark.parametrize("mode", ["bad", "", "websocket"])
    def test_invalid_transport_mode_raises(self, mode: str) -> None:
        with pytest.raises(ValueError, match="transport_mode"):
            MCPClientHub(mode, persistent=False)  # type: ignore[arg-type]

    def test_stdio_requires_command(self) -> None:
        with pytest.raises(ValueError, match="stdio transport requires"):
            MCPClientHub("stdio", persistent=False)

    @pytest.mark.parametrize("mode", ["sse", "streamable_http"])
    def test_http_modes_require_endpoint(self, mode: str) -> None:
        with pytest.raises(ValueError, match="requires a non-empty endpoint"):
            MCPClientHub(mode, persistent=False)  # type: ignore[arg-type]

    def test_args_must_be_list_of_strings(self) -> None:
        with pytest.raises(ValueError, match="args"):
            MCPClientHub(
                "stdio", persistent=False, command="python", args=("server.py",)
            )  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="args"):
            MCPClientHub(
                "stdio", persistent=False, command="python", args=["server.py", 1]
            )  # type: ignore[list-item]

    def test_persistent_must_be_bool(self) -> None:
        with pytest.raises(TypeError, match="persistent must be a bool"):
            MCPClientHub("stdio", "yes", command="python")  # type: ignore[arg-type]


class TestMCPClientHubConstructionExtras:
    def test_defaults_are_none(self) -> None:
        hub = MCPClientHub("stdio", persistent=False, command="python")

        assert hub.read_timeout_seconds is None
        assert hub.client_kwargs is None
        assert hub.session_kwargs is None

    def test_read_timeout_seconds_is_coerced_to_float(self) -> None:
        hub = MCPClientHub(
            "stdio", persistent=False, command="python", read_timeout_seconds=5
        )

        assert hub.read_timeout_seconds == 5.0

    def test_client_kwargs_and_session_kwargs_are_stored(self) -> None:
        hub = MCPClientHub(
            "stdio",
            persistent=False,
            command="python",
            client_kwargs={"cwd": "/tmp"},
            session_kwargs={"client_info": None},
        )

        assert hub.client_kwargs == {"cwd": "/tmp"}
        assert hub.session_kwargs == {"client_info": None}

    @pytest.mark.parametrize("param_name", ["client_kwargs", "session_kwargs"])
    def test_non_mapping_kwargs_raise(self, param_name: str) -> None:
        with pytest.raises(ValueError, match=f"{param_name} must be a mapping"):
            MCPClientHub(
                "stdio", persistent=False, command="python", **{param_name: "bad"}
            )  # type: ignore[arg-type]

    @pytest.mark.parametrize("param_name", ["client_kwargs", "session_kwargs"])
    def test_kwargs_properties_are_immutable_mapping(self, param_name: str) -> None:
        hub = MCPClientHub(
            "stdio", persistent=False, command="python", **{param_name: {"a": 1}}
        )

        with pytest.raises(TypeError):
            getattr(hub, param_name)["a"] = 2  # type: ignore[index]

    def test_streamable_http_headers_and_http_client_collide(self) -> None:
        with pytest.raises(ValueError, match="Cannot set both headers"):
            MCPClientHub(
                "streamable_http",
                persistent=False,
                endpoint="http://localhost:8000/mcp",
                headers={"Authorization": "Bearer x"},
                client_kwargs={"http_client": object()},
            )

    def test_streamable_http_headers_alone_is_fine(self) -> None:
        hub = MCPClientHub(
            "streamable_http",
            persistent=False,
            endpoint="http://localhost:8000/mcp",
            headers={"Authorization": "Bearer x"},
        )

        assert hub.headers == {"Authorization": "Bearer x"}

    def test_streamable_http_client_kwargs_without_http_client_is_fine(self) -> None:
        hub = MCPClientHub(
            "streamable_http",
            persistent=False,
            endpoint="http://localhost:8000/mcp",
            headers={"Authorization": "Bearer x"},
            client_kwargs={"terminate_on_close": False},
        )

        assert hub.client_kwargs == {"terminate_on_close": False}

    def test_collision_check_does_not_apply_to_sse(self) -> None:
        sentinel = object()
        hub = MCPClientHub(
            "sse",
            persistent=False,
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
            persistent=False,
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
            persistent=False,
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
            persistent=False,
            endpoint="http://localhost:8000/sse",
            headers={"Authorization": "Bearer secret"},
        )

        with pytest.raises(TypeError):
            hub.headers["Authorization"] = "changed"  # type: ignore[index]

    def test_headers_setter_revalidates(self) -> None:
        hub = MCPClientHub("sse", persistent=False, endpoint="http://localhost:8000/sse")

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
                persistent=False,
                endpoint="http://localhost:8000/sse",
                headers=headers,  # type: ignore[arg-type]
            )


class TestMCPClientHubLocalHelpers:
    def test_to_dict_shape(self) -> None:
        hub = MCPClientHub(
            "stdio",
            persistent=False,
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
            "persistent": False,
            "is_connected": False,
        }

    def test_unpack_transport_streams_accepts_two_tuple(self) -> None:
        hub = MCPClientHub("stdio", persistent=False, command="python")

        assert hub._unpack_transport_streams(("read", "write")) == ("read", "write")

    def test_unpack_transport_streams_accepts_three_tuple(self) -> None:
        hub = MCPClientHub("stdio", persistent=False, command="python")

        assert hub._unpack_transport_streams(("read", "write", "session")) == (
            "read",
            "write",
        )

    def test_unpack_transport_streams_rejects_bad_shape(self) -> None:
        hub = MCPClientHub("stdio", persistent=False, command="python")

        with pytest.raises(MCPConnectionError, match="Unexpected transport stream shape"):
            hub._unpack_transport_streams(("only-one",))


class TestMCPClientHubOperationsWithoutRealServer:
    def test_list_tools_uses_fake_session(self) -> None:
        session = FakeSession()
        hub = FakeHub(
            session,
            transport_mode="stdio",
            persistent=False,
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
            persistent=False,
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
        hub = FakeHub(
            FakeSession(), transport_mode="stdio", persistent=False, command="python"
        )

        with pytest.raises(ValueError, match="remote_name"):
            hub.call_tool("   ", {"query": "hello"})

    def test_call_tool_rejects_non_mapping_inputs(self) -> None:
        hub = FakeHub(
            FakeSession(), transport_mode="stdio", persistent=False, command="python"
        )

        with pytest.raises(ValueError, match="inputs"):
            hub.call_tool("search", ["not", "mapping"])  # type: ignore[arg-type]

    def test_list_tools_wraps_session_failure_as_connection_error(self) -> None:
        hub = FakeHub(
            FailingListSession(), transport_mode="stdio", persistent=False, command="python"
        )

        with pytest.raises(MCPConnectionError, match="Failed to list MCP tools"):
            hub.list_tools()

    def test_call_tool_wraps_session_failure_as_tool_error(self) -> None:
        hub = FakeHub(
            FailingCallSession(), transport_mode="stdio", persistent=False, command="python"
        )

        with pytest.raises(MCPToolError, match="Failed to call MCP tool 'search'"):
            hub.call_tool("search", {"query": "hello"})


class TestMCPClientHubRefresh:
    def test_refresh_with_nothing_provided_raises(self) -> None:
        hub = MCPClientHub(
            "sse",
            persistent=False,
            endpoint="http://localhost:8000/sse",
            headers={"X-Old": "yes"},
        )

        with pytest.raises(ValueError, match="requires at least one of"):
            hub.refresh()

    def test_refresh_with_headers_updates_stored_headers(self) -> None:
        hub = MCPClientHub("sse", persistent=False, endpoint="http://localhost:8000/sse")

        hub.refresh(headers={"X-New": "yes"})

        assert hub.headers == {"X-New": "yes"}

    def test_refresh_header_validation_applies(self) -> None:
        hub = MCPClientHub("sse", persistent=False, endpoint="http://localhost:8000/sse")

        with pytest.raises(ValueError, match="headers must be a mapping"):
            hub.refresh(headers="bad")  # type: ignore[arg-type]

    def test_refresh_client_kwargs_alone_updates_only_client_kwargs(self) -> None:
        hub = MCPClientHub(
            "sse",
            persistent=False,
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
            persistent=False,
            endpoint="http://localhost:8000/sse",
            headers={"X-Old": "yes"},
            client_kwargs={"timeout": 10},
        )

        hub.refresh(session_kwargs={"client_info": None})

        assert hub.session_kwargs == {"client_info": None}
        assert hub.headers == {"X-Old": "yes"}
        assert hub.client_kwargs == {"timeout": 10}

    def test_refresh_replaces_client_kwargs_wholesale_not_merged(self) -> None:
        hub = MCPClientHub("sse", persistent=False, endpoint="http://localhost:8000/sse")

        hub.refresh(client_kwargs={"a": 1})
        hub.refresh(client_kwargs={"b": 2})

        assert hub.client_kwargs == {"b": 2}

    def test_refresh_client_kwargs_validation_applies(self) -> None:
        hub = MCPClientHub("sse", persistent=False, endpoint="http://localhost:8000/sse")

        with pytest.raises(ValueError, match="client_kwargs must be a mapping"):
            hub.refresh(client_kwargs="bad")  # type: ignore[arg-type]

    def test_refresh_revalidates_streamable_http_collision(self) -> None:
        hub = MCPClientHub(
            "streamable_http",
            persistent=False,
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
            persistent=False,
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
    `_connect_session`/`_do_operation` to reach `operation(session)`."""

    async def initialize(self) -> None:
        return None

    async def __aenter__(self) -> "FakeRealSession":
        return self

    async def __aexit__(self, *exc_info: Any) -> bool:
        return False


class TestDoOperationErrorWrapping:
    """
    Exercises the *real* (non-`FakeHub`-overridden) `_do_operation` by
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
        return MCPClientHub("stdio", persistent=False, command="python")

    def test_typed_exception_from_operation_passes_through_unwrapped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hub = self._make_hub(monkeypatch)

        async def op(session: Any) -> Any:
            raise MCPToolError("Failed to call MCP tool 'x': boom")

        with pytest.raises(MCPToolError, match=r"^Failed to call MCP tool 'x': boom$"):
            asyncio.run(hub._do_operation(op))

    def test_unexpected_exception_from_operation_is_wrapped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        hub = self._make_hub(monkeypatch)

        async def op(session: Any) -> Any:
            raise ValueError("boom")

        with pytest.raises(MCPConnectionError, match="Failed MCP operation"):
            asyncio.run(hub._do_operation(op))


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


class TestDoOperationKwargsForwarding:
    """
    Proves client_kwargs/session_kwargs/read_timeout_seconds actually reach
    the underlying transport constructors and ClientSession, by
    monkeypatching those symbols with recording fakes (same style as
    `TestDoOperationErrorWrapping._make_hub`).
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

        hub = MCPClientHub(
            "stdio", persistent=False, command="python", client_kwargs={"cwd": "/tmp"}
        )
        asyncio.run(hub._do_operation(_identity_op))

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
            persistent=False,
            endpoint="http://localhost:8000/sse",
            client_kwargs={"timeout": 10},
        )
        asyncio.run(hub._do_operation(_identity_op))

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
            persistent=False,
            endpoint="http://localhost:8000/mcp",
            client_kwargs={"terminate_on_close": False},
        )
        asyncio.run(hub._do_operation(_identity_op))

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
            persistent=False,
            command="python",
            read_timeout_seconds=12.5,
            session_kwargs={"client_info": None},
        )
        asyncio.run(hub._do_operation(_identity_op))

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
            persistent=False,
            endpoint="http://localhost:8000/sse",
            client_kwargs={"env": {"X": "1"}},
        )

        # This failure originates in connection setup (an unexpected kwarg
        # rejected by sse_client), not in the operation call -- wrapped by
        # _connect_session's own except clause, a distinct message from
        # _do_operation's "Failed MCP operation" wrapping.
        with pytest.raises(MCPConnectionError, match="Failed to establish MCP session"):
            asyncio.run(hub._do_operation(_identity_op))


class FakeRealCallableSession(FakeSession, FakeRealSession):
    """Combines FakeSession's list_tools()/call_tool() behavior with
    FakeRealSession's async-context-manager protocol (initialize/__aenter__/
    __aexit__) -- what standing in for `ClientSession` needs to support for
    persistent-mode tests that actually drive list_tools()/call_tool()
    through a live (faked) session, not just prove kwargs forwarding."""


class RecordingSessionFactory:
    """Builds a fresh FakeRealCallableSession per call and records how many
    times it was invoked -- proves session reuse (or lack of it) without
    needing a real transport/subprocess."""

    def __init__(self, session_cls: type = None) -> None:  # type: ignore[assignment]
        self.session_cls = session_cls or FakeRealCallableSession
        self.call_count = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return self.session_cls()


def _patch_stdio_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_factory: RecordingSessionFactory | None = None,
) -> RecordingSessionFactory:
    factory = session_factory or RecordingSessionFactory()
    monkeypatch.setattr(
        client_hub_module,
        "stdio_client",
        lambda server_params: FakeAsyncCM(("read", "write")),
    )
    monkeypatch.setattr(client_hub_module, "ClientSession", factory)
    return factory


class TestMCPClientHubPersistentConnect:
    def test_persistent_construction_connects_eagerly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = _patch_stdio_fakes(monkeypatch)

        hub = MCPClientHub("stdio", persistent=True, command="python")
        try:
            assert factory.call_count == 1
            assert hub.is_connected is True
        finally:
            hub.close()

    def test_persistent_session_is_reused_across_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = _patch_stdio_fakes(monkeypatch)

        hub = MCPClientHub("stdio", persistent=True, command="python")
        try:
            hub.list_tools()
            hub.call_tool("search", {"query": "hi"})
            hub.list_tools()

            # One connect at construction, zero more across three calls.
            assert factory.call_count == 1
        finally:
            hub.close()

    def test_non_persistent_reconnects_every_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = _patch_stdio_fakes(monkeypatch)

        hub = MCPClientHub("stdio", persistent=False, command="python")

        hub.list_tools()
        hub.list_tools()

        assert factory.call_count == 2

    def test_persistent_construction_failure_leaves_no_running_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def failing_stdio_client(server_params: Any) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(client_hub_module, "stdio_client", failing_stdio_client)

        before = threading.active_count()
        with pytest.raises(MCPConnectionError):
            MCPClientHub("stdio", persistent=True, command="python")
        after = threading.active_count()

        assert after == before


class TestMCPClientHubClose:
    def test_close_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_stdio_fakes(monkeypatch)
        hub = MCPClientHub("stdio", persistent=True, command="python")

        hub.close()
        assert hub.is_connected is False
        hub.close()  # second call: safe no-op
        assert hub.is_connected is False

    def test_close_on_non_persistent_hub_is_a_no_op(self) -> None:
        hub = MCPClientHub("stdio", persistent=False, command="python")

        hub.close()  # must not raise -- nothing was ever opened

        assert hub.is_connected is False

    def test_use_after_close_falls_back_to_fresh_per_call_connect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = _patch_stdio_fakes(monkeypatch)
        hub = MCPClientHub("stdio", persistent=True, command="python")
        hub.close()
        assert factory.call_count == 1

        result = hub.list_tools()

        assert list(result) == ["search"]
        assert factory.call_count == 2  # a fresh, one-off connect, not a raise
        assert hub.is_connected is False  # still not held open afterward

    def test_context_manager_closes_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_stdio_fakes(monkeypatch)

        with MCPClientHub("stdio", persistent=True, command="python") as hub:
            assert hub.is_connected is True

        assert hub.is_connected is False

    def test_atexit_registered_on_persistent_construction_and_unregistered_on_close(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_stdio_fakes(monkeypatch)
        registered: list[Any] = []
        unregistered: list[Any] = []
        monkeypatch.setattr(client_hub_module.atexit, "register", registered.append)
        monkeypatch.setattr(client_hub_module.atexit, "unregister", unregistered.append)

        hub = MCPClientHub("stdio", persistent=True, command="python")
        assert registered == [hub.close]

        hub.close()
        assert unregistered == [hub.close]


class TestMCPClientHubPersistentRefresh:
    def test_refresh_on_persistent_sse_reconnects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = RecordingSessionFactory()
        monkeypatch.setattr(
            client_hub_module,
            "sse_client",
            lambda url, headers=None, **kw: FakeAsyncCM(("read", "write")),
        )
        monkeypatch.setattr(client_hub_module, "ClientSession", factory)

        hub = MCPClientHub("sse", persistent=True, endpoint="http://localhost:8000/sse")
        try:
            assert factory.call_count == 1

            hub.refresh(headers={"X-New": "yes"})

            assert factory.call_count == 2
            assert hub.headers == {"X-New": "yes"}
            assert hub.is_connected is True
        finally:
            hub.close()

    def test_refresh_on_persistent_stdio_warns_and_respawns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        factory = _patch_stdio_fakes(monkeypatch)
        hub = MCPClientHub("stdio", persistent=True, command="python")
        try:
            assert factory.call_count == 1

            with caplog.at_level("WARNING"):
                hub.refresh(client_kwargs={"cwd": "/tmp"})

            assert factory.call_count == 2
            assert hub.client_kwargs == {"cwd": "/tmp"}
            assert any(
                "restarts the underlying server process" in record.message
                for record in caplog.records
            )
        finally:
            hub.close()

    def test_refresh_failure_on_persistent_hub_keeps_old_session_alive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempt = {"n": 0}

        def flaky_stdio_client(server_params: Any) -> Any:
            attempt["n"] += 1
            if attempt["n"] > 1:
                raise RuntimeError("second connect fails")
            return FakeAsyncCM(("read", "write"))

        monkeypatch.setattr(client_hub_module, "stdio_client", flaky_stdio_client)
        monkeypatch.setattr(
            client_hub_module, "ClientSession", lambda *a, **k: FakeRealSession()
        )

        hub = MCPClientHub("stdio", persistent=True, command="python")
        try:
            with pytest.raises(MCPConnectionError):
                hub.refresh(client_kwargs={"cwd": "/tmp"})

            # Rolled back: config unchanged, hub still connected on the
            # original session.
            assert hub.client_kwargs is None
            assert hub.is_connected is True
        finally:
            hub.close()


class SlowFakeSession(FakeRealCallableSession):
    """Yields control mid-call so concurrent dispatches onto the shared
    background loop actually get a chance to interleave."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        await asyncio.sleep(0.01)
        return await super().call_tool(name, arguments)


class TestMCPClientHubConcurrency:
    def test_concurrent_calls_against_one_persistent_hub_all_succeed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = SlowFakeSession()
        monkeypatch.setattr(
            client_hub_module,
            "stdio_client",
            lambda server_params: FakeAsyncCM(("read", "write")),
        )
        monkeypatch.setattr(client_hub_module, "ClientSession", lambda *a, **k: session)

        hub = MCPClientHub("stdio", persistent=True, command="python")
        results: list[Any] = []
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            try:
                result = hub.call_tool("search", {"query": f"q{i}"})
                with lock:
                    results.append(result)
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        try:
            threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            assert errors == []
            assert len(results) == 5
            assert len(session.called_tools) == 5
        finally:
            hub.close()


class TestMCPClientHubToDictPersistence:
    def test_to_dict_reflects_persistence_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_stdio_fakes(monkeypatch)
        hub = MCPClientHub("stdio", persistent=True, command="python")
        try:
            data = hub.to_dict()
            assert data["persistent"] is True
            assert data["is_connected"] is True
        finally:
            hub.close()

        assert hub.to_dict()["is_connected"] is False


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific subprocess/event-loop check")
class TestWindowsBackgroundLoopSubprocessSupport:
    """
    Regression guard, not a fix: confirms start_background_loop()'s loop
    actually supports subprocess spawning on Windows (the concrete gap this
    pass investigated -- see the spec's Validation notes). If a future
    dependency ever silently overrides the ambient asyncio event-loop
    policy, this test is what would catch it.
    """

    def test_background_loop_supports_anyio_subprocess(self) -> None:
        import anyio

        loop, thread = start_background_loop()
        try:

            async def spawn() -> bytes:
                process = await anyio.open_process([sys.executable, "--version"])
                try:
                    return await process.stdout.receive()
                finally:
                    await process.wait()

            future = asyncio.run_coroutine_threadsafe(spawn(), loop)
            output = future.result(timeout=15)
            assert b"Python" in output
        finally:
            stop_background_loop(loop, thread)
