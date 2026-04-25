from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Awaitable, Callable

import pytest

from atomic_agentic.mcp.MCPClientHub import MCPClientHub


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
        "headers",
        [
            "bad",
            {"ok": 1},
            {1: "ok"},
        ],
    )
    def test_invalid_headers_raise(self, headers: object) -> None:
        with pytest.raises(ValueError, match="headers"):
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

        with pytest.raises(RuntimeError, match="Unexpected transport stream shape"):
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

        with pytest.raises(RuntimeError, match="remote_name"):
            hub.call_tool("   ", {"query": "hello"})

    def test_call_tool_rejects_non_mapping_inputs(self) -> None:
        hub = FakeHub(FakeSession(), transport_mode="stdio", command="python")

        with pytest.raises(RuntimeError, match="inputs"):
            hub.call_tool("search", ["not", "mapping"])  # type: ignore[arg-type]
