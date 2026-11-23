from __future__ import annotations

import asyncio
from typing import Any, Mapping

import pytest

from atomic_agentic.exceptions import ToolDefinitionError, ToolInvocationError
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results.tools import MCPToolResult
from atomic_agentic.mcp.mcp_client_hub import MCPClientHub
from atomic_agentic.tools.mcp import MCPProxyTool
from atomic_agentic.constants.core import NO_VAL

def param(
    name: str,
    index: int,
    *,
    kind: str = ParamSpec.KEYWORD_ONLY,
    type_: str = "Any",
    default: Any = None,
) -> ParamSpec:
    kwargs: dict[str, Any] = {
        "name": name,
        "index": index,
        "kind": kind,
        "type": type_,
    }
    if default is not None:
        kwargs["default"] = default
    return ParamSpec(**kwargs)


def search_metadata(
    *,
    description: str = "Remote search tool.",
    return_type: str = "str",
    extraction_mode: str = "extract_result",
    raw_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "description": description,
        "parameters": [
            param("query", 0, type_="str"),
            param("top_k", 1, type_="int", default=5),
        ],
        "return_type": return_type,
        "extraction_mode": extraction_mode,
        "raw_metadata": dict(raw_metadata or {"title": "Search"}),
    }


class FakeMCPClientHub(MCPClientHub):
    """Fake MCP hub that satisfies isinstance checks without opening transports."""

    def __init__(
        self,
        *,
        tools: dict[str, dict[str, Any]] | None = None,
        result: Any | None = None,
        async_result: Any | None = None,
        async_error: Exception | None = None,
        error: Exception | None = None,
        hub_name: str | None = None,
    ) -> None:
        self._hub_name = hub_name
        self._tools = {"search": search_metadata()} if tools is None else tools
        self.result = (
            result
            if result is not None
            else {
                "structuredContent": {"result": "remote result"},
                "content": ["content block"],
                "isError": False,
            }
        )
        self.async_result = async_result
        self.async_error = async_error
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.async_calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def transport_mode(self) -> str:
        return "stdio"

    @property
    def location(self) -> str:
        return "python"

    @property
    def name(self) -> str | None:
        return self._hub_name

    def list_tools(self) -> dict[str, dict[str, Any]]:
        return dict(self._tools)

    def call_tool(self, remote_name: str, inputs: Mapping[str, Any]) -> Any:
        self.calls.append((remote_name, dict(inputs)))
        if self.error is not None:
            raise self.error
        return self.result

    async def async_call_tool(self, remote_name: str, inputs: Mapping[str, Any]) -> Any:
        self.async_calls.append((remote_name, dict(inputs)))
        if self.async_error is not None:
            raise self.async_error
        return self.async_result if self.async_result is not None else self.result

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport_mode": self.transport_mode,
            "location": self.location,
        }


def make_tool(
    *,
    hub: FakeMCPClientHub | None = None,
    remote_name: str = "search",
) -> MCPProxyTool:
    return MCPProxyTool(hub or FakeMCPClientHub(), remote_name)


class TestMCPProxyToolConstruction:
    def test_valid_construction_with_existing_hub(self) -> None:
        hub = FakeMCPClientHub()
        tool = make_tool(hub=hub)

        assert tool.client_hub is hub
        assert tool.remote_name == "search"
        assert tool.name == "search"
        assert tool.description == "Remote search tool."
        assert tool.full_name == "MCPProxyTool.search"

    def test_remote_name_is_stripped(self) -> None:
        tool = make_tool(remote_name="  search  ")

        assert tool.remote_name == "search"
        assert tool.name == "search"

    def test_description_is_replaceable_through_setter(self) -> None:
        tool = make_tool()

        tool.description = "Local description."

        assert tool.description == "Local description."

    def test_description_falls_back_to_stub_when_remote_description_missing(self) -> None:
        hub = FakeMCPClientHub(
            tools={
                "search": search_metadata(description=""),
            }
        )

        tool = make_tool(hub=hub)

        assert tool.description == "MCP proxy tool 'search'"

    def test_missing_remote_name_raises(self) -> None:
        with pytest.raises(ToolDefinitionError, match="name"):
            make_tool(remote_name="   ")

    def test_non_hub_client_hub_raises(self) -> None:
        with pytest.raises(TypeError, match="client_hub"):
            MCPProxyTool(object(), "search")  # type: ignore[arg-type]

    def test_constructor_takes_only_hub_and_name(self) -> None:
        hub = FakeMCPClientHub()

        with pytest.raises(TypeError):
            MCPProxyTool(hub, "search", "local_search")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            MCPProxyTool(hub, "search", description="x")  # type: ignore[call-arg]

    def test_missing_remote_tool_raises(self) -> None:
        hub = FakeMCPClientHub(tools={})

        with pytest.raises(ToolDefinitionError, match="not found"):
            make_tool(hub=hub)


class TestMCPProxyToolSignatureAndMetadata:
    def test_parameters_and_return_type_come_from_mcp_metadata(self) -> None:
        tool = make_tool()

        assert [(p.name, p.kind, p.type, p.default) for p in tool.parameters] == [
            ("query", ParamSpec.KEYWORD_ONLY, ("str",), NO_VAL),
            ("top_k", ParamSpec.KEYWORD_ONLY, ("int",), 5),
        ]
        assert tool.return_type == ("str",)

    def test_mcpdata_property_returns_copy(self) -> None:
        tool = make_tool()

        snapshot = tool.mcpdata
        snapshot["description"] = "mutated"

        assert tool.mcpdata["description"] == "Remote search tool."

    def test_raw_metadata_returns_copy(self) -> None:
        tool = make_tool()

        raw = tool.raw_metadata
        raw["title"] = "Mutated"

        assert tool.raw_metadata == {"title": "Search"}

    def test_raw_metadata_returns_empty_dict_when_missing_or_not_mapping(self) -> None:
        hub = FakeMCPClientHub(
            tools={
                "search": {
                    **search_metadata(),
                    "raw_metadata": ["not", "mapping"],
                }
            }
        )
        tool = make_tool(hub=hub)

        assert tool.raw_metadata == {}

    def test_invalid_metadata_parameters_raises(self) -> None:
        hub = FakeMCPClientHub(
            tools={
                "search": {
                    **search_metadata(),
                    "parameters": [{"not": "ParamSpec"}],
                }
            }
        )

        with pytest.raises(TypeError, match="list\\[ParamSpec\\]"):
            make_tool(hub=hub)

    def test_invalid_metadata_return_type_raises(self) -> None:
        hub = FakeMCPClientHub(
            tools={
                "search": {
                    **search_metadata(),
                    "return_type": 123,
                }
            }
        )

        with pytest.raises(TypeError, match="return_type"):
            make_tool(hub=hub)


class TestMCPProxyToolInvocation:
    def test_extract_result_mode_returns_structured_result(self) -> None:
        hub = FakeMCPClientHub(
            result={
                "structuredContent": {"result": "answer"},
                "content": ["ignored"],
                "isError": False,
            }
        )
        tool = make_tool(hub=hub)

        assert tool.invoke({"query": "hello"}).result == "answer"
        assert hub.calls == [("search", {"query": "hello", "top_k": 5})]

    def test_structured_content_mode_returns_structured_content(self) -> None:
        hub = FakeMCPClientHub(
            tools={"search": search_metadata(extraction_mode="structured_content")},
            result={
                "structuredContent": {"answer": 42},
                "isError": False,
            },
        )
        tool = make_tool(hub=hub)

        assert tool.invoke({"query": "hello"}).result == {"answer": 42}

    def test_content_blocks_mode_returns_content(self) -> None:
        hub = FakeMCPClientHub(
            tools={"search": search_metadata(extraction_mode="content_blocks")},
            result={
                "content": [{"type": "text", "text": "hello"}],
                "isError": False,
            },
        )
        tool = make_tool(hub=hub)

        assert tool.invoke({"query": "hello"}).result == [{"type": "text", "text": "hello"}]

    def test_extract_result_requires_structured_content(self) -> None:
        hub = FakeMCPClientHub(result={"content": [], "isError": False})
        tool = make_tool(hub=hub)

        with pytest.raises(ToolInvocationError, match="structuredContent is required"):
            tool.invoke({"query": "hello"})

    def test_extract_result_requires_mapping_structured_content(self) -> None:
        hub = FakeMCPClientHub(
            result={"structuredContent": ["bad"], "isError": False}
        )
        tool = make_tool(hub=hub)

        with pytest.raises(ToolInvocationError, match="must be mapping-like"):
            tool.invoke({"query": "hello"})

    def test_extract_result_requires_result_key(self) -> None:
        hub = FakeMCPClientHub(
            result={"structuredContent": {"answer": 42}, "isError": False}
        )
        tool = make_tool(hub=hub)

        with pytest.raises(ToolInvocationError, match="missing required 'result'"):
            tool.invoke({"query": "hello"})

    def test_structured_content_mode_requires_structured_content(self) -> None:
        hub = FakeMCPClientHub(
            tools={"search": search_metadata(extraction_mode="structured_content")},
            result={"content": [], "isError": False},
        )
        tool = make_tool(hub=hub)

        with pytest.raises(ToolInvocationError, match="structuredContent is required"):
            tool.invoke({"query": "hello"})

    def test_content_blocks_mode_requires_content(self) -> None:
        hub = FakeMCPClientHub(
            tools={"search": search_metadata(extraction_mode="content_blocks")},
            result={"structuredContent": {}, "isError": False},
        )
        tool = make_tool(hub=hub)

        with pytest.raises(ToolInvocationError, match="missing 'content'"):
            tool.invoke({"query": "hello"})

    def test_invalid_extraction_mode_raises(self) -> None:
        hub = FakeMCPClientHub(
            tools={"search": search_metadata(extraction_mode="bad_mode")}
        )
        tool = make_tool(hub=hub)

        with pytest.raises(ToolInvocationError, match="invalid MCP extraction_mode"):
            tool.invoke({"query": "hello"})

    def test_remote_is_error_true_raises_with_structured_payload(self) -> None:
        hub = FakeMCPClientHub(
            result={
                "structuredContent": {"error": "boom"},
                "content": ["ignored"],
                "isError": True,
            }
        )
        tool = make_tool(hub=hub)

        with pytest.raises(ToolInvocationError, match="isError=True"):
            tool.invoke({"query": "hello"})

    def test_non_mapping_result_envelope_raises(self) -> None:
        hub = FakeMCPClientHub(result=["bad"])
        tool = make_tool(hub=hub)

        with pytest.raises(ToolInvocationError, match="non-mapping result envelope"):
            tool.invoke({"query": "hello"})

    def test_execute_wraps_non_tool_invocation_errors(self) -> None:
        hub = FakeMCPClientHub(error=RuntimeError("transport boom"))
        tool = make_tool(hub=hub)

        with pytest.raises(ToolInvocationError, match="invocation failed"):
            tool.invoke({"query": "hello"})

    def test_execute_preserves_tool_invocation_error(self) -> None:
        hub = FakeMCPClientHub(error=ToolInvocationError("known failure"))
        tool = make_tool(hub=hub)

        with pytest.raises(ToolInvocationError, match="known failure"):
            tool.invoke({"query": "hello"})

    def test_execute_rejects_positional_args(self) -> None:
        tool = make_tool()

        with pytest.raises(ToolInvocationError, match="do not accept positional"):
            tool.execute((1,), {})

    def test_to_arg_kwarg_returns_empty_args_and_copy_of_kwargs(self) -> None:
        tool = make_tool()
        inputs = {"query": "hello"}

        args, kwargs = tool.to_arg_kwarg(inputs)

        assert args == ()
        assert kwargs == {"query": "hello"}
        assert kwargs is not inputs


class TestMCPProxyToolAsync:
    def test_async_invoke_calls_fake_async_call_tool_and_extracts_result(self) -> None:
        hub = FakeMCPClientHub(
            async_result={
                "structuredContent": {"result": "async answer"},
                "isError": False,
            }
        )
        tool = make_tool(hub=hub)

        result = asyncio.run(tool.async_invoke({"query": "hello"}))

        assert result.result == "async answer"
        assert hub.async_calls == [("search", {"query": "hello", "top_k": 5})]

    def test_async_execute_rejects_positional_args(self) -> None:
        tool = make_tool()

        with pytest.raises(ToolInvocationError, match="do not accept positional"):
            asyncio.run(tool.async_execute((1,), {}))

    def test_async_execute_wraps_non_tool_invocation_errors(self) -> None:
        hub = FakeMCPClientHub(async_error=RuntimeError("transport boom"))
        tool = make_tool(hub=hub)

        with pytest.raises(ToolInvocationError, match="async MCP invocation failed"):
            asyncio.run(tool.async_invoke({"query": "hello"}))

    def test_async_execute_preserves_tool_invocation_error(self) -> None:
        hub = FakeMCPClientHub(async_error=ToolInvocationError("known failure"))
        tool = make_tool(hub=hub)

        with pytest.raises(ToolInvocationError, match="known failure"):
            asyncio.run(tool.async_invoke({"query": "hello"}))


class TestMCPProxyToolMakeResult:
    def test_invoke_returns_mcp_tool_result(self) -> None:
        tool = make_tool()

        result = tool.invoke({"query": "hello"})

        assert isinstance(result, MCPToolResult)

    def test_result_carries_transport_mode_from_hub(self) -> None:
        tool = make_tool()

        result = tool.invoke({"query": "hello"})

        assert result.transport_mode == "stdio"

    def test_result_carries_remote_name(self) -> None:
        tool = make_tool()

        result = tool.invoke({"query": "hello"})

        assert result.remote_name == "search"

    def test_result_carries_location(self) -> None:
        tool = make_tool()

        result = tool.invoke({"query": "hello"})

        assert result.location == "python"

    def test_to_dict_includes_transport_fields(self) -> None:
        tool = make_tool()

        data = tool.invoke({"query": "hello"}).to_dict()

        assert data["transport_mode"] == "stdio"
        assert data["remote_name"] == "search"
        assert data["location"] == "python"
        assert "endpoint" not in data
        assert "command" not in data

    def test_async_invoke_also_returns_mcp_tool_result(self) -> None:
        hub = FakeMCPClientHub(
            async_result={
                "structuredContent": {"result": "async answer"},
                "isError": False,
            }
        )
        tool = make_tool(hub=hub)

        result = asyncio.run(tool.async_invoke({"query": "hello"}))

        assert isinstance(result, MCPToolResult)
        assert result.transport_mode == "stdio"
        assert result.remote_name == "search"


class TestMCPProxyToolResultHook:
    def test_execute_returns_payload_with_transport_fields(self) -> None:
        tool = make_tool()

        payload = tool.execute((), {"query": "hello"})

        assert set(payload) == {"transport_mode", "remote_name", "location", "result"}
        assert payload["transport_mode"] == "stdio"
        assert payload["remote_name"] == tool.remote_name
        assert payload["location"] == "python"

    def test_async_execute_returns_payload_with_transport_fields(self) -> None:
        hub = FakeMCPClientHub(
            async_result={"structuredContent": {"result": "x"}, "isError": False}
        )
        tool = make_tool(hub=hub)

        payload = asyncio.run(tool.async_execute((), {"query": "hello"}))

        assert payload == {
            "transport_mode": "stdio",
            "remote_name": tool.remote_name,
            "location": "python",
            "result": "x",
        }

    def test_invoke_populates_transport_fields(self) -> None:
        tool = make_tool()

        result = tool.invoke({"query": "hello"})

        assert type(result) is MCPToolResult
        assert result.transport_mode == "stdio"
        assert result.remote_name == tool.remote_name
        assert result.location == "python"

    def test_async_invoke_populates_transport_fields(self) -> None:
        hub = FakeMCPClientHub(
            async_result={"structuredContent": {"result": "x"}, "isError": False}
        )
        tool = make_tool(hub=hub)

        result = asyncio.run(tool.async_invoke({"query": "hello"}))

        assert type(result) is MCPToolResult
        assert result.transport_mode == "stdio"
        assert result.remote_name == tool.remote_name

    def test_no_make_result_and_no_hook_overrides(self) -> None:
        tool = make_tool()

        assert not hasattr(tool, "make_result")
        assert "_invoke_impl" not in MCPProxyTool.__dict__
        assert "_async_invoke_impl" not in MCPProxyTool.__dict__


class TestNaming:
    def test_named_hub_prefixes_tool_name(self) -> None:
        tool = make_tool(hub=FakeMCPClientHub(hub_name="files"))

        assert tool.name == "files_search"
        assert tool.remote_name == "search"

    def test_unnamed_hub_uses_raw_name(self) -> None:
        tool = make_tool(hub=FakeMCPClientHub(hub_name=None))

        assert tool.name == "search"

    def test_call_path_uses_raw_remote_name(self) -> None:
        hub = FakeMCPClientHub(hub_name="files")
        tool = make_tool(hub=hub)

        tool.invoke({"query": "hello"})

        assert hub.calls[0][0] == "search"

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ToolDefinitionError):
            make_tool(remote_name="missing")

    def test_blank_name_raises(self) -> None:
        with pytest.raises(ToolDefinitionError):
            make_tool(remote_name="  ")

    def test_description_remote_fallback_and_setter(self) -> None:
        assert make_tool().description == "Remote search tool."

        hub = FakeMCPClientHub(
            hub_name="files", tools={"search": search_metadata(description="")}
        )
        tool = make_tool(hub=hub)
        assert tool.description == "MCP proxy tool 'files_search'"

        tool.description = "Replaced."
        assert tool.description == "Replaced."

    def test_result_validates_payload_fields(self) -> None:
        result = make_tool().invoke({"query": "hello"})

        assert isinstance(result, MCPToolResult)
        assert (result.transport_mode, result.remote_name, result.location) == (
            "stdio",
            "search",
            "python",
        )

    def test_passthrough_properties_are_gone(self) -> None:
        tool = make_tool()

        for attr in ("endpoint", "command", "args", "headers", "transport_mode"):
            assert not hasattr(tool, attr)


class TestMCPProxyToolHeadersAndSerialization:
    def test_to_dict_includes_remote_and_client_metadata(self) -> None:
        tool = make_tool()

        data = tool.to_dict()

        assert data["type"] == "MCPProxyTool"
        assert data["name"] == "search"
        assert data["remote_name"] == "search"
        assert data["raw_metadata"] == {"title": "Search"}
        assert data["client_hub"]["transport_mode"] == "stdio"
