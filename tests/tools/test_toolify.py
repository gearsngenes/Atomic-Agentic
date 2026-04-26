from __future__ import annotations

from typing import Any, Mapping

import pytest

from atomic_agentic.a2a.PyA2AtomicClient import PyA2AtomicClient
from atomic_agentic.core.Exceptions import ToolDefinitionError
from atomic_agentic.core.Invokable import AtomicInvokable
from atomic_agentic.core.Parameters import ParamSpec
from atomic_agentic.core.sentinels import NO_VAL
from atomic_agentic.mcp.MCPClientHub import MCPClientHub
from atomic_agentic.tools.Toolify import batch_toolify, toolify
from atomic_agentic.tools.adapter import AdapterTool
from atomic_agentic.tools.a2a import PyA2AtomicTool
from atomic_agentic.tools.base import Tool
from atomic_agentic.tools.mcp import MCPProxyTool


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def undocumented(value: str) -> str:
    return value


undocumented.__doc__ = None


class CallableWithoutName:
    def __call__(self, value: str) -> str:
        return value


class CallableWithEmptyName:
    __name__ = ""

    def __call__(self, value: str) -> str:
        return value


def make_param(
    name: str,
    index: int,
    *,
    kind: str = ParamSpec.POSITIONAL_OR_KEYWORD,
    type_: str = "Any",
    default: Any = NO_VAL,
) -> ParamSpec:
    return ParamSpec(
        name=name,
        index=index,
        kind=kind,
        type=type_,
        default=default,
    )


def make_param_dict(
    name: str,
    index: int,
    *,
    kind: str = ParamSpec.POSITIONAL_OR_KEYWORD,
    type_: str = "Any",
    default: Any = NO_VAL,
) -> dict[str, Any]:
    return make_param(
        name,
        index,
        kind=kind,
        type_=type_,
        default=default,
    ).to_dict()


class EchoInvokable(AtomicInvokable):
    def __init__(
        self,
        *,
        name: str = "echo_invokable",
        namespace: str | None = None,
        description: str = "Echo invokable.",
        filter_extraneous_inputs: bool = True,
    ) -> None:
        self.namespace = namespace
        super().__init__(
            name=name,
            description=description,
            parameters=[
                make_param("value", 0, type_="Any"),
            ],
            return_type="dict[str, Any]",
            filter_extraneous_inputs=filter_extraneous_inputs,
        )

    def invoke(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return dict(inputs)

    async def async_invoke(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return dict(inputs)

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        if self.namespace is not None:
            data["namespace"] = self.namespace
        return data


def mcp_metadata(
    *,
    description: str = "Remote MCP search.",
    extraction_mode: str = "extract_result",
) -> dict[str, Any]:
    return {
        "description": description,
        "parameters": [
            make_param("query", 0, kind=ParamSpec.KEYWORD_ONLY, type_="str"),
        ],
        "return_type": "str",
        "extraction_mode": extraction_mode,
        "raw_metadata": {"title": "Search"},
    }


class FakeMCPClientHub(MCPClientHub):
    def __init__(
        self,
        *,
        tools: dict[str, dict[str, Any]] | None = None,
        result: Any | None = None,
    ) -> None:
        self._tools = (
            {
                "search": mcp_metadata(description="Remote MCP search."),
                "summarize": mcp_metadata(description="Remote MCP summarize."),
            }
            if tools is None
            else tools
        )
        self.result = (
            result
            if result is not None
            else {
                "structuredContent": {"result": "mcp result"},
                "isError": False,
            }
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._headers: Mapping[str, str] | None = None

    @property
    def transport_mode(self) -> str:
        return "stdio"

    @property
    def endpoint(self) -> str | None:
        return None

    @property
    def command(self) -> str | None:
        return "python"

    @property
    def args(self) -> tuple[str, ...] | None:
        return ("fake_mcp_server.py",)

    @property
    def headers(self) -> Mapping[str, str] | None:
        return self._headers

    @headers.setter
    def headers(self, value: Mapping[str, str] | None) -> None:
        self._headers = value

    def list_tools(self) -> dict[str, dict[str, Any]]:
        return dict(self._tools)

    def call_tool(self, remote_name: str, inputs: Mapping[str, Any]) -> Any:
        self.calls.append((remote_name, dict(inputs)))
        return self.result

    async def _acall_tool(self, remote_name: str, inputs: Mapping[str, Any]) -> Any:
        self.calls.append((remote_name, dict(inputs)))
        return self.result

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport_mode": self.transport_mode,
            "command": self.command,
            "args": list(self.args or ()),
            "has_headers": self.headers is not None,
        }


def a2a_metadata(
    *,
    name: str = "echo",
    description: str = "Remote A2A echo.",
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": [
            make_param_dict("value", 0, type_="Any"),
        ],
        "return_type": "dict[str, Any]",
        "filter_extraneous_inputs": True,
        "invokable_type": "EchoInvokable",
    }


class FakePyA2AtomicClient(PyA2AtomicClient):
    def __init__(
        self,
        *,
        invokables: dict[str, dict[str, Any]] | None = None,
        result: Any | None = None,
    ) -> None:
        self._url = "http://example.test/a2a"
        self._headers: Mapping[str, str] | None = None
        self._agent_card = type(
            "FakeAgentCard",
            (),
            {
                "name": "fake_agent",
                "description": "Fake A2A agent.",
            },
        )()
        self._invokables = (
            {
                "echo": a2a_metadata(name="echo", description="Remote A2A echo."),
                "classify": a2a_metadata(
                    name="classify",
                    description="Remote A2A classify.",
                ),
            }
            if invokables is None
            else invokables
        )
        self.result = result if result is not None else {"a2a": True}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.metadata_calls: list[str] = []

    @property
    def url(self) -> str:
        return self._url

    @property
    def headers(self) -> Mapping[str, str] | None:
        return self._headers

    @headers.setter
    def headers(self, value: Mapping[str, str] | None) -> None:
        self._headers = value

    @property
    def agent_card(self) -> Any:
        return self._agent_card

    def list_invokables(self) -> dict[str, dict[str, Any]]:
        return dict(self._invokables)

    def get_invokable_metadata(self, remote_name: str) -> dict[str, Any]:
        self.metadata_calls.append(remote_name)
        return dict(self._invokables[remote_name])

    def call_invokable(self, remote_name: str, inputs: Mapping[str, Any]) -> Any:
        self.calls.append((remote_name, dict(inputs)))
        return self.result

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "FakePyA2AtomicClient",
            "url": self.url,
            "has_headers": self.headers is not None,
            "agent_name": self.agent_card.name,
        }


class TestToolifyCallable:
    def test_toolify_callable_returns_tool(self) -> None:
        tool = toolify(add, namespace="tests", description="Add values.")

        assert isinstance(tool, Tool)
        assert type(tool) is Tool
        assert tool.name == "add"
        assert tool.namespace == "tests"
        assert tool.description == "Add values."
        assert tool.full_name == "Tool.tests.add"
        assert tool.invoke({"a": 2, "b": 3}) == 5

    def test_toolify_callable_uses_inferred_name_when_name_missing(self) -> None:
        tool = toolify(add, namespace="tests", description="Add values.")

        assert tool.name == "add"

    def test_toolify_callable_applies_overrides(self) -> None:
        tool = toolify(
            add,
            name="sum_values",
            namespace="math",
            description="Sum values.",
            filter_extraneous_inputs=False,
        )

        assert tool.name == "sum_values"
        assert tool.namespace == "math"
        assert tool.description == "Sum values."
        assert tool.filter_extraneous_inputs is False
        assert tool.full_name == "Tool.math.sum_values"

    def test_toolify_callable_uses_docstring_description(self) -> None:
        tool = toolify(add, namespace="tests")

        assert tool.description == "Add two integers."

    def test_toolify_callable_uses_undescribed_fallback_description(self) -> None:
        tool = toolify(undocumented, namespace="tests")

        assert tool.description == "undescribed"

    def test_toolify_callable_rejects_non_string_description(self) -> None:
        with pytest.raises(ToolDefinitionError, match="description"):
            toolify(add, description=123)  # type: ignore[arg-type]

    def test_toolify_callable_requires_name_when_name_cannot_be_inferred(self) -> None:
        with pytest.raises(ToolDefinitionError, match="name"):
            toolify(CallableWithoutName())

    def test_toolify_callable_rejects_empty_resolved_name(self) -> None:
        with pytest.raises(ToolDefinitionError, match="name"):
            toolify(CallableWithEmptyName())


class TestToolifyExistingTool:
    def test_existing_tool_returns_same_instance(self) -> None:
        original = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        result = toolify(original)

        assert result is original

    def test_existing_tool_mutates_explicit_overrides_only(self) -> None:
        original = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
            filter_extraneous_inputs=True,
        )

        result = toolify(
            original,
            name="sum_values",
            namespace="math",
            description="Sum values.",
            filter_extraneous_inputs=False,
        )

        assert result is original
        assert original.name == "sum_values"
        assert original.namespace == "math"
        assert original.description == "Sum values."
        assert original.filter_extraneous_inputs is False
        assert original.full_name == "Tool.math.sum_values"

    def test_existing_tool_without_overrides_preserves_metadata(self) -> None:
        original = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
            filter_extraneous_inputs=False,
        )

        result = toolify(original)

        assert result is original
        assert original.name == "add"
        assert original.namespace == "tests"
        assert original.description == "Add values."
        assert original.filter_extraneous_inputs is False


class TestToolifyAtomicInvokable:
    def test_atomic_invokable_becomes_adapter_tool(self) -> None:
        invokable = EchoInvokable()

        tool = toolify(invokable)

        assert isinstance(tool, AdapterTool)
        assert tool.component is invokable
        assert tool.name == "echo_invokable"
        assert tool.description == "Echo invokable."
        assert tool.invoke({"value": 123}) == {"value": 123}

    def test_atomic_invokable_uses_schema_and_return_type(self) -> None:
        invokable = EchoInvokable()

        tool = toolify(invokable)

        assert tool.parameters == invokable.parameters
        assert tool.return_type == invokable.return_type

    def test_atomic_invokable_applies_overrides(self) -> None:
        invokable = EchoInvokable(filter_extraneous_inputs=False)

        tool = toolify(
            invokable,
            name="local_echo",
            namespace="wrapped",
            description="Wrapped echo.",
            filter_extraneous_inputs=True,
        )

        assert isinstance(tool, AdapterTool)
        assert tool.name == "local_echo"
        assert tool.namespace == "wrapped"
        assert tool.description == "Wrapped echo."
        assert tool.filter_extraneous_inputs is True


class TestToolifyMCPClientHub:
    def test_mcp_client_hub_requires_remote_name(self) -> None:
        with pytest.raises(ToolDefinitionError, match="remote_name"):
            toolify(FakeMCPClientHub())

    def test_mcp_client_hub_becomes_mcp_proxy_tool(self) -> None:
        hub = FakeMCPClientHub()

        tool = toolify(hub, remote_name="search")

        assert isinstance(tool, MCPProxyTool)
        assert tool.remote_name == "search"
        assert tool.name == "search"
        assert tool.namespace == "mcp"

    def test_mcp_client_hub_applies_overrides(self) -> None:
        hub = FakeMCPClientHub()

        tool = toolify(
            hub,
            remote_name="search",
            name="local_search",
            namespace="remote",
            description="Local search.",
            filter_extraneous_inputs=False,
        )

        assert isinstance(tool, MCPProxyTool)
        assert tool.name == "local_search"
        assert tool.namespace == "remote"
        assert tool.description == "Local search."
        assert tool.filter_extraneous_inputs is False

    def test_mcp_proxy_tool_invokes_fake_hub(self) -> None:
        hub = FakeMCPClientHub(result={"structuredContent": {"result": "ok"}})
        tool = toolify(hub, remote_name="search")

        assert tool.invoke({"query": "hello"}) == "ok"
        assert hub.calls == [("search", {"query": "hello"})]


class TestToolifyPyA2AtomicClient:
    def test_a2a_client_requires_remote_name(self) -> None:
        with pytest.raises(ToolDefinitionError, match="remote_name"):
            toolify(FakePyA2AtomicClient())

    def test_a2a_client_becomes_pya2atomic_tool(self) -> None:
        client = FakePyA2AtomicClient()

        tool = toolify(client, remote_name="echo")

        assert isinstance(tool, PyA2AtomicTool)
        assert tool.remote_name == "echo"
        assert tool.name == "echo"
        assert tool.namespace == "fake_agent"

    def test_a2a_client_applies_overrides(self) -> None:
        client = FakePyA2AtomicClient()

        tool = toolify(
            client,
            remote_name="echo",
            name="local_echo",
            namespace="a2a_tools",
            description="Local echo.",
            filter_extraneous_inputs=False,
        )

        assert isinstance(tool, PyA2AtomicTool)
        assert tool.name == "local_echo"
        assert tool.namespace == "a2a_tools"
        assert tool.description == "Local echo."
        assert tool.filter_extraneous_inputs is False

    def test_a2a_tool_invokes_fake_client(self) -> None:
        client = FakePyA2AtomicClient(result={"ok": True})
        tool = toolify(client, remote_name="echo")

        assert tool.invoke({"value": 123}) == {"ok": True}
        assert client.calls == [("echo", {"value": 123})]


class TestToolifyInvalidInputs:
    def test_toolify_none_raises_tool_definition_error(self) -> None:
        with pytest.raises(ToolDefinitionError, match="expected either"):
            toolify(None)  # type: ignore[arg-type]

    def test_toolify_unsupported_component_type_raises(self) -> None:
        with pytest.raises(ToolDefinitionError, match="unsupported"):
            toolify(object())  # type: ignore[arg-type]


class TestBatchToolifyLocalSources:
    def test_batch_toolify_empty_or_none_returns_empty_list(self) -> None:
        assert batch_toolify([]) == []
        assert batch_toolify(None) == []

    def test_batch_toolify_mixed_callables_and_invokable_preserves_order(self) -> None:
        invokable = EchoInvokable()

        tools = batch_toolify([add, invokable], batch_namespace="batch")

        assert len(tools) == 2
        assert type(tools[0]) is Tool
        assert isinstance(tools[1], AdapterTool)
        assert tools[0].full_name == "Tool.batch.add"
        assert tools[1].full_name == "AdapterTool.batch.echo_invokable"

    def test_batch_toolify_applies_batch_namespace_to_local_tools(self) -> None:
        tools = batch_toolify([add, multiply], batch_namespace="math")

        assert [tool.full_name for tool in tools] == [
            "Tool.math.add",
            "Tool.math.multiply",
        ]

    def test_batch_toolify_applies_batch_filter_inputs(self) -> None:
        tools = batch_toolify([add, EchoInvokable()], batch_filter_inputs=False)

        assert [tool.filter_extraneous_inputs for tool in tools] == [False, False]


class TestBatchToolifyRemoteExpansion:
    def test_batch_toolify_expands_mcp_hub_tools(self) -> None:
        hub = FakeMCPClientHub()

        tools = batch_toolify([hub])

        assert len(tools) == 2
        assert all(isinstance(tool, MCPProxyTool) for tool in tools)
        assert [tool.remote_name for tool in tools] == ["search", "summarize"]

    def test_batch_toolify_expands_a2a_client_invokables(self) -> None:
        client = FakePyA2AtomicClient()

        tools = batch_toolify([client])

        assert len(tools) == 2
        assert all(isinstance(tool, PyA2AtomicTool) for tool in tools)
        assert [tool.remote_name for tool in tools] == ["echo", "classify"]

    def test_batch_toolify_mixed_sources_preserves_expanded_order(self) -> None:
        hub = FakeMCPClientHub()
        client = FakePyA2AtomicClient()

        tools = batch_toolify([add, hub, client, multiply], batch_namespace="batch")

        assert [type(tool) for tool in tools] == [
            Tool,
            MCPProxyTool,
            MCPProxyTool,
            PyA2AtomicTool,
            PyA2AtomicTool,
            Tool,
        ]
        assert [tool.name for tool in tools] == [
            "add",
            "search",
            "summarize",
            "echo",
            "classify",
            "multiply",
        ]

    def test_batch_toolify_applies_batch_namespace_to_remote_tools(self) -> None:
        tools = batch_toolify(
            [
                FakeMCPClientHub(),
                FakePyA2AtomicClient(),
            ],
            batch_namespace="remote_batch",
        )

        assert [tool.namespace for tool in tools] == [
            "remote_batch",
            "remote_batch",
            "remote_batch",
            "remote_batch",
        ]

    def test_batch_toolify_applies_batch_filter_inputs_to_remote_tools(self) -> None:
        tools = batch_toolify(
            [
                FakeMCPClientHub(),
                FakePyA2AtomicClient(),
            ],
            batch_filter_inputs=False,
        )

        assert all(tool.filter_extraneous_inputs is False for tool in tools)

    def test_batch_toolify_remote_tools_can_invoke_fake_backends(self) -> None:
        hub = FakeMCPClientHub(result={"structuredContent": {"result": "mcp ok"}})
        client = FakePyA2AtomicClient(result={"a2a": "ok"})

        tools = batch_toolify([hub, client])
        mcp_tool = tools[0]
        a2a_tool = tools[2]

        assert mcp_tool.invoke({"query": "hello"}) == "mcp ok"
        assert a2a_tool.invoke({"value": 123}) == {"a2a": "ok"}

        assert hub.calls == [("search", {"query": "hello"})]
        assert client.calls == [("echo", {"value": 123})]
