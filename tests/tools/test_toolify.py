from __future__ import annotations

from typing import Any, Mapping

import pytest

from atomic_agentic.a2a.A2AClientHub import A2AClientHub
from atomic_agentic.a2a.PyA2AtomicClient import PyA2AtomicClient
from atomic_agentic.constants.python_a2a import PYA2A_RESULT_KEY
from atomic_agentic.exceptions import ToolDefinitionError
from atomic_agentic.core.Invokable import AtomicInvokable
from atomic_agentic.models.a2a_sdk import A2AtomicSkillMetadata
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.mcp.MCPClientHub import MCPClientHub
from atomic_agentic.tools.Toolify import batch_toolify, toolify
from atomic_agentic.tools.a2a_sdk import A2AProxyTool
from atomic_agentic.tools.python_a2a import PyA2AtomicTool
from atomic_agentic.tools.base import Tool
from atomic_agentic.tools.mcp import MCPProxyTool
from atomic_agentic.agents.tools import return_tool


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
        namespace: str = "default",
        description: str = "Echo invokable.",
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            namespace=namespace,
            parameters=[
                make_param("value", 0, type_="Any"),
            ],
            return_type="dict[str, Any]",
        )

    def invoke(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return dict(inputs)

    async def async_invoke(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return dict(inputs)


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

    async def async_call_tool(self, remote_name: str, inputs: Mapping[str, Any]) -> Any:
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

    def call_invokable(self, remote_name: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((remote_name, dict(inputs)))
        return {PYA2A_RESULT_KEY: self.result}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "FakePyA2AtomicClient",
            "url": self.url,
            "has_headers": self.headers is not None,
            "agent_name": self.agent_card.name,
        }


def a2a_sdk_skill_metadata(
    *,
    remote_name: str = "add",
    description: str = "Remote A2A-sdk skill.",
) -> A2AtomicSkillMetadata:
    return A2AtomicSkillMetadata(
        remote_name=remote_name,
        description=description,
        extra_description="",
        params=(
            make_param("a", 0, type_="int"),
            make_param("b", 1, type_="int"),
        ),
        return_type="int",
    )


class FakeA2AClientHub(A2AClientHub):
    def __init__(
        self,
        *,
        skills: dict[str, A2AtomicSkillMetadata] | None = None,
        card_name: str = "Fake A2A-sdk Agent",
        card_description: str = "Fake A2A-sdk agent.",
        result: Any | None = None,
    ) -> None:
        self._skills = (
            {"add": a2a_sdk_skill_metadata(remote_name="add")}
            if skills is None
            else skills
        )
        self._card = type(
            "FakeAgentCard",
            (),
            {"name": card_name, "description": card_description},
        )()
        self.result = result if result is not None else 42
        self.skill_calls: list[tuple[str, dict[str, Any]]] = []

    @property
    def agent_card(self) -> Any:
        return self._card

    @property
    def transport_mode(self) -> str:
        return "JSONRPC"

    @property
    def base_url(self) -> str:
        return "http://example.test/a2a-sdk"

    @property
    def persistent(self) -> bool:
        return False

    def get_atomic_skills(self) -> dict[str, A2AtomicSkillMetadata]:
        return dict(self._skills)

    def call_atomic_skill(self, skill_id: str, inputs: Mapping[str, Any]) -> Any:
        self.skill_calls.append((skill_id, dict(inputs)))
        return self.result

    async def async_call_atomic_skill(self, skill_id: str, inputs: Mapping[str, Any]) -> Any:
        return self.call_atomic_skill(skill_id, inputs)


class TestToolifyCallable:
    def test_toolify_callable_returns_tool(self) -> None:
        tool = toolify(add, namespace="tests", description="Add values.")

        assert isinstance(tool, Tool)
        assert type(tool) is Tool
        assert tool.name == "add"
        assert tool.namespace == "tests"
        assert tool.description == "Add values."
        assert tool.full_name == "Tool.tests.add"
        assert tool.invoke({"a": 2, "b": 3}).result == 5

    def test_toolify_callable_uses_inferred_name_when_name_missing(self) -> None:
        tool = toolify(add, namespace="tests", description="Add values.")

        assert tool.name == "add"

    def test_toolify_callable_applies_overrides(self) -> None:
        tool = toolify(
            add,
            name="sum_values",
            namespace="math",
            description="Sum values.",
        )

        assert tool.name == "sum_values"
        assert tool.namespace == "math"
        assert tool.description == "Sum values."
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

    def test_toolify_callable_rejects_remote_name(self) -> None:
        with pytest.raises(ToolDefinitionError, match="remote_name"):
            toolify(add, remote_name="remote_add")


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

    def test_existing_tool_with_overrides_returns_wrapper_without_mutating_original(self) -> None:
        original = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        result = toolify(
            original,
            name="sum_values",
            namespace="math",
            description="Sum values.",
        )

        assert result is not original
        assert type(result) is Tool
        assert result.function is original

        assert result.name == "sum_values"
        assert result.namespace == "math"
        assert result.description == "Sum values."
        assert result.full_name == "Tool.math.sum_values"

        assert original.name == "add"
        assert original.namespace == "tests"
        assert original.description == "Add values."
        assert original.full_name == "Tool.tests.add"

        assert result.invoke({"a": 2, "b": 3}).result == 5

    def test_existing_tool_without_overrides_preserves_metadata(self) -> None:
        original = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        result = toolify(original)

        assert result is original
        assert original.name == "add"
        assert original.namespace == "tests"
        assert original.description == "Add values."

    def test_existing_tool_wrapper_preserves_omitted_metadata(self) -> None:
        original = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        result = toolify(original, name="sum_values")

        assert result is not original
        assert result.function is original
        assert result.name == "sum_values"
        assert result.namespace == "tests"
        assert result.description == "Add values."

        assert original.name == "add"
        assert original.namespace == "tests"
        assert original.description == "Add values."

    def test_existing_tool_rejects_remote_name(self) -> None:
        original = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        with pytest.raises(ToolDefinitionError, match="remote_name"):
            toolify(original, remote_name="remote_add")


class TestToolifyAtomicInvokableWrapping:
    def test_non_tool_atomic_invokable_wraps_by_reference(self) -> None:
        invokable = EchoInvokable()

        tool = toolify(invokable)

        assert type(tool) is Tool
        assert tool.function is invokable
        assert tool.name == "echo_invokable"
        assert tool.description == "Echo invokable."
        assert tool.invoke({"value": 123}).result == {"value": 123}


class TestToolifyReturnToolIdentity:
    def test_toolify_return_tool_with_override_delegates_by_reference(self) -> None:
        result = toolify(return_tool, name="alias")

        assert result is not return_tool
        assert result.function is return_tool
        assert result.name == "alias"


class TestToolifyAtomicInvokableNamespace:
    def test_route1_existing_tool_returned_unchanged(self) -> None:
        t = Tool(function=add, name="add", namespace="ns", description="d")
        result = toolify(t)
        assert result is t


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
        )

        assert isinstance(tool, MCPProxyTool)
        assert tool.name == "local_search"
        assert tool.namespace == "remote"
        assert tool.description == "Local search."

    def test_mcp_proxy_tool_invokes_fake_hub(self) -> None:
        hub = FakeMCPClientHub(result={"structuredContent": {"result": "ok"}})
        tool = toolify(hub, remote_name="search")

        assert tool.invoke({"query": "hello"}).result == "ok"
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
        )

        assert isinstance(tool, PyA2AtomicTool)
        assert tool.name == "local_echo"
        assert tool.namespace == "a2a_tools"
        assert tool.description == "Local echo."

    def test_a2a_tool_invokes_fake_client(self) -> None:
        client = FakePyA2AtomicClient(result={"ok": True})
        tool = toolify(client, remote_name="echo")

        assert tool.invoke({"value": 123}).result == {"ok": True}
        assert client.calls == [("echo", {"value": 123})]


class TestToolifyA2AClientHub:
    def test_remote_name_present_selects_skill_mode(self) -> None:
        hub = FakeA2AClientHub()

        tool = toolify(hub, remote_name="add")

        assert isinstance(tool, A2AProxyTool)
        assert tool.skill_id == "add"
        assert tool.name == "add"

    def test_remote_name_absent_selects_generic_mode_not_an_error(self) -> None:
        hub = FakeA2AClientHub()

        tool = toolify(hub)

        assert isinstance(tool, A2AProxyTool)
        assert tool.skill_id is None
        assert tool.name == "send_parts"

    def test_remote_name_blank_selects_generic_mode(self) -> None:
        hub = FakeA2AClientHub()

        tool = toolify(hub, remote_name="   ")

        assert tool.skill_id is None

    def test_unknown_skill_id_raises_tool_definition_error(self) -> None:
        hub = FakeA2AClientHub()

        with pytest.raises(ToolDefinitionError, match="skill_id"):
            toolify(hub, remote_name="does_not_exist")

    def test_applies_overrides(self) -> None:
        hub = FakeA2AClientHub()

        tool = toolify(
            hub,
            remote_name="add",
            name="local_add",
            namespace="a2a_tools",
            description="Local add.",
        )

        assert isinstance(tool, A2AProxyTool)
        assert tool.name == "local_add"
        assert tool.namespace == "a2a_tools"
        assert tool.description == "Local add."

    def test_namespace_falls_back_to_sanitized_card_name(self) -> None:
        hub = FakeA2AClientHub(card_name="My Cool Agent")

        tool = toolify(hub, remote_name="add")

        assert tool.namespace == "My_Cool_Agent"

    def test_namespace_falls_back_to_a2a_when_card_name_unusable(self) -> None:
        # "123 Agent" sanitizes to "123_Agent" -- starts with a digit, so it
        # never becomes a valid identifier.
        hub = FakeA2AClientHub(card_name="123 Agent")

        tool = toolify(hub, remote_name="add")

        assert tool.namespace == "a2a"

    def test_explicit_namespace_wins_over_card_name(self) -> None:
        hub = FakeA2AClientHub(card_name="My Cool Agent")

        tool = toolify(hub, remote_name="add", namespace="explicit_ns")

        assert tool.namespace == "explicit_ns"

    def test_skill_mode_invokes_fake_hub(self) -> None:
        hub = FakeA2AClientHub(result=17)
        tool = toolify(hub, remote_name="add")

        assert tool.invoke({"a": 12, "b": 5}).result == 17
        assert hub.skill_calls == [("add", {"a": 12, "b": 5})]


class TestToolifyInvalidInputs:
    def test_toolify_none_raises_tool_definition_error(self) -> None:
        with pytest.raises(ToolDefinitionError, match="expected a non-empty"):
            toolify(None)  # type: ignore[arg-type]

    def test_toolify_unsupported_component_type_raises(self) -> None:
        with pytest.raises(ToolDefinitionError, match="unsupported"):
            toolify(object())  # type: ignore[arg-type]


class TestBatchToolifyLocalSources:
    def test_batch_toolify_empty_or_none_returns_empty_list(self) -> None:
        assert batch_toolify([]) == []
        assert batch_toolify(None) == []

    def test_batch_toolify_applies_batch_namespace_to_local_tools(self) -> None:
        tools = batch_toolify([add, multiply], batch_namespace="math")

        assert [tool.full_name for tool in tools] == [
            "Tool.math.add",
            "Tool.math.multiply",
        ]

    def test_batch_toolify_wraps_non_tool_atomic_invokable(self) -> None:
        invokable = EchoInvokable()

        tools = batch_toolify([add, invokable], batch_namespace="batch")

        assert len(tools) == 2
        assert type(tools[0]) is Tool
        assert type(tools[1]) is Tool
        assert tools[1].function is invokable
        assert tools[0].full_name == "Tool.batch.add"
        assert tools[1].full_name == "Tool.batch.echo_invokable"

    def test_batch_toolify_existing_tool_namespace_override_wraps_without_mutating_original(self) -> None:
        original = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        tools = batch_toolify([original], batch_namespace="batch")

        assert len(tools) == 1
        assert tools[0] is not original
        assert tools[0].function is original
        assert tools[0].full_name == "Tool.batch.add"
        assert original.full_name == "Tool.tests.add"


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

    def test_batch_toolify_remote_tools_can_invoke_fake_backends(self) -> None:
        hub = FakeMCPClientHub(result={"structuredContent": {"result": "mcp ok"}})
        client = FakePyA2AtomicClient(result={"a2a": "ok"})

        tools = batch_toolify([hub, client])
        mcp_tool = tools[0]
        a2a_tool = tools[2]

        assert mcp_tool.invoke({"query": "hello"}).result == "mcp ok"
        assert a2a_tool.invoke({"value": 123}).result == {"a2a": "ok"}

        assert hub.calls == [("search", {"query": "hello"})]
        assert client.calls == [("echo", {"value": 123})]


class TestBatchToolifyA2AClientHub:
    def test_expands_skills_plus_one_trailing_generic_tool(self) -> None:
        hub = FakeA2AClientHub(
            skills={
                "add": a2a_sdk_skill_metadata(remote_name="add"),
                "multiply": a2a_sdk_skill_metadata(remote_name="multiply"),
            }
        )

        tools = batch_toolify([hub])

        assert len(tools) == 3
        assert all(isinstance(tool, A2AProxyTool) for tool in tools)
        assert [tool.skill_id for tool in tools] == ["add", "multiply", None]
        assert tools[-1].name == "send_parts"

    def test_zero_skill_hub_still_yields_one_generic_tool(self) -> None:
        hub = FakeA2AClientHub(skills={})

        tools = batch_toolify([hub])

        assert len(tools) == 1
        assert tools[0].skill_id is None

    def test_applies_batch_namespace_to_all_produced_tools(self) -> None:
        hub = FakeA2AClientHub(
            skills={"add": a2a_sdk_skill_metadata(remote_name="add")}
        )

        tools = batch_toolify([hub], batch_namespace="batch_ns")

        assert [tool.namespace for tool in tools] == ["batch_ns", "batch_ns"]

    def test_mixed_sources_preserves_expanded_order(self) -> None:
        hub = FakeA2AClientHub(
            skills={"add": a2a_sdk_skill_metadata(remote_name="add")}
        )

        tools = batch_toolify([add, hub], batch_namespace="batch")

        assert [type(tool) for tool in tools] == [Tool, A2AProxyTool, A2AProxyTool]
        assert [tool.name for tool in tools] == ["add", "add", "send_parts"]
