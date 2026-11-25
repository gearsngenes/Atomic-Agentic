from __future__ import annotations

import asyncio
from typing import Any, Mapping

import pytest

from atomic_agentic.a2a.a2a_client_hub import A2AClientHub
from atomic_agentic.exceptions import A2AProxyError, MCPToolError, ToolDefinitionError
from atomic_agentic.invokables.atomic_invokable import AtomicInvokable
from atomic_agentic.models.a2a_sdk import A2AtomicSkillMetadata
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.mcp.mcp_client_hub import MCPClientHub
from atomic_agentic.adapters import to_tools, to_tool
from atomic_agentic.tools.a2a_sdk import A2AProxyTool
from atomic_agentic.tools.base import Tool
from atomic_agentic.tools.mcp import MCPProxyTool
from atomic_agentic.agents.tools import return_tool

from ..agents.conftest import make_stub_tool_agent


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


class EchoInvokable(AtomicInvokable):
    def __init__(
        self,
        *,
        name: str = "echo_invokable",
        description: str = "Echo invokable.",
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            parameters=[
                make_param("value", 0, type_="Any"),
            ],
            return_type="dict[str, Any]",
        )

    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"result": dict(inputs)}

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"result": dict(inputs)}


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
        hub_name: str | None = None,
    ) -> None:
        self._hub_name = hub_name
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
        return self.result

    async def async_call_tool(self, remote_name: str, inputs: Mapping[str, Any]) -> Any:
        self.calls.append((remote_name, dict(inputs)))
        return self.result

    def to_dict(self) -> dict[str, Any]:
        return {
            "transport_mode": self.transport_mode,
            "location": self.location,
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
        hub_name: str | None = None,
    ) -> None:
        self._hub_name = hub_name
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
    def location(self) -> str:
        return "http://example.test/a2a-sdk"

    @property
    def name(self) -> str | None:
        return self._hub_name

    @property
    def persistent(self) -> bool:
        return False

    def list_skills(self) -> dict[str, A2AtomicSkillMetadata]:
        return dict(self._skills)

    def call_atomic_skill(self, remote_name: str, inputs: Mapping[str, Any]) -> Any:
        self.skill_calls.append((remote_name, dict(inputs)))
        return self.result

    async def async_call_atomic_skill(self, remote_name: str, inputs: Mapping[str, Any]) -> Any:
        return self.call_atomic_skill(remote_name, inputs)


class TestToToolCallable:
    def test_to_tool_callable_returns_tool(self) -> None:
        tool = to_tool(add, description="Add values.")

        assert isinstance(tool, Tool)
        assert type(tool) is Tool
        assert tool.name == "add"
        assert tool.description == "Add values."
        assert tool.full_name == "Tool.add"
        assert tool.invoke({"a": 2, "b": 3}).result == 5

    def test_to_tool_callable_uses_inferred_name_when_name_missing(self) -> None:
        tool = to_tool(add, description="Add values.")

        assert tool.name == "add"

    def test_to_tool_callable_applies_overrides(self) -> None:
        tool = to_tool(
            add,
            name="sum_values",
            description="Sum values.",
        )

        assert tool.name == "sum_values"
        assert tool.description == "Sum values."
        assert tool.full_name == "Tool.sum_values"

    def test_to_tool_callable_uses_docstring_description(self) -> None:
        tool = to_tool(add)

        assert tool.description == "Add two integers."

    def test_to_tool_callable_uses_default_description_fallback(self) -> None:
        tool = to_tool(undocumented)

        assert tool.description == "No description available."

    def test_to_tool_callable_rejects_non_string_description(self) -> None:
        with pytest.raises(ToolDefinitionError, match="description"):
            to_tool(add, description=123)  # type: ignore[arg-type]

    def test_to_tool_callable_requires_name_when_name_cannot_be_inferred(self) -> None:
        with pytest.raises(ToolDefinitionError, match="name"):
            to_tool(CallableWithoutName())

    def test_to_tool_callable_rejects_empty_resolved_name(self) -> None:
        with pytest.raises(ToolDefinitionError, match="name"):
            to_tool(CallableWithEmptyName())

    def test_to_tool_rejects_remote_name_keyword(self) -> None:
        with pytest.raises(TypeError, match="remote_name"):
            to_tool(add, remote_name="remote_add")  # type: ignore[call-arg]


class TestToToolExistingTool:
    def test_existing_tool_returns_same_instance(self) -> None:
        original = Tool(
            function=add,
            name="add",
            description="Add values.",
        )

        result = to_tool(original)

        assert result is original

    def test_existing_tool_with_overrides_returns_wrapper_without_mutating_original(self) -> None:
        original = Tool(
            function=add,
            name="add",
            description="Add values.",
        )

        result = to_tool(
            original,
            name="sum_values",
            description="Sum values.",
        )

        assert result is not original
        assert type(result) is Tool
        assert result.function is original

        assert result.name == "sum_values"
        assert result.description == "Sum values."
        assert result.full_name == "Tool.sum_values"

        assert original.name == "add"
        assert original.description == "Add values."
        assert original.full_name == "Tool.add"

        assert result.invoke({"a": 2, "b": 3}).result == 5

    def test_existing_tool_without_overrides_preserves_metadata(self) -> None:
        original = Tool(
            function=add,
            name="add",
            description="Add values.",
        )

        result = to_tool(original)

        assert result is original
        assert original.name == "add"
        assert original.description == "Add values."

    def test_existing_tool_wrapper_preserves_omitted_metadata(self) -> None:
        original = Tool(
            function=add,
            name="add",
            description="Add values.",
        )

        result = to_tool(original, name="sum_values")

        assert result is not original
        assert result.function is original
        assert result.name == "sum_values"
        assert result.description == "Add values."

        assert original.name == "add"
        assert original.description == "Add values."

    def test_existing_tool_rejects_remote_name_keyword(self) -> None:
        original = Tool(function=add, name="add", description="Add values.")

        with pytest.raises(TypeError, match="remote_name"):
            to_tool(original, remote_name="remote_add")  # type: ignore[call-arg]


class TestToToolPositionalOverrides:
    """`name` and `description` are positional-or-keyword."""

    def test_callable_accepts_positional_name_and_description(self) -> None:
        def add(a: int, b: int) -> int:
            return a + b

        tool = to_tool(add, "adder", "Adds two ints.")

        assert tool.name == "adder"
        assert tool.description == "Adds two ints."

    def test_existing_tool_accepts_positional_name(self) -> None:
        def add(a: int, b: int) -> int:
            return a + b

        original = to_tool(add)

        renamed = to_tool(original, "renamed")

        assert renamed is not original
        assert renamed.name == "renamed"
        assert renamed.function is original

    def test_no_fourth_positional_parameter(self) -> None:
        with pytest.raises(TypeError):
            to_tool(FakeMCPClientHub(), None, None, "search")  # type: ignore[misc]


class TestToToolExistingProxyTool:
    """`to_tool()`'s route 1 ("already a Tool") must work uniformly across
    plain `Tool` and every proxy-tool subtype (`MCPProxyTool`,
    `A2AProxyTool` in both skill and generic mode) --
    an explicit Pass 4 checklist item that was previously unverified."""

    def test_mcp_proxy_tool_without_overrides_returned_unchanged(self) -> None:
        original = MCPProxyTool(FakeMCPClientHub(), "search")

        result = to_tool(original)

        assert result is original

    def test_mcp_proxy_tool_with_overrides_wraps_by_reference(self) -> None:
        original = MCPProxyTool(FakeMCPClientHub(), "search")

        result = to_tool(original, name="local_search")

        assert result is not original
        assert type(result) is Tool
        assert result.function is original
        assert result.name == "local_search"
        assert result.invoke({"query": "hello"}).result == "mcp result"

    def test_a2a_proxy_tool_skill_mode_without_overrides_returned_unchanged(self) -> None:
        original = A2AProxyTool(FakeA2AClientHub(), "add")

        result = to_tool(original)

        assert result is original

    def test_a2a_proxy_tool_skill_mode_with_overrides_wraps_by_reference(self) -> None:
        original = A2AProxyTool(FakeA2AClientHub(), "add")

        result = to_tool(original, name="local_add")

        assert result is not original
        assert type(result) is Tool
        assert result.function is original
        assert result.name == "local_add"
        assert result.invoke({"a": 1, "b": 2}).result == 42

    def test_a2a_proxy_tool_generic_mode_without_overrides_returned_unchanged(self) -> None:
        original = A2AProxyTool(FakeA2AClientHub())

        result = to_tool(original)

        assert result is original

    def test_a2a_proxy_tool_generic_mode_with_overrides_wraps_by_reference(self) -> None:
        original = A2AProxyTool(FakeA2AClientHub())

        result = to_tool(original, name="local_send")

        assert result is not original
        assert type(result) is Tool
        assert result.function is original
        assert result.name == "local_send"


class TestToToolAtomicInvokableWrapping:
    def test_non_tool_atomic_invokable_wraps_by_reference(self) -> None:
        invokable = EchoInvokable()

        tool = to_tool(invokable)

        assert type(tool) is Tool
        assert tool.function is invokable
        assert tool.name == "echo_invokable"
        assert tool.description == "Echo invokable."
        assert tool.invoke({"value": 123}).result == {"value": 123}


class TestToToolReturnToolIdentity:
    def test_to_tool_return_tool_with_override_delegates_by_reference(self) -> None:
        result = to_tool(return_tool, name="alias")

        assert result is not return_tool
        assert result.function is return_tool
        assert result.name == "alias"


class TestToToolMCPClientHub:
    def test_mcp_client_hub_without_name_raises_hub_not_found(self) -> None:
        with pytest.raises(MCPToolError, match="not found"):
            to_tool(FakeMCPClientHub())

    def test_mcp_client_hub_becomes_mcp_proxy_tool(self) -> None:
        hub = FakeMCPClientHub()

        tool = to_tool(hub, "search")

        assert isinstance(tool, MCPProxyTool)
        assert tool.remote_name == "search"
        assert tool.name == "search"

    def test_named_hub_prefixes_proxy_name(self) -> None:
        tool = to_tool(FakeMCPClientHub(hub_name="files"), name="search")

        assert type(tool) is MCPProxyTool
        assert tool.name == "files_search"
        assert tool.remote_name == "search"

    def test_description_applied_through_setter_keeps_proxy_class(self) -> None:
        tool = to_tool(FakeMCPClientHub(), name="search", description="Local search.")

        assert type(tool) is MCPProxyTool
        assert tool.name == "search"
        assert tool.description == "Local search."

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(MCPToolError):
            to_tool(FakeMCPClientHub(), name="nonexistent")

    def test_untrimmed_name_raises_showing_what_was_typed(self) -> None:
        with pytest.raises(MCPToolError, match=r"' search '"):
            to_tool(FakeMCPClientHub(), name=" search ")

    def test_blank_name_raises(self) -> None:
        with pytest.raises(MCPToolError):
            to_tool(FakeMCPClientHub(), name="")

    def test_mcp_proxy_tool_invokes_fake_hub(self) -> None:
        hub = FakeMCPClientHub(result={"structuredContent": {"result": "ok"}})
        tool = to_tool(hub, name="search")

        assert tool.invoke({"query": "hello"}).result == "ok"
        assert hub.calls == [("search", {"query": "hello"})]


class TestToToolA2AClientHub:
    def test_name_present_selects_skill_mode(self) -> None:
        hub = FakeA2AClientHub()

        tool = to_tool(hub, name="add")

        assert isinstance(tool, A2AProxyTool)
        assert tool.remote_name == "add"
        assert tool.name == "add"

    def test_named_hub_prefixes_skill_and_generic_names(self) -> None:
        hub = FakeA2AClientHub(hub_name="calc")

        assert to_tool(hub, name="add").name == "calc_add"
        assert to_tool(hub).name == "calc_send_parts"

    def test_name_absent_selects_generic_mode_not_an_error(self) -> None:
        hub = FakeA2AClientHub()

        tool = to_tool(hub)

        assert isinstance(tool, A2AProxyTool)
        assert tool.remote_name is None
        assert tool.name == "send_parts"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_name_blank_raises_a2a_proxy_error(self, blank: str) -> None:
        with pytest.raises(A2AProxyError):
            to_tool(FakeA2AClientHub(), name=blank)

    def test_name_is_not_stripped(self) -> None:
        with pytest.raises(A2AProxyError):
            to_tool(FakeA2AClientHub(), name=" add ")

    def test_unknown_skill_raises_a2a_proxy_error(self) -> None:
        hub = FakeA2AClientHub()

        with pytest.raises(A2AProxyError, match="not found"):
            to_tool(hub, name="does_not_exist")

    def test_description_applied_through_setter_keeps_proxy_class(self) -> None:
        tool = to_tool(FakeA2AClientHub(), name="add", description="Local add.")

        assert type(tool) is A2AProxyTool
        assert tool.name == "add"
        assert tool.description == "Local add."

    def test_skill_mode_invokes_fake_hub(self) -> None:
        hub = FakeA2AClientHub(result=17)
        tool = to_tool(hub, name="add")

        assert tool.invoke({"a": 12, "b": 5}).result == 17
        assert hub.skill_calls == [("add", {"a": 12, "b": 5})]


class TestToToolDecoratorForm:
    def test_bare_decorator(self) -> None:
        @to_tool
        def plus(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        assert type(plus) is Tool
        assert plus.name == "plus"
        assert plus.description == "Add two integers."

    def test_empty_call_returns_decorator(self) -> None:
        decorated = to_tool()(add)

        assert type(decorated) is Tool
        assert decorated.name == "add"
        assert decorated.description == "Add two integers."

    def test_keyword_overrides(self) -> None:
        @to_tool(name="adder", description="my description")
        def plus(a, b):
            return a + b

        assert plus.name == "adder"
        assert plus.description == "my description"

    def test_name_only_keeps_docstring_description(self) -> None:
        decorated = to_tool(name="sum_two")(add)

        assert decorated.name == "sum_two"
        assert decorated.description == "Add two integers."

    def test_call_form_still_returns_tool(self) -> None:
        result = to_tool(add)

        assert type(result) is Tool
        assert result.name == "add"

    def test_matches_call_form(self) -> None:
        decorated = to_tool(name="sum_two", description="Sum values.")(add)
        direct = to_tool(add, "sum_two", "Sum values.")

        assert decorated.name == direct.name
        assert decorated.description == direct.description
        assert decorated.parameters == direct.parameters

    def test_unannotated_parameters_work(self) -> None:
        @to_tool
        def plus(a, b):
            return a + b

        assert plus.invoke({"a": 1, "b": 2}).result == 3

    def test_sync_function_invokes(self) -> None:
        assert to_tool(add).invoke({"a": 2, "b": 3}).result == 5

    def test_async_function_awaits(self) -> None:
        async def async_add(a: int, b: int) -> int:
            return a + b

        decorated = to_tool(async_add)

        assert asyncio.run(decorated.async_invoke({"a": 2, "b": 3})).result == 5

    def test_nameless_callable_raises(self) -> None:
        with pytest.raises(ToolDefinitionError, match="cannot infer a name"):
            to_tool()(CallableWithoutName())

    def test_positional_string_raises_keyword_hint(self) -> None:
        with pytest.raises(ToolDefinitionError, match="keyword arguments only"):
            to_tool("name")  # type: ignore[arg-type]

    def test_non_callable_raises_unsupported_type(self) -> None:
        with pytest.raises(ToolDefinitionError, match="unsupported `component` type"):
            to_tool(42)  # type: ignore[arg-type]

    def test_explicit_none_is_not_a_decorator(self) -> None:
        with pytest.raises(ToolDefinitionError):
            to_tool(None)  # type: ignore[arg-type]

    def test_decorated_function_registers_on_tool_agent(self) -> None:
        agent = make_stub_tool_agent()

        assert agent.register_tool(to_tool(add)) is True
        assert agent.has_tool("add")


class TestToToolInvalidInputs:
    def test_to_tool_none_raises_tool_definition_error(self) -> None:
        with pytest.raises(ToolDefinitionError, match="expected a non-empty"):
            to_tool(None)  # type: ignore[arg-type]

    def test_to_tool_unsupported_component_type_raises(self) -> None:
        with pytest.raises(ToolDefinitionError, match="unsupported"):
            to_tool(object())  # type: ignore[arg-type]


class TestToToolsLocalSources:
    def test_to_tools_empty_or_none_returns_empty_list(self) -> None:
        assert to_tools([]) == []
        assert to_tools(None) == []

    def test_to_tools_names_local_tools(self) -> None:
        tools = to_tools([add, multiply])

        assert [tool.full_name for tool in tools] == [
            "Tool.add",
            "Tool.multiply",
        ]

    def test_to_tools_wraps_non_tool_atomic_invokable(self) -> None:
        invokable = EchoInvokable()

        tools = to_tools([add, invokable])

        assert len(tools) == 2
        assert type(tools[0]) is Tool
        assert type(tools[1]) is Tool
        assert tools[1].function is invokable
        assert tools[0].full_name == "Tool.add"
        assert tools[1].full_name == "Tool.echo_invokable"

    def test_to_tools_returns_existing_tool_unchanged(self) -> None:
        original = Tool(
            function=add,
            name="add",
            description="Add values.",
        )

        tools = to_tools([original])

        assert tools == [original]
        assert tools[0] is original


class TestToToolsRemoteExpansion:
    def test_to_tools_expands_mcp_hub_tools(self) -> None:
        hub = FakeMCPClientHub()

        tools = to_tools([hub])

        assert len(tools) == 2
        assert all(isinstance(tool, MCPProxyTool) for tool in tools)
        assert [tool.remote_name for tool in tools] == ["search", "summarize"]

    def test_to_tools_keeps_names_and_prefixes_for_mcp_hub(self) -> None:
        hub = FakeMCPClientHub(hub_name="files")

        tools = to_tools([hub])

        assert [tool.name for tool in tools] == ["files_search", "files_summarize"]
        assert [tool.remote_name for tool in tools] == ["search", "summarize"]

    def test_to_tools_on_real_hub_adds_no_fetch_beyond_construction(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fetches: list[int] = []

        def fake_fetch(self: MCPClientHub) -> dict[str, dict[str, Any]]:
            fetches.append(1)
            return {
                "search": mcp_metadata(description="Remote MCP search."),
                "summarize": mcp_metadata(description="Remote MCP summarize."),
            }

        monkeypatch.setattr(MCPClientHub, "_fetch_tools", fake_fetch)
        hub = MCPClientHub("python", "stdio", name="files")
        assert len(fetches) == 1

        tools = to_tools([hub])
        single = to_tool(hub, "search", description="Local search.")

        assert len(fetches) == 1
        assert [tool.name for tool in tools] == ["files_search", "files_summarize"]
        assert [tool.remote_name for tool in tools] == ["search", "summarize"]
        assert single.description == "Local search."

    def test_to_tool_on_real_hub_unknown_name_raises_mcp_tool_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            MCPClientHub, "_fetch_tools", lambda self: {"search": mcp_metadata()}
        )
        hub = MCPClientHub("python", "stdio")

        with pytest.raises(MCPToolError, match="nonexistent"):
            to_tool(hub, "nonexistent")

    def test_to_tools_mixed_sources_preserves_expanded_order(self) -> None:
        hub = FakeMCPClientHub()

        tools = to_tools([add, hub, multiply])

        assert [type(tool) for tool in tools] == [
            Tool,
            MCPProxyTool,
            MCPProxyTool,
            Tool,
        ]
        assert [tool.name for tool in tools] == [
            "add",
            "search",
            "summarize",
            "multiply",
        ]


    def test_to_tools_remote_tools_can_invoke_fake_backends(self) -> None:
        hub = FakeMCPClientHub(result={"structuredContent": {"result": "mcp ok"}})
        a2a_hub = FakeA2AClientHub(result=17)

        tools = to_tools([hub, a2a_hub])
        mcp_tool = tools[0]
        a2a_tool = tools[2]

        assert mcp_tool.invoke({"query": "hello"}).result == "mcp ok"
        assert a2a_tool.invoke({"a": 1, "b": 2}).result == 17

        assert hub.calls == [("search", {"query": "hello"})]
        assert a2a_hub.skill_calls == [("add", {"a": 1, "b": 2})]


class TestToToolsHubPrefixAndDuplicates:
    def test_named_hubs_prefix_names_in_order(self) -> None:
        tools = to_tools([FakeMCPClientHub(hub_name="m"), FakeA2AClientHub(hub_name="a")])

        assert [t.name for t in tools] == [
            "m_search",
            "m_summarize",
            "a_add",
            "a_send_parts",
        ]

    def test_unnamed_hubs_keep_duplicates_without_raising(self) -> None:
        tools = to_tools([FakeA2AClientHub(), FakeA2AClientHub()])

        assert [t.name for t in tools] == ["add", "send_parts", "add", "send_parts"]


class TestToToolsA2AClientHub:
    def test_expands_skills_plus_one_trailing_generic_tool(self) -> None:
        hub = FakeA2AClientHub(
            skills={
                "add": a2a_sdk_skill_metadata(remote_name="add"),
                "multiply": a2a_sdk_skill_metadata(remote_name="multiply"),
            }
        )

        tools = to_tools([hub])

        assert len(tools) == 3
        assert all(isinstance(tool, A2AProxyTool) for tool in tools)
        assert [tool.remote_name for tool in tools] == ["add", "multiply", None]
        assert tools[-1].name == "send_parts"

    def test_zero_skill_hub_still_yields_one_generic_tool(self) -> None:
        hub = FakeA2AClientHub(skills={})

        tools = to_tools([hub])

        assert len(tools) == 1
        assert tools[0].remote_name is None


    def test_mixed_sources_preserves_expanded_order(self) -> None:
        hub = FakeA2AClientHub(
            skills={"add": a2a_sdk_skill_metadata(remote_name="add")}
        )

        tools = to_tools([add, hub])

        assert [type(tool) for tool in tools] == [Tool, A2AProxyTool, A2AProxyTool]
        assert [tool.name for tool in tools] == ["add", "add", "send_parts"]
