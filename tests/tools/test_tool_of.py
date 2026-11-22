from __future__ import annotations

import asyncio

import pytest

from atomic_agentic.exceptions import ToolDefinitionError
from atomic_agentic.tools.base import Tool

from ..agents.conftest import make_stub_tool_agent


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def async_add(a: int, b: int) -> int:
    """Add two integers asynchronously."""
    return a + b


class TestToolOfForms:
    def test_bare_decorator(self) -> None:
        @Tool.of
        def plus(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        assert type(plus) is Tool
        assert plus.name == "plus"
        assert plus.description == "Add two integers."

    def test_empty_parentheses(self) -> None:
        decorated = Tool.of()(add)

        assert type(decorated) is Tool
        assert decorated.name == "add"
        assert decorated.description == "Add two integers."

    def test_keyword_overrides(self) -> None:
        @Tool.of(name="adder", description="my description")
        def plus(a, b):
            return a + b

        assert plus.name == "adder"
        assert plus.description == "my description"

    def test_name_only_keeps_docstring_description(self) -> None:
        decorated = Tool.of(name="sum_two")(add)

        assert decorated.name == "sum_two"
        assert decorated.description == "Add two integers."

    def test_direct_call(self) -> None:
        result = Tool.of(add)

        assert type(result) is Tool
        assert result.name == "add"


class TestToolOfBehavior:
    def test_matches_constructor(self) -> None:
        decorated = Tool.of(name="sum_two", description="Sum values.")(add)
        direct = Tool(add, "sum_two", "Sum values.")

        assert decorated.name == direct.name
        assert decorated.description == direct.description
        assert decorated.parameters == direct.parameters

    def test_unannotated_parameters_work(self) -> None:
        @Tool.of
        def plus(a, b):
            return a + b

        assert plus.invoke({"a": 1, "b": 2}).result == 3

    def test_sync_function_invokes(self) -> None:
        assert Tool.of(add).invoke({"a": 2, "b": 3}).result == 5

    def test_async_function_awaits(self) -> None:
        decorated = Tool.of(async_add)

        assert asyncio.run(decorated.async_invoke({"a": 2, "b": 3})).result == 5


class TestToolOfErrors:
    def test_positional_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="keyword arguments only"):
            Tool.of("name")  # type: ignore[arg-type]

    def test_non_callable_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="keyword arguments only"):
            Tool.of(42)  # type: ignore[arg-type]

    def test_constructor_still_rejects_missing_function(self) -> None:
        with pytest.raises(TypeError):
            Tool()  # type: ignore[call-arg]
        with pytest.raises(ToolDefinitionError):
            Tool(None)  # type: ignore[arg-type]


class TestToolOfIntegration:
    def test_decorated_function_registers_on_tool_agent(self) -> None:
        agent = make_stub_tool_agent()

        assert agent.register_tool(Tool.of(add)) is True
        assert agent.has_tool("add")
