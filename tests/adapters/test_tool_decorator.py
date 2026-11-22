from __future__ import annotations

import asyncio

import pytest

from atomic_agentic.adapters import to_tool, tool
from atomic_agentic.exceptions import ToolDefinitionError
from atomic_agentic.tools.base import Tool

from ..agents.conftest import make_stub_tool_agent


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def async_add(a: int, b: int) -> int:
    """Add two integers asynchronously."""
    return a + b


class CallableWithoutName:
    def __call__(self, value: str) -> str:
        return value


class TestToolDecoratorForms:
    def test_bare_decorator(self) -> None:
        @tool
        def plus(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        assert type(plus) is Tool
        assert plus.name == "plus"
        assert plus.description == "Add two integers."

    def test_empty_parentheses(self) -> None:
        @tool()
        def plus(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        assert type(plus) is Tool
        assert plus.name == "plus"
        assert plus.description == "Add two integers."

    def test_keyword_overrides(self) -> None:
        @tool(name="sum_two", description="Sum values.")
        def plus(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        assert plus.name == "sum_two"
        assert plus.description == "Sum values."

    def test_name_only_keeps_docstring_description(self) -> None:
        decorated = tool(name="sum_two")(add)

        assert decorated.name == "sum_two"
        assert decorated.description == "Add two integers."

    def test_direct_call(self) -> None:
        result = tool(add)

        assert type(result) is Tool
        assert result.name == "add"
        assert result.description == "Add two integers."


class TestToolDecoratorBehavior:
    def test_sync_function_invokes(self) -> None:
        assert tool(add).invoke({"a": 2, "b": 3}).result == 5

    def test_async_function_awaits(self) -> None:
        decorated = tool(async_add)

        assert asyncio.run(decorated.async_invoke({"a": 2, "b": 3})).result == 5

    def test_matches_to_tool(self) -> None:
        decorated = tool(name="sum_two", description="Sum values.")(add)
        direct = to_tool(add, "sum_two", "Sum values.")

        assert decorated.name == direct.name
        assert decorated.description == direct.description
        assert decorated.parameters == direct.parameters

    def test_redecorating_a_tool_wraps_it_under_the_new_identity(self) -> None:
        original = tool(add)

        wrapper = tool(name="renamed")(original)

        assert wrapper is not original
        assert wrapper.name == "renamed"
        assert original.name == "add"
        assert wrapper.invoke({"a": 1, "b": 2}).result == 3


class TestToolDecoratorErrors:
    def test_positional_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="keyword arguments only"):
            tool("name")  # type: ignore[call-overload]

    def test_non_callable_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="keyword arguments only"):
            tool(42)  # type: ignore[call-overload]

    def test_callable_without_name_raises_tool_definition_error(self) -> None:
        with pytest.raises(ToolDefinitionError):
            tool(CallableWithoutName())

    def test_callable_without_name_succeeds_with_name(self) -> None:
        decorated = tool(name="echo")(CallableWithoutName())

        assert decorated.name == "echo"


class TestToolDecoratorIntegration:
    def test_decorated_function_registers_on_tool_agent(self) -> None:
        agent = make_stub_tool_agent()

        assert agent.register_tool(tool(add)) is True
        assert agent.has_tool("add")
