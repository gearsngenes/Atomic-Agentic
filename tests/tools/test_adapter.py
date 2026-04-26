from __future__ import annotations

import asyncio

from atomic_agentic.tools.adapter import AdapterTool
from atomic_agentic.tools.base import Tool


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def make_base_tool() -> Tool:
    return Tool(
        function=add,
        name="add",
        namespace="math",
        description="Add values.",
    )


class TestAdapterTool:
    def test_wraps_concrete_tool_and_exposes_component_schema(self) -> None:
        base_tool = make_base_tool()
        adapter = AdapterTool(base_tool)

        assert adapter.component is base_tool
        assert adapter.name == "add"
        assert adapter.namespace == "wrapped_math"
        assert adapter.description == "Add values."
        assert adapter.full_name == "AdapterTool.wrapped_math.add"
        assert adapter.parameters == base_tool.parameters
        assert adapter.return_type == base_tool.return_type

    def test_invoke_forwards_dict_inputs_to_component(self) -> None:
        base_tool = make_base_tool()
        adapter = AdapterTool(base_tool)

        assert adapter.invoke({"a": 2, "b": 3}) == 5

    def test_async_invoke_forwards_to_component_async_invoke(self) -> None:
        base_tool = make_base_tool()
        adapter = AdapterTool(base_tool)

        result = asyncio.run(adapter.async_invoke({"a": 2, "b": 3}))

        assert result == 5

    def test_component_setter_swaps_component_and_refreshes_schema(self) -> None:
        base_tool = make_base_tool()
        replacement = Tool(
            function=multiply,
            name="multiply",
            namespace="math",
            description="Multiply values.",
        )
        adapter = AdapterTool(base_tool)

        adapter.component = replacement

        assert adapter.component is replacement
        assert adapter.name == "multiply"
        assert adapter.description == "Multiply values."
        assert adapter.parameters == replacement.parameters
        assert adapter.return_type == replacement.return_type
        assert adapter.invoke({"a": 3, "b": 4}) == 12

    def test_to_dict_includes_nested_component(self) -> None:
        base_tool = make_base_tool()
        adapter = AdapterTool(base_tool)

        data = adapter.to_dict()

        assert data["type"] == "AdapterTool"
        assert data["name"] == "add"
        assert data["namespace"] == "wrapped_math"
        assert data["component"]["type"] == "Tool"
        assert data["component"]["name"] == "add"
        assert data["component"]["namespace"] == "math"
