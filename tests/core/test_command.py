# tests/core/test_command.py

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from atomic_agentic.core.Invokable import StructuredInvokable, Command
from atomic_agentic.tools.base import Tool


def add(*, a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def add_with_default(*, a: int, b: int = 10) -> int:
    """Add two integers with a default second value."""
    return a + b


async def async_add(*, a: int, b: int) -> int:
    """Asynchronously add two integers."""
    return a + b


def make_add_tool(
    *,
    filter_extraneous_inputs: bool = True,
) -> Tool:
    return Tool(
        function=add,
        name="add",
        namespace="tests",
        description="Add two integers.",
        filter_extraneous_inputs=filter_extraneous_inputs,
    )


class TestCommandConstruction:
    def test_constructs_no_input_invokable_from_tool(self) -> None:
        executor = make_add_tool()

        command = Command(
            executor=executor,
            fixed_inputs={"a": 2, "b": 3},
            name="add_two_and_three",
            description="Add 2 and 3.",
        )

        assert command.executor is executor
        assert command.fixed_inputs == {"a": 2, "b": 3}
        assert command.parameters == []
        assert command.return_type == "int"
        assert command.filter_extraneous_inputs is False
        assert command.name == "add_two_and_three"
        assert command.description == "Add 2 and 3."
        assert command.signature == "Command.add_two_and_three() -> int"

    def test_defaults_identity_when_name_and_description_are_omitted(self) -> None:
        executor = make_add_tool()

        command = Command(
            executor=executor,
            fixed_inputs={"a": 2, "b": 3},
        )

        assert command.name == "add_command"
        assert "Command wrapper for Tool.tests.add" in command.description

    def test_requires_atomic_invokable_executor(self) -> None:
        with pytest.raises(TypeError, match="AtomicInvokable"):
            Command(
                executor=object(),  # type: ignore[arg-type]
                fixed_inputs={},
            )

    def test_requires_mapping_fixed_inputs(self) -> None:
        executor = make_add_tool()

        with pytest.raises(TypeError, match="fixed_inputs must be a mapping"):
            Command(
                executor=executor,
                fixed_inputs=[("a", 1), ("b", 2)],  # type: ignore[arg-type]
            )

    def test_fixed_inputs_property_returns_copy(self) -> None:
        command = Command(
            executor=make_add_tool(),
            fixed_inputs={"a": 2, "b": 3},
        )

        returned = command.fixed_inputs
        returned["a"] = 999

        assert command.fixed_inputs == {"a": 2, "b": 3}


class TestCommandFixedInputValidation:
    def test_missing_required_fixed_input_raises_at_construction(self) -> None:
        executor = make_add_tool()

        with pytest.raises(TypeError, match="missing required input"):
            Command(
                executor=executor,
                fixed_inputs={"a": 2},
            )

    def test_missing_optional_fixed_input_is_allowed(self) -> None:
        executor = Tool(
            function=add_with_default,
            name="add_with_default",
            namespace="tests",
            description="Add two integers with a default.",
        )

        command = Command(
            executor=executor,
            fixed_inputs={"a": 5},
        )

        assert command.invoke({}) == 15

    def test_strict_executor_rejects_extra_fixed_input_at_construction(self) -> None:
        executor = make_add_tool(filter_extraneous_inputs=False)

        with pytest.raises(TypeError, match="unexpected input key"):
            Command(
                executor=executor,
                fixed_inputs={"a": 2, "b": 3, "unused": 4},
            )

    def test_filtering_executor_drops_extra_fixed_input_at_construction(self) -> None:
        executor = make_add_tool(filter_extraneous_inputs=True)

        command = Command(
            executor=executor,
            fixed_inputs={"a": 2, "b": 3, "unused": 4},
        )

        assert command.fixed_inputs == {"a": 2, "b": 3}
        assert command.invoke({}) == 5


class TestCommandFilterPolicy:
    def test_filter_extraneous_inputs_is_fixed_to_false(self) -> None:
        command = Command(
            executor=make_add_tool(),
            fixed_inputs={"a": 2, "b": 3},
        )

        assert command.filter_extraneous_inputs is False

        command.filter_extraneous_inputs = False
        assert command.filter_extraneous_inputs is False

        with pytest.raises(ValueError, match="fixed to False"):
            command.filter_extraneous_inputs = True

    @pytest.mark.parametrize("value", ["false", 0, 1, None, [], {}])
    def test_filter_extraneous_inputs_rejects_non_bool_assignment(
        self,
        value: object,
    ) -> None:
        command = Command(
            executor=make_add_tool(),
            fixed_inputs={"a": 2, "b": 3},
        )

        with pytest.raises(TypeError, match="must be a bool"):
            command.filter_extraneous_inputs = value  # type: ignore[assignment]


class TestCommandInvocation:
    def test_invoke_empty_mapping_delegates_to_executor_with_fixed_inputs(self) -> None:
        command = Command(
            executor=make_add_tool(),
            fixed_inputs={"a": 2, "b": 3},
        )

        assert command.invoke({}) == 5

    def test_call_without_arguments_delegates_to_executor(self) -> None:
        command = Command(
            executor=make_add_tool(),
            fixed_inputs={"a": 2, "b": 3},
        )

        assert command() == 5

    def test_invoke_rejects_runtime_inputs(self) -> None:
        command = Command(
            executor=make_add_tool(),
            fixed_inputs={"a": 2, "b": 3},
        )

        with pytest.raises(TypeError, match="unexpected input key"):
            command.invoke({"a": 100})

    def test_invoke_requires_mapping(self) -> None:
        command = Command(
            executor=make_add_tool(),
            fixed_inputs={"a": 2, "b": 3},
        )

        with pytest.raises(TypeError, match="inputs must be a mapping"):
            command.invoke(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_call_rejects_positional_arguments(self) -> None:
        command = Command(
            executor=make_add_tool(),
            fixed_inputs={"a": 2, "b": 3},
        )

        with pytest.raises(TypeError, match="at most 0 positional"):
            command(1)

    def test_call_rejects_keyword_arguments(self) -> None:
        command = Command(
            executor=make_add_tool(),
            fixed_inputs={"a": 2, "b": 3},
        )

        with pytest.raises(TypeError, match="unexpected keyword"):
            command(a=100)


class TestCommandAsyncInvocation:
    def test_async_invoke_delegates_to_executor_async_path(self) -> None:
        executor = Tool(
            function=async_add,
            name="async_add",
            namespace="tests",
            description="Asynchronously add two integers.",
        )
        command = Command(
            executor=executor,
            fixed_inputs={"a": 2, "b": 3},
        )

        result = asyncio.run(command.async_invoke({}))

        assert result == 5

    def test_async_call_without_arguments_delegates_to_async_invoke(self) -> None:
        executor = Tool(
            function=async_add,
            name="async_add",
            namespace="tests",
            description="Asynchronously add two integers.",
        )
        command = Command(
            executor=executor,
            fixed_inputs={"a": 2, "b": 3},
        )

        result = asyncio.run(command.async_call())

        assert result == 5

    def test_async_invoke_rejects_runtime_inputs(self) -> None:
        command = Command(
            executor=make_add_tool(),
            fixed_inputs={"a": 2, "b": 3},
        )

        with pytest.raises(TypeError, match="unexpected input key"):
            asyncio.run(command.async_invoke({"a": 100}))


class TestCommandSerialization:
    def test_to_dict_includes_command_metadata_executor_and_fixed_inputs(self) -> None:
        executor = make_add_tool()
        command = Command(
            executor=executor,
            fixed_inputs={"a": 2, "b": 3},
            name="add_two_and_three",
            description="Add 2 and 3.",
        )

        data = command.to_dict()

        assert data["type"] == "Command"
        assert data["name"] == "add_two_and_three"
        assert data["description"] == "Add 2 and 3."
        assert data["parameters"] == []
        assert data["return_type"] == "int"
        assert data["filter_extraneous_inputs"] is False
        assert data["fixed_inputs"] == {"a": 2, "b": 3}
        assert data["executor"]["type"] == "Tool"
        assert data["executor"]["name"] == "add"


class TestCommandComposition:
    def test_structured_invokable_can_wrap_command(self) -> None:
        command = Command(
            executor=make_add_tool(),
            fixed_inputs={"a": 2, "b": 3},
            name="add_two_and_three",
            description="Add 2 and 3.",
        )

        wrapper = StructuredInvokable(
            component=command,
            output_schema=["value"],
        )

        result = wrapper.invoke({})

        assert type(result) is dict
        assert result == {"value": 5}
        assert not hasattr(result, "raw_result")
        assert wrapper.parameters == []
