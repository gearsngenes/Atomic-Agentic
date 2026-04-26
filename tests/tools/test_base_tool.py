from __future__ import annotations

import asyncio
from typing import Any

import pytest

from atomic_agentic.core.Exceptions import ToolDefinitionError, ToolInvocationError
from atomic_agentic.core.sentinels import NO_VAL
from atomic_agentic.tools.base import Tool


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def greet(name: str, punctuation: str = "!") -> str:
    """Greet someone."""
    return f"Hello, {name}{punctuation}"


def keyword_only(*, text: str, upper: bool = False) -> str:
    return text.upper() if upper else text


def positional_only(a: int, /, b: int) -> int:
    return a - b


def collect(a: int, *items: int) -> tuple[int, tuple[int, ...]]:
    return a, items


def collect_kwargs(a: int, **extras: Any) -> dict[str, Any]:
    return {"a": a, **extras}


def fail() -> None:
    raise RuntimeError("boom")


async def async_add(a: int, b: int) -> int:
    return a + b


async def async_fail() -> None:
    raise RuntimeError("async boom")


class TestToolConstruction:
    def test_valid_construction_from_callable(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        assert tool.name == "add"
        assert tool.namespace == "tests"
        assert tool.description == "Add values."
        assert tool.full_name == "Tool.tests.add"
        assert tool.function is add
        assert tool.module == add.__module__
        assert tool.qualname == add.__qualname__

    def test_default_namespace_and_inferred_metadata(self) -> None:
        tool = Tool(function=add)

        assert tool.name == "add"
        assert tool.namespace == "default"
        assert tool.description == "Add two integers."
        assert tool.full_name == "Tool.default.add"

    def test_non_callable_function_raises(self) -> None:
        with pytest.raises(ToolDefinitionError, match="callable"):
            Tool(function=123)  # type: ignore[arg-type]

    def test_parameters_and_return_type_are_extracted(self) -> None:
        tool = Tool(function=greet, namespace="tests", description="Greet.")

        assert [(p.name, p.kind, p.default) for p in tool.parameters] == [
            ("name", "POSITIONAL_OR_KEYWORD", NO_VAL),
            ("punctuation", "POSITIONAL_OR_KEYWORD", "!"),
        ]
        assert tool.return_type == "str"

    def test_to_dict_includes_tool_metadata(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        data = tool.to_dict()

        assert data["type"] == "Tool"
        assert data["name"] == "add"
        assert data["description"] == "Add values."
        assert data["namespace"] == "tests"
        assert data["module"] == add.__module__
        assert data["qualname"] == add.__qualname__
        assert data["return_type"] == "int"


class TestToolBindingBasic:
    def test_required_arguments_bind_and_execute(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        assert tool.invoke({"a": 2, "b": 3}) == 5

    def test_default_values_are_applied(self) -> None:
        tool = Tool(
            function=greet,
            name="greet",
            namespace="tests",
            description="Greet.",
        )

        assert tool.invoke({"name": "Ada"}) == "Hello, Ada!"
        assert tool.invoke({"name": "Ada", "punctuation": "."}) == "Hello, Ada."

    def test_keyword_only_parameters_bind_correctly(self) -> None:
        tool = Tool(
            function=keyword_only,
            name="keyword_only",
            namespace="tests",
            description="Keyword-only text tool.",
        )

        assert tool.invoke({"text": "hello"}) == "hello"
        assert tool.invoke({"text": "hello", "upper": True}) == "HELLO"

    def test_missing_required_parameter_raises(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        with pytest.raises(ToolInvocationError, match="missing required"):
            tool.invoke({"a": 1})

    def test_unknown_input_raises_when_filtering_disabled(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
            filter_extraneous_inputs=False,
        )

        with pytest.raises(ToolInvocationError, match="unknown parameters"):
            tool.invoke({"a": 1, "b": 2, "extra": 3})

    def test_filter_extraneous_inputs_true_filters_unknown_inputs(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
            filter_extraneous_inputs=True,
        )

        assert tool.invoke({"a": 1, "b": 2, "extra": 3}) == 3


class TestToolBindingPositionalKinds:
    def test_positional_only_parameters_bind_as_args(self) -> None:
        tool = Tool(
            function=positional_only,
            name="positional_only",
            namespace="tests",
            description="Positional-only tool.",
        )

        args, kwargs = tool.to_arg_kwarg({"a": 5, "b": 2})

        assert args == (5,)
        assert kwargs == {"b": 2}
        assert tool.invoke({"a": 5, "b": 2}) == 3

    def test_explicit_varargs_payload_switches_pos_or_kw_to_args(self) -> None:
        tool = Tool(
            function=collect,
            name="collect",
            namespace="tests",
            description="Collect varargs.",
        )

        args, kwargs = tool.to_arg_kwarg({"a": 1, "items": [2, 3, 4]})

        assert args == (1, 2, 3, 4)
        assert kwargs == {}
        assert tool.invoke({"a": 1, "items": [2, 3, 4]}) == (1, (2, 3, 4))

    @pytest.mark.parametrize("payload", ["abc", b"abc", bytearray(b"abc")])
    def test_explicit_varargs_rejects_string_like_payloads(self, payload: Any) -> None:
        tool = Tool(
            function=collect,
            name="collect",
            namespace="tests",
            description="Collect varargs.",
        )

        with pytest.raises(TypeError, match="must be a list or tuple"):
            tool.invoke({"a": 1, "items": payload})

    @pytest.mark.parametrize("payload", [1, {"x": 1}, None])
    def test_explicit_varargs_rejects_non_sequence_payloads(self, payload: Any) -> None:
        tool = Tool(
            function=collect,
            name="collect",
            namespace="tests",
            description="Collect varargs.",
        )

        with pytest.raises(TypeError, match="must be a list or tuple"):
            tool.invoke({"a": 1, "items": payload})

    def test_explicit_varkwargs_payload_merges_into_kwargs(self) -> None:
        tool = Tool(
            function=collect_kwargs,
            name="collect_kwargs",
            namespace="tests",
            description="Collect kwargs.",
        )

        args, kwargs = tool.to_arg_kwarg({"a": 1, "extras": {"debug": True}})

        assert args == ()
        assert kwargs == {"a": 1, "debug": True}
        assert tool.invoke({"a": 1, "extras": {"debug": True}}) == {
            "a": 1,
            "debug": True,
        }

    def test_explicit_varkwargs_payload_must_be_mapping(self) -> None:
        tool = Tool(
            function=collect_kwargs,
            name="collect_kwargs",
            namespace="tests",
            description="Collect kwargs.",
        )

        with pytest.raises(TypeError, match="must be a mapping"):
            tool.invoke({"a": 1, "extras": ["bad"]})

    def test_explicit_varkwargs_duplicate_key_raises(self) -> None:
        tool = Tool(
            function=collect_kwargs,
            name="collect_kwargs",
            namespace="tests",
            description="Collect kwargs.",
        )

        with pytest.raises(ToolInvocationError, match="duplicate key"):
            tool.invoke({"a": 1, "extras": {"a": 99}})

    def test_unknown_inputs_are_collected_into_varkwargs_by_filter_inputs(self) -> None:
        tool = Tool(
            function=collect_kwargs,
            name="collect_kwargs",
            namespace="tests",
            description="Collect kwargs.",
            filter_extraneous_inputs=False,
        )

        assert tool.invoke({"a": 1, "debug": True}) == {
            "a": 1,
            "debug": True,
        }


class TestToolExecution:
    def test_invoke_wraps_callable_exception(self) -> None:
        tool = Tool(
            function=fail,
            name="fail",
            namespace="tests",
            description="Failing tool.",
        )

        with pytest.raises(ToolInvocationError, match="invocation failed"):
            tool.invoke({})

    def test_async_invoke_works_for_sync_callable(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        result = asyncio.run(tool.async_invoke({"a": 2, "b": 3}))

        assert result == 5

    def test_async_invoke_awaits_async_callable(self) -> None:
        tool = Tool(
            function=async_add,
            name="async_add",
            namespace="tests",
            description="Async add values.",
        )

        result = asyncio.run(tool.async_invoke({"a": 2, "b": 3}))

        assert result == 5

    def test_async_invoke_wraps_async_callable_exception(self) -> None:
        tool = Tool(
            function=async_fail,
            name="async_fail",
            namespace="tests",
            description="Async failing tool.",
        )

        with pytest.raises(ToolInvocationError, match="invocation failed"):
            asyncio.run(tool.async_invoke({}))


class TestToolMutableMetadata:
    def test_namespace_setter_updates_full_name(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        tool.namespace = "math"

        assert tool.namespace == "math"
        assert tool.full_name == "Tool.math.add"
