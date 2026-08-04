from __future__ import annotations

import asyncio
from typing import Any

import pytest

from atomic_agentic.exceptions import ToolDefinitionError, ToolInvocationError
from atomic_agentic.core.Invokable import AtomicInvokable, StructuredInvokable
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.tools.base import Tool


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def add_no_doc(a: int, b: int) -> int:
    return a + b


add_no_doc.__doc__ = None


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


def raise_tool_invocation_error() -> None:
    raise ToolInvocationError("deliberate invocation error")


async def async_raise_tool_invocation_error() -> None:
    raise ToolInvocationError("deliberate async invocation error")


async def async_add(a: int, b: int) -> int:
    return a + b


async def async_fail() -> None:
    raise RuntimeError("async boom")


def returns_coroutine(a: int, b: int) -> Any:
    """Plain sync callable that hands the sync path an awaitable to drive."""

    async def _add() -> int:
        return a + b

    return _add()


async def returns_awaitable(a: int, b: int) -> Any:
    """Async callable whose awaited result is itself awaitable."""

    async def _add() -> int:
        return a + b

    return _add()


def make_param(
    name: str,
    index: int,
    kind: str = ParamSpec.POSITIONAL_OR_KEYWORD,
    *,
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

    def test_callable_without_docstring_uses_fallback_description(self) -> None:
        tool = Tool(function=add_no_doc)

        assert tool.name == "add_no_doc"
        assert tool.description == "No description available."

    def test_blank_callable_name_and_description_use_fallbacks(self) -> None:
        tool = Tool(function=add, name="   ", description="")

        assert tool.name == "add"
        assert tool.description == "Add two integers."

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


class TestToolExtraDescription:
    def test_plain_callable_backed_tool_has_no_extra_description(self) -> None:
        tool = Tool(function=add, name="add", namespace="tests")

        assert tool._extra_description() == ""
        assert tool.description == tool._description

    def test_invokable_backed_tool_chains_wrapped_extra_description(self) -> None:
        inner = Tool(function=add, name="add", namespace="tests")
        structured = StructuredInvokable(component=inner, output_schema=["result"])
        tool = Tool(function=structured, name="structured_add", namespace="tests")

        assert tool._extra_description() == "Output schema: [result]"



class TestToolBindingBasic:
    def test_required_arguments_bind_and_execute(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        assert tool.invoke({"a": 2, "b": 3}).result == 5

    def test_default_values_are_applied(self) -> None:
        tool = Tool(
            function=greet,
            name="greet",
            namespace="tests",
            description="Greet.",
        )

        assert tool.invoke({"name": "Ada"}).result == "Hello, Ada!"
        assert tool.invoke({"name": "Ada", "punctuation": "."}).result == "Hello, Ada."

    def test_keyword_only_parameters_bind_correctly(self) -> None:
        tool = Tool(
            function=keyword_only,
            name="keyword_only",
            namespace="tests",
            description="Keyword-only text tool.",
        )

        assert tool.invoke({"text": "hello"}).result == "hello"
        assert tool.invoke({"text": "hello", "upper": True}).result == "HELLO"

    def test_missing_required_parameter_raises(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        with pytest.raises(TypeError, match="missing required"):
            tool.invoke({"a": 1})

    def test_unknown_input_is_filtered(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        assert tool.invoke({"a": 1, "b": 2, "extra": 3}).result == 3


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
        assert tool.invoke({"a": 5, "b": 2}).result == 3

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
        assert tool.invoke({"a": 1, "items": [2, 3, 4]}).result == (1, (2, 3, 4))

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
        assert tool.invoke({"a": 1, "extras": {"debug": True}}).result == {
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

        with pytest.raises(TypeError, match="would overwrite bound keyword"):
            tool.invoke({"a": 1, "extras": {"a": 99}})

    def test_unknown_inputs_are_collected_into_varkwargs_by_filter_inputs(self) -> None:
        tool = Tool(
            function=collect_kwargs,
            name="collect_kwargs",
            namespace="tests",
            description="Collect kwargs.",
        )

        assert tool.invoke({"a": 1, "debug": True}).result == {
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

        assert result.result == 5

    def test_async_invoke_awaits_async_callable(self) -> None:
        tool = Tool(
            function=async_add,
            name="async_add",
            namespace="tests",
            description="Async add values.",
        )

        result = asyncio.run(tool.async_invoke({"a": 2, "b": 3}))

        assert result.result == 5

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
    def test_namespace_is_read_only(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        assert tool.namespace == "tests"
        assert tool.full_name == "Tool.tests.add"
        with pytest.raises(AttributeError):
            tool.namespace = "math"

class TestToolExecutionEdgeCases:
    def test_sync_execute_runs_awaitable_result_from_sync_callable(self) -> None:
        tool = Tool(
            function=returns_coroutine,
            name="returns_coroutine",
            namespace="tests",
            description="Sync callable returning a coroutine.",
        )

        assert tool.invoke({"a": 2, "b": 3}).result == 5

    def test_sync_execute_reraises_tool_invocation_error_without_rewrapping(self) -> None:
        tool = Tool(
            function=raise_tool_invocation_error,
            name="raise_tool_invocation_error",
            namespace="tests",
            description="Sync callable that raises ToolInvocationError directly.",
        )

        with pytest.raises(ToolInvocationError) as exc_info:
            tool.invoke({})

        assert str(exc_info.value) == "deliberate invocation error"

    def test_async_execute_awaits_doubly_awaitable_result(self) -> None:
        tool = Tool(
            function=returns_awaitable,
            name="returns_awaitable",
            namespace="tests",
            description="Async callable returning an awaitable.",
        )

        result = asyncio.run(tool.async_invoke({"a": 2, "b": 3}))

        assert result.result == 5

    def test_async_execute_reraises_tool_invocation_error_without_rewrapping(self) -> None:
        tool = Tool(
            function=async_raise_tool_invocation_error,
            name="async_raise_tool_invocation_error",
            namespace="tests",
            description="Async callable that raises ToolInvocationError directly.",
        )

        with pytest.raises(ToolInvocationError) as exc_info:
            asyncio.run(tool.async_invoke({}))

        assert str(exc_info.value) == "deliberate async invocation error"


class AddInvokable(AtomicInvokable):
    """Small AtomicInvokable test double for natural-callable Tool wrapping."""

    def __init__(self, *, name: str = "add_invokable", namespace: str = "tests") -> None:
        super().__init__(
            name=name,
            namespace=namespace,
            description="Add invokable.",
            parameters=[
                make_param("a", 0, type_="int"),
                make_param("b", 1, type_="int", default=0),
            ],
            return_type="int",
        )

    def invoke(self, inputs):
        filtered = self.filter_inputs(inputs)
        return int(filtered["a"]) + int(filtered.get("b", 0))

    async def async_invoke(self, inputs):
        return self.invoke(inputs)


class TestToolNaturalAtomicInvokableWrapping:
    def test_construction_infers_name_and_description_from_invokable(self) -> None:
        invokable = AddInvokable(name="add_invokable")
        tool = Tool(function=invokable)

        assert tool.name == "add_invokable"
        assert tool.description == "Add invokable."
        assert tool.function is invokable

    def test_schema_reuses_invokable_declared_parameters(self) -> None:
        tool = Tool(function=AddInvokable())

        assert [(p.name, p.kind, p.default) for p in tool.parameters] == [
            ("a", ParamSpec.POSITIONAL_OR_KEYWORD, NO_VAL),
            ("b", ParamSpec.POSITIONAL_OR_KEYWORD, 0),
        ]
        assert tool.return_type == "int"

    def test_sync_invoke_returns_unwrapped_result(self) -> None:
        tool = Tool(function=AddInvokable())

        assert tool.invoke({"a": 2, "b": 3}).result == 5

    def test_async_invoke_dispatches_through_invokable_async_call(self) -> None:
        tool = Tool(function=AddInvokable())

        result = asyncio.run(tool.async_invoke({"a": 2, "b": 3}))

        assert result.result == 5

    def test_explicit_name_and_description_override_invokable_inference(self) -> None:
        tool = Tool(function=AddInvokable(), name="custom", description="Custom.")

        assert tool.name == "custom"
        assert tool.description == "Custom."
