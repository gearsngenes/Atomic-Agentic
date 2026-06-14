from __future__ import annotations

import asyncio
from typing import Any, Mapping

import pytest

from atomic_agentic.core.Exceptions import ToolDefinitionError, ToolInvocationError
from atomic_agentic.core.Invokable import AtomicInvokable
from atomic_agentic.core.Parameters import ParamSpec
from atomic_agentic.core.constants import NO_VAL
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


class AddInvokable(AtomicInvokable):
    """Small AtomicInvokable test double for Tool wrapping."""

    def __init__(
        self,
        *,
        name: str = "add_invokable",
        description: str = "Add invokable.",
        filter_extraneous_inputs: bool = True,
    ) -> None:
        self.last_inputs: dict[str, Any] | None = None
        super().__init__(
            name=name,
            description=description,
            parameters=[
                make_param("a", 0, type_="int"),
                make_param("b", 1, type_="int", default=0),
            ],
            return_type="int",
            filter_extraneous_inputs=filter_extraneous_inputs,
        )

    def invoke(self, inputs: Mapping[str, Any]) -> int:
        filtered = self.filter_inputs(inputs)
        self.last_inputs = dict(filtered)
        return int(filtered["a"]) + int(filtered.get("b", 0))


class AsyncTrackingInvokable(AddInvokable):
    """Invokable that records whether Tool.async_execute used async_invoke()."""

    def __init__(self, *, fail_async: bool = False) -> None:
        self.async_invoke_used = False
        self.async_call_used = False
        self.call_used = False
        self.fail_async = fail_async
        super().__init__(
            name="async_tracking_invokable",
            description="Async tracking invokable.",
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_used = True
        return super().__call__(*args, **kwargs)

    async def async_call(self, *args: Any, **kwargs: Any) -> Any:
        self.async_call_used = True
        return await super().async_call(*args, **kwargs)

    async def async_invoke(self, inputs: Mapping[str, Any]) -> Any:
        self.async_invoke_used = True
        if self.fail_async:
            raise RuntimeError("async invokable boom")
        return await super().async_invoke(inputs)


class BadReturnTypeInvokable(AddInvokable):
    """Invokable whose return_type property reports a non-str value."""

    @property
    def return_type(self) -> Any:
        return 123


class VariadicInvokable(AtomicInvokable):
    """AtomicInvokable test double exposing a variadic parameter spec."""

    def __init__(self) -> None:
        super().__init__(
            name="variadic_invokable",
            description="Invokable with a variadic keyword parameter.",
            parameters=[
                make_param("a", 0, type_="int"),
                make_param("extras", 1, kind=ParamSpec.VAR_KEYWORD, type_="dict"),
            ],
            return_type="int",
        )

    def invoke(self, inputs: Mapping[str, Any]) -> int:
        filtered = self.filter_inputs(inputs)
        return int(filtered["a"])


class RaisingToolInvocationErrorInvokable(AtomicInvokable):
    """Invokable whose invoke/async_invoke raise ToolInvocationError directly."""

    def __init__(self) -> None:
        super().__init__(
            name="raises_tool_invocation_error",
            description="Invokable that raises ToolInvocationError directly.",
            parameters=[],
            return_type="None",
        )

    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        raise ToolInvocationError("deliberate invocation error")

    async def async_invoke(self, inputs: Mapping[str, Any]) -> Any:
        raise ToolInvocationError("deliberate async invocation error")


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
        assert tool.wraps_invokable is False
        assert tool.module == add.__module__
        assert tool.qualname == add.__qualname__

    def test_default_namespace_and_inferred_metadata(self) -> None:
        tool = Tool(function=add)

        assert tool.name == "add"
        assert tool.namespace == "default"
        assert tool.description == "Add two integers."
        assert tool.full_name == "Tool.default.add"
        assert tool.wraps_invokable is False

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
        assert data["wraps_invokable"] is False
        assert "invokable_function" not in data


class TestToolInvokableBacking:
    def test_invokable_construction_uses_invokable_metadata_by_default(self) -> None:
        invokable = AddInvokable()
        tool = Tool(function=invokable)

        assert tool.name == invokable.name
        assert tool.namespace == "default"
        assert tool.description == invokable.description
        assert tool.full_name == f"Tool.default.{invokable.name}"
        assert tool.function is invokable
        assert tool.wraps_invokable is True

    def test_invokable_construction_allows_name_and_description_overrides(self) -> None:
        invokable = AddInvokable()
        tool = Tool(
            function=invokable,
            name="wrapped_add",
            description="Wrapped add invokable.",
        )

        assert tool.name == "wrapped_add"
        assert tool.description == "Wrapped add invokable."
        assert tool.wraps_invokable is True

    def test_invokable_construction_treats_blank_metadata_as_omitted(self) -> None:
        invokable = AddInvokable()
        tool = Tool(function=invokable, name="", description="   ")

        assert tool.name == invokable.name
        assert tool.description == invokable.description

    def test_invokable_tool_reuses_declared_schema(self) -> None:
        invokable = AddInvokable()
        tool = Tool(function=invokable)

        assert [(p.name, p.kind, p.default) for p in tool.parameters] == [
            ("a", ParamSpec.POSITIONAL_OR_KEYWORD, NO_VAL),
            ("b", ParamSpec.POSITIONAL_OR_KEYWORD, 0),
        ]
        assert tool.return_type == invokable.return_type
        assert "a" in tool.signature
        assert "b" in tool.signature
        assert "inputs" not in tool.signature

    def test_invokable_tool_sync_invoke_uses_dict_first_path(self) -> None:
        invokable = AddInvokable()
        tool = Tool(function=invokable)

        assert tool.invoke({"a": 2, "b": 3}).result == 5
        assert invokable.last_inputs == {"a": 2, "b": 3}

    def test_invokable_tool_applies_defaults_before_dict_first_invoke(self) -> None:
        invokable = AddInvokable()
        tool = Tool(function=invokable)

        assert tool.invoke({"a": 7}).result == 7
        assert invokable.last_inputs == {"a": 7, "b": 0}

    def test_invokable_tool_module_qualname_come_from_invoke_method(self) -> None:
        invokable = AddInvokable()
        tool = Tool(function=invokable)

        assert tool.module == invokable.invoke.__module__
        assert tool.qualname == invokable.invoke.__qualname__

    def test_to_dict_includes_invokable_function_when_wrapping_invokable(self) -> None:
        invokable = AddInvokable()
        tool = Tool(
            function=invokable,
            name="wrapped_add",
            namespace="tests",
            description="Wrapped add invokable.",
        )

        data = tool.to_dict()

        assert data["type"] == "Tool"
        assert data["name"] == "wrapped_add"
        assert data["namespace"] == "tests"
        assert data["return_type"] == "int"
        assert data["wraps_invokable"] is True
        assert data["invokable_function"] == invokable.to_dict()


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

    def test_unknown_input_raises_when_filtering_disabled(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
            filter_extraneous_inputs=False,
        )

        with pytest.raises(TypeError, match="unexpected input key"):
            tool.invoke({"a": 1, "b": 2, "extra": 3})

    def test_filter_extraneous_inputs_true_filters_unknown_inputs(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
            filter_extraneous_inputs=True,
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
            filter_extraneous_inputs=False,
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

    def test_async_invoke_uses_invokable_async_invoke(self) -> None:
        invokable = AsyncTrackingInvokable()
        tool = Tool(function=invokable)

        result = asyncio.run(tool.async_invoke({"a": 2, "b": 3}))

        assert result.result == 5
        assert invokable.async_invoke_used is True
        assert invokable.async_call_used is False
        assert invokable.call_used is False

    def test_async_invoke_wraps_invokable_async_invoke_exception(self) -> None:
        invokable = AsyncTrackingInvokable(fail_async=True)
        tool = Tool(function=invokable)

        with pytest.raises(ToolInvocationError, match="invocation failed"):
            asyncio.run(tool.async_invoke({"a": 2, "b": 3}))

        assert invokable.async_invoke_used is True
        assert invokable.async_call_used is False
        assert invokable.call_used is False


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

    def test_function_setter_refreshes_callable_schema(self) -> None:
        tool = Tool(
            function=add,
            name="stable_name",
            namespace="tests",
            description="Stable description.",
        )

        tool.function = greet

        assert tool.function is greet
        assert tool.wraps_invokable is False
        assert tool.name == "stable_name"
        assert tool.description == "Stable description."
        assert [(p.name, p.default) for p in tool.parameters] == [
            ("name", NO_VAL),
            ("punctuation", "!"),
        ]
        assert tool.return_type == "str"
        assert tool.invoke({"name": "Ada"}).result == "Hello, Ada!"

    def test_function_setter_switches_callable_to_invokable(self) -> None:
        tool = Tool(
            function=add,
            name="stable_name",
            namespace="tests",
            description="Stable description.",
        )
        invokable = AddInvokable()

        tool.function = invokable

        assert tool.function is invokable
        assert tool.wraps_invokable is True
        assert tool.name == "stable_name"
        assert tool.description == "Stable description."
        assert [p.name for p in tool.parameters] == ["a", "b"]
        assert tool.return_type == "int"
        assert tool.invoke({"a": 4, "b": 6}).result == 10

    def test_function_setter_switches_invokable_to_callable(self) -> None:
        invokable = AddInvokable()
        tool = Tool(function=invokable)

        tool.function = greet

        assert tool.function is greet
        assert tool.wraps_invokable is False
        assert tool.name == invokable.name
        assert tool.description == invokable.description
        assert [(p.name, p.default) for p in tool.parameters] == [
            ("name", NO_VAL),
            ("punctuation", "!"),
        ]
        assert tool.return_type == "str"
        assert tool.invoke({"name": "Ada", "punctuation": "."}).result == "Hello, Ada."

    def test_function_setter_rejects_non_callable_without_mutating_state(self) -> None:
        tool = Tool(
            function=add,
            name="add",
            namespace="tests",
            description="Add values.",
        )

        old_function = tool.function
        old_parameters = tool.parameters
        old_return_type = tool.return_type
        old_module = tool.module
        old_qualname = tool.qualname
        old_wraps_invokable = tool.wraps_invokable

        with pytest.raises(ToolDefinitionError, match="callable"):
            tool.function = 123  # type: ignore[assignment]

        assert tool.function is old_function
        assert tool.parameters == old_parameters
        assert tool.return_type == old_return_type
        assert tool.module == old_module
        assert tool.qualname == old_qualname
        assert tool.wraps_invokable is old_wraps_invokable


class TestToolExecutionEdgeCases:
    def test_function_setter_rejects_non_string_return_type_from_invokable(self) -> None:
        tool = Tool(function=AddInvokable())

        with pytest.raises(TypeError, match="return_type must be str"):
            tool.function = BadReturnTypeInvokable()

    def test_invokable_dict_first_binding_skips_variadic_parameter_defaults(self) -> None:
        invokable = VariadicInvokable()
        tool = Tool(function=invokable)

        args, kwargs = tool.to_arg_kwarg({"a": 1})

        assert args == ()
        assert kwargs == {"a": 1}
        assert "extras" not in kwargs

    def test_sync_execute_runs_awaitable_result_from_sync_callable(self) -> None:
        tool = Tool(
            function=returns_coroutine,
            name="returns_coroutine",
            namespace="tests",
            description="Sync callable returning a coroutine.",
        )

        assert tool.invoke({"a": 2, "b": 3}).result == 5

    def test_sync_execute_reraises_tool_invocation_error_without_rewrapping(self) -> None:
        invokable = RaisingToolInvocationErrorInvokable()
        tool = Tool(function=invokable)

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
        invokable = RaisingToolInvocationErrorInvokable()
        tool = Tool(function=invokable)

        with pytest.raises(ToolInvocationError) as exc_info:
            asyncio.run(tool.async_invoke({}))

        assert str(exc_info.value) == "deliberate async invocation error"
