from __future__ import annotations

import asyncio
from collections import namedtuple
from collections.abc import Mapping
from typing import Any

import pytest

from atomic_agentic.core.Exceptions import PackagingError
from atomic_agentic.core.Invokable import AtomicInvokable
from atomic_agentic.core.Parameters import ParamSpec
from atomic_agentic.core.sentinels import NO_VAL
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.StructuredInvokable import (
    StructuredInvokable,
    StructuredResultDict,
)


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


def echo_value(value: Any) -> Any:
    """Return the provided value."""
    return value


def return_mapping(value: Any = 1) -> dict[str, Any]:
    """Return a mapping."""
    return {"a": value, "b": 2}


def return_sequence() -> list[int]:
    """Return a sequence."""
    return [1, 2, 3]


def return_scalar() -> int:
    """Return a scalar."""
    return 42


def return_none_mapping() -> dict[str, Any]:
    """Return a mapping with None."""
    return {"a": None, "b": 2}


def dummy_component() -> Tool:
    return Tool(
        function=echo_value,
        name="echo_value",
        namespace="tests",
        description="Echo test value.",
        filter_extraneous_inputs=True,
    )


def structured(
    *,
    output_schema: Any = None,
    component: AtomicInvokable | None = None,
    **kwargs: Any,
) -> StructuredInvokable:
    return StructuredInvokable(
        component=component or dummy_component(),
        output_schema=output_schema,
        **kwargs,
    )


class RecordingInvokable(AtomicInvokable):
    def __init__(
        self,
        *,
        raw_result: Any,
        name: str = "recording_invokable",
        description: str = "Recording invokable.",
        filter_extraneous_inputs: bool = True,
        raise_sync: Exception | None = None,
        raise_async: Exception | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            parameters=[
                make_param("value", 0, type_="Any"),
            ],
            return_type="Any",
            filter_extraneous_inputs=filter_extraneous_inputs,
        )
        self.raw_result = raw_result
        self.raise_sync = raise_sync
        self.raise_async = raise_async
        self.calls: list[dict[str, Any]] = []
        self.async_calls: list[dict[str, Any]] = []

    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        self.calls.append(dict(inputs))
        if self.raise_sync is not None:
            raise self.raise_sync
        return self.raw_result

    async def async_invoke(self, inputs: Mapping[str, Any]) -> Any:
        self.async_calls.append(dict(inputs))
        if self.raise_async is not None:
            raise self.raise_async
        return self.raw_result


class ModelDumpObject:
    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return {"a": 1, "b": 2, "mode": mode}


class ModelDumpNoModeObject:
    def model_dump(self) -> dict[str, Any]:
        return {"a": 10, "b": 20}


class PlainObject:
    def __init__(self) -> None:
        self.a = 1
        self.b = 2
        self._private = "hidden"


class UnsupportedObject:
    __slots__ = ()


class TestStructuredResultDict:
    def test_copy_preserves_mapping_items_and_raw_result(self) -> None:
        result = StructuredResultDict({"a": 1}, raw_result={"raw": True})

        copied = result.copy()

        assert copied == {"a": 1}
        assert copied.raw_result == {"raw": True}
        assert copied is not result

    def test_raw_result_is_not_mapping_item(self) -> None:
        result = StructuredResultDict({"a": 1}, raw_result={"raw": True})

        assert "raw_result" not in result
        assert result.raw_result == {"raw": True}


class TestStructuredInvokableConstruction:
    def test_requires_atomic_invokable_component(self) -> None:
        with pytest.raises(TypeError, match="AtomicInvokable"):
            StructuredInvokable(component=object())  # type: ignore[arg-type]

    def test_inherits_component_identity_parameters_and_filter_flag(self) -> None:
        component = Tool(
            function=return_mapping,
            name="return_mapping",
            namespace="tests",
            description="Return a mapping.",
            filter_extraneous_inputs=False,
        )

        wrapper = StructuredInvokable(component=component, output_schema=["a"])

        assert wrapper.component is component
        assert wrapper.name == component.name
        assert "Return a mapping." in wrapper.description
        assert wrapper.parameters == component.parameters
        assert wrapper.filter_extraneous_inputs is False
        assert wrapper.return_type == "StructuredResultDict[str, Any]"

    def test_explicit_identity_and_filter_override_component(self) -> None:
        component = Tool(
            function=return_mapping,
            name="return_mapping",
            namespace="tests",
            description="Return a mapping.",
            filter_extraneous_inputs=False,
        )

        wrapper = StructuredInvokable(
            component=component,
            name="structured_mapping",
            description="Structured mapping.",
            output_schema=["a"],
            filter_extraneous_inputs=True,
        )

        assert wrapper.name == "structured_mapping"
        assert "Structured mapping." in wrapper.description
        assert wrapper.filter_extraneous_inputs is True

    def test_output_schema_normalizes_string_list_and_properties(self) -> None:
        wrapper = structured(output_schema=["a", "b"])

        assert [spec.name for spec in wrapper.output_schema] == ["a", "b"]
        assert [spec.name for spec in wrapper.named_output_fields] == ["a", "b"]
        assert wrapper.output_vararg is None
        assert wrapper.output_varkwarg is None
        assert wrapper.output_has_varargs is False
        assert wrapper.output_has_varkwargs is False

    def test_output_schema_with_variadic_sinks_exposes_sink_names(self) -> None:
        schema = [
            make_param("a", 0),
            make_param("rest", 1, kind=ParamSpec.VAR_POSITIONAL),
            make_param("extras", 2, kind=ParamSpec.VAR_KEYWORD),
        ]

        wrapper = structured(output_schema=schema)

        assert wrapper.output_vararg == "rest"
        assert wrapper.output_varkwarg == "extras"
        assert wrapper.output_has_varargs is True
        assert wrapper.output_has_varkwargs is True

    def test_none_output_schema_packages_to_empty_mapping(self) -> None:
        wrapper = structured(output_schema=None)

        assert wrapper.output_schema == []
        assert wrapper.package({"a": 1}) == {}
        assert "Output schema: [<empty>]" in wrapper.description

    def test_to_dict_includes_component_schema_and_policy_knobs(self) -> None:
        wrapper = structured(
            output_schema=["a"],
            map_single_fields=False,
            map_extras=False,
            ignore_unhandled=True,
            absent_value_mode=StructuredInvokable.FILL,
            default_absent_value="missing",
            none_is_absent=True,
            coerce_to_collection=True,
        )

        data = wrapper.to_dict()

        assert data["type"] == "StructuredInvokable"
        assert data["component"]["type"] == "Tool"
        assert data["output_schema"][0]["name"] == "a"
        assert data["map_single_fields"] is False
        assert data["map_extras"] is False
        assert data["ignore_unhandled"] is True
        assert data["absent_value_mode"] == "FILL"
        assert data["default_absent_value"] == "missing"
        assert data["none_is_absent"] is True
        assert data["coerce_to_collection"] is True


class TestStructuredInvokablePolicyValidation:
    @pytest.mark.parametrize(
        "attr",
        [
            "map_single_fields",
            "map_extras",
            "ignore_unhandled",
            "none_is_absent",
            "coerce_to_collection",
        ],
    )
    def test_bool_policy_setters_reject_non_bool(self, attr: str) -> None:
        wrapper = structured(output_schema=["a"])

        with pytest.raises(TypeError, match=attr):
            setattr(wrapper, attr, "yes")

    def test_description_rejects_non_string(self) -> None:
        wrapper = structured(output_schema=["a"])

        with pytest.raises(TypeError, match="description"):
            wrapper.description = 123  # type: ignore[assignment]

    def test_description_rejects_empty_string(self) -> None:
        wrapper = structured(output_schema=["a"])

        with pytest.raises(ValueError, match="description cannot be empty"):
            wrapper.description = "   "

    @pytest.mark.parametrize("mode", ["raise", "RAISE", "drop", "DROP", "fill", "FILL"])
    def test_absent_value_mode_accepts_valid_modes_case_insensitively(
        self,
        mode: str,
    ) -> None:
        wrapper = structured(output_schema=["a"])

        wrapper.absent_value_mode = mode

        assert wrapper.absent_value_mode == mode.upper()

    def test_absent_value_mode_rejects_invalid_string(self) -> None:
        wrapper = structured(output_schema=["a"])

        with pytest.raises(ValueError, match="absent_value_mode"):
            wrapper.absent_value_mode = "ignore"

    def test_absent_value_mode_rejects_non_string(self) -> None:
        wrapper = structured(output_schema=["a"])

        with pytest.raises(TypeError, match="absent_value_mode"):
            wrapper.absent_value_mode = 123  # type: ignore[assignment]


class TestStructuredInvokableMappingPackaging:
    def test_mapping_exact_fields_are_copied(self) -> None:
        wrapper = structured(output_schema=["a", "b"])

        assert wrapper.package({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_mapping_uses_schema_defaults_for_missing_fields(self) -> None:
        schema = [
            make_param("a", 0),
            make_param("b", 1, default=99),
        ]
        wrapper = structured(output_schema=schema)

        assert wrapper.package({"a": 1}) == {"a": 1, "b": 99}

    def test_mapping_extras_raise_without_varkwarg_sink(self) -> None:
        wrapper = structured(output_schema=["a"])

        with pytest.raises(PackagingError, match="unhandled mapping extras"):
            wrapper.package({"a": 1, "extra": 2})

    def test_mapping_extras_drop_when_ignore_unhandled_true(self) -> None:
        wrapper = structured(output_schema=["a"], ignore_unhandled=True)

        assert wrapper.package({"a": 1, "extra": 2}) == {"a": 1}

    def test_mapping_extras_route_to_varkwarg_sink(self) -> None:
        schema = [
            make_param("a", 0),
            make_param("extras", 1, kind=ParamSpec.VAR_KEYWORD),
        ]
        wrapper = structured(output_schema=schema)

        assert wrapper.package({"a": 1, "x": 2, "y": 3}) == {
            "a": 1,
            "extras": {"x": 2, "y": 3},
        }

    def test_map_extras_backfills_missing_named_fields(self) -> None:
        wrapper = structured(output_schema=["a", "b"], ignore_unhandled=True)

        assert wrapper.package({"a": 1, "extra": 2}) == {"a": 1, "b": 2}

    def test_map_extras_false_leaves_missing_field_unresolved(self) -> None:
        wrapper = structured(
            output_schema=["a", "b"],
            map_extras=False,
            ignore_unhandled=True,
        )

        assert wrapper.package({"a": 1, "extra": 2}) == {"a": 1, "b": NO_VAL}

    def test_explicit_vararg_payload_is_peeled_from_mapping(self) -> None:
        schema = [
            make_param("a", 0),
            make_param("rest", 1, kind=ParamSpec.VAR_POSITIONAL),
        ]
        wrapper = structured(output_schema=schema)

        assert wrapper.package({"a": 1, "rest": [2, 3]}) == {
            "a": 1,
            "rest": (2, 3),
        }

    def test_explicit_varkwarg_payload_is_peeled_from_mapping(self) -> None:
        schema = [
            make_param("a", 0),
            make_param("extras", 1, kind=ParamSpec.VAR_KEYWORD),
        ]
        wrapper = structured(output_schema=schema)

        assert wrapper.package({"a": 1, "extras": {"x": 2}}) == {
            "a": 1,
            "extras": {"x": 2},
        }

    def test_explicit_vararg_payload_must_be_list_or_tuple(self) -> None:
        schema = [
            make_param("a", 0),
            make_param("rest", 1, kind=ParamSpec.VAR_POSITIONAL),
        ]
        wrapper = structured(output_schema=schema)

        with pytest.raises(ValueError, match="output vararg"):
            wrapper.package({"a": 1, "rest": "bad"})

    def test_explicit_varkwarg_payload_must_be_mapping(self) -> None:
        schema = [
            make_param("a", 0),
            make_param("extras", 1, kind=ParamSpec.VAR_KEYWORD),
        ]
        wrapper = structured(output_schema=schema)

        with pytest.raises(ValueError, match="output varkwarg"):
            wrapper.package({"a": 1, "extras": ["bad"]})

    def test_explicit_varkwarg_overlapping_with_leftover_extras_raises(self) -> None:
        schema = [
            make_param("a", 0),
            make_param("extras", 1, kind=ParamSpec.VAR_KEYWORD),
        ]
        wrapper = structured(output_schema=schema, map_extras=False)

        with pytest.raises(PackagingError, match="overlapping keys"):
            wrapper.package({"a": 1, "extras": {"x": 2}, "x": 3})


class TestStructuredInvokableSequencePackaging:
    def test_sequence_fills_named_fields_by_position(self) -> None:
        wrapper = structured(output_schema=["a", "b"])

        assert wrapper.package([1, 2]) == {"a": 1, "b": 2}

    def test_sequence_extras_raise_without_vararg_sink(self) -> None:
        wrapper = structured(output_schema=["a"])

        with pytest.raises(PackagingError, match="unhandled positional extras"):
            wrapper.package([1, 2, 3])

    def test_sequence_extras_route_to_vararg_sink(self) -> None:
        schema = [
            make_param("a", 0),
            make_param("rest", 1, kind=ParamSpec.VAR_POSITIONAL),
        ]
        wrapper = structured(output_schema=schema)

        assert wrapper.package([1, 2, 3]) == {"a": 1, "rest": (2, 3)}

    def test_sequence_extras_drop_when_ignore_unhandled_true(self) -> None:
        wrapper = structured(output_schema=["a"], ignore_unhandled=True)

        assert wrapper.package([1, 2, 3]) == {"a": 1}

    @pytest.mark.parametrize("raw", ["abc", b"abc", bytearray(b"abc")])
    def test_string_like_values_are_treated_as_scalars(self, raw: Any) -> None:
        wrapper = structured(output_schema=["value"])

        assert wrapper.package(raw) == {"value": raw}


class TestStructuredInvokableScalarPackaging:
    def test_scalar_maps_to_single_named_field(self) -> None:
        wrapper = structured(output_schema=["value"])

        assert wrapper.package(42) == {"value": 42}

    def test_scalar_maps_to_only_missing_named_field(self) -> None:
        schema = [
            make_param("a", 0),
            make_param("b", 1, default=1),
        ]
        wrapper = structured(output_schema=schema)

        assert wrapper.package(42) == {"a": 42, "b": 1}

    def test_scalar_with_multiple_missing_fields_raises(self) -> None:
        wrapper = structured(output_schema=["a", "b"])

        with pytest.raises(PackagingError, match="too much ambiguity"):
            wrapper.package(42)

    def test_scalar_with_all_fields_default_filled_raises_unless_ignored(self) -> None:
        schema = [
            make_param("a", 0, default=1),
            make_param("b", 1, default=2),
        ]
        wrapper = structured(output_schema=schema)

        with pytest.raises(PackagingError, match="too much ambiguity"):
            wrapper.package(42)

    def test_scalar_with_all_fields_default_filled_can_be_ignored(self) -> None:
        schema = [
            make_param("a", 0, default=1),
            make_param("b", 1, default=2),
        ]
        wrapper = structured(output_schema=schema, ignore_unhandled=True)

        assert wrapper.package(42) == {"a": 1, "b": 2}

    def test_map_single_fields_false_stores_whole_mapping(self) -> None:
        wrapper = structured(output_schema=["payload"], map_single_fields=False)

        assert wrapper.package({"a": 1}) == {"payload": {"a": 1}}

    def test_map_single_fields_false_stores_whole_sequence(self) -> None:
        wrapper = structured(output_schema=["payload"], map_single_fields=False)

        assert wrapper.package([1, 2, 3]) == {"payload": [1, 2, 3]}


class TestStructuredInvokablePassthrough:
    def test_passthrough_returns_mapping_source_as_is(self) -> None:
        wrapper = structured(output_schema=StructuredInvokable.PASSTHROUGH)

        assert wrapper.package({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_passthrough_rejects_non_string_keys(self) -> None:
        wrapper = structured(output_schema=StructuredInvokable.PASSTHROUGH)

        with pytest.raises(PackagingError, match="string keys"):
            wrapper.package({1: "bad"})

    def test_passthrough_none_is_absent_converts_none_to_no_val(self) -> None:
        wrapper = structured(
            output_schema=StructuredInvokable.PASSTHROUGH,
            none_is_absent=True,
        )

        assert wrapper.package({"a": None, "b": 2}) == {"a": NO_VAL, "b": 2}

    def test_passthrough_rejects_non_mapping_raw_output(self) -> None:
        wrapper = structured(output_schema=StructuredInvokable.PASSTHROUGH)

        with pytest.raises(PackagingError, match="can only be used with mapping"):
            wrapper.package([1, 2, 3])

    def test_passthrough_then_missing_values_can_fill(self) -> None:
        wrapper = structured(
            output_schema=StructuredInvokable.PASSTHROUGH,
            none_is_absent=True,
            absent_value_mode=StructuredInvokable.FILL,
            default_absent_value="filled",
        )

        packaged = wrapper.package({"a": None})

        assert wrapper.handle_missing_values(packaged) == {"a": "filled"}


class TestStructuredInvokableObjectCoercion:
    def test_coerce_false_treats_object_as_scalar(self) -> None:
        obj = PlainObject()
        wrapper = structured(output_schema=["payload"], coerce_to_collection=False)

        assert wrapper.package(obj) == {"payload": obj}

    def test_model_dump_object_can_coerce_to_mapping(self) -> None:
        wrapper = structured(
            output_schema=["a", "b"],
            coerce_to_collection=True,
            ignore_unhandled=True,
        )

        assert wrapper.package(ModelDumpObject()) == {"a": 1, "b": 2}

    def test_model_dump_type_error_fallback_calls_without_mode(self) -> None:
        wrapper = structured(output_schema=["a", "b"], coerce_to_collection=True)

        assert wrapper.package(ModelDumpNoModeObject()) == {"a": 10, "b": 20}

    def test_asdict_object_can_coerce_to_mapping(self) -> None:
        Point = namedtuple("Point", ["a", "b"])
        wrapper = structured(output_schema=["a", "b"], coerce_to_collection=True)

        assert wrapper.package(Point(1, 2)) == {"a": 1, "b": 2}

    def test_plain_object_dict_can_coerce_to_mapping_and_ignores_private_attrs(self) -> None:
        wrapper = structured(output_schema=["a", "b"], coerce_to_collection=True)

        assert wrapper.package(PlainObject()) == {"a": 1, "b": 2}

    def test_unsupported_object_falls_back_to_scalar(self) -> None:
        obj = UnsupportedObject()
        wrapper = structured(output_schema=["payload"], coerce_to_collection=True)

        assert wrapper.package(obj) == {"payload": obj}


class TestStructuredInvokableMissingValues:
    def test_none_is_absent_converts_named_none_to_no_val(self) -> None:
        wrapper = structured(
            output_schema=["a", "b"],
            none_is_absent=True,
        )

        assert wrapper.package({"a": None, "b": 2}) == {"a": NO_VAL, "b": 2}

    def test_none_is_absent_filters_none_from_vararg_sink(self) -> None:
        schema = [
            make_param("a", 0),
            make_param("rest", 1, kind=ParamSpec.VAR_POSITIONAL),
        ]
        wrapper = structured(output_schema=schema, none_is_absent=True)

        assert wrapper.package([1, None, 3]) == {"a": 1, "rest": (3,)}

    def test_none_is_absent_filters_none_from_varkwarg_sink(self) -> None:
        schema = [
            make_param("a", 0),
            make_param("extras", 1, kind=ParamSpec.VAR_KEYWORD),
        ]
        wrapper = structured(output_schema=schema, none_is_absent=True)

        assert wrapper.package({"a": 1, "x": None, "y": 2}) == {
            "a": 1,
            "extras": {"y": 2},
        }

    def test_raise_mode_raises_for_missing_fields(self) -> None:
        wrapper = structured(output_schema=["a"])

        with pytest.raises(ValueError, match="missing required field"):
            wrapper.handle_missing_values({"a": NO_VAL})

    def test_drop_mode_removes_missing_fields(self) -> None:
        wrapper = structured(
            output_schema=["a", "b"],
            absent_value_mode=StructuredInvokable.DROP,
        )

        assert wrapper.handle_missing_values({"a": NO_VAL, "b": 2}) == {"b": 2}

    def test_fill_mode_fills_missing_fields(self) -> None:
        wrapper = structured(
            output_schema=["a", "b"],
            absent_value_mode=StructuredInvokable.FILL,
            default_absent_value="filled",
        )

        assert wrapper.handle_missing_values({"a": NO_VAL, "b": 2}) == {
            "a": "filled",
            "b": 2,
        }

    def test_handle_missing_values_without_missing_returns_plain_dict_copy(self) -> None:
        wrapper = structured(output_schema=["a"])
        source = {"a": 1}

        result = wrapper.handle_missing_values(source)

        assert result == {"a": 1}
        assert result is not source


class TestStructuredInvokableInvoke:
    def test_invoke_returns_structured_result_and_preserves_raw_result(self) -> None:
        component = Tool(
            function=return_mapping,
            name="return_mapping",
            namespace="tests",
            description="Return a mapping.",
        )
        wrapper = structured(component=component, output_schema=["a", "b"])

        result = wrapper.invoke({"value": 10})

        assert isinstance(result, StructuredResultDict)
        assert result == {"a": 10, "b": 2}
        assert result.raw_result == {"a": 10, "b": 2}

    def test_invoke_applies_missing_value_handling(self) -> None:
        component = Tool(
            function=return_none_mapping,
            name="return_none_mapping",
            namespace="tests",
            description="Return none mapping.",
        )
        wrapper = structured(
            component=component,
            output_schema=["a", "b"],
            none_is_absent=True,
            absent_value_mode=StructuredInvokable.FILL,
            default_absent_value="filled",
        )

        result = wrapper.invoke({})

        assert result == {"a": "filled", "b": 2}
        assert result.raw_result == {"a": None, "b": 2}

    def test_invoke_filters_inputs_before_calling_component(self) -> None:
        component = RecordingInvokable(raw_result={"value": 123})
        wrapper = structured(component=component, output_schema=["value"])

        result = wrapper.invoke({"value": 123, "extra": "ignored"})

        assert result == {"value": 123}
        assert component.calls == [{"value": 123}]

    def test_async_invoke_mirrors_sync_contract(self) -> None:
        component = RecordingInvokable(raw_result={"value": 123})
        wrapper = structured(component=component, output_schema=["value"])

        result = asyncio.run(wrapper.async_invoke({"value": 123, "extra": "ignored"}))

        assert isinstance(result, StructuredResultDict)
        assert result == {"value": 123}
        assert result.raw_result == {"value": 123}
        assert component.async_calls == [{"value": 123}]

    def test_component_sync_exception_propagates(self) -> None:
        component = RecordingInvokable(
            raw_result={},
            raise_sync=RuntimeError("component failed"),
        )
        wrapper = structured(component=component, output_schema=["value"])

        with pytest.raises(RuntimeError, match="component failed"):
            wrapper.invoke({"value": 1})

    def test_component_async_exception_propagates(self) -> None:
        component = RecordingInvokable(
            raw_result={},
            raise_async=RuntimeError("async component failed"),
        )
        wrapper = structured(component=component, output_schema=["value"])

        with pytest.raises(RuntimeError, match="async component failed"):
            asyncio.run(wrapper.async_invoke({"value": 1}))
