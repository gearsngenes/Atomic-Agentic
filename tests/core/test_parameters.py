from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, TypedDict

import pytest

from atomic_agentic.core.Exceptions import SchemaError
from atomic_agentic.core.Parameters import (
    ParamSpec,
    extract_io,
    is_valid_parameter_order,
    to_paramspec_list,
)
from atomic_agentic.core.sentinels import NO_VAL


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


class TestParamSpec:
    def test_paramspec_exposes_mapping_and_attribute_views(self) -> None:
        spec = ParamSpec(
            name="x",
            index=0,
            kind=ParamSpec.POSITIONAL_OR_KEYWORD,
            type="int",
            default=NO_VAL,
        )

        assert spec["name"] == "x"
        assert spec["index"] == 0
        assert spec["kind"] == ParamSpec.POSITIONAL_OR_KEYWORD
        assert spec["type"] == "int"

        assert spec.name == "x"
        assert spec.index == 0
        assert spec.kind == ParamSpec.POSITIONAL_OR_KEYWORD
        assert spec.type == "int"
        assert spec.default is NO_VAL

    def test_paramspec_stores_default_when_present(self) -> None:
        spec = ParamSpec(
            name="limit",
            index=0,
            kind=ParamSpec.POSITIONAL_OR_KEYWORD,
            type="int",
            default=10,
        )

        assert spec["default"] == 10
        assert spec.default == 10

    def test_paramspec_omits_default_key_when_default_is_no_val(self) -> None:
        spec = make_param("x", 0)

        assert "default" not in spec
        assert spec.default is NO_VAL

    def test_paramspec_is_mapping_immutable_via_setitem(self) -> None:
        spec = make_param("x", 0)

        with pytest.raises(TypeError):
            spec["name"] = "y"

    def test_paramspec_is_mapping_immutable_via_delitem(self) -> None:
        spec = make_param("x", 0)

        with pytest.raises(TypeError):
            del spec["name"]

    def test_paramspec_to_dict_round_trip_without_default(self) -> None:
        spec = make_param("x", 0, type_="str")

        data = spec.to_dict()
        round_trip = ParamSpec.from_dict(data)

        assert data == {
            "name": "x",
            "index": 0,
            "kind": ParamSpec.POSITIONAL_OR_KEYWORD,
            "type": "str",
        }
        assert round_trip.name == spec.name
        assert round_trip.index == spec.index
        assert round_trip.kind == spec.kind
        assert round_trip.type == spec.type
        assert round_trip.default is NO_VAL

    def test_paramspec_to_dict_round_trip_with_default(self) -> None:
        spec = make_param("limit", 0, type_="int", default=10)

        data = spec.to_dict()
        round_trip = ParamSpec.from_dict(data)

        assert data == {
            "name": "limit",
            "index": 0,
            "kind": ParamSpec.POSITIONAL_OR_KEYWORD,
            "type": "int",
            "default": 10,
        }
        assert round_trip.default == 10

    def test_paramspec_from_dict_requires_mapping(self) -> None:
        with pytest.raises(TypeError):
            ParamSpec.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "data",
        [
            {},
            {"name": "x", "index": 0, "kind": ParamSpec.POSITIONAL_OR_KEYWORD},
            {"name": "x", "index": "0", "kind": ParamSpec.POSITIONAL_OR_KEYWORD, "type": "Any"},
            {"name": 123, "index": 0, "kind": ParamSpec.POSITIONAL_OR_KEYWORD, "type": "Any"},
            {"name": "x", "index": 0, "kind": 123, "type": "Any"},
            {"name": "x", "index": 0, "kind": ParamSpec.POSITIONAL_OR_KEYWORD, "type": 123},
        ],
    )
    def test_paramspec_from_dict_rejects_malformed_mapping(
        self,
        data: Mapping[str, Any],
    ) -> None:
        with pytest.raises(TypeError):
            ParamSpec.from_dict(data)


class TestExtractIO:
    def test_extract_io_rejects_non_callable(self) -> None:
        with pytest.raises(TypeError):
            extract_io(123)  # type: ignore[arg-type]

    def test_extract_io_extracts_basic_parameters_and_return_type(self) -> None:
        def sample(x: int, y: str = "default") -> bool:
            return bool(x and y)

        parameters, return_type = extract_io(sample)

        assert return_type == "bool"
        assert [param.name for param in parameters] == ["x", "y"]
        assert [param.kind for param in parameters] == [
            ParamSpec.POSITIONAL_OR_KEYWORD,
            ParamSpec.POSITIONAL_OR_KEYWORD,
        ]
        assert [param.type for param in parameters] == ["int", "str"]
        assert parameters[0].default is NO_VAL
        assert parameters[1].default == "default"

    def test_extract_io_infers_type_from_default_when_annotation_missing(self) -> None:
        def sample(limit=10):
            return limit

        parameters, return_type = extract_io(sample)

        assert return_type == "Any"
        assert parameters[0].name == "limit"
        assert parameters[0].type == "int"
        assert parameters[0].default == 10

    def test_extract_io_handles_varargs_keyword_only_and_varkwargs(self) -> None:
        def sample(x: int, *args: str, debug: bool, **extras: float) -> None:
            return None

        parameters, return_type = extract_io(sample)

        assert return_type == "Any"
        assert [(param.name, param.kind, param.type) for param in parameters] == [
            ("x", ParamSpec.POSITIONAL_OR_KEYWORD, "int"),
            ("args", ParamSpec.VAR_POSITIONAL, "str"),
            ("debug", ParamSpec.KEYWORD_ONLY, "bool"),
            ("extras", ParamSpec.VAR_KEYWORD, "float"),
        ]

    def test_extract_io_handles_positional_only_parameters(self) -> None:
        def sample(x: int, /, y: int) -> int:
            return x + y

        parameters, return_type = extract_io(sample)

        assert return_type == "int"
        assert [(param.name, param.kind) for param in parameters] == [
            ("x", ParamSpec.POSITIONAL_ONLY),
            ("y", ParamSpec.POSITIONAL_OR_KEYWORD),
        ]

    def test_extract_io_formats_builtin_generics(self) -> None:
        def sample(values: list[int], config: dict[str, Any]) -> list[str]:
            return [str(config.get("prefix", "")) + str(value) for value in values]

        parameters, return_type = extract_io(sample)

        assert [param.type for param in parameters] == ["list[int]", "dict[str, Any]"]
        assert return_type == "list[str]"

    def test_extract_io_preserves_string_annotations(self) -> None:
        def sample(value: "CustomType") -> "OtherType":
            return value  # type: ignore[return-value]

        parameters, return_type = extract_io(sample)

        assert parameters[0].type == "CustomType"
        assert return_type == "OtherType"

    def test_extract_io_formats_optional_annotation(self) -> None:
        def sample(value: Optional[int]) -> Optional[str]:
            return str(value) if value is not None else None

        parameters, return_type = extract_io(sample)

        assert parameters[0].type in {"Union[int, NoneType]", "Optional[int]", "int | None"}
        assert return_type in {"Union[str, NoneType]", "Optional[str]", "str | None"}


class TestParameterOrderValidation:
    def test_valid_empty_parameter_list(self) -> None:
        assert is_valid_parameter_order([]) is True

    def test_valid_full_parameter_order(self) -> None:
        parameters = [
            make_param("a", 0, ParamSpec.POSITIONAL_ONLY),
            make_param("b", 1, ParamSpec.POSITIONAL_OR_KEYWORD),
            make_param("args", 2, ParamSpec.VAR_POSITIONAL),
            make_param("debug", 3, ParamSpec.KEYWORD_ONLY),
            make_param("extras", 4, ParamSpec.VAR_KEYWORD),
        ]

        assert is_valid_parameter_order(parameters) is True

    def test_rejects_non_list_input(self) -> None:
        with pytest.raises(TypeError):
            is_valid_parameter_order(tuple())  # type: ignore[arg-type]

    def test_rejects_non_paramspec_items(self) -> None:
        with pytest.raises(TypeError):
            is_valid_parameter_order(["x"])  # type: ignore[list-item]

    def test_rejects_duplicate_parameter_names(self) -> None:
        parameters = [
            make_param("x", 0),
            make_param("x", 1),
        ]

        with pytest.raises(SchemaError, match="Duplicate"):
            is_valid_parameter_order(parameters)

    def test_rejects_unknown_parameter_kind(self) -> None:
        parameters = [
            make_param("x", 0, "UNKNOWN_KIND"),
        ]

        with pytest.raises(SchemaError, match="Unknown parameter kind"):
            is_valid_parameter_order(parameters)

    def test_rejects_out_of_order_kinds(self) -> None:
        parameters = [
            make_param("debug", 0, ParamSpec.KEYWORD_ONLY),
            make_param("x", 1, ParamSpec.POSITIONAL_OR_KEYWORD),
        ]

        with pytest.raises(SchemaError, match="Invalid parameter order"):
            is_valid_parameter_order(parameters)

    def test_rejects_multiple_varargs(self) -> None:
        parameters = [
            make_param("args", 0, ParamSpec.VAR_POSITIONAL),
            make_param("more_args", 1, ParamSpec.VAR_POSITIONAL),
        ]

        with pytest.raises(SchemaError, match="Only one VAR_POSITIONAL"):
            is_valid_parameter_order(parameters)

    def test_rejects_multiple_varkwargs(self) -> None:
        parameters = [
            make_param("extras", 0, ParamSpec.VAR_KEYWORD),
            make_param("more_extras", 1, ParamSpec.VAR_KEYWORD),
        ]

        with pytest.raises(SchemaError, match="Only one VAR_KEYWORD"):
            is_valid_parameter_order(parameters)

    def test_rejects_varargs_with_default(self) -> None:
        parameters = [
            make_param("args", 0, ParamSpec.VAR_POSITIONAL, default=()),
        ]

        with pytest.raises(SchemaError, match="cannot have a default"):
            is_valid_parameter_order(parameters)

    def test_rejects_varkwargs_with_default(self) -> None:
        parameters = [
            make_param("extras", 0, ParamSpec.VAR_KEYWORD, default={}),
        ]

        with pytest.raises(SchemaError, match="cannot have a default"):
            is_valid_parameter_order(parameters)

    def test_rejects_required_positional_after_defaulted_positional(self) -> None:
        parameters = [
            make_param("x", 0, default=1),
            make_param("y", 1),
        ]

        with pytest.raises(SchemaError, match="cannot follow"):
            is_valid_parameter_order(parameters)

    def test_allows_required_keyword_only_after_defaulted_keyword_only(self) -> None:
        parameters = [
            make_param("x", 0, default=1),
            make_param("args", 1, ParamSpec.VAR_POSITIONAL),
            make_param("optional_flag", 2, ParamSpec.KEYWORD_ONLY, default=False),
            make_param("required_flag", 3, ParamSpec.KEYWORD_ONLY),
        ]

        assert is_valid_parameter_order(parameters) is True


class TestToParamSpecList:
    def test_none_normalizes_to_empty_list(self) -> None:
        assert to_paramspec_list(None) == []

    def test_empty_sequence_normalizes_to_empty_list(self) -> None:
        assert to_paramspec_list([]) == []
        assert to_paramspec_list(()) == []
        assert to_paramspec_list(set()) == []

    def test_typed_dict_class_normalizes_annotations(self) -> None:
        class Config(TypedDict):
            query: str
            top_k: int

        parameters = to_paramspec_list(Config)

        assert [(param.name, param.index, param.kind, param.type) for param in parameters] == [
            ("query", 0, ParamSpec.POSITIONAL_OR_KEYWORD, "str"),
            ("top_k", 1, ParamSpec.POSITIONAL_OR_KEYWORD, "int"),
        ]

    def test_list_of_paramspecs_is_reindexed_into_fresh_specs(self) -> None:
        original = [
            make_param("x", 10, type_="int"),
            make_param("y", 11, type_="str", default="hello"),
        ]

        parameters = to_paramspec_list(original)

        assert parameters is not original
        assert [(param.name, param.index, param.type, param.default) for param in parameters] == [
            ("x", 0, "int", NO_VAL),
            ("y", 1, "str", "hello"),
        ]
        assert parameters[0] is not original[0]
        assert parameters[1] is not original[1]

    def test_rejects_unsupported_schema_type(self) -> None:
        with pytest.raises(SchemaError):
            to_paramspec_list(123)  # type: ignore[arg-type]

    def test_rejects_mixed_sequence_types(self) -> None:
        with pytest.raises(SchemaError):
            to_paramspec_list(["x", make_param("y", 1)])  # type: ignore[list-item]

    def test_rejects_sequence_of_non_strings_and_non_paramspecs(self) -> None:
        with pytest.raises(SchemaError):
            to_paramspec_list([1, 2, 3])  # type: ignore[list-item]


class TestToParamSpecListStringGrammar:
    def test_plain_string_names_create_positional_or_keyword_parameters(self) -> None:
        parameters = to_paramspec_list(["x", "y"])

        assert [(param.name, param.index, param.kind, param.type) for param in parameters] == [
            ("x", 0, ParamSpec.POSITIONAL_OR_KEYWORD, "Any"),
            ("y", 1, ParamSpec.POSITIONAL_OR_KEYWORD, "Any"),
        ]

    def test_slash_marker_converts_previous_plain_names_to_positional_only(self) -> None:
        parameters = to_paramspec_list(["x", "y", "/"])

        assert [(param.name, param.index, param.kind) for param in parameters] == [
            ("x", 0, ParamSpec.POSITIONAL_ONLY),
            ("y", 1, ParamSpec.POSITIONAL_ONLY),
        ]

    def test_star_marker_converts_following_plain_names_to_keyword_only(self) -> None:
        parameters = to_paramspec_list(["x", "*", "debug", "limit"])

        assert [(param.name, param.index, param.kind) for param in parameters] == [
            ("x", 0, ParamSpec.POSITIONAL_OR_KEYWORD),
            ("debug", 1, ParamSpec.KEYWORD_ONLY),
            ("limit", 2, ParamSpec.KEYWORD_ONLY),
        ]

    def test_named_star_creates_var_positional_and_keyword_only_section(self) -> None:
        parameters = to_paramspec_list(["x", "*args", "debug"])

        assert [(param.name, param.index, param.kind) for param in parameters] == [
            ("x", 0, ParamSpec.POSITIONAL_OR_KEYWORD),
            ("args", 1, ParamSpec.VAR_POSITIONAL),
            ("debug", 2, ParamSpec.KEYWORD_ONLY),
        ]

    def test_double_star_creates_var_keyword_parameter(self) -> None:
        parameters = to_paramspec_list(["x", "**extras"])

        assert [(param.name, param.index, param.kind) for param in parameters] == [
            ("x", 0, ParamSpec.POSITIONAL_OR_KEYWORD),
            ("extras", 1, ParamSpec.VAR_KEYWORD),
        ]

    def test_full_mixed_string_schema_normalizes_to_python_signature_order(self) -> None:
        parameters = to_paramspec_list([
            "a",
            "b",
            "/",
            "c",
            "*args",
            "debug",
            "limit",
            "**extras",
        ])

        assert [(param.name, param.index, param.kind) for param in parameters] == [
            ("a", 0, ParamSpec.POSITIONAL_ONLY),
            ("b", 1, ParamSpec.POSITIONAL_ONLY),
            ("c", 2, ParamSpec.POSITIONAL_OR_KEYWORD),
            ("args", 3, ParamSpec.VAR_POSITIONAL),
            ("debug", 4, ParamSpec.KEYWORD_ONLY),
            ("limit", 5, ParamSpec.KEYWORD_ONLY),
            ("extras", 6, ParamSpec.VAR_KEYWORD),
        ]

    @pytest.mark.parametrize(
        "schema",
        [
            ["/"],
            ["x", "/", "/"],
            ["x", "*", "*"],
            ["x", "*args", "*more"],
            ["x", "*args", "*"],
            ["x", "**extras", "after"],
            ["x", "**extras", "**more"],
            ["x", "*", "/"],
            ["x", "**"],
            ["x", "*"],
        ],
    )
    def test_invalid_marker_sequences_raise_schema_error(self, schema: list[str]) -> None:
        with pytest.raises(SchemaError):
            to_paramspec_list(schema)

    @pytest.mark.parametrize(
        "schema",
        [
            [""],
            ["   "],
            ["bad-name"],
            ["123bad"],
            ["*"],
            ["*bad-name"],
            ["**bad-name"],
        ],
    )
    def test_invalid_string_names_raise_schema_error(self, schema: list[str]) -> None:
        with pytest.raises(SchemaError):
            to_paramspec_list(schema)