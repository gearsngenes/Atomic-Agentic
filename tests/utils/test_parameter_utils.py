from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

import pytest

from atomic_agentic.exceptions import SchemaError
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.utils.parameters import (
    _validate_parameter_order,
    extract_io,
    is_valid_parameter_order,
    to_paramspec_list,
)
from atomic_agentic.constants.core import NO_VAL


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

        assert return_type == "None"
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

        assert parameters[0].type in {"CustomType", "'CustomType'"}
        assert return_type in {"OtherType", "'OtherType'"}

    def test_extract_io_formats_optional_annotation(self) -> None:
        def sample(value: Optional[int]) -> Optional[str]:
            return str(value) if value is not None else None

        parameters, return_type = extract_io(sample)

        assert parameters[0].type in {"Union[int, NoneType]", "Union[int, None]", "Optional[int]", "int | None"}
        assert return_type in {"Union[str, NoneType]", "Union[str, None]", "Optional[str]", "str | None"}


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
            _validate_parameter_order(parameters)

    def test_paramspec_rejects_unknown_parameter_kind(self) -> None:
        with pytest.raises(ValueError, match="ParamSpec.kind"):
            ParamSpec(
                name="x",
                index=0,
                kind="UNKNOWN",
                type="Any",
            )

    def test_rejects_out_of_order_kinds(self) -> None:
        parameters = [
            make_param("debug", 0, ParamSpec.KEYWORD_ONLY),
            make_param("x", 1, ParamSpec.POSITIONAL_OR_KEYWORD),
        ]

        with pytest.raises(SchemaError, match="Invalid parameter order"):
            _validate_parameter_order(parameters)

    def test_rejects_multiple_varargs(self) -> None:
        parameters = [
            make_param("args", 0, ParamSpec.VAR_POSITIONAL),
            make_param("more_args", 1, ParamSpec.VAR_POSITIONAL),
        ]

        with pytest.raises(SchemaError, match="Only one VAR_POSITIONAL"):
            _validate_parameter_order(parameters)

    def test_rejects_multiple_varkwargs(self) -> None:
        parameters = [
            make_param("extras", 0, ParamSpec.VAR_KEYWORD),
            make_param("more_extras", 1, ParamSpec.VAR_KEYWORD),
        ]

        with pytest.raises(SchemaError, match="Only one VAR_KEYWORD"):
            _validate_parameter_order(parameters)

    def test_rejects_varargs_with_default(self) -> None:
        parameters = [
            make_param("args", 0, ParamSpec.VAR_POSITIONAL, default=()),
        ]

        with pytest.raises(SchemaError, match="cannot have a default"):
            _validate_parameter_order(parameters)

    def test_rejects_varkwargs_with_default(self) -> None:
        parameters = [
            make_param("extras", 0, ParamSpec.VAR_KEYWORD, default={}),
        ]

        with pytest.raises(SchemaError, match="cannot have a default"):
            _validate_parameter_order(parameters)

    def test_rejects_required_positional_after_defaulted_positional(self) -> None:
        parameters = [
            make_param("x", 0, default=1),
            make_param("y", 1),
        ]

        with pytest.raises(SchemaError, match="cannot follow"):
            _validate_parameter_order(parameters)

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


class TestExtractIOAnnotated:
    def test_annotated_param_extracts_base_type_and_description(self) -> None:
        def sample(x: Annotated[str, "the x value"]) -> str:
            return x

        parameters, _ = extract_io(sample)

        assert parameters[0].type == "str"
        assert parameters[0].description == "the x value"

    def test_plain_param_has_none_description(self) -> None:
        def sample(x: int) -> None:
            pass

        parameters, _ = extract_io(sample)

        assert parameters[0].description is None

    def test_annotated_first_string_metadata_wins(self) -> None:
        def sample(x: Annotated[int, "first", "second"]) -> None:
            pass

        parameters, _ = extract_io(sample)

        assert parameters[0].description == "first"

    def test_annotated_non_string_metadata_before_string_is_skipped(self) -> None:
        def sample(x: Annotated[float, 999, "valid"]) -> None:
            pass

        parameters, _ = extract_io(sample)

        assert parameters[0].type == "float"
        assert parameters[0].description == "valid"

    def test_annotated_whitespace_only_description_coerced_to_none(self) -> None:
        def sample(x: Annotated[str, "   "]) -> None:
            pass

        parameters, _ = extract_io(sample)

        assert parameters[0].description is None

    def test_annotated_return_type_unwrapped_description_discarded(self) -> None:
        def sample() -> Annotated[str, "return doc"]:
            return "x"

        _, return_type = extract_io(sample)

        assert return_type == "str"

    def test_annotated_varargs_description_extracted(self) -> None:
        def sample(*args: Annotated[int, "positional items"]) -> None:
            pass

        parameters, _ = extract_io(sample)

        assert parameters[0].kind == ParamSpec.VAR_POSITIONAL
        assert parameters[0].type == "int"
        assert parameters[0].description == "positional items"

    def test_annotated_varkwargs_description_extracted(self) -> None:
        def sample(**kwargs: Annotated[str, "keyword items"]) -> None:
            pass

        parameters, _ = extract_io(sample)

        assert parameters[0].kind == ParamSpec.VAR_KEYWORD
        assert parameters[0].description == "keyword items"

    def test_annotated_mixed_described_and_plain_params(self) -> None:
        def sample(
            x: Annotated[str, "has description"],
            y: int,
            z: Annotated[bool, "also described"] = True,
        ) -> None:
            pass

        parameters, _ = extract_io(sample)

        assert parameters[0].description == "has description"
        assert parameters[1].description is None
        assert parameters[2].description == "also described"

    def test_annotated_keyword_only_description_extracted(self) -> None:
        def sample(*, flag: Annotated[bool, "enable feature"]) -> None:
            pass

        parameters, _ = extract_io(sample)

        assert parameters[0].kind == ParamSpec.KEYWORD_ONLY
        assert parameters[0].description == "enable feature"


class TestToParamSpecListTypedDictAnnotated:
    def test_typed_dict_annotated_field_extracts_description(self) -> None:
        class Config(TypedDict):
            query: Annotated[str, "the search string"]
            limit: int

        parameters = to_paramspec_list(Config)

        query_param = next(p for p in parameters if p.name == "query")
        limit_param = next(p for p in parameters if p.name == "limit")
        assert query_param.type == "str"
        assert query_param.description == "the search string"
        assert limit_param.description is None

    def test_typed_dict_plain_fields_have_none_description(self) -> None:
        class Config(TypedDict):
            x: int
            y: str

        parameters = to_paramspec_list(Config)

        assert all(p.description is None for p in parameters)
