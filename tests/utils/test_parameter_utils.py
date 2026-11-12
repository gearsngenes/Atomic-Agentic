from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Annotated, Any, Literal, Optional, TypedDict

import pytest

from atomic_agentic.exceptions import SchemaError
from atomic_agentic.core.Invokable import AtomicInvokable
from atomic_agentic.models.parameters import ParameterReport, ParamSpec
from atomic_agentic.utils.parameters import (
    _insertion_category,
    _normalize_prompt_template,
    _try_parse_clean_field,
    _validate_parameter_order,
    apply_parameter_reports,
    build_parameter_report,
    build_parameter_reports,
    insert_by_category,
    is_valid_parameter_order,
    n_way_kind_compatible,
    n_way_type_witness,
    semantically_compatible,
    semantically_identical,
    to_paramspec_list,
)
from atomic_agentic.core.core_api import extract_io, parameter_overlap, parameter_collisions
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


class _AddInvokable(AtomicInvokable):
    def __init__(self) -> None:
        super().__init__(
            name="add_invokable",
            namespace="tests",
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


class TestExtractIO:
    def test_extract_io_rejects_non_callable(self) -> None:
        with pytest.raises(TypeError):
            extract_io(123)  # type: ignore[arg-type]

    def test_extract_io_returns_atomic_invokable_declared_schema_directly(self) -> None:
        invokable = _AddInvokable()

        parameters, return_type = extract_io(invokable)

        assert parameters == invokable.parameters
        assert return_type == invokable.return_type == "int"

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
        assert [param.type for param in parameters] == [("int",), ("str",)]
        assert parameters[0].default is NO_VAL
        assert parameters[1].default == "default"

    def test_extract_io_infers_type_from_default_when_annotation_missing(self) -> None:
        def sample(limit=10):
            return limit

        parameters, return_type = extract_io(sample)

        assert return_type == "Any"
        assert parameters[0].name == "limit"
        assert parameters[0].type == ("int",)
        assert parameters[0].default == 10

    def test_extract_io_handles_varargs_keyword_only_and_varkwargs(self) -> None:
        def sample(x: int, *args: str, debug: bool, **extras: float) -> None:
            return None

        parameters, return_type = extract_io(sample)

        assert return_type == "None"
        assert [(param.name, param.kind, param.type) for param in parameters] == [
            ("x", ParamSpec.POSITIONAL_OR_KEYWORD, ("int",)),
            ("args", ParamSpec.VAR_POSITIONAL, ("str",)),
            ("debug", ParamSpec.KEYWORD_ONLY, ("bool",)),
            ("extras", ParamSpec.VAR_KEYWORD, ("float",)),
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

        assert [param.type for param in parameters] == [("list[int]",), ("dict[str, Any]",)]
        assert return_type == "list[str]"

    def test_extract_io_preserves_string_annotations(self) -> None:
        def sample(value: "CustomType") -> "OtherType":
            return value  # type: ignore[return-value]

        parameters, return_type = extract_io(sample)

        assert parameters[0].type in {("CustomType",), ("'CustomType'",)}
        assert return_type in {"OtherType", "'OtherType'"}

    def test_extract_io_formats_optional_annotation(self) -> None:
        def sample(value: Optional[int]) -> Optional[str]:
            return str(value) if value is not None else None

        parameters, return_type = extract_io(sample)

        assert parameters[0].type == ("None", "int")
        assert return_type == "None | str"

    def test_extract_io_pep604_union_matches_typing_union(self) -> None:
        def sample(value: int | None) -> str | None:
            return str(value) if value is not None else None

        parameters, return_type = extract_io(sample)

        assert parameters[0].type == ("None", "int")
        assert return_type == "None | str"

    def test_extract_io_literal_none_is_not_corrupted_to_any(self) -> None:
        def sample(value: Literal[None]) -> None:
            return None

        parameters, _ = extract_io(sample)

        assert parameters[0].type == ("Literal[None]",)


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
            ("query", 0, ParamSpec.POSITIONAL_OR_KEYWORD, ("str",)),
            ("top_k", 1, ParamSpec.POSITIONAL_OR_KEYWORD, ("int",)),
        ]

    def test_list_of_paramspecs_is_reindexed_into_fresh_specs(self) -> None:
        original = [
            make_param("x", 10, type_="int"),
            make_param("y", 11, type_="str", default="hello"),
        ]

        parameters = to_paramspec_list(original)

        assert parameters is not original
        assert [(param.name, param.index, param.type, param.default) for param in parameters] == [
            ("x", 0, ("int",), NO_VAL),
            ("y", 1, ("str",), "hello"),
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
            ("x", 0, ParamSpec.POSITIONAL_OR_KEYWORD, ("Any",)),
            ("y", 1, ParamSpec.POSITIONAL_OR_KEYWORD, ("Any",)),
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

        assert parameters[0].type == ("str",)
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

        assert parameters[0].type == ("float",)
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
        assert parameters[0].type == ("int",)
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
        assert query_param.type == ("str",)
        assert query_param.description == "the search string"
        assert limit_param.description is None

    def test_typed_dict_plain_fields_have_none_description(self) -> None:
        class Config(TypedDict):
            x: int
            y: str

        parameters = to_paramspec_list(Config)

        assert all(p.description is None for p in parameters)


class TestTryParseCleanField:
    def test_bare_identifier(self) -> None:
        assert _try_parse_clean_field("role") == "role"

    def test_identifier_with_conversion(self) -> None:
        assert _try_parse_clean_field("role!r") == "role"

    def test_identifier_with_format_spec(self) -> None:
        assert _try_parse_clean_field("age:>10") == "age"

    def test_identifier_with_conversion_and_format_spec(self) -> None:
        assert _try_parse_clean_field("role!r:>10") == "role"

    def test_empty_string_is_not_clean(self) -> None:
        assert _try_parse_clean_field("") is None

    def test_digit_leading_is_not_clean(self) -> None:
        assert _try_parse_clean_field("0") is None

    def test_dotted_access_is_not_clean(self) -> None:
        assert _try_parse_clean_field("obj.attr") is None

    def test_indexed_access_is_not_clean(self) -> None:
        assert _try_parse_clean_field("x[0]") is None

    def test_invalid_conversion_flag_is_not_clean(self) -> None:
        assert _try_parse_clean_field("name!z") is None

    def test_conversion_flag_with_no_char_is_not_clean(self) -> None:
        assert _try_parse_clean_field("name!") is None

    def test_trailing_garbage_after_identifier_is_not_clean(self) -> None:
        assert _try_parse_clean_field("name garbage") is None


class TestNormalizePromptTemplate:
    def test_plain_identifier_fields_discovered_and_preserved(self) -> None:
        normalized, discovered = _normalize_prompt_template(
            "You are a {role} expert in {domain}."
        )
        assert normalized == "You are a {role} expert in {domain}."
        assert discovered == ["role", "domain"]

    def test_duplicate_placeholder_deduplicated_in_first_appearance_order(self) -> None:
        normalized, discovered = _normalize_prompt_template(
            "Hello {name}, I said hello {name}!"
        )
        assert normalized == "Hello {name}, I said hello {name}!"
        assert discovered == ["name"]

    def test_conversion_and_format_spec_preserved(self) -> None:
        normalized, discovered = _normalize_prompt_template(
            "Quote: {role!r} and spec: {age:>10}."
        )
        assert normalized == "Quote: {role!r} and spec: {age:>10}."
        assert discovered == ["role", "age"]

    def test_stray_unmatched_open_brace_never_raises(self) -> None:
        normalized, discovered = _normalize_prompt_template(
            "you are an expert on the history of the { character"
        )
        assert discovered == []
        assert normalized.format() == (
            "you are an expert on the history of the { character"
        )

    def test_stray_unmatched_close_brace_never_raises(self) -> None:
        normalized, discovered = _normalize_prompt_template(
            "this has a } lone brace."
        )
        assert discovered == []
        assert normalized.format() == "this has a } lone brace."

    def test_already_escaped_literal_passes_through_unchanged(self) -> None:
        normalized, discovered = _normalize_prompt_template(
            "Return JSON like {{ \"result\": \"value\" }}."
        )
        assert normalized == "Return JSON like {{ \"result\": \"value\" }}."
        assert discovered == []
        assert normalized.format() == 'Return JSON like { "result": "value" }.'

    def test_unescaped_json_is_escaped_and_round_trips(self) -> None:
        raw = 'Return JSON like { "result": "value" }.'
        normalized, discovered = _normalize_prompt_template(raw)
        assert discovered == []
        assert normalized.format() == raw

    def test_empty_braces_are_escaped_not_discovered(self) -> None:
        normalized, discovered = _normalize_prompt_template("Empty {} braces.")
        assert discovered == []
        assert normalized.format() == "Empty {} braces."

    def test_digit_index_is_escaped_not_discovered(self) -> None:
        normalized, discovered = _normalize_prompt_template("Index {0} value.")
        assert discovered == []
        assert normalized.format() == "Index {0} value."

    def test_dotted_and_indexed_access_escaped_not_discovered(self) -> None:
        normalized, discovered = _normalize_prompt_template(
            "Dotted {obj.attr} and indexed {x[0]}."
        )
        assert discovered == []
        assert normalized.format() == "Dotted {obj.attr} and indexed {x[0]}."

    def test_nested_format_spec_whole_span_escaped_not_discovered(self) -> None:
        raw = "Nested spec: {value:{width}} done."
        normalized, discovered = _normalize_prompt_template(raw)
        assert discovered == []
        assert normalized.format() == raw

    def test_deeply_nested_stray_braces_never_raise_and_round_trip(self) -> None:
        raw = "Multiple stray: { { {ok} } }"
        normalized, discovered = _normalize_prompt_template(raw)
        assert discovered == []
        assert normalized.format() == raw

    def test_invalid_conversion_flag_field_is_escaped_not_discovered(self) -> None:
        normalized, discovered = _normalize_prompt_template("Bad: {name!z} here.")
        assert discovered == []
        assert normalized.format() == "Bad: {name!z} here."

    def test_renormalizing_normalized_template_is_idempotent(self) -> None:
        raw = 'Mixed { "a": 1 } and {role} and {value:{width}} and stray {.'
        normalized_once, discovered_once = _normalize_prompt_template(raw)
        normalized_twice, discovered_twice = _normalize_prompt_template(normalized_once)
        assert normalized_twice == normalized_once
        assert discovered_twice == discovered_once


class TestSemanticallyCompatible:
    def test_same_type_same_non_variadic_kind_is_compatible(self) -> None:
        a = make_param("x", 0, ParamSpec.POSITIONAL_OR_KEYWORD, type_="int")
        b = make_param("x", 0, ParamSpec.POSITIONAL_OR_KEYWORD, type_="int")
        assert semantically_compatible(a, b) is True

    def test_either_side_any_type_is_compatible_despite_type_mismatch(self) -> None:
        a = make_param("x", 0, type_="int")
        b = make_param("x", 0, type_="Any")
        assert semantically_compatible(a, b) is True
        assert semantically_compatible(b, a) is True

    def test_mismatched_non_any_types_are_incompatible(self) -> None:
        a = make_param("x", 0, type_="int")
        b = make_param("x", 0, type_="str")
        assert semantically_compatible(a, b) is False

    def test_two_different_non_variadic_kinds_are_kind_compatible(self) -> None:
        a = make_param("x", 0, ParamSpec.POSITIONAL_ONLY, type_="int")
        b = make_param("x", 0, ParamSpec.KEYWORD_ONLY, type_="int")
        assert semantically_compatible(a, b) is True

    def test_same_variadic_kind_is_compatible(self) -> None:
        a = make_param("args", 0, ParamSpec.VAR_POSITIONAL, type_="int")
        b = make_param("args", 0, ParamSpec.VAR_POSITIONAL, type_="int")
        assert semantically_compatible(a, b) is True

    def test_different_variadic_kinds_are_incompatible(self) -> None:
        a = make_param("x", 0, ParamSpec.VAR_POSITIONAL, type_="int")
        b = make_param("x", 0, ParamSpec.VAR_KEYWORD, type_="int")
        assert semantically_compatible(a, b) is False

    def test_variadic_vs_non_variadic_is_incompatible(self) -> None:
        a = make_param("x", 0, ParamSpec.VAR_POSITIONAL, type_="int")
        b = make_param("x", 0, ParamSpec.POSITIONAL_OR_KEYWORD, type_="int")
        assert semantically_compatible(a, b) is False

    def test_bare_generic_origin_is_compatible_with_any_parameterization(self) -> None:
        a = make_param("x", 0, type_="dict")
        b = make_param("x", 0, type_="dict[str, Any]")
        assert semantically_compatible(a, b) is True
        assert semantically_compatible(b, a) is True

    def test_same_origin_incompatible_args_are_incompatible(self) -> None:
        a = make_param("x", 0, type_="dict[str, int]")
        b = make_param("x", 0, type_="dict[str, str]")
        assert semantically_compatible(a, b) is False

    def test_same_origin_any_wildcarded_arg_is_compatible(self) -> None:
        a = make_param("x", 0, type_="dict[str, int]")
        b = make_param("x", 0, type_="dict[str, Any]")
        assert semantically_compatible(a, b) is True

    def test_literal_reordered_values_are_compatible(self) -> None:
        def sample_ab(x: Literal["a", "b"]) -> None:
            pass

        def sample_ba(x: Literal["b", "a"]) -> None:
            pass

        a = extract_io(sample_ab)[0][0]
        b = extract_io(sample_ba)[0][0]
        assert semantically_compatible(a, b) is True

    def test_union_member_overlap_is_compatible(self) -> None:
        a = make_param("x", 0, type_=("int", "str"))
        b = make_param("x", 0, type_="str")
        assert semantically_compatible(a, b) is True


class TestSemanticallyIdentical:
    def _base(self) -> ParamSpec:
        return ParamSpec(
            name="x",
            index=0,
            kind=ParamSpec.KEYWORD_ONLY,
            type="int",
            default=1,
            description="the x value",
        )

    def test_fully_matching_specs_are_identical(self) -> None:
        a = self._base()
        b = self._base()
        assert semantically_identical(a, b) is True

    def test_identical_implies_compatible(self) -> None:
        a = self._base()
        b = self._base()
        assert semantically_identical(a, b) is True
        assert semantically_compatible(a, b) is True

    def test_type_mismatch_breaks_identity(self) -> None:
        a = self._base()
        b = replace(self._base(), type="str")
        assert semantically_identical(a, b) is False

    def test_kind_mismatch_breaks_identity(self) -> None:
        a = self._base()
        b = replace(self._base(), kind=ParamSpec.POSITIONAL_OR_KEYWORD)
        assert semantically_identical(a, b) is False

    def test_default_mismatch_breaks_identity(self) -> None:
        a = self._base()
        b = replace(self._base(), default=2)
        assert semantically_identical(a, b) is False

    def test_description_mismatch_breaks_identity(self) -> None:
        a = self._base()
        b = replace(self._base(), description="a different description")
        assert semantically_identical(a, b) is False

    def test_name_mismatch_breaks_identity(self) -> None:
        a = self._base()
        b = replace(self._base(), name="y")
        assert semantically_identical(a, b) is False

    def test_literal_reordered_values_are_identical(self) -> None:
        def sample_ab(x: Literal["a", "b"]) -> None:
            pass

        def sample_ba(x: Literal["b", "a"]) -> None:
            pass

        a = extract_io(sample_ab)[0][0]
        b = extract_io(sample_ba)[0][0]
        assert semantically_identical(a, b) is True


class TestParameterOverlapAndCollisions:
    def test_empty_sources_produce_no_overlap_or_collisions(self) -> None:
        assert parameter_overlap([]) == []
        assert parameter_collisions([]) == []

    def test_disjoint_names_produce_no_overlap_or_collisions(self) -> None:
        def source_a(x: int) -> None: ...
        def source_b(y: int) -> None: ...
        assert parameter_overlap([source_a, source_b]) == []
        assert parameter_collisions([source_a, source_b]) == []

    def test_compatible_shared_name_is_overlap_not_collision(self) -> None:
        def source_a(x: int) -> None: ...
        def source_b(x) -> None: ...  # noqa: ANN001 -- deliberately untyped -> "Any"
        assert parameter_overlap([source_a, source_b]) == ["x"]
        assert parameter_collisions([source_a, source_b]) == []

    def test_incompatible_shared_name_is_collision_not_overlap(self) -> None:
        def source_a(x: int) -> None: ...
        def source_b(x: str) -> None: ...
        assert parameter_overlap([source_a, source_b]) == []
        assert parameter_collisions([source_a, source_b]) == ["x"]

    def test_mixed_overlap_and_collisions_partition_cleanly(self) -> None:
        def source_a(compatible: int, colliding: int, only_in_a: int) -> None: ...
        def source_b(compatible, colliding: str, only_in_b: int) -> None: ...  # noqa: ANN001

        overlap = parameter_overlap([source_a, source_b])
        collisions = parameter_collisions([source_a, source_b])

        assert overlap == ["compatible"]
        assert collisions == ["colliding"]
        assert set(overlap) & set(collisions) == set()

    def test_results_follow_first_source_order(self) -> None:
        def source_a(second: int, first: int) -> None: ...
        def source_b(first: int, second: int) -> None: ...
        assert parameter_overlap([source_a, source_b]) == ["second", "first"]

    def test_name_shared_by_two_of_three_is_overlap_by_default(self) -> None:
        def source_a(x: int) -> None: ...
        def source_b(x: int) -> None: ...
        def source_c(y: int) -> None: ...
        assert parameter_overlap([source_a, source_b, source_c]) == ["x"]

    def test_unanimous_only_excludes_names_not_shared_by_every_source(self) -> None:
        def source_a(x: int) -> None: ...
        def source_b(x: int) -> None: ...
        def source_c(y: int) -> None: ...
        assert parameter_overlap([source_a, source_b, source_c], unanimous_only=True) == []

    def test_unanimous_only_includes_names_shared_by_every_source(self) -> None:
        def source_a(x: int) -> None: ...
        def source_b(x: int) -> None: ...
        def source_c(x: int) -> None: ...
        assert parameter_overlap([source_a, source_b, source_c], unanimous_only=True) == ["x"]

    def test_parameter_collisions_has_no_unanimous_only_parameter(self) -> None:
        import inspect
        assert "unanimous_only" not in inspect.signature(parameter_collisions).parameters

    def test_parameter_collisions_flags_two_source_disagreement_with_more_sources_present(self) -> None:
        def source_a(x: int) -> None: ...
        def source_b(x: str) -> None: ...
        def source_c(y: int) -> None: ...
        assert parameter_collisions([source_a, source_b, source_c]) == ["x"]

    def test_accepts_atomic_invokable_sources(self) -> None:
        invokable = _AddInvokable()
        def source_b(a: int) -> None: ...
        assert parameter_overlap([invokable, source_b]) == ["a"]


class TestInsertionCategory:
    def test_positional_only_without_default(self) -> None:
        spec = make_param("x", 0, ParamSpec.POSITIONAL_ONLY)
        assert _insertion_category(spec) == (0, 0)

    def test_positional_only_with_default(self) -> None:
        spec = make_param("x", 0, ParamSpec.POSITIONAL_ONLY, default=1)
        assert _insertion_category(spec) == (0, 1)

    def test_positional_or_keyword_without_default(self) -> None:
        spec = make_param("x", 0, ParamSpec.POSITIONAL_OR_KEYWORD)
        assert _insertion_category(spec) == (1, 0)

    def test_positional_or_keyword_with_default(self) -> None:
        spec = make_param("x", 0, ParamSpec.POSITIONAL_OR_KEYWORD, default=1)
        assert _insertion_category(spec) == (1, 1)

    def test_var_positional(self) -> None:
        spec = make_param("args", 0, ParamSpec.VAR_POSITIONAL)
        assert _insertion_category(spec) == (2, 0)

    def test_keyword_only_without_default_is_tier_zero(self) -> None:
        spec = make_param("x", 0, ParamSpec.KEYWORD_ONLY)
        assert _insertion_category(spec) == (3, 0)

    def test_keyword_only_with_default_is_still_tier_zero(self) -> None:
        spec = make_param("x", 0, ParamSpec.KEYWORD_ONLY, default=1)
        assert _insertion_category(spec) == (3, 0)

    def test_var_keyword(self) -> None:
        spec = make_param("extras", 0, ParamSpec.VAR_KEYWORD)
        assert _insertion_category(spec) == (4, 0)


class TestInsertByCategory:
    def test_empty_items_returns_fresh_equal_content_list(self) -> None:
        composed = [make_param("a", 5, type_="int")]
        result = insert_by_category(composed, [])

        assert result is not composed
        assert result[0] is not composed[0]
        assert [(p.name, p.index, p.type) for p in result] == [("a", 0, ("int",))]

    def test_simple_keyword_only_append(self) -> None:
        composed = [make_param("x", 0, ParamSpec.POSITIONAL_OR_KEYWORD)]
        items = [make_param("flag", 0, ParamSpec.KEYWORD_ONLY, default=False)]

        result = insert_by_category(composed, items)

        assert [p.name for p in result] == ["x", "flag"]
        assert [p.index for p in result] == [0, 1]

    def test_required_positional_or_keyword_graft_lands_before_defaulted_existing(self) -> None:
        # composed ~ def pre(a, /, b=1) -- a positional-only required, b
        # positional-or-keyword defaulted. Grafting a new required
        # positional-or-keyword param must land before b, not after, or the
        # result would violate the required-after-defaulted rule.
        composed = [
            make_param("a", 0, ParamSpec.POSITIONAL_ONLY, type_="int"),
            make_param("b", 1, ParamSpec.POSITIONAL_OR_KEYWORD, type_="int", default=1),
        ]
        items = [make_param("c", 0, ParamSpec.POSITIONAL_OR_KEYWORD, type_="int")]

        result = insert_by_category(composed, items)

        assert [p.name for p in result] == ["a", "c", "b"]
        assert is_valid_parameter_order(result) is True

    def test_batch_items_in_same_category_preserve_relative_order(self) -> None:
        items = [
            make_param("p1", 0, ParamSpec.KEYWORD_ONLY),
            make_param("p2", 0, ParamSpec.KEYWORD_ONLY),
        ]

        result = insert_by_category([], items)

        assert [p.name for p in result] == ["p1", "p2"]

    def test_existing_entries_land_before_new_entries_in_same_category(self) -> None:
        composed = [make_param("existing", 0, ParamSpec.KEYWORD_ONLY)]
        items = [make_param("new", 0, ParamSpec.KEYWORD_ONLY)]

        result = insert_by_category(composed, items)

        assert [p.name for p in result] == ["existing", "new"]

    def test_duplicate_name_raises_schema_error(self) -> None:
        composed = [make_param("x", 0, ParamSpec.KEYWORD_ONLY)]
        items = [make_param("x", 0, ParamSpec.KEYWORD_ONLY)]

        with pytest.raises(SchemaError, match="Duplicate"):
            insert_by_category(composed, items)


class TestParamSpecTypeConstruction:
    def test_bare_string_type_is_wrapped_into_a_one_tuple(self) -> None:
        spec = ParamSpec(name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type="int")
        assert spec.type == ("int",)

    def test_list_type_is_normalized_into_a_sorted_tuple(self) -> None:
        spec = ParamSpec(name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type=["str", "int"])
        assert spec.type == ("int", "str")

    def test_tuple_type_is_deduplicated(self) -> None:
        spec = ParamSpec(name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type=("int", "int", "str"))
        assert spec.type == ("int", "str")

    def test_non_str_non_sequence_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            ParamSpec(name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type=123)  # type: ignore[arg-type]

    def test_empty_tuple_type_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ParamSpec(name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type=())

    def test_non_str_member_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            ParamSpec(name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type=("int", 5))  # type: ignore[arg-type]

    def test_to_dict_from_dict_round_trips_multi_member_type(self) -> None:
        spec = ParamSpec(
            name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type=("int", "str"), default=1,
        )
        rebuilt = ParamSpec.from_dict(spec.to_dict())
        assert rebuilt.type == ("int", "str")
        assert rebuilt == spec

    def test_to_dict_emits_a_list_for_type(self) -> None:
        spec = ParamSpec(name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type=("int", "str"))
        assert spec.to_dict()["type"] == ["int", "str"]


class TestNWayTypeWitness:
    def test_empty_input_returns_empty_frozenset(self) -> None:
        assert n_way_type_witness([]) == frozenset()

    def test_all_abstain_returns_any_fallback(self) -> None:
        assert n_way_type_witness([("Any",), ("Any",), ("Any",)]) == frozenset({"Any"})

    def test_single_tuple_is_its_own_witness(self) -> None:
        assert n_way_type_witness([("int",)]) == frozenset({"int"})

    def test_mixed_any_tuple_leak_regression_still_excludes_any_from_candidacy(self) -> None:
        # The bug this algorithm was designed to close: "Any" must never be a
        # CANDIDATE, even sitting inside a mixed opinionated tuple like
        # ("Any", "int") -- it must not manufacture a false witness that
        # papers over ("int",) vs ("str",)'s genuine, flat conflict. "Any"
        # still works as a receiving-side wildcard (it satisfies its own
        # tuple's membership test trivially), but "int" fails against the
        # ("str",) tuple, and "str" fails against the ("int",) tuple, so the
        # witness set must come back empty.
        witness = n_way_type_witness([("Any", "int"), ("int",), ("str",)])
        assert witness == frozenset()

    def test_triangle_pairwise_compatible_but_no_universal_witness_is_empty(self) -> None:
        # Helly-property counterexample, N=3: every pair shares a token
        # (A-B: str, B-C: bool, A-C: int) but no single token is common to
        # all three -- the witness standard (unlike a weaker pairwise-all-
        # pairs check) correctly reports no shared type.
        a = ("int", "str")
        b = ("str", "bool")
        c = ("bool", "int")
        assert n_way_type_witness([a, b, c]) == frozenset()

    def test_genuine_three_way_witness_is_found_and_full_set_is_returned(self) -> None:
        # Every opinionated tuple actually shares "int"; "str" only appears
        # in one tuple and must not leak into the witness set.
        witness = n_way_type_witness([("int", "str"), ("int",), ("int",)])
        assert witness == frozenset({"int"})

    def test_abstaining_source_alongside_opinionated_sources_is_never_consulted(self) -> None:
        # A pure-Any source neither contributes to nor rescues the witness
        # search once at least one opinionated source exists.
        assert n_way_type_witness([("Any",), ("int",), ("int",)]) == frozenset({"int"})

    def test_abstaining_source_does_not_rescue_a_genuine_opinionated_conflict(self) -> None:
        assert n_way_type_witness([("Any",), ("int",), ("str",)]) == frozenset()

    def test_generator_input_is_materialized_and_consumed_safely(self) -> None:
        witness = n_way_type_witness((t for t in [("int",), ("int",)]))
        assert witness == frozenset({"int"})


class TestNWayKindCompatible:
    def test_no_variadic_present_is_always_compatible_regardless_of_mix(self) -> None:
        assert n_way_kind_compatible([
            ParamSpec.POSITIONAL_ONLY,
            ParamSpec.POSITIONAL_OR_KEYWORD,
            ParamSpec.KEYWORD_ONLY,
        ]) is True

    def test_single_kind_repeated_is_compatible(self) -> None:
        assert n_way_kind_compatible([ParamSpec.KEYWORD_ONLY, ParamSpec.KEYWORD_ONLY]) is True

    def test_single_shared_variadic_kind_is_compatible(self) -> None:
        assert n_way_kind_compatible([ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_POSITIONAL]) is True

    def test_two_distinct_variadic_kinds_are_incompatible(self) -> None:
        assert n_way_kind_compatible([ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD]) is False

    def test_one_variadic_plus_one_non_variadic_is_incompatible(self) -> None:
        assert n_way_kind_compatible([ParamSpec.VAR_POSITIONAL, ParamSpec.POSITIONAL_OR_KEYWORD]) is False

    def test_empty_input_has_no_variadic_and_is_compatible(self) -> None:
        assert n_way_kind_compatible([]) is True


class TestBuildParameterReport:
    def test_single_observation_is_a_trivial_pass_through(self) -> None:
        spec = make_param("x", 0, type_="int")
        report = build_parameter_report("x", [spec])

        assert report.parameter_name == "x"
        assert report.witness_types == frozenset({"int"})
        assert report.winner_source == 0
        assert report.kind_compatible is True
        assert report.observations == (spec,)
        assert report.is_identical is True

    def test_dense_none_gaps_are_skipped_for_witness_and_kind_computation(self) -> None:
        spec_b = make_param("x", 0, type_="int")
        report = build_parameter_report("x", [None, spec_b, None])

        assert report.witness_types == frozenset({"int"})
        assert report.observations == (None, spec_b, None)

    def test_winner_source_is_the_first_present_index_not_the_first_list_index(self) -> None:
        spec_b = make_param("x", 0, type_="int", default=1)
        spec_c = make_param("x", 0, type_="int", default=2)
        report = build_parameter_report("x", [None, spec_b, spec_c])

        assert report.winner_source == 1

    def test_is_identical_false_when_present_observations_differ(self) -> None:
        spec_a = make_param("x", 0, type_="int", default=1)
        spec_b = make_param("x", 0, type_="int", default=2)
        report = build_parameter_report("x", [spec_a, spec_b])

        # Compatible (same type/kind) but not identical (different default).
        assert report.witness_types == frozenset({"int"})
        assert report.is_identical is False

    def test_is_identical_true_when_present_observations_all_match_winner(self) -> None:
        spec_a = make_param("x", 0, type_="int", default=1)
        spec_b = make_param("x", 0, type_="int", default=1)
        report = build_parameter_report("x", [spec_a, spec_b])

        assert report.is_identical is True

    def test_kind_compatible_reflects_n_way_kind_compatible(self) -> None:
        spec_a = make_param("x", 0, ParamSpec.VAR_POSITIONAL, type_="Any")
        spec_b = make_param("x", 0, ParamSpec.VAR_KEYWORD, type_="Any")
        report = build_parameter_report("x", [spec_a, spec_b])

        assert report.kind_compatible is False


class TestBuildParameterReports:
    def test_first_seen_order_preserved_across_sources(self) -> None:
        source_a = [make_param("second", 0), make_param("first", 1)]
        source_b = [make_param("third", 0)]

        reports = build_parameter_reports([source_a, source_b])

        assert [r.parameter_name for r in reports] == ["second", "first", "third"]

    def test_multi_name_grouping_across_disjoint_source_subsets(self) -> None:
        source_a = [make_param("x", 0, type_="int"), make_param("shared", 1, type_="int")]
        source_b = [make_param("y", 0, type_="str")]
        source_c = [make_param("shared", 0, type_="int")]

        reports = build_parameter_reports([source_a, source_b, source_c])
        by_name = {r.parameter_name: r for r in reports}

        assert set(by_name) == {"x", "shared", "y"}
        assert by_name["x"].observations == (source_a[0], None, None)
        assert by_name["y"].observations == (None, source_b[0], None)
        assert by_name["shared"].observations == (source_a[1], None, source_c[0])
        assert by_name["shared"].is_identical is True

    def test_empty_sources_yields_empty_reports(self) -> None:
        assert build_parameter_reports([]) == []
        assert build_parameter_reports([[], []]) == []

    def test_report_entries_are_parameterreport_instances(self) -> None:
        reports = build_parameter_reports([[make_param("x", 0)]])
        assert isinstance(reports[0], ParameterReport)


class _DummyReconciliationError(Exception):
    """Distinct exception class, deliberately not Agent/Workflow-shaped --
    proves error_cls is actually threaded through apply_parameter_reports
    rather than hardcoded to one caller's exception type."""


class TestApplyParameterReports:
    def test_empty_witness_raises_error_cls(self) -> None:
        spec_a = make_param("x", 0, type_="int")
        spec_b = make_param("x", 0, type_="str")
        report = build_parameter_report("x", [spec_a, spec_b])
        assert report.witness_types == frozenset()

        with pytest.raises(_DummyReconciliationError, match="no compatible reconciliation"):
            apply_parameter_reports(
                [report], ("a", "b"), error_cls=_DummyReconciliationError, stacklevel=1
            )

    def test_kind_incompatible_raises_error_cls(self) -> None:
        spec_a = make_param("x", 0, ParamSpec.VAR_POSITIONAL, type_="Any")
        spec_b = make_param("x", 0, ParamSpec.VAR_KEYWORD, type_="Any")
        report = build_parameter_report("x", [spec_a, spec_b])
        assert report.kind_compatible is False

        with pytest.raises(_DummyReconciliationError, match="no compatible reconciliation"):
            apply_parameter_reports(
                [report], ("a", "b"), error_cls=_DummyReconciliationError, stacklevel=1
            )

    def test_all_identical_reports_construct_without_warning(self) -> None:
        spec_a = make_param("x", 0, type_="int", default=1)
        spec_b = make_param("x", 0, type_="int", default=1)
        report = build_parameter_report("x", [spec_a, spec_b])
        assert report.is_identical is True

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            constructed = apply_parameter_reports(
                [report], ("a", "b"), error_cls=_DummyReconciliationError, stacklevel=1
            )

        assert caught == []
        assert len(constructed) == 1
        assert constructed[0].name == "x"
        assert constructed[0].type == ("int",)
        assert constructed[0].default == 1

    def test_mixed_identical_and_non_identical_reports_single_grouped_warning(self) -> None:
        source_a = [
            make_param("a", 0, type_="int", default=1),
            make_param("b", 1, type_="int", default=1),
            make_param("c", 2, type_="int"),
        ]
        source_b = [
            make_param("a", 0, type_="int", default=2),
            make_param("b", 1, type_="int", default=2),
            make_param("c", 2, type_="int"),
        ]
        reports = build_parameter_reports([source_a, source_b])

        with pytest.warns(UserWarning) as record:
            constructed = apply_parameter_reports(
                reports, ("source_a", "source_b"), error_cls=_DummyReconciliationError, stacklevel=1
            )

        # Exactly one grouped warning, not one per non-identical name.
        assert len(record) == 1
        assert "['a', 'b']" in str(record[0].message)
        assert len(constructed) == 3

    def test_stacklevel_attributes_warning_to_the_supplied_frame_depth(self) -> None:
        spec_a = make_param("x", 0, type_="int", default=1)
        spec_b = make_param("x", 0, type_="int", default=2)
        report = build_parameter_report("x", [spec_a, spec_b])

        def call_site() -> None:
            apply_parameter_reports([report], ("a", "b"), error_cls=_DummyReconciliationError, stacklevel=2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call_site()

        assert len(caught) == 1
        # stacklevel=2 walks one frame up from apply_parameter_reports' own
        # warnings.warn call, landing on call_site's own invocation line --
        # not this test method's frame, and not apply_parameter_reports'
        # own module.
        assert caught[0].filename == __file__
        assert caught[0].lineno == call_site.__code__.co_firstlineno + 1
