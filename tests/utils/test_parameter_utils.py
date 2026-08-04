from __future__ import annotations

from dataclasses import replace
from typing import Annotated, Any, Optional, TypedDict

import pytest

from atomic_agentic.exceptions import SchemaError
from atomic_agentic.core.Invokable import AtomicInvokable
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.utils.parameters import (
    _insertion_category,
    _normalize_prompt_template,
    _try_parse_clean_field,
    _validate_parameter_order,
    insert_by_category,
    is_valid_parameter_order,
    parameter_collisions,
    parameter_overlap,
    semantically_compatible,
    semantically_identical,
    to_paramspec_list,
    variadic_compatible,
)
from atomic_agentic.core.core_api import extract_io
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


class TestParameterOverlapAndCollisions:
    def test_empty_lists_produce_no_overlap_or_collisions(self) -> None:
        assert parameter_overlap([], []) == []
        assert parameter_collisions([], []) == []

    def test_disjoint_names_produce_no_overlap_or_collisions(self) -> None:
        source_a = [make_param("x", 0, type_="int")]
        source_b = [make_param("y", 0, type_="int")]
        assert parameter_overlap(source_a, source_b) == []
        assert parameter_collisions(source_a, source_b) == []

    def test_compatible_shared_name_is_overlap_not_collision(self) -> None:
        source_a = [make_param("x", 0, type_="int")]
        source_b = [make_param("x", 0, type_="Any")]
        assert parameter_overlap(source_a, source_b) == ["x"]
        assert parameter_collisions(source_a, source_b) == []

    def test_incompatible_shared_name_is_collision_not_overlap(self) -> None:
        source_a = [make_param("x", 0, type_="int")]
        source_b = [make_param("x", 0, type_="str")]
        assert parameter_overlap(source_a, source_b) == []
        assert parameter_collisions(source_a, source_b) == ["x"]

    def test_mixed_overlap_and_collisions_partition_cleanly(self) -> None:
        source_a = [
            make_param("compatible", 0, type_="int"),
            make_param("colliding", 1, type_="int"),
            make_param("only_in_a", 2, type_="int"),
        ]
        source_b = [
            make_param("compatible", 0, type_="Any"),
            make_param("colliding", 1, type_="str"),
            make_param("only_in_b", 2, type_="int"),
        ]

        overlap = parameter_overlap(source_a, source_b)
        collisions = parameter_collisions(source_a, source_b)

        assert overlap == ["compatible"]
        assert collisions == ["colliding"]
        assert set(overlap) & set(collisions) == set()
        shared_names = {p.name for p in source_a} & {p.name for p in source_b}
        assert set(overlap) | set(collisions) == shared_names

    def test_results_follow_source_a_order(self) -> None:
        source_a = [
            make_param("second", 0, type_="int"),
            make_param("first", 1, type_="int"),
        ]
        source_b = [
            make_param("first", 0, type_="int"),
            make_param("second", 1, type_="int"),
        ]
        assert parameter_overlap(source_a, source_b) == ["second", "first"]


class TestVariadicCompatible:
    def test_neither_side_has_variadics(self) -> None:
        source_a = [make_param("x", 0)]
        source_b = [make_param("y", 0)]
        assert variadic_compatible(source_a, source_b, set()) is True

    def test_only_one_side_has_a_variadic(self) -> None:
        source_a = [make_param("args", 0, ParamSpec.VAR_POSITIONAL)]
        source_b = [make_param("y", 0)]
        assert variadic_compatible(source_a, source_b, set()) is True

    def test_same_kind_same_name_already_in_shared_names_is_safe(self) -> None:
        source_a = [make_param("args", 0, ParamSpec.VAR_POSITIONAL)]
        source_b = [make_param("args", 0, ParamSpec.VAR_POSITIONAL)]
        assert variadic_compatible(source_a, source_b, {"args"}) is True

    def test_same_kind_different_names_not_in_shared_names_conflicts(self) -> None:
        source_a = [make_param("extra_args", 0, ParamSpec.VAR_POSITIONAL)]
        source_b = [make_param("things", 0, ParamSpec.VAR_POSITIONAL)]
        assert variadic_compatible(source_a, source_b, set()) is False

    def test_same_kind_same_name_not_yet_in_shared_names_still_conflicts(self) -> None:
        # This function trusts the shared_names set given to it -- it does not
        # independently recognize a shared name as "already resolved" unless
        # the caller included it in shared_names.
        source_a = [make_param("args", 0, ParamSpec.VAR_POSITIONAL)]
        source_b = [make_param("args", 0, ParamSpec.VAR_POSITIONAL)]
        assert variadic_compatible(source_a, source_b, set()) is False

    def test_different_variadic_kinds_do_not_conflict(self) -> None:
        source_a = [make_param("args", 0, ParamSpec.VAR_POSITIONAL)]
        source_b = [make_param("extras", 0, ParamSpec.VAR_KEYWORD)]
        assert variadic_compatible(source_a, source_b, set()) is True

    def test_both_kinds_conflicting_independently_still_reports_conflict(self) -> None:
        source_a = [
            make_param("args_a", 0, ParamSpec.VAR_POSITIONAL),
            make_param("extras_a", 1, ParamSpec.VAR_KEYWORD),
        ]
        source_b = [
            make_param("args_b", 0, ParamSpec.VAR_POSITIONAL),
            make_param("extras_b", 1, ParamSpec.VAR_KEYWORD),
        ]
        assert variadic_compatible(source_a, source_b, set()) is False


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
        assert [(p.name, p.index, p.type) for p in result] == [("a", 0, "int")]

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
