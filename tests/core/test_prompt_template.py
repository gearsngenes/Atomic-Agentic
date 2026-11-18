# tests/core/test_prompt_template.py

from __future__ import annotations

import asyncio

import pytest

from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.core.prompt_template import PromptTemplate
from atomic_agentic.exceptions import SchemaError
from atomic_agentic.models.parameters import ParamSpec

POK = ParamSpec.POSITIONAL_OR_KEYWORD


def make(template: str, parameters=None, description: str = "d") -> PromptTemplate:
    return PromptTemplate(
        template=template, name="t", description=description, parameters=parameters
    )


class TestPromptTemplateConstruction:
    def test_static_template_has_no_parameters(self) -> None:
        tpl = make("You are a helpful assistant.", description="static role")

        assert tpl.parameters == []

    def test_single_placeholder_produces_one_parameter(self) -> None:
        tpl = make("You are a {role}.")

        assert len(tpl.parameters) == 1
        assert tpl.parameters[0].name == "role"
        assert tpl.parameters[0].index == 0

    def test_multiple_placeholders_in_template_order(self) -> None:
        tpl = make("You are a {role} expert in {domain}.")

        assert [p.name for p in tpl.parameters] == ["role", "domain"]
        assert [p.index for p in tpl.parameters] == [0, 1]

    def test_duplicate_placeholder_produces_one_parameter(self) -> None:
        tpl = make("Hello {name}, I said hello {name}!")

        assert len(tpl.parameters) == 1
        assert tpl.parameters[0].name == "name"

    def test_discovered_params_are_positional_or_keyword_required_any(self) -> None:
        tpl = make("Say {word}.")
        param = tpl.parameters[0]

        assert param.kind == POK
        assert param.default is NO_VAL
        assert param.type == ("Any",)
        assert param.description is None

    @pytest.mark.parametrize("bad_placeholder", ["{}", "{0}", "{a.b}", "{x[0]}"])
    def test_non_identifier_placeholder_is_not_discovered(self, bad_placeholder: str) -> None:
        tpl = make(f"Hello {bad_placeholder}.")

        assert tpl.parameters == []

    def test_nested_format_spec_field_is_not_discovered(self) -> None:
        assert make("Nested spec: {value:{width}} done.").parameters == []

    def test_invalid_conversion_flag_field_is_not_discovered(self) -> None:
        assert make("Bad: {name!z} here.").parameters == []

    def test_malformed_stray_brace_does_not_raise(self) -> None:
        tpl = make("you are an expert on the history of the { character")

        assert tpl.parameters == []

    def test_template_is_normalized_form(self) -> None:
        tpl = make('Return JSON like { "result": "value" }.')

        assert tpl.template == 'Return JSON like {{ "result": "value" }}.'

    def test_normalizing_again_changes_nothing(self) -> None:
        tpl = make('Hi {name} { "a": 1 }')
        again = make(tpl.template)

        assert again.template == tpl.template
        assert [p.name for p in again.parameters] == ["name"]


class TestExplicitParameters:
    def test_list_order_is_signature_order(self) -> None:
        tpl = make("{a} {b} {c}", parameters=["c", "a", "b"])

        assert [p.name for p in tpl.parameters] == ["c", "a", "b"]
        assert [p.index for p in tpl.parameters] == [0, 1, 2]

    def test_str_and_paramspec_entries_mix(self) -> None:
        spec = ParamSpec(name="b", index=0, kind=POK, type="int", default=3)
        tpl = make("{a} {b}", parameters=["a", spec])

        assert [p.name for p in tpl.parameters] == ["a", "b"]
        assert tpl.parameters[0].kind == POK
        assert tpl.parameters[0].type == ("Any",)
        assert tpl.parameters[1].default == 3

    def test_paramspec_kept_and_reindexed(self) -> None:
        spec = ParamSpec(
            name="b",
            index=7,
            kind=ParamSpec.KEYWORD_ONLY,
            type="int",
            default=5,
            description="the b value",
        )
        tpl = make("{a} {b}", parameters=["a", spec])
        b = tpl.parameters[1]

        assert b.index == 1
        assert b.kind == ParamSpec.KEYWORD_ONLY
        assert b.type == ("int",)
        assert b.default == 5
        assert b.description == "the b value"

    def test_unlisted_placeholder_is_escaped_to_literal(self) -> None:
        tpl = make("{a} and {x}", parameters=["a"])

        assert tpl.template == "{a} and {{x}}"
        assert tpl.invoke({"a": "hi"}).result == "hi and {x}"

    def test_empty_list_gives_no_parameters_and_literal_placeholders(self) -> None:
        tpl = make("{a} and {b}", parameters=[])

        assert tpl.parameters == []
        assert tpl.invoke({}).result == "{a} and {b}"

    def test_positional_only_entry_renders_by_keyword(self) -> None:
        spec = ParamSpec(name="a", index=0, kind=ParamSpec.POSITIONAL_ONLY, type="str")
        tpl = make("Hi {a}.", parameters=[spec])

        assert tpl.parameters[0].kind == ParamSpec.POSITIONAL_ONLY
        assert tpl.invoke({"a": "there"}).result == "Hi there."

    def test_tuple_is_accepted(self) -> None:
        tpl = make("{a} {b}", parameters=("a", "b"))

        assert [p.name for p in tpl.parameters] == ["a", "b"]


class TestConstructionErrors:
    def test_non_str_template_raises(self) -> None:
        with pytest.raises(TypeError, match="template must be a str"):
            make(123)  # type: ignore[arg-type]

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
    def test_blank_template_raises(self, blank: str) -> None:
        with pytest.raises(ValueError):
            make(blank)

    def test_blank_description_raises(self) -> None:
        with pytest.raises(ValueError):
            make("Hello.", description="   ")

    def test_bare_str_parameters_raises(self) -> None:
        with pytest.raises(TypeError):
            make("{a}", parameters="a")  # type: ignore[arg-type]

    def test_non_list_parameters_raises(self) -> None:
        with pytest.raises(TypeError):
            make("{a}", parameters={"a"})  # type: ignore[arg-type]

    def test_entry_neither_str_nor_paramspec_raises(self) -> None:
        with pytest.raises(TypeError):
            make("{a}", parameters=[1])  # type: ignore[list-item]

    @pytest.mark.parametrize("kind", [ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD])
    def test_variadic_paramspec_raises(self, kind: str) -> None:
        spec = ParamSpec(name="a", index=0, kind=kind, type="Any")

        with pytest.raises(ValueError, match="variadic"):
            make("{a}", parameters=[spec])

    def test_listed_name_absent_from_template_raises(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            make("{a}", parameters=["a", "missing"])

    def test_listed_name_only_in_escaped_region_raises(self) -> None:
        with pytest.raises(ValueError, match="'a'"):
            make("Hello {a.b}", parameters=["a"])

    def test_duplicate_listed_name_raises(self) -> None:
        with pytest.raises(TypeError):
            make("{a}", parameters=["a", "a"])

    def test_defaulted_entry_before_required_raises(self) -> None:
        spec = ParamSpec(name="a", index=0, kind=POK, type="str", default="x")

        with pytest.raises(SchemaError):
            make("{a} {b}", parameters=[spec, "b"])

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError):
            PromptTemplate(template="Hello.", name="1bad", description="d")


class TestInvocation:
    def test_static_prompt(self) -> None:
        assert make("You are a helpful assistant.").invoke({}).result == (
            "You are a helpful assistant."
        )

    def test_required_param_filled_from_inputs(self) -> None:
        tpl = make("You are a {role} expert in {domain}.")

        result = tpl.invoke({"role": "Python", "domain": "data science"}).result

        assert result == "You are a Python expert in data science."

    def test_default_fills_when_absent(self) -> None:
        spec = ParamSpec(name="language", index=0, kind=POK, type="str", default="English")
        tpl = make("Respond in {language}.", parameters=[spec])

        assert tpl.invoke({}).result == "Respond in English."

    def test_caller_overrides_default(self) -> None:
        spec = ParamSpec(name="language", index=0, kind=POK, type="str", default="English")
        tpl = make("Respond in {language}.", parameters=[spec])

        assert tpl.invoke({"language": "French"}).result == "Respond in French."

    def test_missing_required_raises_key_error(self) -> None:
        tpl = make("You are a {role}.")

        with pytest.raises(KeyError):
            tpl.invoke({})

    def test_duplicate_placeholder_filled_everywhere(self) -> None:
        tpl = make("Hello {name}, I said hello {name}!")

        assert tpl.invoke({"name": "world"}).result == "Hello world, I said hello world!"

    def test_undeclared_inputs_are_ignored(self) -> None:
        tpl = make("You are a {role}.")

        assert tpl.invoke({"role": "assistant", "irrelevant": "x"}).result == (
            "You are a assistant."
        )

    def test_braces_in_caller_value_pass_through_verbatim(self) -> None:
        tpl = make("Value: {x}")

        assert tpl.invoke({"x": "{y} {{z}}"}).result == "Value: {y} {{z}}"

    @pytest.mark.parametrize(
        ("template", "expected"),
        [
            ("No placeholders at all.", "No placeholders at all."),
            ("{}", "{}"),
            ("{0}", "{0}"),
        ],
    )
    def test_non_identifier_expressions_pass_through_as_literal_text(
        self, template: str, expected: str
    ) -> None:
        assert make(template).invoke({}).result == expected

    def test_mixed_identifier_and_non_identifier(self) -> None:
        tpl = make("Hello {name}! JSON: {'a':1}")

        assert tpl.parameters[0].name == "name"
        assert tpl.invoke({"name": "world"}).result == "Hello world! JSON: {'a':1}"

    def test_malformed_brace_template_renders_with_no_inputs(self) -> None:
        tpl = make("you are an expert on the history of the { character")

        assert tpl.invoke({}).result == "you are an expert on the history of the { character"

    def test_daisy_chain_deferred_field_via_placeholder_shaped_default(self) -> None:
        spec = ParamSpec(
            name="reasoning_trace",
            index=0,
            kind=POK,
            type="str",
            default="{reasoning_trace}",
        )
        tpl = make("Stage one: {reasoning_trace}", parameters=[spec])

        assert tpl.invoke({}).result == "Stage one: {reasoning_trace}"

    def test_async_invoke_returns_same_result(self) -> None:
        tpl = make("Hi {name}.")

        sync = tpl.invoke({"name": "a"}).result
        result = asyncio.run(tpl.async_invoke({"name": "a"})).result

        assert result == sync == "Hi a."

    def test_result_is_str_with_atomic_result_envelope(self) -> None:
        tpl = make("Hi {name}.")
        res = tpl.invoke({"name": "a"})

        assert isinstance(res.result, str)
        assert tpl.return_type == ("str",)
        assert type(res).__name__ == "AtomicResult"
        assert res.invoker_id == tpl.full_name


class TestToDict:
    def test_contains_standard_fields_and_template(self) -> None:
        tpl = make("You are a {role}.", description="my label")
        d = tpl.to_dict()

        assert d["type"] == "PromptTemplate"
        assert d["name"] == "t"
        assert d["description"] == "my label"
        assert d["template"] == "You are a {role}."
        assert d["parameters"] == [p.to_dict() for p in tpl.parameters]
        assert d["return_type"] == ["str"]

    def test_default_included_only_for_optional_params(self) -> None:
        spec = ParamSpec(name="language", index=0, kind=POK, type="str", default="English")
        tpl = make("Respond in {language}.", parameters=[spec])
        assert tpl.to_dict()["parameters"][0]["default"] == "English"

        required = make("You are a {role} expert in {domain}.")
        assert all("default" not in p for p in required.to_dict()["parameters"])
