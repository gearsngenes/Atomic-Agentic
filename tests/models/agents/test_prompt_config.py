# tests/models/agents/test_prompt_config.py

from __future__ import annotations

import pytest

from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.constants.core import NO_VAL


class TestPromptConfigConstruction:
    def test_static_prompt_has_empty_parameters(self) -> None:
        cfg = PromptConfig(template="You are a helpful assistant.", description="static role")

        assert cfg.parameters == ()

    def test_single_placeholder_produces_one_paramspec(self) -> None:
        cfg = PromptConfig(template="You are a {role}.", description="d")

        assert len(cfg.parameters) == 1
        assert cfg.parameters[0].name == "role"
        assert cfg.parameters[0].index == 0

    def test_multiple_placeholders_in_template_order(self) -> None:
        cfg = PromptConfig(template="You are a {role} expert in {domain}.", description="d")

        names = [p.name for p in cfg.parameters]
        assert names == ["role", "domain"]
        assert cfg.parameters[0].index == 0
        assert cfg.parameters[1].index == 1

    def test_duplicate_placeholder_produces_single_paramspec(self) -> None:
        cfg = PromptConfig(template="Hello {name}, I said hello {name}!", description="d")

        assert len(cfg.parameters) == 1
        assert cfg.parameters[0].name == "name"
        assert cfg.parameters[0].index == 0

    def test_discovered_params_are_keyword_only(self) -> None:
        from atomic_agentic.models.parameters import ParamSpec
        cfg = PromptConfig(template="Say {word}.", description="d")

        assert cfg.parameters[0].kind == ParamSpec.KEYWORD_ONLY

    def test_discovered_params_are_required_by_default(self) -> None:
        cfg = PromptConfig(template="Hello {name}.", description="d")

        assert cfg.parameters[0].default is NO_VAL

    def test_defaults_sets_default_on_matching_param(self) -> None:
        cfg = PromptConfig(
            template="Respond in {language}.",
            description="d",
            defaults={"language": "English"},
        )

        assert cfg.parameters[0].default == "English"

    def test_partial_defaults_leave_others_required(self) -> None:
        cfg = PromptConfig(
            template="You are a {role} speaking {language}.",
            description="d",
            defaults={"language": "English"},
        )

        role_param = next(p for p in cfg.parameters if p.name == "role")
        lang_param = next(p for p in cfg.parameters if p.name == "language")
        assert role_param.default is NO_VAL
        assert lang_param.default == "English"

    def test_extra_default_key_not_in_template_raises(self) -> None:
        with pytest.raises(ValueError, match="not found in template"):
            PromptConfig(template="Hello {name}.", description="d", defaults={"typo": "x"})

    def test_non_dict_defaults_raises(self) -> None:
        with pytest.raises(TypeError, match="defaults must be a dict or None"):
            PromptConfig(template="Hello {name}.", description="d", defaults=["bad"])  # type: ignore[arg-type]

    def test_none_defaults_accepted(self) -> None:
        cfg = PromptConfig(template="Hello {name}.", description="d", defaults=None)

        assert cfg.parameters[0].default is NO_VAL

    @pytest.mark.parametrize("bad_placeholder", ["{}", "{0}", "{a.b}", "{x[0]}"])
    def test_non_identifier_placeholder_is_silently_skipped(self, bad_placeholder: str) -> None:
        cfg = PromptConfig(template=f"Hello {bad_placeholder}.", description="d")
        assert cfg.parameters == ()

    def test_non_str_template_raises(self) -> None:
        with pytest.raises(TypeError, match="template must be a str"):
            PromptConfig(template=123, description="d")  # type: ignore[arg-type]

    def test_non_str_description_raises(self) -> None:
        with pytest.raises(TypeError, match="description must be a str"):
            PromptConfig(template="Hello {name}.", description=None)  # type: ignore[arg-type]


class TestPromptConfigRender:
    def test_static_prompt_renders(self) -> None:
        cfg = PromptConfig(template="You are a helpful assistant.", description="d")

        assert cfg.render({}) == "You are a helpful assistant."

    def test_required_param_filled_from_inputs(self) -> None:
        cfg = PromptConfig(
            template="You are a {role} expert in {domain}.",
            description="d",
        )

        result = cfg.render({"role": "Python", "domain": "data science"})

        assert result == "You are a Python expert in data science."

    def test_optional_param_filled_from_default_when_absent(self) -> None:
        cfg = PromptConfig(
            template="Respond in {language}.",
            description="d",
            defaults={"language": "English"},
        )

        assert cfg.render({}) == "Respond in English."

    def test_caller_value_overrides_default(self) -> None:
        cfg = PromptConfig(
            template="Respond in {language}.",
            description="d",
            defaults={"language": "English"},
        )

        assert cfg.render({"language": "French"}) == "Respond in French."

    def test_required_param_absent_raises(self) -> None:
        cfg = PromptConfig(template="You are a {role}.", description="d")

        with pytest.raises(ValueError, match="required parameter 'role' is missing"):
            cfg.render({})

    def test_duplicate_placeholder_filled_everywhere(self) -> None:
        cfg = PromptConfig(template="Hello {name}, I said hello {name}!", description="d")

        assert cfg.render({"name": "world"}) == "Hello world, I said hello world!"

    def test_undeclared_inputs_are_ignored(self) -> None:
        cfg = PromptConfig(template="You are a {role}.", description="d")

        result = cfg.render({"role": "assistant", "irrelevant": "ignored"})

        assert result == "You are a assistant."

    @pytest.mark.parametrize(
        ("template", "expected"),
        [
            ("example json {'a':b} and {INPUT_HERE}", "example json {'a':b} and {INPUT_HERE}"),
            ("No placeholders at all.", "No placeholders at all."),
            ("{}", "{}"),
            ("{0}", "{0}"),
        ],
    )
    def test_non_identifier_expressions_pass_through_as_literal_text(
        self, template: str, expected: str
    ) -> None:
        cfg = PromptConfig(template=template, description="d")
        assert cfg.render({}) == expected

    def test_mixed_identifier_and_non_identifier_in_one_template(self) -> None:
        cfg = PromptConfig(template="Hello {name}! JSON: {'a':1}", description="d")
        assert cfg.parameters[0].name == "name"
        assert cfg.render({"name": "world"}) == "Hello world! JSON: {'a':1}"


class TestPromptConfigToDict:
    def test_to_dict_returns_template_description_and_empty_defaults(self) -> None:
        cfg = PromptConfig(template="You are a {role}.", description="my label")
        result = cfg.to_dict()
        assert result == {"template": "You are a {role}.", "description": "my label", "defaults": {}}

    def test_to_dict_includes_defaults_for_optional_params(self) -> None:
        cfg = PromptConfig(
            template="Respond in {language}.",
            description="lang",
            defaults={"language": "English"},
        )
        assert cfg.to_dict()["defaults"] == {"language": "English"}

    def test_to_dict_excludes_required_params_from_defaults(self) -> None:
        cfg = PromptConfig(template="You are a {role} expert in {domain}.", description="d")
        assert cfg.to_dict()["defaults"] == {}
