# tests/core/test_invokable.py

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Mapping

import pytest

from atomic_agentic.invokables.atomic_invokable import AtomicInvokable
from atomic_agentic.models.results import AtomicResult
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.constants.core import NO_VAL
from atomic_agentic.exceptions import SchemaError


class EchoInvokable(AtomicInvokable):
    """Minimal concrete AtomicInvokable used to test the base contract."""

    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"result": inputs}

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"result": inputs}


def make_param(
    name: str,
    index: int,
    kind: str = ParamSpec.POSITIONAL_OR_KEYWORD,
    *,
    type_: str = "Any",
    default: Any = NO_VAL,
    description: str | None = None,
) -> ParamSpec:
    return ParamSpec(
        name=name,
        index=index,
        kind=kind,
        type=type_,
        default=default,
        description=description,
    )


def make_invokable(
    parameters: list[ParamSpec] | None = None,
    *,
    name: str = "echo",
    description: str = "Echo test invokable.",
    return_type: str | list[str] | set[str] = "dict[str, Any]",
) -> EchoInvokable:
    return EchoInvokable(
        name=name,
        description=description,
        parameters=parameters if parameters is not None else [make_param("x", 0)],
        return_type=return_type,
    )


class TestAtomicInvokableConstruction:
    def test_valid_construction_succeeds(self) -> None:
        invokable = make_invokable()

        assert invokable.name == "echo"
        assert invokable.description == "Echo test invokable."
        assert invokable.return_type == ("dict[str, Any]",)

    @pytest.mark.parametrize("bad_name", ["", "   ", "bad-name", "123bad"])
    def test_invalid_name_raises(self, bad_name: str) -> None:
        with pytest.raises(ValueError):
            make_invokable(name=bad_name)

    @pytest.mark.parametrize("bad_description", ["", "   "])
    def test_invalid_description_raises(self, bad_description: str) -> None:
        with pytest.raises(ValueError):
            make_invokable(description=bad_description)

    def test_non_list_parameters_raise(self) -> None:
        with pytest.raises(TypeError):
            EchoInvokable(
                name="echo",
                description="Echo test invokable.",
                parameters=("x",),  # type: ignore[arg-type]
                return_type="dict[str, Any]",
            )

    def test_non_paramspec_parameter_raises(self) -> None:
        with pytest.raises(TypeError):
            make_invokable(parameters=["x"])  # type: ignore[list-item]

    def test_duplicate_parameter_names_raise(self) -> None:
        params = [
            make_param("x", 0),
            make_param("x", 1),
        ]

        with pytest.raises(TypeError):
            make_invokable(parameters=params)

    def test_invalid_parameter_name_is_rejected_by_paramspec(self) -> None:
        with pytest.raises(ValueError, match="ParamSpec.name"):
            make_param("bad-name", 0)

    def test_mismatched_parameter_index_raises(self) -> None:
        params = [make_param("x", 1)]

        with pytest.raises(TypeError):
            make_invokable(parameters=params)

    def test_non_string_return_type_raises(self) -> None:
        with pytest.raises(TypeError):
            make_invokable(return_type=dict)  # type: ignore[arg-type]

    def test_return_type_str_is_one_member_and_not_split(self) -> None:
        assert make_invokable(return_type="int | str").return_type == ("int | str",)

    @pytest.mark.parametrize(
        "value",
        [["str", "int", "str"], ("int", "str"), {"str", "int"}, frozenset({"int", "str"})],
    )
    def test_return_type_collection_is_sorted_and_deduplicated(self, value) -> None:
        assert make_invokable(return_type=value).return_type == ("int", "str")

    def test_empty_return_type_raises(self) -> None:
        with pytest.raises(ValueError):
            make_invokable(return_type=[])

    def test_blank_return_type_member_raises(self) -> None:
        with pytest.raises(ValueError):
            make_invokable(return_type=["int", "  "])

    def test_non_str_return_type_member_raises(self) -> None:
        with pytest.raises(TypeError):
            make_invokable(return_type=["int", 3])  # type: ignore[list-item]

    def test_signature_joins_return_type_members_once(self) -> None:
        invokable = make_invokable(name="echo", return_type={"str", "None"})

        assert invokable.signature.endswith(") -> None | str")

    def test_constructor_copies_parameters_list(self) -> None:
        params = [make_param("x", 0)]
        invokable = make_invokable(parameters=params)

        params.append(make_param("y", 1))

        assert len(invokable.parameters) == 1
        assert invokable.parameters[0] is params[0]


class TestAtomicInvokableIdentity:
    def test_repr_is_full_name_only(self) -> None:
        invokable = make_invokable()

        assert repr(invokable) == f"<{invokable.full_name}>"

    def test_no_instance_id_attribute(self) -> None:
        assert not hasattr(make_invokable(), "instance_id")

    def test_full_name_uses_base_class_name_and_name(self) -> None:
        invokable = make_invokable(name="sample")

        assert invokable.full_name == "EchoInvokable.sample"

class TestAtomicInvokableIdentity:
    def test_namespace_argument_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            EchoInvokable(  # type: ignore[call-arg]
                name="echo",
                namespace="grp",
                description="Echo test invokable.",
                parameters=[make_param("x", 0)],
                return_type="str",
            )

    def test_no_namespace_attribute(self) -> None:
        assert not hasattr(make_invokable(), "namespace")

    def test_to_dict_has_no_namespace_key(self) -> None:
        assert "namespace" not in make_invokable().to_dict()


class TestAtomicInvokableParameterContract:
    def test_parameters_returns_copy(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("y", 1),
        ])

        params = invokable.parameters
        params.append(make_param("z", 2))

        assert [param.name for param in invokable.parameters] == ["x", "y"]

    def test_has_varargs_detects_var_positional(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("args", 1, ParamSpec.VAR_POSITIONAL),
        ])

        assert invokable.has_varargs is True
        assert invokable.has_varkwargs is False

    def test_has_varkwargs_detects_var_keyword(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("extras", 1, ParamSpec.VAR_KEYWORD),
        ])

        assert invokable.has_varargs is False
        assert invokable.has_varkwargs is True

    def test_signature_renders_variadic_parameters(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0, type_="int"),
            make_param("args", 1, ParamSpec.VAR_POSITIONAL),
            make_param("extras", 2, ParamSpec.VAR_KEYWORD),
        ])

        signature = invokable.signature

        assert signature.startswith("echo(")
        assert "x: int" in signature
        assert "*args: Any" in signature
        assert "**extras: Any" in signature
        assert "-> dict[str, Any]" in signature


class TestAtomicInvokableSignatureRendering:
    """Covers the `/`/`*` marker logic of `signature`."""

    @staticmethod
    def _render(invokable: EchoInvokable, prop: str) -> str:
        return getattr(invokable, prop)

    @pytest.mark.parametrize("prop", ["signature"])
    def test_bare_slash_after_positional_only_run(self, prop: str) -> None:
        invokable = make_invokable(parameters=[
            make_param("a", 0, ParamSpec.POSITIONAL_ONLY, type_="int"),
            make_param("b", 1, ParamSpec.POSITIONAL_ONLY, type_="int"),
            make_param("c", 2, ParamSpec.POSITIONAL_OR_KEYWORD, type_="int"),
        ])

        rendered = self._render(invokable, prop)

        assert "a: int, b: int, /, c: int" in rendered

    @pytest.mark.parametrize("prop", ["signature"])
    def test_bare_star_before_keyword_only_run(self, prop: str) -> None:
        invokable = make_invokable(parameters=[
            make_param("a", 0, ParamSpec.POSITIONAL_OR_KEYWORD, type_="int"),
            make_param("b", 1, ParamSpec.KEYWORD_ONLY, type_="int"),
            make_param("c", 2, ParamSpec.KEYWORD_ONLY, type_="int"),
        ])

        rendered = self._render(invokable, prop)

        assert "a: int, *, b: int, c: int" in rendered

    @pytest.mark.parametrize("prop", ["signature"])
    def test_var_positional_suppresses_bare_star(self, prop: str) -> None:
        invokable = make_invokable(parameters=[
            make_param("a", 0, ParamSpec.POSITIONAL_OR_KEYWORD, type_="int"),
            make_param("args", 1, ParamSpec.VAR_POSITIONAL, type_="int"),
            make_param("b", 2, ParamSpec.KEYWORD_ONLY, type_="int"),
        ])

        rendered = self._render(invokable, prop)

        # A bare "*" would be redundant right after "*args" -- must not appear.
        assert "*args: int, b: int" in rendered
        assert "*args: int, *, b: int" not in rendered

    @pytest.mark.parametrize("prop", ["signature"])
    def test_all_positional_only_still_emits_closing_slash(self, prop: str) -> None:
        invokable = make_invokable(parameters=[
            make_param("a", 0, ParamSpec.POSITIONAL_ONLY, type_="int"),
            make_param("b", 1, ParamSpec.POSITIONAL_ONLY, type_="int"),
        ])

        rendered = self._render(invokable, prop)

        assert "a: int, b: int, /)" in rendered

    @pytest.mark.parametrize("prop", ["signature"])
    def test_default_value_rendered_regardless_of_kind(self, prop: str) -> None:
        invokable = make_invokable(parameters=[
            make_param("a", 0, ParamSpec.POSITIONAL_OR_KEYWORD, type_="int", default=1),
            make_param("b", 1, ParamSpec.KEYWORD_ONLY, type_="int", default=2),
        ])

        rendered = self._render(invokable, prop)

        assert "a: int = 1" in rendered
        assert "b: int = 2" in rendered

    def test_signature_omits_class_prefix(self) -> None:
        invokable = make_invokable(name="echo")

        assert invokable.signature.startswith("echo(")
        assert "EchoInvokable" not in invokable.signature

    def test_fullname_signature_is_removed(self) -> None:
        assert not hasattr(make_invokable(), "fullname_signature")


class TestAtomicInvokableFiltering:
    def test_filter_inputs_requires_mapping(self) -> None:
        invokable = make_invokable()

        with pytest.raises(TypeError):
            invokable.invoke(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_known_inputs_are_retained(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("y", 1),
        ])

        result = invokable.invoke({"x": 1, "y": 2}).result

        assert result == {"x": 1, "y": 2}

    def test_extraneous_inputs_are_dropped_when_filtering_enabled(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
        ])

        result = invokable.invoke({"x": 1, "unused": 2}).result

        assert result == {"x": 1}

    def test_extraneous_inputs_merge_into_varkwargs(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("extras", 1, ParamSpec.VAR_KEYWORD),
        ])

        result = invokable.invoke({"x": 1, "debug": True}).result

        assert result == {
            "x": 1,
            "extras": {"debug": True},
        }

    def test_explicit_varkwargs_merge_with_extraneous_inputs(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("extras", 1, ParamSpec.VAR_KEYWORD),
        ])

        result = invokable.invoke({
            "x": 1,
            "extras": {"explicit": 2},
            "debug": True,
        }).result

        assert result == {
            "x": 1,
            "extras": {
                "explicit": 2,
                "debug": True,
            },
        }

    @pytest.mark.parametrize("bad_value", ["abc", 123, object()])
    def test_invalid_explicit_varargs_payload_raises(self, bad_value: object) -> None:
        invokable = make_invokable(parameters=[
            make_param("args", 0, ParamSpec.VAR_POSITIONAL),
        ])

        with pytest.raises(TypeError):
            invokable.invoke({"args": bad_value})

    @pytest.mark.parametrize("bad_value", ["abc", 123, [("x", 1)], object()])
    def test_invalid_explicit_varkwargs_payload_raises(self, bad_value: object) -> None:
        invokable = make_invokable(parameters=[
            make_param("extras", 0, ParamSpec.VAR_KEYWORD),
        ])

        with pytest.raises(TypeError):
            invokable.invoke({"extras": bad_value})

    @pytest.mark.parametrize("good_value", [[], (), [1, 2], (1, 2)])
    def test_valid_explicit_varargs_payload_is_retained(self, good_value: list[Any] | tuple[Any, ...]) -> None:
        invokable = make_invokable(parameters=[
            make_param("args", 0, ParamSpec.VAR_POSITIONAL),
        ])

        result = invokable.invoke({"args": good_value}).result

        assert result == {"args": good_value}

    def test_valid_explicit_varkwargs_payload_is_retained(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("extras", 0, ParamSpec.VAR_KEYWORD),
        ])

        result = invokable.invoke({"extras": {"x": 1}}).result

        assert result == {"extras": {"x": 1}}

    def test_absent_param_with_default_is_injected(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("y", 1, default=42),
        ])

        result = invokable.invoke({"x": 1}).result

        assert result == {"x": 1, "y": 42}

    def test_present_param_is_not_overwritten_by_default(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0, default=99),
        ])

        result = invokable.invoke({"x": 7}).result

        assert result == {"x": 7}

    def test_varargs_params_are_not_default_injected(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("args", 1, ParamSpec.VAR_POSITIONAL),
            make_param("extras", 2, ParamSpec.VAR_KEYWORD),
        ])

        result = invokable.invoke({"x": 1}).result

        # VAR_POSITIONAL is absent (no injection, no empty list).
        # VAR_KEYWORD gets an empty dict from the existing varkwarg merge logic,
        # not from default injection — our loop skips both variadic kinds.
        assert result == {"x": 1, "extras": {}}


class TestAtomicInvokableCallBinding:
    def test_call_binds_positional_arguments_in_order(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("y", 1),
        ])

        result = invokable(1, 2)

        assert result == {"x": 1, "y": 2}

    def test_call_binds_keyword_arguments_by_name(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("y", 1),
        ])

        result = invokable(x=1, y=2)

        assert result == {"x": 1, "y": 2}

    def test_call_binds_mixed_positional_and_keyword_arguments(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("y", 1),
        ])

        result = invokable(1, y=2)

        assert result == {"x": 1, "y": 2}

    def test_call_rejects_duplicate_positional_and_keyword_binding(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("y", 1),
        ])

        with pytest.raises(TypeError, match="multiple values"):
            invokable(1, x=2)

    def test_call_rejects_too_many_positional_arguments_without_varargs(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
        ])

        with pytest.raises(TypeError, match="at most 1 positional"):
            invokable(1, 2)

    def test_call_collects_extra_positional_arguments_into_varargs(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("args", 1, ParamSpec.VAR_POSITIONAL),
        ])

        result = invokable(1, 2, 3)

        assert result == {
            "x": 1,
            "args": (2, 3),
        }

    def test_call_rejects_unknown_keywords_without_varkwargs(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
        ])

        with pytest.raises(TypeError, match="unexpected keyword"):
            invokable(1, debug=True)

    def test_call_collects_unknown_keywords_into_varkwargs(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("extras", 1, ParamSpec.VAR_KEYWORD),
        ])

        result = invokable(1, debug=True, retries=2)

        assert result == {
            "x": 1,
            "extras": {
                "debug": True,
                "retries": 2,
            },
        }

    def test_call_rejects_positional_only_parameter_passed_as_keyword(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0, ParamSpec.POSITIONAL_ONLY),
        ])

        with pytest.raises(TypeError, match="positional-only"):
            invokable(x=1)

    def test_positional_only_keyword_is_not_routed_into_varkwargs(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0, ParamSpec.POSITIONAL_ONLY),
            make_param("extras", 1, ParamSpec.VAR_KEYWORD),
        ])

        with pytest.raises(TypeError, match="positional-only"):
            invokable(x=1)

    def test_call_treats_varargs_field_keyword_as_unknown_keyword_without_varkwargs(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("args", 1, ParamSpec.VAR_POSITIONAL),
        ])

        with pytest.raises(TypeError, match="unexpected keyword"):
            invokable(1, args=(2, 3))

    def test_call_treats_varargs_field_keyword_as_unknown_keyword_without_varkwargs(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("args", 1, ParamSpec.VAR_POSITIONAL),
        ])

        with pytest.raises(TypeError, match="unexpected keyword"):
            invokable(1, args=(2, 3))

    def test_call_collects_unknown_keywords_and_varkwarg_field_name_into_varkwargs(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("extras", 1, ParamSpec.VAR_KEYWORD),
        ])

        result = invokable(1, debug=True)

        assert result == {
            "x": 1,
            "extras": {"debug": True},
        }

        result = invokable(1, extras={"debug": True})

        assert result == {
            "x": 1,
            "extras": {
                "extras": {"debug": True},
            },
        }


class TestAtomicInvokableAsync:
    def test_async_invoke_runs_async_hook(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
        ])

        result = asyncio.run(invokable.async_invoke({"x": 1}))

        assert result.result == {"x": 1}

    def test_async_call_uses_call_binding(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("y", 1),
        ])

        result = asyncio.run(invokable.async_call(1, y=2))

        assert result == {"x": 1, "y": 2}

    def test_async_call_rejects_duplicate_binding(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("y", 1),
        ])

        with pytest.raises(TypeError, match="multiple values"):
            asyncio.run(invokable.async_call(1, x=2))


class TestAtomicInvokableSerialization:
    def test_to_dict_has_no_instance_id(self) -> None:
        invokable = make_invokable()

        data = invokable.to_dict()

        assert "instance_id" not in data

    def test_to_dict_includes_core_metadata(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0, type_="int"),
        ])

        data = invokable.to_dict()

        assert data["type"] == "EchoInvokable"
        assert data["name"] == "echo"
        assert data["description"] == "Echo test invokable."
        assert data["return_type"] == ["dict[str, Any]"]
        assert data["parameters"] == [
            {
                "name": "x",
                "index": 0,
                "kind": ParamSpec.POSITIONAL_OR_KEYWORD,
                "type": ["int"],
            }
        ]

    def test_to_dict_does_not_include_signature(self) -> None:
        invokable = make_invokable()

        data = invokable.to_dict()

        assert "signature" not in data


class TestAtomicInvokableDescriptionGetter:
    def test_no_described_params_returns_raw_description(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("y", 1),
        ])

        assert invokable.description == invokable._description

    def test_one_described_param_appends_bullet(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("p1", 0),
            make_param("p2", 1, description="second param"),
        ])

        assert invokable.description == "Echo test invokable.\n\n- p2: second param"

    def test_multiple_described_params_append_in_order(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("p1", 0, description="first"),
            make_param("p2", 1),
            make_param("p3", 2, description="third"),
        ])

        assert invokable.description == "Echo test invokable.\n\n- p1: first\n- p3: third"

    def test_all_described_params_all_appear(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("a", 0, description="alpha"),
            make_param("b", 1, description="beta"),
        ])

        assert invokable.description == "Echo test invokable.\n\n- a: alpha\n- b: beta"

    def test_to_dict_description_is_always_raw(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0, description="has a description"),
        ])

        assert invokable.to_dict()["description"] == invokable._description
        assert invokable.to_dict()["description"] != invokable.description


class _ExtraDescriptionChild(EchoInvokable):
    def _extra_description(self) -> str:
        return "child extra"


class _ExtraDescriptionGrandchild(_ExtraDescriptionChild):
    def _extra_description(self) -> str:
        parent = super()._extra_description()
        return f"{parent}\ngrandchild extra"


class TestExtraDescriptionHook:
    def test_default_extra_description_is_empty(self) -> None:
        invokable = make_invokable()

        assert invokable._extra_description() == ""
        assert invokable.description == invokable._description

    def test_subclass_extra_description_appends_after_base(self) -> None:
        invokable = _ExtraDescriptionChild(
            name="child",
            description="Child base.",
            parameters=[make_param("x", 0)],
            return_type="Any",
        )

        assert invokable.description == "Child base.\nchild extra"

    def test_two_level_super_chaining_preserves_ancestor_extra(self) -> None:
        invokable = _ExtraDescriptionGrandchild(
            name="grandchild",
            description="GC base.",
            parameters=[make_param("x", 0)],
            return_type="Any",
        )

        assert invokable.description == "GC base.\nchild extra\ngrandchild extra"

    def test_described_param_bullets_precede_extra_description(self) -> None:
        invokable = _ExtraDescriptionChild(
            name="child",
            description="Child base.",
            parameters=[make_param("x", 0, description="the x value")],
            return_type="Any",
        )

        assert invokable.description == "Child base.\n\n- x: the x value\nchild extra"


def _two_required() -> EchoInvokable:
    return make_invokable(parameters=[make_param("a", 0), make_param("b", 1)])


class TestAtomicInvokableDefaultMutators:
    # setting
    def test_set_defaults_makes_param_optional(self) -> None:
        invokable = _two_required()

        invokable.set_defaults({"a": 1, "b": 2})

        assert [p.default for p in invokable.parameters] == [1, 2]
        assert invokable.filter_inputs({}) == {"a": 1, "b": 2}

    def test_set_defaults_replaces_existing_default(self) -> None:
        invokable = make_invokable(parameters=[make_param("a", 0, default=1)])

        invokable.set_defaults({"a": 5})

        assert invokable.parameters[0].default == 5
        assert invokable.filter_inputs({}) == {"a": 5}

    def test_set_defaults_returns_none(self) -> None:
        invokable = make_invokable()

        assert invokable.set_defaults({"x": 1}) is None

    def test_set_defaults_shares_value_by_reference(self) -> None:
        invokable = make_invokable()
        shared: list[int] = []

        invokable.set_defaults({"x": shared})

        assert invokable.filter_inputs({})["x"] is shared

    def test_set_defaults_accepts_no_val_as_no_default(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("a", 0, ParamSpec.KEYWORD_ONLY, default=1),
        ])

        invokable.set_defaults({"a": NO_VAL})

        assert invokable.parameters[0].default is NO_VAL

    def test_set_defaults_keyword_only_any_order(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("a", 0, ParamSpec.KEYWORD_ONLY, default=1),
            make_param("b", 1, ParamSpec.KEYWORD_ONLY),
        ])

        invokable.set_defaults({"b": 2})
        assert invokable.parameters[1].default == 2

        invokable.set_defaults({"a": 9})
        assert invokable.parameters[0].default == 9

    def test_set_defaults_empty_mapping_is_noop(self) -> None:
        invokable = _two_required()
        before = invokable.parameters

        invokable.set_defaults({})

        assert invokable.parameters == before

    @pytest.mark.parametrize("bad", [["a"], "a", None])
    def test_set_defaults_non_mapping_raises_type_error(self, bad: Any) -> None:
        invokable = make_invokable()

        with pytest.raises(TypeError, match="must be a mapping"):
            invokable.set_defaults(bad)

    def test_set_defaults_unknown_name_raises_value_error(self) -> None:
        invokable = make_invokable()

        with pytest.raises(ValueError, match="unknown parameter.*nope"):
            invokable.set_defaults({"nope": 1})

    @pytest.mark.parametrize("kind", [ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD])
    def test_set_defaults_variadic_name_raises_value_error(self, kind: str) -> None:
        invokable = make_invokable(parameters=[make_param("rest", 0, kind)])

        with pytest.raises(ValueError, match="variadic") as excinfo:
            invokable.set_defaults({"rest": 1})

        assert "unknown" not in str(excinfo.value)

    def test_set_defaults_order_violation_raises_schema_error_naming_param(self) -> None:
        invokable = _two_required()

        with pytest.raises(SchemaError, match="'b'"):
            invokable.set_defaults({"a": 1})

    def test_set_defaults_batch_judged_on_final_state(self) -> None:
        invokable = _two_required()

        invokable.set_defaults({"a": 1, "b": 2})

        assert [p.default for p in invokable.parameters] == [1, 2]

    def test_set_defaults_failure_changes_nothing(self) -> None:
        invokable = _two_required()
        snapshot = invokable.parameters

        with pytest.raises(ValueError):
            invokable.set_defaults({"a": 1, "nope": 2})

        assert invokable.parameters == snapshot

    def test_set_defaults_reflected_in_signature_and_to_dict(self) -> None:
        invokable = _two_required()

        invokable.set_defaults({"a": 1, "b": 2})

        assert "a: Any = 1, b: Any = 2" in invokable.signature
        assert [p["default"] for p in invokable.to_dict()["parameters"]] == [1, 2]

    # removing
    def test_remove_defaults_makes_param_required(self) -> None:
        invokable = make_invokable(parameters=[make_param("a", 0, default=1)])

        invokable.remove_defaults(["a"])

        assert invokable.parameters[0].default is NO_VAL
        assert invokable.filter_inputs({}) == {}

    def test_remove_defaults_returns_none(self) -> None:
        invokable = make_invokable(parameters=[make_param("a", 0, default=1)])

        assert invokable.remove_defaults(["a"]) is None

    def test_remove_defaults_no_default_raises_value_error(self) -> None:
        invokable = make_invokable()

        with pytest.raises(ValueError, match="have no default"):
            invokable.remove_defaults(["x"])

    def test_remove_defaults_unknown_name_raises_value_error(self) -> None:
        invokable = make_invokable()

        with pytest.raises(ValueError, match="unknown parameter"):
            invokable.remove_defaults(["nope"])

    def test_remove_defaults_variadic_name_raises_value_error(self) -> None:
        invokable = make_invokable(parameters=[make_param("rest", 0, ParamSpec.VAR_POSITIONAL)])

        with pytest.raises(ValueError, match="variadic") as excinfo:
            invokable.remove_defaults(["rest"])

        assert "have no default" not in str(excinfo.value)

    def test_remove_defaults_order_violation_raises_schema_error(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("a", 0, default=1),
            make_param("b", 1, default=2),
        ])

        # Removing only "b" would leave a required positional after a defaulted one.
        with pytest.raises(SchemaError, match="'b'"):
            invokable.remove_defaults(["b"])

        invokable.remove_defaults(["a", "b"])
        assert [p.default for p in invokable.parameters] == [NO_VAL, NO_VAL]

    def test_remove_defaults_failure_changes_nothing(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("a", 0, default=1),
            make_param("b", 1, default=2),
        ])
        snapshot = invokable.parameters

        with pytest.raises(SchemaError):
            invokable.remove_defaults(["b"])

        assert invokable.parameters == snapshot

    def test_remove_defaults_bare_string_raises_type_error(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("a", 0, default=1),
            make_param("b", 1, default=2),
        ])

        with pytest.raises(TypeError, match="bare string"):
            invokable.remove_defaults("ab")

    def test_remove_defaults_non_iterable_raises_type_error(self) -> None:
        invokable = make_invokable(parameters=[make_param("a", 0, default=1)])

        with pytest.raises(TypeError):
            invokable.remove_defaults(5)  # type: ignore[arg-type]

    def test_remove_defaults_empty_is_noop(self) -> None:
        invokable = make_invokable(parameters=[make_param("a", 0, default=1)])
        before = invokable.parameters

        invokable.remove_defaults([])

        assert invokable.parameters == before

    def test_remove_defaults_duplicate_names_harmless(self) -> None:
        invokable = make_invokable(parameters=[make_param("a", 0, default=1)])

        invokable.remove_defaults(["a", "a"])

        assert invokable.parameters[0].default is NO_VAL

    # isolation
    def test_shared_spec_unaffected_when_one_invokable_sets_default(self) -> None:
        spec = make_param("x", 0)
        original_hash = hash(spec)
        first = make_invokable(parameters=[spec])
        second = make_invokable(parameters=[spec], name="other")

        first.set_defaults({"x": 1})

        assert first.parameters[0].default == 1
        assert second.parameters[0].default is NO_VAL
        assert spec.default is NO_VAL
        assert hash(spec) == original_hash

    def test_parameters_copy_not_affected_by_later_set(self) -> None:
        invokable = make_invokable()
        taken = invokable.parameters

        invokable.set_defaults({"x": 1})

        assert taken[0].default is NO_VAL


# ---------------------------------------------------------------- #
# Invocation template, logging, __call__ through the template
# ---------------------------------------------------------------- #
class SpyInvokable(EchoInvokable):
    """Counts filter_inputs calls and records what the hooks receive."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.filter_calls = 0
        self.hook_inputs: list[Any] = []

    def filter_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        self.filter_calls += 1
        return super().filter_inputs(inputs)

    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.hook_inputs.append(inputs)
        return super()._invoke_impl(inputs)

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.hook_inputs.append(inputs)
        return await super()._async_invoke_impl(inputs)


@dataclass(frozen=True, slots=True)
class ExtraResult(AtomicResult):
    extra: str = ""


class ExtraInvokable(EchoInvokable):
    result_cls = ExtraResult

    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"result": inputs, "extra": "e"}

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"result": inputs, "extra": "ae"}


class BadPayloadInvokable(EchoInvokable):
    payload: Any = None

    def _invoke_impl(self, inputs: dict[str, Any]) -> Any:
        return self.payload

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> Any:
        return self.payload


class RaisingInvokable(EchoInvokable):
    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("async boom")


def _build(cls: type[AtomicInvokable], **overrides: Any) -> Any:
    kwargs: dict[str, Any] = dict(
        name="echo",
        description="Test invokable.",
        parameters=[make_param("x", 0)],
        return_type="Any",
    )
    kwargs.update(overrides)
    return cls(**kwargs)


_CORE_LOGGER = "atomic_agentic.invokables.atomic_invokable"


def _records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _CORE_LOGGER]


class TestInvokeTemplate:
    def test_returns_atomic_result_with_envelope(self) -> None:
        invokable = make_invokable()

        result = invokable.invoke({"x": 1})

        assert isinstance(result, AtomicResult)
        assert result.result == {"x": 1}
        assert result.invoker_id == invokable.full_name
        assert result.started_at.tzinfo is not None
        assert result.ended_at.tzinfo is not None
        assert result.started_at <= result.ended_at
        assert result.elapsed_time >= 0

    def test_filter_inputs_runs_once_and_hook_gets_filtered(self) -> None:
        invokable = _build(SpyInvokable)

        invokable.invoke({"x": 1, "unused": 2})

        assert invokable.filter_calls == 1
        assert invokable.hook_inputs == [{"x": 1}]

    def test_extra_payload_keys_reach_result_cls(self) -> None:
        result = _build(ExtraInvokable).invoke({"x": 1})

        assert isinstance(result, ExtraResult)
        assert result.extra == "e"
        assert result.result == {"x": 1}

    def test_payload_without_result_raises_type_error(self) -> None:
        invokable = _build(BadPayloadInvokable)
        invokable.payload = {"other": 1}

        with pytest.raises(TypeError):
            invokable.invoke({"x": 1})

    def test_non_mapping_payload_raises_type_error(self) -> None:
        invokable = _build(BadPayloadInvokable)
        invokable.payload = "not a mapping"

        with pytest.raises(TypeError):
            invokable.invoke({"x": 1})

    def test_hook_exception_propagates_without_finished_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        invokable = _build(RaisingInvokable)

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(RuntimeError, match="boom"):
                invokable.invoke({"x": 1})

        messages = [r.getMessage() for r in _records(caplog)]
        assert any("started" in m for m in messages)
        assert not any("finished" in m for m in messages)

    def test_filter_failure_never_calls_hook(self) -> None:
        invokable = _build(SpyInvokable)

        with pytest.raises(TypeError):
            invokable.invoke(["not", "a", "mapping"])  # type: ignore[arg-type]

        assert invokable.hook_inputs == []

    def test_subclass_with_only_sync_hook_cannot_instantiate(self) -> None:
        class SyncOnly(AtomicInvokable):
            def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
                return {"result": inputs}

        with pytest.raises(TypeError):
            _build(SyncOnly)

    # async mirror
    def test_async_returns_atomic_result_with_envelope(self) -> None:
        invokable = make_invokable()

        result = asyncio.run(invokable.async_invoke({"x": 1}))

        assert isinstance(result, AtomicResult)
        assert result.result == {"x": 1}
        assert result.invoker_id == invokable.full_name
        assert result.started_at <= result.ended_at

    def test_async_filter_runs_once_and_hook_gets_filtered(self) -> None:
        invokable = _build(SpyInvokable)

        asyncio.run(invokable.async_invoke({"x": 1, "unused": 2}))

        assert invokable.filter_calls == 1
        assert invokable.hook_inputs == [{"x": 1}]

    def test_async_extra_payload_keys_reach_result_cls(self) -> None:
        result = asyncio.run(_build(ExtraInvokable).async_invoke({"x": 1}))

        assert isinstance(result, ExtraResult)
        assert result.extra == "ae"

    def test_async_hook_exception_propagates_without_finished_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        invokable = _build(RaisingInvokable)

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(RuntimeError, match="async boom"):
                asyncio.run(invokable.async_invoke({"x": 1}))

        messages = [r.getMessage() for r in _records(caplog)]
        assert any("started" in m for m in messages)
        assert not any("finished" in m for m in messages)

    def test_async_filter_failure_never_calls_hook(self) -> None:
        invokable = _build(SpyInvokable)

        with pytest.raises(TypeError):
            asyncio.run(invokable.async_invoke(["nope"]))  # type: ignore[arg-type]

        assert invokable.hook_inputs == []


class TestNoInvokeLock:
    def test_instances_have_no_invoke_lock(self) -> None:
        assert not hasattr(_build(EchoInvokable), "_invoke_lock")


class TestLogLevel:
    def test_default_is_debug(self) -> None:
        assert make_invokable().log_level == logging.DEBUG

    def test_start_and_finished_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        invokable = make_invokable()

        with caplog.at_level(logging.DEBUG):
            invokable.invoke({"x": 1})

        records = _records(caplog)
        assert [r.levelno for r in records] == [logging.DEBUG, logging.DEBUG]
        assert records[0].getMessage() == f"[{invokable.full_name} started]"
        assert records[1].getMessage() == f"[{invokable.full_name} finished]"

    def test_custom_level_is_used(self, caplog: pytest.LogCaptureFixture) -> None:
        invokable = make_invokable()
        invokable.log_level = logging.INFO

        with caplog.at_level(logging.DEBUG):
            invokable.invoke({"x": 1})

        records = _records(caplog)
        assert len(records) == 2
        assert all(r.levelno == logging.INFO for r in records)

    def test_none_disables_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        invokable = make_invokable()
        invokable.log_level = None

        with caplog.at_level(logging.DEBUG):
            invokable.invoke({"x": 1})
            asyncio.run(invokable.async_invoke({"x": 1}))

        assert _records(caplog) == []

    @pytest.mark.parametrize("bad", ["debug", True, 1.5])
    def test_bad_type_raises_type_error(self, bad: Any) -> None:
        with pytest.raises(TypeError, match="log_level"):
            make_invokable().log_level = bad

    def test_async_messages_have_async_prefix(self, caplog: pytest.LogCaptureFixture) -> None:
        invokable = make_invokable()

        with caplog.at_level(logging.DEBUG):
            asyncio.run(invokable.async_invoke({"x": 1}))

        assert [r.getMessage() for r in _records(caplog)] == [
            f"[Async {invokable.full_name} started]",
            f"[Async {invokable.full_name} finished]",
        ]

    def test_filter_failure_logs_start_but_not_finished(self, caplog: pytest.LogCaptureFixture) -> None:
        invokable = _build(SpyInvokable)

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(TypeError):
                invokable.invoke(["x"])  # type: ignore[arg-type]

        messages = [r.getMessage() for r in _records(caplog)]
        assert messages == [f"[{invokable.full_name} started]"]

    def test_async_filter_failure_logs_start_but_not_finished(self, caplog: pytest.LogCaptureFixture) -> None:
        invokable = _build(SpyInvokable)

        with caplog.at_level(logging.DEBUG):
            with pytest.raises(TypeError):
                asyncio.run(invokable.async_invoke(["x"]))  # type: ignore[arg-type]

        messages = [r.getMessage() for r in _records(caplog)]
        assert messages == [f"[Async {invokable.full_name} started]"]


class TestCallThroughTemplate:
    def test_call_returns_unwrapped_payload(self) -> None:
        assert make_invokable()(1) == {"x": 1}

    def test_call_can_return_atomic_result(self) -> None:
        result = make_invokable()(1, return_atomic_result_object=True)

        assert isinstance(result, AtomicResult)
        assert result.result == {"x": 1}

    def test_async_call_returns_unwrapped_payload(self) -> None:
        assert asyncio.run(make_invokable().async_call(1)) == {"x": 1}

    def test_async_call_can_return_atomic_result(self) -> None:
        result = asyncio.run(make_invokable().async_call(1, return_atomic_result_object=True))

        assert isinstance(result, AtomicResult)
        assert result.result == {"x": 1}
