# tests/core/test_invokable.py

from __future__ import annotations

import asyncio
from typing import Any, Mapping

import pytest

from atomic_agentic.core.Invokable import AtomicInvokable
from atomic_agentic.core.Parameters import ParamSpec
from atomic_agentic.core.sentinels import NO_VAL


class EchoInvokable(AtomicInvokable):
    """Minimal concrete AtomicInvokable used to test the base contract."""

    def invoke(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return self.filter_inputs(inputs)


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


def make_invokable(
    parameters: list[ParamSpec] | None = None,
    *,
    name: str = "echo",
    description: str = "Echo test invokable.",
    return_type: str = "dict[str, Any]",
    filter_extraneous_inputs: bool = True,
) -> EchoInvokable:
    return EchoInvokable(
        name=name,
        description=description,
        parameters=parameters if parameters is not None else [make_param("x", 0)],
        return_type=return_type,
        filter_extraneous_inputs=filter_extraneous_inputs,
    )


class TestAtomicInvokableConstruction:
    def test_valid_construction_succeeds(self) -> None:
        invokable = make_invokable()

        assert invokable.name == "echo"
        assert invokable.description == "Echo test invokable."
        assert invokable.return_type == "dict[str, Any]"
        assert invokable.filter_extraneous_inputs is True

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

    def test_invalid_parameter_name_raises(self) -> None:
        params = [make_param("bad-name", 0)]

        with pytest.raises(ValueError):
            make_invokable(parameters=params)

    def test_mismatched_parameter_index_raises(self) -> None:
        params = [make_param("x", 1)]

        with pytest.raises(TypeError):
            make_invokable(parameters=params)

    def test_non_string_return_type_raises(self) -> None:
        with pytest.raises(TypeError):
            make_invokable(return_type=dict)  # type: ignore[arg-type]


class TestAtomicInvokableIdentity:
    def test_instance_id_is_non_empty_string(self) -> None:
        invokable = make_invokable()

        assert isinstance(invokable.instance_id, str)
        assert invokable.instance_id

    def test_instance_id_is_stable_for_same_object(self) -> None:
        invokable = make_invokable()

        assert invokable.instance_id == invokable.instance_id

    def test_instance_id_differs_between_instances(self) -> None:
        first = make_invokable()
        second = make_invokable()

        assert first.instance_id != second.instance_id

    def test_full_name_uses_base_class_name_and_name(self) -> None:
        invokable = make_invokable(name="sample")

        assert invokable.full_name == "EchoInvokable.sample"

    def test_name_mutation_updates_full_name(self) -> None:
        invokable = make_invokable(name="before")

        invokable.name = "after"

        assert invokable.name == "after"
        assert invokable.full_name == "EchoInvokable.after"

    def test_invalid_name_mutation_raises(self) -> None:
        invokable = make_invokable()

        with pytest.raises(ValueError):
            invokable.name = "bad-name"


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

        assert "EchoInvokable.echo" in signature
        assert "x: int" in signature
        assert "*args: Any" in signature
        assert "**extras: Any" in signature
        assert "-> dict[str, Any]" in signature


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

        result = invokable.invoke({"x": 1, "y": 2})

        assert result == {"x": 1, "y": 2}

    def test_extraneous_inputs_are_dropped_when_filtering_enabled(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
        ])

        result = invokable.invoke({"x": 1, "unused": 2})

        assert result == {"x": 1}

    def test_extraneous_inputs_raise_when_filtering_disabled_without_varkwargs(self) -> None:
        invokable = make_invokable(
            parameters=[make_param("x", 0)],
            filter_extraneous_inputs=False,
        )

        with pytest.raises(TypeError, match="unexpected input key"):
            invokable.invoke({"x": 1, "unused": 2})

    def test_extraneous_inputs_merge_into_varkwargs(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
            make_param("extras", 1, ParamSpec.VAR_KEYWORD),
        ])

        result = invokable.invoke({"x": 1, "debug": True})

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
        })

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

        result = invokable.invoke({"args": good_value})

        assert result == {"args": good_value}

    def test_valid_explicit_varkwargs_payload_is_retained(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("extras", 0, ParamSpec.VAR_KEYWORD),
        ])

        result = invokable.invoke({"extras": {"x": 1}})

        assert result == {"extras": {"x": 1}}


class TestAtomicInvokableFilterFlag:
    @pytest.mark.parametrize("value", [True, False])
    def test_filter_extraneous_inputs_accepts_bool_at_construction(self, value: bool) -> None:
        invokable = make_invokable(filter_extraneous_inputs=value)

        assert invokable.filter_extraneous_inputs is value

    @pytest.mark.parametrize("value", ["false", "true", 1, 0, None, [], {}])
    def test_filter_extraneous_inputs_rejects_non_bool_at_construction(self, value: object) -> None:
        with pytest.raises(TypeError):
            make_invokable(filter_extraneous_inputs=value)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", [True, False])
    def test_filter_extraneous_inputs_accepts_bool_assignment(self, value: bool) -> None:
        invokable = make_invokable()

        invokable.filter_extraneous_inputs = value

        assert invokable.filter_extraneous_inputs is value

    @pytest.mark.parametrize("value", ["false", "true", 1, 0, None, [], {}])
    def test_filter_extraneous_inputs_rejects_non_bool_assignment(self, value: object) -> None:
        invokable = make_invokable()

        with pytest.raises(TypeError):
            invokable.filter_extraneous_inputs = value  # type: ignore[assignment]


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
    def test_async_invoke_delegates_to_sync_invoke(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0),
        ])

        result = asyncio.run(invokable.async_invoke({"x": 1}))

        assert result == {"x": 1}

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
    def test_to_dict_includes_stable_identity_fields(self) -> None:
        invokable = make_invokable()

        data = invokable.to_dict()

        assert data["instance_id"] == invokable.instance_id

    def test_to_dict_includes_core_metadata(self) -> None:
        invokable = make_invokable(parameters=[
            make_param("x", 0, type_="int"),
        ])

        data = invokable.to_dict()

        assert data["type"] == "EchoInvokable"
        assert data["name"] == "echo"
        assert data["description"] == "Echo test invokable."
        assert data["return_type"] == "dict[str, Any]"
        assert data["filter_extraneous_inputs"] is True
        assert data["parameters"] == [
            {
                "name": "x",
                "index": 0,
                "kind": ParamSpec.POSITIONAL_OR_KEYWORD,
                "type": "int",
            }
        ]

    def test_to_dict_does_not_include_signature(self) -> None:
        invokable = make_invokable()

        data = invokable.to_dict()

        assert "signature" not in data