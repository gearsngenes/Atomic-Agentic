from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any, Mapping

import pytest

from atomic_agentic.models.parameters import ParamSpec, normalize_type_tuple
from atomic_agentic.constants.core import NO_VAL


class TestNormalizeTypeTuple:
    def test_str_is_one_member_and_not_split(self) -> None:
        assert normalize_type_tuple("int | str", label="t") == ("int | str",)

    @pytest.mark.parametrize(
        "value",
        [["str", "int", "str"], ("int", "str"), {"str", "int"}, frozenset({"int", "str"})],
    )
    def test_collections_are_sorted_and_deduplicated(self, value) -> None:
        assert normalize_type_tuple(value, label="t") == ("int", "str")

    def test_members_are_stripped(self) -> None:
        assert normalize_type_tuple(["  int ", "str"], label="t") == ("int", "str")

    def test_empty_collection_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="t must be non-empty"):
            normalize_type_tuple([], label="t")

    def test_blank_member_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-empty strings"):
            normalize_type_tuple(["int", "   "], label="t")

    def test_non_str_member_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="members must be str"):
            normalize_type_tuple(["int", 3], label="t")  # type: ignore[list-item]

    @pytest.mark.parametrize("value", [3, None, {"a": 1}, iter(["int"])])
    def test_wrong_container_raises_type_error(self, value) -> None:
        with pytest.raises(TypeError, match="must be a str, tuple, list or set"):
            normalize_type_tuple(value, label="t")

    def test_label_names_the_field_in_messages(self) -> None:
        with pytest.raises(TypeError, match="my_field"):
            normalize_type_tuple(3, label="my_field")  # type: ignore[arg-type]

    def test_param_spec_type_accepts_a_set(self) -> None:
        spec = ParamSpec(name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type={"str", "int"})
        assert spec.type == ("int", "str")


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


class TestParamSpec:
    def test_paramspec_attribute_view_is_warning_free(self) -> None:
        spec = ParamSpec(
            name="x",
            index=0,
            kind=ParamSpec.POSITIONAL_OR_KEYWORD,
            type="int",
            default=NO_VAL,
        )

        assert spec.name == "x"
        assert spec.index == 0
        assert spec.kind == ParamSpec.POSITIONAL_OR_KEYWORD
        assert spec.type == ("int",)
        assert spec.default is NO_VAL

    def test_paramspec_mapping_getitem_is_not_supported(self) -> None:
        spec = ParamSpec(
            name="x",
            index=0,
            kind=ParamSpec.POSITIONAL_OR_KEYWORD,
            type="int",
            default=NO_VAL,
        )

        with pytest.raises(TypeError):
            _ = spec["name"]  # type: ignore[index]

    def test_paramspec_mapping_helpers_are_not_supported(self) -> None:
        spec = make_param("x", 0, type_="int")

        with pytest.raises(AttributeError):
            spec.get("name")  # type: ignore[attr-defined]

        with pytest.raises(TypeError):
            _ = "default" in spec  # type: ignore[operator]

        with pytest.raises(TypeError):
            list(spec)  # type: ignore[arg-type]

        with pytest.raises(AttributeError):
            spec.keys()  # type: ignore[attr-defined]

        with pytest.raises(AttributeError):
            spec.items()  # type: ignore[attr-defined]

        with pytest.raises(AttributeError):
            spec.values()  # type: ignore[attr-defined]

    def test_paramspec_stores_default_when_present(self) -> None:
        spec = ParamSpec(
            name="limit",
            index=0,
            kind=ParamSpec.POSITIONAL_OR_KEYWORD,
            type="int",
            default=10,
        )

        assert spec.to_dict()["default"] == 10
        assert spec.default == 10

    def test_paramspec_omits_default_key_when_default_is_no_val(self) -> None:
        spec = make_param("x", 0)

        assert "default" not in spec.to_dict()
        assert spec.default is NO_VAL

    def test_replace_default_leaves_original_spec_and_hash_unchanged(self) -> None:
        spec = ParamSpec(
            name="x",
            index=0,
            kind=ParamSpec.POSITIONAL_OR_KEYWORD,
            type="int",
        )
        original_hash = hash(spec)

        new = replace(spec, default=5)

        assert spec.default is NO_VAL
        assert hash(spec) == original_hash
        assert new.default == 5
        assert (new.name, new.index, new.kind, new.type) == (
            spec.name,
            spec.index,
            spec.kind,
            spec.type,
        )

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
            "type": ["str"],
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
            "type": ["int"],
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


class TestParamSpecDescription:
    def test_description_defaults_to_none(self) -> None:
        spec = make_param("x", 0)
        assert spec.description is None

    def test_description_stores_non_empty_string(self) -> None:
        spec = make_param("x", 0, description="the x value")
        assert spec.description == "the x value"

    def test_description_strips_surrounding_whitespace(self) -> None:
        spec = ParamSpec(
            name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD,
            type="Any", description="  hello  ",
        )
        assert spec.description == "hello"

    def test_description_whitespace_only_coerced_to_none(self) -> None:
        spec = ParamSpec(
            name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD,
            type="Any", description="   ",
        )
        assert spec.description is None

    def test_description_empty_string_coerced_to_none(self) -> None:
        spec = ParamSpec(
            name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD,
            type="Any", description="",
        )
        assert spec.description is None

    def test_description_none_explicit_accepted(self) -> None:
        spec = ParamSpec(
            name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD,
            type="Any", description=None,
        )
        assert spec.description is None

    def test_description_wrong_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            ParamSpec(
                name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                type="Any", description=123,  # type: ignore[arg-type]
            )

    def test_to_dict_omits_description_key_when_none(self) -> None:
        spec = make_param("x", 0)
        assert "description" not in spec.to_dict()

    def test_to_dict_includes_description_key_when_set(self) -> None:
        spec = make_param("x", 0, description="docs")
        assert spec.to_dict()["description"] == "docs"

    def test_to_dict_round_trip_with_description(self) -> None:
        spec = make_param("x", 0, type_="str", description="the value")
        data = spec.to_dict()
        assert data == {
            "name": "x",
            "index": 0,
            "kind": ParamSpec.POSITIONAL_OR_KEYWORD,
            "type": ["str"],
            "description": "the value",
        }
        round_trip = ParamSpec.from_dict(data)
        assert round_trip.description == "the value"

    def test_from_dict_absent_description_key_yields_none(self) -> None:
        spec = ParamSpec.from_dict({
            "name": "x", "index": 0,
            "kind": ParamSpec.POSITIONAL_OR_KEYWORD, "type": "Any",
        })
        assert spec.description is None

    def test_to_parameter_list_copy_preserves_description(self) -> None:
        from atomic_agentic.utils.parameters import to_parameter_list

        original = [make_param("x", 0, description="kept")]
        result = to_parameter_list(original)
        assert result[0].description == "kept"

    def test_to_parameter_list_string_schema_yields_none_description(self) -> None:
        from atomic_agentic.utils.parameters import to_parameter_list

        result = to_parameter_list(["x", "y"])
        assert all(p.description is None for p in result)
