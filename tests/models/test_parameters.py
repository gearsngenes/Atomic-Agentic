from __future__ import annotations

from typing import Any, Mapping

import pytest

from atomic_agentic.models.parameters import ParamSpec
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
        assert spec.type == "int"
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
