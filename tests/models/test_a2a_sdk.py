from __future__ import annotations

import pytest

from atomic_agentic.models.a2a_sdk import A2AtomicSkillMetadata
from atomic_agentic.models.parameters import ParamSpec


def param(*, name: str = "a", index: int = 0, kind: str = ParamSpec.POSITIONAL_OR_KEYWORD,
          type_: str = "int") -> ParamSpec:
    return ParamSpec(name=name, index=index, kind=kind, type=(type_,))


class TestConstruction:
    def test_valid_construction(self) -> None:
        meta = A2AtomicSkillMetadata(
            remote_name="add",
            description="Adds two numbers.",
            extra_description="",
            params=(param(),),
            return_type="int",
        )

        assert meta.remote_name == "add"
        assert meta.description == "Adds two numbers."
        assert meta.params == (param(),)
        assert meta.return_type == "int"

    def test_remote_name_and_return_type_are_stripped(self) -> None:
        meta = A2AtomicSkillMetadata(
            remote_name="  add  ",
            description=None,
            extra_description="",
            params=(),
            return_type="  int  ",
        )

        assert meta.remote_name == "add"
        assert meta.return_type == "int"

    def test_blank_description_normalizes_to_none(self) -> None:
        meta = A2AtomicSkillMetadata(
            remote_name="add", description="   ", extra_description="", params=(), return_type="int"
        )

        assert meta.description is None

    def test_non_str_remote_name_raises(self) -> None:
        with pytest.raises(TypeError, match="remote_name"):
            A2AtomicSkillMetadata(
                remote_name=123, description=None, extra_description="", params=(), return_type="int"  # type: ignore[arg-type]
            )

    def test_empty_remote_name_raises(self) -> None:
        with pytest.raises(ValueError, match="remote_name"):
            A2AtomicSkillMetadata(
                remote_name="   ", description=None, extra_description="", params=(), return_type="int"
            )

    def test_non_str_description_raises(self) -> None:
        with pytest.raises(TypeError, match="description"):
            A2AtomicSkillMetadata(
                remote_name="add", description=123, extra_description="", params=(), return_type="int"  # type: ignore[arg-type]
            )

    def test_non_str_extra_description_raises(self) -> None:
        with pytest.raises(TypeError, match="extra_description"):
            A2AtomicSkillMetadata(
                remote_name="add", description=None, extra_description=None, params=(), return_type="int"  # type: ignore[arg-type]
            )

    def test_non_tuple_params_raises(self) -> None:
        with pytest.raises(TypeError, match="params"):
            A2AtomicSkillMetadata(
                remote_name="add", description=None, extra_description="", params=[param()], return_type="int"  # type: ignore[arg-type]
            )

    def test_params_with_non_paramspec_member_raises(self) -> None:
        with pytest.raises(TypeError, match="params"):
            A2AtomicSkillMetadata(
                remote_name="add", description=None, extra_description="", params=(object(),), return_type="int"  # type: ignore[arg-type]
            )

    def test_empty_return_type_raises(self) -> None:
        with pytest.raises(ValueError, match="return_type"):
            A2AtomicSkillMetadata(
                remote_name="add", description=None, extra_description="", params=(), return_type="   "
            )


class TestToDict:
    def test_round_trips_with_description(self) -> None:
        meta = A2AtomicSkillMetadata(
            remote_name="add",
            description="Adds two numbers.",
            extra_description="extra",
            params=(param(name="a", index=0), param(name="b", index=1)),
            return_type="int",
        )

        d = meta.to_dict()

        assert d == {
            "remote_name": "add",
            "description": "Adds two numbers.",
            "extra_description": "extra",
            "params": [param(name="a", index=0).to_dict(), param(name="b", index=1).to_dict()],
            "return_type": "int",
        }

        rebuilt = A2AtomicSkillMetadata.from_dict(d)
        assert rebuilt == meta

    def test_omits_description_key_when_none(self) -> None:
        meta = A2AtomicSkillMetadata(
            remote_name="add", description=None, extra_description="", params=(), return_type="int"
        )

        d = meta.to_dict()

        assert "description" not in d
        assert A2AtomicSkillMetadata.from_dict(d) == meta


class TestFromDict:
    def test_non_mapping_raises(self) -> None:
        with pytest.raises(TypeError, match="mapping"):
            A2AtomicSkillMetadata.from_dict(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_missing_remote_name_raises(self) -> None:
        with pytest.raises(TypeError, match="remote_name"):
            A2AtomicSkillMetadata.from_dict({"return_type": "int", "params": []})

    def test_missing_return_type_raises(self) -> None:
        with pytest.raises(TypeError, match="return_type"):
            A2AtomicSkillMetadata.from_dict({"remote_name": "add", "params": []})

    def test_params_not_list_raises(self) -> None:
        with pytest.raises(TypeError, match="params"):
            A2AtomicSkillMetadata.from_dict({"remote_name": "add", "return_type": "int", "params": "nope"})

    def test_params_entry_not_mapping_raises(self) -> None:
        with pytest.raises(TypeError, match="params"):
            A2AtomicSkillMetadata.from_dict(
                {"remote_name": "add", "return_type": "int", "params": ["not a mapping"]}
            )

    def test_extra_description_defaults_to_empty_string(self) -> None:
        meta = A2AtomicSkillMetadata.from_dict(
            {"remote_name": "add", "return_type": "int", "params": []}
        )

        assert meta.extra_description == ""
        assert meta.description is None


class TestProtobufIndexCoercion:
    """`MessageToDict` (what `A2AClientHub` decodes an `AgentExtension.params`
    Struct through) has no native int type -- whole numbers always come back
    as float. This is the one boundary `from_dict` coerces at; `ParamSpec`
    itself stays strictly int-only for direct/local construction."""

    def test_whole_number_float_index_coerces_to_int(self) -> None:
        meta = A2AtomicSkillMetadata.from_dict(
            {
                "remote_name": "add",
                "return_type": "int",
                "params": [
                    {"name": "a", "index": 0.0, "kind": ParamSpec.POSITIONAL_OR_KEYWORD, "type": ["int"]},
                    {"name": "b", "index": 1.0, "kind": ParamSpec.POSITIONAL_OR_KEYWORD, "type": ["int"]},
                ],
            }
        )

        assert meta.params[0].index == 0
        assert type(meta.params[0].index) is int
        assert meta.params[1].index == 1
        assert type(meta.params[1].index) is int

    def test_plain_int_index_passes_through_unaffected(self) -> None:
        meta = A2AtomicSkillMetadata.from_dict(
            {
                "remote_name": "add",
                "return_type": "int",
                "params": [{"name": "a", "index": 0, "kind": ParamSpec.POSITIONAL_OR_KEYWORD, "type": ["int"]}],
            }
        )

        assert meta.params[0].index == 0
        assert type(meta.params[0].index) is int

    def test_non_integer_float_index_still_raises(self) -> None:
        # is_integer() is False -- the coercion deliberately does not apply,
        # so ParamSpec.from_dict's own strict int check still rejects it.
        with pytest.raises(TypeError, match="index"):
            A2AtomicSkillMetadata.from_dict(
                {
                    "remote_name": "add",
                    "return_type": "int",
                    "params": [{"name": "a", "index": 0.5, "kind": ParamSpec.POSITIONAL_OR_KEYWORD, "type": ["int"]}],
                }
            )
