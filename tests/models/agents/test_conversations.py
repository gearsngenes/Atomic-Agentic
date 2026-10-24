from __future__ import annotations

from types import MappingProxyType

import pytest

from atomic_agentic.models.agents.context import ContextVariable
from atomic_agentic.models.agents.conversations import Conversation


def make_conv(**variables: ContextVariable) -> Conversation:
    return Conversation(name="c", context=dict(variables))


def cv(name: str, value: object = 1, **kwargs: object) -> ContextVariable:
    return ContextVariable(name=name, value=value, **kwargs)  # type: ignore[arg-type]


class TestConversationContextConstruction:
    def test_defaults_to_empty_read_only_view(self) -> None:
        conv = Conversation(name="c")

        assert isinstance(conv.context, MappingProxyType)
        assert len(conv.context) == 0
        with pytest.raises(TypeError):
            conv.context["x"] = cv("x")  # type: ignore[index]

    def test_constructor_copies_and_shares_variables_by_identity(self) -> None:
        var = cv("x", [1])
        source = {"x": var}

        conv = Conversation(name="c", context=source)
        source["y"] = cv("y")

        assert "y" not in conv.context
        assert conv.context["x"] is var
        assert conv.context["x"].value is var.value

    def test_rejects_bad_context(self) -> None:
        with pytest.raises(ValueError, match="Mapping"):
            Conversation(name="c", context=[])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ContextVariable"):
            Conversation(name="c", context={"x": 1})  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="must equal"):
            Conversation(name="c", context={"y": cv("x")})

    def test_equality_and_hash_ignore_context(self) -> None:
        a = Conversation(name="c", context={"x": cv("x")})
        b = Conversation(name="c")

        assert a == b
        assert hash(a) == hash(b)


class TestRemember:
    def test_new_key_defaults_not_readonly(self) -> None:
        conv = Conversation(name="c")

        conv.remember("x", 5, "five")

        assert conv.context["x"] == cv("x", 5, description="five")
        assert conv.context["x"].readonly is False

    def test_readonly_creation_and_key_normalization(self) -> None:
        conv = Conversation(name="c")

        conv.remember(" x ", 1, readonly=True)

        assert conv.context["x"].readonly is True
        with pytest.raises(ValueError):
            conv.remember("1bad", 1)

    def test_replaces_and_clears_description(self) -> None:
        conv = make_conv(x=cv("x", 1, description="old"))

        conv.remember("x", 2)

        assert conv.context["x"].value == 2
        assert conv.context["x"].description is None

    def test_readonly_collision_raises_and_leaves_context(self) -> None:
        conv = make_conv(x=cv("x", 1, readonly=True))
        before = conv.context

        with pytest.raises(ValueError, match="readonly"):
            conv.remember("x", 2)

        assert conv.context is before
        assert conv.context["x"].value == 1

    def test_earlier_view_is_unchanged(self) -> None:
        conv = make_conv(x=cv("x", 1))
        before = conv.context

        conv.remember("y", 2)

        assert "y" not in before
        assert "y" in conv.context


class TestDescribe:
    def test_replaces_description_only(self) -> None:
        var = cv("x", [1], description="old")
        conv = make_conv(x=var)

        conv.describe("x", "new")

        assert conv.context["x"].description == "new"
        assert conv.context["x"].value is var.value

    def test_none_and_blank_clear(self) -> None:
        conv = make_conv(x=cv("x", description="old"))

        conv.describe("x", "   ")

        assert conv.context["x"].description is None

    def test_non_string_raises(self) -> None:
        conv = make_conv(x=cv("x"))

        with pytest.raises(TypeError):
            conv.describe("x", 3)  # type: ignore[arg-type]

    def test_unknown_key_and_readonly_raise(self) -> None:
        conv = make_conv(x=cv("x", readonly=True))

        with pytest.raises(KeyError):
            conv.describe("nope", "d")
        with pytest.raises(ValueError, match="readonly"):
            conv.describe("x", "d")


class TestForget:
    def test_removes_key_and_prior_view_unchanged(self) -> None:
        conv = make_conv(x=cv("x"), y=cv("y"))
        before = conv.context

        conv.forget("x")

        assert "x" not in conv.context
        assert "x" in before

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(KeyError):
            Conversation(name="c").forget("x")

    def test_readonly_can_be_forgotten(self) -> None:
        conv = make_conv(x=cv("x", readonly=True))

        conv.forget("x")

        assert "x" not in conv.context


class TestBulk:
    def test_remember_many_inserts_and_replaces_wholesale(self) -> None:
        conv = make_conv(x=cv("x", 1, description="old"))

        conv.remember_many([cv("x", 2), cv("y", 3)])

        assert conv.context["x"].value == 2
        assert conv.context["x"].description is None
        assert conv.context["y"].value == 3

    @pytest.mark.parametrize(
        "items, exc",
        [
            ([cv("a"), cv("a")], ValueError),
            ([cv("a"), cv("r")], ValueError),
            ([cv("a"), 1], TypeError),
        ],
    )
    def test_remember_many_failure_leaves_same_view(self, items: list, exc: type) -> None:
        conv = make_conv(r=cv("r", readonly=True))
        before = conv.context

        with pytest.raises(exc):
            conv.remember_many(items)

        assert conv.context is before

    def test_forget_many_removes_all_including_readonly_and_dupes(self) -> None:
        conv = make_conv(x=cv("x"), y=cv("y", readonly=True), z=cv("z"))

        conv.forget_many(["x", "y", "x"])

        assert list(conv.context) == ["z"]

    def test_forget_many_unknown_removes_nothing(self) -> None:
        conv = make_conv(x=cv("x"))
        before = conv.context

        with pytest.raises(KeyError):
            conv.forget_many(["x", "nope"])

        assert conv.context is before
