from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from atomic_agentic.models.workflows.graph import GraphFlowNode, StatePolicySpec
from atomic_agentic.tools.base import Tool


def make_tool(name: str = "node_tool") -> Tool:
    return Tool(
        function=lambda x: x,
        name=name,
        namespace="tests",
        description="A trivial tool.",
    )


class TestGraphFlowNode:
    def test_construction_stores_all_fields(self) -> None:
        invokable = make_tool()
        router = make_tool("router_tool")
        node = GraphFlowNode(
            invokable=invokable,
            incoming=("a", "b"),
            outgoing=("c",),
            routers=(router,),
            priority=2,
        )

        assert node.invokable is invokable
        assert node.incoming == ("a", "b")
        assert node.outgoing == ("c",)
        assert node.routers == (router,)
        assert node.priority == 2

    def test_is_frozen(self) -> None:
        node = GraphFlowNode(
            invokable=make_tool(), incoming=(), outgoing=(), routers=(), priority=1
        )

        with pytest.raises(FrozenInstanceError):
            node.priority = 5  # type: ignore[misc]

    def test_replace_produces_new_instance_with_updated_priority(self) -> None:
        node = GraphFlowNode(
            invokable=make_tool(), incoming=(), outgoing=(), routers=(), priority=1
        )

        updated = replace(node, priority=9)

        assert updated is not node
        assert updated.priority == 9
        assert node.priority == 1
        assert updated.invokable is node.invokable
        assert updated.incoming == node.incoming

    def test_no_type_validation_of_invokable_or_routers(self) -> None:
        # Deliberate, matching CheckerSpec.judge precedent -- construction
        # accepts anything here; validation is GraphFlow.__init__'s job,
        # since this module can't import AtomicInvokable at runtime.
        node = GraphFlowNode(
            invokable="not an invokable",  # type: ignore[arg-type]
            incoming=(),
            outgoing=(),
            routers=("also not an invokable",),  # type: ignore[list-item]
            priority=1,
        )
        assert node.invokable == "not an invokable"
        assert node.routers == ("also not an invokable",)


class TestStatePolicySpec:
    def test_construction_stores_all_fields(self) -> None:
        spec = StatePolicySpec(key="x", raise_on_collision=False, tiebreak="first")

        assert spec.key == "x"
        assert spec.raise_on_collision is False
        assert spec.tiebreak == "first"

    def test_tiebreak_defaults_to_none(self) -> None:
        spec = StatePolicySpec(key="x", raise_on_collision=False)
        assert spec.tiebreak is None

    def test_key_must_be_non_empty_str(self) -> None:
        with pytest.raises(TypeError, match="key must be a non-empty str"):
            StatePolicySpec(key="", raise_on_collision=False)

        with pytest.raises(TypeError, match="key must be a non-empty str"):
            StatePolicySpec(key=123, raise_on_collision=False)  # type: ignore[arg-type]

    def test_raise_on_collision_must_be_bool(self) -> None:
        with pytest.raises(TypeError, match="raise_on_collision must be a bool"):
            StatePolicySpec(key="x", raise_on_collision="yes")  # type: ignore[arg-type]

    def test_raise_on_collision_true_with_tiebreak_raises(self) -> None:
        with pytest.raises(
            ValueError, match="tiebreak must be None when raise_on_collision is True"
        ):
            StatePolicySpec(key="x", raise_on_collision=True, tiebreak="first")

    def test_raise_on_collision_true_with_no_tiebreak_is_valid(self) -> None:
        spec = StatePolicySpec(key="x", raise_on_collision=True)
        assert spec.tiebreak is None

    def test_invalid_tiebreak_value_raises(self) -> None:
        with pytest.raises(ValueError, match="tiebreak must be 'first', 'last', or None"):
            StatePolicySpec(key="x", raise_on_collision=False, tiebreak="middle")

    def test_is_frozen(self) -> None:
        spec = StatePolicySpec(key="x", raise_on_collision=False)
        with pytest.raises(FrozenInstanceError):
            spec.key = "y"  # type: ignore[misc]
