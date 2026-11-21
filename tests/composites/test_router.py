from __future__ import annotations

import asyncio
from types import MappingProxyType
from typing import Any

import pytest

from atomic_agentic.composites import Graph, Pipeline, Router
from atomic_agentic.invokables.prompt_template import PromptTemplate
from atomic_agentic.tools.base import Tool

CALLS: list[str] = []


def pick_kind(kind: Any, **rest: Any) -> Any:
    """Return kind unchanged; extra inputs ride along in the state."""
    return kind


def pick_kind_str(kind: str, **rest: Any) -> str:
    """Return kind unchanged; extra inputs ride along in the state."""
    return kind


def left(x: int = 0) -> str:
    """Left branch."""
    CALLS.append("left")
    return f"left{x}"


def right(x: int = 0) -> str:
    """Right branch."""
    CALLS.append("right")
    return f"right{x}"


def as_list(x: int = 0) -> list[int]:
    """Return a list."""
    CALLS.append("list")
    return [x, x]


def as_int(x: int = 0) -> int:
    """Return an int."""
    return x


def read_route(**state: Any) -> dict[str, Any]:
    """Return the state seen."""
    return dict(state)


def make_tool(function: Any, name: str) -> Tool:
    return Tool(function=function, name=name, description=f"Test tool {name}.")


def build(selector: Any, routes: dict[Any, Any], name: str = "rt") -> Router:
    return Router(name, "Test router.", selector, routes)


@pytest.fixture(autouse=True)
def clear_calls() -> None:
    CALLS.clear()


class TestConstruction:
    def test_rejects_non_dict_routes(self) -> None:
        with pytest.raises(TypeError, match="dict"):
            build(pick_kind, [left])  # type: ignore[arg-type]

    def test_rejects_empty_routes(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            build(pick_kind, {})

    def test_rejects_tuple_selector(self) -> None:
        with pytest.raises(TypeError, match="selector"):
            build((pick_kind, "k"), {"a": left})

    def test_rejects_tuple_route(self) -> None:
        with pytest.raises(TypeError, match="route"):
            build(pick_kind, {"a": (left, "k")})

    def test_accepts_every_entry_form(self) -> None:
        tool = make_tool(left, "mine")
        router = build("{kind}", {"a": tool, "b": "plain text", "c": right})
        assert isinstance(router.selector, PromptTemplate)
        assert router.routes["a"] is tool
        assert isinstance(router.routes["b"], PromptTemplate)
        assert isinstance(router.routes["c"], Tool)

    def test_coerced_names(self) -> None:
        router = build(pick_kind, {"a": left, "b": right}, name="r")
        assert router.selector.name == "r_select"
        assert router.routes["a"].name == "r_route_0"
        assert router.routes["b"].name == "r_route_1"
        assert router.routes["b"].description == "Route 1 of router r"

    def test_selector_and_routes_properties(self) -> None:
        router = build(pick_kind, {"a": left, 2: right})
        assert isinstance(router.routes, MappingProxyType)
        assert list(router.routes) == ["a", 2]
        with pytest.raises(TypeError):
            router.routes["z"] = left  # type: ignore[index]

    def test_parameters_from_selector(self) -> None:
        selector = make_tool(pick_kind, "sel")
        router = build(selector, {"a": left})
        assert router.parameters == selector.parameters

    def test_return_type_common(self) -> None:
        a, b = make_tool(left, "l"), make_tool(right, "r")
        assert build(pick_kind, {"a": a, "b": b}).return_type == a.return_type

    def test_return_type_differs_is_any(self) -> None:
        router = build(pick_kind, {"a": left, "b": as_list})
        assert router.return_type == ("Any",)

    def test_to_dict(self) -> None:
        data = build(pick_kind, {"a": left, 2: right}, name="r").to_dict()
        assert data["selector"]["name"] == "r_select"
        assert [r["key"] for r in data["routes"]] == ["a", 2]
        assert data["routes"][1]["invokable"]["name"] == "r_route_1"

    def test_rejects_invalid_name(self) -> None:
        with pytest.raises(ValueError):
            Router("bad name", "Test router.", pick_kind, {"a": left})


class TestRuntime:
    def test_matching_branch_result_is_returned_raw(self) -> None:
        router = build(pick_kind, {"a": left, "b": as_list, "c": as_int})
        assert router.invoke({"kind": "a", "x": 1}).result == "left1"
        assert router.invoke({"kind": "b", "x": 2}).result == [2, 2]
        assert router.invoke({"kind": "c", "x": 3}).result == 3

    def test_only_chosen_branch_runs(self) -> None:
        router = build(pick_kind, {"a": left, "b": right})
        router.invoke({"kind": "b"})
        assert CALLS == ["right"]

    @pytest.mark.parametrize(
        "key", [1, (1, 2), None, 2.5, frozenset({"z"})], ids=repr
    )
    def test_non_str_keys(self, key: Any) -> None:
        router = build(pick_kind, {key: left, "other": right})
        assert router.invoke({"kind": key}).result == "left0"

    def test_true_and_one_share_a_key(self) -> None:
        router = build(pick_kind, {1: left, "other": right})
        assert router.invoke({"kind": True}).result == "left0"
        assert router.invoke({"kind": 1.0}).result == "left0"

    def test_branch_receives_shared_state(self) -> None:
        router = build(pick_kind, {"a": read_route})
        seen = router.invoke({"kind": "a", "extra": 5}).result
        assert seen["__route__"] == "a"
        assert seen["kind"] == "a"
        assert seen["extra"] == 5

    def test_unknown_value_raises_value_error(self) -> None:
        router = build(pick_kind, {"a": left})
        with pytest.raises(ValueError, match="rt") as info:
            router.invoke({"kind": "zzz"})
        assert "zzz" in str(info.value)
        assert CALLS == []

    def test_unhashable_value_raises_value_error(self) -> None:
        router = build(pick_kind, {"a": left})
        with pytest.raises(ValueError, match="unhashable"):
            router.invoke({"kind": ["a"]})

    def test_async_mirrors_sync(self) -> None:
        router = build(pick_kind, {"a": left, "b": right})
        assert asyncio.run(router.async_invoke({"kind": "b", "x": 4})).result == "right4"
        assert router.invoke({"kind": "b", "x": 4}).result == "right4"
        with pytest.raises(ValueError):
            asyncio.run(router.async_invoke({"kind": "nope"}))

    def test_str_selector_template(self) -> None:
        router = build("{kind}", {"go": left, "stop": right})
        assert router.invoke({"kind": "stop"}).result == "right0"


class TestComposition:
    def test_pipeline_as_branch(self) -> None:
        branch = Pipeline("shape", "Shaped branch.", [(left, "text")])
        router = build(pick_kind, {"a": branch, "b": right})
        assert router.invoke({"kind": "a", "x": 1}).result == {"text": "left1"}
        assert router.invoke({"kind": "b", "x": 1}).result == "right1"

    def test_graph_as_branch(self) -> None:
        inner = Graph(
            "inner",
            "Inner graph.",
            {"n": (left, "out")},
            start="n",
            returns="out",
        )
        router = build(pick_kind_str, {"a": inner})
        assert router.invoke({"kind": "a", "x": 2}).result == "left2"

    def test_pipeline_as_selector(self) -> None:
        selector = Pipeline("sel", "Selector pipeline.", [pick_kind_str])
        router = build(selector, {"a": left, "b": right})
        assert router.invoke({"kind": "a", "x": 1}).result == "left1"

    def test_graph_as_selector(self) -> None:
        selector = Graph(
            "selg",
            "Selector graph.",
            {"n": (pick_kind, "choice")},
            start="n",
            returns="choice",
            parameters=make_tool(pick_kind, "p").parameters,
        )
        router = build(selector, {"a": left, "b": right})
        assert router.invoke({"kind": "b", "x": 1}).result == "right1"

    def test_router_nested_in_pipeline(self) -> None:
        router = build(pick_kind, {"a": left, "b": right})
        pipe = Pipeline("outer", "Outer.", [(router, "routed")])
        assert pipe.invoke({"kind": "a", "x": 1}).result == {"routed": "left1"}


class TestSurface:
    def test_export(self) -> None:
        import atomic_agentic.composites as composites

        assert composites.Router is Router
        assert "Router" in composites.__all__
