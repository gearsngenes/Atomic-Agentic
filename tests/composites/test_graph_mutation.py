from __future__ import annotations

import logging
from typing import Any, Callable

import pytest

from atomic_agentic.composites import Graph
from atomic_agentic.tools.base import Tool

GRAPH_LOGGER = "atomic_agentic.composites.graph"


def echo(**kwargs: Any) -> dict[str, Any]:
    """Echo the keyword arguments."""
    return dict(kwargs)


def abc(**kwargs: Any) -> dict[str, int]:
    """Return a two-key mapping."""
    return {"a": 1, "b": 2}


def writer(payload: dict[str, Any]) -> Any:
    """Build a node that writes a fixed payload."""

    def node(**kwargs: Any) -> dict[str, Any]:
        """Write a payload."""
        return dict(payload)

    return node


def router_returning(value: Any) -> Any:
    def route(**kwargs: Any) -> Any:
        """Return a fixed routing value."""
        return value

    return route


def build(nodes: dict[str, Any], edges: list[Any], start: Any, **kwargs: Any) -> Graph:
    return Graph("g", "Test graph.", nodes, edges, start, **kwargs)


def abc_graph(**kwargs: Any) -> Graph:
    """a -> b, c unconnected, start a."""
    return build({"a": echo, "b": echo, "c": echo}, [("a", "b")], "a", **kwargs)


def snapshot(graph: Graph) -> tuple[Any, ...]:
    return (
        dict(graph.nodes),
        list(graph.nodes),
        graph.start,
        graph.returns,
        graph.collision_handling_order,
        graph.max_traversal_depth,
    )


class TestAddNode:
    def test_appended_in_declaration_order_and_ranks_last(self) -> None:
        graph = build({"a": writer({"v": "a"})}, [], "a", returns="v")
        graph.add_node("z", writer({"v": "z"}))
        graph.set_start(["z", "a"])
        assert list(graph.nodes) == ["a", "z"]
        assert graph.invoke({}).result == "a"

    def test_rejects_duplicate_name(self) -> None:
        with pytest.raises(ValueError, match="already"):
            abc_graph().add_node("a", echo)

    def test_rejects_bad_name_and_entry(self) -> None:
        graph = abc_graph()
        with pytest.raises(TypeError):
            graph.add_node(1, echo)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            graph.add_node("x", 42)
        with pytest.raises(ValueError, match="length 3"):
            graph.add_node("x", (echo, "k", "extra"))

    def test_coerced_named_by_key_and_not_wired(self) -> None:
        graph = abc_graph(collision_handling_order=["a"])
        graph.add_node("x", echo)
        assert isinstance(graph.nodes["x"].invokable, Tool)
        assert graph.nodes["x"].invokable.name == "x"
        assert graph.nodes["x"].outgoing == () and graph.nodes["x"].routers == ()
        assert "x" not in graph.start
        assert graph.collision_handling_order == ("a",)


class TestUpdateNode:
    def test_replaces_invokable_and_binding_keeping_edges_and_position(self) -> None:
        graph = abc_graph()
        graph.update_node("a", (abc, ["a"]))
        assert graph.nodes["a"].outgoing == ("b",)
        assert list(graph.nodes) == ["a", "b", "c"]
        assert graph.invoke({}).result == {"a": 1}

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="not a node"):
            abc_graph().update_node("zzz", echo)

    def test_bad_entry_raises(self) -> None:
        with pytest.raises(TypeError):
            abc_graph().update_node("a", 42)


class TestRemoveNode:
    def test_removes_leaf_with_its_own_edges_and_routers(self) -> None:
        graph = build(
            {"a": echo, "b": echo, "c": echo},
            [("a", "b"), ("c", "b"), ("c", router_returning(None))],
            "a",
        )
        graph.remove_node("c")
        assert list(graph.nodes) == ["a", "b"]
        assert graph.nodes["b"].outgoing == ()
        assert graph.invoke({}).trace == (("a",), ("b",))

    def test_raises_while_another_node_points_to_it(self) -> None:
        with pytest.raises(ValueError, match="'a'"):
            abc_graph().remove_node("b")

    def test_drops_the_node_from_start_and_collision_order(self) -> None:
        graph = abc_graph(collision_handling_order=["c"])
        graph.remove_node("c")
        assert graph.collision_handling_order is None
        graph.remove_node("a")
        assert graph.start == ()

    def test_removing_the_last_node_leaves_an_empty_graph(self) -> None:
        graph = build({"a": echo}, [], "a")
        graph.remove_node("a")
        assert dict(graph.nodes) == {} and graph.start == ()

    def test_self_loop_does_not_block(self) -> None:
        graph = build({"a": echo, "b": echo}, [("b", "b")], "a")
        graph.remove_node("b")
        assert list(graph.nodes) == ["a"]

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="not a node"):
            abc_graph().remove_node("zzz")

    def test_router_returning_a_removed_name_fails_at_run_time(self) -> None:
        graph = build({"a": echo, "b": echo}, [("a", router_returning("b"))], "a")
        graph.remove_node("b")
        with pytest.raises(ValueError, match="not a node"):
            graph.invoke({})


class TestAddEdge:
    def test_fixed_edge_true_then_false(self) -> None:
        graph = abc_graph()
        assert graph.add_edge("b", "c") is True
        assert graph.add_edge("b", "c") is False
        assert graph.nodes["b"].outgoing == ("c",)
        assert graph.invoke({}).trace == (("a",), ("b",), ("c",))

    def test_unknown_source_or_target_raises(self) -> None:
        graph = abc_graph()
        with pytest.raises(ValueError):
            graph.add_edge("zzz", "a")
        with pytest.raises(ValueError):
            graph.add_edge("a", "zzz")

    def test_callable_routers_are_coerced_with_counter_names(self) -> None:
        graph = abc_graph()
        assert graph.add_edge("a", router_returning(None)) is True
        assert graph.add_edge("a", router_returning(None)) is True
        assert [r.name for r in graph.nodes["a"].routers] == ["g_router_0", "g_router_1"]

    def test_stored_invokable_router_is_not_added_twice(self) -> None:
        graph = abc_graph()
        kept = Tool(function=router_returning(None), name="kept", description="Kept.")
        assert graph.add_edge("a", kept) is True
        assert graph.add_edge("a", kept) is False
        assert graph.nodes["a"].routers == (kept,)

    def test_unsupported_target_raises_without_burning_a_router_name(self) -> None:
        graph = abc_graph()
        for bad in (None, 5):
            with pytest.raises(TypeError):
                graph.add_edge("a", bad)  # type: ignore[arg-type]
        graph.add_edge("a", router_returning(None))
        assert graph.nodes["a"].routers[0].name == "g_router_0"


class TestRemoveEdge:
    def test_fixed_edge_true_then_false(self) -> None:
        graph = abc_graph()
        assert graph.remove_edge("a", "b") is True
        assert graph.remove_edge("a", "b") is False
        assert graph.nodes["a"].outgoing == ()

    def test_router_removed_by_identity(self) -> None:
        graph = abc_graph()
        graph.add_edge("a", router_returning(None))
        stored = graph.nodes["a"].routers[0]
        assert graph.remove_edge("a", stored) is True
        assert graph.remove_edge("a", stored) is False
        assert graph.nodes["a"].routers == ()

    def test_raw_callable_target_raises(self) -> None:
        graph = abc_graph()
        with pytest.raises(TypeError, match="routers"):
            graph.remove_edge("a", router_returning(None))  # type: ignore[arg-type]

    def test_unknown_node_raises(self) -> None:
        graph = abc_graph()
        with pytest.raises(ValueError):
            graph.remove_edge("zzz", "a")
        with pytest.raises(ValueError):
            graph.remove_edge("a", "zzz")

    def test_router_names_stay_unique_after_removal(self) -> None:
        graph = abc_graph()
        graph.add_edge("a", router_returning(None))
        graph.remove_edge("a", graph.nodes["a"].routers[0])
        graph.add_edge("a", router_returning(None))
        assert graph.nodes["a"].routers[0].name == "g_router_1"


class TestDisconnectNode:
    def test_removes_edges_in_both_directions_and_routers(self) -> None:
        graph = build(
            {"a": echo, "b": echo, "c": echo},
            [("a", "b"), ("b", "c"), ("c", "b"), ("b", router_returning(None))],
            "a",
        )
        assert graph.disconnect_node("b") is True
        assert graph.nodes["a"].outgoing == ()
        assert graph.nodes["b"].outgoing == () and graph.nodes["b"].routers == ()
        assert graph.nodes["c"].outgoing == ()
        assert graph.disconnect_node("b") is False

    def test_self_loop_is_removed(self) -> None:
        graph = build({"a": echo, "b": echo}, [("b", "b")], "a")
        assert graph.disconnect_node("b") is True
        assert graph.nodes["b"].outgoing == ()

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="not a node"):
            abc_graph().disconnect_node("zzz")

    def test_lets_a_pointed_to_node_be_removed(self) -> None:
        graph = abc_graph()
        graph.disconnect_node("b")
        graph.remove_node("b")
        assert list(graph.nodes) == ["a", "c"]


class TestKnobs:
    def test_set_start(self) -> None:
        graph = abc_graph()
        graph.set_start("b")
        assert graph.start == ("b",)
        graph.set_start(["c", "a"])
        assert graph.start == ("c", "a")

    @pytest.mark.parametrize("start", [["a", "a"], ["zzz"], 5])
    def test_set_start_rejects_bad_values(self, start: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            abc_graph().set_start(start)

    @pytest.mark.parametrize("empty", [[], None])
    def test_set_start_empty_clears_the_start(self, empty: Any) -> None:
        graph = abc_graph()
        graph.set_start(empty)
        assert graph.start == ()

    def test_set_returns(self) -> None:
        graph = abc_graph()
        graph.set_returns("x")
        assert graph.returns == "x"
        graph.set_returns(["x", "y"])
        assert graph.returns == ("x", "y")
        graph.set_returns(None)
        assert graph.returns is None
        with pytest.raises(TypeError):
            graph.set_returns(5)  # type: ignore[arg-type]

    def test_set_collision_handling_order(self) -> None:
        graph = abc_graph()
        graph.set_collision_handling_order(["b", "a"])
        assert graph.collision_handling_order == ("b", "a")
        graph.set_collision_handling_order(None)
        assert graph.collision_handling_order is None
        with pytest.raises(ValueError):
            graph.set_collision_handling_order(["zzz"])
        with pytest.raises(ValueError):
            graph.set_collision_handling_order(["a", "a"])

    def test_depth_setter(self) -> None:
        graph = abc_graph()
        graph.max_traversal_depth = 7
        assert graph.max_traversal_depth == 7
        for bad, error in ((0, ValueError), (True, TypeError), ("5", TypeError)):
            with pytest.raises(error):
                graph.max_traversal_depth = bad  # type: ignore[assignment]


FAILING_CALLS: list[Callable[[Graph], Any]] = [
    lambda g: g.add_node("a", echo),
    lambda g: g.add_node("x", 42),
    lambda g: g.update_node("zzz", echo),
    lambda g: g.update_node("a", 42),
    lambda g: g.remove_node("b"),
    lambda g: g.add_edge("zzz", "a"),
    lambda g: g.add_edge("a", 5),
    lambda g: g.remove_edge("a", 5),
    lambda g: g.disconnect_node("zzz"),
    lambda g: g.set_returns(5),
    lambda g: g.set_collision_handling_order(["zzz"]),
    lambda g: setattr(g, "max_traversal_depth", 0),
]


class TestAtomicity:
    @pytest.mark.parametrize("call", FAILING_CALLS)
    def test_failed_call_leaves_graph_unchanged(self, call: Callable[[Graph], Any]) -> None:
        graph = abc_graph(collision_handling_order=["c"])
        before = snapshot(graph)
        with pytest.raises((TypeError, ValueError)):
            call(graph)
        assert snapshot(graph) == before


class TestMidRunSafety:
    def test_add_edge_during_a_run_applies_to_the_next_run(self) -> None:
        holder: dict[str, Graph] = {}

        def mutate(**kwargs: Any) -> dict[str, Any]:
            """Add an edge to the graph while it runs."""
            holder["g"].add_edge("b", "c")
            return {}

        graph = build({"a": mutate, "b": echo, "c": echo}, [("a", "b")], "a")
        holder["g"] = graph
        assert graph.invoke({}).trace == (("a",), ("b",))
        assert graph.invoke({}).trace == (("a",), ("b",), ("c",))

    def test_remove_edge_during_a_run_applies_to_the_next_run(self) -> None:
        holder: dict[str, Graph] = {}

        def mutate(**kwargs: Any) -> dict[str, Any]:
            """Remove an edge from the graph while it runs."""
            holder["g"].remove_edge("a", "b")
            return {}

        graph = build({"a": mutate, "b": echo}, [("a", "b")], "a")
        holder["g"] = graph
        assert graph.invoke({}).trace == (("a",), ("b",))
        assert graph.invoke({}).trace == (("a",),)


class TestInvariants:
    def test_name_parameters_and_return_type_never_change(self) -> None:
        graph = abc_graph()
        before = (graph.name, list(graph.parameters), graph.return_type)
        graph.add_node("x", echo)
        graph.add_edge("c", "x")
        graph.update_node("b", (abc, ["a"]))
        graph.disconnect_node("b")
        graph.remove_node("b")
        graph.set_start("c")
        graph.set_returns(None)
        assert (graph.name, list(graph.parameters), graph.return_type) == before

    def test_to_dict_reflects_current_state(self) -> None:
        graph = abc_graph()
        graph.add_node("x", (abc, "k"))
        graph.add_edge("c", "x")
        graph.set_start(["c"])
        graph.set_returns("k")
        graph.max_traversal_depth = 9
        data = graph.to_dict()
        assert list(data["nodes"]) == ["a", "b", "c", "x"]
        assert data["nodes"]["x"]["binding"] == "k"
        assert ["c", "x"] in data["edges"]
        assert data["start"] == ["c"]
        assert data["returns"] == "k"
        assert data["max_traversal_depth"] == 9

    def test_a_mutated_graph_runs(self) -> None:
        graph = build({"a": writer({"v": 1})}, [], "a", returns="w")
        graph.add_node("b", writer({"w": 2}))
        graph.add_edge("a", "b")
        assert graph.invoke({}).result == 2


class TestLogging:
    def test_mutators_log_debug_only(self, caplog: pytest.LogCaptureFixture) -> None:
        graph = abc_graph()
        with caplog.at_level(logging.DEBUG, logger=GRAPH_LOGGER):
            graph.add_node("x", echo)
            graph.add_edge("a", "x")
            graph.add_edge("a", "x")
            graph.remove_edge("a", "x")
            graph.disconnect_node("x")
            graph.update_node("x", echo)
            graph.remove_node("x")
        records = [r for r in caplog.records if r.name == GRAPH_LOGGER]
        assert records
        assert all(r.levelno == logging.DEBUG for r in records)
