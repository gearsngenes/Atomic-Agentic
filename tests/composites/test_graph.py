from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any

import pytest

from atomic_agentic.composites import Graph, Pipeline
from atomic_agentic.exceptions import ToolInvocationError
from atomic_agentic.invokables.prompt_template import PromptTemplate
from atomic_agentic.models.parameters import ParamSpec
from atomic_agentic.models.results import GraphResult
from atomic_agentic.tools.base import Tool

GRAPH_LOGGER = "atomic_agentic.composites.graph"


def echo(**kwargs: Any) -> dict[str, Any]:
    """Echo the keyword arguments."""
    return dict(kwargs)


def bump(x: int) -> dict[str, int]:
    """Increase x by one."""
    return {"x": x + 1}


def five(**kwargs: Any) -> int:
    """Return five (a non-mapping)."""
    return 5


def abc(**kwargs: Any) -> dict[str, int]:
    """Return a two-key mapping."""
    return {"a": 1, "b": 2}


def boom(**kwargs: Any) -> dict[str, int]:
    """Always fail."""
    raise ValueError("boom")


def again(x: int) -> Any:
    """Loop back to step until x reaches 3."""
    return "step" if x < 3 else None


def writer(payload: dict[str, Any], log: list[Any] | None = None, label: str = "") -> Any:
    """Build a node that records the state it saw and writes a fixed payload."""

    def node(**kwargs: Any) -> dict[str, Any]:
        """Record the state this node saw and write a payload."""
        if log is not None:
            log.append((label, dict(kwargs)))
        return dict(payload)

    return node


def router_returning(value: Any) -> Any:
    def route(**kwargs: Any) -> Any:
        """Return a fixed routing value."""
        return value

    return route


def build(nodes: dict[str, Any], edges: list[Any], start: Any, **kwargs: Any) -> Graph:
    return Graph("g", "Test graph.", nodes, edges, start, **kwargs)


class TestConstruction:
    def test_rejects_list_nodes(self) -> None:
        with pytest.raises(TypeError, match="dict"):
            build([echo], [], "echo")  # type: ignore[arg-type]

    def test_empty_graph_is_valid(self) -> None:
        graph = Graph("g", "d")
        assert dict(graph.nodes) == {}
        assert graph.start == ()

    def test_empty_nodes_with_a_start_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a node"):
            build({}, [], "n")

    def test_nodes_without_a_start_are_valid(self) -> None:
        assert Graph("g", "d", {"a": echo}).start == ()

    def test_rejects_non_str_node_name(self) -> None:
        with pytest.raises(TypeError):
            build({1: echo}, [], "n")  # type: ignore[dict-item]

    @pytest.mark.parametrize("bad", ["a-b", "1st", "a b"])
    @pytest.mark.parametrize("entry", [echo, "template {x}"])
    def test_rejects_invalid_invokable_name(self, bad: str, entry) -> None:
        with pytest.raises(ValueError, match="alphanumeric/underscore"):
            Graph("g", "d", {bad: entry})

    def test_rejects_wrong_tuple_length(self) -> None:
        with pytest.raises(ValueError, match="length 3"):
            build({"n": (echo, "k", "x")}, [], "n")

    def test_rejects_bad_binding_type(self) -> None:
        with pytest.raises(TypeError):
            build({"n": (echo, 5)}, [], "n")

    @pytest.mark.parametrize("entry", [42, None, object()])
    def test_rejects_unsupported_entry(self, entry: Any) -> None:
        with pytest.raises(TypeError):
            build({"n": entry}, [], "n")

    def test_rejects_non_list_edges(self) -> None:
        with pytest.raises(TypeError, match="list"):
            build({"n": echo}, (), "n")  # type: ignore[arg-type]

    def test_rejects_non_tuple_edge(self) -> None:
        with pytest.raises(TypeError):
            build({"n": echo}, [["n", "n"]], "n")

    def test_rejects_wrong_edge_length(self) -> None:
        with pytest.raises(ValueError, match="length 3"):
            build({"n": echo}, [("n", "n", "n")], "n")

    def test_rejects_unknown_edge_source_and_target(self) -> None:
        with pytest.raises(ValueError, match="source"):
            build({"n": echo}, [("zzz", "n")], "n")
        with pytest.raises(ValueError, match="target"):
            build({"n": echo}, [("n", "zzz")], "n")

    def test_rejects_none_edge_target(self) -> None:
        with pytest.raises(TypeError):
            build({"n": echo}, [("n", None)], "n")

    def test_duplicate_fixed_edges_collapse(self) -> None:
        graph = build({"a": echo, "b": echo}, [("a", "b"), ("a", "b")], "a")
        assert graph.nodes["a"].outgoing == ("b",)

    def test_start_forms(self) -> None:
        assert build({"n": echo}, [], "n").start == ("n",)
        assert build({"a": echo, "b": echo}, [], ["a", "b"]).start == ("a", "b")

    @pytest.mark.parametrize("start", [["a", "a"], ["zzz"], 5])
    def test_rejects_bad_start(self, start: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            build({"a": echo}, [], start)

    @pytest.mark.parametrize("depth", [0, -1])
    def test_rejects_bad_depth_value(self, depth: int) -> None:
        with pytest.raises(ValueError):
            build({"n": echo}, [], "n", max_traversal_depth=depth)

    @pytest.mark.parametrize("depth", [True, "5", 2.5])
    def test_rejects_bad_depth_type(self, depth: Any) -> None:
        with pytest.raises(TypeError):
            build({"n": echo}, [], "n", max_traversal_depth=depth)

    @pytest.mark.parametrize("returns", [5, "", [1], ("a",)])
    def test_rejects_bad_returns(self, returns: Any) -> None:
        with pytest.raises(TypeError):
            build({"n": echo}, [], "n", returns=returns)

    def test_collision_order_validation(self) -> None:
        with pytest.raises(ValueError, match="not a node"):
            build({"a": echo}, [], "a", collision_handling_order=["zzz"])
        with pytest.raises(ValueError, match="repeat"):
            build({"a": echo}, [], "a", collision_handling_order=["a", "a"])
        with pytest.raises(TypeError):
            build({"a": echo}, [], "a", collision_handling_order="a")  # type: ignore[arg-type]

    def test_default_parameters_and_return_type(self) -> None:
        graph = build({"n": echo}, [], "n")
        assert [(p.name, p.kind) for p in graph.parameters] == [("kwargs", ParamSpec.VAR_KEYWORD)]
        assert graph.return_type == ("Any",)

    def test_explicit_parameters_are_used(self) -> None:
        params = [ParamSpec(name="x", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type=("int",))]
        graph = build({"n": bump}, [], "n", parameters=params)
        assert [p.name for p in graph.parameters] == ["x"]

    def test_knob_properties(self) -> None:
        graph = build(
            {"a": echo},
            [],
            "a",
            returns=["x", "y"],
            max_traversal_depth=7,
            collision_handling_order=["a"],
        )
        assert graph.returns == ("x", "y")
        assert graph.max_traversal_depth == 7
        assert graph.collision_handling_order == ("a",)

    def test_nodes_view_is_read_only_and_node_frozen(self) -> None:
        graph = build({"n": echo}, [], "n")
        with pytest.raises(TypeError):
            graph.nodes["x"] = graph.nodes["n"]  # type: ignore[index]
        with pytest.raises(dataclasses.FrozenInstanceError):
            graph.nodes["n"].binding = "k"  # type: ignore[misc]


class TestCoercion:
    def test_callable_is_named_by_key(self) -> None:
        graph = build({"first": echo}, [], "first")
        invokable = graph.nodes["first"].invokable
        assert isinstance(invokable, Tool)
        assert invokable.name == "first"
        assert invokable.description.startswith("Node first of graph g")

    def test_str_becomes_prompt_template(self) -> None:
        graph = build({"hello": "Hello {who}"}, [], "hello", returns="result")
        assert isinstance(graph.nodes["hello"].invokable, PromptTemplate)

    def test_invokable_is_kept_and_may_repeat(self) -> None:
        tool = Tool(function=echo, name="mine", description="Mine.")
        graph = build({"one": tool, "two": (tool, "k")}, [], ["one", "two"])
        assert graph.nodes["one"].invokable is tool
        assert graph.nodes["two"].invokable is tool

    def test_callable_router_is_named_and_invokable_router_is_kept(self) -> None:
        kept = Tool(function=router_returning(None), name="kept", description="Kept.")
        graph = build(
            {"a": echo},
            [("a", router_returning(None)), ("a", kept)],
            "a",
        )
        first, second = graph.nodes["a"].routers
        assert first.name == "g_router_0"
        assert second is kept


class TestSeeding:
    def test_inputs_flow_into_state(self) -> None:
        graph = build({"n": echo}, [], "n")
        assert graph.invoke({"a": 1, "b": 2}).result == {"a": 1, "b": 2}

    def test_named_parameter_collision_raises(self) -> None:
        params = [
            ParamSpec(name="a", index=0, kind=ParamSpec.POSITIONAL_OR_KEYWORD, type=("Any",)),
            ParamSpec(name="rest", index=1, kind=ParamSpec.VAR_KEYWORD, type=("Any",)),
        ]
        graph = build({"n": echo}, [], "n", parameters=params)
        with pytest.raises(TypeError, match="collide"):
            graph.invoke({"a": 1, "rest": {"a": 2}})

    def test_extras_reach_a_kwargs_node(self) -> None:
        log: list[Any] = []
        graph = build({"n": writer({}, log, "n")}, [], "n")
        graph.invoke({"x": 1, "y": 2})
        assert log == [("n", {"x": 1, "y": 2})]


class TestBindings:
    def test_unbound_merges_mapping_whole(self) -> None:
        graph = build({"n": abc}, [], "n")
        assert graph.invoke({}).result == {"a": 1, "b": 2}

    def test_unbound_non_mapping_raises(self) -> None:
        graph = build({"n": five}, [], "n")
        with pytest.raises(TypeError, match="unbound"):
            graph.invoke({})

    def test_str_binding_stores_whole_response(self) -> None:
        graph = build({"n": (five, "k")}, [], "n", returns="k")
        assert graph.invoke({}).result == 5

    def test_list_binding_filters(self) -> None:
        graph = build({"n": (abc, ["a"])}, [], "n")
        assert graph.invoke({}).result == {"a": 1}

    def test_dict_binding_renames(self) -> None:
        graph = build({"n": (abc, {"a": "z"})}, [], "n")
        assert graph.invoke({}).result == {"z": 1}

    @pytest.mark.parametrize("binding", [[], {}])
    def test_empty_binding_writes_nothing_but_still_fires_edges(self, binding: Any) -> None:
        graph = build({"a": (abc, binding), "b": five}, [("a", "b")], "a", returns=None)
        # b is unbound and returns a non-mapping, so reaching it proves the edge fired
        with pytest.raises(TypeError, match="node 'b'"):
            graph.invoke({})

    def test_empty_binding_leaves_state_untouched(self) -> None:
        graph = build({"a": (abc, [])}, [], "a")
        assert graph.invoke({"keep": 1}).result == {"keep": 1}

    def test_missing_selector_is_skipped_with_debug_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        graph = build({"n": (abc, {"a": "a", "zzz": "zzz"})}, [], "n")
        with caplog.at_level(logging.DEBUG, logger=GRAPH_LOGGER):
            assert graph.invoke({}).result == {"a": 1}
        assert any("zzz" in r.getMessage() for r in caplog.records)


class TestSupersteps:
    def test_siblings_read_one_frozen_snapshot(self) -> None:
        log: list[Any] = []
        graph = build(
            {"a": writer({"x": 1}, log, "a"), "b": writer({"y": 2}, log, "b")},
            [],
            ["a", "b"],
        )
        graph.invoke({})
        seen = dict(log)
        assert seen["a"] == {} and seen["b"] == {}

    def test_node_reached_twice_fires_once_per_superstep(self) -> None:
        log: list[Any] = []
        graph = build(
            {"a": echo, "b": echo, "c": writer({}, log, "c")},
            [("a", "c"), ("b", "c")],
            ["a", "b"],
        )
        result = graph.invoke({})
        assert [label for label, _ in log] == ["c"]
        assert result.trace == (("a", "b"), ("c",))

    def test_cycle_refires_in_later_supersteps(self) -> None:
        graph = build({"step": bump}, [("step", again)], "step")
        result = graph.invoke({"x": 0})
        assert result.result == {"x": 3}
        assert result.trace == (("step",), ("step",), ("step",))

    def test_trace_is_declaration_ordered(self) -> None:
        graph = build({"a": echo, "b": echo, "c": echo}, [], ["c", "a"])
        assert graph.invoke({}).trace == (("a", "c"),)


class TestCollisions:
    @staticmethod
    def six(order: Any = None) -> Graph:
        nodes = {
            "a": writer({}),
            "b": writer({"v": "b"}),
            "c": writer({}),
            "d": writer({}),
            "e": writer({"v": "e"}),
            "f": writer({"v": "f"}),
        }
        return build(nodes, [], list(nodes), returns="v", collision_handling_order=order)

    def test_default_is_declaration_order(self) -> None:
        assert self.six().invoke({}).result == "b"

    def test_listed_names_rank_first_in_given_order(self) -> None:
        assert self.six(["f", "e"]).invoke({}).result == "f"
        assert self.six(["e"]).invoke({}).result == "e"

    def test_unlisted_writers_fall_back_to_declaration_order(self) -> None:
        assert self.six(["a", "c"]).invoke({}).result == "b"

    def test_later_supersteps_overwrite(self) -> None:
        graph = build(
            {"a": writer({"v": 1}), "b": writer({"v": 2})},
            [("a", "b")],
            "a",
            returns="v",
            collision_handling_order=["a"],
        )
        assert graph.invoke({}).result == 2


class TestRouters:
    def test_router_reads_post_merge_state(self) -> None:
        def pick(go: str) -> str:
            """Return the node named in state."""
            return go

        graph = build(
            {"s": writer({"go": "t"}), "t": writer({"done": True})},
            [("s", pick)],
            "s",
            returns="done",
        )
        assert graph.invoke({}).result is True

    def test_none_ends_that_path(self) -> None:
        graph = build({"s": echo, "t": echo}, [("s", router_returning(None))], "s")
        assert graph.invoke({}).trace == (("s",),)

    def test_non_str_router_result_raises_type_error(self) -> None:
        graph = build({"s": echo}, [("s", router_returning(5))], "s")
        with pytest.raises(TypeError, match="node name"):
            graph.invoke({})

    def test_unknown_router_name_raises_value_error(self) -> None:
        graph = build({"s": echo}, [("s", router_returning("nope"))], "s")
        with pytest.raises(ValueError, match="not a node"):
            graph.invoke({})

    def test_two_routers_on_one_node_join_one_superstep(self) -> None:
        graph = build(
            {"s": echo, "x": echo, "y": echo},
            [("s", router_returning("x")), ("s", router_returning("y"))],
            "s",
        )
        assert graph.invoke({}).trace == (("s",), ("x", "y"))

    def test_fixed_and_router_targets_join(self) -> None:
        graph = build(
            {"s": echo, "x": echo, "y": echo},
            [("s", "x"), ("s", router_returning("y"))],
            "s",
        )
        assert graph.invoke({}).trace == (("s",), ("x", "y"))


class TestDepthCap:
    def test_empty_queue_on_last_allowed_superstep_is_clean(self) -> None:
        graph = build({"a": echo, "b": echo}, [("a", "b")], "a", max_traversal_depth=2)
        result = graph.invoke({})
        assert result.trace == (("a",), ("b",))

    def test_non_empty_queue_at_cap_raises(self) -> None:
        graph = build(
            {"a": echo, "b": echo, "c": echo},
            [("a", "b"), ("b", "c")],
            "a",
            max_traversal_depth=2,
        )
        with pytest.raises(RuntimeError, match="max_traversal_depth=2"):
            graph.invoke({})

    def test_self_loop_without_exit_raises(self) -> None:
        graph = build({"a": echo}, [("a", "a")], "a", max_traversal_depth=3)
        with pytest.raises(RuntimeError):
            graph.invoke({})


class TestReturns:
    def test_none_is_whole_state(self) -> None:
        assert build({"n": abc}, [], "n").invoke({"z": 0}).result == {"z": 0, "a": 1, "b": 2}

    def test_str_is_one_value(self) -> None:
        assert build({"n": abc}, [], "n", returns="b").invoke({}).result == 2

    def test_list_is_subset_dict(self) -> None:
        assert build({"n": abc}, [], "n", returns=["b"]).invoke({}).result == {"b": 2}

    def test_never_written_key_raises(self) -> None:
        graph = build({"n": abc}, [], "n", returns=["a", "q", "r"])
        with pytest.raises(KeyError, match="q"):
            graph.invoke({})


class TestResultAndModes:
    def test_result_type_and_fields(self) -> None:
        result = build({"n": abc}, [], "n").invoke({})
        assert type(result) is GraphResult
        assert result.trace == (("n",),)

    def test_async_matches_sync(self) -> None:
        graph = build({"step": bump}, [("step", again)], "step")
        sync = graph.invoke({"x": 0})
        asynchronous = asyncio.run(graph.async_invoke({"x": 0}))
        assert sync.result == asynchronous.result
        assert sync.trace == asynchronous.trace

    def test_sync_invoke_inside_a_running_loop(self) -> None:
        graph = build({"n": abc}, [], "n", returns="a")

        async def main() -> Any:
            return graph.invoke({}).result

        assert asyncio.run(main()) == 1

    def test_node_exception_is_not_wrapped_by_the_graph(self) -> None:
        # Tool wraps its own failures; the graph adds no wrapper of its own.
        graph = build({"n": boom}, [], "n")
        with pytest.raises(ToolInvocationError, match="boom") as info:
            graph.invoke({})
        assert "graph" not in str(info.value).lower()

    def test_debug_only_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        graph = build({"a": echo, "b": echo}, [("a", "b")], "a")
        with caplog.at_level(logging.DEBUG, logger=GRAPH_LOGGER):
            graph.invoke({})
        records = [r for r in caplog.records if r.name == GRAPH_LOGGER]
        assert records
        assert all(r.levelno == logging.DEBUG for r in records)


class TestSerialization:
    def test_to_dict(self) -> None:
        graph = build(
            {"a": (abc, {"a": "x"}), "b": echo},
            [("a", "b"), ("a", router_returning(None))],
            ["a"],
            returns=["x"],
            max_traversal_depth=9,
            collision_handling_order=["b"],
        )
        data = graph.to_dict()
        assert data["nodes"]["a"]["binding"] == {"a": "x"}
        assert data["nodes"]["b"]["binding"] is None
        assert data["edges"][0] == ["a", "b"]
        assert data["edges"][1][0] == "a" and isinstance(data["edges"][1][1], dict)
        assert data["start"] == ["a"]
        assert data["returns"] == ["x"]
        assert data["max_traversal_depth"] == 9
        assert data["collision_handling_order"] == ["b"]


class TestSurface:
    def test_export(self) -> None:
        import atomic_agentic.composites as composites

        assert composites.__all__ == ["Graph", "Pipeline", "Router"]
        assert composites.Graph is Graph
        assert composites.Pipeline is Pipeline

    def test_workflows_package_still_imports(self) -> None:
        import atomic_agentic.workflows  # noqa: F401
