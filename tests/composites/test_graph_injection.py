from __future__ import annotations

from typing import Any

import pytest

from atomic_agentic.composites import Graph
from atomic_agentic.invokables.prompt_template import PromptTemplate
from atomic_agentic.adapters import to_tool
from atomic_agentic.exceptions import ToolDefinitionError


def inv(fn: Any, name: str = "n") -> Any:
    """Wrap a callable as an invokable."""
    return to_tool(fn, name=name, description=f"{name} test node")


def echo(**kwargs: Any) -> dict[str, Any]:
    """Echo the keyword arguments."""
    return dict(kwargs)


def put_one(**kwargs: Any) -> int:
    """Return one."""
    return 1


def put_two(**kwargs: Any) -> int:
    """Return two."""
    return 2


def work(**state: Any) -> dict[str, int]:
    """Count how many times the body ran."""
    return {"work": state.get("work", 0) + 1}


def make(**state: Any) -> dict[str, int]:
    """Produce the next draft."""
    return {"draft": state.get("draft", 0) + 1}


def review(draft: int) -> str:
    """Approve the third draft."""
    return "approved" if draft >= 3 else "revise"


def base(**kwargs: Any) -> Graph:
    """a -> b, start a."""
    return Graph("g", "d", {"a": echo, "b": echo}, [("a", "b")], "a", **kwargs)


def snap(graph: Graph) -> tuple[Any, ...]:
    return (dict(graph.nodes), list(graph.nodes), graph.start, graph.collision_handling_order)


KINDS = {
    "sequence": (Graph.add_sequence, "start_from", "end_at"),
    "parallel": (Graph.add_parallel, "fan_out", "fan_in"),
}


class TestEmptyGraph:
    def test_running_an_empty_graph_raises(self) -> None:
        with pytest.raises(RuntimeError, match="set_start"):
            Graph("g", "d").invoke({})

    def test_running_nodes_without_a_start_raises(self) -> None:
        with pytest.raises(RuntimeError, match="set_start"):
            Graph("g", "d", {"a": echo}).invoke({})

    def test_build_then_set_start_then_run(self) -> None:
        graph = Graph("g", "d", returns="work")
        graph.add_sequence({"w0": work, "w1": work})
        with pytest.raises(RuntimeError, match="set_start"):
            graph.invoke({})
        graph.set_start("w0")
        assert graph.invoke({}).result == 2


class TestInjectionCommon:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("bad", [["x"], "x", None, 5])
    def test_rejects_non_dict(self, kind: str, bad: Any) -> None:
        method = KINDS[kind][0]
        with pytest.raises(TypeError, match="dict"):
            method(base(), bad)

    @pytest.mark.parametrize("kind", KINDS)
    def test_rejects_empty_dict(self, kind: str) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            KINDS[kind][0](base(), {})

    @pytest.mark.parametrize("kind", KINDS)
    def test_name_clash_lists_every_name_and_changes_nothing(self, kind: str) -> None:
        graph = base()
        before = snap(graph)
        with pytest.raises(ValueError, match=r"already exist.*'a'.*'b'"):
            KINDS[kind][0](graph, {"a": echo, "x": echo, "b": echo})
        assert snap(graph) == before

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("which", [1, 2])
    def test_unknown_bridge_name_changes_nothing(self, kind: str, which: int) -> None:
        method, first, second = KINDS[kind]
        graph = base()
        before = snap(graph)
        with pytest.raises(ValueError, match="not a node"):
            method(graph, {"x": echo}, **{(first if which == 1 else second): "zzz"})
        assert snap(graph) == before

    @pytest.mark.parametrize("kind", KINDS)
    def test_bad_bridge_values(self, kind: str) -> None:
        method, first, _ = KINDS[kind]
        with pytest.raises(ValueError, match="repeat"):
            method(base(), {"x": echo}, **{first: ["a", "a"]})
        with pytest.raises(TypeError):
            method(base(), {"x": echo}, **{first: 5})
        with pytest.raises(TypeError):
            method(base(), {"x": echo}, **{first: [5]})

    @pytest.mark.parametrize("kind", KINDS)
    def test_bad_entry_changes_nothing(self, kind: str) -> None:
        graph = base()
        before = snap(graph)
        with pytest.raises(ToolDefinitionError):
            KINDS[kind][0](graph, {"x": echo, "y": 42})
        assert snap(graph) == before

    @pytest.mark.parametrize("kind", KINDS)
    def test_every_entry_form_is_accepted(self, kind: str) -> None:
        graph = base()
        result = KINDS[kind][0](
            graph, {"p": PromptTemplate(template="Hello {who}", name="p", description="d"), "q": echo, "r": (echo, "k"), "s": inv(echo, "s")}
        )
        assert result is None
        assert list(graph.nodes) == ["a", "b", "p", "q", "r", "s"]
        assert graph.nodes["r"].binding == "k"

    @pytest.mark.parametrize("kind", KINDS)
    def test_only_adds_and_floats_without_bridges(self, kind: str) -> None:
        graph = base(collision_handling_order=["b"])
        KINDS[kind][0](graph, {"x": echo, "y": echo})
        assert graph.start == ("a",)
        assert graph.collision_handling_order == ("b",)
        assert graph.nodes["a"].outgoing == ("b",)
        assert not any(n in node.outgoing for node in (graph.nodes["a"], graph.nodes["b"]) for n in "xy")


class TestAddSequence:
    def _graph(self) -> Graph:
        return Graph("g", "d", {"a": echo, "b": echo, "t": echo}, [("a", "b")], "a")

    def test_chain_and_bridges_as_lists(self) -> None:
        graph = self._graph()
        graph.add_sequence(
            {"x": echo, "y": echo, "z": echo}, start_from=["a", "b"], end_at=["t", "b"]
        )
        assert graph.nodes["x"].outgoing == ("y",)
        assert graph.nodes["y"].outgoing == ("z",)
        assert graph.nodes["z"].outgoing == ("t", "b")
        assert graph.nodes["a"].outgoing == ("b", "x")
        assert graph.nodes["b"].outgoing == ("x",)

    def test_bridge_names_as_single_strings(self) -> None:
        graph = self._graph()
        graph.add_sequence({"x": echo}, start_from="a", end_at="t")
        assert graph.nodes["a"].outgoing == ("b", "x")
        assert graph.nodes["x"].outgoing == ("t",)

    def test_run_through_a_bound_chain(self) -> None:
        def first(x: int) -> int:
            """Add one."""
            return x + 1

        def second(y: int) -> int:
            """Times ten."""
            return y * 10

        graph = Graph("g", "d", returns="z")
        graph.add_sequence({"first": (first, "y"), "second": (second, "z")})
        graph.set_start("first")
        assert graph.invoke({"x": 1}).result == 20

    def test_mid_graph_insertion(self) -> None:
        graph = base()
        graph.remove_edge("a", "b")
        graph.add_sequence({"m": echo}, start_from="a", end_at="b")
        assert graph.invoke({}).trace == (("a",), ("m",), ("b",))

    def test_repeat_until_single_key(self) -> None:
        graph = Graph("g", "d")
        graph.add_sequence({"make": make, "review": (review, "verdict")}, repeat_until={"verdict": "approved"})
        graph.set_start("make")
        result = graph.invoke({}).result
        assert result["draft"] == 3
        assert result["verdict"] == "approved"

    def test_repeat_until_all_keys_must_match(self) -> None:
        def check(work: int) -> dict[str, Any]:
            """Report approval at two runs and a score at three."""
            return {"approved": work >= 2, "score": 10 if work >= 3 else 0}

        graph = Graph("g", "d")
        graph.add_sequence({"work": work, "check": check}, repeat_until={"approved": True, "score": 10})
        graph.set_start("work")
        assert graph.invoke({}).result["work"] == 3

    def test_missing_key_does_not_match_even_for_none(self) -> None:
        graph = Graph("g", "d", max_traversal_depth=6)
        graph.add_sequence({"work": work}, repeat_until={"flag": None})
        graph.set_start("work")
        with pytest.raises(RuntimeError, match="max_traversal_depth"):
            graph.invoke({})

    def test_never_matching_hits_the_cap(self) -> None:
        graph = Graph("g", "d", max_traversal_depth=6)
        graph.add_sequence({"work": work}, repeat_until={"never": 1})
        graph.set_start("work")
        with pytest.raises(RuntimeError, match="max_traversal_depth"):
            graph.invoke({})

    def test_repeat_until_exits_to_end_at(self) -> None:
        graph = Graph("g", "d", {"publish": (echo, "published")})
        graph.add_sequence(
            {"make": make, "review": (review, "verdict")},
            end_at="publish",
            repeat_until={"verdict": "approved"},
        )
        graph.set_start("make")
        result = graph.invoke({})
        assert result.trace[-1] == ("publish",)
        assert "published" in result.result
        assert graph.nodes["review"].outgoing == ()

    def test_repeat_until_without_end_at_ends_the_run(self) -> None:
        graph = Graph("g", "d")
        graph.add_sequence({"make": make, "review": (review, "verdict")}, repeat_until={"verdict": "approved"})
        graph.set_start("make")
        assert graph.invoke({}).trace[-1] == ("review",)

    def test_body_runs_at_least_once(self) -> None:
        graph = Graph("g", "d")
        graph.add_sequence({"work": work}, repeat_until={"go": True})
        graph.set_start("work")
        assert graph.invoke({"go": True}).trace == (("work",),)

    def test_repeat_until_dict_is_copied(self) -> None:
        expected = {"verdict": "approved"}
        graph = Graph("g", "d")
        graph.add_sequence({"make": make, "review": (review, "verdict")}, repeat_until=expected)
        expected["verdict"] = "never"
        graph.set_start("make")
        assert graph.invoke({}).result["draft"] == 3

    @pytest.mark.parametrize(
        ("bad", "error"),
        [({}, ValueError), ({1: "x"}, TypeError), (["verdict"], TypeError), ("verdict", TypeError)],
    )
    def test_rejects_bad_repeat_until(self, bad: Any, error: type[Exception]) -> None:
        graph = base()
        before = snap(graph)
        with pytest.raises(error):
            graph.add_sequence({"x": echo}, repeat_until=bad)
        assert snap(graph) == before

    def test_end_at_list_with_repeat_until_raises(self) -> None:
        graph = self._graph()
        before = snap(graph)
        with pytest.raises(ValueError, match="single node"):
            graph.add_sequence({"x": echo}, end_at=["a", "b"], repeat_until={"k": 1})
        assert snap(graph) == before

    def test_router_names_do_not_repeat(self) -> None:
        graph = Graph("g", "d")
        graph.add_sequence({"one": echo}, repeat_until={"k": 1})
        graph.add_sequence({"two": echo}, repeat_until={"k": 1})
        first = graph.nodes["one"].routers[0].name
        second = graph.nodes["two"].routers[0].name
        assert first != second


class TestAddParallel:
    def _graph(self, **kwargs: Any) -> Graph:
        graph = Graph("g", "d", {"plan": echo, "join": echo}, [], "plan", **kwargs)
        graph.add_parallel(
            {"b0": (put_one, "k"), "b1": (put_two, "k")}, fan_out="plan", fan_in="join"
        )
        return graph

    def test_edges(self) -> None:
        graph = self._graph()
        assert graph.nodes["plan"].outgoing == ("b0", "b1")
        assert graph.nodes["b0"].outgoing == ("join",)
        assert graph.nodes["b1"].outgoing == ("join",)

    def test_fan_out_and_fan_in_lists(self) -> None:
        graph = Graph("g", "d", {"p": echo, "q": echo, "j": echo, "k": echo})
        graph.add_parallel({"x": echo, "y": echo}, fan_out=["p", "q"], fan_in=["j", "k"])
        assert graph.nodes["p"].outgoing == ("x", "y")
        assert graph.nodes["q"].outgoing == ("x", "y")
        assert graph.nodes["x"].outgoing == ("j", "k")

    def test_branches_run_together_and_the_join_runs_once(self) -> None:
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        graph = Graph("g", "d", {"plan": echo, "join": (add, "j")}, [], "plan", returns="j")
        graph.add_parallel(
            {"b0": (put_one, "a"), "b1": (put_two, "b")}, fan_out="plan", fan_in="join"
        )
        result = graph.invoke({})
        assert result.result == 3
        assert result.trace == (("plan",), ("b0", "b1"), ("join",))

    def test_collision_defaults_to_first_declared_and_is_ranked_separately(self) -> None:
        graph = self._graph(returns="k")
        assert graph.invoke({}).result == 1
        graph.set_collision_handling_order(["b1", "b0"])
        assert graph.invoke({}).result == 2


class TestRecipes:
    def test_routing(self) -> None:
        def classify(**state: Any) -> dict[str, str]:
            """Classify the input."""
            return {"kind": "a"}

        def pick(kind: str) -> str:
            """Choose the handler for the kind."""
            return f"handle_{kind}"

        graph = Graph("route", "d", {"classify": classify}, [], "classify", returns="out")
        graph.add_parallel({"handle_a": (put_one, "out"), "handle_b": (put_two, "out")})
        graph.add_edge("classify", pick)
        assert graph.invoke({}).result == 1

    def test_fixed_count_is_an_unrolled_sequence(self) -> None:
        step = inv(work, "work")
        graph = Graph("rep", "d")
        graph.add_sequence({f"w{i}": step for i in range(3)})
        graph.set_start("w0")
        result = graph.invoke({})
        assert result.result["work"] == 3
        assert len(result.trace) == 3
