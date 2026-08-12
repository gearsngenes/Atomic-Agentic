from __future__ import annotations

import asyncio
from typing import Any

import pytest

from atomic_agentic.exceptions import ExecutionError, ValidationError, WorkflowError
from atomic_agentic.models.results.workflows import GraphFlowResult
from atomic_agentic.models.workflows import GraphFlowNode, StatePolicySpec
from atomic_agentic.tools.base import Tool
from atomic_agentic.workflows.graph import GraphFlow


def T(fn: Any, name: str) -> Tool:
    """Shorthand for a deterministic Tool node/router, no LLM involved."""
    return Tool(function=fn, name=name, namespace="tests", description=f"node {name}.")


def make_node(name: str = "start") -> Tool:
    def fn(x: int) -> dict:
        return {"x": x + 1}

    return T(fn, name)


def make_graph_flow(
    *,
    nodes: dict[str, Any] | None = None,
    edges: list[tuple] | None = None,
    start: str = "start",
    state_policies: list[StatePolicySpec] | None = None,
    priorities: dict[str, int] | None = None,
    max_edge_traversals: int | None = None,
    stop_early: bool = False,
    result_mode: str = GraphFlow.LIST,
    return_keys: list[str] | None = None,
    name: str = "graph_flow",
) -> GraphFlow:
    resolved_nodes = nodes if nodes is not None else {"start": make_node("start")}
    resolved_edges = edges if edges is not None else []
    return GraphFlow(
        name=name,
        namespace="tests",
        description="Graph test flow.",
        nodes=resolved_nodes,
        edges=resolved_edges,
        start=start,
        state_policies=state_policies,
        priorities=priorities,
        max_edge_traversals=max_edge_traversals,
        stop_early=stop_early,
        result_mode=result_mode,
        return_keys=return_keys,
    )


class TestGraphFlowConstruction:
    def test_nodes_must_be_dict(self) -> None:
        with pytest.raises(TypeError, match="nodes must be a dict"):
            make_graph_flow(nodes=[make_node("start")])  # type: ignore[arg-type]

    def test_nodes_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError, match="nodes must not be empty"):
            make_graph_flow(nodes={})

    def test_nodes_values_must_be_atomic_invokable(self) -> None:
        with pytest.raises(TypeError, match="must be AtomicInvokable"):
            make_graph_flow(nodes={"start": object()})  # type: ignore[dict-item]

    def test_start_must_be_str(self) -> None:
        with pytest.raises(TypeError, match="start must be a str"):
            make_graph_flow(start=123)  # type: ignore[arg-type]

    def test_start_must_be_a_configured_node(self) -> None:
        with pytest.raises(KeyError, match="is not a configured node"):
            make_graph_flow(start="missing")

    def test_edges_must_be_a_list(self) -> None:
        with pytest.raises(TypeError, match="edges must be a list"):
            make_graph_flow(edges=())  # type: ignore[arg-type]

    def test_edge_must_be_a_two_tuple(self) -> None:
        with pytest.raises(TypeError, match="2-tuple"):
            make_graph_flow(
                nodes={"start": make_node("start"), "next": make_node("next")},
                edges=[("start", "next", "extra")],  # type: ignore[list-item]
            )

    def test_edge_source_must_be_str(self) -> None:
        with pytest.raises(TypeError, match="edge source must be a str"):
            make_graph_flow(edges=[(123, "start")])  # type: ignore[list-item]

    def test_edge_source_must_be_configured_node(self) -> None:
        with pytest.raises(KeyError, match="edge source"):
            make_graph_flow(edges=[("missing", None)])

    def test_edge_target_str_must_be_configured_node(self) -> None:
        with pytest.raises(KeyError, match="edge target"):
            make_graph_flow(edges=[("start", "missing")])

    def test_edge_target_must_be_str_none_or_atomic_invokable(self) -> None:
        with pytest.raises(TypeError, match="edge target must be"):
            make_graph_flow(edges=[("start", 123)])  # type: ignore[list-item]

    def test_leaf_marker_cannot_combine_with_real_edge(self) -> None:
        with pytest.raises(ValueError, match="cannot combine a leaf marker"):
            make_graph_flow(
                nodes={"start": make_node("start"), "next": make_node("next")},
                edges=[("start", None), ("start", "next")],
            )

    def test_leaf_marker_cannot_combine_with_router(self) -> None:
        router = make_node("router")
        with pytest.raises(ValueError, match="cannot combine a leaf marker"):
            make_graph_flow(
                nodes={"start": make_node("start"), "router": router},
                edges=[("start", None), ("start", router)],
            )

    def test_priorities_must_be_dict(self) -> None:
        with pytest.raises(TypeError, match="priorities must be a dict"):
            make_graph_flow(priorities=["start"])  # type: ignore[arg-type]

    def test_priorities_keys_must_be_configured_nodes(self) -> None:
        with pytest.raises(KeyError, match="priorities key"):
            make_graph_flow(priorities={"missing": 2})

    def test_max_edge_traversals_must_be_int_or_none(self) -> None:
        with pytest.raises(TypeError, match="max_edge_traversals must be an int"):
            make_graph_flow(max_edge_traversals="5")  # type: ignore[arg-type]

    def test_max_edge_traversals_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_edge_traversals must be > 0"):
            make_graph_flow(max_edge_traversals=0)

    def test_stop_early_must_be_bool(self) -> None:
        with pytest.raises(TypeError, match="stop_early must be a bool"):
            make_graph_flow(stop_early="yes")  # type: ignore[arg-type]

    def test_result_mode_must_be_known(self) -> None:
        with pytest.raises(ValueError, match="result_mode must be one of"):
            make_graph_flow(result_mode="bogus")

    def test_return_keys_must_be_list_of_str(self) -> None:
        with pytest.raises(TypeError, match="return_keys must be a list"):
            make_graph_flow(return_keys="x")  # type: ignore[arg-type]

    def test_scalar_mode_allows_at_most_one_return_key(self) -> None:
        with pytest.raises(ValueError, match="allows at most one requested key"):
            make_graph_flow(result_mode=GraphFlow.SCALAR, return_keys=["a", "b"])

    def test_parameters_equal_start_nodes_own_parameters(self) -> None:
        flow = make_graph_flow()
        assert [p.name for p in flow.parameters] == ["x"]

    def test_whole_graph_type_collision_raises_workflow_error(self) -> None:
        def int_x(x: int) -> dict:
            return {"x": x}

        def str_x(x: str) -> dict:
            return {"x": x}

        with pytest.raises(WorkflowError, match="conflicts across nodes"):
            make_graph_flow(nodes={"start": T(int_x, "start"), "other": T(str_x, "other")})

    def test_whole_graph_collision_check_is_not_reachability_aware(self) -> None:
        # "other" is never actually connected to "start" by any edge, but
        # the collision check still runs across every node regardless.
        def int_x(x: int) -> dict:
            return {"x": x}

        def str_x(x: str) -> dict:
            return {"x": x}

        with pytest.raises(WorkflowError):
            make_graph_flow(
                nodes={"start": T(int_x, "start"), "other": T(str_x, "other")}, edges=[]
            )

    def test_any_type_suppresses_collision(self) -> None:
        def any_x(x) -> dict:  # noqa: ANN001 -- deliberately untyped -> "Any"
            return {"x": x}

        def int_x(x: int) -> dict:
            return {"x": x}

        flow = make_graph_flow(nodes={"start": T(int_x, "start"), "other": T(any_x, "other")})
        assert flow is not None

    def test_router_own_parameters_are_not_included_in_collision_check(self) -> None:
        def start_fn(x: int) -> dict:
            return {"x": x}

        def router_fn(x: str) -> str | None:  # deliberately incompatible vs start's x:int
            return None

        router = T(router_fn, "router")

        # Must NOT raise -- routers are outside the collision-check scope,
        # they're never part of self._node_data.
        flow = GraphFlow(
            name="router_scope",
            namespace="tests",
            description=".",
            nodes={"start": T(start_fn, "start")},
            edges=[("start", router)],
            start="start",
        )
        assert flow is not None

    def test_shared_type_widens_across_nodes(self) -> None:
        def start_fn(x: dict) -> dict:
            return {"x": x}

        def other_fn(x: dict[str, int]) -> dict:
            return {"x": x}

        flow = make_graph_flow(
            nodes={"start": T(start_fn, "start"), "other": T(other_fn, "other")}
        )

        x_param = next(p for p in flow.parameters if p.name == "x")
        assert set(x_param.type) == {"dict", "dict[str, int]"}

    def test_kind_default_description_always_come_from_start(self) -> None:
        def start_fn(x: dict = None) -> dict:
            return {"x": x}

        def other_fn(x: dict[str, int] = {"k": 1}) -> dict:  # noqa: B006 -- deliberately distinct default for the test
            return {"x": x}

        flow = make_graph_flow(
            nodes={"start": T(start_fn, "start"), "other": T(other_fn, "other")}
        )

        x_param = next(p for p in flow.parameters if p.name == "x")
        assert x_param.default is None
        assert set(x_param.type) == {"dict", "dict[str, int]"}

    def test_widening_is_reachability_blind(self) -> None:
        # "other" is never connected to "start" by any edge, but its type
        # still widens start's own declared type for the shared name --
        # mirrors test_whole_graph_collision_check_is_not_reachability_aware.
        def start_fn(x: dict) -> dict:
            return {"x": x}

        def other_fn(x: dict[str, int]) -> dict:
            return {"x": x}

        flow = make_graph_flow(
            nodes={"start": T(start_fn, "start"), "other": T(other_fn, "other")}, edges=[]
        )

        x_param = next(p for p in flow.parameters if p.name == "x")
        assert set(x_param.type) == {"dict", "dict[str, int]"}

    def test_widening_does_not_mutate_node_invokables(self) -> None:
        def start_fn(x: dict) -> dict:
            return {"x": x}

        def other_fn(x: dict[str, int]) -> dict:
            return {"x": x}

        start_node = T(start_fn, "start")
        other_node = T(other_fn, "other")
        flow = make_graph_flow(nodes={"start": start_node, "other": other_node})

        assert flow is not None
        assert [p.type for p in start_node.parameters] == [("dict",)]
        assert [p.type for p in other_node.parameters] == [("dict[str, int]",)]

    def test_scalar_return_type_reconciled_from_declaring_nodes(self) -> None:
        def start_fn(x: int) -> dict:
            return {"x": x, "y": 0}

        def reader_fn(y: int) -> dict:
            return {"y": y}

        flow = make_graph_flow(
            nodes={"start": T(start_fn, "start"), "reader": T(reader_fn, "reader")},
            result_mode=GraphFlow.SCALAR,
            return_keys=["y"],
        )
        assert flow.return_type == "int"

    def test_scalar_return_type_any_for_orphan_key(self) -> None:
        flow = make_graph_flow(result_mode=GraphFlow.SCALAR, return_keys=["never_declared"])
        assert flow.return_type == "Any"

    def test_construction_raises_when_any_reader_coexists_with_a_real_type_conflict(self) -> None:
        # "y" declared as int by one node and str by another -- a genuine
        # conflict that has nothing to do with any_reader's presence. The
        # all-pairs collision check catches this directly: int-vs-str is
        # incompatible regardless of what a third, unrelated node declares.
        def start_fn(x: int) -> dict:
            return {"x": x, "y": 0}

        def int_reader(y: int) -> dict:
            return {"y": y}

        def str_reader(y: str) -> dict:
            return {"y": y}

        def any_reader(y) -> dict:  # noqa: ANN001 -- deliberately untyped -> "Any"
            return {"y": y}

        with pytest.raises(WorkflowError, match="conflicts across nodes"):
            make_graph_flow(
                nodes={
                    "start": T(start_fn, "start"),
                    "int_reader": T(int_reader, "int_reader"),
                    "str_reader": T(str_reader, "str_reader"),
                    "any_reader": T(any_reader, "any_reader"),
                },
                result_mode=GraphFlow.SCALAR,
                return_keys=["y"],
            )

    def test_scalar_return_type_drops_any_and_keeps_the_one_concrete_type(self) -> None:
        # "y" declared as Any by one node, int by another -- both pass the
        # collision check (Any is a universal wildcard); the SCALAR
        # return_type should be the one real, informative type, not diluted
        # by the wildcard that contributes no actual constraint.
        def start_fn(x: int) -> dict:
            return {"x": x, "y": 0}

        def int_reader(y: int) -> dict:
            return {"y": y}

        def any_reader(y) -> dict:  # noqa: ANN001 -- deliberately untyped -> "Any"
            return {"y": y}

        flow = make_graph_flow(
            nodes={
                "start": T(start_fn, "start"),
                "int_reader": T(int_reader, "int_reader"),
                "any_reader": T(any_reader, "any_reader"),
            },
            result_mode=GraphFlow.SCALAR,
            return_keys=["y"],
        )
        assert flow.return_type == "int"

    def test_scalar_return_type_unions_genuinely_distinct_compatible_declarations(self) -> None:
        # "y" declared as dict[str, Any] by one node, dict[str, int] by
        # another -- same origin, compatible via the value-position Any
        # wildcard, but neither is a bare origin so neither gets simplified
        # away. Both are real, non-redundant information -- the return type
        # should union them, not collapse to "Any".
        def start_fn(x: int) -> dict:
            return {"x": x, "y": {}}

        def any_value_reader(y: dict[str, Any]) -> dict:
            return {"y": y}

        def int_value_reader(y: dict[str, int]) -> dict:
            return {"y": y}

        flow = make_graph_flow(
            nodes={
                "start": T(start_fn, "start"),
                "any_value_reader": T(any_value_reader, "any_value_reader"),
                "int_value_reader": T(int_value_reader, "int_value_reader"),
            },
            result_mode=GraphFlow.SCALAR,
            return_keys=["y"],
        )
        assert flow.return_type == "dict[str, Any] | dict[str, int]"

    def test_scalar_return_type_simplifies_bare_origin_redundancy(self) -> None:
        # "y" declared as bare dict by one node, dict[str, int] by another,
        # dict[str, Any] by a third -- all mutually compatible (bare dict is
        # a wildcard for its own origin; int-vs-Any at the value position is
        # wildcard-compatible too). The parameterized siblings add no
        # information beyond the bare form, so they get simplified away.
        def start_fn(x: int) -> dict:
            return {"x": x, "y": {}}

        def bare_reader(y: dict) -> dict:
            return {"y": y}

        def int_value_reader(y: dict[str, int]) -> dict:
            return {"y": y}

        def any_value_reader(y: dict[str, Any]) -> dict:
            return {"y": y}

        flow = make_graph_flow(
            nodes={
                "start": T(start_fn, "start"),
                "bare_reader": T(bare_reader, "bare_reader"),
                "int_value_reader": T(int_value_reader, "int_value_reader"),
                "any_value_reader": T(any_value_reader, "any_value_reader"),
            },
            result_mode=GraphFlow.SCALAR,
            return_keys=["y"],
        )
        assert flow.return_type == "dict"

    def test_scalar_return_type_none_when_no_return_keys(self) -> None:
        flow = make_graph_flow(result_mode=GraphFlow.SCALAR, return_keys=None)
        assert flow.return_type == "None"

    def test_non_scalar_return_type_is_mode_name(self) -> None:
        flow = make_graph_flow(result_mode=GraphFlow.DICT, return_keys=["x"])
        assert flow.return_type == "dict"


class TestGraphFlowNodeDataAssembly:
    def test_incoming_reflects_fixed_edges_only(self) -> None:
        flow = make_graph_flow(
            nodes={"start": make_node("start"), "mid": make_node("mid")},
            edges=[("start", "mid")],
        )
        assert flow.nodes["mid"].incoming == ("start",)

    def test_router_edges_never_populate_incoming(self) -> None:
        router = make_node("router")
        flow = make_graph_flow(
            nodes={"start": make_node("start"), "mid": make_node("mid")},
            edges=[("start", "mid"), ("mid", router)],
        )
        # router is attached to "mid" but its dynamic target isn't known
        # statically -- nothing's incoming reflects it.
        assert flow.nodes["start"].incoming == ()

    def test_outgoing_and_routers_split_correctly(self) -> None:
        router = make_node("router")
        flow = make_graph_flow(
            nodes={"start": make_node("start"), "mid": make_node("mid"), "router": router},
            edges=[("start", "mid"), ("start", router)],
        )
        assert flow.nodes["start"].outgoing == ("mid",)
        assert flow.nodes["start"].routers == (router,)

    def test_node_with_no_edges_or_routers_has_empty_topology(self) -> None:
        flow = make_graph_flow()
        assert flow.nodes["start"].outgoing == ()
        assert flow.nodes["start"].routers == ()
        assert flow.nodes["start"].incoming == ()

    def test_priority_defaults_to_one(self) -> None:
        flow = make_graph_flow()
        assert flow.nodes["start"].priority == 1

    def test_priority_from_priorities_dict(self) -> None:
        flow = make_graph_flow(priorities={"start": 5})
        assert flow.nodes["start"].priority == 5

    def test_self_loop_permitted(self) -> None:
        flow = make_graph_flow(edges=[("start", "start")])
        assert flow.nodes["start"].outgoing == ("start",)
        assert flow.nodes["start"].incoming == ("start",)

    def test_same_invokable_may_back_multiple_node_names(self) -> None:
        shared = make_node("shared_tool")
        flow = make_graph_flow(nodes={"start": shared, "alias": shared})
        assert flow.nodes["start"].invokable is shared
        assert flow.nodes["alias"].invokable is shared

    def test_nodes_property_entries_cannot_be_reassigned(self) -> None:
        flow = make_graph_flow()
        with pytest.raises(TypeError):
            flow.nodes["start"] = None  # type: ignore[index]


class TestGraphFlowMutators:
    def test_set_state_policy_registers_and_visible(self) -> None:
        flow = make_graph_flow()
        flow.set_state_policy("x", True)
        assert flow.state_policies["x"].raise_on_collision is True

    def test_set_state_policy_overwrites_existing(self) -> None:
        flow = make_graph_flow()
        flow.set_state_policy("x", False, "first")
        flow.set_state_policy("x", False, "last")
        assert flow.state_policies["x"].tiebreak == "last"

    def test_set_state_policy_validates_via_statepolicyspec(self) -> None:
        flow = make_graph_flow()
        with pytest.raises(ValueError, match="tiebreak must be None"):
            flow.set_state_policy("x", True, "first")

    def test_remove_state_policy_removes_registered(self) -> None:
        flow = make_graph_flow(
            state_policies=[StatePolicySpec(key="x", raise_on_collision=True)]
        )
        flow.remove_state_policy("x")
        assert "x" not in flow.state_policies

    def test_remove_state_policy_rejects_unregistered(self) -> None:
        flow = make_graph_flow()
        with pytest.raises(ValueError, match="no state policy is configured"):
            flow.remove_state_policy("missing")

    def test_bulk_state_policies_replay_set_state_policy(self) -> None:
        flow = make_graph_flow(
            state_policies=[StatePolicySpec(key="x", raise_on_collision=False, tiebreak="first")]
        )
        assert flow.state_policies["x"].tiebreak == "first"

    def test_set_priority_updates_node_data(self) -> None:
        flow = make_graph_flow()
        flow.set_priority("start", 9)
        assert flow.nodes["start"].priority == 9

    def test_set_priority_rejects_unknown_node(self) -> None:
        flow = make_graph_flow()
        with pytest.raises(KeyError, match="is not a configured node"):
            flow.set_priority("missing", 2)

    def test_set_priority_rejects_non_int(self) -> None:
        flow = make_graph_flow()
        with pytest.raises(TypeError, match="priority must be an int"):
            flow.set_priority("start", "high")  # type: ignore[arg-type]

    def test_max_edge_traversals_setter_validates(self) -> None:
        flow = make_graph_flow()
        flow.max_edge_traversals = 5
        assert flow.max_edge_traversals == 5

        with pytest.raises(TypeError):
            flow.max_edge_traversals = "5"  # type: ignore[assignment]

        with pytest.raises(ValueError):
            flow.max_edge_traversals = 0

        flow.max_edge_traversals = None
        assert flow.max_edge_traversals is None

    def test_stop_early_setter_validates(self) -> None:
        flow = make_graph_flow()
        flow.stop_early = True
        assert flow.stop_early is True

        with pytest.raises(TypeError):
            flow.stop_early = "yes"  # type: ignore[assignment]

    def test_include_trace_mutable_inherited_from_base(self) -> None:
        flow = make_graph_flow()
        assert flow.include_trace is True
        flow.include_trace = False
        assert flow.include_trace is False


class TestGraphFlowExtraDescription:
    def test_unbounded_cap_note(self) -> None:
        flow = make_graph_flow()
        assert flow._extra_description() == "Enters at node 'start'. Unbounded superstep count."

    def test_bounded_cap_note(self) -> None:
        flow = make_graph_flow(max_edge_traversals=7)
        assert flow._extra_description() == "Enters at node 'start'. Bounded to 7 superstep(s)."


class TestGraphFlowSyncInvoke:
    def test_single_node_graph_runs_once_and_terminates_queue_empty(self) -> None:
        flow = make_graph_flow(result_mode=GraphFlow.SCALAR, return_keys=["x"])

        result = flow.invoke({"x": 1})

        assert result.result == 2
        assert result.termination_reason == "queue_empty"
        assert result.edge_traversals == (("start",),)

    def test_fan_out_and_fan_in_converge_in_expected_superstep(self) -> None:
        def start_fn(x: int) -> dict:
            return {"a": x, "b": x}

        def left_fn(a: int) -> dict:
            return {"left_result": a + 1}

        def right_fn(b: int) -> dict:
            return {"right_result": b + 2}

        def merge_fn(left_result: int, right_result: int) -> dict:
            return {"total": left_result + right_result}

        nodes = {
            "start": T(start_fn, "start"),
            "left": T(left_fn, "left"),
            "right": T(right_fn, "right"),
            "merge": T(merge_fn, "merge"),
        }
        edges = [("start", "left"), ("start", "right"), ("left", "merge"), ("right", "merge")]
        flow = GraphFlow(
            name="fanout",
            namespace="tests",
            description=".",
            nodes=nodes,
            edges=edges,
            start="start",
            return_keys=["total"],
            result_mode=GraphFlow.SCALAR,
        )

        result = flow.invoke({"x": 10})

        assert result.result == 23  # (10 + 1) + (10 + 2)
        assert result.edge_traversals == (("start",), ("left", "right"), ("merge",))
        # If ready-set dedup were broken, "merge" would appear twice.
        assert len(result.trace) == 4

    def test_cycle_loops_back_via_router_then_terminates(self) -> None:
        def start_fn(count: int) -> dict:
            return {"count": count + 1}

        def loop_or_stop(count: int) -> str | None:
            return "start" if count < 3 else None

        router = T(loop_or_stop, "router")
        flow = GraphFlow(
            name="cycle",
            namespace="tests",
            description=".",
            nodes={"start": T(start_fn, "start")},
            edges=[("start", router)],
            start="start",
            return_keys=["count"],
            result_mode=GraphFlow.SCALAR,
            max_edge_traversals=10,
        )

        result = flow.invoke({"count": 0})

        assert result.result == 3
        assert result.termination_reason == "queue_empty"
        assert result.edge_traversals == (("start",), ("start",), ("start",))
        assert len(result.trace) == 6  # 3 node fires + 3 router fires, interleaved

    def test_multiple_routers_on_one_node_are_additive(self) -> None:
        def source_fn(x: int) -> dict:
            return {"x": x}

        def router_a_fn(x: int) -> str | None:
            return "target_a"

        def router_b_fn(x: int) -> str | None:
            return "target_b"

        def target_a_fn(x: int) -> dict:
            return {"visited_a": True}

        def target_b_fn(x: int) -> dict:
            return {"visited_b": True}

        router_a = T(router_a_fn, "router_a")
        router_b = T(router_b_fn, "router_b")
        nodes = {
            "source": T(source_fn, "source"),
            "target_a": T(target_a_fn, "target_a"),
            "target_b": T(target_b_fn, "target_b"),
        }
        flow = GraphFlow(
            name="multi_router",
            namespace="tests",
            description=".",
            nodes=nodes,
            edges=[("source", router_a), ("source", router_b)],
            start="source",
            return_keys=["visited_a", "visited_b"],
            result_mode=GraphFlow.DICT,
        )

        result = flow.invoke({"x": 1})

        assert result.result == {"visited_a": True, "visited_b": True}
        assert result.edge_traversals == (("source",), ("target_a", "target_b"))


class TestGraphFlowStateCollisions:
    def _collision_flow(self, **kwargs: Any) -> GraphFlow:
        def start_fn() -> dict:
            return {}

        def writer_a() -> dict:
            return {"shared": "from_a"}

        def writer_b() -> dict:
            return {"shared": "from_b"}

        nodes = {
            "start": T(start_fn, "start"),
            "writer_a": T(writer_a, "writer_a"),
            "writer_b": T(writer_b, "writer_b"),
        }
        edges = [("start", "writer_a"), ("start", "writer_b")]
        return GraphFlow(
            name="collision",
            namespace="tests",
            description=".",
            nodes=nodes,
            edges=edges,
            start="start",
            return_keys=["shared"],
            result_mode=GraphFlow.SCALAR,
            **kwargs,
        )

    def test_default_policy_is_last_writer_wins(self) -> None:
        flow = self._collision_flow()
        result = flow.invoke({})
        assert result.result == "from_b"  # writer_b is last in fixed round order

    def test_priority_resolves_regardless_of_tiebreak_direction(self) -> None:
        flow = self._collision_flow(priorities={"writer_a": 5})
        result = flow.invoke({})
        assert result.result == "from_a"

    def test_explicit_tiebreak_first(self) -> None:
        flow = self._collision_flow(
            state_policies=[StatePolicySpec(key="shared", raise_on_collision=False, tiebreak="first")]
        )
        result = flow.invoke({})
        assert result.result == "from_a"

    def test_configured_policy_with_no_tiebreak_still_defaults_to_last(self) -> None:
        # StatePolicySpec permits raise_on_collision=False with tiebreak
        # left None -- pins down that this resolves the same as the fully
        # lazy (no policy at all) default, not a different/undefined path.
        flow = self._collision_flow(
            state_policies=[StatePolicySpec(key="shared", raise_on_collision=False, tiebreak=None)]
        )
        result = flow.invoke({})
        assert result.result == "from_b"

    def test_raise_on_collision_true_raises_validation_error(self) -> None:
        flow = self._collision_flow(
            state_policies=[StatePolicySpec(key="shared", raise_on_collision=True)]
        )
        with pytest.raises(ExecutionError) as exc_info:
            flow.invoke({})
        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "written by multiple nodes this superstep" in str(exc_info.value.__cause__)

    def test_single_writer_key_never_consults_policy(self) -> None:
        # A key written by exactly one node this round is never a
        # "collision" -- raise_on_collision must not fire for it.
        def start_fn() -> dict:
            return {}

        def only_writer() -> dict:
            return {"solo": "value"}

        flow = GraphFlow(
            name="solo_write",
            namespace="tests",
            description=".",
            nodes={"start": T(start_fn, "start"), "only_writer": T(only_writer, "only_writer")},
            edges=[("start", "only_writer")],
            start="start",
            state_policies=[StatePolicySpec(key="solo", raise_on_collision=True)],
            return_keys=["solo"],
            result_mode=GraphFlow.SCALAR,
        )
        result = flow.invoke({})
        assert result.result == "value"


class TestGraphFlowTermination:
    def _stop_early_flow(self, *, stop_early: bool, return_keys: list[str] | None) -> GraphFlow:
        def start_fn(x: int) -> dict:
            return {"x": x, "target": "ready"}

        def slow_branch_fn(x: int) -> dict:
            return {"slow_done": True}

        nodes = {"start": T(start_fn, "start"), "slow_branch": T(slow_branch_fn, "slow_branch")}
        return GraphFlow(
            name="stop_early_flow",
            namespace="tests",
            description=".",
            nodes=nodes,
            edges=[("start", "slow_branch")],
            start="start",
            return_keys=return_keys,
            result_mode=GraphFlow.SCALAR if return_keys else GraphFlow.LIST,
            stop_early=stop_early,
        )

    def test_stop_early_stops_before_queue_naturally_empties(self) -> None:
        flow = self._stop_early_flow(stop_early=True, return_keys=["target"])

        result = flow.invoke({"x": 1})

        assert result.termination_reason == "early_exit"
        assert result.result == "ready"
        assert result.edge_traversals == (("start",),)

    def test_stop_early_inert_when_no_return_keys_requested(self) -> None:
        flow = self._stop_early_flow(stop_early=True, return_keys=None)

        result = flow.invoke({"x": 1})

        assert result.termination_reason == "queue_empty"
        assert result.edge_traversals == (("start",), ("slow_branch",))

    def _always_looping_flow(self, *, max_edge_traversals: int, return_keys: list[str]) -> GraphFlow:
        def start_fn(count: int) -> dict:
            return {"count": count + 1}

        def always_loop(count: int) -> str:
            return "start"

        router = T(always_loop, "always_loop")
        return GraphFlow(
            name="capped",
            namespace="tests",
            description=".",
            nodes={"start": T(start_fn, "start")},
            edges=[("start", router)],
            start="start",
            return_keys=return_keys,
            result_mode=GraphFlow.SCALAR,
            max_edge_traversals=max_edge_traversals,
        )

    def test_max_edge_traversals_stops_run_as_cap_hit(self) -> None:
        flow = self._always_looping_flow(max_edge_traversals=3, return_keys=["count"])

        result = flow.invoke({"count": 0})

        assert result.termination_reason == "cap_hit"
        assert result.result == 3
        assert len(result.edge_traversals) == 3

    def test_cap_hit_with_missing_requested_key_raises(self) -> None:
        flow = self._always_looping_flow(max_edge_traversals=2, return_keys=["never_produced"])

        with pytest.raises(ExecutionError) as exc_info:
            flow.invoke({"count": 0})

        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "were never populated" in str(exc_info.value.__cause__)


class TestGraphFlowResultModes:
    def _single_node_flow(self, *, result_mode: str, return_keys: list[str] | None) -> GraphFlow:
        return make_graph_flow(
            nodes={"start": make_node("start")}, result_mode=result_mode, return_keys=return_keys
        )

    def test_scalar_with_key(self) -> None:
        flow = self._single_node_flow(result_mode=GraphFlow.SCALAR, return_keys=["x"])
        assert flow.invoke({"x": 1}).result == 2

    def test_scalar_without_key_returns_none(self) -> None:
        flow = self._single_node_flow(result_mode=GraphFlow.SCALAR, return_keys=None)
        assert flow.invoke({"x": 1}).result is None

    def test_list_with_keys(self) -> None:
        flow = self._single_node_flow(result_mode=GraphFlow.LIST, return_keys=["x"])
        assert flow.invoke({"x": 1}).result == [2]

    def test_list_without_keys_returns_empty_list(self) -> None:
        flow = self._single_node_flow(result_mode=GraphFlow.LIST, return_keys=None)
        assert flow.invoke({"x": 1}).result == []

    def test_tuple_with_keys(self) -> None:
        flow = self._single_node_flow(result_mode=GraphFlow.TUPLE, return_keys=["x"])
        assert flow.invoke({"x": 1}).result == (2,)

    def test_tuple_without_keys_returns_empty_tuple(self) -> None:
        flow = self._single_node_flow(result_mode=GraphFlow.TUPLE, return_keys=None)
        assert flow.invoke({"x": 1}).result == ()

    def test_set_with_keys(self) -> None:
        flow = self._single_node_flow(result_mode=GraphFlow.SET, return_keys=["x"])
        assert flow.invoke({"x": 1}).result == {2}

    def test_set_without_keys_returns_empty_set(self) -> None:
        flow = self._single_node_flow(result_mode=GraphFlow.SET, return_keys=None)
        assert flow.invoke({"x": 1}).result == set()

    def test_dict_with_keys(self) -> None:
        flow = self._single_node_flow(result_mode=GraphFlow.DICT, return_keys=["x"])
        assert flow.invoke({"x": 1}).result == {"x": 2}

    def test_dict_without_keys_returns_empty_dict(self) -> None:
        flow = self._single_node_flow(result_mode=GraphFlow.DICT, return_keys=None)
        assert flow.invoke({"x": 1}).result == {}


class TestGraphFlowTraceAndEdgeTraversals:
    def test_trace_none_when_include_trace_disabled(self) -> None:
        flow = make_graph_flow()
        flow.include_trace = False
        result = flow.invoke({"x": 1})
        assert result.trace is None

    def test_edge_traversals_populated_regardless_of_include_trace(self) -> None:
        flow = make_graph_flow()
        flow.include_trace = False
        result = flow.invoke({"x": 1})
        assert result.edge_traversals == (("start",),)

    def test_trace_orders_node_results_before_router_results_per_round(self) -> None:
        def start_fn(x: int) -> dict:
            return {"x": x}

        def router_fn(x: int) -> str | None:
            return None

        router = T(router_fn, "router")
        start_node = T(start_fn, "start")
        flow = GraphFlow(
            name="trace_order",
            namespace="tests",
            description=".",
            nodes={"start": start_node},
            edges=[("start", router)],
            start="start",
        )

        result = flow.invoke({"x": 1})

        assert len(result.trace) == 2
        assert result.trace[0].invoker_id == start_node.instance_id
        assert result.trace[1].invoker_id == router.instance_id


class TestGraphFlowAsyncInvoke:
    def test_async_invoke_runs_cycle_and_terminates(self) -> None:
        def start_fn(count: int) -> dict:
            return {"count": count + 1}

        def loop_or_stop(count: int) -> str | None:
            return "start" if count < 3 else None

        router = T(loop_or_stop, "router")
        flow = GraphFlow(
            name="async_cycle",
            namespace="tests",
            description=".",
            nodes={"start": T(start_fn, "start")},
            edges=[("start", router)],
            start="start",
            return_keys=["count"],
            result_mode=GraphFlow.SCALAR,
            max_edge_traversals=10,
        )

        result = asyncio.run(flow.async_invoke({"count": 0}))

        assert isinstance(result, GraphFlowResult)
        assert result.result == 3
        assert result.termination_reason == "queue_empty"


class TestGraphFlowValidationAndErrors:
    def test_node_non_mapping_result_raises_validation_error(self) -> None:
        def bad_fn(x: int) -> int:
            return x + 1

        flow = GraphFlow(
            name="bad_output",
            namespace="tests",
            description=".",
            nodes={"start": T(bad_fn, "start")},
            edges=[],
            start="start",
        )
        with pytest.raises(ExecutionError) as exc_info:
            flow.invoke({"x": 1})
        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "must produce a mapping-shaped result" in str(exc_info.value.__cause__)

    def test_router_returning_non_string_non_none_raises_validation_error(self) -> None:
        def start_fn(x: int) -> dict:
            return {"x": x}

        def bad_router(x: int) -> int:
            return 42

        router = T(bad_router, "bad_router")
        flow = GraphFlow(
            name="bad_router_flow",
            namespace="tests",
            description=".",
            nodes={"start": T(start_fn, "start")},
            edges=[("start", router)],
            start="start",
        )
        with pytest.raises(ExecutionError) as exc_info:
            flow.invoke({"x": 1})
        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "must return None or a node name" in str(exc_info.value.__cause__)

    def test_router_naming_unconfigured_node_raises_validation_error(self) -> None:
        def start_fn(x: int) -> dict:
            return {"x": x}

        def bad_router(x: int) -> str:
            return "does_not_exist"

        router = T(bad_router, "bad_router")
        flow = GraphFlow(
            name="bad_target_flow",
            namespace="tests",
            description=".",
            nodes={"start": T(start_fn, "start")},
            edges=[("start", router)],
            start="start",
        )
        with pytest.raises(ExecutionError) as exc_info:
            flow.invoke({"x": 1})
        assert isinstance(exc_info.value.__cause__, ValidationError)
        assert "is not a configured node" in str(exc_info.value.__cause__)

    def test_node_invoke_failure_wrapped_with_node_context(self) -> None:
        def raising_fn(x: int) -> dict:
            raise RuntimeError("boom")

        flow = GraphFlow(
            name="raising_flow",
            namespace="tests",
            description=".",
            nodes={"start": T(raising_fn, "start")},
            edges=[],
            start="start",
        )
        with pytest.raises(ExecutionError) as exc_info:
            flow.invoke({"x": 1})
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "node 'start'" in str(exc_info.value.__cause__)

    def test_router_invoke_failure_wrapped_with_router_context(self) -> None:
        def start_fn(x: int) -> dict:
            return {"x": x}

        def raising_router(x: int) -> str | None:
            raise RuntimeError("router boom")

        router = T(raising_router, "raising_router")
        flow = GraphFlow(
            name="raising_router_flow",
            namespace="tests",
            description=".",
            nodes={"start": T(start_fn, "start")},
            edges=[("start", router)],
            start="start",
        )
        with pytest.raises(ExecutionError) as exc_info:
            flow.invoke({"x": 1})
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "router attached to node 'start'" in str(exc_info.value.__cause__)


class TestGraphFlowSerialization:
    def test_to_dict_includes_topology_and_config(self) -> None:
        def start_fn(x: int) -> dict:
            return {"x": x}

        def router_fn(x: int) -> str | None:
            return None

        router = T(router_fn, "router")
        nodes = {"start": T(start_fn, "start"), "next": make_node("next")}
        flow = GraphFlow(
            name="serial",
            namespace="tests",
            description=".",
            nodes=nodes,
            edges=[("start", "next"), ("start", router)],
            start="start",
            priorities={"next": 3},
            state_policies=[StatePolicySpec(key="x", raise_on_collision=False, tiebreak="first")],
            max_edge_traversals=5,
            stop_early=True,
            result_mode=GraphFlow.DICT,
            return_keys=["x"],
        )

        data = flow.to_dict()

        assert data["start"] == "start"
        assert data["priorities"] == {"start": 1, "next": 3}
        assert data["state_policies"] == [
            {"key": "x", "raise_on_collision": False, "tiebreak": "first"}
        ]
        assert data["max_edge_traversals"] == 5
        assert data["stop_early"] is True
        assert data["result_mode"] == "dict"
        assert data["return_keys"] == ["x"]
        assert set(data["nodes"].keys()) == {"start", "next"}
        assert ("start", "next") in data["edges"]
