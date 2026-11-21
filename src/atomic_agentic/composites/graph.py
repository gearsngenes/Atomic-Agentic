from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, ClassVar

from ..constants.core import IDENTIFIER_PATTERN
from ..invokables.atomic_invokable import AtomicInvokable
from ..models.parameters import ParamSpec
from ..models.results import GraphResult
from ..tools.toolification import to_invokable
from ..utils.binding import Binding, normalize_binding, slice_response
from ..utils.core import run_coro_sync

logger = logging.getLogger(__name__)

__all__ = ["Graph"]


class Graph(AtomicInvokable):
    """Superstep-batched node graph over a shared state dict.

    Construction contract
    ---------------------
    - ``nodes`` is an optional dict of name to entry (an invokable, a ``str``, a
      callable, or ``(entry, binding)``); a graph may start empty. Entries are
      coerced through the shared ``to_invokable`` and named by their dict key.
      A node name must be a valid invokable name (alphanumeric/underscore, no
      leading digit) or ``ValueError`` is raised.
    - ``edges`` are ``(source, target)`` pairs: a ``str`` target is a node name
      (fixed edge); a callable or invokable target is a router attached to
      ``source``. A node with no outgoing edges is terminal.
    - ``start`` names the round-0 nodes (several start nodes run in parallel). It
      may be empty while the graph is being built; running with an empty
      ``start`` raises ``RuntimeError``.
    - ``parameters`` default to a single ``**kwargs``; ``return_type`` is always ``Any``
      because ``returns`` is mutable and may switch between a key, a list and the
      whole state. ``name``, ``parameters`` and ``return_type`` are fixed for the
      graph's life.

    Mutation contract
    -----------------
    - Nodes, edges and knobs change only through the mutators below, each of
      which validates fully before changing anything (a failed call leaves the
      graph untouched). Node records are replaced, never edited in place.
    - Relations are sets: ``add_edge`` of an existing edge, ``remove_edge`` of an
      absent one and ``disconnect_node`` with nothing attached return ``False``.
      Names that do not exist, wrong types and invariant violations raise.
    - ``add_node`` appends in declaration order, which ranks the new node below
      existing ones in default collision handling.
    - A run works from a snapshot taken at its start; mutation during a run
      affects only later runs.

    Injection
    ---------
    - ``add_sequence`` and ``add_parallel`` add a dict of named nodes (same entry
      forms as the constructor) and may bridge them to existing nodes
      (``start_from``/``end_at``, ``fan_out``/``fan_in``). They only add edges,
      are atomic, and never touch ``start`` or the collision order.
    - ``add_sequence(..., repeat_until={key: value})`` loops the sequence until
      every key matches in state; the depth cap ends a loop that never matches.

    Runtime contract
    ----------------
    - Call inputs seed state (the ``**kwargs`` payload is flattened into it).
    - Each superstep runs the ready nodes concurrently against one frozen state
      snapshot, binds each response (``None`` merges a mapping whole), merges the
      writes, then runs routers against the merged state. A node fires at most
      once per superstep; cycles re-fire it in later supersteps.
    - Same-key writes in one superstep go to the highest-ranked writer: the
      ``collision_handling_order`` names first, then every other node in
      declaration order.
    - The run ends when nothing is ready. At ``max_traversal_depth`` with nodes
      still ready it raises ``RuntimeError``.
    - ``returns`` selects the result: ``None`` whole state, ``str`` one key's
      value, ``list[str]`` a subset dict.
    """

    result_cls: ClassVar[type[GraphResult]] = GraphResult

    @dataclass(frozen=True)
    class Node:
        """One node: the invokable, its output binding (None = merge a mapping
        response whole), its fixed outgoing node names and its routers."""

        invokable: AtomicInvokable
        binding: Binding | None
        outgoing: tuple[str, ...]
        routers: tuple[AtomicInvokable, ...]

    def __init__(
        self,
        name: str,
        description: str,
        nodes: dict[str, AtomicInvokable | str | Callable[..., Any] | tuple[Any, Binding | None]]
        | None = None,
        edges: list[tuple[str, str | AtomicInvokable | Callable[..., Any]]] | None = None,
        start: str | list[str] | None = None,
        *,
        parameters: list[ParamSpec] | None = None,
        returns: str | list[str] | None = None,
        max_traversal_depth: int = 50,
        collision_handling_order: list[str] | None = None,
    ) -> None:
        # 1. nodes
        nodes = {} if nodes is None else nodes
        if not isinstance(nodes, dict):
            raise TypeError(f"nodes must be a dict, got {type(nodes)!r}")
        built = {
            node_name: self._coerce_node(name, node_name, entry)
            for node_name, entry in nodes.items()
        }

        # 2. edges
        edges = [] if edges is None else edges
        if not isinstance(edges, list):
            raise TypeError(f"edges must be a list, got {type(edges)!r}")
        outgoing: dict[str, list[str]] = {n: [] for n in built}
        routers: dict[str, list[AtomicInvokable]] = {n: [] for n in built}
        router_count = 0
        for edge in edges:
            if not isinstance(edge, tuple):
                raise TypeError(f"each edge must be a (source, target) tuple, got {edge!r}")
            if len(edge) != 2:
                raise ValueError(f"each edge must be (source, target); got length {len(edge)}")
            source, target = edge
            self._check_edge_ends(source, target, built)
            if isinstance(target, str):
                if target not in outgoing[source]:
                    outgoing[source].append(target)
            else:
                routers[source].append(self._coerce_router(name, target, router_count))
                router_count += 1

        # 3. knobs
        start_tuple = self._check_start(start, built)
        depth = self._check_depth(max_traversal_depth)
        returns_value = self._check_returns(returns)
        order_value = self._check_order(collision_handling_order, built)

        # 4. base init
        if parameters is None:
            parameters = [
                ParamSpec(name="kwargs", index=0, kind=ParamSpec.VAR_KEYWORD, type=("Any",))
            ]
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            return_type="Any",
        )

        # 5. store
        self._nodes: dict[str, Graph.Node] = {
            node_name: Graph.Node(
                invokable=invokable,
                binding=binding,
                outgoing=tuple(outgoing[node_name]),
                routers=tuple(routers[node_name]),
            )
            for node_name, (invokable, binding) in built.items()
        }
        self._router_count = router_count
        self._start = start_tuple
        self._returns = returns_value
        self._max_traversal_depth = depth
        self._collision_handling_order = order_value

    # ---------------------------------------------------------------- #
    # Shared validators (constructor and mutators)
    # ---------------------------------------------------------------- #
    @staticmethod
    def _coerce_node(
        graph_name: str, node_name: Any, entry: Any
    ) -> tuple[AtomicInvokable, Binding | None]:
        """Turn a node entry into ``(invokable, binding)``."""
        if not isinstance(node_name, str) or not node_name:
            raise TypeError(f"node names must be non-empty str, got {node_name!r}")
        if not IDENTIFIER_PATTERN.fullmatch(node_name):
            raise ValueError(
                f"{graph_name}: node name {node_name!r} must be alphanumeric/underscore "
                "and not start with a digit"
            )
        if isinstance(entry, tuple):
            if len(entry) != 2:
                raise ValueError(
                    f"node {node_name!r} tuple must be (node, binding); got length {len(entry)}"
                )
            obj, raw_binding = entry
        else:
            obj, raw_binding = entry, None
        binding = normalize_binding(
            raw_binding, label=f"{graph_name} node {node_name!r} binding"
        )
        invokable = to_invokable(
            obj,
            name=node_name,
            description=f"Node {node_name} of graph {graph_name}",
        )
        return invokable, binding

    @staticmethod
    def _coerce_router(graph_name: str, target: Any, index: int) -> AtomicInvokable:
        """Turn a router target into an invokable named from ``index``."""
        return to_invokable(
            target,
            name=f"{graph_name}_router_{index}",
            description=f"Router {index} of graph {graph_name}",
        )

    @staticmethod
    def _check_node_names(names: tuple[str, ...], nodes: Mapping[str, Any], label: str) -> None:
        """Every name must be a str that names a node, with no repeats."""
        for node_name in names:
            if not isinstance(node_name, str):
                raise TypeError(f"{label} entries must be str, got {type(node_name)!r}")
            if node_name not in nodes:
                raise ValueError(f"{label} entry {node_name!r} is not a node")
        if len(set(names)) != len(names):
            raise ValueError(f"{label} must not repeat a node: {list(names)!r}")

    @staticmethod
    def _check_edge_ends(source: Any, target: Any, nodes: Mapping[str, Any]) -> None:
        """The source must be a node; a ``str`` target must be a node."""
        if not isinstance(source, str):
            raise TypeError(f"edge source must be a str, got {type(source)!r}")
        if source not in nodes:
            raise ValueError(f"edge source {source!r} is not a node")
        if isinstance(target, str) and target not in nodes:
            raise ValueError(f"edge target {target!r} is not a node")

    @staticmethod
    def _check_start(start: Any, nodes: Mapping[str, Any]) -> tuple[str, ...]:
        """Normalize ``start`` to a tuple of node names (empty is valid)."""
        if start is None:
            return ()
        if isinstance(start, str):
            start_tuple: tuple[str, ...] = (start,)
        elif isinstance(start, list):
            start_tuple = tuple(start)
        else:
            raise TypeError(f"start must be None, a str or a list[str], got {type(start)!r}")
        Graph._check_node_names(start_tuple, nodes, "start")
        return start_tuple

    @staticmethod
    def _check_returns(returns: Any) -> str | tuple[str, ...] | None:
        """Normalize ``returns``: ``None``, a non-empty str, or a tuple of str."""
        if returns is None:
            return None
        if isinstance(returns, str):
            if not returns:
                raise TypeError("returns must be a non-empty str when given as a str")
            return returns
        if isinstance(returns, list) and all(isinstance(k, str) for k in returns):
            return tuple(returns)
        raise TypeError(f"returns must be None, a str or a list[str], got {returns!r}")

    @staticmethod
    def _check_depth(depth: Any) -> int:
        """The depth cap must be a positive int (not a bool)."""
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise TypeError(f"max_traversal_depth must be an int, got {type(depth)!r}")
        if depth <= 0:
            raise ValueError("max_traversal_depth must be > 0")
        return depth

    @staticmethod
    def _check_order(order: Any, nodes: Mapping[str, Any]) -> tuple[str, ...] | None:
        """Normalize the collision order: ``None`` or a tuple of node names."""
        if order is None:
            return None
        if not isinstance(order, list) or not all(isinstance(n, str) for n in order):
            raise TypeError(
                f"collision_handling_order must be None or a list[str], got {order!r}"
            )
        order_tuple = tuple(order)
        Graph._check_node_names(order_tuple, nodes, "collision_handling_order")
        return order_tuple

    def _require_node(self, name: Any) -> Graph.Node:
        """Return the record of node ``name``; it must be a str naming a node."""
        if not isinstance(name, str):
            raise TypeError(f"node name must be a str, got {type(name)!r}")
        if name not in self._nodes:
            raise ValueError(f"{self.full_name}: {name!r} is not a node")
        return self._nodes[name]

    # ---------------------------------------------------------------- #
    # Read-only surface
    # ---------------------------------------------------------------- #
    @property
    def nodes(self) -> MappingProxyType[str, Graph.Node]:
        """Read-only live view of the node records."""
        return MappingProxyType(self._nodes)

    @property
    def start(self) -> tuple[str, ...]:
        """The round-0 node names."""
        return self._start

    @property
    def returns(self) -> str | tuple[str, ...] | None:
        """The result selection: ``None`` (whole state), a key, or a tuple of keys."""
        return self._returns

    @property
    def max_traversal_depth(self) -> int:
        """The superstep cap."""
        return self._max_traversal_depth

    @max_traversal_depth.setter
    def max_traversal_depth(self, value: int) -> None:
        self._max_traversal_depth = self._check_depth(value)

    @property
    def collision_handling_order(self) -> tuple[str, ...] | None:
        """Node names that win same-key collisions first, in order (``None``: declaration order)."""
        return self._collision_handling_order

    # ---------------------------------------------------------------- #
    # Node mutators
    # ---------------------------------------------------------------- #
    def add_node(self, name: str, entry: Any) -> None:
        """Add a node with no edges, appended in declaration order.

        1. Coerce ``entry`` (same forms as the constructor).
        2. A name that is already a node raises ``ValueError``.
        3. Append the record; ``start`` and the collision order are untouched.
        """
        invokable, binding = self._coerce_node(self.name, name, entry)
        if name in self._nodes:
            raise ValueError(f"{self.full_name}: {name!r} is already a node")
        self._nodes[name] = Graph.Node(invokable, binding, (), ())
        logger.debug("%s: added node %r", self.full_name, name)

    def update_node(self, name: str, entry: Any) -> None:
        """Replace a node's invokable and binding, keeping its edges and position."""
        node = self._require_node(name)
        invokable, binding = self._coerce_node(self.name, name, entry)
        self._nodes[name] = replace(node, invokable=invokable, binding=binding)
        logger.debug("%s: updated node %r", self.full_name, name)

    def remove_node(self, name: str) -> None:
        """Remove a node together with its own outgoing edges and routers.

        Raises while another node has a fixed edge to it. The node is dropped from
        ``start`` and ``collision_handling_order``. A self-loop does not block.
        A router elsewhere that later returns the name fails at run time.
        """
        self._require_node(name)
        pointing = [n for n, node in self._nodes.items() if n != name and name in node.outgoing]
        if pointing:
            raise ValueError(
                f"{self.full_name}: cannot remove {name!r}: nodes {pointing!r} have edges to it"
            )
        start = tuple(n for n in self._start if n != name)
        order = self._collision_handling_order
        if order is not None:
            order = tuple(n for n in order if n != name) or None
        del self._nodes[name]
        self._start = start
        self._collision_handling_order = order
        logger.debug("%s: removed node %r", self.full_name, name)

    # ---------------------------------------------------------------- #
    # Edge mutators (relations are sets: return True if changed)
    # ---------------------------------------------------------------- #
    def add_edge(
        self, source: str, target: str | AtomicInvokable | Callable[..., Any]
    ) -> bool:
        """Add a fixed edge (``str`` target) or a router (anything else).

        Returns ``False`` for a fixed edge already present or a router invokable
        already stored for ``source`` (by identity). A callable is coerced into
        a new router each time.
        """
        node = self._require_node(source)
        if isinstance(target, str):
            self._require_node(target)
            if target in node.outgoing:
                logger.debug("%s: edge %r -> %r already present", self.full_name, source, target)
                return False
            self._nodes[source] = replace(node, outgoing=node.outgoing + (target,))
            logger.debug("%s: added edge %r -> %r", self.full_name, source, target)
            return True
        if isinstance(target, AtomicInvokable) and any(r is target for r in node.routers):
            logger.debug("%s: router already stored on %r", self.full_name, source)
            return False
        router = self._coerce_router(self.name, target, self._router_count)
        self._router_count += 1
        self._nodes[source] = replace(node, routers=node.routers + (router,))
        logger.debug("%s: added router %s on %r", self.full_name, router.name, source)
        return True

    def remove_edge(self, source: str, target: str | AtomicInvokable) -> bool:
        """Remove a fixed edge (``str``) or a stored router (by identity).

        Returns ``False`` when the relation is not present. A raw callable cannot
        match a router (it was coerced); pass the invokable from
        ``graph.nodes[source].routers``.
        """
        node = self._require_node(source)
        if isinstance(target, str):
            self._require_node(target)
            if target not in node.outgoing:
                logger.debug("%s: no edge %r -> %r", self.full_name, source, target)
                return False
            remaining = tuple(t for t in node.outgoing if t != target)
            self._nodes[source] = replace(node, outgoing=remaining)
            logger.debug("%s: removed edge %r -> %r", self.full_name, source, target)
            return True
        if isinstance(target, AtomicInvokable):
            if not any(r is target for r in node.routers):
                logger.debug("%s: router not stored on %r", self.full_name, source)
                return False
            remaining_routers = tuple(r for r in node.routers if r is not target)
            self._nodes[source] = replace(node, routers=remaining_routers)
            logger.debug("%s: removed router %s from %r", self.full_name, target.name, source)
            return True
        raise TypeError(
            f"remove_edge target must be a node name or a stored router invokable "
            f"(see graph.nodes[{source!r}].routers), got {type(target)!r}"
        )

    def disconnect_node(self, name: str) -> bool:
        """Remove every fixed edge to and from ``name`` and all of its routers."""
        self._require_node(name)
        replacements: dict[str, Graph.Node] = {}
        for node_name, node in self._nodes.items():
            if node_name == name:
                if node.outgoing or node.routers:
                    replacements[node_name] = replace(node, outgoing=(), routers=())
            elif name in node.outgoing:
                replacements[node_name] = replace(
                    node, outgoing=tuple(t for t in node.outgoing if t != name)
                )
        self._nodes.update(replacements)
        logger.debug("%s: disconnected %r (changed=%s)", self.full_name, name, bool(replacements))
        return bool(replacements)

    # ---------------------------------------------------------------- #
    # Knob mutators
    # ---------------------------------------------------------------- #
    def set_start(self, start: str | list[str] | None) -> None:
        """Set the round-0 nodes (validated as at construction); empty clears them."""
        self._start = self._check_start(start, self._nodes)

    def set_returns(self, returns: str | list[str] | None) -> None:
        """Set the result selection (validated as at construction)."""
        self._returns = self._check_returns(returns)

    def set_collision_handling_order(self, order: list[str] | None) -> None:
        """Set the collision order (validated as at construction)."""
        self._collision_handling_order = self._check_order(order, self._nodes)

    # ---------------------------------------------------------------- #
    # Injection (add a group of named nodes, optionally bridged to existing ones)
    # ---------------------------------------------------------------- #
    def _stage_nodes(self, nodes: Any) -> dict[str, tuple[AtomicInvokable, Binding | None]]:
        """Validate and coerce a dict of new nodes without changing the graph."""
        if not isinstance(nodes, dict):
            raise TypeError(f"nodes must be a dict, got {type(nodes)!r}")
        if not nodes:
            raise ValueError("nodes must not be empty")
        clashes = [n for n in nodes if n in self._nodes]
        if clashes:
            raise ValueError(f"{self.full_name}: node name(s) already exist: {clashes!r}")
        return {n: self._coerce_node(self.name, n, entry) for n, entry in nodes.items()}

    def _link_names(self, label: str, value: Any) -> tuple[str, ...]:
        """Normalize a bridge argument (``None``, a name or a list) to existing node names."""
        if value is None:
            return ()
        if isinstance(value, str):
            names: tuple[str, ...] = (value,)
        elif isinstance(value, list):
            names = tuple(value)
        else:
            raise TypeError(f"{label} must be None, a str or a list[str], got {type(value)!r}")
        self._check_node_names(names, self._nodes, label)
        return names

    def add_sequence(
        self,
        nodes: dict[str, Any],
        *,
        start_from: str | list[str] | None = None,
        end_at: str | list[str] | None = None,
        repeat_until: dict[str, Any] | None = None,
    ) -> None:
        """Add ``nodes`` as a chain in dict order, optionally bridged and looped.

        1. Validate the new nodes and the bridge names; nothing changes yet.
        2. ``start_from`` nodes get an edge to the first new node; the last new
           node gets an edge to each ``end_at`` node.
        3. ``repeat_until`` replaces that last edge with a router that returns the
           first node until every key matches its value in state (a missing key
           does not match), then the single ``end_at`` node or ``None``. The body
           runs at least once; the depth cap ends a loop that never matches.
        """
        staged = self._stage_nodes(nodes)
        sources = self._link_names("start_from", start_from)
        sinks = self._link_names("end_at", end_at)
        names = list(staged)
        head = names[0]

        router: AtomicInvokable | None = None
        if repeat_until is not None:
            if not isinstance(repeat_until, dict) or not all(
                isinstance(k, str) for k in repeat_until
            ):
                raise TypeError(f"repeat_until must be a dict with str keys, got {repeat_until!r}")
            if not repeat_until:
                raise ValueError("repeat_until must not be empty")
            if len(sinks) > 1:
                raise ValueError("end_at must be a single node when repeat_until is given")
            expected = dict(repeat_until)
            exit_to = sinks[0] if sinks else None

            def decide(**state: Any) -> str | None:
                done = all(k in state and state[k] == v for k, v in expected.items())
                return exit_to if done else head

            router = self._coerce_router(self.name, decide, self._router_count)

        records: dict[str, Graph.Node] = {}
        for i, node_name in enumerate(names):
            invokable, binding = staged[node_name]
            if i + 1 < len(names):
                records[node_name] = Graph.Node(invokable, binding, (names[i + 1],), ())
            elif router is not None:
                records[node_name] = Graph.Node(invokable, binding, (), (router,))
            else:
                records[node_name] = Graph.Node(invokable, binding, sinks, ())
        for source in sources:
            node = self._nodes[source]
            self._nodes[source] = replace(node, outgoing=node.outgoing + (head,))
        self._nodes.update(records)
        if router is not None:
            self._router_count += 1
        logger.debug("%s: added sequence %s", self.full_name, names)

    def add_parallel(
        self,
        nodes: dict[str, Any],
        *,
        fan_out: str | list[str] | None = None,
        fan_in: str | list[str] | None = None,
    ) -> None:
        """Add ``nodes`` as independent branches, optionally bridged.

        Each ``fan_out`` node gets an edge to every branch and every branch gets an
        edge to each ``fan_in`` node (which runs once, after all branches).
        """
        staged = self._stage_nodes(nodes)
        sources = self._link_names("fan_out", fan_out)
        sinks = self._link_names("fan_in", fan_in)
        names = tuple(staged)
        records = {
            n: Graph.Node(invokable, binding, sinks, ()) for n, (invokable, binding) in staged.items()
        }
        for source in sources:
            node = self._nodes[source]
            self._nodes[source] = replace(node, outgoing=node.outgoing + names)
        self._nodes.update(records)
        logger.debug("%s: added parallel branches %s", self.full_name, list(names))

    # ---------------------------------------------------------------- #
    # Run helpers
    # ---------------------------------------------------------------- #
    def _seed_state(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Seed state from filtered inputs, flattening the ``**kwargs`` payload."""
        state = dict(inputs)
        varkw = next(
            (p.name for p in self.parameters if p.kind == ParamSpec.VAR_KEYWORD), None
        )
        if varkw is not None and varkw in state:
            payload = state.pop(varkw)
            clash = sorted(set(payload) & set(state))
            if clash:
                raise TypeError(
                    f"{self.full_name}: extra input keys collide with named parameters: {clash!r}"
                )
            state.update(payload)
        return state

    def _bind(self, node_name: str, node: Graph.Node, response: Any) -> dict[str, Any]:
        """Apply the node's binding to its response and return the state writes."""
        binding = node.binding
        if binding is None:
            if not isinstance(response, Mapping):
                raise TypeError(
                    f"{self.full_name}: node {node_name!r} is unbound and returned "
                    f"{type(response)!r}; bind it with (node, 'key') to store a non-mapping response"
                )
            return dict(response)
        writes = slice_response(
            response,
            binding,
            label=f"{self.full_name} node {node_name!r} binding",
            on_missing="skip",
        )
        if isinstance(binding, dict):
            skipped = set(binding.values()) - set(writes)
            if skipped:
                logger.debug(
                    "%s: node %r skipped missing selector(s) for %s",
                    self.full_name,
                    node_name,
                    sorted(skipped),
                )
        return writes

    def _select_returns(
        self, state: dict[str, Any], returns: str | tuple[str, ...] | None
    ) -> Any:
        """Pick the result out of state; a requested key never written raises."""
        if returns is None:
            return dict(state)
        keys = (returns,) if isinstance(returns, str) else returns
        missing = [k for k in keys if k not in state]
        if missing:
            raise KeyError(f"{self.full_name}: returns key(s) {missing!r} were never populated")
        if isinstance(returns, str):
            return state[returns]
        return {k: state[k] for k in keys}

    def _router_choice(
        self, node_name: str, value: Any, nodes: dict[str, Graph.Node]
    ) -> str | None:
        """Validate one router result: a node name or ``None``."""
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"{self.full_name}: router on node {node_name!r} must return a node name "
                f"or None, got {type(value)!r}"
            )
        if value not in nodes:
            raise ValueError(
                f"{self.full_name}: router on node {node_name!r} returned {value!r}, "
                "which is not a node"
            )
        return value

    # ---------------------------------------------------------------- #
    # Run
    # ---------------------------------------------------------------- #
    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Sync is the async path bridged; there is no second engine."""
        return run_coro_sync(self._async_invoke_impl(inputs))

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run supersteps until nothing is ready; raise at the depth cap."""
        # 1. snapshot of topology and knobs
        nodes = dict(self._nodes)
        start = self._start
        returns = self._returns
        depth = self._max_traversal_depth
        order = self._collision_handling_order or ()
        if not start:
            raise RuntimeError(f"{self.full_name}: no start nodes; call set_start(...) first")
        rank = {n: i for i, n in enumerate([*order, *(n for n in nodes if n not in order)])}

        # 2. state and loop bookkeeping
        state = self._seed_state(inputs)
        ready: set[str] = set(start)
        trace: list[tuple[str, ...]] = []

        while True:
            # 3. this superstep's nodes, in declaration order
            names = [n for n in nodes if n in ready]
            trace.append(tuple(names))
            logger.debug("%s: superstep %d fires %s", self.full_name, len(trace), names)

            # 4. run concurrently against one frozen snapshot
            snapshot = dict(state)
            results = await asyncio.gather(
                *(nodes[n].invokable.async_invoke(snapshot) for n in names)
            )

            # 5. bind, then merge so the highest-ranked writer is applied last
            writes = {
                n: self._bind(n, nodes[n], r.result) for n, r in zip(names, results)
            }
            for n in sorted(names, key=rank.__getitem__, reverse=True):
                state.update(writes[n])

            # 6. routers read the merged state
            calls = [(n, router) for n in names for router in nodes[n].routers]
            router_results = await asyncio.gather(
                *(router.async_invoke(dict(state)) for _, router in calls)
            )

            # 7. next ready set
            next_ready: set[str] = set()
            for n in names:
                next_ready.update(nodes[n].outgoing)
            for (n, _), router_result in zip(calls, router_results):
                choice = self._router_choice(n, router_result.result, nodes)
                if choice is not None:
                    next_ready.add(choice)

            # 8. finish, or raise at the cap, or continue
            if not next_ready:
                break
            if len(trace) >= depth:
                raise RuntimeError(
                    f"{self.full_name}: reached max_traversal_depth={depth} with "
                    f"nodes still ready: {sorted(next_ready)!r}"
                )
            ready = next_ready

        # 9. result
        return {
            "result": self._select_returns(state, returns),
            "trace": tuple(trace),
        }

    # ---------------------------------------------------------------- #
    # Serialization
    # ---------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph's nodes, edges and knobs."""
        data = super().to_dict()
        edges: list[list[Any]] = []
        for node_name, node in self._nodes.items():
            edges.extend([node_name, target] for target in node.outgoing)
            edges.extend([node_name, router.to_dict()] for router in node.routers)
        data.update(
            {
                "nodes": {
                    node_name: {
                        "invokable": node.invokable.to_dict(),
                        "binding": dict(node.binding)
                        if isinstance(node.binding, dict)
                        else node.binding,
                    }
                    for node_name, node in self._nodes.items()
                },
                "edges": edges,
                "start": list(self._start),
                "returns": list(self._returns)
                if isinstance(self._returns, tuple)
                else self._returns,
                "max_traversal_depth": self._max_traversal_depth,
                "collision_handling_order": list(self._collision_handling_order)
                if self._collision_handling_order is not None
                else None,
            }
        )
        return data
