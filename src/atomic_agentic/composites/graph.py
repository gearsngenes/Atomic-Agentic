from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, ClassVar

from ..invokables.atomic_invokable import AtomicInvokable
from ..models.parameters import ParamSpec
from ..models.results import AtomicResult
from ..tools.toolification import to_invokable
from ..utils.binding import Binding, normalize_binding, slice_response
from ..utils.core import run_coro_sync

logger = logging.getLogger(__name__)

__all__ = ["Graph", "GraphResult"]


@dataclass(frozen=True, slots=True)
class GraphResult(AtomicResult):
    """Result of one ``Graph`` run.

    Fields
    ------
    trace:
        Node names fired per superstep (declaration order within a superstep),
        always populated. Routers do not appear.
    hit_depth_limit:
        True when the run stopped at ``max_traversal_depth`` with a non-empty
        ready set (only possible when ``raise_on_depth_limit`` is false).
    """

    trace: tuple[tuple[str, ...], ...] = field(default=(), kw_only=True)
    hit_depth_limit: bool = field(default=False, kw_only=True)


class Graph(AtomicInvokable):
    """Superstep-batched node graph over a shared state dict.

    Construction contract
    ---------------------
    - ``nodes`` is a non-empty dict of name to entry (an invokable, a ``str``, a
      callable, or ``(entry, binding)``). Entries are coerced through the shared
      ``to_invokable`` and named by their dict key.
    - ``edges`` are ``(source, target)`` pairs: a ``str`` target is a node name
      (fixed edge); a callable or invokable target is a router attached to
      ``source``. A node with no outgoing edges is terminal.
    - ``start`` names the round-0 nodes (several start nodes run in parallel).
    - ``parameters`` default to a single ``**kwargs``; ``return_type`` is ``Any``.
      Both are fixed. Topology is fixed in this pass; mutators come later.

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
      still ready it raises (``raise_on_depth_limit``) or stops and returns the
      normal slice with ``hit_depth_limit`` set.
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
        nodes: dict[str, AtomicInvokable | str | Callable[..., Any] | tuple[Any, Binding | None]],
        edges: list[tuple[str, str | AtomicInvokable | Callable[..., Any]]],
        start: str | list[str],
        *,
        parameters: list[ParamSpec] | None = None,
        returns: str | list[str] | None = None,
        max_traversal_depth: int = 50,
        raise_on_depth_limit: bool = True,
        collision_handling_order: list[str] | None = None,
    ) -> None:
        # 1. nodes container
        if not isinstance(nodes, dict):
            raise TypeError(f"nodes must be a non-empty dict, got {type(nodes)!r}")
        if not nodes:
            raise ValueError("nodes must not be empty")

        # 2. node entries
        built: dict[str, tuple[AtomicInvokable, Binding | None]] = {}
        for node_name, entry in nodes.items():
            if not isinstance(node_name, str) or not node_name:
                raise TypeError(f"node names must be non-empty str, got {node_name!r}")
            if isinstance(entry, tuple):
                if len(entry) != 2:
                    raise ValueError(
                        f"node {node_name!r} tuple must be (node, binding); got length {len(entry)}"
                    )
                obj, raw_binding = entry
            else:
                obj, raw_binding = entry, None
            binding = normalize_binding(raw_binding, label=f"{name} node {node_name!r} binding")
            invokable = to_invokable(
                obj,
                name=node_name,
                description=f"Node {node_name} of graph {name}",
            )
            built[node_name] = (invokable, binding)

        # 3. edges
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
            if not isinstance(source, str):
                raise TypeError(f"edge source must be a str, got {type(source)!r}")
            if source not in built:
                raise ValueError(f"edge source {source!r} is not a node")
            if isinstance(target, str):
                if target not in built:
                    raise ValueError(f"edge target {target!r} is not a node")
                if target not in outgoing[source]:
                    outgoing[source].append(target)
            else:
                routers[source].append(
                    to_invokable(
                        target,
                        name=f"{name}_router_{router_count}",
                        description=f"Router {router_count} of graph {name}",
                    )
                )
                router_count += 1

        # 4. start
        if isinstance(start, str):
            start_tuple: tuple[str, ...] = (start,)
        elif isinstance(start, list):
            start_tuple = tuple(start)
        else:
            raise TypeError(f"start must be a str or list[str], got {type(start)!r}")
        if not start_tuple:
            raise ValueError("start must not be empty")
        self._check_node_names(start_tuple, built, "start")

        # 5. depth and flag
        if isinstance(max_traversal_depth, bool) or not isinstance(max_traversal_depth, int):
            raise TypeError(f"max_traversal_depth must be an int, got {type(max_traversal_depth)!r}")
        if max_traversal_depth <= 0:
            raise ValueError("max_traversal_depth must be > 0")
        if not isinstance(raise_on_depth_limit, bool):
            raise TypeError(f"raise_on_depth_limit must be a bool, got {type(raise_on_depth_limit)!r}")

        # 6. returns
        returns_value: str | tuple[str, ...] | None
        if returns is None:
            returns_value = None
        elif isinstance(returns, str):
            if not returns:
                raise TypeError("returns must be a non-empty str when given as a str")
            returns_value = returns
        elif isinstance(returns, list) and all(isinstance(k, str) for k in returns):
            returns_value = tuple(returns)
        else:
            raise TypeError(f"returns must be None, a str or a list[str], got {returns!r}")

        # 7. collision order
        order_value: tuple[str, ...] | None
        if collision_handling_order is None:
            order_value = None
        elif isinstance(collision_handling_order, list) and all(
            isinstance(n, str) for n in collision_handling_order
        ):
            order_value = tuple(collision_handling_order)
            self._check_node_names(order_value, built, "collision_handling_order")
        else:
            raise TypeError(
                f"collision_handling_order must be None or a list[str], got {collision_handling_order!r}"
            )

        # 8. base init
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

        # 9. store
        self._nodes: dict[str, Graph.Node] = {
            node_name: Graph.Node(
                invokable=invokable,
                binding=binding,
                outgoing=tuple(outgoing[node_name]),
                routers=tuple(routers[node_name]),
            )
            for node_name, (invokable, binding) in built.items()
        }
        self._start = start_tuple
        self._returns = returns_value
        self._max_traversal_depth = max_traversal_depth
        self._raise_on_depth_limit = raise_on_depth_limit
        self._collision_handling_order = order_value

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

    @property
    def raise_on_depth_limit(self) -> bool:
        """Whether reaching the cap with nodes still ready raises."""
        return self._raise_on_depth_limit

    @property
    def collision_handling_order(self) -> tuple[str, ...] | None:
        """Node names that win same-key collisions first, in order (``None``: declaration order)."""
        return self._collision_handling_order

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
        """Run supersteps until nothing is ready or the depth cap is reached."""
        # 1. snapshot of topology and knobs
        nodes = dict(self._nodes)
        start = self._start
        returns = self._returns
        depth = self._max_traversal_depth
        raise_on_limit = self._raise_on_depth_limit
        order = self._collision_handling_order or ()
        rank = {n: i for i, n in enumerate([*order, *(n for n in nodes if n not in order)])}

        # 2. state and loop bookkeeping
        state = self._seed_state(inputs)
        ready: set[str] = set(start)
        trace: list[tuple[str, ...]] = []
        hit_depth_limit = False

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

            # 8. finish, cap, or continue
            if not next_ready:
                break
            if len(trace) >= depth:
                if raise_on_limit:
                    raise RuntimeError(
                        f"{self.full_name}: reached max_traversal_depth={depth} with "
                        f"nodes still ready: {sorted(next_ready)!r}"
                    )
                hit_depth_limit = True
                break
            ready = next_ready

        # 9. result
        return {
            "result": self._select_returns(state, returns),
            "trace": tuple(trace),
            "hit_depth_limit": hit_depth_limit,
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
                "raise_on_depth_limit": self._raise_on_depth_limit,
                "collision_handling_order": list(self._collision_handling_order)
                if self._collision_handling_order is not None
                else None,
            }
        )
        return data
