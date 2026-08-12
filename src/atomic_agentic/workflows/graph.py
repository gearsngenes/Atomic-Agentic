from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from types import MappingProxyType
from typing import Any, ClassVar

from ..exceptions import ValidationError, WorkflowError
from ..core.Invokable import AtomicInvokable
from ..models.parameters import ParamSpec, ParameterReport
from ..models.workflows.graph import GraphFlowNode, StatePolicySpec
from ..models.results import AtomicResult
from ..models.results.workflows import GraphFlowResult
from ..utils.core import run_coro_sync
from ..utils.parameters import _simplify_type_set, build_parameter_reports
from .base import Workflow

logger = logging.getLogger(__name__)

__all__ = ["GraphFlow"]


class GraphFlow(Workflow):
    """Cyclic, dynamically-routed node graph with superstep-batched (BSP)
    execution.

    Construction contract
    ----------------------
    - ``nodes``: non-empty ``dict[str, AtomicInvokable]``, alias -> node.
    - ``edges``: ``list[tuple[str, str | None | AtomicInvokable]]``.
      ``(source, "target")`` is a fixed edge; ``(source, None)`` marks
      ``source`` as a declared dead end (mutually exclusive with any real
      edge/router sharing that ``source``); ``(source, router_invokable)``
      attaches a router to ``source`` (multiple such tuples for the same
      ``source`` attach multiple routers).
    - ``start``: single entry node name; ``GraphFlow.parameters`` has
      exactly ``nodes[start].parameters``'s own names, in the same order.
      Any name ``start`` shares with another configured node has its
      ``type`` widened to the full cross-node compatible witness set;
      ``kind``/``default``/``description`` always come from ``start``'s
      own declaration, never any other node's.
    - ``state_policies`` / ``priorities``: mutable post-construction via
      :meth:`set_state_policy`/:meth:`remove_state_policy`/
      :meth:`set_priority` (same shape-vs-behavior justification
      ``IterativeFlow.checkers`` already has -- never touch declared
      ``parameters``/``return_type``).
    - ``max_edge_traversals`` / ``stop_early``: mutable behavioral knobs.
    - ``result_mode`` / ``return_keys``: fixed at construction -- feed
      ``return_type``.
    - Construction runs a whole-graph parameter reconciliation
      (:func:`build_parameter_reports`) across every node's declared
      parameters, independent of reachability from ``start`` -- this both
      raises ``WorkflowError`` on a genuine type/kind conflict between any
      two nodes sharing a name, and supplies the type-widening described
      above.

    Runtime contract
    -----------------
    Execution proceeds in supersteps: the whole ready set runs
    concurrently against a frozen state snapshot; nothing merges until the
    entire superstep completes; a node fires at most once per superstep
    even if reached by multiple edges in that same superstep, but can fire
    again in a later, distinct superstep -- this is what makes cycles work,
    there is no AND-join / wait-for-all-predecessors concept anywhere in
    this design. The run ends when the ready queue is empty, when
    ``max_edge_traversals`` is hit, or (if ``stop_early`` is set and
    ``return_keys`` is non-empty) as soon as every requested key is
    already present in state.

    Result
    ------
    See ``GraphFlowResult`` for per-run fields (``edge_traversals``,
    ``termination_reason``, ``max_edge_traversals``, ``result_mode``,
    ``return_keys``).
    """

    SCALAR: ClassVar[str] = "scalar"
    LIST: ClassVar[str] = "list"
    TUPLE: ClassVar[str] = "tuple"
    SET: ClassVar[str] = "set"
    DICT: ClassVar[str] = "dict"

    _RESULT_MODES: ClassVar[frozenset[str]] = frozenset({SCALAR, LIST, TUPLE, SET, DICT})

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        nodes: dict[str, AtomicInvokable],
        edges: list[tuple[str, str | None | AtomicInvokable]],
        start: str,
        *,
        state_policies: list[StatePolicySpec] | None = None,
        priorities: dict[str, int] | None = None,
        max_edge_traversals: int | None = None,
        stop_early: bool = False,
        result_mode: str = DICT,
        return_keys: list[str] | None = None,
        include_trace: bool = True,
    ) -> None:
        """Initialize the graph workflow.

        Parameters
        ----------
        nodes:
            Non-empty alias -> AtomicInvokable mapping.
        edges:
            ``(source, target)`` pairs. ``target`` is a node name (fixed
            edge), ``None`` (declared dead end), or an ``AtomicInvokable``
            (attaches a router to ``source``).
        start:
            Entry node name; determines this flow's own ``parameters``.
        state_policies:
            Optional bulk-construction list of ``StatePolicySpec``, each
            registered via :meth:`set_state_policy`.
        priorities:
            Optional node-name -> priority mapping; unlisted nodes default
            to priority ``1``.
        max_edge_traversals:
            Optional cap on the number of supersteps a run may take.
        stop_early:
            Whether to stop as soon as every requested key is already
            populated, independent of the ready queue.
        result_mode:
            One of ``SCALAR``/``LIST``/``TUPLE``/``SET``/``DICT``.
        return_keys:
            State keys to project into the result, in requested order.
        include_trace:
            Whether results populate ``trace``. Forwarded to the base
            ``Workflow``.
        """
        if not isinstance(nodes, dict):
            raise TypeError(f"nodes must be a dict[str, AtomicInvokable], got {type(nodes)!r}")
        if not nodes:
            raise ValueError("nodes must not be empty")
        for node_name, invokable in nodes.items():
            if not isinstance(invokable, AtomicInvokable):
                raise TypeError(
                    f"nodes[{node_name!r}] must be AtomicInvokable, got {type(invokable)!r}"
                )

        if not isinstance(start, str):
            raise TypeError(f"start must be a str, got {type(start)!r}")
        if start not in nodes:
            raise KeyError(f"start {start!r} is not a configured node in nodes")
        self._start: str = start

        if not isinstance(edges, list):
            raise TypeError(f"edges must be a list, got {type(edges)!r}")

        leaf_marked: dict[str, bool] = {node_name: False for node_name in nodes}
        fixed_targets: dict[str, list[str]] = {node_name: [] for node_name in nodes}
        routers_by_source: dict[str, list[AtomicInvokable]] = {node_name: [] for node_name in nodes}

        for edge in edges:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise TypeError(f"each edge must be a 2-tuple, got {edge!r}")
            source, target = edge
            if not isinstance(source, str):
                raise TypeError(f"edge source must be a str, got {type(source)!r}")
            if source not in nodes:
                raise KeyError(f"edge source {source!r} is not a configured node in nodes")

            if target is None:
                leaf_marked[source] = True
            elif isinstance(target, str):
                if target not in nodes:
                    raise KeyError(f"edge target {target!r} is not a configured node in nodes")
                fixed_targets[source].append(target)
            elif isinstance(target, AtomicInvokable):
                routers_by_source[source].append(target)
            else:
                raise TypeError(
                    f"edge target must be a str, None, or AtomicInvokable, got {type(target)!r}"
                )

        # A source explicitly marked a dead end can't also fire real edges
        # or routers -- the marker would be meaningless otherwise.
        for node_name in nodes:
            if leaf_marked[node_name] and (fixed_targets[node_name] or routers_by_source[node_name]):
                raise ValueError(
                    f"node {node_name!r} cannot combine a leaf marker (None target) with "
                    "real edges or routers"
                )

        if priorities is not None:
            if not isinstance(priorities, dict):
                raise TypeError(f"priorities must be a dict[str, int], got {type(priorities)!r}")
            for node_name in priorities:
                if node_name not in nodes:
                    raise KeyError(f"priorities key {node_name!r} is not a configured node in nodes")

        # incoming only reflects fixed edges -- a router's target isn't
        # known until runtime, so it can never statically populate another
        # node's incoming tuple.
        incoming: dict[str, list[str]] = {node_name: [] for node_name in nodes}
        for source, targets in fixed_targets.items():
            for target in targets:
                incoming[target].append(source)

        resolved_priorities = priorities or {}
        self._node_data: dict[str, GraphFlowNode] = {
            node_name: GraphFlowNode(
                invokable=invokable,
                incoming=tuple(incoming[node_name]),
                outgoing=tuple(fixed_targets[node_name]),
                routers=tuple(routers_by_source[node_name]),
                priority=resolved_priorities.get(node_name, 1),
            )
            for node_name, invokable in nodes.items()
        }

        # Checkers/state-policies precedent: single validated entry point,
        # bulk construction just replays it -- no duplicated validation.
        self._state_policies: dict[str, StatePolicySpec] = {}
        for spec in state_policies or []:
            self.set_state_policy(spec.key, spec.raise_on_collision, spec.tiebreak)

        if max_edge_traversals is not None:
            if not isinstance(max_edge_traversals, int):
                raise TypeError(
                    f"max_edge_traversals must be an int or None, got {type(max_edge_traversals)!r}"
                )
            if max_edge_traversals <= 0:
                raise ValueError("max_edge_traversals must be > 0")
        self._max_edge_traversals: int | None = max_edge_traversals

        if not isinstance(stop_early, bool):
            raise TypeError(f"stop_early must be a bool, got {type(stop_early)!r}")
        self._stop_early: bool = stop_early

        normalized_mode = str(result_mode).strip().lower()
        if normalized_mode not in self._RESULT_MODES:
            raise ValueError(
                f"result_mode must be one of {sorted(self._RESULT_MODES)!r}, got {result_mode!r}"
            )
        self._result_mode: str = normalized_mode

        if return_keys is not None:
            if not isinstance(return_keys, list) or not all(isinstance(k, str) for k in return_keys):
                raise TypeError(f"return_keys must be a list[str] or None, got {return_keys!r}")
        normalized_return_keys = tuple(return_keys or ())
        if self._result_mode == self.SCALAR and len(normalized_return_keys) > 1:
            raise ValueError("result_mode='scalar' allows at most one requested key")
        self._return_keys: tuple[str, ...] = normalized_return_keys

        # Whole-graph parameter reconciliation -- every node, regardless of
        # whether it's actually reachable from start on any real path.
        # start is forced to source index 0 so its own observation is
        # always the reconciliation anchor: the winning source whenever it
        # declares a name, and therefore the source of kind/default/
        # description on GraphFlow's own advertised parameters.
        node_names_in_order = [start, *(n for n in self._node_data if n != start)]
        reports: list[ParameterReport] = build_parameter_reports(
            [list(self._node_data[n].invokable.parameters) for n in node_names_in_order]
        )

        parameters: list[ParamSpec] = []
        for report in reports:
            if not report.witness_types or not report.kind_compatible:
                conflicting = ", ".join(
                    f"{node_names_in_order[index]!r} ({' | '.join(spec.type)}/{spec.kind})"
                    for index, spec in enumerate(report.observations)
                    if spec is not None
                )
                raise WorkflowError(
                    f"GraphFlow parameter {report.parameter_name!r} conflicts across nodes: "
                    f"{conflicting} (same name, incompatible type/kind)."
                )

            start_spec = report.observations[0]
            if start_spec is not None:
                parameters.append(
                    replace(start_spec, type=tuple(sorted(report.witness_types)))
                )

        if self._result_mode == self.SCALAR:
            if self._return_keys:
                key_report = next(
                    (r for r in reports if r.parameter_name == self._return_keys[0]), None
                )
                if key_report is not None:
                    # Flatten every declaring node's own raw type tuple
                    # (each itself already a proven-mutually-compatible
                    # tuple, per the collision check above) into one set,
                    # drop the wildcard "Any" (adds no real information),
                    # collapse any bare-origin/parameterized-sibling
                    # redundancy (dict + dict[str, int] -> dict), then join
                    # what's left as a union -- same representation
                    # convention as every other return_type in this pass.
                    # Deliberately the full raw union, not narrowed to
                    # key_report.witness_types -- an output question needing
                    # every possibility, not an input question needing
                    # common ground. Empty after dropping "Any" (every node
                    # typed this key Any, or key_report doesn't exist) ->
                    # "Any" is the honest answer, not an arbitrary pick
                    # among genuine candidates.
                    flat_types: set[str] = set()
                    for spec in key_report.observations:
                        if spec is not None:
                            flat_types.update(spec.type)
                    concrete_types = _simplify_type_set(flat_types - {"Any"})
                    return_type = " | ".join(sorted(concrete_types)) if concrete_types else "Any"
                else:
                    return_type = "Any"
            else:
                return_type = "None"
        else:
            return_type = self._result_mode

        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            parameters=parameters,
            return_type=return_type,
            include_trace=include_trace,
        )

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def nodes(self) -> MappingProxyType[str, GraphFlowNode]:
        """Read-only live view of node topology/priority. Entries are
        frozen GraphFlowNode instances -- safe against mutation even
        through this view."""
        return MappingProxyType(self._node_data)

    @property
    def start(self) -> str:
        """Fixed entry node name."""
        return self._start

    @property
    def state_policies(self) -> MappingProxyType[str, StatePolicySpec]:
        """Read-only live view of currently configured state policies."""
        return MappingProxyType(self._state_policies)

    @property
    def max_edge_traversals(self) -> int | None:
        """Current superstep cap. None means unbounded."""
        return self._max_edge_traversals

    @max_edge_traversals.setter
    def max_edge_traversals(self, value: int | None) -> None:
        """Validate and set the superstep cap."""
        if value is not None and not isinstance(value, int):
            raise TypeError(f"max_edge_traversals must be an int or None, got {type(value)!r}")
        if value is not None and value <= 0:
            raise ValueError("max_edge_traversals must be > 0")
        self._max_edge_traversals = value

    @property
    def stop_early(self) -> bool:
        """Whether a run stops as soon as every requested key is populated."""
        return self._stop_early

    @stop_early.setter
    def stop_early(self, value: bool) -> None:
        """Validate and set the early-exit knob."""
        if not isinstance(value, bool):
            raise TypeError(f"stop_early must be a bool, got {type(value)!r}")
        self._stop_early = value

    @property
    def result_mode(self) -> str:
        """Fixed output projection mode."""
        return self._result_mode

    @property
    def return_keys(self) -> tuple[str, ...]:
        """Fixed set of requested state keys, in requested order."""
        return self._return_keys

    # ------------------------------------------------------------------ #
    # Mutators (state_policies / priorities -- not fixed topology)
    # ------------------------------------------------------------------ #
    def set_state_policy(
        self, key: str, raise_on_collision: bool, tiebreak: str | None = None
    ) -> None:
        """Register or overwrite the collision policy for ``key``.

        Upsert semantics -- unlike ``add_checker``'s insert-only raise, a
        second call for the same key just updates it.
        """
        spec = StatePolicySpec(key=key, raise_on_collision=raise_on_collision, tiebreak=tiebreak)
        self._state_policies[key] = spec

    def remove_state_policy(self, key: str) -> None:
        """Remove the collision policy registered for ``key``.

        Raises ``ValueError`` if none is configured there.
        """
        if key not in self._state_policies:
            raise ValueError(f"no state policy is configured for key {key!r}")
        del self._state_policies[key]

    def set_priority(self, node_name: str, priority: int) -> None:
        """Set ``node_name``'s priority.

        Raises ``KeyError`` if ``node_name`` isn't configured, ``TypeError``
        if ``priority`` isn't an int. ``GraphFlowNode`` is frozen, so this
        replaces the whole entry rather than mutating it in place.
        """
        if node_name not in self._node_data:
            raise KeyError(f"{node_name!r} is not a configured node")
        if not isinstance(priority, int):
            raise TypeError(f"priority must be an int, got {type(priority)!r}")
        self._node_data[node_name] = replace(self._node_data[node_name], priority=priority)

    def _extra_description(self) -> str:
        """Fixed note on the graph's entry point and superstep cap."""
        cap_note = (
            f"Bounded to {self._max_edge_traversals} superstep(s)."
            if self._max_edge_traversals is not None
            else "Unbounded superstep count."
        )
        return f"Enters at node {self._start!r}. {cap_note}"

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #
    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> GraphFlowResult:
        """Construct a GraphFlowResult envelope for this workflow's invocation.

        ``edge_traversals``/``termination_reason``/``trace`` arrive via
        ``result_kwargs`` from ``_async_run``, same passthrough pattern
        ``IterativeFlow.make_result`` uses.
        """
        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=GraphFlowResult,
            max_edge_traversals=self._max_edge_traversals,
            result_mode=self._result_mode,
            return_keys=self._return_keys,
            **result_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Synchronously execute the graph's superstep loop.

        Sync is the async path bridged, not a second implementation --
        matches ParallelFlow._run's exact pattern. No independent sync
        engine: GraphFlow's per-round concurrency is exactly what
        asyncio.gather already provides, and every AtomicInvokable's
        default async_invoke already backstops sync-only nodes via a
        background thread -- a hand-rolled sync engine would either
        serialize a round's nodes (weakening the core "everything ready
        runs together" guarantee) or duplicate that existing thread-bridge
        for no behavioral gain.
        """
        return run_coro_sync(self._async_run(inputs))

    async def _async_run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Asynchronously execute the graph's superstep loop."""

        async def run_node(node_name: str, invokable: AtomicInvokable, node_inputs: Mapping[str, Any]) -> AtomicResult:
            try:
                return await invokable.async_invoke(node_inputs)
            except Exception as exc:
                raise RuntimeError(
                    f"{self.full_name}: node {node_name!r} ({invokable.full_name}) "
                    "failed during invoke"
                ) from exc

        async def run_router(node_name: str, router: AtomicInvokable, router_inputs: Mapping[str, Any]) -> AtomicResult:
            try:
                return await router.async_invoke(router_inputs)
            except Exception as exc:
                raise RuntimeError(
                    f"{self.full_name}: router attached to node {node_name!r} "
                    f"({router.full_name}) failed during invoke"
                ) from exc

        state: dict[str, Any] = dict(inputs)
        ready: set[str] = {self._start}
        edge_traversals: list[tuple[str, ...]] = []
        trace_entries: list[AtomicResult] = []
        termination_reason = "queue_empty"

        while ready:
            # Cap counts completed rounds -- stop before starting one more.
            if self._max_edge_traversals is not None and len(edge_traversals) >= self._max_edge_traversals:
                termination_reason = "cap_hit"
                break

            # Fixed nodes-declaration order, filtered to this round's ready
            # set -- deterministic regardless of which edges produced it.
            round_names = [name for name in self._node_data if name in ready]

            # Frozen read snapshot; state itself isn't touched again until
            # this whole round's writes are merged below.
            snapshot = dict(state)

            round_results: list[AtomicResult] = list(
                await asyncio.gather(
                    *[
                        run_node(name, self._node_data[name].invokable, snapshot)
                        for name in round_names
                    ]
                )
            )
            trace_entries.extend(round_results)

            for node_name, result in zip(round_names, round_results):
                if not isinstance(result.result, Mapping):
                    raise ValidationError(
                        f"{self.full_name}: node {node_name!r} must produce a "
                        f"mapping-shaped result, got {type(result.result)!r}"
                    )

            # Flattened across every node in this round -- different
            # nodes' routers have no dependency on each other, so they're
            # gathered together, not per-node.
            router_calls = [
                (node_name, router, round_results[index].result)
                for index, node_name in enumerate(round_names)
                for router in self._node_data[node_name].routers
            ]
            router_results: list[AtomicResult] = list(
                await asyncio.gather(
                    *[
                        run_router(node_name, router, router_inputs)
                        for node_name, router, router_inputs in router_calls
                    ]
                )
            )
            trace_entries.extend(router_results)

            for (node_name, _router, _router_inputs), router_result in zip(router_calls, router_results):
                value = router_result.result
                if value is not None and not isinstance(value, str):
                    raise ValidationError(
                        f"{self.full_name}: router attached to node {node_name!r} must "
                        f"return None or a node name, got {type(value)!r}"
                    )
                if isinstance(value, str) and value not in self._node_data:
                    raise ValidationError(
                        f"{self.full_name}: router attached to node {node_name!r} named "
                        f"{value!r}, which is not a configured node"
                    )

            next_ready: set[str] = set()
            for node_name in round_names:
                next_ready |= set(self._node_data[node_name].outgoing)
            for (_node_name, _router, _router_inputs), router_result in zip(router_calls, router_results):
                if router_result.result is not None:
                    next_ready.add(router_result.result)

            # Collect every key this round's nodes wrote, then resolve
            # same-superstep collisions in one merge pass.
            contributors: dict[str, list[tuple[str, Any]]] = {}
            for node_name, result in zip(round_names, round_results):
                for key, value in result.result.items():
                    contributors.setdefault(key, []).append((node_name, value))

            for key, writers in contributors.items():
                if len(writers) == 1:
                    state[key] = writers[0][1]
                    continue

                policy = self._state_policies.get(key)
                if policy is not None and policy.raise_on_collision:
                    raise ValidationError(
                        f"{self.full_name}: state key {key!r} written by multiple "
                        f"nodes this superstep: {[writer_name for writer_name, _ in writers]!r}"
                    )

                tiebreak = policy.tiebreak if policy is not None else "last"
                max_priority = max(self._node_data[writer_name].priority for writer_name, _ in writers)
                tied = [
                    (writer_name, value)
                    for writer_name, value in writers
                    if self._node_data[writer_name].priority == max_priority
                ]
                state[key] = tied[0][1] if tiebreak == "first" else tied[-1][1]

            edge_traversals.append(tuple(round_names))

            # stop_early only ever has a trigger condition when specific
            # keys were actually requested -- with no return_keys there's
            # nothing to check presence of, so this never fires then.
            if self._stop_early and self._return_keys and all(k in state for k in self._return_keys):
                termination_reason = "early_exit"
                break

            ready = next_ready

        if not self._return_keys:
            if self._result_mode == self.SCALAR:
                payload: Any = None
            elif self._result_mode == self.LIST:
                payload = []
            elif self._result_mode == self.TUPLE:
                payload = ()
            elif self._result_mode == self.SET:
                payload = set()
            else:  # DICT
                payload = {}
        else:
            missing = [k for k in self._return_keys if k not in state]
            if missing:
                raise ValidationError(
                    f"{self.full_name}: requested key(s) {missing!r} were never populated"
                )
            if self._result_mode == self.SCALAR:
                payload = state[self._return_keys[0]]
            elif self._result_mode == self.LIST:
                payload = [state[k] for k in self._return_keys]
            elif self._result_mode == self.TUPLE:
                payload = tuple(state[k] for k in self._return_keys)
            elif self._result_mode == self.SET:
                payload = {state[k] for k in self._return_keys}
            else:  # DICT
                payload = {k: state[k] for k in self._return_keys}

        return payload, {
            "trace": tuple(trace_entries) if self.include_trace else None,
            "edge_traversals": tuple(edge_traversals),
            "termination_reason": termination_reason,
        }

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the fixed graph topology plus current mutable configuration."""
        data = super().to_dict()

        edges_data: list[tuple[str, Any]] = []
        for node_name, entry in self._node_data.items():
            for target in entry.outgoing:
                edges_data.append((node_name, target))
            for router in entry.routers:
                edges_data.append((node_name, router.to_dict()))

        data.update(
            {
                "nodes": {name: entry.invokable.to_dict() for name, entry in self._node_data.items()},
                "edges": edges_data,
                "start": self._start,
                "state_policies": [
                    {"key": k, "raise_on_collision": v.raise_on_collision, "tiebreak": v.tiebreak}
                    for k, v in sorted(self._state_policies.items())
                ],
                "priorities": {name: entry.priority for name, entry in self._node_data.items()},
                "max_edge_traversals": self._max_edge_traversals,
                "stop_early": self._stop_early,
                "result_mode": self._result_mode,
                "return_keys": list(self._return_keys),
            }
        )
        return data
