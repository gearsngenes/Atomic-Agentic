from __future__ import annotations

from collections.abc import Hashable, Mapping
from types import MappingProxyType
from typing import Any, Callable, ClassVar

from ..invokables.atomic_invokable import AtomicInvokable
from ..models.parameters import ParamSpec
from ..models.results import RouterResult
from ..tools.toolification import to_invokable
from .graph import Graph

__all__ = ["Router"]


class Router(AtomicInvokable):
    """A selector plus a dict of routes, run over a private ``Graph``.

    Construction contract
    ---------------------
    - ``selector`` is any entry form (invokable, ``str``, callable); it runs
      first and its whole response is the route value. ``routes`` is a
      non-empty dict of hashable key to entry; keys need not be ``str``.
    - Neither the selector nor a route entry may be a tuple: branches are bare,
      so shape a branch's result by wrapping it in a ``Pipeline``. There is no
      default route.
    - Entries are coerced through the shared ``to_invokable``; coerced names are
      ``<router name>_select`` and ``<router name>_route_{i}`` in dict order.
      The private graph's node names (``select``, ``route_{i}``) are internal.
    - ``parameters`` default to the selector's. Pass ``parameters=[...]`` to
      accept inputs the selector does not declare (they reach the chosen branch
      through the shared state); undeclared inputs are dropped.
      ``return_type`` is the union of the branch return types.
    - Its result is a ``RouterResult``: ``result`` is the branch's raw response,
      ``route`` the selector's value and ``branch`` the chosen branch's name.
    - The router holds its graph by composition and has no mutators.

    Runtime contract
    ----------------
    - The selector's response is looked up in the routes by Python dict
      equality and hashing, so ``1``, ``1.0`` and ``True`` share one key, and a
      ``None`` key matches a ``None`` response. That is documented, not
      special-cased.
    - The chosen branch receives the shared state (the call inputs plus the
      selector's response under ``__route__``); only that branch runs, and its
      raw response (any type) is the result.
    - A selector value with no matching route raises ``ValueError`` and an
      unhashable value raises ``ValueError`` too: both surface as the graph's
      own error for an invalid router target, because an exception raised inside
      the routing closure would be wrapped by the tool layer. The message names
      the router and the offending value.
    - The reserved state keys ``__route__`` and ``__result__`` belong to the
      router; user input keys with those names are the caller's problem.
    """

    result_cls: ClassVar[type[RouterResult]] = RouterResult

    def __init__(
        self,
        name: str,
        description: str,
        selector: AtomicInvokable | str | Callable[..., Any],
        routes: dict[Hashable, AtomicInvokable | str | Callable[..., Any]],
        *,
        parameters: list[ParamSpec] | None = None,
    ) -> None:
        # 1. shape checks
        if not isinstance(routes, dict):
            raise TypeError(f"routes must be a non-empty dict, got {type(routes)!r}")
        if not routes:
            raise ValueError("routes must not be empty")
        if isinstance(selector, tuple):
            raise TypeError("selector must not be a tuple; wrap it in a Pipeline to shape its result")
        for key, entry in routes.items():
            if isinstance(entry, tuple):
                raise TypeError(
                    f"route {key!r} must not be a tuple; wrap the branch in a Pipeline "
                    "to shape its result"
                )

        # 2. coerce the selector and the branches
        selector_invokable = to_invokable(
            selector,
            name=f"{name}_select",
            description=f"Selector of router {name}",
        )
        branches = [
            to_invokable(
                entry,
                name=f"{name}_route_{i}",
                description=f"Route {i} of router {name}",
            )
            for i, entry in enumerate(routes.values())
        ]

        # 3. internal node names and the key map
        key_map = {key: f"route_{i}" for i, key in enumerate(routes)}

        # 4. routing closure; invalid values return a non-node name so the graph raises
        def decide(**state: Any) -> str:
            value = state["__route__"]
            try:
                target = key_map.get(value)
            except TypeError:
                return f"<router {name}: unhashable route value {value!r}>"
            if target is None:
                return f"<router {name}: no route for {value!r}>"
            return target

        # 5. private graph
        if parameters is None:
            parameters = selector_invokable.parameters
        nodes: dict[str, Any] = {"select": (selector_invokable, "__route__")}
        for i, branch in enumerate(branches):
            nodes[f"route_{i}"] = (branch, "__result__")
        graph = Graph(
            name,
            description,
            nodes,
            edges=[("select", decide)],
            start="select",
            parameters=parameters,
            returns=["__route__", "__result__"],
            max_traversal_depth=2,
        )

        # 6. base init
        return_type = {t for branch in branches for t in branch.return_type}
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            return_type=return_type,
        )

        # 7. store
        self._graph = graph
        self._branch_names = {f"route_{i}": branch.name for i, branch in enumerate(branches)}
        self._selector = selector_invokable
        self._routes: Mapping[Hashable, AtomicInvokable] = MappingProxyType(
            dict(zip(routes, branches))
        )

    @property
    def selector(self) -> AtomicInvokable:
        """The coerced selector invokable."""
        return self._selector

    @property
    def routes(self) -> Mapping[Hashable, AtomicInvokable]:
        """Read-only mapping of route key to coerced branch invokable."""
        return self._routes

    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run the private graph; return the branch's raw response and what was chosen."""
        return self._payload(self._graph.invoke(inputs))

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async twin of ``_invoke_impl``."""
        return self._payload(await self._graph.async_invoke(inputs))

    def _payload(self, graph_result: Any) -> dict[str, Any]:
        """Build the ``RouterResult`` payload; the trace's last node is the branch."""
        chosen = graph_result.trace[-1][0]
        return {
            "result": graph_result.result["__result__"],
            "route": graph_result.result["__route__"],
            "branch": self._branch_names[chosen],
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the router with its selector and routes in dict order."""
        data = super().to_dict()
        data.update(
            {
                "selector": self._selector.to_dict(),
                "routes": [
                    {"key": key, "invokable": invokable.to_dict()}
                    for key, invokable in self._routes.items()
                ],
            }
        )
        return data
