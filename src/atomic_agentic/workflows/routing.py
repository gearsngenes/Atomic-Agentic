from __future__ import annotations

import logging
import warnings
from dataclasses import replace
from collections.abc import Hashable, Iterable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any, ClassVar

from ..exceptions import ValidationError, WorkflowError
from ..core.Invokable import AtomicInvokable
from ..constants.core import NO_VAL
from ..models.parameters import ParamSpec
from ..utils.parameters import (
    parameter_collisions,
    parameter_overlap,
    variadic_compatible,
    semantically_identical,
    insert_by_category,
)
from ..models.results.workflows import RoutingFlowResult
from .base import Workflow

logger = logging.getLogger(__name__)

__all__ = ["RoutingFlow"]


class RoutingFlow(Workflow):
    """Route one invocation to exactly one fixed branch based on a router result.

    Overview
    --------
    ``RoutingFlow`` first invokes a fixed router on the outer workflow inputs.
    The router's unwrapped result (``router_result.result``) is the
    "selector": a value used to pick exactly one configured branch, which is
    then invoked with the same outer inputs.

    Construction contract
    ----------------------
    - ``branches`` is either:
        - a non-empty ``list[AtomicInvokable]``, stored as
          ``tuple[AtomicInvokable, ...]``. Selectors must be ``int`` indices.
        - a non-empty ``dict[Hashable, AtomicInvokable]``, stored as
          configured. Selectors must be keys present in this mapping.
      Any invokable, ``Workflow`` or not, is stored exactly as configured
      with no wrapping.
    - ``router`` may be any ``AtomicInvokable``, stored as configured with
      no wrapping. The normalized router is exposed read-only via
      :attr:`router`.
    - ``parameters`` is not a constructor option. The outer schema is always
      derived from ``router`` + ``branches`` per ``schema_mode``:
        - ``STRICT`` (default) -- outer schema equals the router's own
          parameters, verbatim. Every branch's parameter names must already
          exist in the router's schema (raises ``WorkflowError`` otherwise);
          overlapping names are checked for collisions/identity, but nothing
          new is ever introduced.
        - ``PARTIAL`` -- a router-seeded union-fold across router + all
          branches (same reconciliation primitives ``ParallelFlow`` uses),
          gated by requiring each branch to share at least one non-colliding
          parameter name with the router (raises ``WorkflowError``
          otherwise).
        - ``OPEN`` -- identical fold to ``PARTIAL``, without the overlap
          requirement -- a branch may share nothing with the router.
      For ``PARTIAL``/``OPEN``, a parameter a branch introduces that the
      router doesn't already declare becomes optional (``default=None``)
      when it doesn't already carry a default of its own.
    - ``return_type`` is the shared branch return type if all branches agree,
      otherwise a ``" | "``-joined union of each unique branch return type,
      in branch order.
    - ``include_trace`` controls whether invocation results populate
      ``trace`` with the router's and selected branch's own ``AtomicResult``
      objects. Defaults to ``True``; mutable post-construction via the
      inherited ``include_trace`` property.
    - Branch, router, and schema-mode topology are all fixed at
      construction. No mutation API is provided.

    Routing contract
    -----------------
    At runtime, the router's unwrapped result (the "selector") must be:

    - not a ``bool``,
    - for list-configured branches: an ``int`` in ``[0, len(branches))``,
    - for dict-configured branches: a key present in ``branches``.

    Otherwise invocation raises ``ValidationError``.

    Branch invocation semantics
    ----------------------------
    - The router receives the filtered outer workflow inputs.
    - The selected branch also receives the same filtered outer workflow inputs.
    - The router result is used only for selection, not as a handoff payload.
    - Exactly one branch is invoked per run.
    """

    OPEN: ClassVar[str] = "open"
    PARTIAL: ClassVar[str] = "partial"
    STRICT: ClassVar[str] = "strict"
    _SCHEMA_MODES: ClassVar[frozenset[str]] = frozenset({OPEN, PARTIAL, STRICT})

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        branches: tuple[AtomicInvokable, ...] | list[AtomicInvokable] | dict[Hashable, AtomicInvokable],
        router: AtomicInvokable,
        *,
        schema_mode: str = STRICT,
        include_trace: bool = True,
    ) -> None:
        # 1. Topology is fixed at construction: list/tuple branches are selected
        #    by int index, dict branches are selected by key. Every branch is
        #    validated as an AtomicInvokable and stored as configured, no
        #    wrapping.
        normalized_branches: tuple[AtomicInvokable, ...] | dict[Hashable, AtomicInvokable]
        if isinstance(branches, dict):
            if not branches:
                raise ValueError("branches must not be empty")
            if any(not isinstance(branch, AtomicInvokable) for branch in branches.values()):
                raise TypeError("RoutingFlow branches must be AtomicInvokable")
            normalized_branches = dict(branches)
        elif isinstance(branches, (list, tuple)):
            if not branches:
                raise ValueError("branches must not be empty")
            if any(not isinstance(branch, AtomicInvokable) for branch in branches):
                raise TypeError("RoutingFlow branches must be AtomicInvokable")
            normalized_branches = tuple(branches)
        else:
            raise TypeError(
                "branches must be a non-empty list, tuple, or dict of AtomicInvokable nodes, "
                f"got {type(branches)!r}"
            )

        # 2. The router is the first node invoked; its own contract anchors
        #    the outer parameter schema (see _reconcile_parameters).
        if not isinstance(router, AtomicInvokable):
            raise TypeError(f"RoutingFlow router must be AtomicInvokable, got {type(router)!r}")
        normalized_router = router

        # 3. schema_mode picks how far branches may diverge from the
        #    router's own declared parameters -- resolved once, immutable
        #    thereafter.
        if schema_mode not in self._SCHEMA_MODES:
            raise ValueError(
                f"schema_mode must be one of {sorted(self._SCHEMA_MODES)!r}, got {schema_mode!r}"
            )

        branch_values = (
            normalized_branches.values()
            if isinstance(normalized_branches, dict)
            else normalized_branches
        )

        # 4. Outer schema is fully derived -- no manual override.
        declared_parameters = self._reconcile_parameters(normalized_router, branch_values, schema_mode)

        # 5. return_type reflects every branch that could be selected at runtime:
        #    the shared type if all branches agree, otherwise a "|"-joined union
        #    of each unique branch return type, in branch order.
        unique_return_types = list(dict.fromkeys(branch.return_type for branch in branch_values))
        resolved_return_type = (
            unique_return_types[0]
            if len(unique_return_types) == 1
            else " | ".join(unique_return_types)
        )

        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            parameters=declared_parameters,
            return_type=resolved_return_type,
            include_trace=include_trace,
        )

        self._branches: tuple[AtomicInvokable, ...] | dict[Hashable, AtomicInvokable] = normalized_branches
        self._router: AtomicInvokable = normalized_router
        self._schema_mode: str = schema_mode

    # ------------------------------------------------------------------ #
    # Read-only properties
    # ------------------------------------------------------------------ #
    @property
    def branches(self) -> tuple[AtomicInvokable, ...] | MappingProxyType:
        """Return the fixed normalized branch topology.

        ``tuple[AtomicInvokable, ...]`` if configured from a list, or a
        read-only ``MappingProxyType[Hashable, AtomicInvokable]`` if
        configured from a dict.
        """
        if isinstance(self._branches, dict):
            return MappingProxyType(self._branches)
        return self._branches

    @property
    def router(self) -> AtomicInvokable:
        """Return the fixed normalized router."""
        return self._router

    @property
    def schema_mode(self) -> str:
        """Return the fixed parameter-reconciliation mode: OPEN, PARTIAL, or STRICT."""
        return self._schema_mode

    def _extra_description(self) -> str:
        """State the fixed branch count; inline shared branch content when unanimous.

        Since only one branch executes per run, a labeled per-branch dump
        would misrepresent which content applies to any given invocation. A
        branch's own extra description is appended only when every
        configured branch agrees exactly and the shared value is non-empty.
        """
        branch_values = (
            list(self._branches.values())
            if isinstance(self._branches, dict)
            else list(self._branches)
        )
        extras = [b._extra_description() for b in branch_values]
        base = f"Selects 1 of {len(branch_values)} branches at runtime."

        if all(e == extras[0] for e in extras) and extras[0]:
            return f"{base}\n{extras[0]}"
        return base

    # ------------------------------------------------------------------ #
    # Internal reconciliation helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reconcile_parameters(
        router: AtomicInvokable,
        branches: Iterable[AtomicInvokable],
        schema_mode: str,
    ) -> list[ParamSpec]:
        """Derive the outer parameter schema from the router and branches.

        STRICT: the router's own parameters are the fixed ceiling. Each
        branch's parameter names must already exist in that schema (raise
        otherwise); overlapping names are checked for collisions, then for
        exact identity (warn if merely compatible). Nothing is ever added.

        PARTIAL / OPEN: a router-seeded left-fold across every branch, same
        primitives/order ParallelFlow._reconcile_branch_parameters uses.
        PARTIAL additionally requires each branch to share at least one
        non-colliding parameter with the router before folding it in.
        Parameters a branch introduces that the router didn't already have
        become optional (default=None) if they lack their own default --
        not every branch runs on a given invocation.
        """
        combined: list[ParamSpec] = list(router.parameters)
        router_names = {p.name for p in combined}

        if schema_mode == RoutingFlow.STRICT:
            for branch in branches:
                branch_params = list(branch.parameters)

                unknown = {p.name for p in branch_params} - router_names
                if unknown:
                    raise WorkflowError(
                        f"RoutingFlow branch {branch.full_name} declares parameter(s) "
                        f"{sorted(unknown)!r} not present in the router's "
                        f"({router.full_name}) schema; schema_mode='strict' requires "
                        "every branch to be a subset of the router's parameters."
                    )

                collisions = parameter_collisions(combined, branch_params)
                if collisions:
                    raise WorkflowError(
                        f"RoutingFlow branch {branch.full_name} parameter(s) "
                        f"{collisions!r} conflict with the router's declaration "
                        "(same name, incompatible type/kind)."
                    )

                overlap = parameter_overlap(combined, branch_params)
                combined_by_name = {p.name: p for p in combined}
                branch_by_name = {p.name: p for p in branch_params}
                for shared_name in overlap:
                    if not semantically_identical(combined_by_name[shared_name], branch_by_name[shared_name]):
                        warnings.warn(
                            f"Parameter {shared_name!r} in branch {branch.full_name} is "
                            "compatible with the router's declaration but not identical; "
                            "the router's declaration wins.",
                            UserWarning,
                            stacklevel=3,
                        )

            return combined

        # PARTIAL / OPEN share the same union-fold.
        for branch in branches:
            branch_params = list(branch.parameters)

            if schema_mode == RoutingFlow.PARTIAL:
                if not parameter_overlap(combined, branch_params):
                    raise WorkflowError(
                        f"RoutingFlow branch {branch.full_name} shares no parameters "
                        f"with the router ({router.full_name}); schema_mode='partial' "
                        "requires at least one overlapping parameter."
                    )

            collisions = parameter_collisions(combined, branch_params)
            if collisions:
                raise WorkflowError(
                    f"RoutingFlow branch {branch.full_name} parameter(s) "
                    f"{collisions!r} conflict with an earlier declaration "
                    "(same name, incompatible type/kind)."
                )

            overlap = parameter_overlap(combined, branch_params)
            if not variadic_compatible(combined, branch_params, set(overlap)):
                raise WorkflowError(
                    f"RoutingFlow branch {branch.full_name} declares an independent "
                    "variadic parameter of the same kind as an earlier declaration "
                    "under a different name; rename one."
                )

            combined_by_name = {p.name: p for p in combined}
            branch_by_name = {p.name: p for p in branch_params}
            for shared_name in overlap:
                if not semantically_identical(combined_by_name[shared_name], branch_by_name[shared_name]):
                    warnings.warn(
                        f"Parameter {shared_name!r} in branch {branch.full_name} is "
                        "compatible with an earlier declaration but not identical; "
                        "the earlier declaration wins.",
                        UserWarning,
                        stacklevel=3,
                    )

            remainder = [p for p in branch_params if p.name not in overlap]
            combined = insert_by_category(combined, remainder)

        # Anything not originally from the router is optional unless it
        # already carries its own default -- not every branch runs.
        return [
            p if (p.name in router_names or p.default is not NO_VAL)
            else replace(p, default=None)
            for p in combined
        ]

    def _extract_selector(self, selector: Any) -> Hashable:
        """Validate and return the router-provided selector for the configured branches."""
        # bool is a subclass of int, and could otherwise alias a list index
        # (0/1) or collide with int-valued dict keys (True == 1) - reject
        # outright regardless of branch topology.
        if isinstance(selector, bool):
            raise ValidationError(
                f"{self.full_name}: router result must not be a bool, got {selector!r}"
            )

        if isinstance(self._branches, dict):
            # Dict topology: selector must be a hashable key present in
            # self._branches. Unhashable selectors raise TypeError on `in` -
            # surface that as a ValidationError instead.
            try:
                is_valid = selector in self._branches
            except TypeError as exc:
                raise ValidationError(
                    f"{self.full_name}: router result {selector!r} is not a valid "
                    f"(hashable) branch key"
                ) from exc
            if not is_valid:
                raise ValidationError(
                    f"{self.full_name}: router result {selector!r} is not among "
                    f"configured branch keys {tuple(self._branches.keys())!r}"
                )
            return selector

        # List/tuple topology: selector must be an in-range int index.
        if not isinstance(selector, int):
            raise ValidationError(
                f"{self.full_name}: router result must be an int, got {type(selector)!r}"
            )
        if not (0 <= selector < len(self._branches)):
            raise ValidationError(
                f"{self.full_name}: router selected index {selector} out of range "
                f"for {len(self._branches)} configured branch(es)"
            )
        return selector

    # ------------------------------------------------------------------ #
    # Result construction
    # ------------------------------------------------------------------ #
    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> RoutingFlowResult:
        """Construct a RoutingFlowResult envelope for this workflow's invocation.

        No fixed-topology constant to inject here -- selected_branch varies
        per invocation, so it flows through result_kwargs from
        _run/_async_run rather than being injected here.
        """
        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=RoutingFlowResult,
            **result_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Synchronously invoke the router, then exactly one chosen branch."""
        # 1. Router runs first, on the same inputs the outer workflow received.
        logger.info("%s: invoking router (%s)", self.full_name, self._router.full_name)
        router_result = self._router.invoke(inputs)

        # 2. Router's unwrapped result is the selector; validate + resolve it
        #    to exactly one configured branch.
        selected_branch = self._extract_selector(router_result.result)
        branch = self._branches[selected_branch]

        # 3. Selected branch runs on the SAME outer inputs (not the router's
        #    result - the router result is selection-only, not a handoff payload).
        logger.info(
            "%s: router selected branch %r (%s)",
            self.full_name,
            selected_branch,
            branch.full_name,
        )
        selected_result = branch.invoke(inputs)

        return selected_result.result, {
            "selected_branch": selected_branch,
            "trace": (router_result, selected_result) if self.include_trace else None,
        }

    async def _async_run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Asynchronously invoke the router, then exactly one chosen branch."""
        logger.info("[Async %s]: invoking router (%s)", self.full_name, self._router.full_name)
        router_result = await self._router.async_invoke(inputs)

        selected_branch = self._extract_selector(router_result.result)
        branch = self._branches[selected_branch]

        logger.info(
            "[Async %s]: router selected branch %r (%s)",
            self.full_name,
            selected_branch,
            branch.full_name,
        )
        selected_result = await branch.async_invoke(inputs)

        return selected_result.result, {
            "selected_branch": selected_branch,
            "trace": (router_result, selected_result) if self.include_trace else None,
        }

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the fixed router, branch topology, and schema_mode."""
        data = super().to_dict()
        if isinstance(self._branches, dict):
            branches_data: Any = {key: branch.to_dict() for key, branch in self._branches.items()}
        else:
            branches_data = [branch.to_dict() for branch in self._branches]
        data.update(
            {
                "router": self._router.to_dict(),
                "branches": branches_data,
                "schema_mode": self._schema_mode,
            }
        )
        return data
