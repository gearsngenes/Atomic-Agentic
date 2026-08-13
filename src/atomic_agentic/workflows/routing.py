from __future__ import annotations

import logging
import warnings
from dataclasses import replace
from collections.abc import Hashable, Iterable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any

from ..exceptions import ValidationError, WorkflowError
from ..core.Invokable import AtomicInvokable
from ..constants.core import NO_VAL
from ..models.parameters import ParamSpec
from ..utils.parameters import (
    build_parameter_reports,
    n_way_kind_compatible,
    insert_by_category,
)
from ..models.results.workflows import RoutingFlowResult
from .base import Workflow

logger = logging.getLogger(__name__)

__all__ = ["RoutingFlow"]


def _group_by_equality(values: Iterable[Any]) -> list[Any]:
    """Partition an iterable into equality groups, preserving first-seen order.

    Used for grouping declared ``default`` values across branches -- a
    plain ``set()`` isn't safe here since a default may be unhashable
    (e.g. a list).
    """
    groups: list[Any] = []
    for value in values:
        if not any(value == existing for existing in groups):
            groups.append(value)
    return groups


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
      derived from ``router`` + ``branches``, per-name:
        - A name the router declares is reconciled against each declaring
          branch independently (a 2-source witness check per branch,
          router vs. that branch); the final type is the union of every
          declaring branch's own passing witness -- not one shared check
          across all branches at once, since only one branch ever runs.
          An incompatible branch raises ``WorkflowError`` immediately,
          naming that branch. ``kind``/``default``/``description`` are
          always the router's own, unconditionally.
        - A name the router doesn't declare is never rejected -- branches
          may always introduce parameters the router doesn't have. It's
          kind-gated across its declaring branches (raise on conflict --
          the only remaining raise condition for these names), with
          ``description`` from the first-declaring branch. A
          ``VAR_POSITIONAL``/``VAR_KEYWORD`` name keeps ``default=NO_VAL``
          (required -- a default there would raise downstream) with a
          plain type union across declaring branches. A scalar name's type
          is that same union plus ``"None"``, with ``default=None`` --
          unless every declaring branch shares the exact same real
          (non-``NO_VAL``) default, in which case that shared value is used
          instead and ``"None"`` is not added, since there's no fallback
          gap to cover.
      Every construction call emits at most one grouped ``UserWarning``
      naming every router-shared name that reconciled but wasn't identical
      to the router's own declaration (not one warning per name or per
      branch).
    - ``return_type`` is the shared branch return type if all branches agree,
      otherwise a ``" | "``-joined union of each unique branch return type,
      in branch order.
    - ``include_trace`` controls whether invocation results populate
      ``trace`` with the router's and selected branch's own ``AtomicResult``
      objects. Defaults to ``True``; mutable post-construction via the
      inherited ``include_trace`` property.
    - Branch and router topology are fixed at construction. No mutation
      API is provided.

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

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        branches: tuple[AtomicInvokable, ...] | list[AtomicInvokable] | dict[Hashable, AtomicInvokable],
        router: AtomicInvokable,
        *,
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

        branch_values = (
            normalized_branches.values()
            if isinstance(normalized_branches, dict)
            else normalized_branches
        )

        # 3. Outer schema is fully derived -- no manual override.
        declared_parameters = self._reconcile_parameters(normalized_router, branch_values)

        # 4. return_type reflects every branch that could be selected at runtime:
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
    ) -> list[ParamSpec]:
        """Derive the outer parameter schema from the router and branches.

        Every router-declared name is reconciled against each declaring
        branch independently via a 2-source ``build_parameter_reports``
        call (router always source 0, so it's the ``winner_source`` --
        and therefore supplies kind/default/description -- whenever it
        declares the name). A branch's own passing witness is unioned into
        that name's running type across all branches; an empty witness or
        incompatible kind raises immediately, naming the offending branch.

        A name the router doesn't declare is never rejected -- branches may
        always introduce parameters the router doesn't have. Every such
        name is kind-gated across its declaring branches (raise on
        conflict) and takes its ``description`` from the first-declaring
        branch. A ``VAR_POSITIONAL``/``VAR_KEYWORD`` name keeps
        ``default=NO_VAL`` (structurally required) with a plain flat type
        union. A scalar name uses the flat type union plus ``"None"``, with
        ``default=None`` -- unless every declaring branch shares the exact
        same non-``NO_VAL`` default, in which case that shared value is
        used as-is and ``"None"`` is not appended. A lone ``NO_VAL`` never
        counts as an agreed default on its own, so a name only one branch
        declares, with no default of its own, still resolves to
        ``default=None``.
        """
        router_params = list(router.parameters)
        router_by_name = {p.name: p for p in router_params}

        resolved: dict[str, ParamSpec] = dict(router_by_name)
        overlapped: list[str] = []
        non_router_declarations: dict[str, list[tuple[AtomicInvokable, ParamSpec]]] = {}

        for branch in branches:
            reports = build_parameter_reports([router_params, list(branch.parameters)])

            for report in reports:
                router_slot, branch_slot = report.observations

                if router_slot is None:
                    non_router_declarations.setdefault(report.parameter_name, []).append(
                        (branch, branch_slot)
                    )
                    continue

                if branch_slot is None:
                    continue

                if not report.witness_types or not report.kind_compatible:
                    raise WorkflowError(
                        f"RoutingFlow branch {branch.full_name} is incompatible "
                        f"with router {router.full_name} on parameter "
                        f"{report.parameter_name!r}."
                    )

                current = resolved[report.parameter_name]
                resolved[report.parameter_name] = replace(
                    current,
                    type=tuple(sorted(set(current.type) | report.witness_types)),
                )
                if not report.is_identical and report.parameter_name not in overlapped:
                    overlapped.append(report.parameter_name)

        if overlapped:
            warnings.warn(
                f"Parameter(s) {overlapped!r} are compatible with the router's "
                "declaration but not identical across branches; the router's "
                "declaration wins kind/default/description (type is reconciled "
                "to the full compatible witness set).",
                UserWarning,
                stacklevel=3,
            )

        variadic_kinds = {ParamSpec.VAR_POSITIONAL, ParamSpec.VAR_KEYWORD}

        for name, declaring in non_router_declarations.items():
            if not n_way_kind_compatible(spec.kind for _, spec in declaring):
                raise WorkflowError(
                    f"RoutingFlow parameter {name!r} has an incompatible kind "
                    f"across branches {[b.full_name for b, _ in declaring]!r} "
                    "that don't share it with the router."
                )

            _, first_spec = declaring[0]
            union_type = tuple(sorted({t for _, spec in declaring for t in spec.type}))

            if first_spec.kind in variadic_kinds:
                final_type, final_default = union_type, NO_VAL
            else:
                default_groups = _group_by_equality(spec.default for _, spec in declaring)
                if len(default_groups) == 1 and default_groups[0] is not NO_VAL:
                    final_type, final_default = union_type, default_groups[0]
                else:
                    final_type = tuple(sorted(set(union_type) | {"None"}))
                    final_default = None

            resolved[name] = ParamSpec(
                name=name,
                index=0,
                kind=first_spec.kind,
                type=final_type,
                default=final_default,
                description=first_spec.description,
            )

        router_ordered = [resolved[p.name] for p in router_params]
        new_names = [resolved[name] for name in resolved if name not in router_by_name]
        return insert_by_category(router_ordered, new_names)

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
        """Serialize the fixed router and branch topology."""
        data = super().to_dict()
        if isinstance(self._branches, dict):
            branches_data: Any = {key: branch.to_dict() for key, branch in self._branches.items()}
        else:
            branches_data = [branch.to_dict() for branch in self._branches]
        data.update(
            {
                "router": self._router.to_dict(),
                "branches": branches_data,
            }
        )
        return data
