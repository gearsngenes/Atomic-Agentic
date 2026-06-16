from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Any, Optional

from ..core.Exceptions import ValidationError
from ..core.Invokable import AtomicInvokable
from ..models.parameters import ParamSpec
from ..utils.parameters import _validate_parameter_order, to_paramspec_list
from ..models.results.workflows import RoutingFlowResult
from .base import Workflow
from .basic import BasicFlow

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
        - a non-empty ``list[Workflow | AtomicInvokable]``, normalized to
          ``tuple[Workflow, ...]``. Selectors must be ``int`` indices.
        - a non-empty ``dict[Hashable, Workflow | AtomicInvokable]``,
          normalized to ``dict[Hashable, Workflow]``. Selectors must be keys
          present in this mapping.
    - ``router`` may be a ``Workflow`` (kept as-is) or any other
      ``AtomicInvokable`` (wrapped once in ``BasicFlow``). The normalized
      router is exposed read-only via :attr:`router`.
    - ``parameters`` defaults to the normalized router's parameters (the
      router is the first node invoked, so its contract is the natural
      ground truth for the outer workflow). An explicit override may be
      provided when branches require inputs beyond the router's contract.
    - ``filter_extraneous_inputs`` defaults to ``True``.
    - ``return_type`` is the shared branch return type if all branches agree,
      otherwise a ``" | "``-joined union of each unique branch return type,
      in branch order.
    - Branch and router topology are both fixed at construction. No mutation
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
        description: str,
        branches: tuple[Workflow | AtomicInvokable] | list[Workflow | AtomicInvokable] | dict[Hashable, Workflow | AtomicInvokable],
        router: Workflow | AtomicInvokable,
        *,
        parameters: type | list[str] | tuple[str, ...] | set[str] | list[ParamSpec] | None = None,
        filter_extraneous_inputs: Optional[bool] = None,
    ) -> None:
        # Topology is fixed at construction: list/tuple branches are selected
        # by int index, dict branches are selected by key. Both are
        # normalized the same way as branches elsewhere (Workflow kept as-is,
        # AtomicInvokable wrapped in BasicFlow).
        normalized_branches: tuple[Workflow, ...] | dict[Hashable, Workflow]
        if isinstance(branches, dict):
            if not branches:
                raise ValueError("branches must not be empty")
            normalized_branches = {
                key: self._normalize_branch(branch) for key, branch in branches.items()
            }
        elif isinstance(branches, (list, tuple)):
            if not branches:
                raise ValueError("branches must not be empty")
            normalized_branches = tuple(self._normalize_branch(branch) for branch in branches)
        else:
            raise TypeError(
                "branches must be a non-empty list, tuple, or dict of [Workflow | AtomicInvokable] nodes, "
                f"got {type(branches)!r}"
            )

        # The router is the first node invoked, so its contract is the
        # natural default for the outer workflow's inputs.
        normalized_router = self._normalize_branch(router)

        if parameters is None:
            declared_parameters = list(normalized_router.parameters)
        else:
            declared_parameters = to_paramspec_list(parameters)
            _validate_parameter_order(declared_parameters)

        resolved_filter = (
            filter_extraneous_inputs if filter_extraneous_inputs is not None else True
        )

        # return_type reflects every branch that could be selected at runtime:
        # the shared type if all branches agree, otherwise a "|"-joined union
        # of each unique branch return type, in branch order.
        branch_values = (
            normalized_branches.values()
            if isinstance(normalized_branches, dict)
            else normalized_branches
        )
        unique_return_types = list(dict.fromkeys(branch.return_type for branch in branch_values))
        resolved_return_type = (
            unique_return_types[0]
            if len(unique_return_types) == 1
            else " | ".join(unique_return_types)
        )

        super().__init__(
            name=name,
            description=description,
            parameters=list(declared_parameters),
            return_type=resolved_return_type,
            filter_extraneous_inputs=resolved_filter,
        )

        self._branches: tuple[Workflow, ...] | dict[Hashable, Workflow] = normalized_branches
        self._router: Workflow = normalized_router

    # ------------------------------------------------------------------ #
    # Read-only properties
    # ------------------------------------------------------------------ #
    @property
    def branches(self) -> tuple[Workflow, ...] | MappingProxyType:
        """Return the fixed normalized branch topology.

        ``tuple[Workflow, ...]`` if configured from a list, or a read-only
        ``MappingProxyType[Hashable, Workflow]`` if configured from a dict.
        """
        if isinstance(self._branches, dict):
            return MappingProxyType(self._branches)
        return self._branches

    @property
    def router(self) -> Workflow:
        """Return the fixed normalized router."""
        return self._router

    # ------------------------------------------------------------------ #
    # Public retrieval helper
    # ------------------------------------------------------------------ #
    def get_router_decision(self, run_id: str) -> Optional[Any]:
        """Return the validated router selector for one parent routing run.

        Returns
        -------
        Optional[Any]
            - ``None`` if the parent checkpoint is not found, or if the
              router's own checkpoint is no longer retained
            - otherwise, the router's unwrapped result (the selector used to
              choose the executed branch)
        """
        checkpoint = self.get_checkpoint(run_id)
        if checkpoint is None:
            return None

        # The selector itself isn't stored on RoutingFlowResult - it's
        # recovered by following router_run_id back to the router's own
        # checkpoint, whose result IS the validated selector.
        router_checkpoint = self._router.get_checkpoint(checkpoint.result.router_run_id)
        return router_checkpoint.result.result if router_checkpoint else None

    # ------------------------------------------------------------------ #
    # Internal normalization / validation helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_branch(branch: Workflow | AtomicInvokable) -> Workflow:
        """Normalize one configured branch or router into a workflow-shaped node."""
        if isinstance(branch, Workflow):
            return branch
        if isinstance(branch, AtomicInvokable):
            return BasicFlow(component=branch)
        raise TypeError(
            "RoutingFlow branches/router must be Workflow or AtomicInvokable, "
            f"got {type(branch)!r}"
        )

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
        """Construct a RoutingFlowResult envelope for this workflow's invocation."""
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
        selected_key = self._extract_selector(router_result.result)
        selected_branch = self._branches[selected_key]

        # 3. Selected branch runs on the SAME outer inputs (not the router's
        #    result - the router result is selection-only, not a handoff payload).
        logger.info(
            "%s: router selected branch %r (%s)",
            self.full_name,
            selected_key,
            selected_branch.full_name,
        )
        selected_result = selected_branch.invoke(inputs)

        return selected_result.result, {
            "selected_key": selected_key,
            "chosen_branch_run": selected_result.run_id,
            "router_run_id": router_result.run_id,
        }

    async def _async_run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Asynchronously invoke the router, then exactly one chosen branch."""
        logger.info("[Async %s]: invoking router (%s)", self.full_name, self._router.full_name)
        router_result = await self._router.async_invoke(inputs)

        selected_key = self._extract_selector(router_result.result)
        selected_branch = self._branches[selected_key]

        logger.info(
            "[Async %s]: router selected branch %r (%s)",
            self.full_name,
            selected_key,
            selected_branch.full_name,
        )
        selected_result = await selected_branch.async_invoke(inputs)

        return selected_result.result, {
            "selected_key": selected_key,
            "chosen_branch_run": selected_result.run_id,
            "router_run_id": router_result.run_id,
        }

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the fixed router plus fixed branch topology."""
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
