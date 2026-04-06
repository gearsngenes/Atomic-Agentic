from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from ..core.Exceptions import ValidationError
from ..core.Invokable import AtomicInvokable
from ..core.sentinels import NO_VAL
from .StructuredInvokable import StructuredInvokable
from .base import FlowResultDict, Workflow
from .basic import BasicFlow

logger = logging.getLogger(__name__)

__all__ = ["RoutingFlow"]


class RoutingFlow(Workflow):
    """Route one invocation to exactly one fixed branch based on a router result.

    Overview
    --------
    ``RoutingFlow`` is a composite workflow that first invokes a fixed normalized
    router path on the outer workflow inputs. The router's *raw* result is then
    interpreted as a strict zero-based branch index selecting one configured
    branch to invoke.

    Construction contract
    ---------------------
    - ``branches`` must be a non-empty ``list[Workflow | StructuredInvokable]``.
    - Branch topology is fixed at construction and exposed read-only via
      :attr:`branches`.
    - ``router`` is required and must be an ``AtomicInvokable``.
    - The router is normalized once at construction and exposed read-only via
      :attr:`router`.
    - The outer workflow input contract mirrors the normalized router path.
    - No router or branch mutation API is provided.

    Routing contract
    ----------------
    At runtime, the normalized router path must yield a raw result discoverable
    through its checkpoint metadata under ``"run_raw_result"``. That raw result
    must satisfy all of the following:

    - be present,
    - be an ``int``,
    - not be a ``bool``,
    - fall within ``[0, len(branches))``.

    Otherwise invocation raises ``ValidationError``.

    Branch invocation semantics
    ---------------------------
    - The router receives the filtered outer workflow inputs.
    - The selected branch also receives the same filtered outer workflow inputs.
    - The router result is used only for selection, not as a handoff payload.
    - Exactly one branch is invoked per run.

    Metadata
    --------
    Per-run metadata contains:

    - ``router_run_id``:
      Child run id of the normalized router invocation.
    - ``selected_index``:
      The validated zero-based selected branch index.
    - ``selected_branch_instance_id``:
      Instance id of the selected branch.
    - ``selected_branch_full_name``:
      Full name of the selected branch.
    - ``selected_branch_run_id``:
      Child run id of the selected branch invocation.
    """

    def __init__(
        self,
        name: str,
        description: str,
        branches: list[Workflow | StructuredInvokable],
        router: AtomicInvokable,
        *,
        filter_extraneous_inputs: Optional[bool] = None,
    ) -> None:
        if not isinstance(branches, list):
            raise TypeError(
                f"branches must be a non-empty list[Workflow | StructuredInvokable], got {type(branches)!r}"
            )
        if not branches:
            raise ValueError("branches must not be empty")

        normalized_branches = tuple(self._normalize_branch(branch) for branch in branches)
        normalized_router = self._normalize_router(router)

        resolved_filter = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else normalized_router.filter_extraneous_inputs
        )

        super().__init__(
            name=name,
            description=description,
            parameters=normalized_router.parameters,
            filter_extraneous_inputs=resolved_filter,
        )

        self._branches: tuple[Workflow, ...] = normalized_branches
        self._router: Workflow = normalized_router

    # ------------------------------------------------------------------ #
    # Read-only properties
    # ------------------------------------------------------------------ #
    @property
    def branches(self) -> tuple[Workflow, ...]:
        """Return the fixed normalized branch tuple."""
        return self._branches

    @property
    def router(self) -> Workflow:
        """Return the fixed normalized router workflow."""
        return self._router

    # ------------------------------------------------------------------ #
    # Public retrieval helper
    # ------------------------------------------------------------------ #
    def get_router_decision(self, run_id: str) -> Optional[int]:
        """Return the selected branch index for one parent routing run.

        Parameters
        ----------
        run_id:
            Parent ``RoutingFlow`` run id.

        Returns
        -------
        Optional[int]
            - ``None`` if the parent checkpoint is not found
            - the validated selected branch index otherwise

        Raises
        ------
        ValidationError
            If the stored checkpoint metadata does not contain a valid
            ``selected_index`` entry.
        """
        checkpoint = self.get_checkpoint(run_id)
        if checkpoint is None:
            return None

        selected_index = checkpoint.metadata.get("selected_index", NO_VAL)
        if selected_index is NO_VAL:
            raise ValidationError(
                f"{self.full_name}: checkpoint metadata missing 'selected_index' for run_id {run_id!r}"
            )
        if not isinstance(selected_index, int) or isinstance(selected_index, bool):
            raise ValidationError(
                f"{self.full_name}: checkpoint metadata 'selected_index' must be a non-bool int, "
                f"got {type(selected_index)!r}"
            )

        return selected_index

    # ------------------------------------------------------------------ #
    # Internal normalization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_branch(branch: Workflow | StructuredInvokable) -> Workflow:
        """Normalize one configured branch into a workflow-shaped node."""
        if isinstance(branch, Workflow):
            return branch
        if isinstance(branch, StructuredInvokable):
            return BasicFlow(component=branch)
        raise TypeError(
            "RoutingFlow branches must be Workflow or StructuredInvokable, "
            f"got {type(branch)!r}"
        )

    @staticmethod
    def _normalize_router(router: AtomicInvokable) -> Workflow:
        """Normalize a router candidate into a fixed workflow-shaped node."""
        if router is None:
            raise TypeError("router must be an AtomicInvokable, got None")

        if not isinstance(router, AtomicInvokable):
            raise TypeError(
                f"router must be an AtomicInvokable, got {type(router)!r}"
            )

        if isinstance(router, Workflow):
            return router

        if isinstance(router, StructuredInvokable):
            return BasicFlow(component=router)

        return BasicFlow(component=StructuredInvokable(component=router, output_schema=[]))

    # ------------------------------------------------------------------ #
    # Internal validation / metadata helpers
    # ------------------------------------------------------------------ #
    def _extract_selected_index(
        self,
        router_flow: Workflow,
        router_result: FlowResultDict,
    ) -> int:
        """Extract and validate the raw routing decision for one router run."""
        checkpoint = router_flow.get_checkpoint(router_result.run_id)
        if checkpoint is None:
            raise ValidationError(
                f"{self.full_name}: router checkpoint not found for run_id {router_result.run_id!r}"
            )

        raw_decision = checkpoint.metadata.get("run_raw_result", NO_VAL)
        if raw_decision is NO_VAL:
            raise ValidationError(
                f"{self.full_name}: router metadata did not contain 'run_raw_result'"
            )

        if isinstance(raw_decision, bool) or not isinstance(raw_decision, int):
            raise ValidationError(
                f"{self.full_name}: router raw result must be a non-bool int, got {type(raw_decision)!r}"
            )

        if raw_decision < 0 or raw_decision >= len(self._branches):
            raise ValidationError(
                f"{self.full_name}: router selected index {raw_decision} out of range "
                f"for {len(self._branches)} configured branch(es)"
            )

        return raw_decision

    @staticmethod
    def _build_metadata(
        *,
        router_run_id: str,
        selected_index: int,
        selected_branch: Workflow,
        selected_branch_run_id: str,
    ) -> dict[str, Any]:
        """Build per-run outer checkpoint metadata."""
        return {
            "router_run_id": router_run_id,
            "selected_index": selected_index,
            "selected_branch_instance_id": selected_branch.instance_id,
            "selected_branch_full_name": selected_branch.full_name,
            "selected_branch_run_id": selected_branch_run_id,
        }

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Synchronously invoke the router, then exactly one selected branch."""
        logger.info("%s: invoking router (%s)", self.full_name, self._router.full_name)
        router_result = self._router.invoke(inputs)

        if not isinstance(router_result, FlowResultDict):
            raise ValidationError(
                f"{self.full_name}: router returned {type(router_result)!r}, expected FlowResultDict"
            )

        selected_index = self._extract_selected_index(self._router, router_result)
        selected_branch = self._branches[selected_index]

        logger.info(
            "%s: router selected branch %d (%s)",
            self.full_name,
            selected_index,
            selected_branch.full_name,
        )
        selected_result = selected_branch.invoke(inputs)

        if not isinstance(selected_result, FlowResultDict):
            raise ValidationError(
                f"{self.full_name}: selected branch {selected_index} ({selected_branch.full_name}) "
                f"returned {type(selected_result)!r}, expected FlowResultDict"
            )

        metadata = self._build_metadata(
            router_run_id=router_result.run_id,
            selected_index=selected_index,
            selected_branch=selected_branch,
            selected_branch_run_id=selected_result.run_id,
        )
        return metadata, selected_result

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Asynchronously invoke the router, then exactly one selected branch."""
        logger.info(
            "[Async %s]: invoking router (%s)",
            self.full_name,
            self._router.full_name,
        )
        router_result = await self._router.async_invoke(inputs)

        if not isinstance(router_result, FlowResultDict):
            raise ValidationError(
                f"{self.full_name}: async router returned {type(router_result)!r}, expected FlowResultDict"
            )

        selected_index = self._extract_selected_index(self._router, router_result)
        selected_branch = self._branches[selected_index]

        logger.info(
            "[Async %s]: router selected branch %d (%s)",
            self.full_name,
            selected_index,
            selected_branch.full_name,
        )
        selected_result = await selected_branch.async_invoke(inputs)

        if not isinstance(selected_result, FlowResultDict):
            raise ValidationError(
                f"{self.full_name}: async selected branch {selected_index} ({selected_branch.full_name}) "
                f"returned {type(selected_result)!r}, expected FlowResultDict"
            )

        metadata = self._build_metadata(
            router_run_id=router_result.run_id,
            selected_index=selected_index,
            selected_branch=selected_branch,
            selected_branch_run_id=selected_result.run_id,
        )
        return metadata, selected_result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the fixed router plus fixed branch topology."""
        data = super().to_dict()
        data.update(
            {
                "router": self._router.to_dict(),
                "branches": [branch.to_dict() for branch in self._branches],
            }
        )
        return data