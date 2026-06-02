"""Workflow wrappers.

This module contains thin workflow adapters around executable nodes.

`BasicFlow` wraps any `AtomicInvokable` and exposes it as a workflow-shaped node.
Its responsibilities are limited to:

- delegating sync/async execution;
- preserving the wrapped component's input parameters;
- validating that the child result is mapping-shaped;
- emitting lightweight metadata for checkpointing.

`BasicFlow` does not reshape arbitrary outputs. Components that do not naturally
return mapping-shaped values should be wrapped in `StructuredInvokable` or a
future approved output adapter before entering a workflow.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from ..core.Exceptions import ValidationError
from ..core.Invokable import AtomicInvokable
from ..core.constants import NO_VAL
from .base import FlowResultDict, Workflow
from .metadata import BasicFlowRunMetadata

logger = logging.getLogger(__name__)

__all__ = ["BasicFlow"]


class BasicFlow(Workflow[BasicFlowRunMetadata]):
    """Thin workflow adapter for mapping-returning AtomicInvokable children.

    `BasicFlow` delegates directly to its wrapped component. The wrapped
    component may be another Workflow or any other AtomicInvokable.

    Runtime result contract:

    - Workflow children must return the workflow-owned `FlowResultDict` carrier.
      Its `run_id` is recorded as the child workflow run id.
    - Non-workflow AtomicInvokable children must return a mapping-shaped result.
      No child run id exists yet for those children, so metadata stores `NO_VAL`.

    The outer Workflow base then records this BasicFlow's checkpoint and wraps
    the final mapping in a fresh outer `FlowResultDict`.
    """

    def __init__(
        self,
        component: AtomicInvokable,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        filter_extraneous_inputs: Optional[bool] = None,
    ) -> None:
        if not isinstance(component, AtomicInvokable):
            raise TypeError(
                "BasicFlow.component must be an AtomicInvokable, "
                f"got {type(component)!r}"
            )

        self._component = component

        resolved_filter = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else component.filter_extraneous_inputs
        )

        super().__init__(
            name=name or component.name,
            description=description or component.description,
            parameters=component.parameters,
            filter_extraneous_inputs=resolved_filter,
        )

    # ------------------------------------------------------------------ #
    # BasicFlow properties
    # ------------------------------------------------------------------ #
    @property
    def component(self) -> AtomicInvokable:
        """The wrapped executable component."""
        return self._component

    # ------------------------------------------------------------------ #
    # Metadata helpers
    # ------------------------------------------------------------------ #
    def _build_metadata(self, result: Mapping[str, Any]) -> BasicFlowRunMetadata:
        """Build typed checkpoint metadata for the delegated child."""
        child_is_workflow = isinstance(self.component, Workflow)

        if child_is_workflow and not isinstance(result, FlowResultDict):
            raise ValidationError(
                f"{type(self).__name__}.{self.name}: wrapped workflow child "
                f"returned {type(result)!r}, expected FlowResultDict."
            )

        return BasicFlowRunMetadata(
            child_is_workflow=child_is_workflow,
            child_id=self.component.instance_id,
            child_run_id=result.run_id if child_is_workflow else NO_VAL,
        )

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[BasicFlowRunMetadata, Mapping[str, Any]]:
        """Synchronously delegate to the wrapped component."""
        result = self.component.invoke(inputs)

        if not isinstance(result, Mapping):
            raise ValidationError(
                f"{type(self).__name__}.{self.name}: wrapped component must "
                f"return a mapping-shaped result for workflow handoff; "
                f"got {type(result)!r}."
            )

        return self._build_metadata(result), result

    async def _async_run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[BasicFlowRunMetadata, Mapping[str, Any]]:
        """Asynchronously delegate to the wrapped component's native async path."""
        result = await self.component.async_invoke(inputs)

        if not isinstance(result, Mapping):
            raise ValidationError(
                f"{type(self).__name__}.{self.name}: wrapped component must "
                f"return a mapping-shaped result for async workflow handoff; "
                f"got {type(result)!r}."
            )

        return self._build_metadata(result), result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize the workflow wrapper plus its wrapped component snapshot."""
        data = super().to_dict()
        data.update(
            {
                "component": self.component.to_dict(),
            }
        )
        return data
