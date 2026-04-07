"""Workflow wrappers.

This module contains thin workflow adapters around already-structured nodes.

`BasicFlow` wraps either:
- a `StructuredInvokable`, or
- another `Workflow`

and exposes it as a workflow node whose responsibilities are limited to:
- delegating sync/async execution,
- preserving the wrapped component's input parameters,
- emitting lightweight metadata for checkpointing.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from ..core.Exceptions import ValidationError
from .StructuredInvokable import StructuredInvokable, StructuredResultDict
from .base import FlowResultDict, Workflow
from .metadata import BasicFlowRunMetadata, NO_VAL

logger = logging.getLogger(__name__)

__all__ = ["BasicFlow"]


class BasicFlow(Workflow[BasicFlowRunMetadata]):
    """Thin workflow adapter for structured invokables and workflows.

    `BasicFlow` does not perform any output packaging of its own. The wrapped
    component is expected to already return a mapping-shaped result:

    - `StructuredInvokable` -> `StructuredResultDict`
    - `Workflow` -> `FlowResultDict`

    The outer workflow layer then records its own checkpoint and wraps the final
    mapping in a fresh outer `FlowResultDict`.
    """

    def __init__(
        self,
        component: StructuredInvokable | Workflow,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        filter_extraneous_inputs: Optional[bool] = None,
    ) -> None:
        if not isinstance(component, (StructuredInvokable, Workflow)):
            raise TypeError(
                "BasicFlow.component must be a StructuredInvokable or Workflow, "
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
    def component(self) -> StructuredInvokable | Workflow:
        """The wrapped structured component."""
        return self._component

    # ------------------------------------------------------------------ #
    # Metadata helpers
    # ------------------------------------------------------------------ #
    def _build_metadata(self, result: Mapping[str, Any]) -> BasicFlowRunMetadata:
        """Build typed checkpoint metadata from the wrapped component and result carrier."""
        child_is_workflow = isinstance(self.component, Workflow)

        if child_is_workflow:
            if not isinstance(result, FlowResultDict):
                raise ValidationError(
                    f"{type(self).__name__}.{self.name}: wrapped workflow child returned "
                    f"{type(result)!r}, expected FlowResultDict"
                )

            return BasicFlowRunMetadata(
                child_is_workflow=True,
                child_id=self.component.instance_id,
                child_full_name=self.component.full_name,
                child_run_id=result.run_id,
                child_raw_result=NO_VAL,
                has_child_raw_result=False,
                child_raw_result_type="Any",
            )

        if not isinstance(result, StructuredResultDict):
            raise ValidationError(
                f"{type(self).__name__}.{self.name}: wrapped structured child returned "
                f"{type(result)!r}, expected StructuredResultDict"
            )

        raw_result = result.raw_result
        raw_result_type = type(raw_result).__name__ if raw_result is not NO_VAL else "Any"

        return BasicFlowRunMetadata(
            child_is_workflow=False,
            child_id=self.component.instance_id,
            child_full_name=self.component.full_name,
            child_run_id=NO_VAL,
            child_raw_result=raw_result,
            has_child_raw_result=True,
            child_raw_result_type=raw_result_type,
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
                f"{type(self).__name__}.{self.name}: wrapped component returned "
                f"a non-mapping result: {type(result)!r}"
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
                f"{type(self).__name__}.{self.name}: wrapped component returned "
                f"a non-mapping async result: {type(result)!r}"
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