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

logger = logging.getLogger(__name__)

__all__ = ["BasicFlow"]


class BasicFlow(Workflow):
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

    @component.setter
    def component(self, candidate: StructuredInvokable | Workflow) -> None:
        """Swap the wrapped component and refresh only the mirrored parameters."""
        if not isinstance(candidate, (StructuredInvokable, Workflow)):
            raise TypeError(
                "BasicFlow.component must be a StructuredInvokable or Workflow, "
                f"got {type(candidate)!r}"
            )

        self._component = candidate
        self._parameters = candidate.parameters

    # ------------------------------------------------------------------ #
    # Metadata helpers
    # ------------------------------------------------------------------ #
    def _build_metadata(self, result: Mapping[str, Any]) -> dict[str, Any]:
        """Build checkpoint metadata from the wrapped component and result carrier."""
        metadata: dict[str, Any] = {
            "component_type": type(self.component).__name__,
            "component_id": self.component.instance_id,
            "component_full_name": self.component.full_name,
            "result_type": type(result).__name__,
        }

        if isinstance(result, FlowResultDict):
            metadata["run_id"] = result.run_id
        elif isinstance(result, StructuredResultDict):
            metadata["run_raw_result"] = result.raw_result

        return metadata

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
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
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
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