"""Workflow wrappers.

This module contains thin workflow adapters around executable nodes.

`BasicFlow` wraps any `AtomicInvokable` and exposes it as a workflow-shaped node.
Its responsibilities are limited to:

- delegating sync/async execution to the wrapped component;
- preserving the wrapped component's input parameters and return type;
- unwrapping the component's AtomicResult into a BasicFlowResult, recording
  the delegate's identity and run id for correlation.

`BasicFlow` performs no output reshaping or type validation. The payload in
`BasicFlowResult.result` is exactly `component_result.result`, whatever type
that is.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Optional

from ..core.Invokable import AtomicInvokable
from ..models.results.workflows import BasicFlowResult
from .base import Workflow

logger = logging.getLogger(__name__)

__all__ = ["BasicFlow"]


class BasicFlow(Workflow):
    """Thin workflow adapter that transparently delegates to a wrapped component.

    `BasicFlow` delegates directly to its wrapped component, which may be
    another `Workflow` or any other `AtomicInvokable`. The component's result
    payload is passed through unchanged as `BasicFlowResult.result`, and the
    component's identity and run id are recorded on the `BasicFlowResult` for
    correlation (see `results.workflows.BasicFlowResult`).
    """

    def __init__(
        self,
        component: AtomicInvokable,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        namespace: Optional[str] = None,
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
            namespace=namespace or component.namespace,  # inherit when not supplied
            parameters=component.parameters,
            return_type=component.return_type,
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
    # Result construction
    # ------------------------------------------------------------------ #
    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> BasicFlowResult:
        """Construct a BasicFlowResult envelope for this BasicFlow's invocation."""
        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=BasicFlowResult,
            **result_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Workflow run hooks
    # ------------------------------------------------------------------ #
    def _run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Synchronously delegate to the wrapped component."""
        child_result = self.component.invoke(inputs)
        return child_result.result, {
            "child_id": child_result.invoker_id,
            "child_type": type(self.component).__name__,
            "child_run_id": child_result.run_id,
        }

    async def _async_run(self, inputs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Asynchronously delegate to the wrapped component's native async path."""
        child_result = await self.component.async_invoke(inputs)
        return child_result.result, {
            "child_id": child_result.invoker_id,
            "child_type": type(self.component).__name__,
            "child_run_id": child_result.run_id,
        }

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
