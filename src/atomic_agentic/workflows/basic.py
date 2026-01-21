"""Workflow wrappers.

This module contains Workflow adapters and decorators around Tools, Agents, and Workflows.

- `BasicFlow` is the thin adapter that normalizes a Tool/Agent/Workflow
  into the Workflow execution + packaging boundary (replaces ToolFlow/AgentFlow).
- AdapterFlow is an adapter Workflow wrapper designed to generalize/normalize any
  Tool, Agent, or Workflow (composition or otherwise) into a single Workflow Node.
- StateIOFlow is a specialized AdapterFlow for specialized use-cases (like in LanGraph)
  that requires the output schema to align with the input schema.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    Mapping,
    Optional,
    Union,
    List,
)

from ..core.Invokable import AtomicInvokable
from ..core.Parameters import ParamSpec
from .base import (
    Workflow,
    BundlingPolicy,
    MappingPolicy,
    AbsentValPolicy,
    DEFAULT_WF_KEY,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BasicFlow",
    "StateIOFlow",
    "DEFAULT_WF_KEY",
]

class BasicFlow(Workflow):
    """A concrete, generic Workflow wrapper for Tools, Agents, and Workflows.

    BasicFlow wraps a single AtomicInvokable component into a WorkFlow:
    """

    def __init__(
        self,
        component: AtomicInvokable,
        *,
        output_schema: Optional[Union[type, List[Union[str, ParamSpec]], Mapping[str, Any]]] = None,
        bundling_policy: Optional[BundlingPolicy] = BundlingPolicy.BUNDLE,
        mapping_policy: Optional[MappingPolicy] = MappingPolicy.STRICT,
        absent_val_policy: Optional[AbsentValPolicy] = AbsentValPolicy.RAISE,
        default_absent_val: Any = None,
    ) -> None:
        # Store component before calling super
        self._component = component
        # Pass component parameters and return_type to parent for eager initialization and validation
        super().__init__(
            name=component.name,
            description=component.description,
            parameters=component.parameters,
            output_schema=output_schema,
            bundling_policy=bundling_policy,
            mapping_policy=mapping_policy,
            absent_val_policy=absent_val_policy,
            default_absent_val=default_absent_val
        )

    # ------------------------------------------------------------------ #
    # BasicFlow Properties
    # ------------------------------------------------------------------ #
    @property
    def component(self) -> AtomicInvokable:
        return self._component
    @component.setter
    def component(self, candidate: AtomicInvokable) -> None:
        self._component = candidate
        self._parameters = candidate.parameters
        self._return_type = candidate.return_type

    # ------------------------------------------------------------------ #
    # BasicFlow Helpers
    # ------------------------------------------------------------------ #
    def _invoke(self, inputs: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
        # Run the component as a workflow, yielding a mapping result.
        raw = self._component.invoke(inputs)
        meta = {
            "type_executed":type(self._component).__name__,
            "component_executed":self._component.full_name,
        }
        return meta, raw

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "component": self.component.to_dict()
        })
        return d