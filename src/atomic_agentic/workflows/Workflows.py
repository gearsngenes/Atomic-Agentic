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
from collections import OrderedDict
from typing import (
    Any,
    List,
    Mapping,
    Type,
    Tuple,
    Optional,
    Union,
    get_type_hints,
)


from ..core.Exceptions import ValidationError, SchemaError
from ..core.Invokable import AtomicInvokable, ArgumentMap
from ..tools import Tool
from ..agents import Agent
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
        component: Union[AtomicInvokable],
        *,
        output_schema: Optional[Union[list[str], Mapping[str, Any]]] = None,
        bundling_policy: Optional[BundlingPolicy] = BundlingPolicy.BUNDLE,
        mapping_policy: Optional[MappingPolicy] = MappingPolicy.STRICT,
        absent_val_policy: Optional[AbsentValPolicy] = AbsentValPolicy.RAISE,
        default_absent_val: Any = None,
    ) -> None:
        # Store component before calling super so build_args_returns can access it
        self._component = component
        # arguments_map is always taken directly from the normalized component.
        super().__init__(
            name=component.name,
            description=component.description,
            output_schema=output_schema,
            bundling_policy=bundling_policy,
            mapping_policy=mapping_policy,
            absent_val_policy=absent_val_policy,
            default_absent_val=default_absent_val
        )

    # ------------------------------------------------------------------ #
    # properties
    # ------------------------------------------------------------------ #
    @property
    def component(self) -> AtomicInvokable:
        return self._component
    @component.setter
    def component(self, candidate: AtomicInvokable) -> None:
        self._component = candidate
        self._arguments_map, self._return_type = self.build_args_returns()
    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def _invoke(self, inputs: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
        # Run the component as a workflow, yielding a mapping result.
        raw = self._component.invoke(inputs)
        return {
            "type_executed":type(self._component).__name__,
            "component_executed":self._component.name}, raw
    
    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _get_arguments(self) -> ArgumentMap:
        return self.component.arguments_map

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            OrderedDict(
                component=self._component.to_dict(),
            )
        )
        return d


#------------------------------------------------------------------------------#
# Stateful Graph Node Wrapper
#------------------------------------------------------------------------------#
def _is_typed_dict_class(obj: Any) -> bool:
    """
    Runtime check for TypedDict classes.
    TypedDict classes in runtime have .__annotations__ and .__total__ attributes.
    """
    return isinstance(obj, type) and issubclass(obj, dict) and hasattr(obj, "__annotations__") and hasattr(obj, "__total__")

def _extract_state_keys(
    state_schema: Union[
        Type[Any],   # TypedDict subclass
        List[str],
        Mapping[str, Any],
    ]
) -> List[str]:
    """
    Normalize a state_schema into a list of state keys.
    Accepts:
      - TypedDict subclass => extract keys from __annotations__
      - list[str] => use list directly
      - mapping => use its keys()

    Raises SchemaError for unsupported types.
    """
    if isinstance(state_schema, list):
        # list of keys
        return state_schema

    if isinstance(state_schema, dict):
        # mapping of key->default/type
        return list(state_schema.keys())

    if _is_typed_dict_class(state_schema):
        # TypedDict: use get_type_hints to resolve forward refs
        return list(get_type_hints(state_schema).keys())

    raise SchemaError(
        f"state_schema must be TypedDict subclass, list[str], or mapping; got {state_schema!r}"
    )

class StateIOFlow(BasicFlow):
    """
    A specialized AdapterFlow intended for stateful graph nodes (e.g., LangGraph).
    It treats the component as a node that takes a state dict and emits a state update dict.
    Missing output values are always dropped (AbsentValPolicy.DROP).

    The provided `state_schema` defines the universe of keys the node can handle
    for both input and output.

    The component's *declared* input keys (excluding VAR_POSITIONAL/VAR_KEYWORD)
    must be a subset of these state keys.
    """

    def __init__(
        self,
        component: Union[Workflow, Tool, Agent],
        *,
        state_schema: Union[Type[Any], List[str], Mapping[str, Any]],
        mapping_policy: MappingPolicy = MappingPolicy.STRICT,
        bundling_policy: BundlingPolicy = BundlingPolicy.UNBUNDLE,
    ):
        state_keys = _extract_state_keys(state_schema)

        super().__init__(
            component=component,
            output_schema=state_keys,
            mapping_policy=mapping_policy,
            bundling_policy=bundling_policy,
            absent_val_policy=AbsentValPolicy.DROP,
        )

        # Compute "real" input keys from arguments_map (exclude VAR_* parameters)
        component_input_keys: set[str] = set()
        for name, meta in (self.arguments_map or {}).items():
            kind = (meta or {}).get("kind")
            if kind in {"VAR_POSITIONAL", "VAR_KEYWORD"}:
                continue
            component_input_keys.add(name)

        unknown_inputs = component_input_keys.difference(state_keys)
        if unknown_inputs:
            raise SchemaError(
                f"Component inputs {unknown_inputs} must be a subset of state schema keys {state_keys}"
            )

        self._state_schema_keys = tuple(state_keys)
        self._component_input_keys = tuple(sorted(component_input_keys))

    def _invoke(self, inputs: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
        """
        Filter the incoming state down to the component's declared non-var inputs,
        then invoke the wrapped component.
        """
        filtered_inputs = {k: inputs[k] for k in self._component_input_keys if k in inputs}
        raw = self.component.invoke(filtered_inputs)
        return {}, raw

    @property
    def component(self) -> AtomicInvokable:
        return self._component
    
    @component.setter
    def component(self, candidate: AtomicInvokable) -> None:
        if not set(candidate.arguments_map.keys()).issubset(self.state_schema_keys):
            raise SchemaError(f"Candidate must have input keys that are a subset of {self.state_schema_keys}. "
                              f"Instead, got {candidate.arguments_map.keys()}.")
        self._component = candidate
        self._arguments_map, self._return_type = self.build_args_returns()
    
    @property
    def state_schema_keys(self) -> tuple[str, ...]:
        return self._state_schema_keys

    @property
    def absent_val_policy(self) -> AbsentValPolicy:
        return AbsentValPolicy.DROP

    @property
    def output_schema(self) -> Mapping[str, Any]:
        return OrderedDict(self._output_schema)
    
    @output_schema.setter
    def output_schema(self, value: Union[List[str], Mapping[str, Any]]) -> None:
        pass