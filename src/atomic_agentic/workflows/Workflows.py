"""Workflow wrappers.

This module contains Workflow adapters and decorators around Tools, Agents, and Workflows.

- ToolFlow and AgentFlow are thin GoF-style Adapters that normalize a Tool/Agent
  into the Workflow execution + packaging boundary.
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
    Optional,
    Union,
    get_type_hints,
)


from ..core.Exceptions import ValidationError, SchemaError
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
    "BundlingPolicy",
    "MappingPolicy",
    "AbsentValPolicy",
    "Workflow",
    "ToolFlow",
    "AgentFlow",
    "AdapterFlow",
    "StateIOFlow",
    "DEFAULT_WF_KEY",
]


#------------------------------------------------------------------------------#
# Thin Tool Workflow Adapter
#------------------------------------------------------------------------------#
class ToolFlow(Workflow):
    """A thin Workflow boundary around a single :class:`~atomic_agentic.Primitives.Tool`.

    - Inputs are forwarded as a mapping to ``tool.invoke(inputs)``.
    - ``arguments_map`` is proxied from the wrapped tool.
    """

    def __init__(
        self,
        tool: Tool,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        output_schema: Optional[Union[list[str], Mapping[str, Any]]] = None,
        bundling_policy: BundlingPolicy = BundlingPolicy.BUNDLE,
        mapping_policy: MappingPolicy = MappingPolicy.STRICT,
        absent_val_policy: AbsentValPolicy = AbsentValPolicy.RAISE,
    ) -> None:
        if not isinstance(tool, Tool):
            raise ValidationError(f"ToolFlow: tool must be a Tool; got {type(tool).__name__}")

        self._tool = tool

        super().__init__(
            name=name or tool.name,
            description=description or tool.description,
            arguments_map=tool.arguments_map,
            output_schema=output_schema,
            bundling_policy=bundling_policy,
            mapping_policy=mapping_policy,
            absent_val_policy=absent_val_policy,
        )

    @property
    def tool(self) -> Tool:
        return self._tool

    @tool.setter
    def tool(self, value: Tool) -> None:
        if not isinstance(value, Tool):
            raise ValidationError(f"ToolFlow: tool must be a Tool; got {type(value).__name__}")
        self._tool = value
        self._set_io_schemas(arguments_map=self._tool.arguments_map, output_schema=self.output_schema)

    def _invoke(self, inputs: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
        raw = self._tool.invoke(inputs)
        return {"tool_executed": self._tool.full_name}, raw

    def to_dict(self) -> OrderedDict[str, Any]:
        d = super().to_dict()
        d.update(OrderedDict(
                tool=self._tool.to_dict(),
        ))
        return d


#------------------------------------------------------------------------------#
# Thin Agent Workflow Adapter
#------------------------------------------------------------------------------#
class AgentFlow(Workflow):
    """A thin Workflow boundary around a single :class:`~atomic_agentic.Primitives.Agent`.

    - Inputs are forwarded as a mapping to ``agent.invoke(inputs)``.
    - ``arguments_map`` is proxied from the agent.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        output_schema: Optional[Union[list[str], Mapping[str, Any]]] = None,
        bundling_policy: BundlingPolicy = BundlingPolicy.BUNDLE,
        mapping_policy: MappingPolicy = MappingPolicy.STRICT,
        absent_val_policy: AbsentValPolicy = AbsentValPolicy.RAISE,
    ) -> None:
        if not isinstance(agent, Agent):
            raise ValidationError(f"AgentFlow: agent must be an Agent; got {type(agent).__name__}")

        super().__init__(
            name=name or agent.name,
            description=description or agent.description,
            arguments_map=agent.arguments_map,
            output_schema=output_schema,
            bundling_policy=bundling_policy,
            mapping_policy=mapping_policy,
            absent_val_policy=absent_val_policy,
        )

        self._agent = agent

    @property
    def agent(self) -> Agent:
        return self._agent

    @agent.setter
    def agent(self, value: Agent) -> None:
        if not isinstance(value, Agent):
            raise ValidationError(f"AgentFlow: agent must be an Agent; got {type(value).__name__}")
        self._agent = value
        self._set_io_schemas(arguments_map=self._agent.arguments_map, output_schema=self.output_schema)

    def clear_memory(self) -> None:
        super().clear_memory()
        self._agent.clear_memory()

    def _invoke(self, inputs: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
        raw = self._agent.invoke(inputs)
        new_history = self._agent.history[-2:] if self._agent.history else []
        meta: dict[str, Any] = {
            "agent_executed": self._agent.name,
            "new_saved_llm_history": new_history,
        }
        return meta, raw

    def to_dict(self) -> OrderedDict[str, Any]:
        d = super().to_dict()
        d.update(
            OrderedDict(
                agent=self._agent.to_dict(),
            )
        )
        return d


#------------------------------------------------------------------------------#
# Generic Workflow Wrapper/Adapter
#------------------------------------------------------------------------------#
ADAPTER_DEPTH = "__ADAPTER_DEPTH__"

class AdapterFlow(Workflow):
    """A concrete, generic Workflow wrapper for Tools, Agents, and Workflows.

    AdapterFlow wraps a single component that is either:

    - Tool      -> normalized to ToolFlow
    - Agent     -> normalized to AgentFlow
    - Workflow  -> used as-is

    Key behavior:

    - The wrapped component is executed as a Workflow, producing an *already-packaged*
      output mapping.
    - That mapping is returned as this wrapper's "raw" result, so this wrapper's
      packaging rules can re-package / transform the mapping again.
    - AdapterFlow is a mutable configuration boundary: changes to name/description,
      output_schema, bundling_policy, and mapping_policy are mirrored into the wrapped
      component.

    Notes:

    - The wrapped component is intentionally not swappable after construction.
    - arguments_map and input_schema remain read-only and always mirror the component.
    """

    def __init__(
        self,
        component: Union[Workflow, Tool, Agent],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        output_schema: Optional[Union[list[str], Mapping[str, Any]]] = None,
        bundling_policy: Optional[BundlingPolicy] = None,
        mapping_policy: Optional[MappingPolicy] = None,
        absent_val_policy: Optional[AbsentValPolicy] = None,
    ) -> None:
        normalized = self._normalize_component(component=component)

        effective_name = name if name is not None else normalized.name
        effective_description = description if description is not None else normalized.description
        effective_output_schema = output_schema if output_schema is not None else normalized.output_schema
        effective_bundling = (
            BundlingPolicy(bundling_policy) if bundling_policy is not None else normalized.bundling_policy
        )
        effective_mapping = (
            MappingPolicy(mapping_policy) if mapping_policy is not None else normalized.mapping_policy
        )
        effective_absent_val_policy = absent_val_policy if absent_val_policy is not None else normalized.absent_val_policy

        # arguments_map is always taken directly from the normalized component.
        super().__init__(
            name=effective_name,
            description=effective_description,
            arguments_map=normalized.arguments_map,
            output_schema=effective_output_schema,
            bundling_policy=effective_bundling,
            mapping_policy=effective_mapping,
            absent_val_policy=effective_absent_val_policy,
        )

        self._component = normalized
        self._mirror_to_component()

    @property
    def component(self) -> Workflow:
        return self._component

    # ------------------------------------------------------------------ #
    # Mirrored, mutable properties
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return self._name
    @name.setter
    def name(self, value: str) -> None:
        self._name = value
        self._component.name = self.name

    @property
    def description(self) -> str:
        return self._description
    @description.setter
    def description(self, value: str) -> None:
        self._description = value
        self._component.description = self.description

    @property
    def output_schema(self) -> Mapping[str, Any]:
        return self._output_schema
    @output_schema.setter
    def output_schema(self, value: Optional[Union[list[str], Mapping[str, Any]]]) -> None:
        if value is None:
            value = [DEFAULT_WF_KEY]
        self._set_io_schemas(arguments_map=self.arguments_map, output_schema=value)
        self._component.output_schema = self.output_schema

    @property
    def bundling_policy(self) -> BundlingPolicy:
        return self._bundling_policy
    @bundling_policy.setter
    def bundling_policy(self, value: Union[BundlingPolicy, str]) -> None:
        self._bundling_policy = BundlingPolicy(value)
        self._component.bundling_policy = self.bundling_policy

    @property
    def mapping_policy(self) -> MappingPolicy:
        return self._mapping_policy
    @mapping_policy.setter
    def mapping_policy(self, value: Union[MappingPolicy, str]) -> None:
        self._mapping_policy = MappingPolicy(value)
        self._component.mapping_policy = self.mapping_policy
    
    @property
    def absent_val_policy(self) -> AbsentValPolicy:
        return self._absent_val_policy
    @absent_val_policy.setter
    def absent_val_policy(self, value: Union[AbsentValPolicy, str]) -> None:
        self._absent_val_policy = AbsentValPolicy(value)
        self._component.absent_val_policy = self.absent_val_policy

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def _invoke(self, inputs: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
        # Run the component as a workflow, yielding a mapping result.
        raw_mapping = self._component.invoke(inputs)

        raw: Mapping = raw_mapping

        # Avoid definitively unintended duplicate nesting for single-key bundling:
        # If the component already returned {k: value} and we are about to BUNDLE into k again,
        # unwrap to `value` so that this wrapper's packaging produces {k: value} (not {k:{k:...}}).
        if (
            self.bundling_policy == BundlingPolicy.BUNDLE
            and len(self.output_schema) == 1
            and isinstance(raw_mapping, Mapping)
        ):
            schema_key = next(iter(self.output_schema.keys()))
            if len(raw_mapping) == 1 and schema_key in raw_mapping:
                raw = raw_mapping[schema_key]

        # Start metadata from the wrapped component's latest checkpoint (if any).
        latest = self._component.latest_checkpoint
        base_meta: Mapping[str, Any] = latest.metadata if latest is not None else {}
        meta = dict(base_meta)  # shallow copy to avoid ghost mutation

        prior_depth = meta.get(ADAPTER_DEPTH, 0)
        depth = (prior_depth + 1) if isinstance(prior_depth, int) and prior_depth >= 0 else 1
        meta[ADAPTER_DEPTH] = depth

        return meta, raw

    def to_dict(self) -> OrderedDict[str, Any]:
        d = super().to_dict()
        d.update(
            OrderedDict(
                component=self._component.to_dict(),
            )
        )
        return d

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _mirror_to_component(self) -> None:
        """Push this wrapper's mutable configuration down into the component."""
        self._component.name = self.name
        self._component.description = self.description
        self._component.output_schema = self.output_schema
        self._component.bundling_policy = self.bundling_policy
        self._component.mapping_policy = self.mapping_policy
        self._component.absent_val_policy = self.absent_val_policy

        # Ensure our IO schemas remain consistent with the component's arguments_map.
        self._set_io_schemas(arguments_map=self._component.arguments_map, output_schema=self.output_schema)

    @staticmethod
    def _normalize_component(*, component: Union[Workflow, Tool, Agent]) -> Workflow:
        """Normalize Tool/Agent into ToolFlow/AgentFlow; Workflow passes through."""
        if isinstance(component, Workflow):
            return component

        if isinstance(component, Tool):
            return ToolFlow(component)

        if isinstance(component, Agent):
            return AgentFlow(component)

        raise ValidationError(
            "AdapterFlow: component must be a Workflow, Tool, or Agent; "
            f"got {type(component).__name__}"
        )

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

class StateIOFlow(AdapterFlow):
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
    def state_schema_keys(self) -> tuple[str, ...]:
        return self._state_schema_keys

    @property
    def absent_val_policy(self) -> AbsentValPolicy:
        return AbsentValPolicy.DROP

    @property
    def output_schema(self) -> Mapping[str, Any]:
        return OrderedDict(self._output_schema)

