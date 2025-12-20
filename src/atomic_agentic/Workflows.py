"""Workflow wrappers.

This module contains thin Workflow adapters around a single Tool or Agent.

The wrappers delegate execution to the underlying component and rely on the
base :class:`~atomic_agentic.Primitives.Workflow` to package outputs and record
checkpoints.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, Optional, Union

from .Exceptions import ValidationError
from .Primitives import (
    Agent,
    BundlingPolicy,
    DEF_RES_KEY,
    MappingPolicy,
    Tool,
    Workflow,
    ArgumentMap
)

logger = logging.getLogger(__name__)

# Backwards-compatibility alias (many older workflows used this key name).
__all__ = [
    "Workflow",
    "ToolFlow",
    "AgentFlow",
    "DEF_RES_KEY",
]


class ToolFlow(Workflow):
    """A thin Workflow boundary around a single :class:`~atomic_agentic.Primitives.Tool`.

    - Inputs are forwarded as a mapping to ``tool.invoke(inputs)``.
    - ``arguments_map`` is proxied from the wrapped tool.
    """

    def __init__(
        self,
        tool: Union[Tool, Callable[..., Any]],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        output_schema: Optional[Union[list[str], Mapping[str, Any]]] = None,
        bundling_policy: BundlingPolicy = BundlingPolicy.BUNDLE,
        mapping_policy: MappingPolicy = MappingPolicy.STRICT,
    ) -> None:
        super().__init__(
            name=name or self._tool.name,
            description=description or self._tool.description,
            arguments_map=self._tool.arguments_map,
            output_schema=output_schema,
            bundling_policy=bundling_policy,
            mapping_policy=mapping_policy,
        )
        if not isinstance(tool, Tool):
            raise ValueError("ToolFlow: `tool` must be a Tool; got {type(tool).__name__}")
        self._tool = Tool

    @property
    def tool(self) -> Tool:
        return self._tool
    
    @tool.setter
    def tool(self, value: Tool) -> None:
        if not isinstance(value, Tool):
            raise ValueError("ToolFlow: `tool` must be a Tool; got {type(value).__name__}")
        self._tool = value
        self._set_io_schemas(arguments_map=self._tool.arguments_map, output_schema=self.output_schema)

    def _invoke(self, inputs: Mapping[str, Any]) -> tuple[Mapping, Any]:
        raw = self._tool.invoke(inputs)
        return {"tool_executed":self._tool.full_name}, raw

    def to_dict(self) -> OrderedDict[str, Any]:
        d = super().to_dict()
        d.update(OrderedDict(
                tool=self._tool.to_dict(),
        ))
        return d


class AgentFlow(Workflow):
    """A thin Workflow boundary around a single :class:`~atomic_agentic.Primitives.Agent`.

    - Inputs are forwarded as a mapping to ``agent.invoke(inputs)``.
    - ``arguments_map`` is proxied from the agent (mirrors its ``pre_invoke`` Tool).
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
    ) -> None:
        super().__init__(
            name=name or self._agent.name,
            description=description or self._agent.description,
            arguments_map=self._agent.arguments_map,
            output_schema=output_schema,
            bundling_policy=bundling_policy,
            mapping_policy=mapping_policy,
        )
        if not isinstance(agent, Agent):
            raise ValidationError(
                f"AgentFlow: agent must be an Agent; got {type(agent).__name__}"
            )
        self._agent = agent

    @property
    def agent(self) -> Agent:
        return self._agent
    
    @agent.setter
    def agent(self, value: Agent) -> None:
        if not isinstance(value, Agent):
            raise ValidationError(
                f"AgentFlow: agent must be an Agent; got {type(value).__name__}"
            )
        self._agent = value
        self._set_io_schemas(arguments_map=self._agent.arguments_map, output_schema=self.output_schema)

    def clear_memory(self) -> None:
        super().clear_memory()
        self._agent.clear_memory()

    def _invoke(self, inputs: Mapping[str, Any]) -> tuple[Mapping, Any]:
        raw = self._agent.invoke(inputs)
        new_history = self.agent.history[-2:] if self.agent.history else []
        meta = {"new_saved_llm_history": new_history,
                "agent_executed": self.agent.name,
                }
        return meta, raw

    def to_dict(self) -> OrderedDict[str, Any]:
        d = super().to_dict()
        d.update(OrderedDict(
            agent=self._agent.to_dict(),
        ))
        return d


class BasicFlow(Workflow):
    """Decorator-style Workflow wrapper.

    BasicFlow wraps a single component that is either:
    - Tool   -> normalized to ToolFlow
    - Agent  -> normalized to AgentFlow
    - Workflow -> used as-is

    Input contract:
        Always pass-through from the normalized component's arguments_map.

    Output contract:
        If `output_schema` is not provided, inherits component.output_schema.
        If provided, uses wrapper-defined schema/policies to re-package the
        component's workflow output mapping.

    Metadata:
        Returns the wrapped component's latest checkpoint metadata (if any),
        copied and augmented with:
        - __WF_WRAP_DEPTH__ (incremented per BasicFlow nesting layer)
        - __WF_WRAPPER__ info describing the outermost wrapper
    """

    def __init__(
        self,
        component: Union[Workflow, Tool, Agent],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        output_schema: Optional[Union[list[str], Mapping[str, Any]]] = None,
        bundling_policy: BundlingPolicy = BundlingPolicy.BUNDLE,
        mapping_policy: MappingPolicy = MappingPolicy.STRICT,
    ) -> None:
        super().__init__(
            name=name or component.name,
            description=description or component.description,
            arguments_map=component.arguments_map,
            output_schema=output_schema,
            bundling_policy=bundling_policy,
            mapping_policy=mapping_policy,
        )

        self._component: Workflow = self._normalize_component(
            component=component,
            output_schema=output_schema,
            bundling_policy=bundling_policy,
            mapping_policy=mapping_policy,
        )

    @property
    def component(self) -> Workflow:
        return self._component
    
    @component.setter
    def component(self, value: Union[Tool, Agent, Workflow]) -> None:
        normalized_component = self._normalize_component(
            component=value,
            output_schema=self.output_schema,
            bundling_policy=self.bundling_policy,
            mapping_policy=self.mapping_policy,
        )
        self._component = normalized_component
        self._set_io_schemas(arguments_map=self._component.arguments_map, output_schema=self.output_schema)

    def _invoke(self, inputs: Mapping[str, Any]) -> tuple[Mapping, Any]:
        # Run the component as a workflow, yielding a mapping result.
        raw_mapping = self._component.invoke(inputs)

        # Start metadata from the wrapped component's latest checkpoint (if any).
        base_meta: Mapping[str, Any] = {}
        latest = self._component.latest_checkpoint
        base_meta = latest.metadata
        meta = dict(base_meta)  # shallow copy to avoid ghost mutation

        prior_depth = meta.get("__component_executed__", 0)
        if isinstance(prior_depth, int) and prior_depth >= 0:
            depth = prior_depth + 1
        else:
            depth = 1

        meta.update({
            "__component_executed__": self._component.name,
            "__component_depth__" : depth,
            "__component_schema__" : self._component.output_schema,
        })

        # IMPORTANT: return the mapping as "raw" so *this* workflow's packaging rules apply.
        return meta, raw_mapping

    def to_dict(self) -> OrderedDict[str, Any]:
        d = super().to_dict()
        d.update(
            OrderedDict(
                component=self._component.to_dict(),
            )
        )
        return d

    @staticmethod
    def _normalize_component(
        *,
        component: Union[Workflow, Tool, Agent],
        output_schema: Optional[Union[list[str], Mapping[str, Any]]],
        bundling_policy: Optional[BundlingPolicy],
        mapping_policy: Optional[MappingPolicy],
    ) -> Workflow:
        """Normalize Tool/Agent into ToolFlow/AgentFlow; Workflow passes through."""
        if isinstance(component, Workflow):
            return component

        if isinstance(component, Tool):
            # Inner ToolFlow uses wrapper-provided output_schema/policies when provided,
            # otherwise defaults inside ToolFlow apply.
            inner = ToolFlow(
                component,
                output_schema=output_schema,
                bundling_policy=bundling_policy or BundlingPolicy.UNBUNDLE,
                mapping_policy=mapping_policy or MappingPolicy.STRICT,
            )
            return inner

        if isinstance(component, Agent):
            inner = AgentFlow(
                component,
                output_schema=output_schema,
                bundling_policy=bundling_policy or BundlingPolicy.UNBUNDLE,
                mapping_policy=mapping_policy or MappingPolicy.STRICT,
            )
            return inner

        raise ValidationError(
            "BasicFlow: component must be a Workflow, Tool, or Agent; "
            f"got {type(component).__name__}"
        )