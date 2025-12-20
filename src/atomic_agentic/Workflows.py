"""Workflow wrappers.

This module contains Workflow adapters and decorators around Tools, Agents, and Workflows.

- ToolFlow and AgentFlow are thin GoF-style Adapters that normalize a Tool/Agent
  into the Workflow execution + packaging boundary.
- MonoFlow is a Decorator-style Workflow wrapper designed for "russian-doll"
  composition, where each wrapper layer can re-package / transform the mapping
  produced by the wrapped component.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Optional, Union

from .Exceptions import ValidationError
from .Primitives import Agent, ArgumentMap, BundlingPolicy, MappingPolicy, Tool, Workflow

logger = logging.getLogger(__name__)

__all__ = [
    "BundlingPolicy",
    "MappingPolicy",
    "Workflow",
    "ToolFlow",
    "AgentFlow",
    "MonoFlow",
]


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


class MonoFlow(Workflow):
    """Decorator-style Workflow wrapper (russian-doll packaging).

    MonoFlow wraps a single component that is either:
    - Tool      -> normalized to ToolFlow
    - Agent     -> normalized to AgentFlow
    - Workflow  -> used as-is

    Key behavior:
    - The wrapped component is executed as a Workflow, producing an *already-packaged*
      output mapping.
    - That mapping is returned as this wrapper's "raw" result, so this wrapper's
      packaging rules can re-package / transform the mapping again.
    - This enables nested wrappers to repeatedly re-package layer by layer.
    """

    def __init__(
        self,
        component: Union[Workflow, Tool, Agent],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        arguments_map: Optional[ArgumentMap] = None,
        output_schema: Optional[Union[list[str], Mapping[str, Any]]] = None,
        bundling_policy: BundlingPolicy = BundlingPolicy.BUNDLE,
        mapping_policy: MappingPolicy = MappingPolicy.STRICT,
    ) -> None:
        if not isinstance(component, (Workflow, Tool, Agent)):
            raise ValidationError(
                "MonoFlow: component must be a Workflow, Tool, or Agent; "
                f"got {type(component).__name__}"
            )

        super().__init__(
            name=name or component.name,
            description=description or component.description,
            arguments_map=arguments_map or component.arguments_map,
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
        normalized = self._normalize_component(
            component=value,
            output_schema=self.output_schema,
            bundling_policy=self.bundling_policy,
            mapping_policy=self.mapping_policy,
        )
        self._component = normalized
        self._set_io_schemas(arguments_map=self._component.arguments_map, output_schema=self.output_schema)

    def _invoke(self, inputs: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
        # Run the component as a workflow, yielding a mapping result.
        raw_mapping = self._component.invoke(inputs)

        # Start metadata from the wrapped component's latest checkpoint (if any).
        latest = self._component.latest_checkpoint
        base_meta: Mapping[str, Any] = latest.metadata if latest is not None else {}
        meta = dict(base_meta)  # shallow copy to avoid ghost mutation

        # Track nesting depth.
        prior_depth = meta.get("__component_depth__", 0)
        depth = (prior_depth + 1) if isinstance(prior_depth, int) and prior_depth >= 0 else 1

        meta.update(
            {
                "__component_executed__": self._component.name,
                "__component_depth__": depth,
                "__component_schema__": self._component.output_schema,
            }
        )

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
        bundling_policy: BundlingPolicy,
        mapping_policy: MappingPolicy,
    ) -> Workflow:
        """Normalize Tool/Agent into ToolFlow/AgentFlow; Workflow passes through."""
        if isinstance(component, Workflow):
            return component

        if isinstance(component, Tool):
            # Inner ToolFlow uses wrapper-provided output_schema/policies for russian-doll packaging.
            return ToolFlow(
                component,
                output_schema=output_schema,
                bundling_policy=bundling_policy,
                mapping_policy=mapping_policy,
            )

        if isinstance(component, Agent):
            return AgentFlow(
                component,
                output_schema=output_schema,
                bundling_policy=bundling_policy,
                mapping_policy=mapping_policy,
            )

        # Defensive (should be unreachable due to caller guards)
        raise ValidationError(
            "MonoFlow: component must be a Workflow, Tool, or Agent; "
            f"got {type(component).__name__}"
        )
