# modules/ToolAdapters.py
from __future__ import annotations

from typing import Any, Mapping, Dict, List
import inspect
from collections import OrderedDict

from modules.Tools import Tool, ToolDefinitionError, NO_DEFAULT
from modules.Agents import Agent


# =========================
# Agent -> Tool (invoke)
# =========================

class AgentTool(Tool):
    """
    Wrap an Agent as a Tool.

    Metadata
    --------
    - type   = "agent"
    - source = agent.name
    - name   = "invoke"
    - description = agent.description

    Schema exposure
    ---------------
    Mirrors the agent.pre_invoke Tool's call-plan so planners see the correct keys.

    Execution
    ---------
    Accepts a flat mapping via Tool.invoke(...) and forwards it to agent.invoke(inputs).
    """

    def __init__(self, agent: Agent) -> None:
        if not isinstance(agent, Agent):
            raise ToolDefinitionError("AgentInvokeTool requires an Agents.Agent instance.")
        self.pre: Tool = agent.pre_invoke
        self.agent = agent
        if not isinstance(self.pre, Tool):
            raise ToolDefinitionError("AgentInvokeTool requires agent.pre_invoke to be a Tools.Tool.")

        # Use the pre_invoke call-plan to define our binding behavior.
        posonly_names: List[str] = list(self.pre.posonly_order)

        def _agent_wrapper(*_args: Any, **_kwargs: Any) -> Any:
            # Rebuild the flat mapping expected by Agent.invoke
            inputs: Dict[str, Any] = {}
            for i, pname in enumerate(posonly_names):
                if i < len(_args):
                    inputs[pname] = _args[i]
            inputs.update(_kwargs)
            return self.agent.invoke(inputs)

        super().__init__(
            func=_agent_wrapper,
            name="invoke",
            description=agent.description,
            type="agent",
            source=agent.name,
        )

        # Mirror pre_invoke Tool's call-plan (strict; no envelopes)
        self._arguments_map = self.pre.arguments_map
        self.posonly_order = list(self.pre.posonly_order)
        self.p_or_kw_names = list(self.pre.p_or_kw_names)
        self.kw_only_names = list(self.pre.kw_only_names)
        self.required_names = set(self.pre.required_names)
        self.has_varargs = bool(self.pre.has_varargs)
        self.varargs_name = self.pre.varargs_name
        self.has_varkw = bool(self.pre.has_varkw)
        self.varkw_name = self.pre.varkw_name