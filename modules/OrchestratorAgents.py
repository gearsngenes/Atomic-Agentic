import json, re, logging
from typing import Any
from modules.Agents import Agent, ToolAgent
import modules.Prompts as Prompts
from modules.Plugins import Plugin
from modules.LLMEngines import LLMEngine

class ToolOrchestratorAgent(ToolAgent):
    def __init__(self, name: str, description: str, llm_engine: LLMEngine):
        """
        Initializes context and toolbox.
        """
        super().__init__(name, description=description, llm_engine=llm_engine)
        self.role_prompt = Prompts.ORCHESTRATOR_PROMPT
        def _return(val: Any):
            return val
        self._previous_steps = []
        ToolOrchestratorAgent.register(self, _return, "Use this method to return a specific value object or string if the user specifically requests it")

    def register(self, tool: Any, description: str | None = None) -> None:
        """
        Register a tool (callable or Plugin) into self._toolbox.
        """
        if callable(tool):
            source = "__dev_tools__"
            if source not in self._toolbox:
                self._toolbox[source] = {}
            name = tool.__name__
            if name.startswith("<"):
                raise ValueError("Tools must be named functions, not lambdas or internals.")
            key = f"{source}.{name}"
            sig = ToolAgent._build_signature(key, tool)
            desc = sig + (f" — {description}" if description else "")
            self._toolbox[source][key] = {"callable": tool, "description": desc}
        elif isinstance(tool, Plugin):
            plugin_name = tool.__class__.__name__
            source = f"__plugin_{plugin_name}__"
            if source in self._toolbox:
                raise RuntimeError(f"Plugin '{plugin_name}' already registered.")
            self._toolbox[source] = {}
            for method_name, meta in tool.method_map().items():
                key = f"{source}.{method_name}"
                sig = ToolAgent._build_signature(key, meta["callable"])
                self._toolbox[source][key] = {
                    "callable": meta["callable"],
                    "description": f"{sig} — {meta['description']}"
                }
        else:
            raise TypeError("Tool must be a callable or Plugin instance.")

    def strategize(self, prompt: str) -> dict:
        # Build method descriptions
        method_lines = []
        for source, methods in self._toolbox.items():
            method_lines.append(f"SOURCE: {source}")
            for name, meta in methods.items():
                method_lines.append(f"- {name}: {meta['description']}")
        available_methods = "\n".join(method_lines)

        # LLM call
        user_prompt = (
            f"AVAILABLE METHODS:\n{available_methods}\n\n"
            f"USER TASK:\n{prompt}\n\n"
            f"Return a single JSON-formatted object for the next step to be executed in the plan"
        )

        raw = Agent.invoke(self, user_prompt)
        raw = re.sub(r"^```[a-zA-Z]*|```$", "", raw.strip())
        step = json.loads(raw)
        return step

    def execute(self, step: dict) -> Any:
        """
        Executes a tool call after resolving any stepN placeholders.
        """
        def _resolve(val: Any) -> Any:
            if isinstance(val, str):
                match = re.fullmatch(r"\{\{step(\d+)\}\}", val)
                if match:
                    return self._previous_steps[int(match.group(1))]["Step_Result"]
                return re.sub(
                    r"\{\{step(\d+)\}\}",
                    lambda m: str(self._previous_steps[int(m.group(1))]["Step_Result"]),
                    val
                )
            elif isinstance(val, list):
                return [_resolve(v) for v in val]
            elif isinstance(val, dict):
                return {k: _resolve(v) for k, v in val.items()}
            return val

        resolved_args = _resolve(step["args"])
        source = step["source"]
        func_key = step["function"]
        func = self._toolbox[source][func_key]["callable"]
        logging.info(f"[TOOL] {func_key} with args:{"".join(f"\n{k}:{v}" for k,v in resolved_args.items())}")
        return func(**resolved_args)

    def step(self, user_input: str) -> tuple[dict, Any]:
        step_strategy = self.strategize(user_input)
        step_call, explanation, status = step_strategy["step_call"], step_strategy["explanation"], step_strategy["status"]
        step_result = self.execute(step_call)
        return explanation, step_result, status

    def invoke(self, prompt: str) -> Any:
        logging.info(f"+---{'-'*len(self.name + ' Starting')}---+")
        logging.info(f"|   {self.name} Starting   |")
        logging.info(f"+---{'-'*len(self.name + ' Starting')}---+")

        self._previous_steps = []
        task_status = "INCOMPLETE"
        result = None
        while task_status != "COMPLETE":
            if self._previous_steps:
                formatted_history = "\n".join(
                    [f"Step {i}: {step['Step_Purpose']} → {step['Step_Result']}" for i, step in enumerate(self._previous_steps)]
                )
                full_prompt = f"{prompt}\n\nPrevious Steps:\n{formatted_history}"
            else:
                full_prompt = prompt
            explanation, result, task_status = self.step(full_prompt)
            self._previous_steps.append({"Step_Purpose": explanation, "Step_Result": result})

        logging.info(f"+---{'-'*len(self.name + ' Finished')}---+")
        logging.info(f"|   {self.name} Finished   |")
        logging.info(f"+---{'-'*len(self.name + ' Finished')}---+\n")
        return result


class AgenticOrchestratorAgent(ToolOrchestratorAgent):
    """
    Extends ToolOrchestratorAgent to support registration of other agents as callable tools.
    Each registered agent gets its own namespace: '__agent_<name>__' with a single '.invoke' method.
    """
    def __init__(self, name: str, description:str, llm_engine: LLMEngine, granular: bool = False):
        super().__init__(name, description=description, llm_engine=llm_engine)
        self._granular = granular
        self.role_prompt = Prompts.AGENTIC_ORCHESTRATOR_PROMPT

    def register(self, tool: Any, description: str | None = None) -> None:
        if isinstance(tool, Agent):
            source = f"__agent_{tool.name}__"
            if source in self._toolbox:
                raise RuntimeError(f"Agent '{tool.name}' is already registered.")
            self._toolbox[source] = {}

            key = f"{source}.invoke"
            sig = ToolAgent._build_signature(key, tool.invoke)
            desc = sig + f" — Agent description: {tool.description}"
            self._toolbox[source][key] = {"callable": tool.invoke, "description": desc}
        elif self._granular:
            super().register(tool, description)
        else:
            raise RuntimeError(f"{self.name} is not configured for granular registration of Plugin's and individual methods")
