import json, re, logging
from typing import Any
from modules.Agents import Agent, ToolAgent
import modules.Prompts as Prompts
from modules.Plugins import Plugin
from modules.LLMEngines import LLMEngine

class ToolOrchestratorAgent(ToolAgent):
    def __init__(self, name: str, llm_engine: LLMEngine):
        """
        Initializes context and toolbox.
        """
        super().__init__(name, llm_engine)
        self.role_prompt=Prompts.ORCHESTRATOR_PROMPT
        def _return(val:Any):
            return val
        self._previous_steps = []
        self.register(_return, "Use this method to return a specific value object or string if the user specifically requests it")

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

        # clean + parse response
        raw = Agent.invoke(self, user_prompt)
        raw = re.sub(r"^```[a-zA-Z]*|```$", "", raw.strip())
        step = json.loads(raw)
        # print(json.dumps(step, indent = 2))
        return step

    def execute(self, step: dict) -> Any:
        """
        Executes a tool call after resolving any stepN placeholders.
        """
        def _resolve(val: Any) -> Any:
            # Full replacement: "{{stepN}}"
            if isinstance(val, str):
                match = re.fullmatch(r"\{\{step(\d+)\}\}", val)
                if match:
                    return self._previous_steps[int(match.group(1))]["Step_Result"]
                # Partial substitution: e.g. "The answer is {{step0}}"
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
        return func(**resolved_args)

    def step(self, user_input: str) -> tuple[dict, Any]:
        """
        Combines strategize + execute
        """
        step_strategy = self.strategize(user_input)
        step_call, explanation, status = step_strategy["step_call"], step_strategy["explanation"], step_strategy["status"]
        step_result = self.execute(step_call)
        return explanation, step_result, status

    def invoke(self, prompt: str) -> Any:
        self._previous_steps = []
        task_status = "INCOMPLETE"
        result = None
        while task_status != "COMPLETE":
            # Reformat prompt to include history
            if self._previous_steps:
                formatted_history = "\n".join(
                    [f"Step {i}: {step['Step_Purpose']} → {step['Step_Result']}" for i, step in enumerate(self._previous_steps)]
                )
                full_prompt = f"{prompt}\n\nPrevious Steps:\n{formatted_history}"
            else:
                full_prompt = prompt
            explanation, result, task_status = self.step(full_prompt)
            self._previous_steps.append({"Step_Purpose": explanation, "Step_Result": result})
        return result
