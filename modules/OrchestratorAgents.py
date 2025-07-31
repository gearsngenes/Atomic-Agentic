import json, re, logging
from typing import Any
from modules.Agents import Agent, ToolAgent
import modules.Prompts as Prompts
from modules.Plugins import Plugin
from modules.LLMEngines import LLMEngine

class OrchestratorAgent(ToolAgent):
    def __init__(self, name: str, description: str, llm_engine: LLMEngine, allow_agentic = False, allow_mcpo = False):
        """
        Initializes context and toolbox.
        """
        super().__init__(name, description=description, llm_engine=llm_engine, allow_agentic=allow_agentic, allow_mcpo=allow_mcpo)
        self.role_prompt = Prompts.ORCHESTRATOR_PROMPT
        self._context_enabled = True
        self._previous_steps = []

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
            f"Return a single JSON-formatted object for the next step to be executed in the plan."
        )

        raw = Agent.invoke(self, user_prompt)
        raw = re.sub(r"^```[a-zA-Z]*|```$", "", raw.strip())
        step = json.loads(raw)

        # Ensure required keys exist
        assert "step_call" in step and "explanation" in step and "decision_point" in step and "status" in step, \
            "Returned step is missing required fields."

        return step


    def execute(self, step: dict) -> Any:
        """
        Executes a tool call after resolving any stepN placeholders.
        """
        resolved_args = self._resolve(step["args"])
        source = step["source"]
        func_key = step["function"]
        func = self._toolbox[source][func_key]["callable"]
        return func(**resolved_args)

    def step(self, user_input: str) -> tuple[str, Any, str, bool]:
        step_strategy = self.strategize(user_input)
        step_call = step_strategy["step_call"]
        explanation = step_strategy["explanation"]
        decision_point = step_strategy["decision_point"]
        status = step_strategy["status"]

        logging.info(f"[TOOL] {step_call['function']} with args: " + "".join(f"\n{k}: {v}" for k, v in step_call["args"].items()) +f"\nDecision Point? {decision_point}")
        step_result = self.execute(step_call)
        return explanation, step_result, status, decision_point


    def invoke(self, prompt: str) -> Any:
        logging.info(f"+---{'-'*len(self.name + ' Starting')}---+")
        logging.info(f"|   {self.name} Starting   |")
        logging.info(f"+---{'-'*len(self.name + ' Starting')}---+")

        self._previous_steps = []
        task_status = "INCOMPLETE"
        result = None

        while task_status != "COMPLETE":
            # Build context: include all step purposes
            explanation_lines = [f"Step {i}: {s['Step_Purpose']}" for i, s in enumerate(self._previous_steps)]
            context = "\n".join(explanation_lines)

            # Determine whether to include result from most recent step
            if self._previous_steps and self._previous_steps[-1]["decision_point"]:
                last = self._previous_steps[-1]
                result_block = (
                    f"\n\nDecision Point Triggered:\n"
                    f"Result of Step {len(self._previous_steps)-1}:\n"
                    f"{last['result']}"
                )
            else:
                result_block = ""

            full_prompt = f"{prompt}\n\nPrevious Steps:\n{context}{result_block}" if context else prompt

            explanation, result, task_status, is_decision_point = self.step(full_prompt)
            self._previous_steps.append({
                "Step_Purpose": explanation,
                "result": result,
                "decision_point": is_decision_point,
                "completed": True
            })

        logging.info(f"+---{'-'*len(self.name + ' Finished')}---+")
        logging.info(f"|   {self.name} Finished   |")
        logging.info(f"+---{'-'*len(self.name + ' Finished')}---+\n")
        return result
