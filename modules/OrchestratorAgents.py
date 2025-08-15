import json, re, logging
from typing import Any
from modules.Agents import Agent, ToolAgent
import modules.Prompts as Prompts
from modules.LLMEngines import LLMEngine

class OrchestratorAgent(ToolAgent):
    def __init__(self, name: str, description: str, llm_engine: LLMEngine, allow_agentic: bool = False, allow_mcp: bool = False):
        """
        A step-by-step orchestrator that generates one JSON step at a time.
        """
        super().__init__(name, description=description, llm_engine=llm_engine, allow_agentic=allow_agentic, allow_mcp=allow_mcp)
        # Enable built-in history/caching of prompts & responses
        self.context_enabled = True
        self.role_prompt = Prompts.ORCHESTRATOR_PROMPT
        self._previous_steps: list[dict] = []

    def strategize(self, prompt: str) -> dict:
        # 1) Build the AVAILABLE METHODS block
        method_lines = []
        for source, methods in self._toolbox.items():
            method_lines.append(f"SOURCE: {source}")
            for name, meta in methods.items():
                method_lines.append(f"- {name}: {meta['description']}")
        available_methods = "\n".join(method_lines)

        # 2) Always use this simple template
        user_prompt = (
            f"AVAILABLE METHODS:\n{available_methods}\n\n"
            f"{prompt}\n\n"
            f"Return a single JSON-formatted object for the next step to be executed in the plan."
        )

        # 3) Delegate to Agent.invoke (which now includes entire history automatically)
        raw = Agent.invoke(self, user_prompt)
        raw = re.sub(r"^```[a-zA-Z]*|```$", "", raw.strip())
        step = json.loads(raw)

        # 4) Sanity check
        assert all(k in step for k in ("step_call", "explanation", "decision_point", "status")), \
            "Returned JSON must include step_call, explanation, decision_point, and status."

        return step

    def execute(self, step: dict) -> Any:
        """
        Runs the chosen tool, resolving any {{stepN}} placeholders.
        """
        resolved_args = self._resolve(step["step_call"]["args"])
        src = step["step_call"]["source"]
        fn_key = step["step_call"]["function"]
        fn = self._toolbox[src][fn_key]["callable"]
        return fn(**resolved_args)

    def step(self, prompt: str) -> tuple[str, Any, str, bool]:
        strat = self.strategize(prompt)
        call = strat["step_call"]
        explanation = strat["explanation"]
        is_decision = strat["decision_point"]
        status = strat["status"]

        logging.info(
            f"[TOOL] {call['function']} args: {call['args']}\nDecision Point? {is_decision}"
        )
        result = self.execute(strat)
        return explanation, result, status, is_decision

    def invoke(self, prompt: str) -> Any:
        """
        Loop, generating one step at a time, until status == "COMPLETE".
        """
        logging.info(f"+---{'-'*len(self.name + ' Starting')}---+")
        logging.info(f"|   {self.name} Starting   |")
        logging.info(f"+---{'-'*len(self.name + ' Starting')}---+")

        # Reset history & steps
        self.clear_memory()
        self._previous_steps = []

        status = "INCOMPLETE"
        last_result = None
        last_decision = False
        iteration = 0

        while status != "COMPLETE":
            if iteration == 0:
                # very first call: feed the raw user task
                sub_prompt = prompt
            else:
                if last_decision:
                    sub_prompt = (
                        f"The result of the previously executed step was: {{step{len(self._previous_steps)-1}}}: {last_result}\n\n"
                        "Given any previously generated and executed steps, generate the next JSON-formatted step needed to complete the user task. Once you've decided what the next step is based on the result value itself, use the double curly-bracket placeholder when passing the result as an argument to the next method/tool. Do NOT pass the raw value itself, but the placeholder instead."
                    )
                else:
                    sub_prompt = (
                        "Given the previously generated and executed steps, generate the next JSON-formatted step needed to complete the user task."
                    )

            explanation, result, status, is_decision = self.step(sub_prompt)

            # Track for the next iteration
            self._previous_steps.append({
                "explanation": explanation,
                "result": result,
                "decision_point": is_decision,
                "completed":     True
            })
            last_result = result
            last_decision = is_decision
            iteration += 1
        self.clear_memory()
        logging.info(f"+---{'-'*len(self.name + ' Finished')}---+")
        logging.info(f"|   {self.name} Finished   |")
        logging.info(f"+---{'-'*len(self.name + ' Finished')}---+\n")

        return last_result
