import json, re, logging
from typing import Any
from modules.Agents import Agent, ToolAgent
import modules.Prompts as Prompts
from modules.LLMEngines import LLMEngine

class OrchestratorAgent(ToolAgent):
    def __init__(self, name: str, description: str, llm_engine: LLMEngine, allow_agentic: bool = False, allow_mcp: bool = False, max_context_chars: int = 100_000):
        """
        A step-by-step orchestrator that generates one JSON step at a time.
        It always includes the previous step's (truncated) result in the next strategize prompt.
        """
        super().__init__(
            name,
            description=description,
            llm_engine=llm_engine,
            allow_agentic=allow_agentic,
            allow_mcp=allow_mcp,
        )
        # Enable built-in history/caching of prompts & responses
        self.context_enabled = True
        self.role_prompt = Prompts.ORCHESTRATOR_PROMPT
        self._previous_steps: list[dict] = []
        self.max_context_chars = max_context_chars  # limit for last-result text included in context

    # --- helpers ---
    def _stringify(self, value: Any) -> str:
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return "" if value is None else str(value)

    def _truncate(self, s: str | None) -> str | None:
        if s is None:
            return None
        if len(s) <= self.max_context_chars:
            return s
        return s[: self.max_context_chars]

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

        # 3) Call LLM via Agent.invoke (includes message history when context_enabled)
        raw = Agent.invoke(self, user_prompt)
        raw = re.sub(r"^```[a-zA-Z]*|```$", "", raw.strip())
        step = json.loads(raw)

        # 4) Sanity check (decision_point removed)
        assert all(k in step for k in ("step_call", "explanation", "status")), \
            "Returned JSON must include step_call, explanation, and status."

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

    def step(self, prompt: str) -> tuple[str, Any, str]:
        strat = self.strategize(prompt)
        call = strat["step_call"]
        explanation = strat["explanation"]
        status = strat["status"]

        logging.info(f"[TOOL] {call['function']} args: {call['args']}")
        result = self.execute(strat)
        return explanation, result, status

    def invoke(self, prompt: str) -> Any:
        """
        Loop, generating one step at a time, until status == "COMPLETE".
        Always includes the prior step's (truncated) result in the next prompt.
        """
        logging.info(f"+---{'-'*len(self.name + ' Starting')}---+")
        logging.info(f"|   {self.name} Starting   |")
        logging.info(f"+---{'-'*len(self.name + ' Starting')}---+")

        # Reset history & steps
        self.clear_memory()
        self._previous_steps = []

        status = "INCOMPLETE"
        last_result_text = None
        iteration = 0

        while status != "COMPLETE":
            if iteration == 0:
                # very first call: feed the raw user task
                sub_prompt = (
                    "TASK:\n"
                    f"{prompt}\n\n"
                    "Generate the next JSON-formatted step needed to complete the user task."
                )
            else:
                # Always include a preview of the last step's result (truncated)
                last_idx = len(self._previous_steps) - 1
                preview = self._truncate(last_result_text) if last_result_text is not None else ""
                sub_prompt = (
                    f"TASK:\n{prompt}\n\n"
                    f"Previously executed step index: {last_idx}\n"
                    f"Placeholder for its value: {{step{last_idx}}}\n"
                    f"LAST RESULT PREVIEW (truncated to {self.max_context_chars} chars):\n"
                    f"{preview}\n\n"
                    "Using any previously generated and executed steps, generate the next JSON-formatted step "
                    "needed to complete the user task. If you need to pass a previous result as an argument, "
                    "use the {{stepN}} placeholder rather than inlining the preview text."
                )

            explanation, result, status = self.step(sub_prompt)

            # Track for the next iteration
            self._previous_steps.append({
                "explanation": explanation,
                "result": result,
                "completed": True
            })

            last_result_text = self._stringify(result)
            iteration += 1

        self.clear_memory()
        logging.info(f"+---{'-'*len(self.name + ' Finished')}---+")
        logging.info(f"|   {self.name} Finished   |")
        logging.info(f"+---{'-'*len(self.name + ' Finished')}---+\n")

        # Return the final result (from the last completed step)
        return self._previous_steps[-1]["result"] if self._previous_steps else None
