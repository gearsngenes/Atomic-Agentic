import asyncio, inspect, logging, json, re
from typing import Any
from modules.Agents import Agent, ToolAgent
import modules.Prompts as Prompts
from modules.LLMEngines import LLMEngine

class PlannerAgent(ToolAgent):
    """
    Generates and runs an executable plan as a tool,
    with a consistent toolbox: source → {name → {callable, description}}.
    Executes plans step-by-step, storing prior step results in _previous_steps.
    """
    def __init__(self, name: str, description: str, llm_engine: LLMEngine, is_async=False, allow_agentic = False, allow_mcp = False):
        super().__init__(name, description, llm_engine, role_prompt=Prompts.PLANNER_PROMPT, allow_agentic=allow_agentic, allow_mcp=allow_mcp)
        self._is_async = is_async
        self._previous_steps: list[dict] = []

    @property
    def is_async(self): return self._is_async
    @is_async.setter
    def is_async(self, value: bool): self._is_async = value

    @property
    def description(self):
        return f"~~Planner Agent {self.name}~~\nThis agent decomposes tasks into a list of tool calls.\nDescription: {self._description}"
    @description.setter
    def description(self, val): self._description = val

    def strategize(self, prompt: str) -> dict:
        lines = []
        for src, methods in self._toolbox.items():
            lines.append(f"SOURCE: {src}")
            for key, meta in methods.items():
                lines.append(f"- {key}: {meta['description']}")
        block = "\n".join(lines)

        user_prompt = (
            f"AVAILABLE METHODS (use the exact key names):\n{block}\n\n"
            f"TASK:\n{prompt}\n\n"
            "When JSON-encoding your plan, every 'function' field must exactly match one of the keys above."
        )
        raw = Agent.invoke(self, user_prompt).strip()
        raw = re.sub(r'^```[a-zA-Z]*|```$', '', raw)
        steps = list(json.loads(raw))

        if not steps or steps[-1]['function'] != '__dev_tools__._return':
            steps.append({"function": "__dev_tools__._return", "args": {"val": None}})

        tools = {name: meta['callable'] for methods in self._toolbox.values() for name, meta in methods.items()}
        return {"steps": steps, "tools": tools}

    def execute(self, plan: dict) -> Any:
        return asyncio.run(self.execute_async(plan)) if self._is_async else self._execute_sync(plan)

    def _execute_sync(self, plan: dict) -> Any:
        steps, tools = plan["steps"], plan["tools"]
        for i, step in enumerate(steps):
            fn = tools[step["function"]]
            if inspect.iscoroutinefunction(fn):
                raise RuntimeError(f"Function '{step['function']}' is async — use is_async=True.")
            logging.info(f"[TOOL] {step['function']} args: {step.get("args", {})}")
            args = self._resolve(step.get("args", {}))
            result = fn(**args)
            self._previous_steps[i]["result"] = result
            self._previous_steps[i]["completed"] = True
        return self._previous_steps[-1]["result"]

    async def execute_async(self, plan: dict) -> Any:
        steps, tools = plan["steps"], plan["tools"]

        def get_deps(i):
            return {
                int(n) for n in re.findall(r"step(\d+)", json.dumps(steps[i].get("args", {})))
            }

        async def run_step(i: int):
            step = steps[i]
            fn = tools[step["function"]]
            logging.info(f"[TOOL] {step['function']} args: {step.get("args", {})}")
            args = self._resolve(step.get("args", {}))
            if inspect.iscoroutinefunction(fn):
                return await fn(**args)
            return await asyncio.get_running_loop().run_in_executor(None, lambda: fn(**args))

        remaining = set(range(len(steps)))
        completed = set()

        while remaining:
            ready = [i for i in remaining if get_deps(i) <= completed]
            if not ready:
                raise RuntimeError("Circular dependency in plan.")
            results = await asyncio.gather(*(run_step(i) for i in ready))
            for i, result in zip(ready, results):
                self._previous_steps[i]["result"] = result
                self._previous_steps[i]["completed"] = True
                completed.add(i)
                remaining.remove(i)

        return self._previous_steps[-1]["result"]

    def invoke(self, prompt: str):
        logging.info(f"\n+---{'-'*len(self.name + ' Starting')}---+"
                     f"\n|   {self.name} Starting   |"
                     f"\n+---{'-'*len(self.name + ' Starting')}---+")

        self._previous_steps = []  # reset step history
        plan = self.strategize(prompt)
        self._previous_steps = [{"result": None, "completed": False} for _ in plan["steps"]]

        logging.info(f"{self.name} created plan with {len(plan['steps'])} steps")
        result = self.execute(plan)

        logging.info(   f"\n+---{'-'*len(self.name + ' Finished')}---+"
                        f"\n|   {self.name} Finished   |"
                        f"\n+---{'-'*len(self.name + ' Finished')}---+\n")
        return result

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
        logging.info(f"\n+---{'-'*len(self.name + ' Starting')}---+"
                     f"\n|   {self.name} Starting   |"
                     f"\n+---{'-'*len(self.name + ' Starting')}---+")

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
        logging.info(   f"\n+---{'-'*len(self.name + ' Finished')}---+"
                        f"\n|   {self.name} Finished   |"
                        f"\n+---{'-'*len(self.name + ' Finished')}---+\n")

        # Return the final result (from the last completed step)
        return self._previous_steps[-1]["result"] if self._previous_steps else None
