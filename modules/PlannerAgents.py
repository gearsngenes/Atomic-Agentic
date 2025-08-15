import asyncio, inspect, logging, json, re
from typing import Any
from modules.Agents import ToolAgent, Agent
from modules.LLMEngines import LLMEngine
import modules.Prompts as Prompts

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
        logging.info(f"+---{'-'*len(self.name + ' Starting')}---+")
        logging.info(f"|   {self.name} Starting   |")
        logging.info(f"+---{'-'*len(self.name + ' Starting')}---+")

        self._previous_steps = []  # reset step history
        plan = self.strategize(prompt)
        self._previous_steps = [{"result": None, "completed": False} for _ in plan["steps"]]

        logging.info(f"{self.name} created plan with {len(plan['steps'])} steps")
        result = self.execute(plan)

        logging.info(f"+---{'-'*len(self.name + ' Finished')}---+")
        logging.info(f"|   {self.name} Finished   |")
        logging.info(f"+---{'-'*len(self.name + ' Finished')}---+\n")
        return result