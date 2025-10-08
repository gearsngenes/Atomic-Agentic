import asyncio, inspect, logging, json, re
from typing import Any, get_type_hints
from modules.Agents import Agent
import modules.Prompts as Prompts
from modules.LLMEngines import LLMEngine
from abc import ABC, abstractmethod
from modules.Tools import Tool, ToolFactory

class ToolAgent(Agent, ABC):
    def __init__(self, name, description, llm_engine, role_prompt = Prompts.DEFAULT_PROMPT):
        super().__init__(name, description, llm_engine, role_prompt, context_enabled = False)
        # Flattened toolbox: ordered list of Tool objects. Grouping by type/source
        # is computed on-demand by toolbox_by_type().
        self._toolbox: list[Tool] = []
        self._previous_steps: list[dict] = []
        self._mcpo_servers = {}  # optional for MCP, added only if enabled
        self._mcpo_counter = 0
        self._mcp_counter = 0
        def _return(val: Any): return val
        self.register(_return, "Returns the passed-in value. Always use this at the end of a plan.")

    def _resolve(self, obj: Any) -> Any:
        """
        Recursively resolve {{stepN}} references using self._previous_steps.
        Ensures the referenced step is completed before use.
        """
        if isinstance(obj, str):
            match = re.fullmatch(r"\{\{step(\d+)\}\}", obj)
            if match:
                idx = int(match.group(1))
                if idx >= len(self._previous_steps) or not self._previous_steps[idx]["completed"]:
                    raise RuntimeError(f"Step {idx} has not been completed yet.")
                return self._previous_steps[idx]["result"]

            return re.sub(
                r"\{\{step(\d+)\}\}",
                lambda m: str(self._previous_steps[int(m.group(1))]["result"])
                if self._previous_steps[int(m.group(1))]["completed"]
                else RuntimeError(f"Step {m.group(1)} has not been completed yet."),
                obj
            )

        if isinstance(obj, list):
            return [self._resolve(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._resolve(v) for k, v in obj.items()}
        return obj
    @abstractmethod
    def strategize(self, prompt:str)->dict:
        pass
    @abstractmethod
    def execute(self, plan:list[dict])->Any:
        pass
    
    def get_actions_context(self) -> str:
        """Return a textual description of available tools formatted for prompts.

        This preserves the old layout used by the planner/orchestrator prompts by
        grouping tools by type then source.
        """
        # Group tools by type -> source -> [tools] keeping insertion order
        grouped: dict[str, dict[str, list[Tool]]] = {}
        for tool in self._toolbox:
            grouped.setdefault(tool.type, {})
            grouped[tool.type].setdefault(tool.source, []).append(tool)

        context = ""
        for action_type, type_dict in grouped.items():
            context += f"ACTION TYPE: {action_type}s\n"
            for source, tool_list in type_dict.items():
                context += f"- SOURCE: {source}\n"
                for tool in tool_list:
                    context += f"  - {tool.signature}: {tool.description}\n"
            context += "\n"
        return context
    
    def list_tools(self) -> dict:
        """Return a mapping of full tool key -> callable.

        Later-registered tools override earlier ones when keys collide.
        """
        tools: dict[str, Any] = {}
        for tool in self._toolbox:
            tools[tool.full_name] = tool.func
        return tools
    
    def register(self, tool: Any, name: str|None = None, description: str | None = None) -> None:
        # Accept a ready-made Tool instance
        if isinstance(tool, Tool):
            self._toolbox.append(tool)
            return tool.full_name

        # Determine allowed non-Tool types (we keep same external API inputs)
        if callable(tool):
            _type = "function"
        elif isinstance(tool, dict) and "method_map" in tool and "name" in tool:
            _type = "plugin"
        elif isinstance(tool, Agent):
            _type = "agent"
        elif isinstance(tool, str) and tool.endswith("/mcp"):
            _type = "mcp"
        else:
            raise ValueError("Tool must be a callable, Plugin, Agent, or MCP server URL string (if MCP registration is allowed).")

        tools: list[Tool] = ToolFactory.toolify(object=tool, name=name, description=description)
        if not tools:
            return None

        # Append all produced Tool objects. We intentionally allow multiple
        # registrations for the same source; lookup will resolve last-wins.
        self._toolbox.extend(tools)

        # Return a helpful reference string similar to prior behavior.
        if _type == "function":
            # return like: function.default.<method_name>
            return f"function.default.{tools[0].name}"
        else:
            # return like: plugin.<source> or agent.<source> or mcp.<source>
            return f"{tools[0].type}.{tools[0].source}"

    def invoke(self, prompt):
        raise NotImplementedError("ToolAgent is abstract; use strategize() and execute() instead.")

    @property # toolbox should not be editable from the outside
    def toolbox(self):
        # Return a shallow copy of the internal Tool list
        return list(self._toolbox)
    def toolbox_by_type(self) -> dict:
        """Compatibility helper: produce the legacy nested dict view
        { type: { source: [Tool, ...], ... }, ... }
        """
        grouped: dict[str, dict[str, list[Tool]]] = {}
        for tool in self._toolbox:
            grouped.setdefault(tool.type, {})
            grouped[tool.type].setdefault(tool.source, []).append(tool)
        return grouped

class PlannerAgent(ToolAgent):
    def __init__(self, name: str, description: str, llm_engine: LLMEngine, context_enabled = False, is_async=False):
        super().__init__(name = name, description=description, llm_engine=llm_engine, role_prompt=Prompts.PLANNER_PROMPT)
        self.context_enabled = context_enabled
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
        # 1) Build the AVAILABLE METHODS block from Tool objects
        block = self.get_actions_context()

        user_prompt = (
            f"TASK:\n{prompt}\n\n"
            "When JSON-encoding your plan, every 'function' field must exactly match one of the keys above."
        )

        # 2) Ask the LLM for a full plan (array of steps)
        messages = [
            {"role": "system", "content": f"{self.role_prompt}\n\nBelow are the available tools you can decompose a user's task into:\n{block}"},
        ]
        if self.context_enabled:
            messages += self._history
        messages.append({"role": "user", "content": user_prompt})
        raw = self._llm_engine.invoke(messages = messages, file_paths = self._file_paths)
        raw = re.sub(r'^```[a-zA-Z]*|```$', '', raw)
        self._history.append({"role": "assistant", "content": raw})
        steps = list(json.loads(raw))
        # 3) Ensure last step is the canonical return tool
        if not steps or steps[-1].get('function') != 'function.default._return':
            steps.append({"function": "function.default._return", "args": {"val": None}})

        return steps


    def execute(self, plan: list[dict]) -> Any:
        self._previous_steps = [{"result": None, "completed": False} for _ in plan]
        return asyncio.run(self.execute_async(plan)) if self._is_async else self._execute_sync(plan)

    def _execute_sync(self, steps: list[dict]) -> Any:
        tools = self.list_tools()
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

    async def execute_async(self, steps: list[dict]) -> Any:
        tools = self.list_tools()

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
        plan_raw = self._history[-1]["content"] if self._history else ""
        self._history = self._history[:-1]
        logging.info(f"{self.name} created plan with {len(plan)} steps")
        result = self.execute(plan)
        if self.context_enabled:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": f"Generated plan:\n{plan_raw}\nExecuted Result: {str(result)}"})
        self._previous_steps = []
        logging.info(   f"\n+---{'-'*len(self.name + ' Finished')}---+"
                        f"\n|   {self.name} Finished   |"
                        f"\n+---{'-'*len(self.name + ' Finished')}---+\n")
        return result

class OrchestratorAgent(ToolAgent):
    def __init__(self, name: str, description: str, llm_engine: LLMEngine,
                 context_enabled=False, max_context_chars: int = 100_000):
        super().__init__(name, description=description, llm_engine=llm_engine)
        self.context_enabled = context_enabled
        self.role_prompt = Prompts.ORCHESTRATOR_PROMPT
        self._previous_steps: list[dict] = []
        self._running_plan: list[dict] = []  # list of prior step JSONs (compact)
        self.max_context_chars = max_context_chars

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
        return s if len(s) <= self.max_context_chars else s[: self.max_context_chars]

    def strategize(self, prompt: str) -> list[dict]:
        """
        Ask for exactly ONE next-step JSON object using the unified key format.
        """
        available = self.get_actions_context()
        last_result = self._truncate(self._stringify(self._previous_steps[-1]["result"])) if self._previous_steps else None

        sys = {
            "role": "system",
            "content": f"{self.role_prompt}\n\nAVAILABLE METHODS (exact keys):\n{available}"
        }
        msgs = [sys]
        if self.context_enabled:
            msgs += self._history
        user = {
            "role": "user",
            "content": (
                "Generate the next single JSON object needed to complete the user's task.\n"
                f"TASK: {prompt}\n"
                f"RUNNING PLAN SO FAR: {json.dumps(self._running_plan, ensure_ascii=False)}\n"
                + (f"LAST RESULT (truncated): {last_result}\n" if last_result is not None else "")
            )
        }
        msgs.append(user)

        raw = self._llm_engine.invoke(messages=msgs, file_paths=self._file_paths).strip()
        raw = re.sub(r"^```[a-zA-Z]*|```$", "", raw)
        step = json.loads(raw)

        # Validate
        assert all(k in step for k in ("step_call", "explanation", "status")), \
            "Returned JSON must include step_call, explanation, and status."
        assert "function" in step["step_call"] and "args" in step["step_call"], \
            "step_call must include function and args."

        # Memoize compact plan
        compact = {
            "function": step["step_call"]["function"],
            "args": step["step_call"]["args"],
            "status": step["status"]
        }
        self._running_plan.append(compact)
        return [step]

    def execute(self, step_list: list[dict]) -> Any:
        """
        Executes the single chosen tool call, resolving any {{stepN}} placeholders.
        """
        step = step_list[0]
        fn_key = step["step_call"]["function"]
        tools = self.list_tools()
        fn = tools.get(fn_key)
        if fn is None:
            available = ", ".join(sorted(tools.keys()))
            raise KeyError(f"Unknown tool key '{fn_key}'. Available keys: {available}")

        logging.info(f"[TOOL] {fn_key} args: {step['step_call'].get('args', {})}")
        args = self._resolve(step["step_call"].get("args", {}))
        if inspect.iscoroutinefunction(fn):
            # orchestrator executes synchronously; disallow accidental async
            raise RuntimeError(f"Function '{fn_key}' is async — switch to an async orchestrator if needed.")
        return fn(**args)

    def step(self, prompt: str) -> tuple[str, Any, str]:
        strat = self.strategize(prompt)[0]
        explanation = strat["explanation"]
        status = strat["status"]
        result = self.execute([strat])
        return explanation, result, status

    def invoke(self, prompt: str) -> Any:
        """
        Generate/execute one step at a time until status == 'COMPLETE'.
        """
        logging.info(f"\n+---{'-'*len(self.name + ' Starting')}---+"
                     f"\n|   {self.name} Starting   |"
                     f"\n+---{'-'*len(self.name + ' Starting')}---+")
        if not self.context_enabled:
            self.clear_memory()
        self._previous_steps.clear()
        self._running_plan.clear()

        status = "INCOMPLETE"
        while status != "COMPLETE":
            explanation, result, status = self.step(prompt)
            self._previous_steps.append({"explanation": explanation, "result": result, "completed": True})

        if not self.context_enabled:
            self.clear_memory()

        logging.info(f"\n+---{'-'*len(self.name + ' Finished')}---+"
                     f"\n|   {self.name} Finished   |"
                     f"\n+---{'-'*len(self.name + ' Finished')}---+\n")
        return self._previous_steps[-1]["result"] if self._previous_steps else None
