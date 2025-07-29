import asyncio, inspect, logging, json, re, requests
from typing import Any
from modules.Agents import ToolAgent, Agent
from modules.LLMEngines import LLMEngine
from modules.Plugins import Plugin
import modules.Prompts as Prompts

class PlannerAgent(ToolAgent):
    """
    Generates and runs an executable plan as a tool,
    with a consistent toolbox: source → {name → {callable, description}}.
    Executes plans step-by-step, storing prior step results in _previous_steps.
    """
    def __init__(self, name: str, description: str, llm_engine: LLMEngine, is_async=False):
        super().__init__(name, description, llm_engine, role_prompt=Prompts.PLANNER_PROMPT)
        self._toolbox = {"__dev_tools__": {}}
        self._is_async = is_async
        self._previous_steps: list[dict] = []

        def _return(val: Any): return val
        PlannerAgent.register(self, _return, "Returns the passed-in value. Always use this at the end of a plan.")

    @property
    def is_async(self): return self._is_async
    @is_async.setter
    def is_async(self, value: bool): self._is_async = value

    @property
    def description(self):
        return f"~~Planner Agent {self.name}~~\nThis agent decomposes tasks into a list of tool calls.\nDescription: {self._description}"
    @description.setter
    def description(self, val): self._description = val

    def register(self, tool: Any, description: str = None) -> None:
        if callable(tool):
            source = "__dev_tools__"
            name = tool.__name__
            if name.startswith("<"):
                raise ValueError("Tool functions must be named.")
            if not description:
                raise ValueError("Tool functions must have description strings.")
            key = f"{source}.{name}"
            sig = ToolAgent._build_signature(key, tool)
            self._toolbox[source][key] = {"callable": tool, "description": f"{sig} — {description}"}
        elif isinstance(tool, Plugin):
            plugin_name = tool.__class__.__name__
            source = f"__plugin_{plugin_name}__"
            if source in self._toolbox:
                raise RuntimeError(f"Plugin '{plugin_name}' already registered.")
            self._toolbox[source] = {
                f"{source}.{name}": {
                    "callable": meta["callable"],
                    "description": ToolAgent._build_signature(f"{source}.{name}", meta["callable"]) + f" — {meta['description']}"
                }
                for name, meta in tool.method_map().items()
            }
        else:
            raise TypeError("Only functions or Plugin instances can be registered.")

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
        steps = json.loads(raw)

        if not steps or steps[-1]['function'] != '__dev_tools__._return':
            steps.append({"function": "__dev_tools__._return", "args": {"val": None}})

        tools = {name: meta['callable'] for methods in self._toolbox.values() for name, meta in methods.items()}
        return {"steps": steps, "tools": tools}

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

# ────────────────────────────────────────────────────────────────
# 2. AgenticPlannerAgent class
# ────────────────────────────────────────────────────────────────
class AgenticPlannerAgent(PlannerAgent):
    """
    Extends PlannerAgent to allow registration of other agents' invoke methods
    as first-class tools under __agent_<AgentName>__ namespaces.
    """
    def __init__(self, name: str, description: str, llm_engine: LLMEngine, granular: bool=False, is_async: bool=False):
        super().__init__(name, description, llm_engine, is_async=is_async)
        self._granular = granular
        self._role_prompt = Prompts.AGENTIC_PLANNER_PROMPT

    @property
    def description(self):
        desc = f"~~AgenticPlanner Agent {self.name}~~\nThis agent decomposes input task prompts into subtasks for other agents and/or calls to tool methods.\nDescription: {self._description}"
        return desc
    @description.setter
    def description(self, value: str):
        self._description = value

    @property
    def granular(self):
        return self._granular

    @granular.setter
    def granular(self, val: bool):
        self._granular = val

    def register(self, tool:Any, description: str=None) -> None:
        if not isinstance(tool, Agent):
            if self.granular:
                return super().register(tool, description)
            raise RuntimeError(f"Not configured for granular registration of methods and plugins")
        agent = tool
        source = f"__agent_{agent.name}__"
        if source in self._toolbox:
            raise RuntimeError(f"Agent '{agent.name}' already registered")
        self._toolbox[source] = {}

        key = f"{source}.invoke"
        sig = ToolAgent._build_signature(key, agent.invoke)
        desc = sig + (f" — Agent description: {agent.description}")
        self._toolbox[source][key] = {"callable": agent.invoke, "description": desc}

# ────────────────────────────────────────────────────────────────
# 3. MCPO PlannerAgent class
# ────────────────────────────────────────────────────────────────
class McpoServerWrapper:
    """
    Wrapper for a single MCP-OpenAPI server. 
    Holds host URL and path descriptions, and exposes mcpo_invoke().
    """
    def __init__(self, name: str, host_url: str, openapi: dict):
        self.name = name
        self.host_url = host_url.rstrip('/')
        # Extract each path’s description from the OpenAPI spec
        self.paths: dict[str, str] = {
            path: meta.get('post', {}).get('description', '')
            for path, meta in openapi.get('paths', {}).items()
        }

    def mcpo_invoke(self, path: str, payload: dict) -> Any:
        url = f"{self.host_url}{path}"
        resp = requests.post(url, json=payload)
        return resp.json()


class McpoPlannerAgent(AgenticPlannerAgent):
    """
    Extends AgenticPlannerAgent to register MCP-OpenAPI servers under
    their own __mcpo_server_i__ namespace, each with a single mcpo_invoke tool.
    """
    def __init__(self,
                 name: str,
                 description: str,
                 llm_engine: LLMEngine,
                 granular: bool = True,
                 is_async: bool = False):
        super().__init__(name, description, llm_engine, granular=granular, is_async=is_async)

        # Counter for naming each server wrapper
        self._mcpo_counter = 0
        # Map host_url -> wrapper instance
        self._mcpo_servers: dict[str, McpoServerWrapper] = {}

        # Override to MCPO-specific prompt
        self._role_prompt = Prompts.MCPO_PLANNER_PROMPT

    @property
    def description(self):
        desc = f"~~MCPO-Planner Agent {self.name}~~\nThis agent decomposes input task prompts into subtasks for other agents, calls to locally registered tool methods, and/or calls to MCPO Tool servers for remotely accessible tools.\nDescription: {self._description}"
        return desc
    @description.setter
    def description(self, value: str):
        self._description = value

    def register(self, tool: Any, description: str = None) -> None:
        # If given a URL string, treat it as a new MCP-O server
        if isinstance(tool, str):
            host = tool.rstrip('/')
            if host in self._mcpo_servers:
                return  # already registered

            # Fetch and parse OpenAPI
            try:
                openapi = requests.get(f"{host}/openapi.json").json()
            except ValueError:
                raise ValueError(f"Invalid MCP-O server URL: {tool}")

            # Instantiate wrapper under a unique namespace
            self._mcpo_counter += 1
            name = f"__mcpo_server_{self._mcpo_counter}__"
            wrapper = McpoServerWrapper(name, host, openapi)
            self._mcpo_servers[host] = wrapper

            # Register its single invoke tool
            source = wrapper.name
            key = f"{source}.mcpo_invoke"
            sig = ToolAgent._build_signature(key, wrapper.mcpo_invoke)
            desc = (
                f"{sig} — Calls the '.mcpo_invoke' method for the {wrapper.name} server.\n"
                "Use exactly one of these paths and payload schemas:\n"
                + "".join(
                    f"  - {path}: {pdesc}\n"
                    for path, pdesc in wrapper.paths.items()
                )
            )
            self._toolbox[source] = {
                key: {"callable": wrapper.mcpo_invoke, "description": desc}
            }

            # Prevent falling through to the base register
            return

        # Otherwise, fall back to AgenticPlannerAgent’s register (agents/plugins)
        super().register(tool, description)