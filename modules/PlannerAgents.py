import logging, requests, inspect, json, re
from typing import Any, get_type_hints
from modules.Agents import Agent, ToolAgent, ChainSequenceAgent
from modules.LLMEngines import LLMEngine
import modules.Prompts as Prompts
from modules.PlanExecutors import PlanExecutor, AsyncPlanExecutor
from modules.Plugins import Plugin

# ────────────────────────────────────────────────────────────────
# 1. PlannerAgent class
# ────────────────────────────────────────────────────────────────
class PlannerAgent(ToolAgent):
    """
    Generates and runs an executable plan as a tool,
    with a consistent toolbox: source → {name → {callable, description}}
    """
    def __init__(self, name: str, llm_engine: LLMEngine, is_async=False):
        super().__init__(name, llm_engine, role_prompt=Prompts.PLANNER_PROMPT)

        # initialize toolbox with dev_tools source
        self._toolbox = {"__dev_tools__": {}}

        # register built-in _return under dev_tools
        def _return(val: Any) -> Any:
            return val
        PlannerAgent.register(self, tool = _return, description="Returns the passed-in value, always use at end.")

        # choose executor
        self._executor = AsyncPlanExecutor() if is_async else PlanExecutor()
        self._is_async = is_async

    @property
    def toolbox(self):
        return {src: methods.copy() for src, methods in self._toolbox.items()}

    @property
    def is_async(self):
        return self._is_async
    @is_async.setter
    def is_async(self, val: bool):
        self._is_async = val
        self._executor = AsyncPlanExecutor() if val else PlanExecutor()

    def register(self, tool: Any, description: str = None) -> None:
        """
        Register a callable or Plugin under the appropriate source namespace.
        """
        # Callable functions go under __dev_tools__
        if callable(tool):
            source = "__dev_tools__"
            name = tool.__name__
            if name.startswith("<"):
                raise ValueError("Tools must be named functions, not lambdas or internals")
            key = f"{source}.{name}"
            sig = ToolAgent._build_signature(key, tool)
            desc = f"{sig}"
            if description:
                desc += f" — {description}"
            self._toolbox[source][key] = {"callable": tool, "description": desc}

        # Plugin instances each get their own namespace
        elif isinstance(tool, Plugin):
            plugin_name = tool.__class__.__name__
            source = f"__plugin_{plugin_name}__"
            if source in self._toolbox:
                raise RuntimeError(f"Plugin '{plugin_name}' already registered in '{self.name}'")
            self._toolbox[source] = {}
            for method, meta in tool.method_map().items():
                key = f"{source}.{method}"
                sig = ToolAgent._build_signature(key, meta['callable'])
                desc = f"{sig} — {meta['description']}"
                self._toolbox[source][key] = {"callable": meta['callable'], "description": desc}
        else:
            raise TypeError("Tool must be a callable or Plugin instance")

    def strategize(self, prompt: str) -> dict:
        # build methods block grouped by source
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

        # flatten tools dict for execution
        tools = {key: meta['callable']
                 for src in self._toolbox.values() for key, meta in src.items()}

        # ensure final return
        if not steps or steps[-1]['function'] != '__dev_tools__._return':
            steps.append({'function': '__dev_tools__._return', 'args': {'val': None}})

        return {'steps': steps, 'tools': tools}

    def execute(self, plan: dict) -> Any:
        return self._executor.execute(plan)

    def invoke(self, prompt: str):
        """
        Plan, execute, and return the final result.
        """
        logging.info(f"+---{'-'*len(self.name + ' Starting')}---+")
        logging.info(f"|   {self.name} Starting   |")
        logging.info(f"+---{'-'*len(self.name + ' Starting')}---+")

        plan = self.strategize(prompt)
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
    def __init__(self, name: str, llm_engine: LLMEngine, granular: bool=False, is_async: bool=False):
        super().__init__(name, llm_engine, is_async=is_async)
        self._granular = granular
        # prepare agent namespace
        self._toolbox['__agent_invoke__'] = {}
        self._role_prompt = Prompts.AGENTIC_PLANNER_PROMPT

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
        if not description:
            return ValueError(f"Pleas provide {agent.name} a description for context.")
        source = f"__agent_{agent.name}__"
        if source in self._toolbox:
            raise RuntimeError(f"Agent '{agent.name}' already registered")
        self._toolbox[source] = {}

        key = f"{source}.invoke"
        sig = ToolAgent._build_signature(key, agent.invoke)
        desc = sig + (f" — This method invokes the {agent.name}.\nAgent description: {description}")
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
                 llm_engine: LLMEngine,
                 granular: bool = True,
                 is_async: bool = False):
        super().__init__(name, llm_engine, granular=granular, is_async=is_async)

        # Counter for naming each server wrapper
        self._mcpo_counter = 0
        # Map host_url -> wrapper instance
        self._mcpo_servers: dict[str, McpoServerWrapper] = {}

        # Override to MCPO-specific prompt
        self._role_prompt = Prompts.MCPO_PLANNER_PROMPT

    def register(self, tool: Any, description: str = None) -> None:
        # If given a URL string, treat it as a new MCP-O server
        if isinstance(tool, str):
            host = tool.rstrip('/')
            if host in self._mcpo_servers:
                return  # already registered

            # Fetch and parse OpenAPI
            try:
                openapi = requests.get(f"{host}/openapi.json").json()
            except Exception:
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