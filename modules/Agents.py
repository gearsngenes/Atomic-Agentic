import os, json, re, inspect, logging
from dotenv import load_dotenv
from openai import OpenAI
from typing import get_type_hints
from typing import Any
load_dotenv()

# internal imports
from modules.Plugins import *
from modules.PlanExecutors import PlanExecutor, AsyncPlanExecutor
import modules.Prompts as Prompts

# ────────────────────────────────────────────────────────────────
# 1.  Agent  (LLM responds to prompts)
# ────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, name, role_prompt: str = Prompts.DEFAULT_PROMPT, llm=None, model: str = "gpt-4o-mini", context_enabled: bool = False):
        self.name = name
        self.llm = llm or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.role_prompt = role_prompt
        self.context_enabled = context_enabled
        self.history = []

    def invoke(self, prompt: str) -> str:
        messages = [{"role": "system", "content": self.role_prompt}]
        if self.context_enabled:
            messages.extend(self.history)  # Include previous messages if context is enabled
        messages.append({"role": "user", "content": prompt})
        response = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )
        response = response.choices[0].message.content.strip()
        if self.context_enabled:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": response})
        return response

# ────────────────────────────────────────────────────────────────
# 2.  Planner Agent  (plans and executes)
# ────────────────────────────────────────────────────────────────
class PlannerAgent(Agent):
    """
    Generates and runs an executable plan as a tool.
    Additionally, can register other agents' invoke()
    methods as tools which it can run`.
    """

    def __init__(self, name: str, model: str = "gpt-4o-mini", is_async = False):
        super().__init__(name = name, role_prompt = Prompts.AGENTIC_PLANNER_PROMPT, model=model)

        # registries -------------------------------------------------
        self.toolbox: dict[str, dict] = {}
        self.plugin_list: list[str] = []

        # always provide a simple "return" helper
        self.register_tool("return", lambda val: val, "return(val:any) – identity helper for the final step.")
        
        # debug / executor
        self.executor = AsyncPlanExecutor() if is_async else PlanExecutor()

    @staticmethod
    def _build_signature(func: callable) -> str:
        sig   = inspect.signature(func)
        hints = get_type_hints(func)
        params = [
            f"{n}: {hints.get(n, Any).__name__}"
            + (f" = {p.default!r}" if p.default is not inspect._empty else "")
            for n, p in sig.parameters.items() if n != "self"
        ]
        rtype = hints.get("return", Any).__name__
        return f"({', '.join(params)}) → {rtype}"

    def register_tool(self, name: str, func: callable, description: str, plugin: str = "", agent: str ="") -> None:
        self.toolbox[name] = {
            "agent":        agent if agent else "__base_agent__",
            "plugin":       plugin if plugin else "__base_registry__",
            "callable":     func,
            "description":  f"{name}{self._build_signature(func)} — {description}",
            "is_async":     inspect.iscoroutinefunction(func),
        }

    def register_plugin(self, plugin: Plugin) -> None:
        pname = plugin.__class__.__name__
        if pname in self.plugin_list:
            raise ValueError(f"Plugin “{pname}” already registered.")
        self.plugin_list.append(pname)
        for method_name, meta in plugin.method_map().items():
            self.register_tool(
                name =          method_name,
                func =          meta["callable"],
                description =   meta["description"],
                plugin =        pname
            )

    def register_agent(self, agent: Agent, description: str | None = None,) -> None:
        """
        Expose `agent.invoke` as a tool called  '<Alias>.invoke'.

        Parameters
        ----------
        agent        : the BasicAgent (or subclass) to expose
        description  : optional extra text appended to auto-description
        """
        # Build signature string "(arg: type, …) → return"
        signature = self._build_signature(agent.invoke)

        # Determine agent type for description
        if isinstance(agent, PlannerAgent):
            agent_type_desc = (
                f"Invokes the {agent.name} planner agent. "
                "This agent expects a full task prompt (str) describing a complex objective. "
                "It will generate and execute a multi-step plan using its registered tools and return the final result. "
                "Use this when you want the agent to autonomously break down and solve a task."
            )
        else:
            agent_type_desc = (
                
                f"Invokes the {agent.name} basic agent. "
                "This agent takes a prompt (str) and generates a text response according to its role-prompt description. "
                "Use this for single-turn LLM completions or simple text generation."
            )

        desc_lines = [
            f"{agent.name}.invoke signature: {signature} -- ",
            agent_type_desc,
            f"Agent description: {description}",
        ]

        self.register_tool(
            name =          f"{agent.name}.invoke",
            func =          agent.invoke, # bound method
            description =   "\n".join(desc_lines),
            agent=          agent.name,
        )

    def create_plan(self, prompt: str) -> dict:
        methods_block = "\n".join(
            f"-{name}: {meta['description']}" for name, meta in self.toolbox.items()
        )

        # 2. Craft user prompt ---------------------------------------
        user_prompt = (
            f"AVAILABLE METHODS\n{methods_block}\n\n"
            f"TASK\n{prompt}\n\n"
            "Respond *only* with the JSON plan."
        )

        # 3. Ask the LLM via BasicAgent ------------------------------
        raw = super().invoke(user_prompt).strip()
        raw = re.sub(r'^```[a-zA-Z]*|```$', '', raw)  # scrub fences if any

        try:
            steps = json.loads(raw)
        except Exception as e:
            raise ValueError(f"JSON parse error: {e}\nLLM output:\n{raw}")

        unknown = [s["function"] for s in steps if s["function"] not in self.toolbox]
        if unknown:
            raise ValueError(f"Plan uses unknown functions: {unknown}")
        
        plan = {"steps": steps, "tools": self.toolbox}
        
        # guarantee last step is "return"
        if not plan["steps"] or plan["steps"][-1]["function"] != "return":
            plan["steps"].append(
                {
                    "function": "return",
                    "args": {"val": None},
                }
            )
        return plan
    
    def execute_plan(self, plan: dict) -> dict:
        return self.executor.execute(plan)
    
    def invoke(self, prompt: str) -> dict:
        """
        Returns
        -------
        { "steps": list[dict], "tools": dict[str, meta] }
        """
        logging.info(f"+---{"-"*len(self.name+" Starting")}---+")
        logging.info(f"|   {self.name} Starting   |")
        logging.info(f"+---{"-"*len(self.name+" Starting")}---+")
        
        plan = self.create_plan(prompt)
        logging.info(f"{self.name} created plan with {len(plan['steps'])} steps")
        
        plan_result = self.execute_plan(plan)
        
        logging.info(f"+---{"-"*len(self.name+" Finished")}---+")
        logging.info(f"|   {self.name} Finished   |")
        logging.info(f"+---{"-"*len(self.name+" Finished")}---+\n\n")
        return plan_result


# ────────────────────────────────────────────────────────────────
# 3.  PolymerAgent  (Invokes a chain of Agents)
# ────────────────────────────────────────────────────────────────
class PolymerAgent(Agent):
    """
    Doubly linked-list analogue of Agent.
    Each PolymerAgent wraps a seed Agent, can be linked to head/tail PolymerAgents,
    and processes outputs through a chain of preprocessor callables.
    """
    def __init__(self, seed: Agent = None, name: str = None, role_prompt: str = None, llm=None, model: str = None, context_enabled: bool = False):
        if not isinstance(seed, Agent):
            raise TypeError("seed must be an Agent instance")
        # initialize the head, tail, and preprocessor list
        self.head:PolymerAgent = None
        self.tail:PolymerAgent = None
        self.preprocessor: list[callable] = []
        # if no seed, create a new Agent with given name and role_prompt
        if not seed:
            self.name = name
            # If no name is provided, raise error
            if not name:
                raise ValueError("Name must be provided if seed is None")
            self.seed = super().__init__(name=name, role_prompt=role_prompt or Prompts.DEFAULT_PROMPT, llm=llm, model=model or "gpt-4o-mini", context_enabled=context_enabled)
        else:
            # Otherwise, use the seed agent's name and initialize seed to input
            self.name:str = seed.name
            self.seed = seed
        

    def register_tool(self, func: callable, index: int = None):
        # Only allow callables that do not return None
        hints = get_type_hints(func)
        rtype = hints.get('return', Any)
        if rtype is type(None):
            raise ValueError("Preprocessor tool cannot have return type None")
        if index is not None:
            self.preprocessor.insert(index, func)
        else:
            self.preprocessor.append(func)


    def sends_to(self, other: 'PolymerAgent'):
        if not isinstance(other, PolymerAgent):
            raise TypeError("other must be a PolymerAgent instance")
        self.tail = other
        other.head = self
        return
    
    
    def invoke(self, prompt: str) -> Any:
        # 1. Pass prompt to seed agent
        result = self.seed.invoke(prompt)
        # 2. Pass through preprocessor chain
        for func in self.preprocessor:
            # Try to match argument count: if func takes >1 arg, pass result as first arg
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            if len(params) == 1:
                result = func(result)
            else:
                # If more than one arg, try to unpack if result is tuple/list
                if isinstance(result, (tuple, list)) and len(result) == len(params):
                    result = func(*result)
                elif isinstance(result, (tuple, list)) and len(result) != len(params):
                    raise ValueError(f"Preprocessor tool {func.__name__} expects {len(params)} args but got {len(result)}")
                else:
                    result = func(result)
        # 3. If tail exists, pass stringified output to tail.invoke
        if self.tail:
            # If result is not str, convert to str
            out = str(result)
            return self.tail.invoke(out)
        else:
            return result