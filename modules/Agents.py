import json, re, inspect, logging
from dotenv import load_dotenv
from typing import get_type_hints
from typing import Any
load_dotenv()

# internal imports
from modules.Plugins import *
from modules.PlanExecutors import PlanExecutor, AsyncPlanExecutor
import modules.Prompts as Prompts
from modules.LLMNuclei import *

# ────────────────────────────────────────────────────────────────
# 1.  Agent  (LLM responds to prompts)
# ────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, name, nucleus:LLMNucleus, role_prompt: str = Prompts.DEFAULT_PROMPT, context_enabled: bool = False):
        self._name = name
        self._nucleus: LLMNucleus = nucleus
        self._role_prompt = role_prompt
        self._context_enabled = context_enabled
        self._history = []

    @property
    def name(self):
        return self._name

    @property
    def role_prompt(self):
        return self._role_prompt

    @property
    def context_enabled(self):
        return self.context_enabled

    @property
    def nucleus(self):
        return self._nucleus

    @property
    def history(self):
        return self._history.copy()

    @context_enabled.setter
    def context_enabled(self, value:bool):
        self._context_enabled = value

    @nucleus.setter
    def nucleus(self, value: LLMNucleus):
        self._nucleus = value

    @name.setter
    def name(self, value: str):
        self._name = value

    @role_prompt.setter
    def role_prompt(self, value: str):
        self._role_prompt = value

    def invoke(self, prompt: str) -> str:
        messages = [{"role": "system", "content": self._role_prompt}]
        if self._context_enabled:
            messages.extend(self._history)  # Include previous messages if context is enabled
        messages.append({"role": "user", "content": prompt})
        response = self._nucleus.invoke(messages).strip()
        if self._context_enabled:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": response})
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

    def __init__(self, name: str, nucleus:LLMNucleus, is_async = False):
        super().__init__(name = name, nucleus=nucleus, role_prompt = Prompts.AGENTIC_PLANNER_PROMPT)

        # registries -------------------------------------------------
        self._toolbox: dict[str, dict] = {}
        self._plugin_list: list[str] = []
        self._agent_list: list[str] = []

        # always provide a simple "return" helper
        self.register_tool("return", lambda val: val, "return(val:any) – identity helper for the final step.")
        
        # is_async/executor
        self._is_async = is_async
        self._executor = AsyncPlanExecutor() if is_async else PlanExecutor()

    @property # role prompt cannot be changed in PlannerAgent
    def role_prompt(self):
        return self._role_prompt
    
    @property # planner should not be remembering chat-history
    def context_enabled(self):
        return self._context_enabled
    
    @property # toolbox should not be editable from the outside
    def toolbox(self):
        return self._toolbox
    
    @property
    def plugin_list(self):
        return self._plugin_list
    
    @property
    def is_async(self):
        return self._is_async
    
    @property
    def executor(self):
        return self._executor
    
    @is_async.setter
    def is_async(self, value: bool):
        self._is_async = value
        self._executor = AsyncPlanExecutor() if value else PlanExecutor()
    

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
        self._toolbox[name] = {
            "agent":        agent if agent else "__base_agent__",
            "plugin":       plugin if plugin else "__base_registry__",
            "callable":     func,
            "description":  f"{name}{self._build_signature(func)} — {description}",
            "is_async":     inspect.iscoroutinefunction(func),
        }

    def register_plugin(self, plugin: Plugin) -> None:
        pname = plugin.__class__.__name__
        if pname in self._plugin_list:
            raise ValueError(f"Plugin “{pname}” already registered.")
        self._plugin_list.append(pname)
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
        
        if agent.name in self._agent_list:
            raise ValueError(f"'{agent.name}' already exists in {self.name}'s list of registered agents.")

        self._agent_list.append(agent.name)

        # Determine agent type for description
        if isinstance(agent, PlannerAgent):
            agent_type_desc = (
                f"Invokes the {agent.name} planner agent. "
                "This agent expects a full task prompt (str) describing a complex objective. "
                "It will generate and execute a multi-step plan using its registered tools and return the final result. "
                "Use this when you want the agent to autonomously break down and solve a task."
            )
        elif isinstance(agent, PolymerAgent):
            agent_type_desc = (
                
                f"Invokes the {agent.name} polymer agent. "
                "This agent takes a prompt (str) and passes it through a chain of other agents, each with their own "
                "'invoke' logic, and before passing their results further down the chain, they preprocess their "
                "result through a queue of methods and then send that preprocessed result to the next agent in the chain. "
                "Use this for deterministic, multi-step, tasks that require chains of LLM completions and method calls."
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
            f"-{name}: {meta['description']}" for name, meta in self._toolbox.items()
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

        unknown = [s["function"] for s in steps if s["function"] not in self._toolbox]
        if unknown:
            raise ValueError(f"Plan uses unknown functions: {unknown}")
        
        plan = {"steps": steps, "tools": self._toolbox}
        
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
        logging.info(f"+---{"-"*len(self._name+" Starting")}---+")
        logging.info(f"|   {self._name} Starting   |")
        logging.info(f"+---{"-"*len(self._name+" Starting")}---+")
        
        plan = self.create_plan(prompt)
        logging.info(f"{self._name} created plan with {len(plan['steps'])} steps")
        
        plan_result = self.execute_plan(plan)
        
        logging.info(f"+---{"-"*len(self._name+" Finished")}---+")
        logging.info(f"|   {self._name} Finished   |")
        logging.info(f"+---{"-"*len(self._name+" Finished")}---+\n\n")
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
    def __init__(self, seed: Agent, name: str|None = None):
        if not seed:
            raise ValueError("'seed' argument must be a non-null Agent instance")
        
        # New Polymer inherits all from the seed agent
        self._seed = seed
        
        # If a name is not provided, then name is automatically set to the seed agent's name 
        self._name = name if name else seed.name
        
        # define the head and tails
        self._head:PolymerAgent = None
        self._tail:PolymerAgent = None

        # define the preprocessor list
        self._preprocessor: list[callable] = []

    # should be settable
    @property
    def seed(self):
        return self._seed
    @property
    def preprocessor(self):
        return self._preprocessor.copy()
    @seed.setter
    def seed(self, value: Agent):
        self._seed = value
    @preprocessor.setter # should be capable of being set in batches
    def preprocessor(self, value: list[callable]):
        self._preprocessor = value

    # should not be settable
    @property
    def head(self):
        return self._head
    @property
    def tail(self):
        return self._tail
    
    # should not be mutable at all, only seed values
    @property
    def role_prompt(self):
        return self.seed.role_prompt
    @property
    def nucleus(self):
        return self.seed.nucleus
    @property
    def context_enabled(self):
        return self.seed.context_enabled
    @property
    def history(self):
        return self.seed.history

    # adds a new tool to the preprocessor chain
    def register_tool(self, func: callable, index: int = None):
        # Only allow callables that do not return None
        hints = get_type_hints(func)
        rtype = hints.get('return', Any)
        if rtype is type(None):
            raise ValueError("Preprocessor tool cannot have return type None")
        if index is not None:
            self._preprocessor.insert(index, func)
        else:
            self._preprocessor.append(func)

    # Chains together one agent to the next
    def talks_to(self, agent_b: 'PolymerAgent') -> None:
        if not agent_b:
            raise TypeError("Cannot link a PolymerAgent to a 'NoneType'")
        if not isinstance(agent_b, PolymerAgent):
            raise TypeError("Must link a PolymerAgent to another PolymerAgent")
        # if a tail agent already exists, decouple it
        tail = self._tail
        if tail:
            tail._head = None
        self._tail = agent_b
        agent_b._head = self

        # Cycle detection: traverse from self, check for repeated agents
        visited = set()
        current = self
        while current:
            agent_id = id(current)
            if agent_id in visited:
                # decouple from agent_b
                agent_b._head = None
                # re-attach previous tail
                self._tail = tail
                if tail:
                    tail._head = self
                raise ValueError(f"Cycle detected in attempting to link agent '{self.name}' to agent '{agent_b.name}'")
            visited.add(agent_id)
            current = current._tail
        return
    
    def pop(self, idx=0)->'PolymerAgent':
        def helper(index,current:PolymerAgent):
            # if reached the end, and not at index 0, return error
            if index and not current._tail:
                raise IndexError("Index is larger than the number of available agents")
            # if reached the desired agent:
            if not idx:
                # Decouple from head
                head, tail = current._head, current._tail
                current._head = None
                current._tail = None
                if head:
                    head._tail = tail
                if tail:
                    tail._head = head
                return current
            return helper(index-1, current._tail)
        return helper(idx, self)
    # invoke calls the seed's invoke
    def invoke(self, prompt: str) -> Any:
        # 1. Pass prompt to seed agent
        result = self.seed.invoke(prompt)
        # 2. Pass through preprocessor chain
        for func in self._preprocessor:
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
        # 3. If tail exists, recursively pass stringified output to tail.invoke
        if self._tail:
            # If result is not str, convert to str
            out = str(result)
            # recursively call the tail.invoke() method
            return self._tail.invoke(out)
        else:
            return result