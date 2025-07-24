import json, re, inspect, logging
from dotenv import load_dotenv
from typing import get_type_hints
from typing import Any
load_dotenv()

# internal imports
from modules.Plugins import *
from modules.PlanExecutors import PlanExecutor, AsyncPlanExecutor
import modules.Prompts as Prompts
from modules.LLMEngines import *

# ────────────────────────────────────────────────────────────────
# 1.  Agent  (LLM responds to prompts)
# ────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, name, llm_engine:LLMEngine, role_prompt: str = Prompts.DEFAULT_PROMPT, context_enabled: bool = False):
        self._name = name
        self._llm_engine: LLMEngine = llm_engine
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
    def llm_engine(self):
        return self._llm_engine

    @property
    def history(self):
        return self._history.copy()

    @context_enabled.setter
    def context_enabled(self, value:bool):
        self._context_enabled = value

    @llm_engine.setter
    def llm_engine(self, value: LLMEngine):
        self._llm_engine = value

    @name.setter
    def name(self, value: str):
        self._name = value

    @role_prompt.setter
    def role_prompt(self, value: str):
        self._role_prompt = value

    def invoke(self, prompt: str):
        messages = [{"role": "system", "content": self._role_prompt}]
        if self._context_enabled:
            messages.extend(self._history)  # Include previous messages if context is enabled
        messages.append({"role": "user", "content": prompt})
        response = self._llm_engine.invoke(messages).strip()
        if self._context_enabled:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": response})
        return response

# ───────────────────────────────────────────────────────────────────────────────
# 2.  PrePostAgent  (calls methods to preprocess and postprocess an Agent's output
# ───────────────────────────────────────────────────────────────────────────────
class PrePostAgent(Agent):
    def __init__(self, name, llm_engine, role_prompt = Prompts.DEFAULT_PROMPT, context_enabled = False):
        Agent.__init__(self, name, llm_engine, role_prompt, context_enabled)
        self._preprocessors: list[callable] = []
        self._postprocessors: list[callable] = []
    
    # adds a new tool to the preprocessor chain
    def add_prestep(self, func: callable, index: int = None):
        # Only allow callables that do not return None
        hints = get_type_hints(func)
        rtype = hints.get('return', Any)
        if rtype is type(None):
            raise ValueError("Preprocessor tool cannot have return type None")
        if index is not None:
            self._preprocessors.insert(index, func)
        else:
            self._preprocessors.append(func)

    # adds a new tool to the postprocessor chain
    def add_poststep(self, func: callable, index: int = None):
        # Only allow callables that do not return None
        hints = get_type_hints(func)
        rtype = hints.get('return', Any)
        if rtype is type(None):
            raise ValueError("Preprocessor tool cannot have return type None")
        if index is not None:
            self._postprocessors.insert(index, func)
        else:
            self._postprocessors.append(func)
    def invoke(self, prompt: str):
        # 1. Pass prompt through preprocessor chain
        preprocessed = prompt
        for func in self._preprocessors:
            # Try to match argument count: if func takes >1 arg, pass result as first arg
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            if len(params) == 1:
                preprocessed = func(preprocessed)
            else:
                # If more than one arg, try to unpack if result is tuple/list
                if isinstance(preprocessed, (tuple, list)) and len(preprocessed) == len(params):
                    preprocessed = func(*preprocessed)
                elif isinstance(preprocessed, (tuple, list)) and len(preprocessed) != len(params):
                    raise ValueError(f"Preprocessor tool {func.__name__} expects {len(params)} args but got {len(preprocessed)}")
                else:
                    preprocessed = func(preprocessed)
        # 2. pass preprocessed result through the LLM
        processed = Agent.invoke(self, str(preprocessed))
        # 3. pass the processed prompt through the postprocessors
        postprocessed = processed
        for func in self._postprocessors:
            # Try to match argument count: if func takes >1 arg, pass result as first arg
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            if len(params) == 1:
                postprocessed = func(postprocessed)
            else:
                # If more than one arg, try to unpack if result is tuple/list
                if isinstance(postprocessed, (tuple, list)) and len(postprocessed) == len(params):
                    postprocessed = func(*postprocessed)
                elif isinstance(postprocessed, (tuple, list)) and len(postprocessed) != len(params):
                    raise ValueError(f"Preprocessor tool {func.__name__} expects {len(params)} args but got {len(preprocessed)}")
                else:
                    postprocessed = func(postprocessed)
        # 4. return post-processed result
        return postprocessed

    @property
    def preprocessors(self):
        return self._preprocessors.copy()
    @preprocessors.setter # should be capable of being set in batches
    def preprocessor(self, value: list[callable]):
        self._preprocessors = value
    @property
    def postprocessors(self):
        return self._postprocessors.copy()
    @postprocessors.setter # should be capable of being set in batches
    def postprocessors(self, value: list[callable]):
        self._postprocessors = value


# ────────────────────────────────────────────────────────────────
# 3.  ChainSequenceAgent  (Invokes a chain of Agents)
# ────────────────────────────────────────────────────────────────
class ChainSequenceAgent(Agent):
    """
    A sequential Chain-of-Agents. Uses a flat internal list.
    Each agent's output is passed as input to the next.
    """
    def __init__(self, name: str | None = None, context_enabled: bool = False):
        self._agents:list[Agent] = []
        self._name = name or "ChainSequence"
        self._context_enabled = context_enabled
        self._role_prompt = ""
        self._history = []
        self._llm_engine = None
    @property
    def agents(self) -> list[Agent]:
        return self._agents

    @property
    def role_prompt(self):
        desc = "You are a chain-sequence agent. You sequentially invoke the following agents:\n"
        return desc + "\n".join(f"- {agent.name}" for agent in self._agents)

    @property
    def llm_engine(self):
        return self._agents[0].llm_engine if self._agents else None
    
    def add(self, agent:Agent, idx:int|None = None):
        if idx != None:
            self._agents.insert(idx, agent)
        else:
            self._agents.append(agent)
    
    def pop(self, idx:int|None = None) -> Agent:
        if idx != None:
            return self._agents.pop(idx)
        return self._agents.pop()
    
    def invoke(self, prompt: str):
        result = prompt
        if not self._agents:
            raise RuntimeError(f"Agents list is empty for ChainSequenceAgent '{self.name}'")
        for agent in self._agents:
            result = agent.invoke(str(result))
        if self._context_enabled:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": str(result)})
        return result

# ────────────────────────────────────────────────────────────────
# 4.  PlannerAgent  (plans and executes)
# ────────────────────────────────────────────────────────────────
from abc import abstractmethod
class _PlannerAgent(Agent):
    def __init__(self, name, llm_engine, role_prompt = Prompts.DEFAULT_PROMPT):
        super().__init__(name, llm_engine, role_prompt, context_enabled = False)
    @abstractmethod
    def create_plan(self, prompt:str)->dict:
        pass
    @abstractmethod
    def execute_plan(self, plan:dict)->Any:
        pass
    @abstractmethod
    def register(self, tool: Any, description: str|None = None) -> None:
        pass

class PlannerAgent(_PlannerAgent):
    """
    Generates and runs an executable plan as a tool.
    Additionally, can register other agents' invoke()
    methods as tools which it can run`.
    """

    def __init__(self, name: str, llm_engine:LLMEngine, is_async = False):
        super().__init__(name = name, llm_engine=llm_engine, role_prompt = Prompts.PLANNER_PROMPT)

        # Toolbox
        self._toolbox: dict[str, dict[str, dict]] = {"dev_tools":{}, "plugin_tools":{}}
        # always provide a simple "return" helper
        def _return(val:Any)->Any:
            return val
        PlannerAgent.register(self,_return, ("This returns whatever value is passed in. "
                                 "Always call '_return' at the END of any plan, returning 'null' by default "
                                 "if the user does not ask for a specific result or result type. Otherwise, "
                                 "call _return for the result of a step specified by the user's task."))
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
    def _build_signature(name: str, func: callable) -> str:
        sig   = inspect.signature(func)
        hints = get_type_hints(func)
        params = [
            f"{n}: {hints.get(n, Any).__name__}"
            + (f" = {p.default!r}" if p.default is not inspect._empty else "")
            for n, p in sig.parameters.items() if n != "self"
        ]
        rtype = hints.get("return", Any).__name__
        signature = f"{name}({', '.join(params)}) → {rtype}"
        return signature

    def register(self, tool: Any, description: str|None = None) -> None:
        if callable(tool):
            if tool.__name__ == "<lambda>":
                raise ValueError("Tools must be defined as named methods, not lambda statements")
            self._toolbox["dev_tools"][tool.__name__] = {
                "callable":     tool,
                "description":  f"{self._build_signature(tool.__name__, tool)} — {description}",
                "is_async":     inspect.iscoroutinefunction(tool),
            }
        elif isinstance(tool, Plugin):
            plugin = tool
            pname = plugin.__class__.__name__
            if pname in self._toolbox["plugin_tools"].keys():
                raise ValueError(f"Plugin “{pname}” already registered.")
            self._toolbox["plugin_tools"][pname] = {}
            plugin_toolbox = self._toolbox["plugin_tools"][pname] 
            for _name, meta in plugin.method_map().items():
                func, description = meta["callable"], meta["description"] 
                plugin_toolbox[f"{pname}.{_name}"] = {
                    "callable":     func,
                    "description":  f"{self._build_signature(f"{pname}.{_name}", func)} — {description}",
                    "is_async":     inspect.iscoroutinefunction(func),
                }
        else:
            raise TypeError("Tool must be of type 'callable' or 'plugin'.")

    def create_plan(self, prompt: str) -> dict:
        developer_methods = (
            "DEVELOPER-REGISTERED METHODS:"
            "\n* ".join(meta['description'] for _, meta in self._toolbox["dev_tools"].items())
        )
        tools = {name:func["callable"] for name, func in self._toolbox["dev_tools"].items()}
        
        plugin_methods = ("\nPLUGIN-REGISTERED METHODS\n")
        for _plugin, _tools in self._toolbox["plugin_tools"].items():
            plugin_methods += f"* PLUGIN: {_plugin}:{"".join(f"\n\t** {meta['description']}" for meta in _tools.values())}"
            tools.update({_name:meta["callable"] for _name,meta in _tools.items()})
        
        methods_block = f"{developer_methods}\n{plugin_methods}"
        # 2. Craft user prompt ---------------------------------------
        user_prompt = (
            f"AVAILABLE METHODS\n{methods_block}\n\n"
            f"TASK\n{prompt}\n\n"
            "Respond *only* with the JSON plan."
        )

        # 3. Ask the LLM via BasicAgent ------------------------------
        raw = Agent.invoke(self, user_prompt).strip()
        raw = re.sub(r'^```[a-zA-Z]*|```$', '', raw)  # scrub fences if any

        try:
            steps = json.loads(raw)
        except Exception as e:
            raise ValueError(f"JSON parse error: {e}\nLLM output:\n{raw}")

        unknown = [s["function"] for s in steps if s["function"] not in tools]
        if unknown:
            raise ValueError(f"Plan uses unknown functions: {unknown}")
        # print(f"STEPS GENERATED:\n{"".join([f"Step {i}: {(step['function'])}, args: {step["args"]}\n" for i, step in enumerate(steps)])}")
        plan = {"steps": steps, "tools": tools}
        
        # guarantee last step is "_return"
        if not plan["steps"] or plan["steps"][-1]["function"] != "return":
            plan["steps"].append(
                {
                    "function": "_return",
                    "args": {"val": None},
                }
            )
        return plan
    
    def execute_plan(self, plan: dict) -> dict:
        return self.executor.execute(plan)
    
    def invoke(self, prompt: str):
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

class AgenticPlannerAgent(PlannerAgent):
    """
    Generates and runs an executable plan as a tool.
    Additionally, can register other agents' invoke()
    methods as tools which it can run`.
    """

    def __init__(self, name: str, llm_engine:LLMEngine, granular:bool = False, is_async = False):
        self._granular = granular
        PlannerAgent.__init__(self, name = name, llm_engine=llm_engine, is_async=is_async)
        self._toolbox.update({"agents":{}})
        # update roleprompt
        self._role_prompt = Prompts.AGENTIC_PLANNER_PROMPT
    @property
    def granular(self):
        return self._granular
    @granular.setter
    def granular(self, value:bool):
        self._granular = value
    def register(self, tool: Any, description: str | None = None,) -> None:
        """
        Expose `agent.invoke` as a tool called  '<Alias>.invoke'.

        Parameters
        ----------
        agent        : the BasicAgent (or subclass) to expose
        description  : optional extra text appended to auto-description
        """
        # Build signature string "(arg: type, …) → return"
        if issubclass(type(tool), Agent):
            agent = tool
            signature = self._build_signature(f"{agent.name}.invoke", agent.invoke)
            
            if agent.name in self._toolbox["agents"].keys():
                raise ValueError(f"'{agent.name}' already exists in {self.name}'s list of registered agents.")

            # Determine agent type for description
            if issubclass(type(agent), _PlannerAgent):
                agent_type_desc = (
                    "This PlannerAgent takes a task prompt (str) describing a complex objective. "
                    "It will generate and execute a multi-step plan using its registered tools and return the final result. "
                    "Use this when you want the agent to autonomously break down and solve a task."
                    "\nThis planner is also capable of generating plans that call the 'invoke()' method of other agents." if isinstance(agent, AgenticPlannerAgent) else ""
                )
            elif isinstance(agent, ChainSequenceAgent):
                agent_type_desc = (
                    "**ChainSequenceAgent**: takes a prompt (str) and passes it through a chain of other agents, each with their own "
                    "'invoke' logic, and then send their result to the next agent in the chain sequenc, specifically the "
                    "agent at their tail pointer. If no next agent is available, they return their result."
                )
            elif isinstance(agent, PrePostAgent):
                agent_type_desc = (
                    "**PrePostAgent**: takes a prompt (str) and passes it through a set of preprocessing methods before "
                    "feeding the input to the standard invoke() method, and the resulting llm output is then sent "
                    "through another set of postprocessing methods and then return the final result."
                )
            else:
                agent_type_desc = (
                    "**Agent**: takes a prompt (str) and generates a response (str) according to its role-prompt description. "
                    "Use this for single-turn LLM completions or simple text generation."
                )

            desc_lines = [
                f"{signature} -- ",
                f"Invokes the {agent.name} Agent.",
                agent_type_desc,
                f"This specific agent's general role: {description}",
            ]
            self._toolbox["agents"].update({agent.name: {"callable":agent.invoke, "description":"\n".join(desc_lines)}})
        else:
            if self.granular:
                super().register(tool, description=description)
            else:
                raise RuntimeError(f"Agentic Planner '{self.name}' was not configured for granular tool registration.")
    def create_plan(self, prompt: str) -> dict:
        # 1. Define the set of methods to be used:
        developer_methods = (
            "DEVELOPER-REGISTERED METHODS:"
            "\n* ".join(meta['description'] for _, meta in self._toolbox["dev_tools"].items())
        )
        tools = {name:func["callable"] for name, func in self._toolbox["dev_tools"].items()}
        
        plugin_methods = "\nPLUGIN-REGISTERED METHODS\n"
        for _plugin, _tools in self._toolbox["plugin_tools"].items():
            plugin_methods += f"* PLUGIN: {_plugin}:{"".join(f"\n\t** {meta['description']}" for meta in _tools.values())}"
            tools.update({_name:meta["callable"] for _name,meta in _tools.items()})
        
        agent_methods = (
            "\nAGENT-REGISTERED INVOKE METHODS\n"
            "".join(f"\n* {name}.invoke: {meta["description"]}\n" for name, meta in self._toolbox["agents"].items())
        )
        tools.update({f"{name}.invoke":meta["callable"] for name, meta in self._toolbox["agents"].items()})
        methods_block = f"{developer_methods}\n{plugin_methods}\n{agent_methods}"

        # 2. Craft user prompt ---------------------------------------
        user_prompt = (
            f"AVAILABLE METHODS\n{methods_block}\n\n"
            f"TASK\n{prompt}\n\n"
            "Respond *only* with the JSON plan."
        )

        # 3. Ask the LLM via BasicAgent ------------------------------
        raw = Agent.invoke(self, user_prompt).strip()
        raw = re.sub(r'^```[a-zA-Z]*|```$', '', raw)  # scrub fences if any

        try:
            steps = json.loads(raw)
        except Exception as e:
            raise ValueError(f"JSON parse error: {e}\nLLM output:\n{raw}")

        unknown = [s["function"] for s in steps if s["function"] not in tools]
        if unknown:
            raise ValueError(f"Plan uses unknown functions: {unknown}")
        
        plan = {"steps": steps, "tools": tools}
        
        # guarantee last step is "_return"
        if not plan["steps"] or plan["steps"][-1]["function"] != "_return":
            plan["steps"].append(
                {
                    "function": "_return",
                    "args": {"val": None},
                }
            )
        return plan
    def invoke(self, prompt: str):
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