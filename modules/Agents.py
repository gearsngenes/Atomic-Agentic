import json, re, inspect, logging
from dotenv import load_dotenv
from typing import Any, get_type_hints
load_dotenv()

# internal imports
from modules.Plugins import *
import modules.Prompts as Prompts
from modules.LLMEngines import *

# ────────────────────────────────────────────────────────────────
# 1.  Agent  (LLM responds to prompts)
# ────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, name, description: str, llm_engine:LLMEngine, role_prompt: str = Prompts.DEFAULT_PROMPT, context_enabled: bool = False):
        self._name = name
        self._llm_engine: LLMEngine = llm_engine
        self._role_prompt = role_prompt
        self._context_enabled = context_enabled
        self._description = description
        self._history = []

    @property
    def name(self):
        return self._name
    
    @property
    def description(self):
        description = f"~~Agent {self.name}~~\nA generic Agent for on Text-Text responses. Description: {self._description}"
        return description

    @property
    def role_prompt(self):
        return self._role_prompt

    @property
    def context_enabled(self):
        return self._context_enabled

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
    
    @description.setter
    def description(self, value: str):
        self._description = value

    @role_prompt.setter
    def role_prompt(self, value: str):
        self._role_prompt = value

    def clear_memory(self):
        self._history = []
    
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
    def __init__(self, name, description, llm_engine, role_prompt = Prompts.DEFAULT_PROMPT, context_enabled = False):
        Agent.__init__(self, name, description, llm_engine, role_prompt, context_enabled)
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
    def description(self):
        desc = f"~~PrePost Agent {self.name}~~\nThis agent preprocesses inputs to the LLM before generating output, and then post-processes the output before returning it.\nDescription:{self._description}"
        return desc
    @description.setter
    def description(self, value: str):
        self._description = value
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
    def __init__(self, name: str, context_enabled: bool = False):
        self._agents:list[Agent] = []
        self._name = name
        self._context_enabled = context_enabled
        self._role_prompt = f"~~ChainSequence Agent {self._name}~~\nThis agent sequentially invokes a list of agents."
        self._history = []
        self._llm_engine = None
    @property
    def agents(self) -> list[Agent]:
        return self._agents.copy()

    @property
    def role_prompt(self):
        return self._role_prompt
    @property
    def description(self):
        desc = f"{self._role_prompt}\nDescription: this agent calls the following agents in order below:\n~~~start~~~{"".join(f"\n{agent._description}" for agent in self._agents)}\n~~~end~~~"
        return desc
    @property
    def llm_engine(self):
        return None
    
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
        if self.context_enabled:
            self._history.append({"role": "user", "content": "input: " + prompt})
        for agent in self._agents:
            result = agent.invoke(str(result))
            if self.context_enabled:
                self._history.append({"role": "assistant", "content": agent.name + ": "+str(result)})
        if self.context_enabled:
            self._history.append({"role": "assistant", "content": self.name + ": " + str(result)})
        return result

# ────────────────────────────────────────────────────────────────
# 4.  Human Agent  (Asks human for input, when provided a prompt)
# ────────────────────────────────────────────────────────────────
class HumanAgent(Agent):
    def __init__(self, name, description, context_enabled = False):
        self._context_enabled = context_enabled
        self._name = name
        self._description = description
        self._llm_engine = None
    def invoke(self, prompt:str):
        return input(f"{prompt}\n{self.name}'s Response: ")

from abc import abstractmethod
# ────────────────────────────────────────────────────────────────
# 5.  Abstract ToolAgent  (Uses Tools and Agents to execute tasks)
# ────────────────────────────────────────────────────────────────
class ToolAgent(Agent):
    def __init__(self, name, description, llm_engine, role_prompt = Prompts.DEFAULT_PROMPT):
        super().__init__(name, description, llm_engine, role_prompt, context_enabled = False)
        self._toolbox:dict[str, dict] = {}
    
    @staticmethod
    def _build_signature(key: str, func: callable) -> str:
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        params = [
            f"{n}: {hints.get(n, Any).__name__}"
            + (f" = {p.default!r}" if p.default is not inspect._empty else "")
            for n, p in sig.parameters.items() if n != "self"
        ]
        rtype = hints.get('return', Any).__name__
        return f"{key}({', '.join(params)}) → {rtype}"
    @abstractmethod
    def strategize(self, prompt:str)->dict:
        pass
    @abstractmethod
    def execute(self, plan:dict)->Any:
        pass
    @abstractmethod
    def register(self, tool: Any, description: str|None = None) -> None:
        pass
    @property # toolbox should not be editable from the outside
    def toolbox(self):
        return self._toolbox.copy()