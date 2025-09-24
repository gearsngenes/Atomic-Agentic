from __future__ import annotations
from modules.Agents import Agent
import modules.Prompts as Prompts
from modules.LLMEngines import *
from abc import ABC, abstractmethod
import asyncio
import logging
import json, re
from modules.Tools import Tool
import inspect
from typing import Callable, Optional


class Workflow(ABC):
    def __init__(self, name: str, description: str):
        self._name = name
        self._description = description
    @abstractmethod
    def invoke(self, prompt: Any):
        pass
    @abstractmethod
    def clear_memory(self):
        pass
    @property
    def name(self):
        return self._name
    @property
    def description(self):
        return self._description

class SingleAgent(Workflow):
    def __init__(self, agent: Agent):
        super().__init__(agent.name, agent.description)
        self.agent = agent
    def clear_memory(self):
        if self.agent and self.agent.context_enabled:
            self.agent.clear_memory()
    def invoke(self, prompt: Any):
        logging.info(f"\n+---{'-'*len(self._name + ' Starting')}---+"
                     f"\n|   {self._name} Starting   |"
                     f"\n+---{'-'*len(self._name + ' Starting')}---+")
        if not self.agent:
            raise ValueError("No agent registered in this workflow.")
        result = self.agent.invoke(str(prompt))
        logging.info(   f"\n+---{'-'*len(self._name + ' Finished')}---+"
                        f"\n|   {self._name} Finished   |"
                        f"\n+---{'-'*len(self._name + ' Finished')}---+\n")
        return result

class ChainOfThought(Workflow):
    def __init__(self, name: str, description: str, steps: list[Agent|Workflow|Tool] = []):
        super().__init__(name, description)
        self._steps: list[Workflow] = [(
            SingleAgent(step) if isinstance(step, Agent) else (
                ToolWorkflow(step) if isinstance(step, Tool) else step
                )
            ) for step in steps]
    def insert_step(self, step: Agent|Workflow|Tool, position: int = None):
        step = SingleAgent(step) if isinstance(step, Agent) else (
            ToolWorkflow(step) if isinstance(step, Tool) else step)
        if position is None:
            self._steps.append(step)
        else:
            self._steps.insert(position, step)
    def pop(self, position: int = -1) -> Workflow:
        if not self._steps:
            raise ValueError("No agents to remove.")
        return self._steps.pop(position)
    def clear_memory(self):
        for step in self._steps:
            step.clear_memory()
    def invoke(self, prompt: Any)->Any:
        logging.info(f"\n+---{'-'*len(self._name + ' Starting')}---+"
                     f"\n|   {self._name} Starting   |"
                     f"\n+---{'-'*len(self._name + ' Starting')}---+")
        if not self._steps:
            raise ValueError("No agents registered in this workflow.")
        current_input = prompt
        previous = "User"
        for step in self._steps:
            logging.info(f"[WORKFLOW] Invoking: {step._name}")
            if isinstance(step, ToolWorkflow):
                # Decide how to forward current_input based on the tool signature
                sig = inspect.signature(step.tool.func)
                params = list(sig.parameters.values())
                bindable = [p for p in params if p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY
                )]
                has_var_pos = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
                is_single_param = (len(bindable) == 1 and not has_var_pos)

                if is_single_param:
                    # Always pass the previous output as a SINGLE value (dict/list/tuple/scalar)
                    current_input = step.invoke(current_input)
                else:
                    # Multi-parameter: preserve shape so ToolWorkflow can bind correctly
                    if isinstance(current_input, dict):
                        current_input = step.invoke(current_input)
                    elif isinstance(current_input, (list, tuple)):
                        current_input = step.invoke(*current_input)
                    else:
                        current_input = step.invoke(current_input)
            else:
                current_input = step.invoke(f"Input from {previous}:\n{current_input}")
            previous = step._name
        logging.info(   f"\n+---{'-'*len(self._name + ' Finished')}---+"
                        f"\n|   {self._name} Finished   |"
                        f"\n+---{'-'*len(self._name + ' Finished')}---+\n")
        return current_input

class MakerChecker(Workflow):
    def __init__(self, name: str, description: str, maker: Agent, checker: Agent, max_revisions: int = 0):
        super().__init__(name, description)
        self.maker = maker
        self.checker = checker
        self.max_revisions = max_revisions
    def clear_memory(self):
        if self.maker and self.maker.context_enabled:
            self.maker.clear_memory()
        if self.checker and self.checker.context_enabled:
            self.checker.clear_memory()
    def invoke(self, prompt: str):
        logging.info(f"\n+---{'-'*len(self._name + ' Starting')}---+"
                     f"\n|   {self._name} Starting   |"
                     f"\n+---{'-'*len(self._name + ' Starting')}---+")
        if not self.maker or not self.checker:
            raise ValueError("Maker and Checker agents must be set.")
        output = self.maker.invoke(f"Create a response for the following input:\n{prompt}")
        count = 0
        while count < self.max_revisions:
            count += 1
            logging.info(f"[WORKFLOW] Revision {count} by {self.maker.name} after review by {self.checker.name}")
            check_prompt = f"Inspect the following output for correctness and quality, and return any major corrections or revision notes you'd suggest to {self.maker.name}:\nOutput:\n{output}\nInput:\n{prompt}\nIs the output correct and high quality? Respond with 'yes' or 'no' and provide feedback if 'no'."
            check_output = self.checker.invoke(check_prompt)
            revision_prompt = f"Feedback from {self.checker.name}:\n{check_output}\nPlease improve the output based on this feedback."
            output = self.maker.invoke(revision_prompt)
        logging.info(   f"\n+---{'-'*len(self._name + ' Finished')}---+"
                        f"\n|   {self._name} Finished   |"
                        f"\n+---{'-'*len(self._name + ' Finished')}---+\n")
        return output

class ConditionalWorkflow(Workflow):
    def __init__(self, name: str, description: str, decider_llm: LLMEngine, branches: list[Agent|Workflow]):
        super().__init__(name, description)
        self.decider = Agent(
            name = name + ".decider",
            description= "decides",
            llm_engine = decider_llm,
            role_prompt=Prompts.CONDITIONAL_DECIDER_PROMPT,
        )
        self.branches: list[Workflow] = [SingleAgent(branch) if isinstance(branch, Agent) else (ToolWorkflow(branch) if isinstance(branch, Tool) else branch) for branch in branches]
    def add_branch(self, branch: Agent|Workflow):
        self.branches.append(branch if isinstance(branch, Workflow) else SingleAgent(branch))
    def remove_branch(self, branch_name: str):
        removed_branch = next((b for b in self.branches if b._name == branch_name), None)
        self.branches = [b for b in self.branches if b._name != branch_name]
        return removed_branch
    def clear_memory(self):
        self.decider.clear_memory()
        for branch in self.branches:
            branch.clear_memory()
    def invoke(self, prompt: Any):
        logging.info(f"\n+---{'-'*len(self._name + ' Starting')}---+"
                     f"\n|   {self._name} Starting   |"
                     f"\n+---{'-'*len(self._name + ' Starting')}---+")
        if not self.decider or not self.branches:
            raise ValueError("Decider and branches must be set.")
        branch_contexts = f"{',\n'.join([f'{branch._name}: {branch.description}' for branch in self.branches])}"
        self.decider.role_prompt = Prompts.CONDITIONAL_DECIDER_PROMPT.format(branches = branch_contexts)
        action_prompt = f"Select the best workflow to handle the following user task: {prompt}"
        decision = self.decider.invoke(action_prompt).strip()
        self.decider.role_prompt=Prompts.CONDITIONAL_DECIDER_PROMPT
        logging.info(f"[WORKFLOW] Decider for {self._name} chose workflow: {decision}")
        for branch in self.branches:
            if branch._name == decision:
                result = branch.invoke(prompt)
                logging.info(   f"\n+---{'-'*len(self._name + ' Finished')}---+"
                        f"\n|   {self._name} Finished   |"
                        f"\n+---{'-'*len(self._name + ' Finished')}---+\n")
                return result
        raise ValueError(f"Decider chose an unknown branch: {decision}")

class Delegator(Workflow):
    def __init__(self, name: str, description: str, delegator_engine: LLMEngine, branches: list[Agent|Workflow|Tool], context_enabled = False):
        super().__init__(name, description)
        self.delegator = Agent(
            name = f"{name}_Delegator",
            description = Prompts.DELEGATOR_PROMPT,
            llm_engine = delegator_engine,
            context_enabled=context_enabled
        )
        self.branches: dict[str, Workflow] = {
            branch._name : (
                    SingleAgent(branch) if isinstance(branch, Agent) else (
                    ToolWorkflow(branch) if isinstance(branch, Tool) else branch)
                ) for branch in branches}
    def add_branch(self, branch: Agent|Workflow|Tool):
        self.branches[branch._name] = SingleAgent(branch) if isinstance(branch, Agent) else (ToolWorkflow(branch) if isinstance(branch, Tool) else branch)
    def remove_branch(self, branch_name: str):
        removed_branch = self.branches.pop(branch_name)
        return removed_branch
    async def _invoke_branch(self, branch: Workflow, prompt: str):
        loop = asyncio.get_event_loop()
        logging.info(f"[WORKFLOW] Invoking branch: {branch._name} with sub-prompt: {prompt[:150]}{'...' if len(prompt) > 150 else ''}")
        return {"branch":branch._name, "result" : await loop.run_in_executor(None, branch.invoke, prompt)}
    def clear_memory(self):
        for branch in self.branches.values():
            branch.clear_memory()
    def invoke(self, prompt: Any):
        logging.info(f"\n+---{'-'*len(self._name + ' Starting')}---+"
                     f"\n|   {self._name} Starting   |"
                     f"\n+---{'-'*len(self._name + ' Starting')}---+")
        if not self.branches:
            raise ValueError("No branches to execute.")
        context = list(zip(self.branches.keys(), [val.description for val in self.branches.values()]))
        subtasks_raw = self.delegator.invoke(f"Decompose the following user request into a JSON-list of workflows and their assigned singular subtasks. The available workflows are described below:\n{json.dumps(context)}\nThe user request to decompose:\n{prompt}")
        subtasks_raw = re.sub(r"```json(.*?)```", r"\1", subtasks_raw, flags=re.DOTALL).strip()
        try:
            subtasks: dict = json.loads(subtasks_raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse delegator output as JSON: {e}\nOutput was:\n{subtasks_raw}")
        for item in subtasks:
            if not self.branches.get(item["workflow"]):
                raise KeyError(f"{item["workflow"]} not in {self.name}.branches")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = [self._invoke_branch(self.branches.get(subtask["workflow"]), subtask["subtask"]) for subtask in subtasks]
        results = loop.run_until_complete(asyncio.gather(*tasks))
        loop.close()
        results = {result["branch"]:result["result"] for result in results}
        for branch in self.branches.keys():
            if not results.get(branch):
                results[branch] = ""
        logging.info(   f"\n+---{'-'*len(self._name + ' Finished')}---+"
                        f"\n|   {self._name} Finished   |"
                        f"\n+---{'-'*len(self._name + ' Finished')}---+\n")
        return results

class ToolWorkflow(Workflow):
    def __init__(self, tool: Tool):
        super().__init__(tool.name, tool.description)
        self.tool = tool
    def clear_memory(self):
        pass
    def invoke(self, *prompt: Any):
        default = self.tool.get_param_defaults(deep=True)
        sig = inspect.signature(self.tool.func)
        params = list(sig.parameters.values())
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        # CASE 1: Single parameter input
        if len(list(default.keys())) == 1:
            # logging.info(f"[WORKFLOW] Single Parameter for {self.name}")
            # single-parameter function: pass the single value, not the whole tuple
            if len(prompt) != 1:
                raise TypeError(f"{self.__class__.__name__}.invoke expected exactly 1 argument for a single-parameter tool")
            return self.tool.func(prompt[0])
        # CASE 2: Dictionary input
        elif len(prompt) == 1 and isinstance(prompt[0], dict):
            # logging.info(f"[WORKFLOW] Dictionary Parsing of inputs for {self.name}")
            provided = dict(prompt[0])
            # Start from defaults and override with provided keys
            call_kwargs = self.tool.get_param_defaults(deep=True)
            # Remove any var-kw placeholder that your Tool might have inserted
            for p in params:
                if p.kind == inspect.Parameter.VAR_KEYWORD and p.name in call_kwargs:
                    call_kwargs.pop(p.name, None)
            # Apply known keys
            for k, v in list(provided.items()):
                if k in call_kwargs:
                    call_kwargs[k] = v
                    provided.pop(k)
            # If there are unknown keys, only allow if function has **kwargs
            if provided and not has_varkw:
                unknown = ", ".join(provided.keys())
                raise TypeError(f"{self.tool.func.__name__}() got unexpected keyword argument(s): {unknown}")
            # If has **kwargs, merge the leftover keys
            if provided and has_varkw:
                call_kwargs.update(provided)
            return self.tool.func(**call_kwargs)
        # CASE 3: Exact positional parameter match
        elif len(prompt) == len(list(default.keys())):
            # logging.info(f"[WORKFLOW] Exact positional matching of parameters for {self.name}")
            return self.tool.func(*prompt)
        # CASE 4: Fallback
        else:
            # logging.info(f"[WORKFLOW] Fallback method for {self.name}")
            param_keys = list(default.keys())[:len(prompt)]
            for i, param in enumerate(prompt):
                default[param_keys[i]] = param
            # pass by keywords, not as positional expansion of dict keys
            return self.tool.func(**default)