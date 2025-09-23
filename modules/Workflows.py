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
    def __init__(self, name: str, description: str, steps: list[Agent|Workflow] = []):
        super().__init__(name, description)
        self._steps: list[Workflow] = [(SingleAgent(step) if isinstance(step, Agent) else step)
                                        for step in steps]
    def insert_step(self, step: Agent|Workflow, position: int = None):
        if position is None:
            self._steps.append(SingleAgent(step) if isinstance(step, Agent) else step)
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
        self.branches: list[Workflow] = [SingleAgent(branch) if isinstance(branch, Agent) else branch for branch in branches]
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
    def __init__(self, name: str, description: str, delegator_engine: LLMEngine, branches: list[Agent|Workflow]):
        super().__init__(name, description)
        self.delegator = Agent(
            name = f"{name}_Delegator",
            description = Prompts.DELEGATOR_PROMPT,
            llm_engine = delegator_engine,
            context_enabled=False
        )
        self.branches: dict[str, Workflow] = {branch._name : (SingleAgent(branch) if isinstance(branch, Agent) else branch) for branch in branches}
    def add_branch(self, branch: Agent|Workflow):
        self.branches[branch._name] = branch if isinstance(branch, Workflow) else SingleAgent(branch)
    def remove_branch(self, branch_name: str):
        removed_branch = self.branches[branch_name]
        self.branches = {name:branch for name, branch in self.branches if branch_name != name}
        return removed_branch
    async def _invoke_branch(self, branch: Workflow, prompt: str):
        loop = asyncio.get_event_loop()
        logging.info(f"[WORKFLOW] Invoking branch: {branch._name} with sub-prompt: {prompt[:150]}{'...' if len(prompt) > 150 else ''}")
        return {"branch":branch._name, "result" : await loop.run_in_executor(None, branch.invoke, prompt)}
    def clear_memory(self):
        for branch in self.branches.values():
            branch.clear_memory()
    def invoke(self, prompt: str):
        logging.info(f"\n+---{'-'*len(self._name + ' Starting')}---+"
                     f"\n|   {self._name} Starting   |"
                     f"\n+---{'-'*len(self._name + ' Starting')}---+")
        if not self.branches:
            raise ValueError("No branches to execute.")
        context = list(zip(self.branches.keys(), [val.description for val in self.branches.values()]))
        print(context)
        subtasks_raw = self.delegator.invoke(f"Decompose the following user request into a JSON-list of workflows and their assigned singular subtasks. The available workflows are described below:\n{json.dumps(context)}\nThe user request to decompose:\n{prompt}")
        subtasks_raw = re.sub(r"```json(.*?)```", r"\1", subtasks_raw, flags=re.DOTALL).strip()
        try:
            subtasks: dict = json.loads(subtasks_raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse delegator output as JSON: {e}\nOutput was:\n{subtasks_raw}")
        print(subtasks)
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