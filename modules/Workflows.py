from __future__ import annotations
from modules.Agents import Agent
import modules.Prompts as Prompts
from modules.LLMEngines import LLMEngine
from abc import ABC, abstractmethod
import asyncio
import logging
import json, re
from modules.Tools import Tool
import json, logging
from typing import Any

import inspect
from typing import Callable

def callable_is_single_param(fn: Callable) -> bool:
    """
    Return True iff `fn` takes exactly one logical input parameter,
    excluding 'self' and excluding varargs/kwargs.
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        # fallback: unknown signature
        return False

    params = []
    for p in sig.parameters.values():
        # ignore bound-instance 'self' if present
        if p.name == "self":
            continue
        # if function uses *args or **kwargs, treat as not single-param
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            return False
        # count positional-only, positional-or-keyword, keyword-only
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY):
            params.append(p)

    return len(params) == 1

WF_RESULT = "__wf_result__"

class Workflow(ABC):
    def __init__(self, name: str, description: str, result_schema: list[str] = [WF_RESULT]):
        self._name = name
        self._description = description
        self._result_schema = result_schema
        self._is_single_param = False
        for s in self._result_schema:
            if not s:
                raise ValueError("Empty strings are not permissible output schema parameter names")

    @property
    def name(self) -> str: return self._name
    
    @property
    def description(self) -> str: return self._description
    
    @property
    def result_schema(self) -> list[str]: return self._result_schema
    @result_schema.setter
    def result_schema(self, val: list[str]):
        if val == None or not len(val):
            raise ValueError("Result schema must be set to a non-empty list of strings")
        self._result_schema = val

    def package_results(self, results: Any)->dict:
        # If results is a dict/list/tuple, check lengths and map by order when equal
        if isinstance(results, dict):
            if len(results) != len(self._result_schema):
                return {self._result_schema[0]: results}
            vals = list(results.values())
            return {k: v for k, v in zip(self._result_schema, vals)}
        if isinstance(results, (list, tuple)):
            if len(results) != len(self._result_schema):
                return {self._result_schema[0]: results}
            return {self._result_schema[i] : results[i] for i in range(len(self._result_schema))}
        # If result is not a collection
        final = {}
        for i, k in enumerate(self._result_schema):
            if i == 0: final[k] = results;continue
            final[k] = None
        return final
    
    @abstractmethod
    def invoke(self, *args: Any, **kwargs: Any) -> dict:
        pass

    @abstractmethod
    def clear_memory(self) -> None:
        pass

class AgentFlow(Workflow):
    def __init__(self, agent: Agent, result_schema: list[str] = [WF_RESULT]):
        super().__init__(name = agent.name,
                         description = agent.description,
                         result_schema = result_schema)
        self.agent: Agent = agent
        self._is_single_param = True
    def clear_memory(self):
        self.agent.clear_memory()
    def invoke(self, *args: Any, **kwargs: Any)->dict:
        logging.info(f"\n+---{'-'*len(self._name + ' Starting')}---+"
                     f"\n|   {self._name} Starting   |"
                     f"\n+---{'-'*len(self._name + ' Starting')}---+")
        try:
            result = self.agent.invoke(*args, **kwargs)
        except TypeError as e:
            logging.info(f"Failed to pass in arguments as is, giving '{e}'. Stringifying...")
            result = self.agent.invoke(f"*args:{args}\n**kwargs:{kwargs}")
        result = self.package_results(result)
        logging.info(   f"\n+---{'-'*len(self._name + ' Finished')}---+"
                        f"\n|   {self._name} Finished   |"
                        f"\n+---{'-'*len(self._name + ' Finished')}---+\n")
        return result

class ToolFlow(Workflow):
    def __init__(self, tool: Tool, result_schema: list[str] = [WF_RESULT]):
        super().__init__(tool.name, tool.description, result_schema)
        self.tool = tool
        self._is_single_param = callable_is_single_param(tool.func)

    def clear_memory(self):
        self.tool.clear_memory()

    def invoke(self, *args: Any, **kwargs: Any)->dict:
        """
        Invoke the underlying function and bind its return into a dict matching
        the keys of `result_schema`.
        """
        # Raw results
        results = self.tool.func(*args, **kwargs)
        # Package into the declared result schema
        results = self.package_results(results)
        return results

class ChainOfThought(Workflow):
    def __init__(self, name: str, description: str,
                 steps: list[Tool|Agent|Workflow] = [], unpack_midsteps:bool = True,
                 result_schema: list[str] = [WF_RESULT]):
        super().__init__(name, description, result_schema)
        self._steps: list[Workflow] = []
        self._unpack_midsteps = unpack_midsteps
        for step in steps:
            if isinstance(step, Tool): new_step = ToolFlow(step)
            elif isinstance(step, Agent): new_step = AgentFlow(step)
            else: new_step = step
            self._steps.append(new_step)
        self._is_single_param = self._steps[0]._is_single_param

    def insert_step(self, step: Agent|Workflow|Tool, schema: list[str] = [WF_RESULT], position: int | None = None):
        if isinstance(step, Agent): step = AgentFlow(step, schema)
        elif isinstance(step, Tool):  step = ToolFlow(step, schema)
        elif isinstance(step, Workflow): step._result_schema = schema
        if position is None: self._steps.append(step, schema)
        else: self._steps.insert(position, step)
        self._is_single_param = self._steps[0]._is_single_param

    def pop(self, position: int = -1) -> Workflow:
        if not self._steps: raise ValueError("No steps to remove.")
        popped = self._steps.pop(position)
        self._is_single_param = self._steps[0]._is_single_param
        return popped

    def clear_memory(self):
        for step in self._steps: step.clear_memory()

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        logging.info(f"\n+---{'-'*len(self._name + ' Starting')}---+"
                     f"\n|   {self._name} Starting   |"
                     f"\n+---{'-'*len(self._name + ' Starting')}---+")
        if not self._steps:
            raise ValueError("No steps registered in this workflow.")

        current = None

        # Iterate schema pipeline
        for step in self._steps:
            logging.info(f"[WORKFLOW] Invoking: {step.name}")
            if step == self._steps[0]:
                current = step.invoke(*args, **kwargs)
                continue
            # If current uses the default result key, handle it as seen below
            if WF_RESULT in current:
                mid_result = current[WF_RESULT]
                if step._is_single_param: current = step.invoke(mid_result)
                elif isinstance(mid_result,(list,tuple)): current = step.invoke(*mid_result)
                elif isinstance(mid_result, dict): current = step.invoke(**mid_result)
                continue
            # Otherwise, treat it as key-word arguments
            current = step.invoke(**current)
        result = self.package_results(current)
        logging.info(f"\n+---{'-'*len(self._name + ' Finished')}---+"
                     f"\n|   {self._name} Finished   |"
                     f"\n+---{'-'*len(self._name + ' Finished')}---+\n")
        return result

class MakerChecker(Workflow):
    """
    Maker-Checker pattern (agents only), *args/**kwargs compatible.
    """
    def __init__(self, name: str, description: str,
                 maker: Workflow, checker: Workflow, early_stop: Agent|Tool|Workflow = None,
                 max_revisions: int = 1,
                 result_schema: list[str] = [WF_RESULT]):
        super().__init__(name, description, result_schema)
        if max_revisions < 0:
            raise ValueError("max_revisions must be >= 0")
        self._is_single_param = maker._is_single_param
        self.max_revisions = max_revisions
        self.maker: Workflow = maker
        self.checker: Workflow = checker
        if not early_stop: self.early_stop = None
        elif isinstance(early_stop, Agent): self.early_stop = AgentFlow(early_stop)
        elif isinstance(early_stop, Tool): self.early_stop = ToolFlow(early_stop)
        else: self.early_stop = early_stop
        if self.early_stop and len(self.early_stop.result_schema) > 1:
            raise ValueError("The Early stop workflow component should only have a single key "
                             "to track whether or the current revision notes warrant approving "
                             "the current draft or not. A single key for a single boolean.")

    def clear_memory(self):
        self.maker.clear_memory()
        self.checker.clear_memory()
        if self.early_stop:
            self.early_stop.clear_memory()

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        logging.info(
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
            f"\n|   {self._name} Starting   |"
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
        )
        rounds: list[dict] = []

        # Prepare first maker user message (no revisions yet)
        logging.info(f"{self.name} Making the first draft")
        draft = self.maker.invoke(*args, **kwargs)

        for i in range(1, self.max_revisions + 1):
            logging.info(f"{self.name} Reviewing draft {i}")
            # Build revions from current draft
            approved = False
            if WF_RESULT in draft:
                draft = draft[WF_RESULT]
            if self.checker._is_single_param: revisions = self.checker.invoke(draft)
            elif isinstance(draft, dict): revisions = self.checker.invoke(**draft)
            elif isinstance(draft, (tuple, list)): revisions = self.checker.invoke(*draft)
            if WF_RESULT in revisions:
                revisions = revisions[WF_RESULT]
            if self.early_stop:
                logging.info(f"[WORKFLOW] {self.early_stop.name} is marking checking if revisions warrant early approval")
                if self.early_stop and self.early_stop._is_single_param: approver_res = self.early_stop.invoke(revisions)
                elif isinstance(revisions, dict): approver_res = self.early_stop.invoke(**revisions)
                elif isinstance(revisions, (tuple, list)): approver_res = self.early_stop.invoke(*revisions)
            approved = self.early_stop != None and approver_res[list(approver_res.keys())[0]]
            rounds.append({
                "approved": approved,
                "revisions": revisions,
                "draft": draft,
            })
            if approved:
                logging.info(f"[WORKFLOW] {self.name} Approved draft {i}")
                break
            # Make a new revised draft
            logging.info(f"{self.name} Making draft {i+1}")
            if self.maker._is_single_param: draft = self.maker.invoke(revisions)
            elif isinstance(revisions, dict): draft = self.maker.invoke(**revisions)
            elif isinstance(revisions, (tuple, list)): draft = self.maker.invoke(*revisions)
        result = self.package_results((rounds, draft))
        logging.info(
            f"\n+---{'-'*len(self._name + ' Finished')}---+"
            f"\n|   {self._name} Finished   |"
            f"\n+---{'-'*len(self._name + ' Finished')}---+\n"
        )
        return result

class Selector(Workflow):
    def __init__(
        self, name: str, description: str,
        branches: list[Workflow], decider: Agent|Workflow|Tool,
        result_schema: list[str] = [WF_RESULT]):
        super().__init__(name, description, result_schema)
        # Initialize the choice agents
        self.branches: list[Workflow] = []
        for branch in branches:
            if isinstance(branch, Agent): self.branches.append(AgentFlow(branch))
            elif isinstance(branch, Tool): self.branches.append(ToolFlow(branch))
            else: self.branches.append(branch)
        if isinstance(decider, Agent): self.decider = AgentFlow(decider)
        elif isinstance(decider, Tool): self.decider = ToolFlow(decider)
        else: self.decider = decider
        if len(self.decider._result_schema) > 1:
            raise ValueError("The decider workflow must have a result schema of length 1, as we only "
                             "expect a single string corresponding to the name of one of the branches.")
        self._is_single_param = self.decider._is_single_param

    def add_branch(self, branch: Agent|Tool|Workflow, schema: list[str] = [WF_RESULT], position: int|None = None) -> None:
        if isinstance(branch, Agent): new_branch = AgentFlow(branch)
        elif isinstance(branch, Tool): new_branch = ToolFlow(branch)
        else: new_branch = branch
        new_branch._result_schema = schema
        if not position: self.branches.append(new_branch)
        else: self.branches.insert(position, new_branch)

    def remove_branch(self, position: int = -1):
        return self.branches.pop(position)

    def clear_memory(self):
        self.decider.clear_memory()
        for branch in self.branches: branch.clear_memory(self)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        logging.info(
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
            f"\n|   {self._name} Starting   |"
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
        )

        if not self.decider or not self.branches:
            raise ValueError("Decider and branches must be set.")

        logging.info(f"[WORKFLOW] Selecting branch via decider on {self._name}")

        decision_obj = self.decider.invoke(*args, **kwargs)
        decision_name = decision_obj[list(decision_obj.keys())[0]]

        if not isinstance(decision_name, str):
            return TypeError(f"{decision_name} is not an instance of type 'str'. "
                             f"Check {self.decider.name}'s return type")
        logging.info(f"[WORKFLOW] Decider chose: {decision_name}")

        # route to the chosen branch with the ORIGINAL payload
        selected: Workflow = None
        for branch in self.branches:
            if branch.name == decision_name:
                selected = branch
                break
        if not selected:
            raise ValueError(f"Decider chose an unknown branch: {decision_name}")
        result = selected.invoke(*args, **kwargs)
        result = self.package_results(result)
        logging.info(
            f"\n+---{'-'*len(self._name + ' Finished')}---+"
            f"\n|   {self._name} Finished   |"
            f"\n+---{'-'*len(self._name + ' Finished')}---+\n"
        )
        return result

class BatchFlow(Workflow):
    """
    Simple parallel workflow: runs provided branches in parallel.

    Invocation semantics:
    - Positional arguments to `invoke` are assigned to branches by position (0 -> branches[0], etc.).
    - Keyword arguments must use branch names to target a specific branch and override positional inputs for that branch.
    - Each branch may be an Agent/Tool/Workflow and will be wrapped as an appropriate Workflow (AgentFlow/ToolFlow) on construction.

    Result schema rules:
    - `result_schema` must either have length 1 or length == number of branches.
      - If length == 1: the list of branch results is returned under that single key.
      - If length == number_of_branches: each branch result is mapped positionally to the schema keys.
    - If `result_schema` is empty, the return value is a dict mapping branch.name -> branch_result.
    """
    def __init__(self, name: str, description: str, branches: list[Agent|Tool|Workflow], result_schema: list[str] = [WF_RESULT]):
        super().__init__(name=name, description=description, result_schema=result_schema)
        self.branches: list[Workflow] = []
        for b in branches:
            if isinstance(b, Agent): self.branches.append(AgentFlow(b))
            elif isinstance(b, Tool): self.branches.append(ToolFlow(b))
            else: self.branches.append(b)

        # Validate schema length now that branches are known
        if len(self._result_schema) not in (0, 1, len(self.branches)):
            raise ValueError("result_schema must be empty, length 1, or match number of branches")

    def clear_memory(self):
        for b in self.branches:
            b.clear_memory()

    def invoke(self, *args: Any, **kwargs: Any) -> dict:
        logging.info(f"[PARALLEL] {self._name} Starting")

        # Build payloads per-branch from positionals and keyword overrides
        payloads = [None] * len(self.branches)
        for i in range(len(self.branches)):
            # positional if provided
            if i < len(args):
                payloads[i] = args[i]
            else:
                payloads[i] = None
        # keyword overrides by branch name
        for i, b in enumerate(self.branches):
            if b.name in kwargs:
                payloads[i] = kwargs[b.name]

        async def _fanout(payloads:list[Any]):
            loop = asyncio.get_running_loop()
            tasks = []
            for i, branch in enumerate(self.branches):
                payload = payloads[i]
                def _call(branch=branch, payload=payload):
                    # Normalize payload invocation similar to other workflows
                    if payload is None:
                        return branch.invoke()
                    if branch._is_single_param:
                        return branch.invoke(payload)
                    if isinstance(payload, (list, tuple)):
                        return branch.invoke(*payload)
                    if isinstance(payload, dict):
                        return branch.invoke(**payload)
                    return branch.invoke(payload)
                tasks.append(loop.run_in_executor(None, _call))
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return results

        try:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(_fanout(payloads))
            finally:
                asyncio.set_event_loop(None)
                loop.close()
        except RuntimeError:
            # fallback for environments that disallow new event loops
            results = asyncio.run(_fanout(payloads))

        # Normalize results: if a branch returned a single-key dict with WF_RESULT,
        # unwrap it so callers don't get nested {branch: {WF_RESULT: ...}}
        normalized = []
        for r in results:
            if isinstance(r, dict) and WF_RESULT in r:
                normalized.append(r[WF_RESULT])
            else: normalized.append(r)

        # If no explicit schema, return mapping branch.name -> result
        if not self._result_schema:
            out = {b.name: r for b, r in zip(self.branches, normalized)}
            logging.info(f"[PARALLEL] {self._name} Finished")
            return out

        # Otherwise schema must be length 1 or length num branches (validated in ctor)
        out = self.package_results(normalized)
        logging.info(f"[PARALLEL] {self._name} Finished")
        return out