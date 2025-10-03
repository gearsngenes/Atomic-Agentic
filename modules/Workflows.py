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


class Workflow(ABC):
    def __init__(self, name: str, description: str, result_schema: list[str], return_raw = False):
        self._name = name
        self._description = description
        self._result_schema = result_schema or []
        self.return_raw = return_raw or not self._result_schema
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
        self._result_schema = val

    def package_results(self, results: Any):
        # If no schema provided, return results unchanged (caller asked for raw results)
        if not self._result_schema or self.return_raw:
            return results

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
        return {self._result_schema[0] : results}
    
    @abstractmethod
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def clear_memory(self) -> None:
        pass

class AgentFlow(Workflow):
    def __init__(self, agent: Agent,
                 result_schema: list[str] = [],
                 return_raw: bool = False):
        super().__init__(name = agent.name,
                         description = agent.description,
                         result_schema = result_schema,
                         return_raw=return_raw)
        self.agent: Agent = agent
    def clear_memory(self):
        self.agent.clear_memory()
    def invoke(self, *args, **kwargs):
        logging.info(f"\n+---{'-'*len(self._name + ' Starting')}---+"
                     f"\n|   {self._name} Starting   |"
                     f"\n+---{'-'*len(self._name + ' Starting')}---+")
        result = self.agent.invoke(*args, **kwargs)
        result = self.package_results(result)
        logging.info(   f"\n+---{'-'*len(self._name + ' Finished')}---+"
                        f"\n|   {self._name} Finished   |"
                        f"\n+---{'-'*len(self._name + ' Finished')}---+\n")
        return result

class ToolFlow(Workflow):
    def __init__(self, tool: Tool, result_schema: list[str] = [], return_raw = False):
        super().__init__(tool.name, tool.description, result_schema, return_raw=return_raw)
        self.tool = tool

    def clear_memory(self):
        self.tool.clear_memory()

    def invoke(self, *args, **kwargs):
        """
        Invoke the underlying function and bind its return into a dict matching
        the keys of `result_schema`.

        Rules (per your request):
        - `result_schema` must be a flat dict of keys, e.g. {"k1": {}, "k2": {}}.
        - If the returned `results` is a dict, list, or tuple:
            - If number of items == number of schema keys: map items in order to keys and return the dict.
            - Otherwise: return {first_key: results} (put the whole result under the first schema key).
        - If `results` is any other type: return {first_key: results}.
        """
        # Raw results
        results = self.tool.func(*args, **kwargs)
        # Package into the declared result schema
        results = self.package_results(results)
        return results

class ChainOfThought(Workflow):
    def __init__(self, name: str, description: str, steps: list[Workflow] = [], result_schema: list[str] = []):
        super().__init__(name, description, result_schema)
        self._steps: list[Workflow] = steps

    def insert_step(self, step: Agent|Workflow|Tool, schema: list[str] = [], position: int | None = None):
        if isinstance(step, Agent): step = AgentFlow(step, schema)
        if isinstance(step, Tool):  step = ToolFlow(step, schema)
        if position is None: self._steps.append(step, schema)
        else: self._steps.insert(position, step)

    def pop(self, position: int = -1) -> Workflow:
        if not self._steps: raise ValueError("No steps to remove.")
        return self._steps.pop(position)

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
            # MUST output key-word representations if they aren't the last step
            if not isinstance(current, dict):
                raise RuntimeError(
                    "The output of the previous step must provide a keyword-representation "
                    f"of the arguments for the current step, but instead got:\n{current}\n"
                    f"Please check {step.name}'s input schema")
            # If current is a dict, pass it as keyword arguments to the next step.
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
                 maker: Workflow,
                 checker: Workflow,
                 early_stop: Agent|Tool|Workflow = None,
                 max_revisions: int = 1,
                 result_schema: list[str] = [],
                 return_raw: bool = False):
        super().__init__(name, description, result_schema, return_raw)
        if max_revisions < 0:
            raise ValueError("max_revisions must be >= 0")
        self.max_revisions = max_revisions
        self.maker: Workflow = maker
        self.checker: Workflow = checker
        if early_stop and isinstance(early_stop, Agent):
            self.early_stop = AgentFlow(early_stop, [])
        elif early_stop and isinstance(early_stop, Tool):
            self.early_stop = ToolFlow(early_stop, [])
        elif early_stop:
            self.early_stop = early_stop
        else:
            self.early_stop = None

    def clear_memory(self):
        self.maker.clear_memory()
        self.checker.clear_memory()

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        logging.info(
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
            f"\n|   {self._name} Starting   |"
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
        )
        rounds: list[dict] = []
        draft = None
        approved = False

        # Prepare first maker user message (no revisions yet)
        logging.info(f"{self.name} Making the first draft")
        draft = self.maker.invoke(*args, **kwargs)

        for i in range(1, self.max_revisions + 1):
            logging.info(f"{self.name} Reviewing draft {i}")
            # Build revions from current draft
            if isinstance(draft, dict):
                revisions = self.checker.invoke(**draft)
            elif isinstance(draft, (tuple, list)):
                revisions = self.checker.invoke(*draft)
            else:
                revisions = self.checker.invoke(draft)
            if isinstance(revisions, dict):
                approved = self.early_stop and self.early_stop.invoke(**revisions)
            elif isinstance(revisions, (tuple, list)):
                approved = self.early_stop and self.early_stop.invoke(*revisions)
            else:
                approved = self.early_stop and self.early_stop.invoke(revisions)
            approved = approved or i == self.max_revisions
            rounds.append({
                "approved": approved,
                "revisions": revisions,
                "draft": draft,
            })
            if approved:
                break
            # Make a new revised draft
            logging.info(f"{self.name} Making draft {i+1}")
            if isinstance(revisions, dict):
                draft = self.maker.invoke(**revisions)
            elif isinstance(revisions, (tuple, list)):
                draft = self.maker.invoke(*revisions)
            else:
                draft = self.maker.invoke(revisions)
        result = self.package_results((rounds, draft))
        logging.info(
            f"\n+---{'-'*len(self._name + ' Finished')}---+"
            f"\n|   {self._name} Finished   |"
            f"\n+---{'-'*len(self._name + ' Finished')}---+\n"
        )
        return result

class FanFlow(Workflow):
    def __init__(self, name: str, description: str,
                 branches: list[Agent|Tool|Workflow] = [], 
                 result_schema: list[str] = [],
                 return_raw: bool = False):
        super().__init__(name, description, result_schema, return_raw)
        self.branches: List[Workflow] = []
        for branch in branches:
            if isinstance(branch, Agent):
                self.branches.append(AgentFlow(branch, [], True))
            elif isinstance(branch, Tool):
                self.branches.append(ToolFlow(branch, [], True))
            else:
                self.branches.append(branch)
    async def _invoke_branch_async(self, branch: Workflow, payload):
        """
        Invoke a branch workflow with normalized payload conventions.
        For ToolFlow: bind using signature-aware logic (as in ToolFlow.invoke).
        For other Workflows/Agents: pass through payload as a single argument,
        unless payload is an explicit callspec [args, kwargs].
        """
        loop = asyncio.get_running_loop()
        # Skip sentinel
        if payload is None:
            return None
        def _call():
            return branch.invoke(payload)
        logging.info(f"[WORKFLOW] Invoking branch: {branch._name}")
        return await loop.run_in_executor(None, _call)
    async def _fanout(self, payloads:list[Any]):
        if len(payloads) != len(self.branches):
            raise ArithmeticError("The number of payloads do not match the number of branches."
                                  f" {self.name} expected {len(self.branches)}, but got {len(payloads)}")
        tasks = [
            self._invoke_branch_async(self.branches[i], payloads[i])
            for i in range(len(self.branches))
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)
    def clear_memory(self):
        for branch in self.branches:
            branch.clear_memory()
    def add_branch(self, branch: Agent|Tool|Workflow, schema: list[str] = [], position: int = -1) -> bool:
        if isinstance(branch, Agent):
            new_branch = AgentFlow(branch, schema)
        elif isinstance(branch, Tool):
            new_branch = ToolFlow(branch, schema)
        else:
            new_branch = branch
            if schema:
                new_branch.result_schema = schema
        self.branches.insert(position, new_branch)
        return True
    def remove_branch(self, position: int = -1) -> Workflow:
        return self.branches.pop(position)
    @abstractmethod
    def invoke(*args, **kwargs):
        pass

class Selector(FanFlow):
    def __init__(
        self, name: str, description: str, branches: List[Workflow],
        decider: LLMEngine|Agent|Workflow|Tool,
        result_schema: list[str] = [],
        return_raw: bool = False):
        super().__init__(name, description, branches)

        # Build decider Agent with SYSTEM role-prompt that lists branches
        self.is_internal_agent = isinstance(decider, LLMEngine)
        if self.is_internal_agent:
            self.decider = AgentFlow(
                Agent(
                    name=f"{name}::Selector",
                    description="Branch selection agent",
                    role_prompt=self._build_decider_system_prompt(),
                    llm_engine=decider,
                ),
                result_schema=[],
                return_raw=True
            )
        elif isinstance(decider, Agent):
            self.decider = AgentFlow(decider,
                                     result_schema=[],
                                     return_raw = True)
        elif isinstance(decider, Tool):
            self.decider = ToolFlow(decider, result_schema=[], return_raw=True)
        else:
            self.decider = decider
            self.decider._result_schema = []
            self.return_raw = True

    # ---- internal helpers ----

    def _build_decider_system_prompt(self) -> str:
        # name: description lines, one per branch (stable, readable)
        branch_lines = ",\n".join(
            f"{b.name}: {b.description}" for b in self.branches
        )
        return Prompts.CONDITIONAL_DECIDER_PROMPT.format(branches=branch_lines)

    def _update_decider_prompt(self) -> None:
        # refresh the decider's SYSTEM prompt when branch set changes
        self.decider.agent.role_prompt = self._build_decider_system_prompt()

    # ---- public API ----
    def add_branch(self, branch: Agent|Tool|Workflow, position: int = -1) -> None:
        FanFlow.add_branch(self, branch, position)
        if self.is_internal_agent:
            self._update_decider_prompt()

    def remove_branch(self, position: int = -1):
        removed = FanFlow.remove_branch(self, position)
        if self.is_internal_agent:
            self._update_decider_prompt()
        return removed

    def clear_memory(self):
        self.decider.clear_memory()
        FanFlow.clear_memory(self)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        logging.info(
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
            f"\n|   {self._name} Starting   |"
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
        )

        if not self.decider or not self.branches:
            raise ValueError("Decider and branches must be set.")

        logging.info(f"[WORKFLOW] Selecting branch via decider on {self._name}")

        # 1) Prefer the original shape (most flexible callers expect this).
        decision_name = self.decider.invoke(*args, **kwargs)
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
        if not self.return_raw:
            result = self.package_results(result)
        logging.info(
            f"\n+---{'-'*len(self._name + ' Finished')}---+"
            f"\n|   {self._name} Finished   |"
            f"\n+---{'-'*len(self._name + ' Finished')}---+\n"
        )
        return result

class Delegator(FanFlow):
    def __init__(self, name: str, description: str, branches: list[Agent | Workflow | Tool],
        task_master: LLMEngine | Agent | Tool | Workflow,
        result_schema:list[str] = [],
        return_raw:bool = False
    ):
        super().__init__(name, description, branches, result_schema, return_raw)

        # Build the task master once; keep only a boolean to know if it's our internal agent
        self._is_internal_agent: bool = isinstance(task_master, LLMEngine)
        default_schema = [b.name for b in self.branches]
        if self._is_internal_agent:
            # Internal agent we OWN; update its role-prompt whenever branches change
            internal_agent = Agent(
                name=f"{name}.delegator",
                description="Delegator decider (internal)",
                llm_engine=task_master,
                role_prompt=Prompts.DELEGATOR_SYSTEM_PROMPT,
            )
            self.task_master: Workflow = AgentFlow(internal_agent, [])
            self._refresh_internal_prompt()
        elif isinstance(task_master, Agent):
            self.task_master: Workflow = AgentFlow(task_master, [], True)
        elif isinstance(task_master, Tool):
            self.task_master: Workflow = ToolFlow(task_master, default_schema)
        elif isinstance(task_master, Workflow):
            self.task_master: Workflow = task_master
            self.task_master.return_raw = False
            self.task_master._result_schema = default_schema
        else:
            raise TypeError("delegator_component must be LLMEngine | Agent | Tool | Workflow")

    # ---------------- Branch management ----------------

    def add_branch(self, branch: Agent | Workflow | Tool, position: int = -1):
        FanFlow.add_branch(self, branch, position)
        if self._is_internal_agent:
            self._refresh_internal_prompt()

    def remove_branch(self, position: int = -1):
        removed = FanFlow.remove_branch(self, position)
        if self._is_internal_agent:
            self._refresh_internal_prompt()
        return removed

    def clear_memory(self):
        # Clear task_master memory if supported
        self.task_master.clear_memory()
        # Clear branches
        FanFlow.clear_memory(self)

    # ---------------- Internal helpers ----------------

    def _refresh_internal_prompt(self):
        """Update the internal decider's role-prompt to include branch list & rules."""
        branch_list = [{"name": b.name, "description": b.description} for b in self.branches]
        dynamic = (
            f"{Prompts.DELEGATOR_SYSTEM_PROMPT}\n\n"
            f"BRANCHES:\n{json.dumps(branch_list, ensure_ascii=False)}\n"
            f"(Remember: include every branch; use null to indicate skipping.)"
        )
        if isinstance(self.task_master, AgentFlow): self.task_master.agent.role_prompt = dynamic

    # ---------------- Public API ----------------

    def invoke(self, *args, **kwargs):
        logging.info(
            f"\n+---{'-' * (len(self._name) + 9)}---+"
            f"\n|   {self._name} Starting   |"
            f"\n+---{'-' * (len(self._name) + 9)}---+"
        )

        if not self.branches:
            raise ValueError("Delegator has no branches to execute.")

        raw = self.task_master.invoke(*args, **kwargs)
        logging.info(f"[WORKFLOW] {self.name} has created task assignments")
        # Normalize decider output to dict
        if isinstance(raw, dict):
            decider_output = raw
        elif isinstance(raw, str):
            cleaned = re.sub(r"```json(.*?)```", r"\1", raw, flags=re.DOTALL).strip()
            try:
                decider_output = json.loads(cleaned)
            except Exception as e:
                raise ValueError(f"Delegator decider returned a string that is not valid JSON dict: {e}\nOutput was:\n{raw}")
        else:
            raise TypeError(
                "Decider must return a dict (branch_name -> payload) or a JSON-string of that dict. "
                f"Instead, Decider recieved {raw} of type {type(raw)}"
            )

        # Validate/complete mapping to cover ALL branches
        payloads = [None] * len(self.branches)  # default None (skip)
        
        logging.info(f"[WORKFLOW] Handing out assignments")
        for i, b in enumerate(self.branches):
            if b.name not in decider_output:
                continue
            payloads[i] = decider_output[b.name]
        logging.info(f"[WORKFLOW] Executing tasks")
        try:
            # Use a fresh loop to avoid conflicts with any outer loops
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(self._fanout(payloads))
            finally:
                asyncio.set_event_loop(None)
                loop.close()
        except RuntimeError:
            # Fallback: if event loop policy prevents new loops, try asyncio.run
            results = asyncio.run(self._fanout(payloads))
        
        if not self.return_raw:
            results = self.package_results(results)
        elif not self.result_schema:
            results = {b.name:result for b, result in zip(self.branches, results)}
        logging.info(
            f"\n+---{'-' * (len(self._name) + 9)}---+"
            f"\n|   {self._name} Finished   |"
            f"\n+---{'-' * (len(self._name) + 9)}---+\n"
        )
        # Tuple aligned to branch order (includes None for skipped)
        return results