"""
Workflows
=========

A small suite of **stateful** workflow primitives that wrap Agents, Tools, or
other Workflows to compose larger behaviors. These classes focus on **plumbing**
(data-shape adaptation, result packaging, and light bookkeeping) and deliberately
avoid any provider-specific logic.

Key concepts
------------
- All workflows expose a uniform `invoke(*args, **kwargs) -> dict` contract whose
  return object always conforms to `result_schema` (list of string keys).
- Each workflow keeps an internal `checkpoints` list with call metadata and a
  normalized snapshot of the last result. The latest normalized result is reachable
  by `latest_result`.
- The constant `WF_RESULT` ("__wf_result__") is the *default* single-key schema
  name used when the result is a single logical value.

Classes
-------
Workflow (abstract)
    Base class that defines schema management, result packaging, and checkpointing.
AgentFlow
    Wraps an `Agent` as a one-step workflow, forwarding `invoke`.
ToolFlow
    Wraps a `Tool` as a one-step workflow, forwarding `invoke`.
ChainFlow
    Linear composition of steps (each a Workflow/Agent/Tool) with shape-adapting
    handoff between steps.
MakerChecker
    Maker–Checker pattern (optionally with early approval) over two Workflows.
Selector
    Branching workflow that delegates to one of several branches based on a decider.
BatchFlow
    Simple parallel fan-out/fan-in that runs branches concurrently and aggregates results.

Notes
-----
- This file only improves documentation clarity. No functional changes were made.
"""

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
from datetime import datetime

import inspect
from typing import Callable


def callable_is_single_param(fn: Callable) -> bool:
    """
    Return True iff `fn` accepts **exactly one** logical parameter.

    Rules
    -----
    - Ignores a bound instance parameter named `self`.
    - Rejects functions that use *args or **kwargs (treated as multi-parameter).
    - Counts POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, and KEYWORD_ONLY as logical params.

    Parameters
    ----------
    fn : Callable
        The function whose signature is inspected.

    Returns
    -------
    bool
        True if there is exactly one logical parameter; False otherwise.
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        # Fallback: unknown signature → treat as not single-parameter
        return False

    params = []
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            return False
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            params.append(p)

    return len(params) == 1


# Default single-key result label used when callers don't provide a custom schema.
WF_RESULT = "__wf_result__"


class Workflow(ABC):
    """
    Abstract workflow base class.

    Responsibilities
    ----------------
    - Maintain a human-readable `name` and `description`.
    - Enforce/output a declared `result_schema` (list of key names).
    - Record per-call `checkpoints` with args/kwargs, timestamp, and normalized result.
    - Provide `package_results(...)` to coerce arbitrary returns into the schema shape.
    - Expose `latest_result` as a dict matching `result_schema`.

    Shape rules for `package_results`
    ---------------------------------
    - If the raw result is a `dict` and its **length matches** `result_schema`,
      keys are mapped **by order** of values into the schema (positional mapping).
      Otherwise the whole dict is stored under the first schema key.
    - If the raw result is a list/tuple and its length matches `result_schema`,
      elements are mapped index-wise; otherwise the whole object goes under the first key.
    - Any other type is placed under the **first** schema key; remaining keys get `None`.

    Parameters
    ----------
    name : str
        Display name for logs and selection.
    description : str
        Short description of the workflow's purpose.
    result_schema : list[str], default [WF_RESULT]
        Keys that define the normalized return dict shape.
    """

    def __init__(self, name: str, description: str, result_schema: list[str] = [WF_RESULT]):
        self._name = name
        self._description = description
        self._result_schema = result_schema
        self._is_single_param = False  # internal hint for handoff shape in composites
        self._checkpoints: list = []

        for s in self._result_schema:
            if not s:
                raise ValueError("Empty strings are not permissible output schema parameter names")

    # ---- Properties ----------------------------------------------------------
    @property
    def name(self) -> str:
        """str: Workflow display name."""
        return self._name

    @property
    def checkpoints(self) -> list:
        """list: A shallow copy of the recorded checkpoints."""
        return list(self._checkpoints)

    @checkpoints.setter
    def checkpoints(self, val: list) -> None:
        """Replace the stored checkpoints with `val`."""
        self._checkpoints = val

    @property
    def latest_result(self) -> dict:
        """
        dict: The most recent result (normalized to `result_schema`).

        If no checkpoints exist, returns a dict containing all schema keys
        mapped to `None`.
        """
        return self._checkpoints[-1]["result"] if self._checkpoints else {k: None for k in self._result_schema}

    @property
    def description(self) -> str:
        """str: Short description of the workflow."""
        return self._description

    @property
    def result_schema(self) -> list[str]:
        """list[str]: The declared output keys for normalized results."""
        return self._result_schema

    @result_schema.setter
    def result_schema(self, val: list[str]):
        """
        Set the result schema.

        Raises
        ------
        ValueError
            If `val` is None or empty.
        """
        if val == None or not len(val):
            raise ValueError("Result schema must be set to a non-empty list of strings")
        self._result_schema = val

    # ---- Utilities -----------------------------------------------------------
    def package_results(self, results: Any) -> dict:
        """
        Normalize an arbitrary return value into a dict conforming to `result_schema`.

        See class docstring *Shape rules for `package_results`* for details.
        """
        if isinstance(results, dict):
            if len(results) != len(self._result_schema):
                return {self._result_schema[0]: results}
            vals = list(results.values())
            return {k: v for k, v in zip(self._result_schema, vals)}

        if isinstance(results, (list, tuple)):
            if len(results) != len(self._result_schema):
                return {self._result_schema[0]: results}
            return {self._result_schema[i]: results[i] for i in range(len(self._result_schema))}

        final = {}
        for i, k in enumerate(self._result_schema):
            if i == 0:
                final[k] = results
                continue
            final[k] = None
        return final

    def clear_memory(self) -> None:
        """Clear all recorded checkpoints for this workflow instance."""
        self._checkpoints = []

    # ---- Contract ------------------------------------------------------------
    @abstractmethod
    def invoke(self, *args: Any, **kwargs: Any) -> dict:
        """
        Execute the workflow.

        The argument handling is workflow-specific; the returned object must always be a
        dict that conforms to `result_schema` (use `package_results` as needed).
        """
        pass


class AgentFlow(Workflow):
    """
    Wrap an `Agent` instance as a single-step workflow.

    Invocation
    ----------
    - Forwards `invoke(*args, **kwargs)` directly to `agent.invoke(...)`.
    - If the agent raises a `TypeError` due to incompatible args, it retries once
      with a stringified payload: `"*args:{args}\\n**kwargs:{kwargs}"`.

    Memory
    ------
    - `clear_memory()` resets both the workflow checkpoints **and** the agent history.
    """

    def __init__(self, agent: Agent, result_schema: list[str] = [WF_RESULT]):
        super().__init__(name=agent.name, description=agent.description, result_schema=result_schema)
        self.agent: Agent = agent
        self._is_single_param = True

    def clear_memory(self):
        """Clear workflow checkpoints and the wrapped agent's memory."""
        Workflow.clear_memory(self)
        self.agent.clear_memory()

    def invoke(self, *args: Any, **kwargs: Any) -> dict:
        """
        Forward to `Agent.invoke` and normalize into `result_schema`.

        Returns
        -------
        dict
            A dict shaped per `result_schema`. The checkpoint stores the *unwrapped*
            value if the only key is `WF_RESULT`.
        """
        logging.info(
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
            f"\n|   {self._name} Starting   |"
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
        )
        try:
            result = self.agent.invoke(*args, **kwargs)
        except TypeError as e:
            logging.info(f"Failed to pass in arguments as is, giving '{e}'. Stringifying...")
            result = self.agent.invoke(f"*args:{args}\n**kwargs:{kwargs}")

        result = self.package_results(result)
        self._checkpoints.append(
            {
                "args": args,
                "kwargs": kwargs,
                "timestamp": str(datetime.now()),
                "result": result if WF_RESULT not in result else result[WF_RESULT],
            }
        )
        logging.info(
            f"\n+---{'-'*len(self._name + ' Finished')}---+"
            f"\n|   {self._name} Finished   |"
            f"\n+---{'-'*len(self._name + ' Finished')}---+\n"
        )
        return result


class ToolFlow(Workflow):
    """
    Wrap a `Tool` as a single-step workflow.

    Behavior
    --------
    - Calls `tool.invoke(*args, **kwargs)`.
    - Determines whether the tool accepts a single argument using
      `callable_is_single_param(tool.func)` (used by composite workflows).
    - Normalizes the return using `package_results`.
    """

    def __init__(self, tool: Tool, result_schema: list[str] = [WF_RESULT]):
        super().__init__(tool.name, tool.description, result_schema)
        self.tool = tool
        self._is_single_param = callable_is_single_param(tool.func)

    def clear_memory(self):
        """Clear workflow checkpoints and forward memory clear to the wrapped tool."""
        Workflow.clear_memory(self)
        self.tool.clear_memory()

    def invoke(self, *args: Any, **kwargs: Any) -> dict:
        """
        Invoke the underlying function and bind its return into a dict matching
        the keys of `result_schema`.
        """
        # Raw results
        result = self.tool.invoke(*args, **kwargs)
        # Package into the declared result schema
        result = self.package_results(result)
        self._checkpoints.append(
            {
                "args": args,
                "kwargs": kwargs,
                "timestamp": str(datetime.now()),
                "result": result if WF_RESULT not in result else result[WF_RESULT],
            }
        )
        return result


class ChainFlow(Workflow):
    """
    Linear composition of steps with shape-aware handoff.

    Handoff semantics
    -----------------
    - The first step receives the original `*args, **kwargs`.
    - If a step's result is a dict containing only `WF_RESULT`, its value is treated as
      the *logical* output for routing to the next step.
    - For subsequent steps:
        * If the receiving step expects a single param (`_is_single_param=True`), pass
          the logical output as a single argument.
        * If the logical output is a `list`/`tuple`, expand positionally.
        * If the logical output is a `dict` (non `WF_RESULT`-wrapped), expand as **kwargs.
        * Otherwise pass as a single positional argument.

    Parameters
    ----------
    steps : list[Tool | Agent | Workflow]
        Items are wrapped to Workflows (ToolFlow/AgentFlow) unless already a Workflow.
    unpack_midsteps : bool
        Preserved legacy flag (handoff behavior is as described above).
    """

    def __init__(
        self,
        name: str,
        description: str,
        steps: list[Tool | Agent | Workflow] = [],
        unpack_midsteps: bool = True,
        result_schema: list[str] = [WF_RESULT],
    ):
        super().__init__(name, description, result_schema)
        self._steps: list[Workflow] = []
        self._unpack_midsteps = unpack_midsteps
        for step in steps:
            if isinstance(step, Tool):
                new_step = ToolFlow(step)
            elif isinstance(step, Agent):
                new_step = AgentFlow(step)
            else:
                new_step = step
            self._steps.append(new_step)
        self._is_single_param = self._steps[0]._is_single_param

    @property
    def steps(self):
        """list[Workflow]: The concrete sequence of step workflows."""
        return self._steps

    @steps.setter
    def steps(self, val: list[Workflow]):
        """Replace the underlying steps with `val`."""
        self._steps = val

    def insert_step(self, step: Agent | Workflow | Tool, schema: list[str] = [WF_RESULT], position: int | None = None):
        """
        Insert a new step at `position`. If `step` is Agent/Tool, it is wrapped.

        Notes
        -----
        - When inserting a `Workflow`, its current `result_schema` is overwritten
          with `schema` to ensure downstream compatibility.
        """
        if isinstance(step, Agent):
            step = AgentFlow(step, schema)
        elif isinstance(step, Tool):
            step = ToolFlow(step, schema)
        elif isinstance(step, Workflow):
            step._result_schema = schema
        if position is None:
            self._steps.append(step, schema)
        else:
            self._steps.insert(position, step)
        self._is_single_param = self._steps[0]._is_single_param

    def pop(self, position: int = -1) -> Workflow:
        """Remove and return the step at `position` (default: last)."""
        if not self._steps:
            raise ValueError("No steps to remove.")
        popped = self._steps.pop(position)
        self._is_single_param = self._steps[0]._is_single_param
        return popped

    def clear_memory(self):
        """Clear workflow checkpoints and each step's memory."""
        Workflow.clear_memory(self)
        for step in self._steps:
            step.clear_memory()

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute steps in order, adapting intermediate shapes as described in
        *Handoff semantics*. The final normalized dict is checkpointed and returned.
        """
        logging.info(
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
            f"\n|   {self._name} Starting   |"
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
        )
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
                if step._is_single_param:
                    current = step.invoke(mid_result)
                elif isinstance(mid_result, (list, tuple)):
                    current = step.invoke(*mid_result)
                elif isinstance(mid_result, dict):
                    current = step.invoke(**mid_result)
                continue

            # Otherwise, treat it as keyword arguments
            current = step.invoke(**current)

        result = self.package_results(current)
        self._checkpoints.append(
            {
                "args": args,
                "kwargs": kwargs,
                "timestamp": str(datetime.now()),
                "result": result if WF_RESULT not in result else result[WF_RESULT],
            }
        )
        logging.info(
            f"\n+---{'-'*len(self._name + ' Finished')}---+"
            f"\n|   {self._name} Finished   |"
            f"\n+---{'-'*len(self._name + ' Finished')}---+\n"
        )
        return result


class MakerChecker(Workflow):
    """
    Maker–Checker pattern over two Workflows, with optional early approval.

    Flow
    ----
    1) `maker.invoke(...)` produces an initial draft.
    2) For up to `max_revisions` rounds:
       a) `checker.invoke(draft)` yields revision notes (shape-adapted).
       b) If `early_stop` is provided, it receives the revision notes and returns
          a single-boolean dict to approve/reject early.
       c) When approved, exit; otherwise feed revisions back to `maker` to produce
          the next draft.

    Result
    ------
    Returns a dict (normalized via `package_results`) containing a 2-tuple:
      (rounds, draft)
    where:
      - `rounds` is a list of dicts with keys: `approved`, `revisions`, `draft`.
      - `draft` is the final draft object (possibly the approved one).
    """

    def __init__(
        self,
        name: str,
        description: str,
        maker: Workflow,
        checker: Workflow,
        early_stop: Agent | Tool | Workflow = None,
        max_revisions: int = 1,
        result_schema: list[str] = [WF_RESULT],
    ):
        super().__init__(name, description, result_schema)
        if max_revisions < 0:
            raise ValueError("max_revisions must be >= 0")
        self._is_single_param = maker._is_single_param
        self.max_revisions = max_revisions
        self.maker: Workflow = maker
        self.checker: Workflow = checker
        if not early_stop:
            self.early_stop = None
        elif isinstance(early_stop, Agent):
            self.early_stop = AgentFlow(early_stop)
        elif isinstance(early_stop, Tool):
            self.early_stop = ToolFlow(early_stop)
        else:
            self.early_stop = early_stop

        if self.early_stop and len(self.early_stop.result_schema) > 1:
            raise ValueError(
                "The Early stop workflow component should only have a single key "
                "to track whether or the current revision notes warrant approving "
                "the current draft or not. A single key for a single boolean."
            )

    def clear_memory(self):
        """Clear checkpoints and internal component memories (maker/checker/early_stop)."""
        Workflow.clear_memory(self)
        self.maker.clear_memory()
        self.checker.clear_memory()
        if self.early_stop:
            self.early_stop.clear_memory()

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the Maker–Checker loop up to `max_revisions`. See class docstring for flow.
        """
        logging.info(
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
            f"\n|   {self._name} Starting   |"
            f"\n+---{'-'*len(self._name + ' Starting')}---+"
        )
        rounds: list[dict] = []

        logging.info(f"{self.name} Making the first draft")
        draft = self.maker.invoke(*args, **kwargs)

        for i in range(1, self.max_revisions + 1):
            logging.info(f"{self.name} Reviewing draft {i}")
            approved = False

            # Unwrap single-key default result
            if WF_RESULT in draft:
                draft = draft[WF_RESULT]

            # Checker invocation (shape-adapted)
            if self.checker._is_single_param: revisions = self.checker.invoke(draft)
            elif isinstance(draft, dict): revisions = self.checker.invoke(**draft)
            elif isinstance(draft, (tuple, list)): revisions = self.checker.invoke(*draft)

            if WF_RESULT in revisions:
                revisions = revisions[WF_RESULT]

            # Optional early approval
            if self.early_stop:
                logging.info(f"[WORKFLOW] {self.early_stop.name} is marking checking if revisions warrant early approval")
                if self.early_stop and self.early_stop._is_single_param: approver_res = self.early_stop.invoke(revisions)
                elif isinstance(revisions, dict): approver_res = self.early_stop.invoke(**revisions)
                elif isinstance(revisions, (tuple, list)): approver_res = self.early_stop.invoke(*revisions)
            approved = self.early_stop != None and approver_res[list(approver_res.keys())[0]]

            rounds.append({"approved": approved, "revisions": revisions, "draft": draft})

            if approved:
                logging.info(f"[WORKFLOW] {self.name} Approved draft {i}")
                break

            # Make a new revised draft
            logging.info(f"{self.name} Making draft {i+1}")
            if self.maker._is_single_param: draft = self.maker.invoke(revisions)
            elif isinstance(revisions, dict): draft = self.maker.invoke(**revisions)
            elif isinstance(revisions, (tuple, list)): draft = self.maker.invoke(*revisions)

        result = self.package_results((rounds, draft))
        self._checkpoints.append(
            {
                "args": args,
                "kwargs": kwargs,
                "timestamp": str(datetime.now()),
                "result": result if WF_RESULT not in result else result[WF_RESULT],
            }
        )
        logging.info(
            f"\n+---{'-'*len(self._name + ' Finished')}---+"
            f"\n|   {self._name} Finished   |"
            f"\n+---{'-'*len(self._name + ' Finished')}---+\n"
        )
        return result


class Selector(Workflow):
    """
    Branching workflow: ask a decider which branch name to execute.

    Components
    ----------
    branches : list[Workflow]
        Candidate paths; input payload is forwarded unchanged to the chosen branch.
    decider : Agent | Workflow | Tool
        A single-key result producer whose value is expected to be a `str` exactly
        matching the `name` of one of the branches.

    Behavior
    --------
    - `invoke(*args, **kwargs)` first calls the decider with the original payload.
    - The decider's single-key value must be a `str` (branch name); that branch
      receives the **original** payload and is executed.
    - The chosen branch's result is normalized and checkpointed.
    """

    def __init__(
        self,
        name: str,
        description: str,
        branches: list[Workflow],
        decider: Agent | Workflow | Tool,
        result_schema: list[str] = [WF_RESULT],
    ):
        super().__init__(name, description, result_schema)
        # Initialize the choice agents
        self.branches: list[Workflow] = []
        for branch in branches:
            if isinstance(branch, Agent): self.branches.append(AgentFlow(branch))
            elif isinstance(branch, Tool): self.branches.append(ToolFlow(branch))
            else: self.branches.append(branch)
        if isinstance(decider, Agent):self.decider = AgentFlow(decider)
        elif isinstance(decider, Tool): self.decider = ToolFlow(decider)
        else: self.decider = decider
        if len(self.decider._result_schema) > 1:
            raise ValueError(
                "The decider workflow must have a result schema of length 1, as we only "
                "expect a single string corresponding to the name of one of the branches."
            )
        self._is_single_param = self.decider._is_single_param

    def add_branch(self, branch: Agent | Tool | Workflow, schema: list[str] = [WF_RESULT], position: int | None = None) -> None:
        """
        Add a branch to `branches`, wrapping as needed and overriding its schema.

        Parameters
        ----------
        branch : Agent | Tool | Workflow
            The new branch to add.
        schema : list[str]
            The branch's `result_schema` to enforce.
        position : int | None
            If provided, insert at this index; otherwise append.
        """
        if isinstance(branch, Agent): new_branch = AgentFlow(branch)
        elif isinstance(branch, Tool): new_branch = ToolFlow(branch)
        else: new_branch = branch
        new_branch._result_schema = schema
        if not position: self.branches.append(new_branch)
        else: self.branches.insert(position, new_branch)

    def remove_branch(self, position: int = -1):
        """Remove and return the branch at `position` (default: last)."""
        return self.branches.pop(position)

    def clear_memory(self):
        """Clear checkpoints, decider memory, and each branch's memory."""
        Workflow.clear_memory(self)
        self.decider.clear_memory()
        for branch in self.branches: branch.clear_memory(self)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """
        Run the decider, route to the selected branch, normalize, checkpoint, return.
        """
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
            raise TypeError(
                f"{decision_name} is not an instance of type 'str'. "
                f"Check {self.decider.name}'s return type"
            )
        logging.info(f"[WORKFLOW] Decider chose: {decision_name}")

        # Route to the chosen branch with the ORIGINAL payload
        selected: Workflow = None
        for branch in self.branches:
            if branch.name == decision_name:
                selected = branch
                break
        if not selected:
            raise ValueError(f"Decider chose an unknown branch: {decision_name}")

        result = selected.invoke(*args, **kwargs)
        result = self.package_results(result)
        self._checkpoints.append(
            {
                "args": args,
                "kwargs": kwargs,
                "timestamp": str(datetime.now()),
                "result": result if WF_RESULT not in result else result[WF_RESULT],
            }
        )
        logging.info(
            f"\n+---{'-'*len(self._name + ' Finished')}---+"
            f"\n|   {self._name} Finished   |"
            f"\n+---{'-'*len(self._name + ' Finished')}---+\n"
        )
        return result


class BatchFlow(Workflow):
    """
    Simple parallel workflow: run multiple branches concurrently and aggregate results.

    Invocation semantics
    --------------------
    - Positional args map to branches by position: `args[0] → branches[0]`, etc.
    - Keyword args may target specific branches **by branch name** and override
      positional payloads for those branches.
    - Each branch is a Workflow (wrapping Agent/Tool as needed in the constructor).

    Result schema rules
    -------------------
    - `result_schema` must either have length **1** or length **== number of branches**.
      - If length == 1: the list of branch results is returned under that single key.
      - If length == number_of_branches: each branch result is mapped positionally.
    - If `result_schema` is empty, the return value is a dict mapping
      `branch.name -> branch_result`.
    """

    def __init__(self, name: str, description: str, branches: list[Agent | Tool | Workflow], result_schema: list[str] = [WF_RESULT]):
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
        """Clear checkpoints and each branch's memory."""
        Workflow.clear_memory(self)
        for b in self.branches:
            b.clear_memory()

    def invoke(self, *args: Any, **kwargs: Any) -> dict:
        """
        Fan out to branches concurrently, normalize each result, then aggregate and return.

        Concurrency
        -----------
        - Uses an event loop created on demand (with a fallback to `asyncio.run` if
          loop creation is restricted).
        - Each branch call is executed in a thread pool via `run_in_executor(None, ...)`
          to avoid blocking the event loop with synchronous branch invocations.

        Payload normalization per branch
        --------------------------------
        - If the branch expects a single parameter: pass the payload as a single arg.
        - If the payload is a list/tuple: expand positionally.
        - If the payload is a dict: expand as keyword args.
        - If the branch returns `{WF_RESULT: value}`, the value is unwrapped in the
          aggregated list to avoid nested `{WF_RESULT: ...}` wrappers.
        """
        logging.info(f"[PARALLEL] {self._name} Starting")

        # Build payloads per-branch from positionals and keyword overrides
        payloads = [None] * len(self.branches)
        for i in range(len(self.branches)):
            if i < len(args):
                payloads[i] = args[i]
            else:
                payloads[i] = None

        for i, b in enumerate(self.branches):
            if b.name in kwargs:
                payloads[i] = kwargs[b.name]

        async def _fanout(payloads: list[Any]):
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
            # Fallback for environments that disallow new event loops
            results = asyncio.run(_fanout(payloads))

        # Normalize results: unwrap {WF_RESULT: X} to X for readability/consistency
        normalized = []
        for r in results:
            if isinstance(r, dict) and WF_RESULT in r:
                normalized.append(r[WF_RESULT])
            else:
                normalized.append(r)

        # Aggregate per declared schema (length validated in __init__)
        result = self.package_results(normalized)
        self._checkpoints.append(
            {
                "args": args,
                "kwargs": kwargs,
                "timestamp": str(datetime.now()),
                "result": result if WF_RESULT not in result else result[WF_RESULT],
            }
        )
        logging.info(f"[PARALLEL] {self._name} Finished")
        return result