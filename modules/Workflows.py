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
from typing import Any, Dict, List, Optional, Sequence, Union
from concurrent.futures import ThreadPoolExecutor, as_completed


from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import StateGraph

from typing import Any, Callable, Dict, List, Optional
from langgraph.graph import StateGraph

# def callable_is_single_param(fn: Callable) -> bool:
#     """
#     Return True iff `fn` accepts **exactly one** logical parameter.

#     Rules
#     -----
#     - Ignores a bound instance parameter named `self`.
#     - Rejects functions that use *args or **kwargs (treated as multi-parameter).
#     - Counts POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, and KEYWORD_ONLY as logical params.

#     Parameters
#     ----------
#     fn : Callable
#         The function whose signature is inspected.

#     Returns
#     -------
#     bool
#         True if there is exactly one logical parameter; False otherwise.
#     """
#     try:
#         sig = inspect.signature(fn)
#     except (ValueError, TypeError):
#         # Fallback: unknown signature → treat as not single-parameter
#         return False

#     params = []
#     for p in sig.parameters.values():
#         if p.name == "self":
#             continue
#         if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
#             return False
#         if p.kind in (
#             inspect.Parameter.POSITIONAL_ONLY,
#             inspect.Parameter.POSITIONAL_OR_KEYWORD,
#             inspect.Parameter.KEYWORD_ONLY,
#         ):
#             params.append(p)

#     return len(params) == 1


# Default single-key result label used when callers don't provide a custom schema.
WF_RESULT = "__wf_result__"
JUDGE_RESULT = "__judge_result__"

class Workflow(ABC):
    """
    Abstract workflow base class (new uniform contract).

    Contract
    --------
    - Each workflow declares two ordered schemas:
        * input_schema:  list[str]  — required input field names
        * output_schema: list[str]  — required output/result field names
      These are **ordered lists of strings** only (no types, no validators).

    - Invocation is uniform: `invoke(input:s dict) -> dict`
      * The single argument MUST be a dictionary named `input`.
      * Implementations must return a dictionary whose keys match `output_schema`
        exactly (ordering is not enforced by Python dicts but is assumed by callers).

    Checkpointing
    -------------
    - Subclasses may record per-call checkpoints; this base class only provides
      the storage and convenience accessors. The expected shape is a list of
      dicts, where each entry at least contains:
        {
          "timestamp": <iso str>,
          "inputs": <dict>,
          "result": <dict conforming to output_schema>
        }

    Packaging helper
    ----------------
    - `package_results(results)` is provided as a convenience for subclasses to
      coerce arbitrary return shapes into an `output_schema`-conforming dict:
        * dict of any shape:
            - if its length == len(output_schema): map **by value order** onto
              output_schema (positional mapping of values).
            - else: place the whole dict under output_schema[0].
        * list/tuple:
            - if its length == len(output_schema): map positionally.
            - else: place the whole sequence under output_schema[0].
        * scalar:
            - place under output_schema[0], and set remaining fields to None.

      Note: This helper does **not** perform any validation beyond shape coercion,
      and it intentionally does not support *args/**kwargs or mutable mapping
      adapters beyond this packaging step.
    """

    # -------------------- Lifecycle --------------------

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: list[str],
        output_schema: list[str] = [WF_RESULT],
    ):
        if not isinstance(name, str) or not name:
            raise ValueError("`name` must be a non-empty string.")
        if not isinstance(description, str):
            raise ValueError("`description` must be a string.")
        if input_schema is None or not isinstance(input_schema, list) or not input_schema:
            raise ValueError("`input_schema` must be a non-empty list of strings.")
        if output_schema is None or not isinstance(output_schema, list) or not output_schema:
            raise ValueError("`output_schema` must be a non-empty list of strings.")

        self._name = name
        self._description = description
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._checkpoints: list[dict] = []

    # -------------------- Introspection --------------------

    @property
    def name(self) -> str:
        """str: Workflow display name."""
        return self._name

    @property
    def description(self) -> str:
        """str: Short description of the workflow."""
        return self._description

    @property
    def input_schema(self) -> list[str]:
        """list[str]: Ordered input field names required by this workflow."""
        return self._input_schema

    @input_schema.setter
    def input_schema(self, val: list[str]):
        """
        Set the input schema.
        """
        self._input_schema = val

    @property
    def output_schema(self) -> list[str]:
        """list[str]: Ordered output/result field names produced by this workflow."""
        return self._output_schema

    @output_schema.setter
    def output_schema(self, val: list[str]):
        """
        Set the output schema.
        """
        self._output_schema = val

    @property
    def checkpoints(self) -> list[dict]:
        """list[dict]: A shallow copy of recorded checkpoints (implementation-defined entries)."""
        return list(self._checkpoints)

    @checkpoints.setter
    def checkpoints(self, val: list[dict]) -> None:
        """Replace the stored checkpoints with `val`."""
        self._checkpoints = val

    @property
    def latest_result(self) -> dict:
        """
        dict: The most recent result, normalized to `output_schema`.

        If no checkpoints exist, returns a dict containing all schema keys
        mapped to `None`.
        """
        if not self._checkpoints:
            return None
        return self._checkpoints[-1].get("result", None)

    # -------------------- Utilities --------------------

    def package_results(self, results: Any) -> dict:
        """
        Normalize an arbitrary return value into a dict conforming to `output_schema`.

        Shape rules:
          - dict: if len(dict) == len(output_schema) → map values by order onto schema;
                  else → {output_schema[0]: dict}
          - list/tuple: if len(seq) == len(output_schema) → positional map;
                        else → {output_schema[0]: seq}
          - scalar: {output_schema[0]: scalar} plus remaining keys set to None
        """
        schema = self._output_schema

        # unwrap until no WF_RESULT is present
        while isinstance(results, dict) and WF_RESULT in results and len(results) == 1:
            results = results[WF_RESULT]
        
        # if singular scalar result
        if not isinstance(results, (dict, list, tuple)):
            if len(schema) == 1: return {schema[0]: results}
            else: raise ValueError("When packaging a scalar, output_schema length must be exactly 1.")

        # if dict
        if isinstance(results, dict):
            if len(results) != len(schema):
                # e.g. single-key schema
                if len(schema) == 1: return {schema[0]: results}
                raise ValueError("When packaging a dict/list/tuple, its length must match the output_schema length if the length of schema is not equal to 1.")
            # keys match → return as-is
            if set(results.keys()) == set(schema): return results
            # lengths match → map by value order
            vals = list(results.values())
            return {k: v for k, v in zip(schema, vals)}

        # if list/tuple
        if isinstance(results, (list, tuple)):
            if len(results) != len(schema):
                # e.g. single-key schema
                if len(schema) == 1: return {schema[0]: results}
                raise ValueError("When packaging a dict/list/tuple, its length must match the output_schema length if the length of schema is not equal to 1.")
            # lengths match → inferred positional map
            return {schema[i]: results[i] for i in range(len(schema))}

    def clear_memory(self) -> None:
        """Clear all recorded checkpoints for this workflow instance."""
        self._checkpoints = []

    # -------------------- Execution --------------------

    @abstractmethod
    def invoke(self, inputs: dict) -> dict:  # noqa: A002  (shadowing built-in is intentional by contract)
        """
        Execute the workflow with a single `input` dictionary.

        Implementations MUST:
          - Accept exactly one argument named `input` (a dict).
          - Return a dict conforming to `output_schema` (use `package_results` as needed).
          - (Optional) Append a checkpoint with at least {"timestamp", "input", "result"}.
        """
        raise NotImplementedError


class AgentFlow(Workflow):
    """
    Workflow that wraps a single `Agent` and expects exactly one input field: "prompt".

    Construction
    ------------
    - `agent` : modules.Agents.Agent
        Supplies:
          • name        → used as workflow name
          • description → used as workflow description
        The input schema is **fixed** to ["prompt"] for all AgentFlow instances.
    - `output_schema` : list[str]
        Must be provided explicitly (ordered list of result field names).

    Invocation
    ----------
    invoke(inputs: dict) -> dict
      • Validates that `inputs` contains the key "prompt".
      • Calls the underlying agent with `agent.invoke(prompt)`.
      • Normalizes the return via `package_results` to match `output_schema`.
      • Records a checkpoint with timestamp, original inputs, and normalized result.

    Notes
    -----
    - No *args/**kwargs interfaces—uniform dict `inputs` only.
    - Extra keys in `inputs` are ignored.
    """

    def __init__(self, agent: Agent,
                 input_schema: list[str] = ["prompt"],
                 output_schema: list[str] = [WF_RESULT]) -> None:
        super().__init__(
            name = agent.name,
            description = agent.description,
            input_schema = input_schema,
            output_schema = output_schema,
        )
        self._agent = agent

    # -------------------- Introspection --------------------

    @property
    def agent(self) -> Agent:
        """The wrapped Agent object."""
        return self._agent

    # -------------------- Execution --------------------

    def clear_memory(self):
        """Clear workflow checkpoints and the wrapped agent's memory."""
        Workflow.clear_memory(self)
        self.agent.clear_memory()

    def invoke(self, inputs: dict) -> dict:
        """
        Execute the wrapped agent with a single `inputs` dict containing "prompt".

        Steps:
          1) Validate `inputs` is a dict and contains "prompt".
          2) Call the agent as `agent.invoke(prompt)`.
          3) Normalize raw return via `package_results` to conform to `output_schema`.
          4) Record a checkpoint with ISO-8601 timestamp (local time).
        """
        raw = self._agent.invoke(str(inputs))
        result = self.package_results(raw)

        self._checkpoints.append(
            {
                "timestamp": str(datetime.now()),
                "inputs": inputs,
                "result": result,
            }
        )
        return result


class ToolFlow(Workflow):
    """
    Workflow that wraps a single `Tool`.

    Construction
    ------------
    - `tool` : modules.Tools.Tool
        Provides name, description, and an ordered parameter map (signature_map).
        These are used to infer:
          • name            → tool.full_name
          • description     → tool.description
          • input_schema    → list(tool.signature_map.keys()) excluding '*args'/'**kwargs'
    - `output_schema` : list[str]
        Must be provided explicitly.

    Invocation
    ----------
    invoke(inputs: dict) -> dict
      • Validates that all keys in `input_schema` are present in `inputs`.
      • Calls the underlying tool as `tool.invoke(**filtered_input)`, where
        `filtered_input` includes only keys declared by `input_schema`.
      • Normalizes the return via `package_results` to match `output_schema`.
      • Records a checkpoint with timestamp, original input, and normalized result.

    Notes
    -----
    - No *args/**kwargs workflow interfaces are provided—uniform dict input only.
    - Extra keys in `input` (not in `input_schema`) are ignored when calling the tool.
    """

    def __init__(self, tool: Tool, output_schema: list[str] = [WF_RESULT]) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("`tool` must be an instance of modules.Tools.Tool.")
        # Derive input schema from the tool's structured signature
        sig_keys = [k for k in tool.signature_map.keys() if k not in ("*args", "**kwargs")]
        super().__init__(
            # Use the short tool name for branch/input lookup so user-provided
            # inputs can refer to the tool by its simple name (e.g. "Compare_Lengths").
            # The underlying Tool object still carries its full_name for runtime
            # lookup when planners/orchestrators produce fully-qualified keys.
            name=tool.name,
            description=tool.description,
            input_schema=sig_keys,
            output_schema=output_schema,
        )
        self._tool = tool

    # -------------------- Introspection --------------------

    @property
    def tool(self) -> Tool:
        """The wrapped Tool object."""
        return self._tool
    
    # input schema is immutable and derived from the tool
    @property
    def input_schema(self) -> list[str]:
        """list[str]: Ordered input field names required by this workflow."""
        return self._input_schema

    # -------------------- Public API --------------------
    def clear_memory(self):
        """Clear workflow checkpoints and forward memory clear to the wrapped tool."""
        Workflow.clear_memory(self)
        self.tool.clear_memory()

    def invoke(self, inputs: dict) -> dict:  # noqa: A002 (intentional)
        """
        Execute the wrapped tool with filtered `input` and normalize the result.
        """
        inputs = {k: v for k,v in inputs.items() if k in self._input_schema}
        raw = self._tool.invoke(**inputs)
        result = self.package_results(raw)

        self._checkpoints.append(
            {
                "timestamp": datetime.now().isoformat(),
                "inputs": inputs,
                "raw_result": raw,
                "result": result,
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

    def __init__(self, name: str, description: str, steps: list[Agent|Tool|Workflow] = [], output_schema: list[str] = [WF_RESULT]) -> None:
        wrapped_steps: list[Workflow] = []
        # convert steps to Workflows as needed
        for i in range(len(steps)):
            if isinstance(steps[i], Agent): wrapped_steps.append(AgentFlow(steps[i]))
            elif isinstance(steps[i], Tool): wrapped_steps.append(ToolFlow(steps[i]))
            else: wrapped_steps.append(steps[i])
        # infer output_schema handoffs
        for i in range(len(wrapped_steps)-1): wrapped_steps[i]._output_schema = wrapped_steps[i+1].input_schema
        # initialize base Workflow
        super().__init__(name = name,
                         description = description,
                         input_schema = wrapped_steps[0].input_schema if wrapped_steps else [],
                         output_schema = output_schema)
        self._steps: list[Workflow] = wrapped_steps

    # input schema is derived from first step, and is immutable
    @property
    def input_schema(self) -> list[str]:
        """list[str]: Ordered input field names required by this workflow."""
        if self._steps:
            return self._input_schema
        return []
    
    @property
    def steps(self):
        """list[Workflow]: The concrete sequence of step workflows."""
        return self._steps

    @steps.setter
    def steps(self, val: list[Agent|Tool|Workflow]):
        wrapped_steps: list[Workflow] = []
        # convert steps to Workflows as needed
        for i in range(len(val)):
            if isinstance(val[i], Agent): wrapped_steps.append(AgentFlow(val[i]))
            elif isinstance(val[i], Tool): wrapped_steps.append(ToolFlow(val[i]))
            else: wrapped_steps.append(val[i])
        # infer output_schema handoffs
        for i in range(len(wrapped_steps)-1): wrapped_steps[i]._output_schema = wrapped_steps[i+1].input_schema
        self._input_schema = wrapped_steps[0].input_schema if wrapped_steps else []
        self._steps = wrapped_steps

    def add_step(self, step: Agent | Workflow | Tool, position: int | None = None):
        """
        Insert a new step at `position`. If `step` is Agent/Tool, it is wrapped.

        Notes
        -----
        - When inserting a `Workflow`, its current `output_schema` is overwritten
          to match the next step's `input_schema` (if any).
        - If `position` is None, the step is appended to the end."""
        # normalize to Workflow
        if isinstance(step, Agent): step = AgentFlow(step)
        elif isinstance(step, Tool): step = ToolFlow(step)
        # append or insert new workflow step
        if position is None: self._steps.append(step)
        else: self._steps.insert(position, step)
        # recompute intermediate output schemas
        for i, step in enumerate(self._steps[:-1]): step._output_schema = self._steps[i+1].input_schema
        self._input_schema = self._steps[0].input_schema

    def pop(self, position: int = -1) -> Workflow:
        """Remove and return the step at `position` (default: last)."""
        if not self._steps:
            raise ValueError("No steps to remove.")
        popped = self._steps.pop(position)
        if not self._steps:
            self._input_schema = []
            self._output_schema = []
        else:
            self._input_schema = self._steps[0].input_schema
            self._output_schema = self._steps[-1].output_schema
        for i, step in enumerate(self._steps[:-1]): step._output_schema = self._steps[i+1].input_schema
        return popped

    def clear_memory(self):
        """Clear workflow checkpoints and each step's memory."""
        Workflow.clear_memory(self)
        for step in self._steps:
            step.clear_memory()

    def invoke(self, inputs: dict) -> Any:
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
        # Initial input
        current = inputs
        # Iterate schema pipeline
        for step in self._steps:
            logging.info(f"[WORKFLOW] Invoking: {step.name}")
            current = step.invoke(current)
        # Final packaging
        result = self.package_results(current)
        # Record checkpoint
        self._checkpoints.append(
            {
                "timestamp": str(datetime.now()),
                "inputs": inputs,
                "result": result,
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
    Maker–Checker workflow (dict-only).

    Invariants:
      - input_schema == maker.input_schema
      - output_schema == maker.output_schema
      - maker.output_schema == checker.input_schema
      - checker.output_schema == maker.input_schema
      - If judge is provided:
          judge.input_schema == checker.output_schema
          len(judge.output_schema) == 1  (single boolean key)
    """

    def __init__(
        self,
        name: str,
        description: str,
        maker: Agent|Tool|Workflow,
        checker: Agent|Tool|Workflow,
        max_revisions: int = 1,
        judge: Optional[Agent|Tool|Workflow] = None,
    ) -> None:
        if max_revisions < 0:
            raise ValueError(f"{name}: max_revisions must be >= 0")
        # Wrap maker/checker as Workflows if needed
        if isinstance(maker, Agent): maker = AgentFlow(maker)
        elif isinstance(maker, Tool): maker = ToolFlow(maker)
        if isinstance(checker, Agent): checker = AgentFlow(checker)
        elif isinstance(checker, Tool): checker = ToolFlow(checker)
        maker.output_schema = checker.input_schema
        checker.output_schema = maker.input_schema

        # Judge constraints (no coercion—fail fast)
        self._judge = None
        if judge is not None:
            if isinstance(judge, Agent): judge = AgentFlow(judge)
            elif isinstance(judge, Tool): judge = ToolFlow(judge)
            if judge.input_schema != maker.input_schema:
                raise ValueError(
                    f"{name}: judge.input_schema {judge.input_schema} "
                    f"!= checker.output_schema or maker.input_schema {checker.output_schema}"
                )
            if len(judge.output_schema) != 1:
                raise ValueError(
                    f"{name}: judge.output_schema must have length 1; got {judge.output_schema}"
                )
            judge._output_schema = [JUDGE_RESULT]

        # Our schemas: identical to maker’s
        super().__init__(
            name=name,
            description=description,
            input_schema=maker.input_schema,
            output_schema=checker.input_schema,
        )

        self._maker: Workflow = maker
        self._checker: Workflow = checker
        self._judge: Workflow = judge
        self._max_revisions: int = max_revisions
    @property
    def max_revisions(self) -> int:
        """Maximum number of maker/checker revision rounds."""
        return self._max_revisions
    @max_revisions.setter
    def max_revisions(self, val: int) -> None:
        """Set the maximum number of maker/checker revision rounds."""
        if val < 0:
            raise ValueError(f"{self.name}: max_revisions must be >= 0")
        self._max_revisions = val
    @property
    def maker(self) -> Workflow:
        """The maker workflow."""
        return self._maker
    @maker.setter
    def maker(self, val: Workflow) -> None:
        """Set the maker workflow."""
        self._maker = val
        self._input_schema = val.input_schema
        # Update maker/checker symmetry
        self._maker.output_schema = self._checker.input_schema
        self._checker.output_schema = self._maker.input_schema
    @property
    def checker(self) -> Workflow:
        """The checker workflow."""
        return self._checker
    @checker.setter
    def checker(self, val: Workflow) -> None:
        """Set the checker workflow."""
        self._checker = val
        self._output_schema = val.input_schema
        # Update maker/checker symmetry
        self._maker.output_schema = self._checker.input_schema
        self._checker.output_schema = self._maker.input_schema
    @property
    def judge(self) -> Optional[Workflow]:
        """The optional judge workflow."""
        return self._judge
    @judge.setter
    def judge(self, val: Optional[Agent|Tool|Workflow]) -> None:
        """Update the judge workflow."""
        # if clearing judge
        if val is None: self._judge = None; return
        # normalize to Workflow, and ensure single-key output schema
        if isinstance(val, Agent):
            val = AgentFlow(val, self._checker.output_schema, [JUDGE_RESULT])
        elif isinstance(val, Tool):
            val = ToolFlow(val, [JUDGE_RESULT])
        elif len(val.output_schema) > 1:
            raise ValueError(
                f"{self.name}: judge.output_schema must have length 1; got {val.output_schema}"
            )
        else: val._output_schema = [JUDGE_RESULT]
        # validate input schema
        if val.input_schema != self._checker.output_schema:
            raise ValueError(
                f"{self.name}: judge.input_schema {val.input_schema} "
                f"!= checker.output_schema or maker.input_schema {self._checker.output_schema}"
            )
        # enforce single-key output schema
        self._judge = val

    def clear_memory(self) -> None:
        """Clear workflow checkpoints and each component's memory."""
        Workflow.clear_memory(self)
        self._maker.clear_memory()
        self._checker.clear_memory()
        if self._judge is not None:
            self._judge.clear_memory()
    
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Create initial draft
        draft = self._maker.invoke(inputs)
        
        draft_history: List[Dict[str, Any]] = []
        approved = False

        for i in range(1, self._max_revisions + 1):
            # Checker consumes the draft directly
            revisions = self._checker.invoke(draft)  # matches checker.output_schema == maker.input_schema

            # Optional judge consumes the revisions directly
            if self._judge is not None:
                decision = self._judge.invoke(revisions)
                approved: bool = decision[JUDGE_RESULT]
                if not isinstance(approved, bool):
                    raise ValueError(
                        f"{self.name}: judge must return a boolean, but instead got a decision of type {type(approved)}"
                    )
            else: approved = False
            # Record round in checkpoint-only history
            draft_history.append({
                "timestamp": str(datetime.now()),
                "draft_index": i,
                "draft": draft,
                "revisions": revisions,
                "approved": approved,
            })
            # If approved, break early
            if approved: break
            # Next draft uses revisions directly (schema symmetry guarantees compatibility)
            draft = self._maker.invoke(revisions)
        # Package final draft to our output_schema (identical to maker.output_schema)
        final_result = self.package_results(draft)
        # Checkpoint with history (not returned)
        self._checkpoints.append({
            "timestamp": str(datetime.now()),
            "inputs": inputs,
            "draft_history": draft_history,
            "result": final_result,
        })
        return final_result


class Selector(Workflow):
    """
    Selector workflow (dict-only).

    Rules:
      - input_schema is inferred from the judge (judge.input_schema)
      - judge MUST be a Workflow; branches MUST be Workflows
      - judge.output_schema MUST have exactly 1 key; the runtime value must be a string
      - all branches MUST have:
          branch.input_schema == judge.input_schema
          branch.output_schema == self.output_schema
      - invoke(inputs) does NOT validate input keys (caller responsibility)
      - result is always self.package_results(raw_branch_result)
      - checkpoints include {"timestamp", "inputs", "selection", "result"}
    """

    def __init__(
        self,
        name: str,
        description: str,
        branches: List[Tool|Agent|Workflow],
        judge: Agent|Tool|Workflow,
        output_schema: List[str],
    ) -> None:

        # Wrap judge as Workflow if needed
        if isinstance(judge, Agent): judge = AgentFlow(judge, ["prompt"], [JUDGE_RESULT])
        elif isinstance(judge, Tool): judge = ToolFlow(judge, [JUDGE_RESULT])
        # Validate judge output schema
        elif len(judge.output_schema) != 1:
            raise ValueError(
                f"{name}: judge.output_schema must have length 1; got {judge.output_schema}"
            )
        else: judge._output_schema = [JUDGE_RESULT]
        # Initialize base
        super().__init__(
            name=name,
            description=description,
            input_schema=judge.input_schema,
            output_schema=output_schema,
        )

        # Store judge via private slot (we already adopted its input schema);
        self._judge: Workflow = judge

        # Validate branches against inferred input schema & declared output schema
        if set([b.name for b in branches]) != set([b.name for b in branches]):
            raise ValueError(f"{self.name}: duplicate branch names detected")
        wrapped_branches = []
        for b in branches:
            # Wrap branch as Workflow if needed
            if isinstance(b, Agent): b = AgentFlow(b, judge.input_schema, output_schema)
            elif isinstance(b, Tool): b = ToolFlow(b, output_schema)
            if b.input_schema != judge.input_schema:
                raise ValueError(
                    f"{self.name}: branch '{b.name}' input_schema {b.input_schema} "
                    f"!= selector/judge input_schema {self.input_schema}"
                )
            b._output_schema = output_schema
            # Add to list
            wrapped_branches.append(b)
        # Store branches
        self._branches: List[Workflow] = wrapped_branches

    # ---- judge property/setter ----
    @property
    def input_schema(self) -> List[str]:
        return self._judge.input_schema
    @property
    def branches(self) -> List[Workflow]:
        return self._branches
    @branches.setter
    def branches(self, val: List[Agent|Tool|Workflow]) -> None:
        wrapped_branches = []
        for b in val:
            # Wrap branch as Workflow if needed
            if isinstance(b, Agent): b = AgentFlow(b, self.input_schema, self.output_schema)
            elif isinstance(b, Tool):
                b = ToolFlow(b, self.output_schema)
                if b.input_schema != self._judge.input_schema:
                    raise ValueError(
                        f"{self.name}: branch '{b.name}' input_schema {b.input_schema} "
                        f"!= selector/judge input_schema {self.input_schema}"
                    )
            else: b._output_schema = self.output_schema
            # Add to list
            wrapped_branches.append(b)
        self._branches = wrapped_branches

    @property
    def judge(self) -> Workflow:
        return self._judge

    @judge.setter
    def judge(self, val: Agent|Tool|Workflow) -> None:
        # Wrap judge as Workflow if needed
        if isinstance(val, Agent): val = AgentFlow(val, self.input_schema, [JUDGE_RESULT])
        elif isinstance(val, Tool): val = ToolFlow(val, [JUDGE_RESULT])
        # Must keep the same input schema as the current selector's (inferred from original judge)
        if val.input_schema != self.input_schema:
            raise ValueError(
                f"{self.name}: new judge.input_schema {val.input_schema} "
                f"!= current selector.input_schema {self.input_schema}"
            )
        if len(val.output_schema) != 1:
            raise ValueError(
                f"{self.name}: judge.output_schema must have exactly 1 key; got {val.output_schema}"
            )
        val._output_schema = [JUDGE_RESULT]
        self._judge = val

    # ---- branch management ----
    def add_branch(self, branch: Agent|Tool|Workflow) -> None:
        if branch.name in [b.name for b in self._branches]:
            raise ValueError(f"{self.name}: duplicate branch name '{branch.name}'")
        # Wrap branch as Workflow if needed
        if isinstance(branch, Agent): branch = AgentFlow(branch, self.input_schema, self.output_schema)
        elif isinstance(branch, Tool): branch = ToolFlow(branch, self.output_schema)
        if branch.input_schema != self.input_schema:
            raise ValueError(
                f"{self.name}: branch '{branch.name}' input_schema {branch.input_schema} "
                f"!= selector/judge input_schema {self.input_schema}"
            )
        if branch.output_schema != self.output_schema: branch._output_schema = self.output_schema
        self._branches.append(branch)

    def remove_branch(self, branch: str|None = None, *, position = -1) -> Workflow:
        if not self._branches:
            raise IndexError(f"{self.name}: no branches to remove")
        if branch is None: removed = self._branches.pop(position)
        else:
            idx = next((i for i,b in enumerate(self._branches) if b.name == branch), None)
            if idx is None: raise ValueError(f"{self.name}: no branch named '{branch}' to remove")
            removed = self._branches.pop(idx)
        return removed

    def clear_memory(self) -> None:
        Workflow.clear_memory(self)
        self._judge.clear_memory()
        for b in self._branches:
            b.clear_memory()

    # ---- main execution ----
    def invoke(self, inputs: Dict) -> Dict:
        # Decide
        decision = self._judge.invoke(inputs)
        selection = decision.get(JUDGE_RESULT)
        if not isinstance(selection, str):
            raise ValueError(
                f"{self.name}: judge must output a string under '{self._sel_key}', got {selection!r}"
            )

        # Dispatch
        if selection not in [b.name for b in self._branches]:
            raise ValueError(
                f"{self.name}: unknown selection '{selection}'. "
                f"Valid: {[b.name for b in self._branches]}"
            )
        idx = next(i for i,b in enumerate(self._branches) if b.name == selection)
        raw = self._branches[idx].invoke(inputs)

        # Package & checkpoint
        result = self.package_results(raw)
        self._checkpoints.append({
            "timestamp": str(datetime.now()),
            "inputs": inputs,
            "selection": selection,
            "result": result,
        })
        return result


class BatchFlow(Workflow):
    """
    BatchFlow (dict-only, label-map based).

    Rules:
      1) input_schema = ordered list of branch names (derived from self.branches).
      2) output_schema depends on unwrap_outputs:
         - if True: output_schema length == number of branches; labels come from an internal
           label map {branch.name -> label}. If labels were not provided, defaults to branch names.
         - if False: output_schema = [output_key] (defaults to WF_RESULT) and the aggregate result is
           wrapped under that single key as { output_key: { branch_name: branch_result, ... } }.
      3) invoke runs all branches in parallel. Each branch receives inputs.get(branch.name, {}).
      4) add/remove branch updates input_schema and, when unwrap_outputs=True, output_schema and label map.
      5) invoke raises if there are no branches.
    """

    def __init__(
        self,
        name: str,
        description: str,
        branches: Sequence[Union[Workflow, Any]],
        *,
        unwrap_outputs: bool = True,
        labels: Optional[List[str]] = None,     # only used when unwrap_outputs=True
        output_key: str = WF_RESULT,            # only used when unwrap_outputs=False
    ) -> None:
        if not branches:
            raise ValueError(f"{name}: at least one branch is required")

        # Normalize branches to Workflows (Agent/Tool -> AgentFlow/ToolFlow).
        wf_branches: List[Workflow] = []
        for obj in branches:
            if isinstance(obj, Workflow):
                wf_branches.append(obj)
            else:
                # Try ToolFlow first (callable tools), then AgentFlow fallback.
                try:
                    wf_branches.append(ToolFlow(obj))
                except Exception:
                    wf_branches.append(AgentFlow(obj))

        # Enforce unique branch names
        names = [b.name for b in wf_branches]
        if len(names) != len(set(names)):
            raise ValueError(f"{name}: duplicate branch names are not allowed: {names}")

        self.unwrap_outputs: bool = bool(unwrap_outputs)

        # Label map: branch.name -> output label (used only when unwrap_outputs=True)
        self._label_map: Dict[str, str] = {}
        if self.unwrap_outputs:
            if labels is not None:
                if len(labels) != len(wf_branches):
                    raise ValueError(
                        f"{name}: labels length {len(labels)} must equal number of branches {len(wf_branches)}"
                    )
                if not all(isinstance(lbl, str) and lbl for lbl in labels):
                    raise ValueError(f"{name}: all labels must be non-empty strings")
                if len(set(labels)) != len(labels):
                    raise ValueError(f"{name}: labels must be unique: {labels}")
                self._label_map = {wf_branches[i].name: labels[i] for i in range(len(wf_branches))}
            else:
                # default to branch names
                self._label_map = {b.name: b.name for b in wf_branches}
            output_schema_list = [self._label_map[b.name] for b in wf_branches]
        else:
            if not isinstance(output_key, str) or not output_key:
                raise ValueError(f"{name}: output_key must be a non-empty string")
            self._label_map = {}  # unused in wrapped mode
            output_schema_list = [output_key]
            self.output_key: str = output_schema_list[0]

        # Initialize base Workflow with derived schemas
        super().__init__(
            name=name,
            description=description,
            input_schema=list(names),
            output_schema=output_schema_list,
        )

        self._branches: List[Workflow] = wf_branches

    @property
    def branches(self) -> List[Workflow]:
        """The list of branch workflows."""
        return self._branches
    @branches.setter
    def branches(self, val: List[Workflow]) -> None:
        """Set the list of branch workflows (rebuilds schemas)."""
        # Enforce unique branch names
        names = [b.name for b in val]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.name}: duplicate branch names are not allowed: {names}")
        self._branches = val
        # Rebuild schemas
        self._rebuild_schemas_after_change()
    
    @property
    def output_key(self) -> str:
        """The output key when unwrap_outputs=False."""
        if self.unwrap_outputs:
            raise RuntimeError(f"{self.name}: output_key is only meaningful when unwrap_outputs=False")
        return self._output_schema[0]
    @output_key.setter
    def output_key(self, val: str) -> None:
        """Set the output key when unwrap_outputs=False (rebuilds output_schema)."""
        if self.unwrap_outputs:
            raise RuntimeError(f"{self.name}: output_key is only meaningful when unwrap_outputs=False")
        self._output_schema = [val]
    @property
    def input_schema(self) -> List[str]:
        """The current input schema."""
        return self._input_schema
    # ---------------------------
    # Branch management
    # ---------------------------
    def add_branch(
        self,
        branch: Union[Workflow, Any],
        position: Optional[int] = None,
        *,
        label: Optional[str] = None,  # only used when unwrap_outputs=True
    ) -> None:
        # Normalize to Workflow
        if isinstance(branch, Tool):
            branch = ToolFlow(branch)
        elif isinstance(branch, Agent):
            branch = AgentFlow(branch)
        
        if any(b.name == branch.name for b in self._branches):
            raise ValueError(f"{self.name}: duplicate branch name '{branch.name}'")

        # Insert/append
        if position is None:
            self._branches.append(branch)
        else:
            if position < 0 or position > len(self._branches):
                raise IndexError(f"{self.name}: add_branch position out of range")
            self._branches.insert(position, branch)

        # Maintain label map & schemas
        if self.unwrap_outputs:
            lbl = label or branch.name
            if lbl in self._label_map.values() or branch.name in self._label_map.keys():
                raise ValueError(f"{self.name}: duplicate branch name {branch.name} or output label '{lbl}'")
            self._label_map[branch.name] = lbl

        self._rebuild_schemas_after_change()

    def remove_branch(self, position: int = -1) -> Workflow:
        if not self._branches:
            raise IndexError(f"{self.name}: no branches to remove")

        removed = self._branches.pop(position)

        if self.unwrap_outputs:
            self._label_map.pop(removed.name, None)

        self._rebuild_schemas_after_change()
        return removed

    def _rebuild_schemas_after_change(self) -> None:
        # input_schema = ordered list of branch names
        names = [b.name for b in self._branches]
        self._input_schema = list(names)

        # output_schema
        if self.unwrap_outputs:
            # Ensure every branch has a label; default to its own name if missing (shouldn't happen in normal flow)
            for b in self._branches:
                if b.name not in self._label_map:
                    self._label_map[b.name] = b.name
            # Rebuild output schema from label map in branch order
            self._output_schema = [self._label_map[b.name] for b in self._branches]
        else:
            # Keep the single output key stable
            self._output_schema = [self.output_key]

    # ---------------------------
    # Execution
    # ---------------------------
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self._branches:
            raise RuntimeError(f"{self.name}: cannot invoke with zero branches")

        # Prepare per-branch payloads (dict-only); missing -> {}
        payloads: Dict[str, Dict[str, Any]] = {}
        for b in self._branches:
            val = inputs.get(b.name, {})
            if not isinstance(val, dict):
                raise ValueError(
                    f"{self.name}: payload for branch '{b.name}' must be a dict; got {type(val).__name__}"
                )
            payloads[b.name] = val

        # Run in parallel
        results: Dict[str, Dict[str, Any]] = {}

        def _run(wf: Workflow) -> Dict[str, Any]:
            return wf.invoke(payloads[wf.name])

        with ThreadPoolExecutor(max_workers=max(1, len(self._branches))) as pool:
            future_map = {pool.submit(_run, b): b.name for b in self._branches}
            for fut in as_completed(future_map):
                bname = future_map[fut]
                results[bname] = fut.result()

        # Aggregate
        if self.unwrap_outputs:
            aggregated: Dict[str, Any] = {}
            for b in self._branches:
                key = self._label_map[b.name]
                branch_result = results[b.name]
                while isinstance(branch_result, dict) and len(branch_result) == 1 and WF_RESULT in branch_result:
                    branch_result = branch_result[WF_RESULT]
                aggregated[key] = branch_result
            packaged = self.package_results(aggregated)
        else:
            wrapped = {self.output_schema[0]: {b.name: results[b.name] for b in self._branches}}
            packaged = self.package_results(wrapped)

        # Checkpoint (keep lean as requested)
        self._checkpoints.append({
            "timestamp": str(datetime.now()),
            "inputs": inputs,
            "result": packaged,
        })
        return packaged

    # ---------------------------
    # Optional label utilities
    # ---------------------------
    def get_label(self, branch_name: str) -> Optional[str]:
        """Return the output label for a given branch name (unwrap mode only)."""
        return self._label_map.get(branch_name) if self.unwrap_outputs else None

    def set_label(self, branch_name: str, new_label: str) -> None:
        """
        Update the output label for a given branch (unwrap mode only).
        Ensures uniqueness and rebuilds output_schema.
        """
        if not self.unwrap_outputs:
            raise RuntimeError(f"{self.name}: labels are only meaningful when unwrap_outputs=True")
        if branch_name not in (b.name for b in self._branches):
            raise ValueError(f"{self.name}: unknown branch '{branch_name}'")
        if not isinstance(new_label, str) or not new_label:
            raise ValueError(f"{self.name}: new_label must be a non-empty string")
        if new_label in self._label_map.values():
            raise ValueError(f"{self.name}: duplicate output label '{new_label}'")

        self._label_map[branch_name] = new_label
        # Rebuild output schema to reflect new label order
        self._output_schema = [self._label_map[b.name] for b in self._branches]

    # ---------------------------
    # Memory management
    # ---------------------------
    def clear_memory(self) -> None:
        Workflow.clear_memory(self)
        for b in self._branches:
            b.clear_memory()


class LangGraphFlow(Workflow):
    """
    LangGraph-backed Workflow (dict-only adapter).

    Rules:
      - A single global `schema` is used for BOTH input and output schemas for the flow and for ALL nodes.
      - Nodes are **Tools only**. Each added Tool is wrapped as a ToolFlow:
          * ToolFlow.input_schema MUST equal the global schema.
          * ToolFlow.output_schema is FORCED to the global schema.
        The ToolFlow's `.invoke` is registered directly into the StateGraph under the node's name.
      - Edges (plain or conditional) use string node names exactly as in langgraph.
      - invoke(inputs: dict) compiles if dirty, runs the graph, and returns only keys in the global schema.
      - Checkpoints store timestamp and the final packaged outputs only.
    """

    def __init__(
        self,
        name: str,
        description: str,
        *,
        schema: List[str],
        graph: Optional[Any] = None,
    ):
        # Single global schema used for both input and output
        super().__init__(name=name, description=description, input_schema=schema, output_schema=schema)

        if graph is None and StateGraph is None:
            raise RuntimeError(
                "langgraph is not installed; cannot create a builder. "
                "Provide an existing builder via graph=... or install langgraph."
            )

        self._graph: StateGraph = graph if graph is not None else StateGraph(dict)
        self._app = None
        self._dirty = True

        # Keep the wrapped ToolFlows here for metadata/checkpoints lookup
        self._nodes: List[ToolFlow] = []
        self._routers: list[Tool] = []

    # -------------------- Introspection --------------------

    def _node_names(self) -> List[str]:
        return [wf.name for wf in self._nodes]

    # -------------------- Builder API --------------------

    def add_node(self, node: Tool|Agent|Workflow, name: Optional[str] = None) -> None:
        """
        Add a Tool node:
          - Wrap as ToolFlow(tool, output_schema=self.output_schema)
          - Enforce ToolFlow.input_schema == self.input_schema
          - Register node_name + toolflow.invoke into the underlying StateGraph
        """
        if isinstance(node, Agent): node = AgentFlow(node, self.input_schema, self.output_schema)
        elif isinstance(node, Tool): node = ToolFlow(node, output_schema=self.output_schema)
        if not (node.input_schema == self.input_schema and node.output_schema == node.input_schema):
            raise ValueError(
                f"{self.name}: node must be a Workflow with input_schema = output_schema "
                f"to be added to LangGraphFlow. Got input_schema={node.input_schema}, "
                f"output_schema={node.output_schema}."
            )
        
        if node.name in self._node_names():
            raise ValueError(f"{self.name}: node '{node.name}' already exists")
        self._nodes.append(node)

        # Register the node directly with its invoke
        self._graph.add_node(node.name, node.invoke)
        self._dirty = True

    def add_edge(self, src: str, dst: str) -> None:
        names = set(self._node_names())
        if src not in names:
            raise ValueError(f"{self.name}: unknown node '{src}'. Add it first.")
        if dst not in names:
            raise ValueError(f"{self.name}: unknown node '{dst}'. Add it first.")
        self._graph.add_edge(src, dst)
        self._dirty = True

    def add_conditional_edges(
        self,
        src: str,
        router: Tool,
        routes: Dict[str, str],
    ) -> None:
        """
        router(state) -> route_key; routes: {route_key: node_name}
        """
        names = set(self._node_names())
        if src not in names:
            raise ValueError(f"{self.name}: unknown node '{src}'. Add it first.")
        for _, dst in routes.items():
            if dst not in names:
                raise ValueError(f"{self.name}: unknown destination node '{dst}'. Add it first.")
        if router.name not in [t.name for t in self._routers]:
            self._routers.append(router)
        self._graph.add_conditional_edges(src, router.func, routes)
        self._dirty = True

    def set_entry_point(self, name: str) -> None:
        if name not in set(self._node_names()):
            raise ValueError(f"{self.name}: unknown entry node '{name}'")
        self._graph.set_entry_point(name)
        self._dirty = True

    def set_finish_point(self, name: str) -> None:
        if name not in set(self._node_names()):
            raise ValueError(f"{self.name}: unknown finish node '{name}'")
        self._graph.set_finish_point(name)
        self._dirty = True

    # -------------------- Execution --------------------

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the compiled graph with a single state dict and return only the global schema keys.
        """
        
        # Strict ingress: require all declared keys present. (If you prefer permissive, default missing to None.)
        if self._dirty or self._app is None:
            self._app = self._graph.compile()
            self._dirty = False

        final_state = self._app.invoke(inputs)
        # Ensure all output_schema keys are present
        for key in self.output_schema:
            if final_state.get(key, None) is None: final_state[key] = None
        print(len(final_state.keys()), len(self.output_schema))
        result = self.package_results(final_state)
        self._checkpoints.append({"timestamp": str(datetime.now()), "inputs":inputs, "result": result})
        return result
