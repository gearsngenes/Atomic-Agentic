# ============================================================
# Workflows
# ============================================================
"""
Workflows
=========

Workflows are instantiated, stateful orchestration objects that accept a **single mapping** of inputs
and return a **mapping** shaped by an explicit `output_schema`. They provide deterministic boundary
control around Tools, Agents, or other Workflows via:

• `output_schema: list[str]` — the exact keys a workflow returns (required; defaults to ["__wf_result__"]
  when not provided to the constructor).
• `bundle_all: bool` — optional single-key envelope that wraps otherwise non-conforming results.

Inputs are delegated to the wrapped component(s). Workflows do **not** own an input schema. For
documentation and UI generation, wrappers expose a **read-only** `arguments_map` that mirrors the
wrapped component’s declared input contract.

Packaging Rules (in order)
--------------------------
1) Repeatedly unwrap `{__wf_result__: ...}` to prevent nested envelopes in composites.
2) If the raw result is a mapping:
   • Exact match to `output_schema` → reorder to schema order and return.
   • Subset of `output_schema` → pad missing keys with `None` and return.
   • Otherwise → if `bundle_all` and single-key schema, wrap; else error.
3) If the raw result is a sequence/iterable (non-string):
   • If single-key schema → put the entire list under that key.
   • If lengths match → zip positionally.
   • If shorter → pad trailing keys with `None`.
   • If longer → error.
4) If the raw result is a set:
   • If single-key schema → wrap as a list under that key.
   • Else → error.
5) If scalar (including str):
   • If single-key schema → wrap under the sole key.
   • Else → error.

Bundling (`bundle_all=True`) is applied **after** mapping alignment to avoid double-enveloping. It
requires a single-key `output_schema`.

This module provides:
• `Workflow`  — base class with deterministic packaging & checkpointing.
• `AgentFlow` — wraps an `Agent`, proxies `arguments_map`, forwards mapping to `agent.invoke`.
• `ToolFlow`  — wraps a `Tool`,  proxies `arguments_map`, forwards mapping to `tool.invoke`.
"""

from __future__ import annotations

# =========================
# Standard Library
# =========================
from abc import ABC, abstractmethod
from dataclasses import is_dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from collections.abc import Mapping, Iterable, Sequence, Set
from collections import OrderedDict
import time
import asyncio, threading
# =========================
# Atomic-Agentic Modules
# =========================
from .Agents import Agent
from .Tools import Tool

# =========================
# Constants & Exceptions
# =========================
WF_RESULT = "__wf_result__"
JUDGE_RESULT = "__judge_result__"


class WorkflowError(Exception):
    """Base class for workflow-related errors."""


class ValidationError(WorkflowError, ValueError):
    """Raised for input/type validation failures."""


class SchemaError(ValidationError):
    """Raised when `output_schema` is malformed or incompatible with options."""


class PackagingError(ValidationError):
    """Raised when a raw result cannot be normalized to `output_schema`."""


class ExecutionError(WorkflowError, RuntimeError):
    """Raised when a wrapped component fails during `_process_inputs`."""


# =========================
# Helpers
# =========================
def _is_namedtuple(x: Any) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_fields")


def _namedtuple_as_mapping(x: Any) -> Dict[str, Any]:
    return {f: getattr(x, f) for f in x._fields}


MAPPABLE_KINDS = (Mapping,)
STRINGISH = (str, bytes, bytearray)


# =========================
# Base Workflow
# =========================
class Workflow(ABC):
    """
    Deterministic orchestration boundary.

    Public API
    ----------
    invoke(inputs: Mapping) -> Dict[str, Any]
        Executes `_process_inputs(inputs)` in the subclass, then packages the raw
        result into a dict shaped by `output_schema`.

    Checkpointing
    -------------
    Each `invoke` appends a checkpoint dict to `self._checkpoints` with:
      • time     — ISO timestamp (ms precision)
      • inputs   — the original inputs (as passed)
      • raw      — the raw result returned by `_process_inputs`
      • result   — the packaged output (dict)

    Notes
    -----
    • Workflows do **not** declare/validate an input schema.
    • Subclasses that wrap a single component should expose a read-only `arguments_map`
      property for documentation (no setter).
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        *,
        output_schema: Optional[List[str]] = None,
        bundle_all: bool = True,
    ) -> None:
        self._name = str(name)
        self._description = str(description)
        # Output schema is not optional as a private field; default when absent.
        self._output_schema: List[str] = list(output_schema) if output_schema else [WF_RESULT]
        self._bundle_all: bool = bool(bundle_all)
        self._checkpoints: List[Dict[str, Any]] = []

        self._validate_output_schema(self._output_schema)
        if self._bundle_all and len(self._output_schema) != 1:
            raise SchemaError(f"{self._name}: bundle_all=True requires single-key output_schema")

    # ---------- Introspection ----------
    @property
    def name(self) -> str:
        return self._name
    @name.setter
    def name(self, val: str)-> None:
        self._name = val

    @property
    def description(self) -> str:
        return self._description
    @description.setter
    def description(self, val: str) -> None:
        self._description = val

    @property
    def checkpoints(self) -> list[dict]:
        return self._checkpoints
    
    def clear_memory(self) -> None:
        self._checkpoints = []

    @property
    @abstractmethod
    def arguments_map(self) -> OrderedDict[str, Any]:
        """
        Read-only mapping describing the expected inputs of the wrapped component.
        Base class does not implement it; wrappers should proxy to the component.
        """
        raise NotImplementedError
    
    @property
    def input_schema(self) -> List[str]:
        return [k for k in self.arguments_map]

    # ---------- Boundary controls ----------
    @property
    def output_schema(self) -> List[str]:
        return list(self._output_schema)

    @output_schema.setter
    def output_schema(self, value: List[str]) -> None:
        self._validate_output_schema(value)
        # if new output schema != 1, then not bundling
        self._bundle_all = self._bundle_all and len(value) == 1
        self._output_schema = list(value)

    @property
    def bundle_all(self) -> bool:
        return self._bundle_all

    @bundle_all.setter
    def bundle_all(self, value: bool) -> None:
        value = bool(value)
        if value and len(self._output_schema) != 1:
            raise SchemaError(f"{self._name}: bundle_all requires single-key output_schema")
        self._bundle_all = value

    # ---------- Serialization ----------
    def to_dict(self) -> OrderedDict[str, Any]:
        """
        JSON-friendly spec: no kind/component refs.
        """
        return OrderedDict({
            "name": self.name,
            "description": self.description,
            "arguments_map": OrderedDict(self.arguments_map),
            "output_schema": list(self._output_schema),
            "bundle_all": bool(self._bundle_all),
        })

    # ---------- Execution ----------
    def invoke(self, inputs: Mapping[str, Any]) -> OrderedDict[str, Any]:
        start = time.time()
        timestamp = datetime.now()
        if not isinstance(inputs, Mapping):
            raise ValidationError(f"{self._name}: inputs must be a mapping, got {type(inputs).__name__}")
        raw = self._process_inputs(dict(inputs))
        result = self.package_results(raw)
        end = time.time()
        
        self._checkpoints.append(
            {
                "timestamp": timestamp.isoformat(timespec="milliseconds"),
                "inputs": inputs,
                "raw": raw,
                "result": result,
                "duration": end - start
            }
        )
        return result

    # ---------- Packaging ----------
    def package_results(self, results: Any) -> OrderedDict[str, Any]:
        """
        Normalize `results` to a dict matching `self._output_schema`, unwrapping WF_RESULT
        first and preferring mapping alignment before any bundling.
        """
        schema = self._output_schema
        if not schema:
            raise SchemaError(f"{self._name}: output_schema is empty")

        # Named records → mapping
        if _is_namedtuple(results):
            results = _namedtuple_as_mapping(results)
        elif is_dataclass(results):
            results = asdict(results)

        # Unwrap {WF_RESULT: ...} repeatedly
        while isinstance(results, MAPPABLE_KINDS) and len(results) == 1 and WF_RESULT in results:
            results = results[WF_RESULT]

        # Mapping path (exact → subset)
        if isinstance(results, MAPPABLE_KINDS):
            results:OrderedDict = OrderedDict(dict(results))
            res_keys = set(results.keys())
            sch_keys = set(schema)
            if res_keys == sch_keys:
                # reorder to schema order
                return OrderedDict(results)
            if res_keys.issubset(sch_keys):
                # pad missing with None
                return OrderedDict({k: results.get(k, None) for k in schema})
            if len(res_keys) == len(sch_keys):
                return OrderedDict({self.output_schema[i] : results[list(results.keys())[i]] for i in range(len(self.output_schema))})
            if len(res_keys) < len(sch_keys):
                filled: OrderedDict = OrderedDict()
                for k in schema: filled[k] = None
                for i, k in enumerate(list(results.keys())): filled[schema[i]] = results[k]
                return filled
            # if bundle_all
            if self.bundle_all:
                if len(schema) != 1:
                    raise PackagingError(f"{self._name}: bundle_all requires single-key schema")
                return {schema[0]: dict(results)}
            raise PackagingError(
                f"{self._name}: Failed to map keys {results.keys()} not to output schema {self.output_schema}; "
                f"change the schema or enable bundle_all under a single key"
            )

        # Sets (unordered)
        if isinstance(results, Set) and not isinstance(results, STRINGISH):
            if len(schema) == 1:
                return {schema[0]: list(results)}
            raise PackagingError(f"{self._name}: set-like results require single-key schema or pre-coercion")

        # Sequences / other iterables (non-string)
        if isinstance(results, (Sequence, Iterable)) and not isinstance(results, STRINGISH):
            # package whole iter/seq inside a single key if bundling
            if self.bundle_all:
                return {schema[0]: results}
            results = list(results)
            # If more keys than schema covers, raise
            if len(results) > len(schema):
                raise PackagingError(
                    f"{self._name}: sequence length {len(results)} exceeds schema length {len(schema)}"
                )
            # If shorter than schema, pad out
            if len(results) < len(schema):
                results = results + [None] * (len(schema) - len(results))
            return {k: v for k, v in zip(schema, results)}
        # Scalar fallback
        if len(schema) == 1:
            return {schema[0]: results}
        raise PackagingError(
            f"{self._name}: cannot package type {type(results).__name__} into multi-key schema"
        )

    # ---------- Extension point ----------
    @abstractmethod
    def _process_inputs(self, inputs: Dict[str, Any]) -> Any:
        """
        Subclasses must implement: perform the wrapped work and return a raw result.
        The base class will package it according to `output_schema`/`bundle_all`.
        """
        raise NotImplementedError

    # ---------- Validators ----------
    @staticmethod
    def _validate_output_schema(schema: List[str]) -> None:
        if not isinstance(schema, list) or not schema or not all(isinstance(k, str) for k in schema):
            raise SchemaError("output_schema must be a non-empty list[str]")


# =========================
# AgentFlow
# =========================
class AgentFlow(Workflow):
    """
    Wraps a single `Agent`.

    • Inputs are forwarded **as a mapping** to `agent.invoke(inputs)`.
    • `arguments_map` is a **read-only** proxy to `agent.arguments_map`.
    • Outputs are normalized by the base `Workflow` packager.
    """

    def __init__(
        self,
        agent: Agent,
        name: str | None = None,
        *,
        description: str = "",
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
    ) -> None:
        super().__init__(
            name=name or agent.name,
            description=agent.description or description,
            output_schema=output_schema,
            bundle_all=bundle_all,
        )
        self._agent = agent

    @property
    def agent(self) -> Agent:
        return self._agent
    @agent.setter
    def agent(self, val: Agent) -> None:
        self._agent = val

    @property
    def arguments_map(self) -> Mapping[str, Any]:
        return self.agent.arguments_map

    def _process_inputs(self, inputs: Dict[str, Any]) -> Any:
        # Dict-only invoke; Agent handles its own validation & pre-invoke shaping.
        return self._agent.invoke(inputs)
    
    def clear_memory(self):
        super().clear_memory()
        self.agent.clear_memory()
    
    def to_dict(self) -> OrderedDict[str, Any]:
        base = super().to_dict()
        base.update({"agent" : self.agent.to_dict()})
        return base


# =========================
# ToolFlow
# =========================
class ToolFlow(Workflow):
    """
    Wraps a single `Tool`.

    • Inputs are forwarded **as a mapping** to `tool.invoke(inputs)` (no kwargs expansion).
    • `arguments_map` is a **read-only** proxy to `tool.arguments_map`.
    • Outputs are normalized by the base `Workflow` packager.
    """

    def __init__(
        self,
        tool: Tool | Callable,
        name: Optional[str] = None,
        *,
        description: str = "",
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
    ) -> None:
        if isinstance(tool, Callable):
            tool = Tool(
                func=tool,
                name=(name or tool.__name__) or "unnamed_callable",
                description= (description or tool.__doc__) or "",
                type="function",
                source="default",
            )
        super().__init__(
            name=name or tool.name,
            description=tool.description,
            output_schema=output_schema,
            bundle_all=bundle_all,
        )
        self._tool = tool

    @property
    def tool(self) -> Tool:
        return self._tool
    @tool.setter
    def tool(self, val: Tool|Callable) -> None:
        if isinstance(val, Callable):
            val = Tool(
                func=val,
                name=self.name,
                description= (self.description or val.__doc__) or "",
                type="function",
                source="default",
            )
        self._tool = val

    @property
    def arguments_map(self) -> Mapping[str, Any]:
        return self._tool.arguments_map

    def _process_inputs(self, inputs: Dict[str, Any]) -> Any:
        # Dict-only invoke; Tool handles its own validation & binding.
        return self._tool.invoke(inputs)
    
    def to_dict(self) -> OrderedDict[str, Any]:
        base = super().to_dict()
        base.update(
            {"tool":self.tool.to_dict()}
        )
        return base
    

# wraps all incoming objects into workflow classes and adjusts their input/output schemas based on optional parameters
def _to_workflow(obj: Callable | Agent | Tool | Workflow) -> Workflow:
        if isinstance(obj, Workflow): return obj
        if isinstance(obj, Agent): return AgentFlow(obj)
        if isinstance(obj, Tool|Callable): return ToolFlow(obj)
        raise ValidationError(f"Object must be Agent, Tool, Callable, or Workflow. Got unexpected '{type(obj).__name__}'.")


class ChainFlow(Workflow):
    """
    Sequentially composes a list of steps (ToolFlow | AgentFlow | Workflow), forwarding the
    mapping output of each step to the next step's input.

    Key properties
    --------------
    • `arguments_map` (read-only): proxies the *first* step's `arguments_map`. If no steps,
      returns an empty OrderedDict.
    • `output_schema` / `bundle_all` (overlay): define the final, normalized shape of the
      ChainFlow's result. The **last step** is force-aligned to these overlay values during
      reconciliation to avoid schema drift.

    Reconciliation (pairwise A → B)
    -------------------------------
    For each adjacent pair, we set A.output_schema := B.input_schema under these rules:
      1) If A.out ⊆ B.in → set A.out := B.in
      2) Else if A.bundle_all → set A.out := B.in
      3) Else if len(A.out) == len(B.in) → set A.out := B.in
      4) Else → raise ValidationError

    Empty chain behavior
    --------------------
    • `arguments_map` → OrderedDict()
    • `_process_inputs` → raises ExecutionError

    Notes
    -----
    • Workflows do **not** own an input schema; we derive a step's `input_schema` from
      its `arguments_map` (read-only) when reconciling hand-offs.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        steps: Optional[List[Callable|Tool|Agent|Workflow]] = None,  # Tool | Agent | Workflow accepted; normalized via _to_workflow
        *,
        output_schema: Optional[List[str]] = None,
        bundle_all: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            output_schema=output_schema,
            bundle_all=bundle_all,
        )
        self._steps: List[Workflow] = []
        self._steps = [ _to_workflow(s) for s in steps ]
        self._reconcile_all()

    # ---------------------------- documentation proxy ----------------------------

    @property
    def arguments_map(self) -> OrderedDict[str, Any]:
        # Empty chain → empty arg map, so derived input_schema (elsewhere) is []
        if not self._steps:
            return OrderedDict()
        return self._steps[0].arguments_map  # read-only proxy

    # ------------------------------- step management ------------------------------

    @property
    def steps(self) -> List[Workflow]:
        return list(self._steps)

    @steps.setter
    def steps(self, new_steps: Optional[List[Callable|Tool|Agent|Workflow]]) -> None:
        self._steps = [ _to_workflow(s) for s in (new_steps or []) ]
        self._reconcile_all()

    def add_step(self, step: Callable|Tool|Agent|Workflow, *, position: int | None = None) -> None:
        wf = _to_workflow(step)
        if position is None:
            self._steps.append(wf)
        else:
            self._steps.insert(position, wf)
        self._reconcile_all()

    def pop(self, index: int = -1) -> Optional[Workflow]:
        if not self._steps:
            raise IndexError(f"ChainFlow.{self.name}.steps is empty, no workflows to pop.")
        removed = self._steps.pop(index)
        self._reconcile_all()
        if not self._steps:
            # Preserve prior invariant for empty overlay
            self._bundle_all = False
        return removed
    
    def clear_memory(self):
        super().clear_memory()
        for step in self.steps:
            step.clear_memory()

    # --------------------------------- execution ---------------------------------

    def _process_inputs(self, inputs: Mapping[str, Any]) -> Any:
        if not self._steps:
            raise ExecutionError(f"{self._name}: cannot invoke an empty ChainFlow")
        curr: Any = inputs
        for i, step in enumerate(self._steps):
            try:
                curr = step.invoke(curr)
            except Exception as e:
                raise ExecutionError(
                    f"{self._name}: step {i} ('{step.name}') failed: {e}"
                ) from e
        return curr

    # --------------------------------- internals ---------------------------------

    def _apply_tail_overlay(self) -> None:
        """Force the last child to use the chain's overlay schema and bundling."""
        if not self._steps:
            return
        tail = self._steps[-1]
        tail.output_schema = list(self.output_schema)
        tail.bundle_all = bool(self.bundle_all)

    def _reconcile_pair(self, A: Workflow, B: Workflow, idx: int) -> None:
        """
        Apply A→B reconciliation rules. `idx` is the position of B (for diagnostics).
        """
        # Derive B's input schema from its arguments_map (read-only)
        b_in = list(B.input_schema)
        a_out = list(A.output_schema)
        # When B declares no inputs, raise an error, as the transfer of data must be enabled mid-way.
        if len(b_in) == 0:
            raise ValueError(f"ChainFlow.{self.name}: Non-initial step {idx} must have a non-empty input schema for data transfer")

        # 1) Subset → adopt B.in
        if set(a_out).issubset(set(b_in)):
            A.output_schema = b_in
            return

        # 2) Bundled A → adopt B.in, giving potential for matching
        if A.bundle_all:
            A.output_schema = b_in # potentially disables bundling
            return

        # 3) Positional compatibility by length
        if len(a_out) == len(b_in):
            A.output_schema = b_in
            return

        # 4) Fail fast, lengths don't match
        raise ValidationError(
            f"{self._name}: incompatible hand-off at step {idx-1}→{idx}; "
            f"A.out={a_out} vs B.in={b_in}. Enable bundling or provide an adapter. "
            f"A.bundle_all={A.bundle_all} vs B.bundle_all={B.bundle_all}"
        )

    def _reconcile_all(self) -> None:
        """Reconcile all adjacent pairs and then enforce tail overlay."""
        if not self._steps:
            return
        for i in range(1, len(self._steps)):
            self._reconcile_pair(self._steps[i - 1], self._steps[i], idx=i)
        self._apply_tail_overlay()
    
    def to_dict(self) -> OrderedDict[str, Any]:
        base = super().to_dict()
        base.update({"steps":[step.to_dict() for step in self.steps]})
        return base


class MakerChecker(Workflow):
    """
    Maker–Checker composite with an optional Judge.

    Inputs
    ------
    • Derived from the Maker: this composite's `arguments_map` proxies `maker.arguments_map`.
      Consequently, the composite's `input_schema` getter (from the base Workflow) yields
      `list(maker.arguments_map.keys())`.

    Outputs
    -------
    • Returns the Maker's final draft (a mapping). The base `Workflow.invoke()` applies the
      composite overlay (this flow's `output_schema` / `bundle_all`) at the boundary.

    Participants
    ------------
    • maker:  ToolFlow | AgentFlow | Workflow (required)
    • checker: ToolFlow | AgentFlow | Workflow (required)
    • judge:  ToolFlow | AgentFlow | Workflow (optional)
        - Must accept the same inputs as the maker (set-equality on input keys).
        - Must return a dict with {JUDGE_RESULT: bool}.
        - Is configured with `output_schema=[JUDGE_RESULT]` and `bundle_all=False`.

    Parity
    ------
    • maker.output_schema  := checker.input_schema  (adopt consumer's order)
    • checker.output_schema := maker.input_schema   (adopt consumer's order)

    Revisions
    ---------
    • max_revisions >= 0 (0 means a single Maker→Checker pass returning the first draft).
    • If no Judge, we run (max_revisions + 1) Maker→Checker passes, feeding the Checker's
      revisions back into the Maker each iteration.

    Errors
    ------
    • ValidationError: negative `max_revisions`; Judge result shape/type invalid.
    • SchemaError: Judge input schema not set-equal to Maker input schema.
    • ExecutionError: underlying participant invocation failures.

    Notes
    -----
    • No `_input_schema` is stored or mutated anywhere. All input introspection relies on
      `arguments_map` / base `input_schema` getter of participants.
    """

    def __init__(
        self,
        name: str,
        description: str,
        maker: Callable | Tool | Agent | Workflow,
        checker: Callable | Tool | Agent | Workflow,
        judge: Callable | Tool | Agent | Workflow,
        *,
        max_revisions: int = 0,
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
    ) -> None:
        # Base overlay only (no input_schema arg in base!)
        super().__init__(
            name=name,
            description=description,
            output_schema=(list(output_schema) if output_schema else [WF_RESULT]),
            bundle_all=bundle_all,
        )

        # Validate max_revisions
        try:
            self._max_revisions = int(max_revisions)
        except Exception as e:
            raise ValidationError(f"{self._name}: max_revisions must be an int; got {max_revisions!r}") from e
        if self._max_revisions < 0:
            raise ValidationError(f"{self._name}: max_revisions must be >= 0; got {self._max_revisions}")

        # Normalize participants (NO input-schema args here)
        self._maker: Workflow = _to_workflow(maker)
        self._checker: Workflow = _to_workflow(checker)
        self._judge: Optional[Workflow] = None

        # Enforce parity between maker and checker
        self._reconcile_pair()

        # Optional judge
        if judge is not None:
            j_wf = _to_workflow(judge)
            # Judge result contract
            j_wf.output_schema = [JUDGE_RESULT]
            j_wf.bundle_all = False

            if set(self._maker.input_schema) != set(j_wf.input_schema):
                raise SchemaError(
                    f"{self._name}: judge input keys {j_wf.input_schema} must match maker input keys {self._maker.input_schema}"
                )
            self._judge = j_wf

    # -------------------------------------------------------------------------
    # Read-only input contract (proxied from maker)
    # -------------------------------------------------------------------------
    @property
    def arguments_map(self) -> "OrderedDict[str, Any]":
        return self._maker.arguments_map

    # -------------------------------------------------------------------------
    # Participant properties (setters re-enforce parity and judge compatibility)
    # -------------------------------------------------------------------------
    @property
    def maker(self) -> Workflow:
        return self._maker

    @maker.setter
    def maker(self, value: Any) -> None:
        wf = _to_workflow(value)
        self._maker = wf
        self._reconcile_pair()
        # Judge compatibility after parity
        if self._judge is not None:
            maker_in = list(self._maker.input_schema)
            judge_in = list(self._judge.input_schema)
            if set(judge_in) != set(maker_in):
                self._judge = None

    @property
    def checker(self) -> Workflow:
        return self._checker

    @checker.setter
    def checker(self, value: Any) -> None:
        wf = _to_workflow(value)
        self._checker = wf
        self._reconcile_pair()
        # Judge remains unaffected; it compares against maker only.

    @property
    def judge(self) -> Optional[Workflow]:
        return self._judge

    @judge.setter
    def judge(self, value: Optional[Any]) -> None:
        if value is None:
            self._judge = None
            return
        j_wf = _to_workflow(value)
        j_wf.output_schema = [JUDGE_RESULT]
        j_wf.bundle_all = False

        maker_in = list(self._maker.input_schema)
        judge_in = list(j_wf.input_schema)
        if set(judge_in) != set(maker_in):
            raise SchemaError(
                f"{self._name}: judge input keys {judge_in} must match maker input keys {maker_in}"
            )
        self._judge = j_wf

    # -------------------------------------------------------------------------
    # Core execution
    # -------------------------------------------------------------------------
    def _process_inputs(self, inputs: Dict[str, Any]) -> Any:
        """
        Run Maker → Checker, optionally loop under Judge until approved
        or until `max_revisions` is reached.

        Returns the Maker's latest draft (raw). Base Workflow will package
        to this composite's overlay schema/bundling.
        """
        current = inputs
        last_draft: Optional[Dict[str, Any]] = None

        for _rev in range(self._max_revisions):
            # Maker
            try:
                draft = self._maker.invoke(current)
            except Exception as e:
                raise ExecutionError(f"{self._name}: maker invocation failed: {e}") from e

            # Checker
            try:
                revisions = self._checker.invoke(draft)
            except Exception as e:
                raise ExecutionError(f"{self._name}: checker invocation failed: {e}") from e

            last_draft = draft  # maker's output is the draft we ultimately return

            # Judge (optional)
            if self._judge is not None:
                try:
                    decision = self._judge.invoke(revisions)
                except Exception as e:
                    raise ExecutionError(f"{self._name}: judge invocation failed: {e}") from e

                if not isinstance(decision, dict) or JUDGE_RESULT not in decision:
                    raise ValidationError(
                        f"{self._name}: judge must return a dict with key '{JUDGE_RESULT}'"
                    )
                ok = decision[JUDGE_RESULT]
                if not isinstance(ok, bool):
                    raise ValidationError(
                        f"{self._name}: '{JUDGE_RESULT}' must be a boolean; got {type(ok).__name__}"
                    )
                if ok:
                    break  # approved
                # not approved → feed revisions back to maker
                current = revisions
            else:
                # No judge: if more cycles remain, feed revisions; otherwise finish
                current = revisions

        return last_draft
    
    def clear_memory(self):
        super().clear_memory()
        self.maker.clear_memory()
        self.checker.clear_memory()
        if self.judge is not None:
            self.judge.clear_memory()
    
    def to_dict(self) -> OrderedDict[str, Any]:
        base = super().to_dict()
        base.update({
            "maker": self.maker.to_dict(),
            "checker": self.checker.to_dict(),
            "judge": self.judge.to_dict() if self.judge else None
        })
        return base

    # -------------------------------------------------------------------------
    # Internal parity enforcement (no input-schema writes; use getters)
    # -------------------------------------------------------------------------
    def _reconcile_pair(self) -> None:
        """
        Enforce output schema parity between Maker and Checker:
          maker.output_schema  := checker.input_schema
          checker.output_schema := maker.input_schema
        (Order is adopted from the consumer in each direction.)
        """
        maker_in = list(self._maker.input_schema)
        checker_in = list(self._checker.input_schema)

        # Adopt consumer's order for each producer's outputs
        self._maker.output_schema = checker_in
        self._checker.output_schema = maker_in


class Selector(Workflow):
    """
    One-line
    --------
    Conditional router that uses a judge to choose a single branch and return that branch’s result.

    Purpose
    -------
    Separate decision logic (judge) from execution (branches). Normalizes judge output and applies a selector-level
    overlay at the boundary, leaving branch internals untouched.

    Contract
    --------
    • Judge: Workflow|Tool|Agent (wrapped); `output_schema = [JUDGE_RESULT]`, `bundle_all=False`.
      At runtime it MUST return `{JUDGE_RESULT: <branch_name:str>}`.
    • Selector `input_schema` mirrors the judge’s `input_schema` (private mirroring; read-only externally).
    • Branches: each Workflow|Tool|Agent (wrapped). Every branch’s `input_schema` must be set-equal to the judge’s.
      Branch `output_schema` is independent of the selector overlay.
    • Execution: `_process_inputs` calls judge → picks matching branch by name → returns that branch’s RAW result.
      Base `package_results` then applies the selector’s `output_schema`/`bundle_all` overlay for the caller.

    Key Parameters
    --------------
    • judge: decision workflow (normalized on set).
    • branches: list of candidate workflows; non-matching branches are dropped on judge change.
    • output_schema / bundle_all: overlay for the selector boundary only.

    Inputs & Outputs
    ----------------
    • Inputs: must match judge/branch schemas (set-equal with ordering normalized per branch).
    • Outputs: branch result packaged by the selector overlay at the boundary.

    Error Conditions
    ----------------
    • `ValidationError`: no branches; malformed judge output; missing branch for selected name.
    • `SchemaError`: invalid judge/branch schemas.
    • `ExecutionError`: judge or selected branch failure.

    Performance/Concurrency
    -----------------------
    Single-branch execution per call; deterministic selection given the judge’s result.

    Example
    -------
    >>> sel = Selector(judge=decider, branches=[foo_flow, bar_flow])
    >>> sel.output_schema = ["result"]  # overlay
    >>> sel.invoke(inputs)["result"]

    See Also
    --------
    Workflow, AgentFlow, ToolFlow
    """
    def __init__(
        self,
        name: str,
        description: str,
        branches: List[Callable|Agent|Tool|Workflow],
        judge: Callable|Agent|Tool|Workflow,
        *,
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
        name_collision_policy = "fail_fast" #"skip" #"replace"
    ) -> None:
        # Initialize base with judge input schema mirrored into selector
        super().__init__(
            name=name,
            description=description,
            output_schema=output_schema,
            bundle_all=bundle_all,
        )
        # ---- Wrap & normalize judge
        wrapped_judge: Workflow = _to_workflow(judge)
        wrapped_judge.output_schema = [JUDGE_RESULT]
        wrapped_judge.bundle_all = False
        # Mirror via private slot to match project conventions
        self._judge: Workflow = wrapped_judge
        self._name_collision_policy:str = name_collision_policy
        self._branches: OrderedDict[str, Workflow] = OrderedDict()
        # ---- Ingest branches
        wrapped_branches: List[Workflow] = [_to_workflow(b) for b in branches]
        if any(set(b.input_schema) != set(self.input_schema) for b in wrapped_branches):
            item = next((b for b in wrapped_branches if b.input_schema != self.input_schema), None)
            raise ValidationError(
                f"{self.name}: unexpected input schema: {item.name}: {item.input_schema}\nnot matching judge's input schema: {self.input_schema}"
            )
        if len(set([b.name for b in branches])) < len(branches):
            if self._name_collision_policy == "fail_fast":
                raise ValidationError(
                    f"{self.name}: duplicate branch names detected. Names given are {[b.name for b in branches]}."
                )
        for branch in wrapped_branches:
            if self._name_collision_policy == "replace" or branch.name not in self._branches:
                self._branches[branch.name] = branch
            continue

    # -------------------- Properties --------------------
    @property
    def arguments_map(self) -> OrderedDict:
        return self._judge.arguments_map
    @property
    def judge(self) -> Workflow:
        return self._judge

    @judge.setter
    def judge(self, val: Callable|Agent|Tool|Workflow) -> None:
        # Wrap provided value
        new_judge = _to_workflow(val)
        new_judge.output_schema = [JUDGE_RESULT]
        new_judge.bundle_all = False
        # Mirror input schema via private slot
        self._judge = new_judge
        self._input_schema = list(self._judge.input_schema)

        # Filter branches to only those that match by set-equivalence
        survivors: OrderedDict[str, Workflow] = OrderedDict()
        for bname, branch in self._branches.items():
            if set(branch.input_schema) == set(self._input_schema):
                survivors[bname] = branch
        self._branches = survivors

    @property
    def branches(self) -> OrderedDict[str, Workflow]:
        return OrderedDict(self._branches)

    @branches.setter
    def branches(self, vals: List[Callable|Agent|Tool|Workflow]) -> None:
        # Re-ingest + validate (same as constructor)
        wrapped: OrderedDict[str, Workflow] = OrderedDict()
        for b in vals:
            wb = _to_workflow(b)
            if set(wb.input_schema) != set(self._input_schema):
                raise ValidationError(
                    f"{self.name}: branch '{wb.name}' input_schema {wb.input_schema} "
                    f"!= selector/judge input_schema {list(self._input_schema)} (set mismatch)"
                )
            if wb.name in wrapped:
                if self._name_collision_policy == "fail_fast":
                    raise ValidationError(f"{self.name}: duplicate branch name '{wb.name}'")
                elif self._name_collision_policy == "skip":
                    continue
                else: pass
            wrapped[wb.name] = wb
        self._branches = wrapped

    # -------------------- Public API --------------------
    def add_branch(self, branch: Callable|Agent|Tool|Workflow) -> None:
        # Wrap
        wb = _to_workflow(branch)
        # Validate input schema set-equivalence
        if set(wb.input_schema) != set(self.input_schema):
            raise ValidationError(
                f"{self.name}: branch '{wb.name}' input_schema {wb.input_schema} "
                f"!= selector/judge input_schema {list(self._input_schema)} (set mismatch)"
            )
        # check for duplicates
        if wb.name in self._branches:
            if self._name_collision_policy == "fail_fast":
                raise ValidationError(f"{self.name}: duplicate branch name '{wb.name}'")
            elif self._name_collision_policy == "skip":
                return
        self._branches[wb.name] = wb

    def remove_branch(self, name: str) -> Workflow:
        if len(self._branches) == 0:
            raise KeyError(f"{self.name}: no branches to remove")
        return self._branches.pop(name)

    def clear_memory(self) -> None:
        Workflow.clear_memory(self)
        self._judge.clear_memory()
        for b in self._branches:
            self._branches[b].clear_memory()
            
    def to_dict(self) -> OrderedDict[str, Any]:
        base = super().to_dict()
        base.update({
            "judge": self.judge.to_dict(),
            "name_collision_policy":self._name_collision_policy,
            "branches": [branch.to_dict() for k,branch in self.branches.items()]
        })
        return base

    # -------------------- Execution --------------------
    def _process_inputs(self, inputs: Mapping[str, Any]) -> Any:
        """
        Decide a branch using the judge and return the *raw* result from that branch's invoke().
        Base Workflow.invoke() will package this result according to the Selector's own
        output_schema/bundle_all configuration.
        """
        # fail fast if no branches
        if not self._branches:
            raise ValidationError(f"{self.name}: There are no branches to choose from")
        # have judge select branch to invoke
        selection_raw = self._judge.invoke(inputs)
        # fail if selection isn't formatted correctly or returning a string
        if not isinstance(selection_raw, dict) or JUDGE_RESULT not in selection_raw:
            raise ValidationError(f"{self.name}: judge must return a dict with key '{JUDGE_RESULT}'; got {selection_raw!r}")
        selection = selection_raw[JUDGE_RESULT]
        if not isinstance(selection, str) or not selection.strip():
            raise ValidationError(f"{self.name}: judge must output a non-empty string under '{JUDGE_RESULT}', got {selection!r}")
        # Find and run branch with selection name (either the ._name or .name)
        selected_branch = next((b for b in self._branches.values() if b.name == selection or b.name == selection), None)
        if selected_branch is None:
            raise ValidationError(f"{self.name}: No branches matching the name for selection '{selection}'.")
        return selected_branch.invoke(inputs)


class MapFlow(Workflow):
    """
    MapFlow
    -------
    Tailored fan-out that routes per-branch payloads from `inputs[branch_name]` to
    parallel branches and aggregates results in declared branch order.

    Schema surfacing
    ----------------
    `arguments_map` is a read-only proxy to an internal Tool (`_schema_supplier`) whose
    keyword-only parameters mirror the current branch names and are annotated as
    `dict[str, any]`. The supplier is rebuilt whenever branches are added/removed.
    `input_schema` remains derived from `arguments_map` via the base Workflow.

    Inputs
    ------
    Mapping[str, Mapping] where keys are the current branch names. Unknown keys
    raise `ValidationError`. A branch without an input is:
      • flatten=False: present in output as `None` (not invoked)
      • flatten=True: skipped (not invoked)

    Outputs
    -------
    flatten=False (default):
        OrderedDict[str, dict | None] in branch order.
    flatten=True:
        Left-to-right merged OrderedDict. Plain dict results are expanded;
        special envelopes `{WF_RESULT: …}` / `{JUDGE_RESULT: …}` and non-dicts
        are stored under the branch name itself.

    Name-collision policy (branch names)
    ------------------------------------
    Duplicates honor: "fail_fast" | "skip" | "replace" (default "fail_fast"),
    consistent with ScatterFlow.

    Errors
    ------
    ValidationError: unknown input key; non-mapping branch payload; flatten key collision.
    ExecutionError: at least one branch failed (first exception is chained).
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        branches: list[Callable | Agent | Tool | Workflow] | None = None,
        *,
        flatten: bool = False,
        output_schema: list[str] = [WF_RESULT],
        bundle_all: bool = True,
        name_collision_policy: str = "fail_fast",  # "skip" | "replace"
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            output_schema=output_schema,
            bundle_all=bundle_all,
        )
        self._flatten: bool = bool(flatten)
        self._name_collision_policy: str = name_collision_policy
        self._branches: OrderedDict[str, Workflow] = OrderedDict()
        self._schema_supplier: Tool | None = None  # built from branch names

        # Wrap and insert initial branches honoring name-collision policy
        for obj in (branches or []):
            self._insert_branch(_to_workflow(obj))

        # Build initial schema supplier
        self._rebuild_schema_supplier()

    # -------------------- Introspection --------------------

    @property
    def arguments_map(self) -> OrderedDict[str, Any]:
        """Read-only proxy to the dynamic schema supplier."""
        if self._schema_supplier is None:
            return OrderedDict()
        return self._schema_supplier.arguments_map

    @property
    def branches(self) -> OrderedDict[str, Workflow]:
        """Shallow copy of branches in declared order."""
        return OrderedDict(self._branches)

    # -------------------- Branch management --------------------

    def _insert_branch(self, wf: Workflow) -> None:
        """Insert honoring name-collision policy."""
        if wf.name in self._branches:
            if self._name_collision_policy == "fail_fast":
                raise ValidationError(f"{self.name}: duplicate branch name '{wf.name}'")
            if self._name_collision_policy == "skip":
                return
            # "replace": fall through
        self._branches[wf.name] = wf

    def add_branch(self, obj: Callable | Agent | Tool | Workflow) -> None:
        """Add a branch and rebuild the schema supplier."""
        self._insert_branch(_to_workflow(obj))
        self._rebuild_schema_supplier()

    def remove_branch(self, name: str) -> Workflow:
        """Remove a branch by name and rebuild the schema supplier."""
        try:
            removed = self._branches.pop(name)
        except KeyError:
            raise KeyError(f"{self.name}: unknown branch '{name}'")
        self._rebuild_schema_supplier()
        return removed
    
    def to_dict(self):
        base = super().to_dict()
        base.update({
            "name_collision_policy":self._name_collision_policy,
            "branches": [branch.to_dict() for k,branch in self.branches.items()],
            "flatten": self._flatten,
        })
        return base

    # -------------------- Memory --------------------

    def clear_memory(self) -> None:
        Workflow.clear_memory(self)
        for wf in self._branches.values():
            wf.clear_memory()
    # -------------------- Execution --------------------

    def _process_inputs(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Execute branches concurrently with asyncio (loop-aware), preserving branch order.
        Unknown top-level keys are rejected. Per-branch payload must be a Mapping.
        """
        if not self._branches:
            raise ValidationError(f"{self.name}: no branches configured")

        # Partition: which branches to run vs skip
        to_run: list[tuple[str, Workflow, Mapping[str, Any]]] = []
        missing: list[str] = []
        for n, wf in self._branches.items():
            if n in inputs:
                payload = inputs[n]
                if not isinstance(payload, Mapping):
                    raise ValidationError(f"{self.name}: payload for branch '{n}' must be a Mapping; got {type(payload).__name__}")
                to_run.append((n, wf, payload))
            else:
                missing.append(n)

        async def _run_all():
            async def _one(n: str, wf: Workflow, payload: Mapping[str, Any]):
                return n, await asyncio.to_thread(wf.invoke, dict(payload))
            return await asyncio.gather(
                *(_one(n, wf, p) for n, wf, p in to_run),
                return_exceptions=True
            )

        def _run_coro_in_worker(coro):
            box = {}
            def _runner():
                box["res"] = asyncio.run(coro)
            t = threading.Thread(target=_runner, daemon=True)
            t.start(); t.join()
            return box["res"]

        gathered = _run_coro_in_worker(_run_all()) if to_run else []

        # Collect results in declared order; propagate first exception
        results_by_branch: OrderedDict[str, Any] = OrderedDict()
        first_exc: BaseException | None = None

        # Seed missing entries (only when not flattening)
        if not self._flatten:
            for n in missing:
                results_by_branch[n] = None

        for n, item in zip((n for n, _, _ in to_run), gathered):
            if isinstance(item, BaseException):
                if first_exc is None:
                    first_exc = item
                continue
            name, res = item
            results_by_branch[name] = res

        if first_exc is not None:
            raise ExecutionError(f"{self.name}: at least one branch failed") from first_exc

        if not self._flatten:
            # Include non-executed branches earlier; now stitch executed ones in declared order
            # Ensure final order matches self._branches
            ordered: OrderedDict[str, Any] = OrderedDict()
            for n in self._branches.keys():
                ordered[n] = results_by_branch.get(n, None)
            return ordered

        # ---- Flatten path ----
        flat: OrderedDict[str, Any] = OrderedDict()
        SPECIAL_KEYS = (WF_RESULT, JUDGE_RESULT)

        for n in self._branches.keys():
            # Only consider executed branches
            if not any(n == x for x, _, _ in to_run):
                continue
            payload = results_by_branch.get(n, None)
            if isinstance(payload, dict):
                # Unwrap special envelope → store under branch name
                if any(k in payload for k in SPECIAL_KEYS):
                    for sk in SPECIAL_KEYS:
                        if sk in payload:
                            flat[n] = payload[sk]
                            break
                else:
                    # Plain mapping: left-to-right merge, fail fast on duplicates
                    for k, v in payload.items():
                        if k in flat:
                            raise ValidationError(f"{self.name}: flatten key collision for '{k}' (originating branch '{n}')")
                        flat[k] = v
            else:
                # Non-mapping → store under branch name
                flat[n] = payload

        return flat

    # -------------------- Schema supplier --------------------

    def _rebuild_schema_supplier(self) -> None:
        """
        Build the dynamic Tool whose KW-only parameters mirror branch names.
        """
        from typing import Dict, Any
        import inspect

        names = list(self._branches.keys())

        def _identity(**kwargs) -> Dict[str, Any]:
            # Not used for execution; only to present a truthful signature.
            return kwargs

        # Build a synthetic signature: (*, <branch>: dict[str, any], ...)
        params = [
            inspect.Parameter(
                name=n,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=Dict[str, Any]
            )
            for n in names
        ]
        _identity.__signature__ = inspect.Signature(
            parameters=params,
            return_annotation=Dict[str, Any]
        )

        self._schema_supplier = Tool(
            func=_identity,
            name=f"{self.name}_inputs",
            description="Dynamic MapFlow input schema supplier (one KW-only dict per branch).",
            type="function",
            source="mapflow",
        )


class ScatterFlow(Workflow):
    """
    ScatterFlow
    -----------
    Broadcast fan-out that sends the **same** validated input mapping to all branches in parallel and aggregates.

    Contract
    --------
    • Branch set invariant: every branch MUST have an `input_schema` set-equal to the others.
      - At construction: all provided branches are checked; mismatch → ValidationError.
      - On add_branch: if non-empty, the new branch is checked against the current first branch; mismatch → ValidationError.
      - If empty, the first added branch establishes the reference schema.
    • arguments_map: read-only proxy to the **current first branch** (empty if no branches).
      input_schema is derived from arguments_map (getter-only; no private writes).
    • Name handling: branches are stored in an OrderedDict[str, Workflow] (insertion order).
      Duplicates honor `name_collision_policy`: "fail_fast" | "skip" | "replace" (default "fail_fast"),
      consistent with Selector.  # mirrors Selector’s policy knob
    • Execution: the same `inputs` mapping is dispatched to every branch concurrently via asyncio.
      Results aggregate either per-branch or flattened by merge.
    • Flatten: right-biased (latest branch wins). One-key envelopes {WF_RESULT: …}/{JUDGE_RESULT: …} are unwrapped
      before merging; non-mapping results are inserted under the branch’s name.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        branches: list[Callable | Agent | Tool | Workflow] | None = None,
        *,
        flatten: bool = False,
        output_schema: list[str] = [WF_RESULT],
        bundle_all: bool = True,
        name_collision_policy: str = "fail_fast",  # "skip" | "replace"
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            output_schema=output_schema,
            bundle_all=bundle_all,
        )
        self._flatten: bool = bool(flatten)
        self._name_collision_policy: str = name_collision_policy
        self._branches: OrderedDict[str, Workflow] = OrderedDict()

        # 1) Wrap all incoming branches first
        wrapped: list[Workflow] = [_to_workflow(obj) for obj in (branches or [])]

        # 2) If non-empty, enforce set-equal input_schema across all wrapped branches
        if wrapped:
            ref_schema = set(wrapped[0].input_schema)
            bad = next((b for b in wrapped if set(b.input_schema) != ref_schema), None)
            if bad is not None:
                raise ValidationError(
                    f"{self.name}: branch '{bad.name}' input_schema {bad.input_schema} "
                    f"!= reference schema {list(ref_schema)}"
                )

        # 3) Only after schema validation, insert honoring name_collision_policy
        for wf in wrapped:
            self._insert_branch_after_validation(wf)

    # -------------------- Properties --------------------

    @property
    def arguments_map(self) -> OrderedDict[str, Any]:
        """Read-only proxy to the first branch's arguments_map; empty if no branches."""
        first = next(iter(self._branches.values()), None)
        return first.arguments_map if first is not None else OrderedDict()

    @property
    def branches(self) -> OrderedDict[str, Workflow]:
        """Return a shallow copy of branches in declared order."""
        return OrderedDict(self._branches)

    # -------------------- Branch Management --------------------

    def _insert_branch_after_validation(self, wf: Workflow) -> None:
        """
        Insert a branch assuming its schema is already validated (or set the reference if empty)
        and honoring the name-collision policy.
        """
        if wf.name in self._branches:
            if self._name_collision_policy == "fail_fast":
                raise ValidationError(f"{self.name}: duplicate branch name '{wf.name}'")
            if self._name_collision_policy == "skip":
                return
            # "replace": fall through to overwrite
        self._branches[wf.name] = wf

    def add_branch(self, obj: Callable | Agent | Tool | Workflow) -> None:
        """
        Add a single branch. If non-empty, enforce set-equal input_schema vs current first branch before inserting.
        """
        wf = _to_workflow(obj)
        first = next(iter(self._branches.values()), None)
        if len(self._branches)>0:
            if set(wf.input_schema) != set(self.input_schema):
                raise ValidationError(
                    f"{self.name}: new branch '{wf.name}' input_schema {wf.input_schema} "
                    f"!= reference schema {list(first.input_schema)}"
                )
        # After schema validity, honor name-collision policy
        self._insert_branch_after_validation(wf)

    def remove_branch(self, name: str) -> Workflow:
        """Remove a branch by exact name and return it."""
        try:
            return self._branches.pop(name)
        except KeyError:
            raise KeyError(f"{self.name}: unknown branch '{name}'")

    def clear_memory(self) -> None:
        """Clear own checkpoints and cascade to children."""
        Workflow.clear_memory(self)
        for wf in self._branches.values():
            wf.clear_memory()

    # -------------------- Execution --------------------

    def _process_inputs(self, inputs: Mapping[str, Any]) -> Any:
        """
        Run all branches in parallel, map results by branch name, and optionally flatten.

        Returns
        -------
        Mapping[str, Any] when flatten=False:
            {branch_name: branch_result, ...}
        Mapping[str, Any] when flatten=True:
            Right-biased merged mapping. For branch results that are mappings with special
            keys (WF_RESULT/JUDGE_RESULT), unwrap and store under the branch name itself.
            For plain mappings (no special keys), merge their pairs into the flat dict.
            For non-mappings, store under the branch name.
        """
        if not self._branches:
            raise ValidationError(f"{self.name}: no branches configured")

        async def _run_all():
            async def _one(n: str, wf: Workflow):
                # run sync wf.invoke in default thread pool
                return n, await asyncio.to_thread(wf.invoke, inputs)
            return await asyncio.gather(
                *(_one(n, wf) for n, wf in self._branches.items()),
                return_exceptions=True
            )

        def _run_coro_in_worker(coro):
            """Always run in a short-lived thread with its own event loop (simple & robust)."""
            box = {}
            def _runner():
                box["res"] = asyncio.run(coro)
            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join()
            return box["res"]

        gathered = _run_coro_in_worker(_run_all())

        # Map: branch name -> result (preserve branch order)
        results_by_branch: OrderedDict[str, Any] = OrderedDict()
        first_exc: BaseException | None = None
        for n, item in zip(self._branches.keys(), gathered):
            if isinstance(item, BaseException):
                if first_exc is None:
                    first_exc = item
                continue
            # item is (name, result)
            name, res = item
            # Safety: ensure zipped names match returned names
            results_by_branch[name] = res

        if first_exc is not None:
            raise ExecutionError(f"{self.name}: at least one branch failed") from first_exc

        if not self._flatten:
            return results_by_branch

        # ---- Flatten path ----
        flat: OrderedDict[str, Any] = OrderedDict()
        SPECIAL_KEYS = (WF_RESULT, JUDGE_RESULT)

        for name in self._branches.keys():
            if name not in results_by_branch:
                continue
            payload = results_by_branch[name]

            if isinstance(payload, dict):
                # If it has a special key, unwrap and store under the branch name itself.
                # Prioritize WF_RESULT over JUDGE_RESULT if both exist (rare).
                if any(k in payload for k in SPECIAL_KEYS):
                    for sk in SPECIAL_KEYS:
                        if sk in payload:
                            flat[name] = payload[sk]
                            break
                else:
                    # Plain mapping: merge pairs directly (right-biased by branch order).
                    for k, v in payload.items():
                        flat[k] = v
            else:
                # Non-mapping: store under the branch name.
                flat[name] = payload

        return flat
    
    def to_dict(self):
        base = super().to_dict()
        base.update({
            "name_collision_policy":self._name_collision_policy,
            "branches": [branch.to_dict() for k,branch in self.branches.items()],
            "flatten": self._flatten,
        })
        return base
