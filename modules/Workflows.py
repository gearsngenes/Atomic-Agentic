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
from typing import Any, Dict, List, Optional
from collections.abc import Mapping, Iterable, Sequence, Set
from collections import OrderedDict
import time
# =========================
# Atomic-Agentic Modules
# =========================
from .Agents import Agent
from .Tools import Tool

# =========================
# Constants & Exceptions
# =========================
WF_RESULT = "__wf_result__"


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
    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-friendly spec: no kind/component refs.
        """
        return {
            "name": self._name,
            "description": self._description,
            "arguments_map": dict(self.arguments_map),
            "output_schema": list(self._output_schema),
            "bundle_all": bool(self._bundle_all),
        }

    # ---------- Execution ----------
    def invoke(self, inputs: Mapping[str, Any]) -> Dict[str, Any]:
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
    def package_results(self, results: Any) -> Dict[str, Any]:
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
            res_keys = set(results.keys())
            sch_keys = set(schema)
            if res_keys == sch_keys:
                # reorder to schema order
                return {k: results[k] for k in schema}
            if res_keys.issubset(sch_keys):
                # pad missing with None
                return {k: results.get(k, None) for k in schema}
            # if lengths match but keys don't 100% match
            shared = res_keys.intersection(sch_keys)
            res_only = list(res_keys - shared)
            missing = list(sch_keys - shared)
            if len(res_only) == len(missing) and len(missing)>0:
                pos_mapped: OrderedDict = OrderedDict()
                NOT_FILLED = object()
                for k in schema: pos_mapped[k] = NOT_FILLED
                for k in shared: pos_mapped[k] = results[k]
                for i, k in enumerate(missing): pos_mapped[k] = results[res_only[i]]
                return pos_mapped
            # mismatch: try bundling if allowed
            if self._bundle_all:
                if len(schema) != 1:
                    raise PackagingError(f"{self._name}: bundle_all requires single-key schema")
                return {schema[0]: dict(results)}
            raise PackagingError(
                f"{self._name}: mapping keys {sorted(res_keys - sch_keys)} not in schema {sorted(sch_keys)}; "
                f"provide adapter or enable bundle_all"
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
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
    ) -> None:
        super().__init__(
            name=name or f"workflow_{agent.name}",
            description=agent.description,
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
        tool: Tool,
        name: Optional[str] = None,
        *,
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
    ) -> None:
        super().__init__(
            name=name or f"workflow_{tool.name}",
            description=tool.description,
            output_schema=output_schema,
            bundle_all=bundle_all,
        )
        self._tool = tool

    @property
    def tool(self) -> Tool:
        return self._tool
    @tool.setter
    def tool(self, val: Tool) -> None:
        self._tool = val

    @property
    def arguments_map(self) -> Mapping[str, Any]:
        return self._tool.arguments_map

    def _process_inputs(self, inputs: Dict[str, Any]) -> Any:
        # Dict-only invoke; Tool handles its own validation & binding.
        return self._tool.invoke(inputs)
    

# wraps all incoming objects into workflow classes and adjusts their input/output schemas based on optional parameters
def _to_workflow(obj: Agent | Tool | Workflow) -> Workflow:
        if isinstance(obj, Workflow): return obj
        if isinstance(obj, Agent): return AgentFlow(obj)
        if isinstance(obj, Tool): return ToolFlow(obj)
        raise ValidationError(f"Object must be Agent, Tool, or Workflow. Got unexpected '{type(obj).__name__}'.")


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
        steps: Optional[List[Any]] = None,  # Tool | Agent | Workflow accepted; normalized via _to_workflow
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
    def steps(self, new_steps: Optional[List[Any]]) -> None:
        self._steps = [ _to_workflow(s) for s in (new_steps or []) ]
        self._reconcile_all()

    def add_step(self, step: Any, *, position: int | None = None) -> None:
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


class MakerChecker(Workflow):
    """
    One-line
    --------
    Iterative maker→checker composite that loops drafts through review (and optional judge) until approved or capped.

    Purpose
    -------
    Encapsulate a write–review loop with strict schema parity between participants. Exposes composite-level overlays
    for external callers while preserving internal participant schemas.

    Contract
    --------
    • Parity: `maker.output_schema == checker.input_schema` and `checker.output_schema == maker.input_schema`.
    • Judge (optional): `input_schema == maker.input_schema`, `output_schema == [JUDGE_RESULT]`, `bundle_all=False`,
      returns `{JUDGE_RESULT: bool}`.
    • On init or when replacing maker/checker, the composite mirrors MAKER externally:
      `self._input_schema = maker.input_schema`; composite `output_schema` defaults to `maker.output_schema`.
    • Callers MAY set composite `output_schema`/`bundle_all` as an overlay; internal participant schemas are unchanged.
    • Loop:
      1) `draft = maker.invoke(inputs)`
      2) `revisions = checker.invoke(draft)`
      3) If judge: `ok = judge.invoke(revisions)[JUDGE_RESULT]` (bool). If ok → stop; else `draft = maker.invoke(revisions)` and repeat.
      4) Stop at `max_revisions` if not approved.
    • Composite returns the latest MAKER draft; base `package_results` applies the composite overlay at the boundary.

    Key Parameters
    --------------
    • maker, checker: Workflow|Tool|Agent (wrapped); must satisfy parity; set via properties (reconciles pair).
    • judge: optional Workflow|Tool|Agent (wrapped and normalized as above).
    • max_revisions: positive int safety cap.

    Inputs & Outputs
    ----------------
    • Inputs: must match `maker.input_schema` (composite mirrors maker).
    • Outputs: packaged draft per composite overlay (does not mutate maker/checker schemas).

    Error Conditions
    ----------------
    • `SchemaError`/`ValidationError`: parity violations; incompatible judge; bad overlays.
    • `ValidationError`: judge must return a single boolean under `JUDGE_RESULT`.
    • `ExecutionError`: participant failure.

    Performance/Concurrency
    -----------------------
    Deterministic loop; one participant call per step per iteration.

    Example
    -------
    >>> mc = MakerChecker(maker=mk, checker=ck, max_revisions=3)
    >>> mc.invoke(seed_inputs)  # returns packaged maker draft

    See Also
    --------
    Workflow, ToolFlow, AgentFlow
    """
    # --------------------------
    # Init
    # --------------------------
    def __init__(
        self,
        name: str,
        description: str,
        maker: Agent | Tool | Workflow,
        checker: Agent | Tool | Workflow,
        judge: Optional[Agent | Tool | Workflow] = None,
        max_revisions: int = 0,
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True
    ) -> None:
        # Normalize participants
        maker_wf = _to_workflow(maker)
        checker_wf = _to_workflow(checker)

        # Validate max_revisions early using provided name for clear messaging
        if max_revisions < 0:
            raise ValidationError(f"{name or '<MakerChecker>'}: max_revisions must be >= 0; got {max_revisions}")

        # Initialize base mirroring the *maker* input schema
        super().__init__(
            name=name,
            description=description,
            input_schema=list(maker_wf.input_schema),
            output_schema=list(output_schema),
            bundle_all = bundle_all,
        )

        # Instance state
        self._maker: Workflow = maker_wf
        self._checker: Workflow = checker_wf
        self.max_revisions: int = int(max_revisions)
        self._judge: Optional[Workflow] = None

        logger.debug("%s: MakerChecker init maker=%s checker=%s max_revisions=%d", self.name, maker_wf.name, checker_wf.name, max_revisions)

        # Enforce maker <-> checker parity before initial mirror.
        self._reconcile_pair()

        # Judge initial sanity (if provided)
        if judge is not None:
            # build judge
            j_wf = _to_workflow(judge)
            j_wf.output_schema = [JUDGE_RESULT]
            j_wf.bundle_all = False
            # if maker and judge inputs don't match, this judge is not compatible
            if set(j_wf.input_schema) != set(j_wf.input_schema):
                raise SchemaError(
                    f"{self.name}: judge.input_schema {j_wf.input_schema} != maker.input_schema {self._maker.input_schema}"
                )
            self._judge = j_wf

        # Mirror maker endpoints externally (again, after reconcile)
        self._refresh_endpoints()


    # --------------------------
    # Utilities
    # --------------------------
    def _reconcile_pair(self) -> None:
        """
        Enforce maker-checker symmetry
        """
        self._maker.output_schema = self._checker.input_schema
        self._checker.output_schema = self._maker.input_schema
        # If judge's input schema length no longer matches maker, drop judge and adapter
        if self._judge is not None and set(self._judge.input_schema) != set(self._maker.input_schema):
            logger.info("%s: dropping judge due to input_schema mismatch (judge=%s maker=%s)", self.name, self._judge.input_schema, self._maker.input_schema)
            self._judge = None

    def _refresh_endpoints(self) -> None:
        """
        Mirror maker endpoints; used on initialization and when replacing maker/checker.
        Do NOT call this at invoke-time so that explicit composite-level overlays set
        by callers are not clobbered between invocations.
        """
        # must use private for input_schema
        self._input_schema = list(self._maker.input_schema)
        # set judge to None if judge.input_schema length no longer matches
        if self._judge is not None and len(self._maker.input_schema) != len(self._judge.input_schema):
            logger.info("%s: judge input_schema no longer matches maker; dropping judge", self.name)
            self._judge = None
        # update adapter if judge present
        logger.debug("%s: refreshed endpoints -> input_schema=%s", self.name, self._input_schema)

    # --------------------------
    # Participant accessors
    # --------------------------
    @property
    def maker(self) -> Workflow:
        return self._maker

    @maker.setter
    def maker(self, value: Agent | Tool | Workflow) -> None:
        self._maker = _to_workflow(value)
        # Maintain parity before commit
        self._reconcile_pair()   # M.out -> C.in
        # Mirror endpoints to maker for this composite (input/output/bundle)
        self._refresh_endpoints()

    @property
    def checker(self) -> Workflow:
        return self._checker

    @checker.setter
    def checker(self, value: Agent | Tool | Workflow) -> None:
        self._checker = _to_workflow(value)
        # Maintain parity before commit
        self._reconcile_pair()     # C.out  -> M.in
        # Mirror endpoints to maker for this composite (input/output/bundle)
        self._refresh_endpoints()

    @property
    def judge(self) -> Optional[Workflow]:
        return self._judge

    @judge.setter
    def judge(self, value: Optional[Agent | Tool | Workflow]) -> None:
        j_wf = _to_workflow(value) if value is not None else None
        if j_wf is None: self._judge = None; return
        # Judge must consume maker.input; produce a boolean at JUDGE_RESULT
        if set(j_wf.input_schema) != set(self._maker.input_schema):
            raise SchemaError(
                f"{self.name}: judge.input_schema {j_wf.input_schema} != maker.input_schema length {self._maker.input_schema}"
            )
        self._judge = j_wf
        self._judge.output_schema = [JUDGE_RESULT]
        self._judge.bundle_all = False

    # --------------------------
    # Runtime
    # --------------------------
    def _process_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministic loop:
          1) maker(draft_0) -> draft_1
          2) checker(draft_1) -> revisions_1
          3) if judge: ok = judge(revisions_1)[__judge_result__] (bool)
             - if ok: break and return draft_1
             - else: draft_2 = maker(revisions_1), repeat
          4) stop after max_revisions or when judge approves.

        Returns the latest maker draft (dict keyed by maker.output_schema).
        """
        # Initial draft from maker (keys == maker.output_schema)
        draft = self._maker.invoke(inputs)

        if self.max_revisions == 0:
            return draft

        for _ in range(self.max_revisions):
            # Checker consumes maker.input_schema and emits a revision bundle (== maker.input_schema)
            revisions = self._checker.invoke(draft)

            if self._judge is not None:
                decision = self._judge.invoke(revisions)  # expects maker.input_schema
                ok = decision.get(JUDGE_RESULT)
                if not isinstance(ok, bool):
                    logger.error("%s: judge returned invalid decision %r", self.name, decision)
                    raise ValidationError(
                        f"{self.name}: judge must return a single boolean at key '{JUDGE_RESULT}'; got {decision!r}"
                    )
                if ok: break

            # Feedback: maker uses revisions to produce a new draft
            draft = self._maker.invoke(revisions)

        return draft


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
        branches: List[Agent|Tool|Workflow],
        judge: Agent|Tool|Workflow,
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
    ) -> None:
        # ---- Wrap & normalize judge
        wrapped_judge: Workflow = _to_workflow(judge, out_sch = [JUDGE_RESULT])
        wrapped_judge.bundle_all = False

        # Initialize base with judge input schema mirrored into selector
        super().__init__(
            name=name,
            description=description,
            input_schema=list(wrapped_judge.input_schema),
            output_schema=output_schema,
            bundle_all=bundle_all,
        )

        # Mirror via private slot to match project conventions
        self._judge: Workflow = wrapped_judge
        self._input_schema = list(self._judge.input_schema)

        # ---- Ingest branches
        wrapped_branches: List[Workflow] = [_to_workflow(b, list(self._input_schema), output_schema) for b in branches]
        if len(set([b.name for b in branches])) < len(branches):
            raise ValidationError(
                f"{self.name}: duplicate branch names detected. Names given are {[b.name for b in branches]}."
            )
        if any(set(b.input_schema) != set(self._input_schema) for b in wrapped_branches):
            raise ValidationError(
                f"{self.name}: unexpected input schema not matching judge's input schema: {self._input_schema}"
            )
        self._branches: List[Workflow] = wrapped_branches

    # -------------------- Properties --------------------
    @property
    def judge(self) -> Workflow:
        return self._judge

    @judge.setter
    def judge(self, val: Agent|Tool|Workflow) -> None:
        # Wrap provided value
        new_judge = _to_workflow(val)
        new_judge.output_schema = [JUDGE_RESULT]
        new_judge.bundle_all = False
        # Mirror input schema via private slot
        self._judge = new_judge
        self._input_schema = list(self._judge.input_schema)

        # Filter branches to only those that match by set-equivalence
        survivors: List[Workflow] = []
        for b in self._branches:
            try:
                if set(b.input_schema) == set(self._input_schema):
                    survivors.append(b)
            except Exception:
                # Extremely defensive: drop anything malformed
                logging.warning(f"{self.name}: branch {b.name} did not match the new judge's input schema. it is being dropped.")
                continue
        self._branches = survivors

    @property
    def branches(self) -> List[Workflow]:
        return list(self._branches)

    @branches.setter
    def branches(self, vals: List[Agent|Tool|Workflow]) -> None:
        # Re-ingest + validate (same as constructor)
        seen_names: set[str] = set()
        wrapped: List[Workflow] = []
        for b in vals:
            wb = _to_workflow(b, in_sch=self._input_schema, out_sch=self._output_schema)
            if set(wb.input_schema) != set(self._input_schema):
                raise ValidationError(
                    f"{self.name}: branch '{wb.name}' input_schema {wb.input_schema} "
                    f"!= selector/judge input_schema {list(self._input_schema)} (set mismatch)"
                )
            if wb.name in seen_names:
                raise ValidationError(f"{self.name}: duplicate branch name '{wb.name}'")
            seen_names.add(wb.name)
            wrapped.append(wb)
        self._branches = wrapped

    # -------------------- Public API --------------------
    def add_branch(self, branch: Agent|Tool|Workflow) -> None:
        # Wrap
        wb = _to_workflow(branch, self.input_schema, self.output_schema)
        # Validate input schema set-equivalence
        if set(wb.input_schema) != set(self.input_schema):
            raise ValidationError(
                f"{self.name}: branch '{wb.name}' input_schema {wb.input_schema} "
                f"!= selector/judge input_schema {list(self._input_schema)} (set mismatch)"
            )
        # Unique name
        if any(b.name == wb.name for b in self._branches):
            raise ValidationError(f"{self.name}: duplicate branch name '{wb.name}'")

        self._branches.append(wb)

    def remove_branch(self, name: str|None = None, *, position: int = -1) -> Workflow:
        if not self._branches:
            raise IndexError(f"{self.name}: no branches to remove")
        if name is None:
            return self._branches.pop(position)
        idx = next((i for i,b in enumerate(self._branches) if b.name == name), None)
        if idx is None:
            raise ValueError(f"{self.name}: no branch named '{name}' to remove")
        return self._branches.pop(idx)

    def clear_memory(self) -> None:
        Workflow.clear_memory(self)
        self._judge.clear_memory()
        for b in self._branches:
            b.clear_memory()

    # -------------------- Execution --------------------
    def _process_inputs(self, inputs: Dict[str, Any]) -> Any:
        """
        Decide a branch using the judge and return the *raw* result from that branch's invoke().
        Base Workflow.invoke() will package this result according to the Selector's own
        output_schema/bundle_all configuration.
        """
        # fail fast if no branches
        if not self._branches:
            logger.error("%s: no branches available for selection", self.name)
            raise ValidationError(f"{self.name}: There are no branches to choose from")
        # have judge select branch to invoke
        selection_raw = self._judge.invoke(inputs)
        # fail if selection isn't formatted correctly or returning a string
        if not isinstance(selection_raw, dict) or JUDGE_RESULT not in selection_raw:
            logger.error("%s: judge returned malformed selection %r", self.name, selection_raw)
            raise ValidationError(f"{self.name}: judge must return a dict with key '{JUDGE_RESULT}'; got {selection_raw!r}")
        selection = selection_raw[JUDGE_RESULT]
        if not isinstance(selection, str) or not selection.strip():
            logger.error("%s: judge produced empty or non-string selection %r", self.name, selection)
            raise ValidationError(f"{self.name}: judge must output a non-empty string under '{JUDGE_RESULT}', got {selection!r}")
        # Find and run branch with selection name (either the ._name or .name)
        selected_branch = next((b for b in self._branches if b._name == selection or b.name == selection), None)
        if selected_branch is None:
            logger.error("%s: no branch matched selection '%s'", self.name, selection)
            raise ValidationError(f"{self.name}: No branches matching the name for selection '{selection}'.")
        logger.debug("%s: selector chose branch '%s'", self.name, selected_branch.name)
        return selected_branch.invoke(inputs)


class MapFlow(Workflow):
    """
    One-line
    --------
    Tailored fan-out that routes per-branch payloads from `inputs[branch.name]` to parallel branches and aggregates.

    Purpose
    -------
    Executes a heterogeneous set of child workflows concurrently where each branch gets its own dict payload keyed by
    the branch’s name. `input_schema` auto-mirrors the ordered list of branch names; packaging/bundling remain a
    boundary concern handled by the base Workflow.

    Contract
    --------
    • `input_schema` is dynamic and equals `[b.name for b in branches]` (ordered); it is rebuilt on add/remove.
    • Only branches with a present payload are invoked; the payload MUST be a dict or a `ValidationError` is raised.
    • Each executed branch MUST return a dict or a `ValidationError` is raised.
    • When `flatten=False` (default): return `{branch.name: dict | None, ...}`; missing payload ⇒ `None`.
    • When `flatten=True`: merge executed branch dicts; collisions raise `ValidationError`.
      – Single-key envelopes `{WF_RESULT: …}` / `{JUDGE_RESULT: …}` are unwrapped first; if the inner value is a dict
        it is merged; otherwise it is placed under `flat[branch.name]`.
    • Child schemas are not mutated; selector/chain-level overlays apply only at the MapFlow boundary.

    Key Parameters
    --------------
    • branches: list[Workflow|Agent|Tool]; wrapped via `_to_workflow(..., in_sch=None, out_sch=None)`. Names must be unique.
    • flatten: bool (default False) to merge or keep per-branch results.
    • output_schema: list[str] (default `[WF_RESULT]`) — overlay for the MapFlow boundary.
    • bundle_all: bool (default True) — obeys base rules.
    • max_workers: int (cap for ThreadPool).

    Inputs & Outputs
    ----------------
    • Inputs: keys must be a subset of current branch names. Unknown keys → `ValidationError`. Missing keys allowed.
    • Outputs: either per-branch dict or a single flattened dict, then packaged by base with the MapFlow overlay.

    Error Conditions
    ----------------
    • `ValidationError`: duplicate branch name; non-dict payload; non-dict branch result; key collision on flatten; unknown input key.
    • `ExecutionError`: any branch raises; original exception is chained.
    • `IndexError`/`ValueError`: remove with no branches / remove unknown name.

    Performance/Concurrency
    -----------------------
    Parallel via `ThreadPoolExecutor` (up to `max_workers`). Result aggregation keeps declared branch order.

    Example
    -------
    >>> mf = MapFlow(name="mf", description="per-branch", branches=[foo, bar], flatten=True)
    >>> mf.invoke({"foo": {"x": 1}, "bar": {"y": 2}})  # {"x": 1, "y": 2}

    See Also
    --------
    Workflow, AgentFlow, ToolFlow, ScatterFlow
    """


    def __init__(
        self,
        name: str,
        description: str,
        branches: List[Union[Workflow, Agent, Tool]] = [],
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
        flatten: bool = False,
        *,
        max_workers: int = 16
    ) -> None:
        # Initialize base with placeholder input_schema; we will set the private field afterwards
        super().__init__(
            name=name,
            description=description,
            input_schema=[],  # will be rebuilt from branch names
            output_schema=output_schema,
            bundle_all=bundle_all,
        )

        self._flatten: bool = bool(flatten)
        self._branches: List[Workflow] = []
        self._max_workers = max_workers

        # Ingest/wrap branches; enforce unique names
        seen: set[str] = set()
        for obj in branches or []:
            wf = _to_workflow(obj, in_sch=None, out_sch=None)
            if wf.name in seen:
                raise ValueError(f"{self.name}: duplicate branch name '{wf.name}'")
            seen.add(wf.name)
            self._branches.append(wf)

        # Input schema mirrors current branch names (ordered)
        self._rebuild_input_schema_from_branch_names()

    # -------------------- Properties --------------------

    @property
    def branches(self) -> List[Workflow]:
        """Read-only view of branches (shallow copy)."""
        return list(self._branches)

    # -------------------- Branch Management --------------------

    def add_branch(self, obj: Union[Workflow, Agent, Tool]) -> None:
        """
        Add a branch. Name must be unique. Input schema is rebuilt from branch names.
        """
        wf = _to_workflow(obj, in_sch=None, out_sch=None)
        if any(b.name == wf.name for b in self._branches):
            raise ValidationError(f"{self.name}: duplicate branch name '{wf.name}'")
        self._branches.append(wf)
        self._rebuild_input_schema_from_branch_names()

    def remove_branch(self, name: Optional[str] = None, *, position: int = -1) -> Workflow:
        """
        Remove a branch by exact name or by position (default: last). Returns the removed workflow.
        Input schema is rebuilt from branch names.
        """
        if not self._branches:
            raise IndexError(f"{self.name}: no branches to remove")

        if name is not None:
            for i, b in enumerate(self._branches):
                if b.name == name:
                    removed = self._branches.pop(i)
                    self._rebuild_input_schema_from_branch_names()
                    return removed
            raise ValueError(f"{self.name}: no branch named '{name}' to remove")

        removed = self._branches.pop(position)
        self._rebuild_input_schema_from_branch_names()
        return removed

    # -------------------- Memory --------------------

    def clear_memory(self) -> None:
        Workflow.clear_memory(self)
        for b in self._branches:
            try:
                b.clear_memory()
            except Exception:
                logger.warning("%s: branch.clear_memory raised during clear_memory", self.name, exc_info=True)

    # -------------------- Execution --------------------

    def _process_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tailored fan-out:
        
        - For each branch:
            * If inputs contains branch.name:
                - require inputs[branch.name] to be a dict; otherwise raise ValueError
                - schedule branch.invoke(inputs[branch.name])
            * If inputs does not contain branch.name:
                - flatten=False: result for branch.name := None (not invoked)
                - flatten=True: branch is ignored (not invoked)

        Returns:
            - flatten=False: { branch.name: dict | None, ... } (all branches)
            - flatten=True:  flattened dict merged from executed branches with collision checks;
                             special single-key envelopes {WF_RESULT: ...}/{JUDGE_RESULT: ...}
                             are inserted as {branch.name: <raw>} instead of flattened.
        """
        if not self._branches:
            logger.error("%s: no branches available", self.name)
            raise ValidationError(f"{self.name}: No branches available")

        # Partition branches: which have payloads, which don't
        to_run: List[tuple[str, Workflow, Dict[str, Any]]] = []
        missing_names: List[str] = []

        for b in self._branches:
            if b.name in inputs:
                payload = inputs[b.name]
                if not isinstance(payload, dict):
                    raise ValidationError(
                        f"{self.name}: payload for branch '{b.name}' must be a dict; "
                        f"got {type(payload).__name__}"
                    )
                to_run.append((b.name, b, payload))
            else:
                missing_names.append(b.name)

        results_by_branch: Dict[str, Optional[Dict[str, Any]]] = {}

        # Initialize defaults for missing branches (flatten=False only)
        if not self._flatten:
            for name in missing_names:
                results_by_branch[name] = None

        first_exc: Optional[BaseException] = None
        failing_branch: Optional[str] = None

        if to_run:
            max_workers = min(len(to_run), self._max_workers)
            futures: List[tuple[str, Any]] = []
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for name, wf, payload in to_run:
                    futures.append((name, pool.submit(wf.invoke, payload)))

                # Collect in declared branch order for determinism
                for branch_name, fut in futures:
                    if first_exc is not None:
                        if not fut.done():
                            fut.cancel()
                        continue
                    try:
                        res = fut.result()
                        if not isinstance(res, dict):
                            raise ValidationError(
                                f"{self.name}:{branch_name} returned non-dict result "
                                f"({type(res).__name__}); expected dict"
                            )
                        results_by_branch[branch_name] = res
                    except BaseException as e:
                        first_exc = e
                        failing_branch = branch_name
                        # Attempt to cancel remaining
                        for _, f in futures:
                            if not f.done():
                                f.cancel()

        if first_exc is not None:
            logger.exception("%s: branch %s failed during MapFlow execution", self.name, failing_branch)
            raise ExecutionError(f"{self.name}:{failing_branch} failed") from first_exc

        if not self._flatten:
            # Ensure all branches are present in the output (ordered by declaration)
            # (Executed branches already filled; missing branches are None)
            return {b.name: results_by_branch.get(b.name, None) for b in self._branches}

        # Flattening path: merge only executed branch dicts
        flat: Dict[str, Any] = {}
        special_keys = {WF_RESULT, JUDGE_RESULT}

        for b in self._branches:
            if b.name not in results_by_branch:
                # either missing input or (in theory) not executed; skip entirely
                continue
            payload = results_by_branch[b.name]
            if payload is None:
                # shouldn't occur here because we don't insert None for flatten=True
                continue

            # unwrap any special keys if using special keys
            if _is_namedtuple(payload):
                payload = _namedtuple_as_mapping(payload)
            elif is_dataclass(payload):
                payload = asdict(payload)
            while isinstance(payload, MAPPABLE_KINDS) and len(payload.keys()) == 1 and list(payload.keys())[0] in special_keys:
                payload = payload[list(payload.keys())[0]]
            if not isinstance(payload, dict):
                flat[b.name] = payload
                continue

            # Normal flatten: merge keys with collision check
            for k, v in payload.items():
                if k in flat:
                    raise ValidationError(
                        f"{self.name}: flatten key collision for '{k}' "
                        f"(originating branch '{b.name}')"
                    )
                flat[k] = v
        return flat

    # -------------------- Helpers --------------------

    def _rebuild_input_schema_from_branch_names(self) -> None:
        """Mirror current branch names into the private input schema (ordered)."""
        self._input_schema = [b.name for b in self._branches]


class ScatterFlow(Workflow):
    """
    One-line
    --------
    Broadcast fan-out that sends the same validated input dict to all branches in parallel and aggregates.

    Purpose
    -------
    Executes multiple child workflows concurrently against a fixed, shared input schema. Enforces set-equality of
    branch input schemas (broadcast contract). Aggregation is either per-branch or flattened; packaging/bundling are
    applied at the ScatterFlow boundary by the base Workflow.

    Contract
    --------
    • `input_schema` is fixed at construction and immutable thereafter.
    • Every branch MUST accept an input schema set-equal to the broadcast schema.
      – At construction, non-conforming branches are pruned.
      – On `add_branch`, non-conforming branches raise `ValueError`.
    • Each branch is invoked exactly once per call with the same `inputs` dict and MUST return a dict; otherwise `ValidationError`.
    • When `flatten=False` (default): return `{branch.name: dict_result, ...}`.
    • When `flatten=True`: merge branch results; collisions raise `ValidationError`.
      – Single-key envelopes `{WF_RESULT: …}` / `{JUDGE_RESULT: …}` are unwrapped first; if the inner value is a dict it
        is merged, otherwise it is stored under `flat[branch.name]`.
    • Child schemas are not mutated; selector/chain-level overlays apply only at the ScatterFlow boundary.

    Key Parameters
    --------------
    • input_schema: list[str] (fixed broadcast contract).
    • branches: list[Workflow|Agent|Tool]; wrapped via `_to_workflow(..., in_sch=input_schema, out_sch=None)`. Names must be unique.
    • flatten: bool (default False) to merge or keep per-branch results.
    • output_schema: list[str] (default `[WF_RESULT]`) — overlay for the ScatterFlow boundary.
    • bundle_all: bool (default True) — obeys base rules.
    • max_workers: int (cap for ThreadPool).

    Inputs & Outputs
    ----------------
    • Inputs: keys must be a subset of `input_schema`; unknown keys → `ValidationError`. Missing keys allowed (branch defaults may apply).
    • Outputs: per-branch map or flattened dict, then packaged by base with the ScatterFlow overlay.

    Error Conditions
    ----------------
    • `ValidationError`: no branches; non-dict branch result; key collision on flatten; unknown input key.
    • `ValueError`: duplicate branch name; add of non-conforming branch.
    • `ExecutionError`: any branch raises; original exception is chained.
    • `IndexError`/`ValueError`: remove with no branches / remove unknown name.

    Performance/Concurrency
    -----------------------
    Parallel via `ThreadPoolExecutor` (up to `max_workers`). Aggregation preserves declared branch order for determinism.

    Example
    -------
    >>> sf = ScatterFlow(name="sf", description="broadcast", input_schema=["text"], branches=[a, b], flatten=True)
    >>> sf.invoke({"text": "hello"})  # merged keys from a & b

    See Also
    --------
    Workflow, AgentFlow, ToolFlow, MapFlow
    """
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: List[str],
        branches: List[Union[Workflow, Agent, Tool]] = [],
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
        flatten: bool = False,
        *,
        max_workers: int = 16
    ) -> None:
        # Initialize base with the broadcast schema
        super().__init__(
            name=name,
            description=description,
            input_schema=list(input_schema),
            output_schema=list(output_schema),
            bundle_all=bundle_all,
        )

        self._flatten: bool = bool(flatten)
        self._branches: List[Workflow] = []
        self._max_workers = max_workers

        # Ingest/wrap branches; enforce unique names; prune non-conforming schemas
        wrapped: List[Workflow] = []
        seen: set[str] = set()
        for obj in branches or []:
            wf = _to_workflow(obj, in_sch=self._input_schema)
            if wf.name in seen:
                raise ValueError(f"{self.name}: duplicate branch name '{wf.name}'")
            seen.add(wf.name)
            # Prune if schema not set-equivalent
            if set(wf.input_schema) == set(self._input_schema):
                wrapped.append(wf)
        self._branches = wrapped

    # -------------------- Properties --------------------

    @property
    def branches(self) -> List[Workflow]:
        """Read-only view of branches (shallow copy)."""
        return list(self._branches)

    # -------------------- Branch Management --------------------

    def add_branch(self, obj: Union[Workflow, Agent, Tool]) -> None:
        """
        Add a branch. Name must be unique. The branch's input_schema must be
        set-equivalent to the fixed broadcast schema or this raises ValueError.
        """
        wf = _to_workflow(obj, in_sch=self._input_schema)
        if any(b.name == wf.name for b in self._branches):
            raise ValueError(f"{self.name}: duplicate branch name '{wf.name}'")
        if set(wf.input_schema) != set(self._input_schema):
            raise ValueError(
                f"{self.name}: branch '{wf.name}' input_schema {wf.input_schema} "
                f"!= broadcast input_schema {list(self._input_schema)} (set mismatch)"
            )
        self._branches.append(wf)

    def remove_branch(self, name: Optional[str] = None, *, position: int = -1) -> Workflow:
        """
        Remove a branch by exact name or by position (default: last). Returns the removed workflow.
        """
        if not self._branches:
            raise IndexError(f"{self.name}: no branches to remove")
        if name is not None:
            for i, b in enumerate(self._branches):
                if b.name == name:
                    return self._branches.pop(i)
            raise ValueError(f"{self.name}: no branch named '{name}' to remove")
        # by position
        return self._branches.pop(position)

    # -------------------- Memory --------------------

    def clear_memory(self) -> None:
        Workflow.clear_memory(self)
        for b in self._branches:
            try:
                b.clear_memory()
            except Exception:
                logger.warning("%s: branch.clear_memory raised during clear_memory", self.name, exc_info=True)

    # -------------------- Execution --------------------

    def _process_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fan out across branches concurrently and return:
            - { branch.name: dict_result, ... } when self._flatten is False
            - a single flattened dict when self._flatten is True

        Every branch receives the same validated `inputs` dict. Each result must be a dict.
        """
        if not self._branches:
            logger.error("%s: no branches available", self.name)
            raise ValidationError(f"{self.name}: No branches available")

        max_workers = min(len(self._branches), self._max_workers) if self._branches else 1
        futures: List[tuple[str, Any]] = []
        results_by_branch: Dict[str, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for b in self._branches:
                futures.append((b.name, pool.submit(b.invoke, inputs)))

            first_exc: Optional[BaseException] = None
            failing_branch: Optional[str] = None

            # Collect in declared branch order for determinism
            for branch_name, fut in futures:
                if first_exc is not None:
                    if not fut.done():
                        fut.cancel()
                    continue
                try:
                    res = fut.result()
                    if not isinstance(res, dict):
                        raise ValidationError(
                            f"{self.name}:{branch_name} returned non-dict result "
                            f"({type(res).__name__}); expected dict"
                        )
                    results_by_branch[branch_name] = res
                except BaseException as e:
                    first_exc = e
                    failing_branch = branch_name
                    # Attempt to cancel remaining futures
                    for _, f in futures:
                        if not f.done():
                            f.cancel()

            if first_exc is not None:
                logger.exception("%s: branch %s failed during ScatterFlow execution", self.name, failing_branch)
                raise ExecutionError(f"{self.name}:{failing_branch} failed") from first_exc

        if not self._flatten:
            return results_by_branch

        # Flattening path
        flat: Dict[str, Any] = {}
        special_keys = {WF_RESULT, JUDGE_RESULT}

        for b in self._branches:
            if b.name not in results_by_branch:
                # either missing input or (in theory) not executed; skip entirely
                continue
            payload = results_by_branch[b.name]
            if payload is None:
                # shouldn't occur here because we don't insert None for flatten=True
                continue

            # unwrap any special keys if using special keys
            if _is_namedtuple(payload):
                payload = _namedtuple_as_mapping(payload)
            elif is_dataclass(payload):
                payload = asdict(payload)
            while isinstance(payload, MAPPABLE_KINDS) and len(payload.keys()) == 1 and list(payload.keys())[0] in special_keys:
                payload = payload[list(payload.keys())[0]]
            if not isinstance(payload, dict):
                flat[b.name] = payload
                continue

            # Normal flatten: merge keys with collision check
            for k, v in payload.items():
                if k in flat:
                    raise ValidationError(
                        f"{self.name}: flatten key collision for '{k}' "
                        f"(originating branch '{b.name}')"
                    )
                flat[k] = v
        return flat