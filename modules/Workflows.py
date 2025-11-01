"""
Workflows
=========

This module defines the Workflow system — a collection of deterministic, schema-based
coordinators that connect and orchestrate Agents, Tools, and other Workflows into
reliable pipelines of logic. Each workflow operates under a uniform call signature
(`invoke(inputs: dict) -> dict`) and uses ordered `input_schema` and `output_schema`
definitions to enforce consistent data flow between components. The goal is to provide
a predictable, provider-agnostic backbone for building structured, multi-step reasoning
and execution chains across any combination of agentic or deterministic logic units.

Workflows are built on the principle of schema determinism: every step declares its
expected inputs and outputs, ensuring that data passed between nodes is validated and
normalized at runtime. This design yields reproducible, debuggable behavior, making it
possible to trace and verify complex orchestration sequences with confidence.

The module supports a range of coordination patterns commonly used in agentic
architectures, including:\n
    - ToolFlow and AgentFlow — wrapping tools or agents in schema-aware workflow nodes.
    - ChainFlow — linear or chain-of-thought pipelines that pass results step-to-step.
    - BatchFlow — parallel fan-out of multiple branches with aggregated results.
    - MakerChecker — iterative maker-reviewer cycles with optional automated judgment.
    - Selector — strategy-style conditional routing between alternative branches.
    - LangGraphFlow — graph-based orchestration using uniform schema propagation.

All workflows share a consistent checkpointing system that records timestamps, input
snapshots, and normalized outputs after every invocation, enabling full traceability
and validation of runs. They can be freely composed or subclassed to introduce new
coordination patterns, provided the schema-based invocation contract and deterministic
checkpoint behavior are maintained.
"""

# ============================================================
# Standard Library Imports
# ============================================================
from __future__ import annotations  # Enables forward type hints for class references
from abc import ABC, abstractmethod  # Abstract base classes
from datetime import date, datetime, time                    # For timestamping checkpoints
from typing import Any, Dict, List, Optional, Set, Union, Tuple, TypedDict  # Type hinting
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel execution
import logging                 # Logging support
from copy import deepcopy
import json
from decimal import Decimal
from uuid import UUID
from pathlib import Path


# ============================================================
# Atomic-Agentic Modules
# ============================================================
from .Tools import Tool             # Used by ToolFlow for wrapping Tools as Workflows
from .Agents import Agent           # Used by AgentFlow for wrapping Agents as Workflows

# ============================================================
# Third-Party Dependencies
# ============================================================
from langgraph.graph import StateGraph, END

# ============================================================
# Constants
# ============================================================
WF_RESULT = "__wf_result__"         # Default single-output field for packaging results
JUDGE_RESULT = "__judge_result__"   # Standard field for boolean judge outputs


class Workflow(ABC):
    """
    Abstract base for deterministic, schema-driven orchestration.

    Mission
    -------
    A Workflow coordinates a single synchronous "step" that wraps a callable
    unit (e.g., a Python method, Tool, Agent, or another Workflow) behind a
    uniform interface. It enforces input/output schemas, performs strict input
    validation, delegates execution to `_process_inputs(inputs)`, and then
    packages the raw result into a dict conforming to `output_schema`.

    Contract
    --------
    - Entry point: `invoke(inputs: dict) -> dict`
      - Validates `inputs` against `input_schema`.
      - Delegates to `_process_inputs(inputs)` for the actual execution.
      - Packages the raw result via `package_results(result)` to match
        `output_schema`, honoring bundling/unwrap options.
      - Records a timestamped checkpoint of the final packaged output.

    Input Validation
    ----------------
    - Keys in `inputs` must be a subset of `input_schema`. Any unexpected key
      raises `ValueError`.
    - Missing keys allowed; unexpected keys raise.
    - Inputs are treated as immutable; internal logic operates on deep copies.

    Output Packaging
    ----------------
    - Regardless of the internal processes, the final output will always be
      a dictionary with keys matching an `output_schema`.
    - Depending on initialization settings, if the output_schema is a single
      key, then the raw output will then be bundled under that single key,
      regardless of type.
    - Packaging raw outputs to match the output schema is handled by a method
      `package_results`
    
    Determinism & Checkpointing
    ---------------------------
    - All transformations are pure with respect to `inputs`.
    - Every successful `invoke` stores a deep-copied, timestamped checkpoint of
      the final packaged output for inspection/reset.
    - `clear_memory()` resets internal checkpoints/state without affecting
      wrapped callables (unless a subclass explicitly cascades clearing).

    Extensibility
    -------------
    - Subclasses implement `_process_inputs(inputs: dict) -> Any`.
    - Subclasses SHOULD NOT bypass `invoke`; always call `invoke` to get
      consistent validation, packaging, and checkpointing.
    """


    def __init__(
        self,
        name: str,
        description: str,
        input_schema: List[str],
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
    ) -> None:
        self._name = name
        self._description = description
        self._input_schema: List[str] = input_schema or []
        self._output_schema: List[str] = output_schema
        self._bundle_all = (bool(bundle_all) and len(self._output_schema) == 1) or self._output_schema == [WF_RESULT]
        self._checkpoints: List[Dict[str, Any]] = []

        # Basic validation
        if self._input_schema is None:
            raise ValueError(f"{self.name}: input_schema must be a non-empty list of field names")
        if not self._output_schema:
            raise ValueError(f"{self.name}: output_schema must be a non-empty list of field names")
        if any(not isinstance(k, str) or not k for k in self._input_schema):
            raise ValueError(f"{self.name}: input_schema must contain non-empty strings")
        if any(not isinstance(k, str) or not k for k in self._output_schema):
            raise ValueError(f"{self.name}: output_schema must contain non-empty strings")
        if self._bundle_all and len(self._output_schema) != 1:
            raise ValueError(
                f"{self.name}: bundle_all=True requires single-key output_schema, got {self._output_schema}"
            )

    # -----------------------------
    # Read-only configuration
    # -----------------------------
    @property
    def input_schema(self) -> List[str]:
        return list(self._input_schema)

    @property
    def output_schema(self) -> List[str]:
        """Get the current output schema (copy)."""
        return list(self._output_schema)
    
    @property
    def bundle_all(self) -> bool:
        return self._bundle_all
    
    @property
    def description(self):
        return self._description
    
    @property
    def name(self) -> str:
        """Distinguished workflow name based on the tool/agent/workflow it wraps."""
        return self._name
    
    # -----------------------------
    # Mutatable attributes
    # -----------------------------
    @output_schema.setter
    def output_schema(self, schema: List[str]) -> None:
        """Set a new output schema; must be non-empty strings. If bundle_all=True, schema must be length 1."""
        if not schema:
            raise ValueError("Output Schema must be set to a non-empty list of strings.")
        if any((not isinstance(key, str) or not key.strip()) for key in schema):
            raise ValueError("Output schema must be a list of strings.")
        # unbundle if more than 1 key
        if len(schema) > 1: self._bundle_all = False
        # bundle if using default result
        if schema == [WF_RESULT]: self._bundle_all = True
        self._output_schema = list(schema)
    
    @bundle_all.setter
    def bundle_all(self, flag: bool) -> None:
        """Enable/disable bundle-all mode; when enabling, output_schema must be length 1."""
        # don't enable if schema is longer than 1
        if flag and (len(self._output_schema) != 1):
            raise ValueError(
                f"{self.name}: enabling bundle_all requires single-key output_schema; got {self._output_schema}"
            )
        # always allow us to disable bundling
        self._bundle_all = flag

    # -----------------------------
    # Checkpointing API
    # -----------------------------
    @property
    def checkpoints(self) -> List[Dict[str, Any]]:
        """A deep copy of the checkpoints history (append-only audit log)."""
        # requires: from copy import deepcopy
        return deepcopy(self._checkpoints)

    @property
    def latest_result(self) -> Optional[Dict[str, Any]]:
        """The most recent packaged result, or None if no invocations yet."""
        if not self._checkpoints: return None
        return deepcopy(self._checkpoints[-1].get("result"))

    def clear_memory(self) -> None:
        """Clear workflow checkpoints."""
        self._checkpoints = []

    # -----------------------------
    # Packaging
    # -----------------------------
    def package_results(self, results: Any) -> Dict[str, Any]:
        """
        Normalize `results` into a dict matching `output_schema`.

        Packaging Rules (strict, deterministic):
          1) Unwrap while results == {WF_RESULT: x} (single key) → results = x.
          2) If results is scalar (not dict/list/tuple):
               - If single-key schema: {schema[0]: results}
               - Else: raise.
          3) If results is a sequence (list/tuple):
               - If bundle_all: {schema[0]: results}
               - Else:
                   * if len(seq) > len(schema): raise (too many results)
                   * pad with None to len(schema), map positionally to schema keys
          4) If results is a dict:
               - If keys match: results
               - If bundle_all: {schema[0]: results}
               - Else if results.keys() ⊆ schema_keys:
                   * return {k: results.get(k, None)} for k in schema (pad missing with None)
                 Else:
                   * if len(results) > len(schema): raise (too many results)
                   * positional map by insertion-order of values() to schema keys
                     and pad with None if fewer

        Raises
        ------
        ValueError
            If `results` cannot be deterministically packaged into `output_schema`.
        """
        schema = self._output_schema
        if not schema:
            raise ValueError(f"{self.name}: output_schema must be non-empty")

        # (1) Unwrap nested {WF_RESULT: x} until its no longer the sole key
        while isinstance(results, dict) and len(results) == 1 and WF_RESULT in results:
            results = results[WF_RESULT]

        # (2) Scalar path
        if not isinstance(results, (dict, list, tuple)):
            if len(schema) == 1: return {schema[0]: results}
            raise ValueError(
                f"{self.name}: scalar result cannot be packaged into multi-key schema {schema}"
            )

        # Sequence path (3)
        if isinstance(results, (list, tuple)):
            # if bundle, do so regardless of # of entries
            if self.bundle_all: return {schema[0]: results}
            # mapping values to schema 1-1
            elif len(results) == len(schema):
                return {k:v for k,v in zip(schema, results)}
            # pad out missing positional values
            elif len(results) < len(schema):
                padded = list(results) + [None] * (len(schema) - len(results))
                return {k: v for k, v in zip(schema, padded)}
            # if length is greater than schema
            raise ValueError(
                f"{self.name}: We received a result length {len(results)}, "
                f"which exceeds the schema length {len(schema)}."
            )
        # Dict path (4)
        assert isinstance(results, dict)
        # No duplicate key wrapping, even if bundle_all
        if set(list(results.keys())) == set(schema): return results
        # else if bundling outputs, do so regardless of # of keys
        if self.bundle_all: return {schema[0] : results}
        # else if keys a subset of schema
        if set(list(results.keys())).issubset(set(schema)): return {k: results.get(k, None) for k in schema}
        # else if not a subset of schema keys and not bundling, raise an error
        if len(results.keys()) != len(schema):
            # else if there are unrecognizeable keys or result is longer/shorter than schema
            raise ValueError(
                f"{self.name}: We accept a dict that is a subset of {schema} "
                "or a dict with a number of keys matching the expected schema length. "
                f"Instead, we got a dictionary with {len(results)} keys that isn't a "
                "subset of our desired schema."
            )
        # if key sets are different and not bundle all, positionally pad values
        key_map = {s:r for s,r in zip(schema, list(results.keys()))}
        return {s:results[key_map[s]] for s in schema}
    
    # -----------------------------
    # Template method contract
    # -----------------------------
    @abstractmethod
    def _process_inputs(self, inputs: dict) -> Any:
        """
        Subclass-specific processing for a single invocation.

        Implementations should:
          - Assume `inputs` may omit some fields from `input_schema` (missing keys are allowed).
          - Forbid relying on extra/unexpected keys; `invoke` already validates and rejects them.
          - Return any Python object (scalar, sequence, or dict). The base class will package it.

        Parameters
        ----------
        inputs : dict
            A dictionary of inputs keyed by `input_schema`. Missing keys are allowed and should
            be handled by the subclass (e.g., via defaults).

        Returns
        -------
        Any
            The raw result to be normalized by `package_results`.
        """
        pass

    def invoke(self, inputs: dict) -> dict:
        """
        Execute the workflow using the template method pattern:

          1) Validate inputs against `input_schema` (unexpected keys → error).
          2) Delegate to subclass `_process_inputs(inputs)`.
          3) Normalize the raw result using `package_results`.
          4) Append a checkpoint and return the normalized dict.

        Parameters
        ----------
        inputs : dict
            Input data; keys must be a subset of `input_schema` (unexpected keys raise).

        Returns
        -------
        dict
            Normalized output matching `output_schema`.
        """
        # Forbid unexpected input keys; allow missing keys
        for key in inputs.keys():
            if key not in self._input_schema:
                raise ValueError(f"{self.name}: Received unexpected input key '{key}', not in input_schema {self._input_schema}")

        # Process and package
        raw = self._process_inputs(inputs)
        result = self.package_results(raw)

        # Checkpoint (deep copies to protect audit history)
        self._checkpoints.append({
            "timestamp": datetime.now().isoformat(),
            "inputs": deepcopy(Workflow._sanitize_for_json(inputs)),
            "raw": deepcopy(Workflow._sanitize_for_json(raw)),
            "result": deepcopy(Workflow._sanitize_for_json(result)),
        })

        return result

    @staticmethod
    def _sanitize_for_json(value):
        """
        Recursively transform `value` into a JSON-serializable form that preserves
        readability. Non-serializable leaves are wrapped with a marker.

        Returns a structure safe for `json.dumps(...)`.
        """
        PRIMITIVES = (str, int, float, bool, type(None))
        if isinstance(value, PRIMITIVES):
            return value

        # Common quasi-primitives → stringified with type marker
        if isinstance(value, (datetime, date, time)): return value.isoformat()
        if isinstance(value, Path): return str(value)
        if isinstance(value, Decimal): return float(value)
        if isinstance(value, UUID): return str(value)
        if isinstance(value, Exception): return str(value)

        # dict: ensure string keys; recurse on values
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                key_str = k if isinstance(k, str) else str(k)
                out[key_str] = Workflow._sanitize_for_json(v)
            return out

        # list/tuple: recurse to list
        if isinstance(value, (list, tuple, set, frozenset)):
            return [Workflow._sanitize_for_json(v) for v in list(value)]

        # Fallback for arbitrary objects
        try:
            # if it has a reasonable __json__ hook or similar in the future, could be extended
            return {
                "__type__": type(value).__name__,
                "value": str(value),
            }
        except Exception:
            return {
                "__type__": type(value).__name__,
                "value": repr(value),
            }


class AgentFlow(Workflow):
    """
    Wraps an Agent into a schema-aware Workflow node.

    Input/Output
    ------------
    - input_schema: defaults to ["prompt"], but can be any non-empty string list
    - output_schema: defaults to [WF_RESULT] unless overridden.

    Invocation
    ----------
    - The 'prompt' value is sanitized for JSON readability:
        * JSON-native primitives (str, int, float, bool, None) pass through unchanged.
        * dict/list/tuple recurse with the same rule.
        * sets are converted to {"__type__": "set", "values": [...]}
        * non-serializable leaves (e.g., datetime, Path, custom objects) are
          converted to {"__stringified__": true, "__type__": "<TypeName>", "value": "<str()>"}
      This ensures `json.dumps(...)` never fails while remaining human-readable.
    - If 'prompt' is already a string, it is passed as-is.
      Otherwise the sanitized structure is JSON-encoded (UTF-8, no ASCII escaping).

    Notes
    -----
    - Payload values that are not strings or collections of strings are coerced to strings during dict sanitation.
    """

    def __init__(self,
                 agent: Agent,
                 input_schema: list[str] = ["prompt"],
                 output_schema: List[str] = [WF_RESULT],
                 bundle_all: bool = True,
    ):
        if input_schema == []:
            raise TypeError("AgentFlow: Cannot initialize agent-flows with empty input schemas")
        super().__init__(
            name = agent.name,
            description = agent.description,
            input_schema = input_schema,
            output_schema = output_schema,
            bundle_all=bundle_all
        )
        self._agent = agent

    # -------------------- Introspection --------------------
    @property
    def agent(self) -> Agent:
        """The wrapped Agent object."""
        return self._agent
    @agent.setter
    def agent(self, val: Agent) -> None:
        """Replaces agent"""
        self._agent = val

    # -------------------- Execution --------------------
    def clear_memory(self):
        """Clear workflow checkpoints and the wrapped agent's memory."""
        Workflow.clear_memory(self)
        self.agent.clear_memory()

    def _process_inputs(self, inputs: dict) -> str:
        """
        Execute the wrapped agent using JSON-readable prompt normalization.

        Behavior:
        - Return the agent.invoke(stringified inputs (or value at key 'prompt'))
        """
        # Reject empty inputs
        if not len(inputs):
            raise TypeError(f"{self.name}: Cannot provide empty inputs to an agent flow.")
        # reject
        if list(inputs.keys()) == ["prompt"]:
            return self.agent.invoke(str(inputs["prompt"]))
        sanitized = Workflow._sanitize_for_json(inputs)
        stringified_inputs = json.dumps(sanitized, ensure_ascii=False)
        return self.agent.invoke(stringified_inputs)


class ToolFlow(Workflow):
    """
    Workflow wrapper for a single `Tool`.

    This flow enforces a dict-based invocation contract over a Tool’s callable,
    preserves the Tool’s input signature as an immutable `input_schema`, and
    allows controlled customization of the output channel via `output_schema`
    and `bundle_all`.

    Design
    ------
    - Identity:
        * `name` is exposed as "workflow.<tool.name>" to distinguish it from the Tool.
        * `description` is derived from the Tool and is immutable in ToolFlow.
    - Schemas:
        * `input_schema` is derived from the Tool’s signature (ordered) and is read-only.
        * `output_schema` is configurable post-init via a validated setter.
        * `bundle_all` is configurable; when True, `output_schema` must have length 1.
    - Execution:
        * Do not override `invoke`; base `Workflow.invoke` handles validation,
          packaging, and checkpointing. ToolFlow implements `_process_inputs`
          and calls the underlying Tool with the provided `inputs` as **kwargs**.
    """

    def __init__(
        self,
        tool: Tool,
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
    ) -> None:
        # Filter out varargs/kwargs conventional placeholders if present
        input_schema: List[str] = [p for p in list(tool.signature_map.keys()) if p not in ("*args", "**kwargs")]

        # If base currently forbids empty, this will raise during super().__init__.
        super().__init__(
            name=tool.name,                         # base will store; we expose a property wrapper
            description=tool._description,           # immutable, derived below
            input_schema=input_schema,
            bundle_all=bundle_all,
            output_schema=output_schema,
        )
        # Store the tool reference
        self._tool: Tool = tool

    # ---------------------------------------------------------------------
    # Identity & metadata (read-only)
    # ---------------------------------------------------------------------
    @property
    def tool(self) -> Tool:
        """The wrapped tool (read-only)."""
        return self._tool

    @property
    def description(self):
        template = """
        Wraps the tool `{tool_name}` in a workflow. It expects a key-word dictionary
        representation for the inputs to be passed into the tool for the following keys:
        {keys}
        
        Wrapped Tool description: {description}
        """
        keys = "".join([f"\n- {key}" for key in self.input_schema]) if self.input_schema else "\n(no parameters)"
        tool_desc = self.tool._description
        return template.format(tool_name=self.tool.name, keys=keys, description=tool_desc.strip())


    # ---------------------------------------------------------------------
    # Execution (template method hook)
    # ---------------------------------------------------------------------
    def _process_inputs(self, inputs: dict) -> Any:
        """
        Delegate to the underlying Tool using dict → **kwargs calling convention.
        Missing keys are allowed (Tool defaults apply). Unexpected keys are
        already rejected by the base `invoke`.
        """
        return self._tool.invoke(**inputs)

    # ---------------------------------------------------------------------
    # Memory
    # ---------------------------------------------------------------------
    def clear_memory(self) -> None:
        """
        Clear checkpoints and cascade to the Tool if it exposes a clear_memory hook.
        """
        super().clear_memory()
        self._tool.clear_memory()

# wraps all incoming objects into workflow classes and adjusts their input/output schemas based on optional parameters
def _to_workflow(obj: Agent | Tool | Workflow, in_sch:list[str]|None = None, out_sch:list[str]|None = None) -> Workflow:
        if isinstance(obj, Workflow):
            obj.output_schema = out_sch or obj.output_schema
            return obj
        if isinstance(obj, Agent): return AgentFlow(obj, input_schema=in_sch or ["prompt"], output_schema=out_sch or [WF_RESULT])
        if isinstance(obj, Tool): return ToolFlow(obj, output_schema=out_sch or [WF_RESULT])
        raise TypeError(f"Object must be Agent, Tool, or Workflow. Got unexpected '{type(obj).__name__}'.")


class ChainFlow(Workflow):
    """
    ChainFlow
    ---------
    Linear pipeline of steps (each a Workflow/ToolFlow/AgentFlow) with strict internal
    reconciliation and a ChainFlow-level output contract (overlay) that shapes the final
    result without mutating the last child.

    STATES
    ======
    • Empty (len(steps) == 0)
      - input_schema/output_schema: freely set; they mirror each other.
      - bundle_all: ALWAYS False (setter rejects enabling).
      - _process_inputs: identity (returns inputs unchanged).

    • Non-empty (len(steps) >= 1)
      - input_schema: derived from first step; setter rejected.
      - output_schema: ChainFlow-level overlay ONLY (no child mutation):
          * If len(schema) > 1  -> force bundle_all = False automatically.
          * If len(schema) == 1 -> do NOT auto-enable bundle_all; caller decides.
          * If effective bundle_all == False -> require len(schema) >= len(last.output_schema).
      - bundle_all: ChainFlow-level overlay ONLY:
          * Enabling True requires len(output_schema) == 1.
          * Disabling False always allowed.

    INTERNAL HANDOFFS
    =================
    For each adjacent (A -> B):
      - If A.bundle_all: A.out := B.in (length may differ); assert persisted equality.
      - Else: require len(A.out) == len(B.in); then A.out := B.in.
    """

    def __init__(
        self,
        name: str,
        description: str,
        steps: List[Agent | Tool | Workflow] | None = None,
        output_schema: List[str] | None = None,
        *,
        zero_step_schema: list[str] = [WF_RESULT],
    ):
        # validate zero_step_schema
        if zero_step_schema == []:
            raise TypeError("ChainFlow: While a chain-flow lacks any intermediate steps, the default identity schema cannot be an empty list")
        if any(not isinstance(key, str) for key in zero_step_schema):
            raise TypeError("ChainFlow: The default schema assigned must be a list of strings, only")
        # build steps
        self._steps: list[Workflow] = []
        if steps:
            self._steps = [_to_workflow(s) for s in steps]
        super().__init__(
            name=name,
            description=description,
            input_schema=zero_step_schema,
            output_schema=zero_step_schema,
            bundle_all= zero_step_schema == [WF_RESULT] or (self._steps and self._steps[-1].bundle_all),
        )
        # Reserve for when no steps are present
        self._default_schema = zero_step_schema
        self._reconcile_all()
        self._mirror_endpoints_from_children()
        # if a custom output schema is provided, set it to that.
        if output_schema != None:
            self.output_schema = output_schema

    # -------------------- Steps management --------------------

    @property
    def steps(self) -> list[Workflow]:
        return list(self._steps)

    @steps.setter
    def steps(self, value: list[Agent | Tool | Workflow] | None):
        value = value or []
        self._steps = [_to_workflow(s) for s in value]
        if not self._steps:
            # Return to empty state invariants
            self._bundle_all = False
            # Keep user-configurable input/output overlays; they remain mirrored by setters.
            return
        self._reconcile_all()
        self._mirror_endpoints_from_children()

    def add_step(self, step: Agent | Tool | Workflow, position: int | None = None) -> None:
        wrapped = _to_workflow(step)
        if position is None:
            self._steps.append(wrapped)
        else:
            if position < 0 or position > len(self._steps):
                raise IndexError(f"{self.name}: add_step position out of range")
            self._steps.insert(position, wrapped)
        self._reconcile_all()
        self._mirror_endpoints_from_children()

    def pop(self, index: int = -1) -> Workflow:
        removed = self._steps.pop(index)
        if not self._steps:
            self._bundle_all = False  # empty state invariant
            return removed
        self._reconcile_all()
        self._mirror_endpoints_from_children()
        return removed

    # -------------------- Execution --------------------
    def _process_inputs(self, inputs: dict) -> dict:
        if not self._steps:
            return inputs  # identity
        current = inputs
        for step in self._steps:
            current = step.invoke(current)  # each child returns a dict
        return current  # ChainFlow overlay packaging happens in base invoke()

    # -------------------- Internals --------------------
    def _reconcile_all(self) -> None:
        if not self._steps:
            return
        for A, B in zip(self._steps, self._steps[1:]):
            a_out = list(A.output_schema)
            b_in  = list(B.input_schema)
            # Skip if they match already
            if a_out == b_in:
                continue
            '''
            Mutate A's output schema if one of the following is true:
            1) A is bundled (theoretically unbundled A out could align with B in)
            2) A out length = B out length (positional mapping of keys 1-1)
            3) A out is a subset of B in (pad out missing keys)
            '''
            if (A.bundle_all) or (len(a_out) == len(b_in)) or (set(a_out).issubset(set(b_in))):
                A.output_schema = b_in
                continue
            # If none of the above are valid, then raise an error
            raise ValueError(
                    f"{self.name}: length mismatch {A.name} → {B.name}: "
                    f"len(out)={len(a_out)} != len(in)={len(b_in)} (adapter required)"
                )

    def _mirror_endpoints_from_children(self) -> None:
        # if no steps are present, then chainflow becomes an identity flow
        if not self._steps:
            self._input_schema = self._default_schema
            self.output_schema = self._default_schema
        first = self._steps[0]
        last  = self._steps[-1]
        self._input_schema  = list(first.input_schema)
        self._output_schema = list(last.output_schema)
        self._bundle_all    = bool(last.bundle_all)


class MakerChecker(Workflow):
    """
    Maker-Checker composite workflow.

    Contracts:
      - maker.output_schema == checker.input_schema
      - checker.output_schema == maker.input_schema
      - judge (optional):
            input_schema == maker.input_schema
            output_schema == [__judge_result__]
            bundle_all == False

    Composability (ChainFlow-style overlays on THIS composite only):
      - On __init__ and on maker/checker replacement, this composite mirrors `maker`
        for its external endpoints (input_schema, output_schema, bundle_all).
      - Callers (e.g., ChainFlow) may set THIS composite's output_schema / bundle_all to relabel outputs.
        Overlay rules (using maker as reference):
          * If len(schema) > 1  -> force bundle_all=False.
          * If len(schema) == len(maker.output_schema) -> accept (positional relabel allowed).
          * If len(schema) > len(maker.output_schema)  -> require superset of maker.output_schema; else raise.
          * If len(schema) < len(maker.output_schema)  -> raise.
          * bundle_all=True requires single-key output_schema.

    Judge volatility policy:
      - If maker.input_schema changes (directly or via checker parity), judge is auto-dropped (None).
      - At invoke time, judge compatibility is rechecked and auto-dropped if stale.
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
    ) -> None:
        # Normalize participants
        maker_wf   = _to_workflow(maker)
        checker_wf = _to_workflow(checker)
        if max_revisions < 0:
            raise ValueError(f"{self.name}: max_revisions must be >= 0; got {max_revisions}")
        # Initialize base mirroring the *maker* endpoints
        super().__init__(
            name=name,
            description=description,
            input_schema=list(maker_wf.input_schema),
            output_schema=list(checker_wf.input_schema),
            bundle_all=maker_wf.bundle_all,
        )

        self._maker: Workflow = maker_wf
        self._checker: Workflow = checker_wf
        self.max_revisions: int = int(max_revisions)
        self._judge: Optional[Workflow] = None
        self._to_judge_adapter: dict = {}

        # Enforce maker <-> checker parity before initial mirror.
        self._reconcile_pair()
        
        # Judge initial sanity (if provided)
        if judge is not None:
            # build judge
            j_wf = _to_workflow(judge)
            j_wf.output_schema = [JUDGE_RESULT]
            j_wf.bundle_all = False
            # if lengths don't match
            if len(j_wf.input_schema) != len(self._maker.input_schema):
                raise ValueError(
                    f"{self.name}: Expected {len(self._maker.input_schema)} but only got {len(j_wf.input_schema)}."
                )
            # initialize mapping of maker input keys to judge input keys
            if set(j_wf.input_schema) == set(self._maker.input_schema):
                self._to_judge_adapter = {key:key for key in self._maker.input_schema}
            else:
                self._to_judge_adapter = {
                    m_key : j_key for m_key, j_key in zip(self._maker.input_schema, j_wf.input_schema)
                }
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
        if self._judge is not None and len(self._judge.input_schema) != self._maker.input_schema:
            self._judge = None
            self._to_judge_adapter = {}
        if self._judge:
            self._to_judge_adapter = {m_key : j_key for m_key, j_key in zip(self._maker.input_schema, self._judge.input_schema)}

    def _refresh_endpoints(self) -> None:
        """
        Mirror maker endpoints; used on initialization and when replacing maker/checker.
        Do NOT call this at invoke-time so that explicit composite-level overlays set
        by callers are not clobbered between invocations.
        """
        # must use private for input_schema
        self._input_schema = list(self._maker.input_schema)
        # must use output setter
        self.output_schema = list(self._maker.output_schema)
        # must use bundle all setter
        self.bundle_all = self._maker.bundle_all
        # set judge to None if judge.input_schema length no longer matches
        if len(self._maker.input_schema) != len(self._judge.input_schema):
            self._judge = None
            self._to_judge_adapter = {}
        # update adapter
        self._to_judge_adapter = {
            m:j for m,j in zip(self._maker.input_schema, self._judge.input_schema)
        }

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
        if len(j_wf) != len(self._maker.input_schema):
            raise ValueError(
                f"{self.name}: Expected {len(self._maker.input_schema)} but only got {len(j_wf.input_schema)}."
            )
        else:
            self._to_judge_adapter = {m:j for m,j in zip(self._maker.input_schema, j_wf.input_schema)}
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
                    raise ValueError(
                        f"{self.name}: judge must return a single boolean at key '{JUDGE_RESULT}'; got {decision!r}"
                    )
                if ok: break

            # Feedback: maker uses revisions to produce a new draft
            draft = self._maker.invoke(revisions)

        return draft


class Selector(Workflow):
    """
    Conditional router that delegates to one of several branches based on a judge workflow.

    Conventions
    -----------
    - Judge is any (Workflow|Agent|Tool). If Agent/Tool is supplied, it is wrapped into an
      AgentFlow/ToolFlow owned by this Selector.
    - The judge's *output_schema* is forcibly `[JUDGE_RESULT]`. At runtime, the judge must
      return a dict like `{JUDGE_RESULT: "<branch_name>"}`.
    - The Selector's *input_schema* strictly mirrors the judge's input schema. Because the
      base class exposes `input_schema` as read-only, we update via private `_input_schema`.
    - Each branch is any (Workflow|Agent|Tool). If Agent/Tool, it is wrapped into a Flow.
      Branch **input_schema** must set-equal the Selector's input schema. Branch outputs are
      independent from the Selector's advertised `output_schema`/`bundle_all`.
    - Packaging/checkpointing occur in `Workflow.invoke()`. `_process_inputs()` only chooses
      a branch and returns that branch's *raw* result.
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
            raise ValueError(
                f"{self.name}: duplicate branch names detected. Names given are {[b.name for b in branches]}."
            )
        if any(set(b.input_schema) != set(self._input_schema) for b in wrapped_branches):
            raise ValueError(
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
                raise ValueError(
                    f"{self.name}: branch '{wb.name}' input_schema {wb.input_schema} "
                    f"!= selector/judge input_schema {list(self._input_schema)} (set mismatch)"
                )
            if wb.name in seen_names:
                raise ValueError(f"{self.name}: duplicate branch name '{wb.name}'")
            seen_names.add(wb.name)
            wrapped.append(wb)
        self._branches = wrapped

    # -------------------- Public API --------------------
    def add_branch(self, branch: Agent|Tool|Workflow) -> None:
        # Wrap
        wb = _to_workflow(branch, self.input_schema, self.output_schema)
        # Validate input schema set-equivalence
        if set(wb.input_schema) != set(self.input_schema):
            raise ValueError(
                f"{self.name}: branch '{wb.name}' input_schema {wb.input_schema} "
                f"!= selector/judge input_schema {list(self._input_schema)} (set mismatch)"
            )
        # Unique name
        if any(b.name == wb.name for b in self._branches):
            raise ValueError(f"{self.name}: duplicate branch name '{wb.name}'")

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
            raise ValueError(f"{self.name}: There are no branches to choose from, thus failure.")
        # have judge select branch to invoke
        selection_raw = self._judge.invoke(inputs)
        # fail if selection isn't formatted correctly or returning a string
        if not isinstance(selection_raw, dict) or JUDGE_RESULT not in selection_raw:
            raise ValueError(f"{self.name}: judge must return a dict with key '{JUDGE_RESULT}'; got {selection_raw!r}")
        selection = selection_raw[JUDGE_RESULT]
        if not isinstance(selection, str) or not selection.strip():
            raise ValueError(f"{self.name}: judge must output a non-empty string under '{JUDGE_RESULT}', got {selection!r}")
        # Find and run branch with selection name (either the ._name or .name)
        selected_branch = next((b for b in self._branches if b._name == selection or b.name == selection), None)
        if selected_branch is None:
            raise RuntimeError(f"{self.name}: No branches matching the name for selection '{selection}'.")
        return selected_branch.invoke(inputs)


class MapFlow(Workflow):
    """
    MapFlow (tailored fan-out).

    Run multiple branches in parallel where each branch receives its own payload from
    `inputs[branch.name]`. The input schema is dynamic and always equals the ordered list
    of current branch names. Missing branch names in `inputs` do not error:
      - flatten=False  -> result for that branch is None
      - flatten=True   -> branch is ignored (not invoked)

    Packaging/bundling/checkpointing are handled by Workflow.invoke(); this subclass only
    implements _process_inputs().

    Execution
    ---------
    - Enqueue only branches that have a present payload and that payload is a dict.
      (If a present payload is not a dict, raise ValueError naming the branch.)
    - Run concurrently (ThreadPool), collect results in declared branch order (deterministic).
    - Each executed branch must return a dict, else TypeError naming the branch.
    - When flatten=False (default): return { branch.name: dict|None, ... } (all branches present).
    - When flatten=True: merge keys from executed branch dicts into a single dict.
        * If a branch result is exactly {WF_RESULT: ...} or {JUDGE_RESULT: ...}, do not flatten;
          instead insert as {branch.name: <raw_result_dict>} (prevents special keys at top-level).
        * Key collisions raise ValueError with the offending key and branch identity.

    Notes
    -----
    - Branches are wrapped via _to_workflow(obj, in_sch=None, out_sch=None). MapFlow does not
      mutate child schemas; overlays are applied only at the MapFlow boundary by base invoke().
    """

    def __init__(
        self,
        name: str,
        description: str,
        branches: List[Union[Workflow, Agent, Tool]] = [],
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
        flatten: bool = False,
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
        wf = self._to_workflow(obj, in_sch=None, out_sch=None)
        if any(b.name == wf.name for b in self._branches):
            raise ValueError(f"{self.name}: duplicate branch name '{wf.name}'")
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
                pass

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
            raise RuntimeError(f"{self.name}: No branches available")

        # Partition branches: which have payloads, which don't
        to_run: List[tuple[str, Workflow, Dict[str, Any]]] = []
        missing_names: List[str] = []

        for b in self._branches:
            if b.name in inputs:
                payload = inputs[b.name]
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"{self.name}: payload for branch '{b.name}' must be a dict; "
                        f"got {type(payload).__name__}"
                    )
                to_run.append((b.name, b, payload))
            else:
                missing_names.append(b.name)

        # Concurrency: execute only those with payloads
        from concurrent.futures import ThreadPoolExecutor

        results_by_branch: Dict[str, Optional[Dict[str, Any]]] = {}

        # Initialize defaults for missing branches (flatten=False only)
        if not self._flatten:
            for name in missing_names:
                results_by_branch[name] = None

        first_exc: Optional[BaseException] = None
        failing_branch: Optional[str] = None

        if to_run:
            max_workers = min(len(to_run), 16)
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
                            raise TypeError(
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
            raise RuntimeError(f"{self.name}:{failing_branch} failed") from first_exc

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

            # Special-envelope rule: exact single-key {WF_RESULT: ...} or {JUDGE_RESULT: ...}
            if len(payload) == 1:
                only_key = next(iter(payload))
                if only_key in special_keys:
                    if b.name in flat and not isinstance(payload[only_key], dict):
                        raise ValueError(
                            f"{self.name}: flatten collision for branch name '{b.name}'"
                        )
                    if isinstance(payload[only_key], dict) and not any(not isinstance(k, str) for k in payload[only_key].keys()):
                        payload = payload[only_key]
                    else:
                        flat[b.name] = payload
                        continue
            else: pass

            # Normal flatten: merge keys with collision check
            for k, v in payload.items():
                if k in flat:
                    raise ValueError(
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
    ScatterFlow (broadcast fan-out).

    Broadcast a single validated input dict to multiple child workflows in parallel and
    aggregate results. Packaging/bundling/checkpointing are handled by Workflow.invoke();
    this subclass only implements _process_inputs().

    Input schema
    ------------
    - Fixed at construction time and immutable thereafter.
    - Every branch must accept the same input schema (set-equivalent).
    - Branches are wrapped via _to_workflow(obj, in_sch=<broadcast schema>, out_sch=None).
    - At construction, non-conforming branches are pruned.
    - On add, non-conforming branches raise ValueError.

    Execution
    ---------
    - _process_inputs runs branches concurrently (ThreadPool), collects results in the
      declared branch order, and returns either:
        * { branch.name: dict_result, ... } when flatten=False (default), or
        * a single flattened dict when flatten=True, merging keys from each branch result.

    Result rules
    ------------
    - Each branch's invoke() MUST return a dict; non-dict results raise TypeError.
    - When flatten=True:
        * If a branch result dict has exactly one key and that key is WF_RESULT or JUDGE_RESULT,
          the flattened output receives { branch.name: <raw_result_dict> } (do not flatten this
          envelope) to avoid redundant/special keys polluting the top level.
        * Otherwise, merge the branch result's items into the flattened dict.
          Key collisions raise ValueError with the offending key and branch identities.
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
            b.clear_memory()

    # -------------------- Execution --------------------

    def _process_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fan out across branches concurrently and return:
            - { branch.name: dict_result, ... } when self._flatten is False
            - a single flattened dict when self._flatten is True

        Every branch receives the same validated `inputs` dict. Each result must be a dict.
        """
        if not self._branches:
            raise RuntimeError(f"{self.name}: No branches available")

        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(len(self._branches), 16) if self._branches else 1
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
                        raise TypeError(
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
                raise RuntimeError(f"{self.name}:{failing_branch} failed") from first_exc

        if not self._flatten:
            return results_by_branch

        # Flattening path
        flat: Dict[str, Any] = {}
        special_keys = {WF_RESULT, JUDGE_RESULT}

        for bname, payload in results_by_branch.items():
            # Special-envelope rule: if payload is exactly {WF_RESULT: ...} or {JUDGE_RESULT: ...},
            # DO NOT flatten; map under branch name instead to avoid special keys at top-level.
            if len(payload) == 1:
                only_key = next(iter(payload))
                if only_key in special_keys:
                    if bname in flat and not isinstance(payload[only_key], dict):
                        raise ValueError(
                            f"{self.name}: flatten collision for branch name '{bname}'"
                        )
                    if isinstance(payload[only_key], dict) and not any(not isinstance(key, str) for key in payload[only_key].keys()):
                        payload = payload[only_key]
                    else:
                        flat[bname] = payload[only_key]
                        continue
                else: pass
            # Normal flatten: merge keys, enforcing no collisions
            for k, v in payload.items():
                if k in flat:
                    raise ValueError(
                        f"{self.name}: flatten key collision for '{k}' "
                        f"(originating branch '{bname}')"
                    )
                flat[k] = v

        return flat