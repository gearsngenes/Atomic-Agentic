"""
Workflows
=========

Overview
--------
Workflows are **instantiated, stateful pipeline objects** that accept and return **single-dict payloads**.
They provide **deterministic orchestration** of heterogeneous components (Tools, Agents, and other Workflows)
through explicit **input/output schemas** and **normalized packaging**. This yields predictable composition,
clear contracts, and auditability across complex, multi-step tasks.

Why Use Workflows
-----------------
• **Determinism & Contracts**: Inputs are validated against declared schemas; outputs are packaged to a declared shape.  
• **Composability**: Mix Tools, Agents, and Workflows without bespoke glue code.  
• **Traceability**: Timestamped checkpoints capture inputs, raw results, and packaged outputs.  
• **Provider-Agnostic**: Decouple orchestration from LLM/tool vendors and SDKs.  
• **Safety & Guardrails**: Schema checks, packaging rules, and boundary overlays prevent silent drift.

Core Guarantees (Module-Level Contract)
---------------------------------------
• **Single call shape**: `invoke(inputs: dict) -> dict` for every Workflow.  
• **Schema determinism**: `input_schema` and `output_schema` define accepted keys and returned keys.  
• **Packaging normalization**: Scalars, sequences, mappings, and records are normalized to exactly `output_schema`.
  Optional `bundle_all` allows single-key envelopes when appropriate.  
• **Error taxonomy**:  
  – `ValidationError`: inputs violate schema/structure.  
  – `SchemaError`: incompatible or illegal schema configurations.  
  – `PackagingError`: result cannot be normalized to the declared outputs.  
  – `ExecutionError`: underlying step raised.

Key Concepts
------------
• **Boundary Overlays**: Composite patterns (e.g., Chain/Selector/MakerChecker/Map/Scatter) can expose an external
  `output_schema`/`bundle_all` **at the boundary** without mutating children.  
• **Hybrid Mapping (dicts)**: Partial overlaps map by key and pad/strict as configured; envelopes can be used
  when arbitrary mappings must be preserved deterministically.  
• **State & Memory**: Checkpoints for observability; optional clearing/reset between runs.

Workflow Patterns (What they enable)
------------------------------------
• **ToolFlow** — *Capability Adaptation*  
  Enable any callable “tool” to participate in schema-checked orchestration. It binds dict inputs to the tool’s
  parameters and normalizes the tool’s return for downstream steps. Use when you want **typed, reusable operations**
  (parsers, converters, retrievers) to plug into pipelines.

• **AgentFlow** — *LLM/Agent Integration*  
  Bridge dict-shaped inputs to an Agent’s prompting/inference interface and bring the Agent’s output back into
  schema-checked form. Use when you need **reasoning or generation** but still want **deterministic I/O** and packaging.

• **ChainFlow** — *Linear Transformation*  
  Compose steps **sequentially** so each step’s packaged outputs become the next step’s inputs. Use when you need
  **progressive transformation** (extract → analyze → summarize) with **clear handoff contracts**.

• **MakerChecker** — *Quality Loop & Governance*  
  Run an iterative **produce → review (→ optional judge)** cycle until approved or capped. Use when you need
  **policy enforcement, drafting/revision**, and explicit acceptance criteria separating *creation* from *evaluation*.

• **Selector** — *Policy Routing*  
  Use a judge to select **one** branch to run from many. Use when choice depends on **inputs, policy, or heuristics**
  (e.g., route to a specialist agent/tool). Keeps **decision logic** separate from **execution paths**.

• **MapFlow** — *Heterogeneous Fan-Out by Name*  
  Dispatch **different payloads** to **different branches** in parallel (payloads keyed by branch name), then aggregate.
  Use when each branch performs **distinct work** on **distinct inputs** and you want **structured aggregation** or
  optional flattening.

• **ScatterFlow** — *Broadcast Fan-Out / Ensembling*  
  Broadcast the **same inputs** to multiple branches in parallel (enforce schema alignment), then aggregate.
  Use for **ensembles, consensus, redundancy**, or when comparing multiple approaches on the same data.

Concurrency & Ordering
----------------------
Fan-out patterns parallelize branches while preserving **deterministic aggregation order**. Boundaries ensure that
non-deterministic underlying steps still yield **deterministic packaged outputs**.

Best Practices
--------------
• Treat schemas as **public contracts**; prefer adapters when step keys differ.  
• Use `bundle_all` **only** with single-key envelopes and only when you truly need to preserve an arbitrary mapping.  
• Keep overlays at **composite boundaries**; avoid mutating child schemas for local convenience.  
• Rely on checkpoints for **debugging, audits, and reproducibility**.  
• Fail **fast and loud** on schema mismatches rather than silently coercing ambiguous shapes.

Out of Scope (What this docstring avoids)
-----------------------------------------
• Vendor specifics (LLM/provider APIs), HTTP plumbing, or SDK nuances.  
• Private helper semantics or logging internals.  
• Lengthy examples; see examples/tests for concrete usage patterns.

Glossary
--------
**Schema**: Ordered list of keys defining accepted inputs or produced outputs.  
**Packaging**: Normalization process that maps any step’s return into `output_schema`.  
**Overlay**: External output schema/bundling applied at a composite boundary without mutating children.  
**Envelope**: Single-key dict used to preserve arbitrary mappings or consolidate results predictably.
"""

# ============================================================
# Standard Library Imports
# ============================================================
from __future__ import annotations  # Enables forward type hints for class references
from abc import ABC, abstractmethod  # Abstract base classes
from datetime import date, datetime, time                    # For timestamping checkpoints
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import logging                 # Logging support
from copy import deepcopy
import json
from decimal import Decimal
from uuid import UUID
from pathlib import Path
from collections.abc import Mapping, Iterable, Sequence, Set
from dataclasses import is_dataclass, asdict


# ============================================================
# Atomic-Agentic Modules
# ============================================================
from .Tools import Tool             # Used by ToolFlow for wrapping Tools as Workflows
from .Agents import Agent           # Used by AgentFlow for wrapping Agents as Workflows
# ============================================================
# Constants
# ============================================================
WF_RESULT = "__wf_result__"         # Default single-output field for packaging results
JUDGE_RESULT = "__judge_result__"   # Standard field for boolean judge outputs

# Module logger for targeted observability
logger = logging.getLogger(__name__)


# Domain-specific exceptions
class WorkflowError(Exception):
    """Base class for workflow-related errors."""


class ValidationError(WorkflowError, ValueError):
    """Raised for input/output validation failures."""


class SchemaError(ValidationError):
    """Raised when schema definitions are malformed or incompatible."""


class PackagingError(ValidationError):
    """Raised when a raw result cannot be deterministically packaged into the
    configured output schema."""


class ExecutionError(WorkflowError, RuntimeError):
    """Raised when runtime processing (i.e., _process_inputs) fails."""


def _is_namedtuple(x) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_fields")

def _namedtuple_as_mapping(x) -> dict:
    return {f: getattr(x, f) for f in x._fields}

# Primary, tweakable groupings
MAPPABLE_KINDS  = (Mapping,)     # dict, defaultdict, ChainMap, etc.
ITERABLE_KINDS  = (Iterable,)    # list/tuple/deque/set/range/generator/... (strings excluded below)

# Small utility tuples for exclusions/special cases
STRINGISH       = (str, bytes, bytearray)


class Workflow(ABC):
    """
    One-line
    --------
    Abstract base for deterministic, schema-driven orchestration with single-dict I/O.

    Purpose
    -------
    An abstract class that orchestrates “steps” (Tools, Agents, or other Workflows) behind a uniform 
    `invoke(inputs: dict) -> dict`. Validates inputs against `input_schema`, delegates to 
    `_process_inputs`, and deterministically packages results to fit an `output_schema`. Maintains 
    timestamped checkpoints of inputs & results (pre-packaged and packaged) for future reference.

    Contract
    --------
    • Entry: `invoke(inputs: dict) -> dict` (do not override).
    • Subclasses MUST implement `_process_inputs(inputs: dict) -> Any`.
    • Inputs: keys MUST be a subset of `input_schema`; unexpected keys raise `ValidationError`.
      Missing keys are allowed; subclass/tool/agent may apply defaults.
    • Packaging: `package_results(result) -> dict` enforces `output_schema` with:
      – Namedtuple/dataclass → mapping.
      – Dict: exact/subset/superset → reordered/ padded/ filtered; hybrid mapping pairs extra
        result keys to remaining schema keys by position; otherwise, if `bundle_all=True` and
        `len(output_schema)==1`, nest under that key; else `PackagingError`.
      – Set-like → require `bundle_all` (single-key) or error.
      – Sequence/Iterable → positional mapping (padded or capped + exhaustion check); optional bundling.
      – 1-key fallback: any value ⇒ `{schema[0]: value}`.
    • Mutability: `output_schema` setter updates schema (and disables bundling if length > 1).
      `bundle_all` setter requires single-key `output_schema` when enabling.

    Key Parameters
    --------------
    • name: str; description: str.
    • input_schema: list[str] (read-only via property).
    • output_schema: list[str] (setter validates; may disable bundling).
    • bundle_all: bool (setter validates single-key when True).

    Checkpointing
    -------------
    • Appends deep-copied `{timestamp, inputs, raw, result}` per successful invoke.
    • Access via `checkpoints`; reset with `clear_memory()`.

    Errors
    ------
    `ValidationError`, `SchemaError`, `PackagingError`, `ExecutionError`.

    Example
    -------
    >>> out = wf.invoke({"prompt": "hello"})
    >>> out  # keys exactly match wf.output_schema
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: List[str],
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
    ) -> None:
        # In/Out schema validation
        if input_schema is None:
            raise SchemaError("Workflow: input_schema must be a non-empty list of field names")
        if output_schema is None or output_schema == []:
            raise SchemaError(f"Workflow: output_schema must be a non-empty list of field names")
        if any(not isinstance(k, str) or not k for k in input_schema):
            raise SchemaError(f"Workflow: input_schema must contain non-empty strings")
        if any(not isinstance(k, str) or not k for k in output_schema):
            raise SchemaError(f"Workflow: output_schema must contain non-empty strings")
        if bundle_all and len(output_schema) != 1:
            raise SchemaError(
                f"Workflow: bundle_all=True requires single-key output_schema, got {output_schema}"
            )

        self._name = name
        self._description = description
        self._input_schema: List[str] = list(input_schema)
        self._output_schema: List[str] = list(output_schema)
        self._bundle_all = bundle_all
        self._checkpoints: List[Dict[str, Any]] = []

        # Observability: record how this Workflow was configured
        logger.debug("%s: initialized with input_schema=%s output_schema=%s bundle_all=%s", self._name, self._input_schema, self._output_schema, self._bundle_all)
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
        if schema is None:
            raise TypeError("Workflow: expected a list of strings, but got a 'NoneType'")
        if schema == []:
            raise SchemaError("Workflow: Output Schema must be set to a non-empty list of strings.")
        if any((not isinstance(key, str) or not key.strip()) for key in schema):
            raise TypeError("Workflow: Output schema must be a list of strings. Mismatched type found in schema.")
        # unbundle if more than 1 key
        if len(schema) > 1: self._bundle_all = False
        self._output_schema = list(schema)
        logger.debug("%s: output_schema set to %s (bundle_all=%s)", self._name, self._output_schema, self._bundle_all)
    
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
    def package_results(self, results: object) -> dict:
        schema = self.output_schema
        if not schema:
            raise SchemaError(f"{self.name}: output_schema is empty")

        # 1) Named "record" types → treat as mapping
        if _is_namedtuple(results):
            results = _namedtuple_as_mapping(results)
        elif is_dataclass(results):
            results = asdict(results)

        # 2) Keep unwrapping WF_RESULT until it's not a 1-key mapping with WF_RESULT
        while isinstance(results, MAPPABLE_KINDS) and len(results) == 1 and WF_RESULT in results:
            results = results[WF_RESULT]

        # 3) Mapping path (hybrid mapping enabled)
        if isinstance(results, MAPPABLE_KINDS):
            # preserve declared order for schema, and insertion order for results
            schema_list = list(schema)
            schema_set  = set(schema)

            res_keys_list = list(results.keys())
            res_keys_set  = set(res_keys_list)

            # Fast paths
            if res_keys_set == schema_set:
                # exact name match → reorder to schema order
                return {k: results[k] for k in schema_list}

            if res_keys_set.issubset(schema_set):
                # results missing some schema keys → pad with None
                return {k: results.get(k, None) for k in schema_list}

            if schema_set.issubset(res_keys_set):
                # results have extras; keep only schema keys, preserve schema order
                return {k: results.get(k, None) for k in schema_list}

            # Hybrid path:
            #  - common keys map by name
            #  - missing on each side must be of equal cardinality
            common_keys = res_keys_set.intersection(schema_set)

            # preserve insertion order for "only in results"
            only_in_results = [k for k in res_keys_list if k not in schema_set]
            # preserve declared order for "only in schema"
            only_in_schema  = [k for k in schema_list if k not in res_keys_set]

            if len(only_in_results) == len(only_in_schema):
                # Build output starting with named matches
                out = {k: results[k] for k in schema_list if k in common_keys}
                # Pair remaining by position: results' insertion order → schema order
                for src_key, dst_key in zip(only_in_results, only_in_schema):
                    out[dst_key] = results.get(src_key)
                # Ensure all schema keys exist (defensive; should be complete)
                for k in schema_list:
                    out.setdefault(k, None)
                # Optional: log that hybrid mapping occurred (helps debugging)
                logger.debug(f"{self.name}: hybrid dict mapping used: "
                            f"matched={sorted(common_keys)}, "
                            f"positional={list(zip(only_in_results, only_in_schema))}")
                return out

            # Mismatch and cannot hybrid-map: try bundling; else fail fast
            if self.bundle_all:
                if len(schema_list) != 1:
                    raise PackagingError(f"{self.name}: bundle_all requires single-key schema")
                return {schema_list[0]: dict(results)}

            raise PackagingError(
                f"{self.name}: mapping keys cannot be aligned: "
                f"only_in_results={only_in_results}, only_in_schema={only_in_schema}. "
                f"Provide an adapter, adjust schema, or enable bundle_all."
            )


        # 4) Scalars (including strings/bytes)
        if isinstance(results, STRINGISH) or not isinstance(results, ITERABLE_KINDS):
            if len(schema) == 1:
                return {schema[0]: results}
            raise PackagingError(
                f"{self.name}: scalar result cannot fit multi-key schema {schema}"
            )

        # 5) Set-like (unordered) — disallow positional use unless bundled
        if isinstance(results, Set):
            if self.bundle_all:
                if len(schema) != 1:
                    raise PackagingError(f"{self.name}: bundle_all requires single-key schema")
                # document: order not guaranteed; if you need order, pre-sort upstream
                return {schema[0]: list(results)}
            raise PackagingError(
                f"{self.name}: set-like results are unordered; bundle_all or pre-coerce to a sequence"
            )

        # 6) Sequence (ordered) — safe for positional mapping
        if isinstance(results, Sequence) and not isinstance(results, STRINGISH):
            seq = list(results)
            if self.bundle_all:
                if len(schema) != 1:
                    raise PackagingError(f"{self.name}: bundle_all requires single-key schema")
                return {schema[0]: seq}
            if len(seq) > len(schema):
                raise PackagingError(
                    f"{self.name}: too many positional items ({len(seq)}) for schema of length {len(schema)}"
                )
            padded = seq + [None] * (len(schema) - len(seq))
            return {k: v for k, v in zip(schema, padded)}

        # 7) Iterable fallback (e.g., generators) — consume deterministically
        #    Only safe for positional mapping; cap to schema length, then assert exhaustion.
        if isinstance(results, ITERABLE_KINDS):
            if self.bundle_all:
                if len(schema) != 1:
                    raise PackagingError(f"{self.name}: bundle_all requires single-key schema")
                return {schema[0]: list(results)}
            it = iter(results)
            collected = []
            for _ in range(len(schema)):
                try:
                    collected.append(next(it))
                except StopIteration:
                    break
            # ensure no overflow beyond schema
            try:
                next(it)
                raise PackagingError(
                    f"{self.name}: iterable produced more items than schema length {len(schema)}"
                )
            except StopIteration:
                pass
            collected += [None] * (len(schema) - len(collected))
            return {k: v for k, v in zip(schema, collected)}

        # 8) Final fallback
        if len(schema) == 1:
            return {schema[0]: results}
        raise PackagingError(
            f"{self.name}: cannot package type {type(results).__name__} into multi-key schema"
        )

    
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
        # Observability: log invocation start (debug-level payload)
        logger.debug("%s: invoke called with inputs=%s", self.name, Workflow._sanitize_for_json(inputs))

        # Forbid unexpected input keys; allow missing keys
        unexpected = set(inputs.keys()) - set(self._input_schema)
        if unexpected:
            logger.error("%s: unexpected input keys %s (allowed: %s)", self.name, unexpected, self._input_schema)
            raise ValidationError(f"{self.name}: Received unexpected input keys {unexpected}, not in input_schema {self._input_schema}")

        # Process and package (catch runtime errors to add context)
        start_ts = datetime.now()
        try:
            raw = self._process_inputs(inputs)
        except Exception as e:
            logger.exception("%s: _process_inputs raised an exception", self.name)
            raise ExecutionError(f"{self.name}: execution failed") from e
        result = self.package_results(raw)
        end_ts = datetime.now()
        duration_ms = (end_ts - start_ts).total_seconds() * 1000.0
        logger.info("%s: invoke completed in %.1fms", self.name, duration_ms)

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
        # Preserve Decimal precision by converting to string rather than float
        if isinstance(value, Decimal):
            logger.debug("%s: sanitizing Decimal value to string to avoid precision loss", type(value).__name__)
            return str(value)
        if isinstance(value, UUID): return str(value)
        # Represent exceptions in a structured, descriptive form
        if isinstance(value, Exception):
            return {
                "__exception__": type(value).__name__,
                "message": str(value),
            }

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
    One-line
    --------
    Workflow wrapper for an `Agent` that converts dict inputs into a single prompt string.

    Purpose
    -------
    Normalizes agent calls for composition: validates inputs, converts them to a prompt string,
    calls `agent.invoke(prompt: str)`, then packages the raw agent result to `output_schema`.

    Contract
    --------
    • Default schemas: `input_schema=["prompt"]` (configurable, non-empty), `output_schema=[WF_RESULT]`.
    • Inputs: subset of `input_schema`; unexpected keys → `ValidationError`.
    • Prompt formation:
      – If inputs contain exactly one key `["prompt"]`, pass `str(inputs["prompt"])` directly.
      – Otherwise, sanitize the entire `inputs` via `Workflow._sanitize_for_json`, then `json.dumps(...)`
        (UTF-8, no ASCII escaping) and pass that string to `agent.invoke(...)`.
    • Single agent call per `invoke`. Result is packaged by base `package_results`.
    • Checkpointing via base class.

    Key Parameters
    --------------
    • agent: `Agent` (stateful; provides `invoke(prompt: str) -> str`).
    • name: optional override; description derived from `agent.description`.
    • output_schema, bundle_all: standard Workflow setters/validation apply.

    Errors
    ------
    • `SchemaError` for empty `input_schema`.
    • `ValidationError` for unexpected input keys or empty inputs.
    • `PackagingError` if stringification/sanitization fails to serialize.
    • `ExecutionError` if `agent.invoke` raises.

    Example
    -------
    >>> af = AgentFlow(agent=my_agent)  # input_schema defaults to ["prompt"], name defaults to my_agent.name
    >>> af.invoke({"prompt": "Summarize this."})[WF_RESULT]
    """

    def __init__(self,
                 agent: Agent,
                 name: str = None,
                 input_schema: list[str] = ["prompt"],
                 output_schema: List[str] = [WF_RESULT],
                 bundle_all: bool = True,
    ):
        if input_schema == []:
            raise SchemaError("AgentFlow: Cannot initialize agent-flows with empty input schemas")
        super().__init__(
            name = name or agent.name,
            description = agent.description,
            input_schema = input_schema,
            output_schema = output_schema,
            bundle_all=bundle_all
        )
        self._agent = agent
        logger.debug("%s: AgentFlow initialized wrapping agent=%s input_schema=%s output_schema=%s bundle_all=%s",
                     self.name, getattr(agent, "name", None), self.input_schema, self.output_schema, self.bundle_all)

    # -------------------- Introspection --------------------
    @property
    def agent(self) -> Agent:
        """The wrapped Agent object."""
        return self._agent
    @agent.setter
    def agent(self, val: Agent) -> None:
        """Replaces agent"""
        self._agent = val
        self._description = self._agent.description

    # -------------------- Execution --------------------
    def clear_memory(self):
        """Clear workflow checkpoints and the wrapped agent's memory."""
        Workflow.clear_memory(self)
        try:
            self.agent.clear_memory()
        except Exception:
            logger.warning("%s: agent.clear_memory() raised an exception", self.name, exc_info=True)

    def _process_inputs(self, inputs: dict) -> str:
        """
        Execute the wrapped agent using JSON-readable prompt normalization.

        Behavior:
        - Return the agent.invoke(stringified inputs (or value at key 'prompt'))
        """
        # Reject empty inputs
        if not inputs:
            logger.error("%s: empty inputs provided to AgentFlow", self.name)
            raise ValidationError(f"{self.name}: AgentFlow requires a non-empty inputs dict (e.g., a 'prompt' key)")
        # reject
        if list(inputs.keys()) == ["prompt"]:
            return self.agent.invoke(str(inputs["prompt"]))
        sanitized = Workflow._sanitize_for_json(inputs)
        # stringify the sanitized prompt payload for agents that expect a string
        try:
            stringified_inputs = json.dumps(sanitized, ensure_ascii=False)
        except Exception as e:
            logger.exception("%s: failed to JSON-encode sanitized inputs for agent invocation", self.name)
            raise PackagingError(f"{self.name}: failed to JSON-encode inputs for agent invocation") from e

        logger.debug("%s: invoking agent %s with stringified inputs (len=%d)", self.name, getattr(self.agent, "name", None), len(stringified_inputs))
        try:
            return self.agent.invoke(stringified_inputs)
        except Exception as e:
            logger.exception("%s: agent.invoke raised an exception", self.name)
            # Wrap runtime errors to give callers a clear, domain-specific exception
            raise ExecutionError(f"{self.name}: agent invocation failed") from e


class ToolFlow(Workflow):
    """
    One-line
    --------
    Workflow wrapper for a single `Tool` with kwargs binding and schema-aware packaging.

    Purpose
    -------
    Adapts a Tool’s callable to the Workflow contract: validates `inputs`, calls the Tool once
    using dict→`**kwargs`, and packages the raw return to `output_schema`.

    Contract
    --------
    • `input_schema` is derived from `tool.signature_map` (ordered) excluding “*args”/“**kwargs”; read-only.
    • `output_schema` defaults to `[WF_RESULT]` (override allowed). `bundle_all` optional.
    • Inputs: keys MUST be a subset of `input_schema`; unexpected keys → `ValidationError`.
      Missing keys are allowed—Tool-level defaults apply.
    • Execution: `self._tool.invoke(**inputs)` exactly once per call.
    • Packaging obeys base rules (scalars/tuple/dict/hybrid/sequence/set/iterable handling).
    • Checkpointing via base class.

    Key Parameters
    --------------
    • tool: `Tool` (exposes `.invoke`, `.signature_map`, `.name`, `.description`).
    • name: optional override (defaults to tool.name).
    • output_schema, bundle_all: standard Workflow setters/validation apply.

    Errors
    ------
    • `ValidationError` for unexpected keys; arity/shape mismatches raise `PackagingError`.
    • `ExecutionError` if the Tool raises.

    Example
    -------
    >>> tf = ToolFlow(tool=my_tool)  # input_schema derived from signature_map
    >>> tf.invoke({"text": "hello"})[WF_RESULT]
    """
    def __init__(
        self,
        tool: Tool,
        name = None,
        output_schema: List[str] = [WF_RESULT],
        bundle_all: bool = True,
    ) -> None:
        # Filter out varargs/kwargs conventional placeholders if present
        input_schema: List[str] = [p for p in list(tool.signature_map.keys()) if p not in ("*args", "**kwargs")]

        # If base currently forbids empty, this will raise during super().__init__.
        super().__init__(
            name=name or tool.name,                         # base will store; we expose a property wrapper
            description=tool._description,           # immutable, derived below
            input_schema=input_schema,
            bundle_all=bundle_all,
            output_schema=output_schema,
        )
        # Store the tool reference                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        self._tool: Tool = tool
        logger.debug("%s: ToolFlow initialized wrapping tool=%s input_schema=%s output_schema=%s bundle_all=%s",
                     self.name, tool.name, self.input_schema, self.output_schema, self.bundle_all)

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
        keys = "\n".join([f"- {key}" for key in self.input_schema]) if self.input_schema else "\n(no parameters)"
        tool_desc = self.tool._description
        return template.format(tool_name=self.tool.name, keys=keys, description=tool_desc.strip()).strip()


    # ---------------------------------------------------------------------
    # Execution (template method hook)
    # ---------------------------------------------------------------------
    def _process_inputs(self, inputs: dict) -> Any:
        """
        Delegate to the underlying Tool using dict → **kwargs calling convention.
        Missing keys are allowed (Tool defaults apply). Unexpected keys are
        already rejected by the base `invoke`.
        """
        logger.debug("%s: invoking tool %s", self.name, self.tool.name)
        try:
            return self._tool.invoke(**inputs)
        except Exception as e:
            logger.exception("%s: tool.invoke raised an exception", self.name)
            # Wrap runtime errors for callers
            raise ExecutionError(f"{self.name}: tool invocation failed") from e

    # ---------------------------------------------------------------------
    # Memory
    # ---------------------------------------------------------------------
    def clear_memory(self) -> None:
        """
        Clear checkpoints and cascade to the Tool if it exposes a clear_memory hook.
        """
        super().clear_memory()
        try:
            self._tool.clear_memory()
        except Exception:
            # Clearing memory is best-effort; log and continue
            logger.warning("%s: tool.clear_memory() raised an exception during clear_memory", self.name, exc_info=True)

# wraps all incoming objects into workflow classes and adjusts their input/output schemas based on optional parameters
def _to_workflow(obj: Agent | Tool | Workflow, in_sch:list[str]|None = None, out_sch:list[str]|None = None) -> Workflow:
        if isinstance(obj, Workflow):
            obj.output_schema = out_sch or obj.output_schema
            return obj
        if isinstance(obj, Agent): return AgentFlow(obj, input_schema=in_sch or ["prompt"], output_schema=out_sch or [WF_RESULT])
        if isinstance(obj, Tool): return ToolFlow(obj, output_schema=out_sch or [WF_RESULT])
        raise ValidationError(f"Object must be Agent, Tool, or Workflow. Got unexpected '{type(obj).__name__}'.")


class ChainFlow(Workflow):
    """
    One-line
    --------
    Linear, schema-checked pipeline that invokes steps in order and hands each step’s packaged output to the next.

    Purpose
    -------
    Compose heterogeneous steps (Workflow/ToolFlow/AgentFlow) into a deterministic chain. The chain exposes its own
    `output_schema` as an overlay for the chain boundary without mutating the final child’s internal schema.

    Contract
    --------
    • Steps are invoked sequentially; step N+1 receives the PACKAGED dict returned by step N’s `invoke()`.
    • When empty, ChainFlow acts as an identity flow: `input_schema` mirrors `output_schema`.
    • When non-empty, `input_schema` mirrors the first child; do not set it directly.
    • Chain-level `output_schema` is an overlay applied to the final child’s raw result by base `package_results`.
    • If ChainFlow `output_schema` has length > 1, `bundle_all` is forced False for the chain overlay.
    • Upstream→downstream handoff:
      – If upstream `bundle_all=True`: downstream must accept the single-key envelope.
      – If upstream `bundle_all=False`: len(upstream.output) MUST equal len(downstream.input_schema), or raise.
      – No implicit hybrid key remapping between children; insert explicit adapters if keys differ.

    Key Parameters
    --------------
    • steps: list[Workflow|Tool|Agent]; wrapped to Workflows; set via `steps`, `add_step`, `pop`.
    • output_schema / bundle_all: overlay for the chain boundary; reconciled independently of children.

    Inputs & Outputs
    ----------------
    • Inputs: must match first child’s `input_schema`.
    • Outputs: the chain’s final return is the base-packaged dict using the chain overlay (`output_schema`, `bundle_all`)
      applied to the final child’s raw result.

    Error Conditions
    ----------------
    • `SchemaError`/`ValidationError`: step adjacency mismatch during reconciliation; invalid overlays.
    • `ExecutionError`: child failure; original exception chained.

    Performance/Concurrency
    -----------------------
    Sequential by design; ordering is stable and deterministic.

    Example
    -------
    >>> cf = ChainFlow(name="C").add_step(t1).add_step(t2)
    >>> cf.output_schema = ["answer"]
    >>> cf.invoke({"text": "hello"})["answer"]

    See Also
    --------
    Workflow, ToolFlow, AgentFlow
    """


    def __init__(
        self,
        name: str,
        description: str,
        steps: List[Agent | Tool | Workflow] | None = None,
        output_schema: List[str]  = [WF_RESULT],
        bundle_all = True
    ):
        # build steps
        self._steps: list[Workflow] = []
        if steps:
            self._steps = [_to_workflow(s) for s in steps]
        logger.debug("%s: ChainFlow initialized with %d step(s): %s", name, len(self._steps), [s.name for s in self._steps])
        super().__init__(
            name=name,
            description=description,
            input_schema=output_schema,
            output_schema=output_schema,
            bundle_all= bundle_all,
        )
        # Reserve for when no steps are present
        self._reconcile_all()
        self._mirror_endpoints_from_children()

    # -------------------- Steps management --------------------

    @property
    def steps(self) -> list[Workflow]:
        return list(self._steps)

    @steps.setter
    def steps(self, value: list[Agent | Tool | Workflow] | None):
        value = value or []
        self._steps = [_to_workflow(s) for s in value]
        if not self._steps:
            # Keep user-configurable input/output overlays; they remain mirrored by setters.
            logger.debug("%s: steps set to empty; ChainFlow reverting to default input schema=%s", self.name, self._output_schema)
        self._reconcile_all()
        self._mirror_endpoints_from_children()

    def add_step(self, step: Agent | Tool | Workflow, position: int | None = None) -> None:
        wrapped = _to_workflow(step)
        if position is None:
            self._steps.append(wrapped)
        else:
            if position < 0 or position > len(self._steps):
                raise IndexError(f"{self.name}: add_step position {position} out of range")
            self._steps.insert(position, wrapped)
        logger.debug("%s: added step '%s' at position=%s", self.name, wrapped.name, str(position))
        self._reconcile_all()
        self._mirror_endpoints_from_children()

    def pop(self, index: int = -1) -> Workflow:
        removed = self._steps.pop(index)
        if not self._steps:
            self._bundle_all = False  # empty state invariant
            return removed
        self._reconcile_all()
        self._mirror_endpoints_from_children()
        logger.debug("%s: popped step '%s' (index=%d); %d steps remain", self.name, removed.name, index, len(self._steps))
        return removed

    # -------------------- Execution --------------------
    def _process_inputs(self, inputs: dict) -> dict:
        if not self._steps:
            self._mirror_endpoints_from_children()
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
            # If none of the above are valid, then raise a domain-specific validation error
            logger.error("%s: reconcilation failed between %s and %s: out=%s in=%s", self.name, A.name, B.name, a_out, b_in)
            raise ValidationError(
                    f"{self.name}: length mismatch {A.name} → {B.name}: "
                    f"len(out)={len(a_out)} != len(in)={len(b_in)} (adapter required)"
                )

    def _mirror_endpoints_from_children(self) -> None:
        # if no steps are present, then chainflow becomes an identity flow
        if not self._steps:
            self._input_schema = self.output_schema
            logger.debug("%s: mirrored endpoints to default input schema=%s (no steps)", self.name, self._output_schema)
            return
        first = self._steps[0]
        self._input_schema  = list(first.input_schema)
        logger.debug("%s: mirrored endpoints from children: input_schema=%s output_schema=%s bundle_all=%s", self.name, self._input_schema, self._output_schema, self._bundle_all)


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