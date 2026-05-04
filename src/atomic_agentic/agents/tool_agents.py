"""
ToolAgents: LLM-Driven Iterative Tool Calling with Persistent Blackboard Memory

This module provides an extensible framework for building intelligent agents that
use **Large Language Models (LLMs)** to decide which tools to invoke, observe
results, and either plan or react accordingly.

Core Concept
------------
Rather than executing a fixed sequence of operations, ToolAgents maintain an
interactive execution loop:

1. **LLM decides**: The LLM examines the current state and decides which tools to invoke
2. **Tools execute**: Selected tools run and produce results
3. **Run state updates**: Results are stored in the invocation's running blackboard
4. **Loop continues**: The LLM observes results and decides next steps, or terminates
5. **Memory persists**: If `context_enabled=True`, completed tool slots are merged into
   the persisted blackboard and the completed invocation is stored as a ToolAgentTurn

The canonical memory model separates storage from rendering:

- Agent memory is stored as turn objects (`AgentTurn` / `ToolAgentTurn`)
- Tool execution results are stored as blackboard slots
- A ToolAgentTurn stores the half-open blackboard span produced by one invocation
- Future LLM-facing messages are rendered from turns and their associated blackboard spans

Execution Persistence
---------------------
Tool invocations are tracked in an execution blackboard:

- Each step records: **tool name**, **arguments** (possibly containing placeholders),
  **resolved arguments**, and **execution result** (or error)
- If `context_enabled=True`, the blackboard is persisted between invoke() calls,
  allowing new runs to reference prior results
- LLM-facing message history is rendered from stored turns rather than stored as the
  canonical memory format

Blackboard Architecture
~~~~~~~~~~~~~~~~~~~~~~~
The **blackboard pattern** is used internally to store and manage tool execution state:

- **Running blackboard**: Current invocation's tool calls (ephemeral, local to this run)
- **Cached blackboard**: Prior invocation results persisted from previous runs
- **Placeholders**: Tool arguments can reference results from:

  - ``<<__sN__>>`` – result from running step N (current invoke, 0-based index)
  - ``<<__cN__>>`` – result from cache entry N (prior invokes, 0-based index)

Placeholders are resolved at execution time to their concrete values, enabling dynamic
data flow and automatic dependency management.

Intelligent Iteration Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Subclasses implement different iteration approaches:

- **PlanActAgent**: LLM generates a complete plan upfront, then the system executes it
  in topologically-sorted concurrent batches.
- **ReActAgent**: LLM emits one tool call per turn, observes the current-run result,
  then decides the next step.

Execution Model
~~~~~~~~~~~~~~~
**Template-Method Pattern**: ``ToolAgent`` owns the invariant iteration loop; subclasses
provide domain-specific planning/iteration logic via abstract hooks.

**Concurrent Execution**: Batches of independent tool calls execute concurrently with
gather-based error handling.

**Termination**: Agents invoke the canonical ``return`` tool to signal completion and
return a final value.

Subclass Responsibilities
-------------------------
Subclasses must implement two abstract methods:

**_initialize_run_state(messages)** → ``RS``
  Initialize and snapshot the execution state for this invoke:
  - Copy the incoming LLM-facing messages into run-local state
  - Snapshot prior cached results if context is enabled
  - Allocate running blackboard slots for current-run tool calls

**_prepare_next_batch(state)** → ``RS``
  Prepare the next executable batch:
  - Decide which tools to invoke based on current state
  - Validate tool names, dependencies, and placeholders
  - Resolve placeholders into concrete arguments
  - Populate `state.prepared_steps`

The run state is extensible: ``RS`` is a TypeVar bound to ``ToolAgentRunState``,
allowing subclasses to carry domain-specific fields such as batches, cursors, or
planning metadata.

Concrete Subclasses
-------------------
- **PlanActAgent**: One-shot planner; queries LLM once to generate an entire plan,
  then executes in concurrent batches. Fast, deterministic, no replanning.
- **ReActAgent**: Iterative actor; queries LLM once per step, reacts to each result.
  Fully adaptive, but requires more LLM turns and sequential execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from dataclasses import asdict, dataclass, field
import logging
import re
import string
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Mapping, Optional, Sequence, List, TypeVar, Iterable
import pprint

from .base import Agent
from .data_classes import AgentTurn, ToolAgentTurn, BlackboardSlot
from ..core.Exceptions import (
    ToolAgentError,
    ToolDefinitionError,
    ToolInvocationError,
    ToolRegistrationError,
)
from ..core.Invokable import AtomicInvokable
from ..core.sentinels import NO_VAL
from ..engines.LLMEngines import LLMEngine
from ..tools import Tool, toolify, batch_toolify
from ..core.Prompts import PLANNER_PROMPT, ORCHESTRATOR_PROMPT
from ..core.Exceptions import ToolAgentError
from ..mcp import MCPClientHub
from ..a2a import PyA2AtomicClient


logger = logging.getLogger(__name__)

# Canonical placeholders:
#   <<__si__>>  : result of plan-local step i (0-based within the running plan)
#   <<__ci__>> : result of cache entry i (0-based within persisted cache)
_STEP_TOKEN: re.Pattern[str] = re.compile(r"<<__s(\d+)__>>")
_CACHE_TOKEN: re.Pattern[str] = re.compile(r"<<__c(\d+)__>>")

def extract_dependencies(obj: Any, placeholder_pattern: re.Pattern[str]) -> set[int]:
    """
    Recursively extract all placeholder references from an object.

    Scans the object for occurrences of a given placeholder pattern (e.g., ``<<__sN__>>``)
    and returns the set of all referenced indices. Used during planning to extract
    dependencies between steps.

    Parameters
    ----------
    obj : Any
        Object to scan. Typically a dict (tool args) but can be any nested structure
        (lists, tuples, dicts, sets, scalars).
    placeholder_pattern : re.Pattern[str]
        Compiled regex pattern matching placeholders. Usually:
        - ``_STEP_TOKEN`` for step refs (``<<__sN__>>``)
        - ``_CACHE_TOKEN`` for cache refs (``<<__cN__>>``)

    Returns
    -------
    set[int]
        Set of all indices found (0-based). Empty set if no placeholders found.

    Validation
    ~~~~~~~~~~
    This method performs **NO validation** of the found indices:
    - Does NOT check bounds (N might be >= blackboard length)
    - Does NOT check execution status (referenced slot might not be executed yet)
    - Purely structural scanning

    Validation happens later in ``_resolve_placeholders()`` at prepare time.

    Examples
    --------
    >>> pattern = _STEP_TOKEN  # Matches <<__sN__>>
    >>> obj = {\"query\": \"<<__s0__>>\", \"context\": [\"<<__s1__>>\", \"<<__s0__>>\"]}
    >>> extract_dependencies(obj, pattern)
    {0, 1}

    >>> obj = {\"static\": \"no placeholders here\"}
    >>> extract_dependencies(obj, pattern)
    set()
    """
    deps: set[int] = set()

    def walk(x: Any) -> None:
        if isinstance(x, str):
            for m in placeholder_pattern.finditer(x):
                deps.add(int(m.group(1)))
            return
        if isinstance(x, dict):
            for k, v in x.items():
                walk(k)
                walk(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)
            return

    walk(obj)
    return deps

# --------------------------------------------------------------------------- #
# Canonical final-return tool (shared common resource)
# --------------------------------------------------------------------------- #
def _return(val: Any) -> Any:
    return val


return_tool = Tool(
    function=_return,
    name="return",
    namespace="ToolAgents",
    description=(
        "Returns the passed-in value. Tool agents should use this to signal completion."
    ),
)


# --------------------------------------------------------------------------- #
# Support classes / types
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ToolAgentRunState:
    """
    Base run state contract for ``ToolAgent`` invocations.

    This dataclass encapsulates all mutable state during a single ``invoke()`` call,
    enabling subclasses to extend it with domain-specific planning/execution fields.

    Blackboard Architecture
    -----------------------
    **cache_blackboard** : list[BlackboardSlot]
        Snapshot of the persisted ``self._blackboard`` at invoke start when context is
        enabled. Previous invocation results are available here and can be referenced
        via ``<<__cN__>>`` placeholders.

    **running_blackboard** : list[BlackboardSlot]
        Plan-local slots (0-based indices) created during this invoke. Populated by
        ``_initialize_run_state()``, planned by ``_prepare_next_batch()``, and
        executed by ``_execute_prepared_batch()``.
        If ``context_enabled=True``, executed slots are persisted and merged into
        ``self._blackboard`` after ``invoke()`` completes.

    Placeholder Semantics & Resolvability
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    **Cached Placeholder** (``<<__cN__>>``)
        Resolvable iff ``0 <= N < len(cache_blackboard)`` AND
        ``cache_blackboard[N].is_executed() == True``.

    **Step Placeholder** (``<<__sN__>>``)
        Resolvable iff ``0 <= N < len(running_blackboard)`` AND
        ``running_blackboard[N].is_executed() == True``.

    Execution State Machine
    -----------------------
    **messages** : list[dict[str, str]]
        Run-local LLM-facing messages used during this invocation. ReAct may augment
        this list with current-run observations. These messages are not the canonical
        stored memory format; completed invocations are stored as turns.

    **executed_steps** : set[int]
        Running plan indices that have been executed.

    **prepared_steps** : list[int]
        Running plan indices ready for execution in the next batch. Must be set by
        ``_prepare_next_batch()`` and consumed by the base execution path.

    **tool_calls_used** : int
        Count of non-return tool calls executed so far.

    **is_done** : bool
        Loop termination flag. Set to True when the ``return`` tool executes.

    **return_value** : Any | NO_VAL
        Agent's raw final output. Set when the return tool executes. This becomes the
        raw response passed into the base Agent post-processing and turn pipeline.
    """
    messages: list[dict[str, str]]

    cache_blackboard: list[BlackboardSlot]
    running_blackboard: list[BlackboardSlot]

    # Plan-local indices whose results are available (optional fast-path bookkeeping).
    executed_steps: set[int] = field(default_factory=set)

    # Exact plan-local indices to execute next; must be set by _prepare_next_batch.
    prepared_steps: list[int] = field(default_factory=list)

    # Bookkeeping:
    tool_calls_used: int = 0  # non-return calls executed in this run

    # Completion:
    is_done: bool = False
    return_value: Any = NO_VAL  # NO_VAL by default; set once return tool executes

    # Run metadata:
    run_metadata: dict[str, Any] = field(default_factory=dict)

RS = TypeVar("RS", bound=ToolAgentRunState)


# --------------------------------------------------------------------------- #
# Base ToolAgent
# --------------------------------------------------------------------------- #
class ToolAgent(Agent, ABC, Generic[RS]):
    """
    Abstract base class implementing the template-method pattern for tool-using agents.

    This class owns the invariant iteration loop; subclasses provide domain-specific
    planning and batch preparation strategies. The architecture uses a blackboard slot
    system with sentinel-driven state and placeholder-based dependency management.

    Template-Method Loop
    --------------------
    The ``_invoke(messages=...)`` method (FINAL; do not override) orchestrates::

    1. state = _initialize_run_state(messages=messages)  [subclass hook]
    2. while not state.is_done:
        state = _prepare_next_batch(state)             [subclass hook]
        state = _execute_prepared_batch(state)         [base implementation]
        [completion check: if return tool executed, is_done=True]
    3. if context_enabled:
        blackboard_start = len(self._blackboard)
        state = update_blackboard(state)
        blackboard_end = len(self._blackboard)
    4. return state.return_value, {"blackboard_start": ..., "blackboard_end": ...}

    The returned metadata is consumed by ``_make_turn(...)`` to construct a
    ``ToolAgentTurn``. The turn stores the half-open blackboard span produced by the
    invocation. Future LLM-facing messages are rendered by ``render_turn(...)``.

    Subclass Responsibilities
    -------------------------
    Subclasses must implement two abstract methods:

    **_initialize_run_state(messages)** → ``RS`` (TypeVar[ToolAgentRunState])
        Initialize and return a run state for this invoke. Must:
        - Copy incoming LLM-facing messages into run-local state
        - Snapshot cached blackboard entries if context is enabled
        - Create an appropriate running blackboard
        - Initialize ``executed_steps``, ``prepared_steps``, and completion state

    **_prepare_next_batch(state)** → ``RS``
        Prepare exactly one executable batch per loop iteration:
        - Generate next tool calls via LLM, precomputed plan, or another strategy
        - Validate tool names, placeholder dependencies, and budget
        - Resolve placeholders with ``self._resolve_placeholders(...)``
        - Fill ``state.prepared_steps`` with indices ready to execute
        - Return the updated state

    Key Features
    ~~~~~~~~~~~~
    **Concurrent Execution**: Each prepared batch runs through async tool invocation and
    gather-based result collection.

    **Placeholder Resolution**: Supported syntaxes:
        - ``<<__sN__>>`` – reference to running step N
        - ``<<__cN__>>`` – reference to cache entry N
        Full-string placeholders preserve types; inline placeholders render via ``repr()``.

    **Return Semantics**: The canonical ``return_tool`` is registered automatically.
        When return executes, ``state.return_value`` is set and the loop exits.

    **Budget Enforcement**: If ``tool_calls_limit`` is set, non-return tool calls are
        tracked and exceeding the limit raises.

    **Context Persistence**: If ``context_enabled=True``, the completed run blackboard
        is merged into ``self._blackboard`` and the produced span is stored on the
        ToolAgentTurn for future rendering.

    Generic Type Parameter
    ~~~~~~~~~~~~~~~~~~~~~~
    ``RS`` is a TypeVar bound to ``ToolAgentRunState``. Subclasses provide a concrete
    runtime-specific state class such as ``PlanActRunState`` or ``ReActRunState``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: str,
        filter_extraneous_inputs: Optional[bool] = None,
        context_enabled: bool = False,
        *,
        tool_calls_limit: Optional[int] = None,
        peek_at_cache: bool = False,
        response_preview_limit: Optional[int] = None,
        blackboard_preview_limit: Optional[int] = None,
        preview_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        history_window: Optional[int] = None,
    ) -> None:
        template = self._validate_role_prompt_template(role_prompt)

        if preview_limit is not None:
            if response_preview_limit is not None:
                raise ToolAgentError(
                    "preview_limit and response_preview_limit cannot both be provided; "
                    "preview_limit is a compatibility alias for response_preview_limit."
                )
            response_preview_limit = preview_limit

        super().__init__(
            name=name,
            description=description,
            llm_engine=llm_engine,
            filter_extraneous_inputs=filter_extraneous_inputs,
            role_prompt=template,
            context_enabled=context_enabled,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            history_window=history_window,
            response_preview_limit=response_preview_limit,
        )

        self._toolbox: dict[str, Tool] = {}
        self._blackboard: list[BlackboardSlot] = []

        self._peek_at_cache = False
        self.peek_at_cache = peek_at_cache

        self.blackboard_preview_limit = blackboard_preview_limit

        self._tool_calls_limit: Optional[int] = None
        self.tool_calls_limit = tool_calls_limit

        # Always include canonical return tool (avoid collisions by skipping).
        self.register(return_tool, name_collision_mode="skip")

    # ------------------------------------------------------------------ #
    # Agent Properties
    # ------------------------------------------------------------------ #
    @property
    def role_prompt(self) -> str:
        """
        ToolAgent role prompt is a template requiring:
          - {TOOLS}
          - {TOOL_CALLS_LIMIT}
        """
        template = self._role_prompt
        limit_text = "unlimited" if self._tool_calls_limit is None else str(self._tool_calls_limit)
        try:
            return template.format(
                TOOLS=self.actions_context(),
                TOOL_CALLS_LIMIT=limit_text,
            )
        except Exception as exc:  # pragma: no cover
            raise ToolAgentError(f"Failed to format ToolAgent role_prompt template: {exc}") from exc

    # ------------------------------------------------------------------ #
    # ToolAgent Properties
    # ------------------------------------------------------------------ #
    @property
    def tool_calls_limit(self) -> Optional[int]:
        """Max allowed non-return tool calls per invoke() run. None means unlimited."""
        return self._tool_calls_limit

    @tool_calls_limit.setter
    def tool_calls_limit(self, value: Optional[int]) -> None:
        if value is None:
            self._tool_calls_limit = None
            return
        if type(value) is not int or value < 0:
            raise ToolAgentError("tool_calls_limit must be None or an int >= 0.")
        self._tool_calls_limit = value

    def blackboard_serialized(self, peek: bool = False) -> list[dict[str, Any]]:
        """
        Read-only serialized view of the persisted blackboard.
        (Keeps backward-friendly dict shape.)
        """
        result = []
        if peek:
            result = [slot.to_dict() for slot in self._blackboard]
        else:
            dicts = [slot.to_dict() for slot in self._blackboard]
            for _dict in dicts:
                _dict.pop("resolved_args")
                _dict.pop("result")
                result.append(_dict)
            
        return result
    
    @property
    def blackboard(self) -> list[BlackboardSlot]:
        return [slot.copy() for slot in self._blackboard]
    
    @property
    def peek_at_cache(self) -> bool:
        return self._peek_at_cache

    @peek_at_cache.setter
    def peek_at_cache(self, val: bool) -> None:
        if type(val) is not bool:
            raise ToolAgentError("peek_at_cache must be a boolean.")
        self._peek_at_cache = val
    
    @property
    def blackboard_preview_limit(self) -> Optional[int]:
        """Character limit for cached blackboard result previews. None means no truncation."""
        return self._blackboard_preview_limit
    
    @blackboard_preview_limit.setter
    def blackboard_preview_limit(self, value: Optional[int]) -> None:
        if value is None:
            self._blackboard_preview_limit = None
            return
        if type(value) is not int or value <= 0:
            raise ToolAgentError("blackboard_preview_limit must be None or a positive integer > 0.")
        self._blackboard_preview_limit = value

    @property
    def preview_limit(self) -> Optional[int]:
        """Compatibility alias for response_preview_limit."""
        return self.response_preview_limit
    
    @preview_limit.setter
    def preview_limit(self, value: Optional[int]) -> None:
        self.response_preview_limit = value

    def _preview_blackboard_result(self, result: Any) -> str:
        """Render and optionally truncate a cached blackboard result preview."""
        try:
            text = repr(result)
        except Exception:
            text = str(result)

        if (
            self._blackboard_preview_limit is not None
            and len(text) > self._blackboard_preview_limit
        ):
            text = text[: self._blackboard_preview_limit] + "..."

        return text

    # ------------------------------------------------------------------ #
    # Memory management
    # ------------------------------------------------------------------ #
    def clear_memory(self):
        super().clear_memory()
        self._blackboard.clear()
    # ------------------------------------------------------------------ #
    # Prompt Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_role_prompt_template(template: Any) -> str:
        """
        ToolAgent requires a non-empty role prompt template containing:
          - {TOOLS}
          - {TOOL_CALLS_LIMIT}
        Additional simple named format fields are allowed.
        """
        if not isinstance(template, str):
            raise ToolAgentError(
                f"ToolAgent role_prompt must be a non-empty str template; got {type(template).__name__!r}."
            )

        cleaned = template.strip()
        if not cleaned:
            raise ToolAgentError("ToolAgent role_prompt template cannot be empty.")

        fmt = string.Formatter()
        fields: set[str] = set()

        for _literal, field_name, _format_spec, _conversion in fmt.parse(cleaned):
            if field_name is None:
                continue
            if field_name == "":
                raise ToolAgentError(
                    "ToolAgent role_prompt template may not use positional fields '{}'. "
                    "Use named placeholders like {TOOLS} and {TOOL_CALLS_LIMIT}."
                )
            if any(ch in field_name for ch in ".[]"):
                raise ToolAgentError(
                    f"ToolAgent role_prompt template contains unsupported field expression {{{field_name}}}. "
                    "Only simple named placeholders are supported."
                )
            fields.add(field_name)

        required = {"TOOLS", "TOOL_CALLS_LIMIT"}
        missing = required - fields
        if missing:
            raise ToolAgentError(
                f"ToolAgent role_prompt template missing required placeholder(s): {', '.join(sorted(missing))}."
            )

        return cleaned

    # ------------------------------------------------------------------ #
    # Toolbox Helpers
    # ------------------------------------------------------------------ #
    def actions_context(self) -> str:
        """String representation of all tools in the toolbox for prompt injection."""
        tools = list(self._toolbox.values())
        return "\n".join(f"-- {t}" for t in tools)

    def list_tools(self) -> dict[str, Tool]:
        return dict(self._toolbox)

    def has_tool(self, tool_full_name: str) -> bool:
        return tool_full_name in self._toolbox

    def get_tool(self, tool_full_name: str) -> Tool:
        tool = self._toolbox.get(tool_full_name)
        if tool is None:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool {tool_full_name!r}.")
        return tool

    def remove_tool(self, tool_full_name: str) -> bool:
        return self._toolbox.pop(tool_full_name, None) is not None

    def clear_tools(self) -> None:
        self._toolbox.clear()

    def register(
        self,
        component: AtomicInvokable | Callable | MCPClientHub | PyA2AtomicClient,
        name: Optional[str] = None,
        description: Optional[str] = None,
        namespace: Optional[str] = None,
        *,
        remote_name: Optional[str] = None,
        filter_extraneous_inputs: Optional[bool] = None,
        name_collision_mode: str = "raise",  # raise|skip|replace
    ) -> str:
        # Validate collision policy
        if name_collision_mode not in ("raise", "skip", "replace"):
            raise ToolRegistrationError(
                "name_collision_mode must be one of: 'raise', 'skip', 'replace'."
            )

        # Normalize via toolify (single source of truth)
        try:
            tool = toolify(
                component=component,
                name=name,
                description=description,
                namespace=namespace,
                filter_extraneous_inputs=filter_extraneous_inputs,
                remote_name=remote_name,
            )
        except Exception as e:
            raise ToolRegistrationError(
                f"{type(self).__name__}.{self.name}: failed to toolify component: {e}"
            ) from e

        # Collision handling
        key = tool.full_name

        if key in self._toolbox:
            if name_collision_mode == "raise":
                raise ToolRegistrationError(
                    f"{type(self).__name__}.{self.name}: tool already registered: {key}"
                )
            if name_collision_mode == "skip":
                return key
            # replace → fall through

        self._toolbox[key] = tool
        return key

    def batch_register(
        self,
        sources: List[AtomicInvokable | Callable | MCPClientHub | PyA2AtomicClient],
        *,
        name_collision_mode: str = "raise",
        batch_filter_inputs: Optional[bool] = None,
        batch_namespace: Optional[str] = None,
    ) -> list[str]:
        if name_collision_mode not in ("raise", "skip", "replace"):
            raise ToolRegistrationError(
                "name_collision_mode must be one of: 'raise', 'skip', 'replace'."
            )

        if not sources:
            raise ValueError("ToolAgents.batch_register() expects a non-empty list of callables, AtomicInvokables, or MCPClientHubs")
        try:
            tool_list = batch_toolify(
                sources=list(sources),
                batch_namespace=batch_namespace,
                batch_filter_inputs=batch_filter_inputs,
            )
        except ToolDefinitionError:
            raise
        except Exception as exc:  # pragma: no cover
            raise ToolRegistrationError(f"batch_toolify failed: {exc}") from exc

        registered: list[str] = []
        for tool in tool_list:
            key = tool.full_name
            if key in self._toolbox:
                if name_collision_mode == "raise":
                    raise ToolRegistrationError(
                        f"{type(self).__name__}.{self.name}: tool already registered: {key}"
                    )
                if name_collision_mode == "skip":
                    continue

            self._toolbox[key] = tool
            registered.append(key)

        return registered

    # ------------------------------------------------------------------ #
    # Placeholder resolution helpers (prepare-time)
    # ------------------------------------------------------------------ #
    def _resolve_placeholders(self, obj: Any, *, state: ToolAgentRunState) -> Any:
        """
        Resolve all placeholders in an object to their concrete values.

        This method recursively traverses the object structure and replaces placeholder
        references with their concrete results. Placeholders are references to previously
        executed steps (``<<__sN__>>``) or cached prior results (``<<__cN__>>``).

        Supported Placeholder Formats
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        - ``<<__sN__>>`` – Result from running step N (0-based, plan-local to this invoke)
        - ``<<__cN__>>`` – Result from cache entry N (0-based, from persisted blackboard)

        Two resolution modes apply depending on placeholder position:

        1. **Full-String Placeholder** (e.g., entire value is ``\"<<__s0__>>\"``):
           - Returns the referenced result as-is, **preserving its type**
           - Example: if ref is ``[1, 2, 3]``, returns the list, not ``\"[1, 2, 3]\"``

        2. **Inline Placeholder** (e.g., within ``\"prefix <<__s0__>> suffix\"``):
           - Replaces placeholder with ``repr(result)``, then returns the string
           - Fallback to ``str(result)`` if ``repr()`` fails
           - Example: if ref is ``[1, 2, 3]``, inline becomes ``\"prefix [1, 2, 3] suffix\"``

        Readiness Validation
        ~~~~~~~~~~~~~~~~~~~~
        Before resolution, validates that all referenced slots are marked **executed**:

        - Referenced slot must exist (index within bounds)
        - Referenced slot must satisfy ``is_executed()``

        Raises ``ToolAgentError`` if any reference is invalid or unexecuted.

        Parameters
        ----------
        obj : Any
            Object to resolve (typically a dict of args). Can be nested (lists, tuples,
            dicts, sets, or scalars).
        state : ToolAgentRunState
            Execution state containing cache_blackboard and running_blackboard.

        Returns
        -------
        Any
            Resolved object with all placeholders replaced. Structure is preserved;
            only placeholder tokens are replaced.

        Raises
        ------
        ToolAgentError
            If a referenced placeholder is out of bounds or not executed.

        Examples
        --------
        >>> # Full-string placeholder (preserves type)
        >>> obj = {\"result\": \"<<__s0__>>\"}
        >>> # If step 0 result is [1, 2, 3]:
        >>> resolved[\"result\"]  # → [1, 2, 3] (list, not string)

        >>> # Inline placeholder (coerced to string)
        >>> obj = {\"message\": \"Step 0 returned: <<__s0__>>\"}
        >>> # If step 0 result is [1, 2, 3]:
        >>> resolved[\"message\"]  # → \"Step 0 returned: [1, 2, 3]\" (string)
        """
        cache = state.cache_blackboard
        running = state.running_blackboard

        needed_cache: set[int] = set(extract_dependencies(obj, placeholder_pattern=_CACHE_TOKEN))
        needed_steps: set[int] = set(extract_dependencies(obj, placeholder_pattern=_STEP_TOKEN))

        # ----------------------------
        # 2) Validate readiness.
        # ----------------------------
        for idx in sorted(needed_cache):
            if idx < 0 or idx >= len(cache):
                raise ToolAgentError(
                    f"Cache reference {idx} out of range (cache length={len(cache)})."
                )
            if not cache[idx].is_executed():
                raise ToolAgentError(f"Referenced cache {idx} is not yet executed.")

        for idx in sorted(needed_steps):
            if idx < 0 or idx >= len(running):
                raise ToolAgentError(
                    f"Step reference {idx} out of range (running plan length={len(running)})."
                )
            if not running[idx].is_executed():
                raise ToolAgentError(f"Referenced step {idx} is not executed.")

        # ----------------------------
        # 3) Resolve recursively.
        # ----------------------------
        def resolve_str(s: str) -> Any:
            # Exact placeholder -> preserve type
            m_cache = _CACHE_TOKEN.fullmatch(s)
            if m_cache:
                return cache[int(m_cache.group(1))].result

            m_step = _STEP_TOKEN.fullmatch(s)
            if m_step:
                return running[int(m_step.group(1))].result

            # Inline substitution
            def repl_cache(m: re.Match[str]) -> str:
                idx = int(m.group(1))
                val = cache[idx].result
                try:
                    return repr(val)
                except Exception:
                    return str(val)

            def repl_step(m: re.Match[str]) -> str:
                idx = int(m.group(1))
                val = running[idx].result
                try:
                    return repr(val)
                except Exception:
                    return str(val)

            out = _CACHE_TOKEN.sub(repl_cache, s)
            out = _STEP_TOKEN.sub(repl_step, out)
            return out

        def resolve(x: Any) -> Any:
            if isinstance(x, str):
                return resolve_str(x)
            if isinstance(x, list):
                return [resolve(v) for v in x]
            if isinstance(x, tuple):
                return tuple(resolve(v) for v in x)
            if isinstance(x, set):
                return {resolve(v) for v in x}
            if isinstance(x, dict):
                return {resolve(k): resolve(v) for k, v in x.items()}
            return x

        return resolve(obj)

    # ------------------------------------------------------------------ #
    # Execution (base-owned)
    # ------------------------------------------------------------------ #
    def _execute_prepared_batch(self, state: RS) -> RS:
        """
        Execute all steps in the currently prepared batch concurrently.

        This is a core base-owned method (do not override). It executes all steps in
        ``state.prepared_steps`` using the tool async-invoke path, records results in
        the running blackboard, and handles termination if the return tool is executed.

        Batch Semantics
        ~~~~~~~~~~~~~~~
        - All steps in ``prepared_steps`` are **concurrent**
        - Multi-step batches use ``asyncio.gather(..., return_exceptions=True)``
        under a single ``asyncio.run(...)``
        - This version favors compactness over strict fail-fast cancellation
        - **Ordering**: Results are stored in the blackboard; order is immaterial

        Validation & Safety Checks
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        Before execution, validates:

        1. **prepared_steps is non-empty**: Raises ToolAgentError if empty
        2. **No duplicates**: Raises if any index appears twice
        3. **Bounds**: Each index must be 0 <= idx < len(running_blackboard)
        4. **Not already executed**: Raises if slot already has result set
        5. **Is prepared**: Slot must be prepared for execution
        6. **Tool exists**: Tool name must be registered in toolbox
        7. **Budget enforcement**: Non-return calls don't exceed tool_calls_limit

        Execution Flow
        ~~~~~~~~~~~~~~
        1. Validate all preconditions (as above)
        2. Count non-return vs. return tool calls
        3. If tool_calls_limit set, check budget
        4. Execute concurrently via ``asyncio.gather(..., return_exceptions=True)``
        5. If any gathered result is an exception, identify the first such step,
        store the error on that slot, mark it failed, and raise
        6. Otherwise, store results in ``slot.result`` and mark slots executed
        7. If return tool executed: set ``state.return_value`` and ``state.is_done = True``
        8. Update ``state.executed_steps``, ``state.tool_calls_used``
        9. Clear ``prepared_steps`` (consumed)

        Parameters
        ----------
        state : RS
            Run state with prepared_steps populated. After execution, updated with
            results and completion flags.

        Returns
        -------
        RS
            Updated state with results recorded and completion status set.

        Raises
        ------
        ToolAgentError
            On any validation failure (preconditions, budget, tool not found, etc.)
            or if any tool invocation raises.

        Side Effects
        ~~~~~~~~~~~~
        - ``state.running_blackboard[idx].result`` is set for executed steps
        - ``state.running_blackboard[idx].error`` is set on failure
        - ``state.running_blackboard[idx].status`` is updated to executed/failed
        - ``state.executed_steps`` is updated with executed indices
        - ``state.tool_calls_used`` incremented by non-return call count
        - ``state.prepared_steps`` is cleared
        - ``state.is_done`` and ``state.return_value`` set if return tool executed
        """
        indices = list(state.prepared_steps)
        if not indices:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: no prepared steps to execute (prepared_steps is empty)."
            )

        if len(indices) != len(set(indices)):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: prepared_steps contains duplicates: {indices!r}."
            )

        board = state.running_blackboard
        board_len = len(board)

        non_return_planned = 0
        return_indices: list[int] = []

        for idx in indices:
            if not isinstance(idx, int):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepared step index must be int; got {type(idx).__name__!r}."
                )
            if idx < 0 or idx >= board_len:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepared step index {idx} out of range "
                    f"(running plan length={board_len})."
                )

            slot = board[idx]

            # slot.step is plan-local during the run; keep it consistent.
            if isinstance(slot.step, int) and slot.step != idx:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: running slot step mismatch at index {idx}: slot.step={slot.step}."
                )

            if slot.is_executed() or idx in state.executed_steps:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepared step {idx} is already executed."
                )
            if not slot.is_prepared():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: slot {idx} is not prepared for execution."
                )

            tool_name = slot.tool
            if tool_name is NO_VAL or not isinstance(tool_name, str) or not tool_name.strip():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: slot {idx} has invalid tool name: {tool_name!r}."
                )

            if tool_name == return_tool.full_name:
                return_indices.append(idx)
            else:
                non_return_planned += 1

        if len(return_indices) > 1:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: multiple return tool calls in one batch: {return_indices!r}."
            )

        # Budget enforcement (non-return only).
        if self._tool_calls_limit is not None:
            if state.tool_calls_used + non_return_planned > self._tool_calls_limit:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool_calls_limit exceeded "
                    f"(limit={self._tool_calls_limit}, used={state.tool_calls_used}, planned={non_return_planned})."
                )

        async def run_batch() -> list[tuple[int, Any]]:
            coros: list[Any] = []

            for idx in indices:
                slot = board[idx]
                tool_name = slot.tool
                tool = self.get_tool(tool_name)

                logger.debug(
                    f"{type(self).__name__}.{self.name}:\nTool: {tool_name}\nArgs: {slot.args}\n\n"
                )

                coros.append(tool.async_invoke(slot.resolved_args))

            raw_results = await asyncio.gather(*coros, return_exceptions=True)

            first_error = next(
                (
                    (idx, raw)
                    for idx, raw in zip(indices, raw_results)
                    if isinstance(raw, BaseException)
                ),
                None,
            )

            if first_error is not None:
                idx, raw_error = first_error

                if isinstance(raw_error, ToolInvocationError):
                    board[idx].error = raw_error
                    board[idx].status = "failed"
                    raise raw_error

                wrapped = ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool call failed at step {idx} for {board[idx].tool!r}: {raw_error}"
                )
                board[idx].error = wrapped
                board[idx].status = "failed"
                raise wrapped from raw_error

            return [(idx, result) for idx, result in zip(indices, raw_results)]

        results = asyncio.run(run_batch())

        for idx, result in results:
            board[idx].result = result
            board[idx].error = NO_VAL
            board[idx].status = "executed"

        # Post-execution bookkeeping
        for idx in indices:
            state.executed_steps.add(idx)

        state.tool_calls_used += non_return_planned
        state.prepared_steps.clear()

        if return_indices:
            ret_idx = return_indices[0]
            state.return_value = board[ret_idx].result
            state.is_done = True

        return state

    async def _async_execute_prepared_batch(self, state: RS) -> RS:
        """
        Async analog of ``_execute_prepared_batch(...)``.

        Executes all currently prepared steps concurrently using each tool's
        ``async_invoke(...)`` path, records results into the running blackboard,
        and updates return/completion bookkeeping.

        This method intentionally preserves the current compact gather-based
        semantics rather than introducing stricter cancellation machinery.
        """
        indices = list(state.prepared_steps)
        if not indices:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: no prepared steps to execute "
                "(prepared_steps is empty)."
            )

        if len(indices) != len(set(indices)):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: prepared_steps contains duplicates: "
                f"{indices!r}."
            )

        board = state.running_blackboard
        board_len = len(board)

        non_return_planned = 0
        return_indices: list[int] = []

        for idx in indices:
            if not isinstance(idx, int):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepared step index must be int; "
                    f"got {type(idx).__name__!r}."
                )
            if idx < 0 or idx >= board_len:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepared step index {idx} out of range "
                    f"(running plan length={board_len})."
                )

            slot = board[idx]

            # During a run, slot.step should remain plan-local and match its position.
            if isinstance(slot.step, int) and slot.step != idx:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: running slot step mismatch at index {idx}: "
                    f"slot.step={slot.step}."
                )

            if slot.is_executed() or idx in state.executed_steps:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepared step {idx} is already executed."
                )

            if not slot.is_prepared():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: slot {idx} is not prepared for execution."
                )

            tool_name = slot.tool
            if tool_name is NO_VAL or not isinstance(tool_name, str) or not tool_name.strip():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: slot {idx} has invalid tool name: "
                    f"{tool_name!r}."
                )

            # Validate existence early so failures happen before gather starts.
            self.get_tool(tool_name)

            if tool_name == return_tool.full_name:
                return_indices.append(idx)
            else:
                non_return_planned += 1

        if len(return_indices) > 1:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: multiple return tool calls in one batch: "
                f"{return_indices!r}."
            )

        if self._tool_calls_limit is not None:
            if state.tool_calls_used + non_return_planned > self._tool_calls_limit:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool_calls_limit exceeded "
                    f"(limit={self._tool_calls_limit}, used={state.tool_calls_used}, "
                    f"planned={non_return_planned})."
                )

        coros: list[Any] = []
        for idx in indices:
            slot = board[idx]
            tool_name = slot.tool
            tool = self.get_tool(tool_name)

            logger.debug(
                f"{type(self).__name__}.{self.name}:\n"
                f"Tool: {tool_name}\n"
                f"Args: {slot.args}\n\n"
            )

            coros.append(tool.async_invoke(slot.resolved_args))

        raw_results = await asyncio.gather(*coros, return_exceptions=True)

        first_error = next(
            (
                (idx, raw)
                for idx, raw in zip(indices, raw_results)
                if isinstance(raw, BaseException)
            ),
            None,
        )

        if first_error is not None:
            idx, raw_error = first_error

            if isinstance(raw_error, ToolInvocationError):
                board[idx].error = raw_error
                board[idx].status = "failed"
                raise raw_error

            wrapped = ToolAgentError(
                f"{type(self).__name__}.{self.name}: tool call failed at index {idx} "
                f"for {board[idx].tool!r}: {raw_error}"
            )
            board[idx].error = wrapped
            board[idx].status = "failed"
            raise wrapped from raw_error

        for idx, result in zip(indices, raw_results):
            board[idx].result = result
            board[idx].error = NO_VAL
            board[idx].status = "executed"
            state.executed_steps.add(idx)

        state.tool_calls_used += non_return_planned
        state.prepared_steps = []

        for idx in reversed(indices):
            if board[idx].tool == return_tool.full_name:
                if state.return_value is not NO_VAL:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: return tool executed more than once."
                    )
                state.return_value = board[idx].result
                state.is_done = True
                break

        return state

    # ------------------------------------------------------------------ #
    # Finalization helpers
    # ------------------------------------------------------------------ #
    def update_blackboard(self, state: ToolAgentRunState) -> ToolAgentRunState:
        """
        Persist executed run steps into the agent's persisted blackboard cache.

        This method is called at the end of ``invoke()`` if ``context_enabled=True``.
        It merges the current run's executed steps into the persistent blackboard,
        enabling future invokes to reference results via cached placeholders.

        Persistence Policy
        ~~~~~~~~~~~~~~~~~~
        1. **Trim empty/unplanned tail**: Remove trailing empty slots from running blackboard
           (slots with no tool assigned)
        2. **Rewrite placeholders**: All ``<<__sN__>>`` step references in appended slots'
           args are rewritten to ``<<__c{new_global_index}__>>`` cache references.
           This ensures cached args never contain step-local placeholders.
        3. **Merge into cache**: Append rewritten executed slots to persist them
        4. **Trim cache tail**: Remove trailing empty slots from final cache

        Placeholder Rewriting Example
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Initial state:
        - cache_blackboard has 5 entries (indices 0-4)
        - running_blackboard has 3 executed entries (indices 0-2)

        Rewriting in appended slots:
        - Step 0's args contain ``<<__s1__>>`` → rewritten to ``<<__c6__>>`` (5 + 1)
        - Step 1's args contain ``<<__s0__>>`` → rewritten to ``<<__c5__>>`` (5 + 0)
        - Step 2's args contain ``<<__s1__>>`` → rewritten to ``<<__c6__>>`` (5 + 1)

        After persistence:
        - cache_blackboard now has 8 entries
        - Future invokes can use ``<<__c5__>>``, ``<<__c6__>>``, ``<<__c7__>>``
          to reference steps 0, 1, 2 respectively

        Parameters
        ----------
        state : ToolAgentRunState
            Run state with executed steps in running_blackboard and cache snapshot.

        Returns
        -------
        ToolAgentRunState
            Updated state after blackboard persistence.

        Side Effects
        ~~~~~~~~~~~~
        - ``self._blackboard`` is replaced with merged cache + appended slots
        """
        base_cache: list[BlackboardSlot] = [slot.copy() for slot in state.cache_blackboard]
        base_len = len(base_cache)

        running: list[BlackboardSlot] = list(state.running_blackboard)

        # 1) Trim empty/unplanned tail from running plan to avoid caching unused slots.
        last = len(running) - 1
        while last >= 0 and running[last].is_empty():
            last -= 1
        running = running[: last + 1]

        def rewrite_step_to_cache_placeholders(obj: Any) -> Any:
            """
            Rewrite <<__sj__>> -> <<__c{base_len + j}__>> recursively.
            Leaves <<__ck__>> unchanged.
            """
            if isinstance(obj, str):
                # exact placeholder: still rewrite as a string placeholder (we are rewriting args,
                # not resolving)
                def repl(m: re.Match[str]) -> str:
                    j = int(m.group(1))
                    return f"<<__c{base_len + j}__>>"

                return _STEP_TOKEN.sub(repl, obj)

            if isinstance(obj, list):
                return [rewrite_step_to_cache_placeholders(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(rewrite_step_to_cache_placeholders(v) for v in obj)
            if isinstance(obj, set):
                return {rewrite_step_to_cache_placeholders(v) for v in obj}
            if isinstance(obj, dict):
                return {
                    rewrite_step_to_cache_placeholders(k): rewrite_step_to_cache_placeholders(v)
                    for k, v in obj.items()
                }
            return obj

        # 2) Append executed running slots as new cache slots with global cache indices.
        appended: list[BlackboardSlot] = []
        for local_i, slot in enumerate(running):
            if not slot.is_executed():
                continue

            new_slot = BlackboardSlot(
                step = base_len + local_i,
                tool = slot.tool,
                args = rewrite_step_to_cache_placeholders(slot.args),
                resolved_args = slot.resolved_args,
                result = slot.result,
                error = slot.error,
                status = "executed",
                step_dependencies = slot.step_dependencies,
                await_step = slot.await_step,
            )
            appended.append(new_slot)

        combined = base_cache + appended

        # 3) Trim empty tail from combined cache.
        if combined:
            last2 = len(combined) - 1
            while last2 >= 0 and combined[last2].is_empty():
                last2 -= 1
            combined = combined[: last2 + 1]

        self._blackboard = combined
        return state

    # ------------------------------------------------------------------ #
    # Template Method (FINAL)
    # ------------------------------------------------------------------ #
    def _invoke(self, *, messages: list[dict[str, str]]) -> tuple[Any, Mapping[str, Any]]:
        """
        FINAL template method (do not override in subclasses).

        Requires subclasses to implement:
        - _initialize_run_state(...)
        - _prepare_next_batch(...)
        """
        if not messages:
            raise ToolAgentError("ToolAgent._invoke requires a non-empty messages list.")

        state = self._initialize_run_state(messages=messages)

        if not isinstance(state, ToolAgentRunState):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: _initialize_run_state must return a ToolAgentRunState (or subclass)."
            )

        while not state.is_done:
            logger.debug(f"{type(self).__name__}.{self.name} has made {state.tool_calls_used} this run")
            # Invariant: prepare must not be called with a pending prepared batch.
            if state.prepared_steps:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: violation: prepared_indices is non-empty before prepare. "
                    f"Execute must follow prepare before preparing again."
                )

            state = self._prepare_next_batch(state)

            # Inline empty-batch check (per design): raise here, not inside helpers.
            if not state.prepared_steps:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepare produced an empty batch (prepared_indices is empty)."
                )

            state = self._execute_prepared_batch(state)

        blackboard_start: int | None = None
        blackboard_end: int | None = None

        # Persist run outputs into cache if context is enabled.
        if self.context_enabled:
            blackboard_start = len(self._blackboard)
            state = self.update_blackboard(state)
            blackboard_end = len(self._blackboard)

        # Collect the final turn metadata.
        turn_metadata = {
            "blackboard_start": blackboard_start,
            "blackboard_end": blackboard_end,
        }
        # Add any additional run metadata from the run state
        turn_metadata.update(state.run_metadata)

        return state.return_value, turn_metadata

    async def _ainvoke(
        self,
        *,
        messages: list[dict[str, str]],
    ) -> tuple[Any, Mapping[str, Any]]:
        """
        Async ToolAgent template method.

        Mirrors the sync `_invoke(...)` loop, but offloads the current sync planning hooks
        to worker threads and awaits the async batch executor for tool execution.

        Subclasses keep the same minimal contract:
        - `_initialize_run_state(...)` stays sync
        - `_prepare_next_batch(...)` stays sync
        - only the execution phase is natively async

        Returns
        -------
        tuple[Any, Mapping[str, Any]]
            The raw return-tool value and metadata containing `blackboard_start` and
            `blackboard_end`.
        """
        if not messages:
            raise ToolAgentError("ToolAgent._ainvoke requires a non-empty messages list.")

        state = await asyncio.to_thread(self._initialize_run_state, messages=messages)

        if not isinstance(state, ToolAgentRunState):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: _initialize_run_state must return "
                "a ToolAgentRunState (or subclass)."
            )

        while not state.is_done:
            logger.debug(
                f"{type(self).__name__}.{self.name} has made {state.tool_calls_used} this run"
            )

            # Invariant: prepare must not be called with a pending prepared batch.
            if state.prepared_steps:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: violation: prepared_steps is non-empty "
                    "before prepare. Execute must follow prepare before preparing again."
                )

            state = await asyncio.to_thread(self._prepare_next_batch, state)

            # Keep the same empty-batch guard as the sync path.
            if not state.prepared_steps:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepare produced an empty batch "
                    "(prepared_steps is empty)."
                )

            state = await self._async_execute_prepared_batch(state)

        blackboard_start: int | None = None
        blackboard_end: int | None = None

        if self.context_enabled:
            blackboard_start = len(self._blackboard)
            state = self.update_blackboard(state)
            blackboard_end = len(self._blackboard)

        # Collect the final turn metadata.
        turn_metadata = {
            "blackboard_start": blackboard_start,
            "blackboard_end": blackboard_end,
        }
        # Add any additional run metadata from the run state
        turn_metadata.update(state.run_metadata)

        return state.return_value, turn_metadata

    def _make_turn(
        self,
        *,
        prompt: str,
        raw_response: Any,
        final_response: Any,
        **metadata: Any,
    ) -> ToolAgentTurn:
        """Construct the stored ToolAgentTurn for one completed invocation.

        Consumes the blackboard span metadata returned by `_invoke(...)` or `_ainvoke(...)`.
        The span is half-open: `blackboard_start` is inclusive and `blackboard_end` is
        exclusive.
        """
        allowed = {"blackboard_start", "blackboard_end"}
        extra = set(metadata) - allowed
        if extra:
            raise ToolAgentError(
                f"{type(self).__name__}._make_turn received unexpected metadata: "
                f"{sorted(extra)!r}"
            )

        blackboard_start = metadata.get("blackboard_start")
        blackboard_end = metadata.get("blackboard_end")

        if blackboard_start is None or blackboard_end is None:
            if blackboard_start is not None or blackboard_end is not None:
                raise ToolAgentError(
                    "blackboard_start and blackboard_end must both be None or both be integers."
                )
        else:
            if type(blackboard_start) is not int or type(blackboard_end) is not int:
                raise ToolAgentError("blackboard_start and blackboard_end must be integers or None.")
            if blackboard_start < 0 or blackboard_end < blackboard_start:
                raise ToolAgentError(
                    "blackboard_start and blackboard_end must satisfy 0 <= start <= end."
                )

        return ToolAgentTurn(
            prompt=prompt,
            raw_response=raw_response,
            final_response=final_response,
            blackboard_start=blackboard_start,
            blackboard_end=blackboard_end,
        )

    def render_turn(self, turn: AgentTurn) -> list[dict[str, str]]:
        """Render one stored ToolAgentTurn into LLM-facing user/assistant messages.

        The base assistant response is rendered through `Agent.render_turn(...)`, preserving
        `assistant_response_source` and `response_preview_limit` behavior. If the turn has
        a non-empty blackboard span, this method appends a cached-step block containing
        each produced step's unresolved args. Result previews are included only when
        `peek_at_cache=True` and are bounded by `blackboard_preview_limit`.
        """
        if not isinstance(turn, ToolAgentTurn):
            raise ToolAgentError(
                f"render_turn expected ToolAgentTurn, got {type(turn)!r}"
            )

        messages = super().render_turn(turn)
        user_message = messages[0]
        assistant_response = messages[1]["content"]

        start = turn.blackboard_start
        end = turn.blackboard_end
        if start is None or end is None or start == end:
            return messages

        if start < 0 or end < start or end > len(self._blackboard):
            raise ToolAgentError(
                f"Invalid blackboard span for rendered turn: start={start!r}, end={end!r}, "
                f"blackboard_length={len(self._blackboard)}."
            )

        extracted: list[dict[str, Any]] = []
        for slot in self._blackboard[start:end]:
            step: dict[str, Any] = {
                "step": slot.step,
                "tool": slot.tool,
                "args": slot.args,
            }
            if self.peek_at_cache:
                step["result"] = self._preview_blackboard_result(slot.result)
            extracted.append(step)

        newest_dump = pprint.pformat(extracted, indent=2, width=160, sort_dicts=False)
        assistant_content = (
            f"RESPONSE:\n{assistant_response}\n\n"
            f"CACHED STEPS #{start}-{end - 1} PRODUCED:\n\n"
            f"{newest_dump}"
        )

        return [
            user_message,
            {"role": "assistant", "content": assistant_content},
        ]

    # ------------------------------------------------------------------ #
    # String to JSON Objects helper
    # ------------------------------------------------------------------ #
    def _extract_from_json_string(self, raw_text: str) -> Any:
        """
        Extract the largest decodable JSON array/object from a possibly noisy string.

        This helper is intentionally shape-neutral:
        - It does not require the decoded value to be a list.
        - It does not require the decoded value to be a dict.
        - It does not validate PlanAct/ReAct-specific fields.

        It preserves the current permissive parsing style used by the older
        `_str_to_steps(...)` and `_str_to_dict(...)` helpers:
        - Strip a single common markdown fence wrapper if present.
        - Scan for candidate JSON array/object starts.
        - Decode with `json.JSONDecoder().raw_decode(...)`.
        - Return the candidate with the largest decoded span.

        Parameters
        ----------
        raw_text : str
            Raw LLM output that may contain a JSON array/object surrounded by
            prose, markdown fences, or other text.

        Returns
        -------
        Any
            The decoded Python value for the largest valid JSON array/object found.

        Raises
        ------
        ToolAgentError
            If `raw_text` is empty/non-string or no valid JSON array/object can be found.
        """
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: LLM returned empty output."
            )

        text = raw_text.strip()

        # Strip a single fenced block wrapper if present.
        # Examples:
        # ```json
        # [...]
        # ```
        text = re.sub(r"^\s*```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text).strip()

        decoder = json.JSONDecoder()

        best_val: Any = NO_VAL
        best_span_len: int = -1

        # Candidate starts: JSON arrays or objects.
        # This intentionally mirrors the existing PlanAct/ReAct parser needs.
        for match in re.finditer(r"[\[{]", text):
            start = match.start()
            try:
                val, end_rel = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue

            if end_rel > best_span_len:
                best_span_len = end_rel
                best_val = val

        if best_val is NO_VAL:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: failed to find a valid JSON array/object in LLM output."
            )

        return best_val

    # ------------------------------------------------------------------ #
    # Dictionary Validation & Conversion Helpers
    # ------------------------------------------------------------------ #
    def _validate_tool_step_dict(
        self,
        data: Mapping[str, Any],
        *,
        expected_step: int,
        allow_await: bool,
        context: str,
    ) -> dict[str, Any]:
        """
        Validate and normalize one raw LLM-produced tool-step mapping.

        This helper is strategy-neutral enough to be shared by PlanAct and ReAct:
        - PlanAct should call it with allow_await=True.
        - ReAct should call it with allow_await=False.

        The raw LLM-produced step number is treated as advisory. The caller-provided
        expected_step is authoritative, so the returned dict always has "step" set to
        expected_step.

        Parameters
        ----------
        data : Mapping[str, Any]
            Raw parsed JSON object representing one tool step.

        expected_step : int
            Authoritative run-local step index to assign to the returned dict.

        allow_await : bool
            Whether the planner-facing "await" scheduling barrier is allowed.

        context : str
            Human-readable label used in error messages, such as "plan step" or
            "next step".

        Returns
        -------
        dict[str, Any]
            Shallow normalized dict with "step" set to expected_step.

        Raises
        ------
        ToolAgentError
            If the raw step shape is invalid.
        """
        if type(expected_step) is not int or expected_step < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} expected_step must be an int >= 0; "
                f"got {expected_step!r}."
            )

        if type(allow_await) is not bool:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} allow_await must be a bool; "
                f"got {type(allow_await).__name__!r}."
            )

        if not isinstance(context, str) or not context.strip():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: context must be a non-empty string."
            )

        if not isinstance(data, Mapping):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} must be a mapping; "
                f"got {type(data).__name__!r}."
            )

        allowed = {"step", "tool", "args"}
        if allow_await:
            allowed.add("await")

        extra = set(data.keys()) - allowed
        if extra:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} contains unsupported keys: "
                f"{sorted(extra)!r}."
            )

        required = {"tool", "args"}
        missing = required - set(data.keys())
        if missing:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} missing required keys: "
                f"{sorted(missing)!r}."
            )

        normalized = dict(data)
        normalized["step"] = expected_step

        tool = normalized["tool"]
        if not isinstance(tool, str) or not tool.strip():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} 'tool' must be a non-empty string."
            )

        args = normalized["args"]
        if not isinstance(args, dict):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} 'args' must be a dict; "
                f"got {type(args).__name__!r}."
            )

        if "await" in normalized:
            if not allow_await:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: {context} must not include 'await'."
                )

            await_step = normalized["await"]
            if type(await_step) is not int or await_step < 0:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: {context} 'await' must be an int >= 0."
                )

            if tool == return_tool.full_name:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: return step must not include 'await'."
                )

        return normalized

    def _tool_step_dict_to_slot(
        self,
        data: Mapping[str, Any],
        *,
        step: int,
        allow_await: bool,
        context: str,
    ) -> BlackboardSlot:
        """
        Convert a normalized tool-step mapping into a planned BlackboardSlot.

        This helper does not validate plan chronology or cache-reference bounds.
        Subclasses remain responsible for validating dependencies against their own
        lifecycle constraints.

        Dependency semantics
        --------------------
        - step_dependencies stores data dependencies extracted from <<__sN__>> args.
        - await_step stores the explicit planner-facing "await" barrier separately.
        - await_step is not folded into step_dependencies here.

        Parameters
        ----------
        data : Mapping[str, Any]
            Normalized step mapping, typically returned by _validate_tool_step_dict(...).

        step : int
            Authoritative run-local step index for the slot.

        allow_await : bool
            Whether to preserve an "await" scheduling barrier from data.

        context : str
            Human-readable label used in error messages.

        Returns
        -------
        BlackboardSlot
            Planned blackboard slot with unresolved args and extracted data dependencies.

        Raises
        ------
        ToolAgentError
            If slot construction fails.
        """
        if type(step) is not int or step < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} step must be an int >= 0; "
                f"got {step!r}."
            )

        if type(allow_await) is not bool:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} allow_await must be a bool; "
                f"got {type(allow_await).__name__!r}."
            )

        if not isinstance(context, str) or not context.strip():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: context must be a non-empty string."
            )

        if not isinstance(data, Mapping):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} must be a mapping; "
                f"got {type(data).__name__!r}."
            )

        tool = data.get("tool", NO_VAL)
        args = data.get("args", NO_VAL)

        if not isinstance(tool, str) or not tool.strip():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} 'tool' must be a non-empty string."
            )

        if not isinstance(args, dict):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} 'args' must be a dict; "
                f"got {type(args).__name__!r}."
            )

        await_step = NO_VAL
        if "await" in data:
            if not allow_await:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: {context} must not include 'await'."
                )
            await_step = data["await"]

        if await_step is not NO_VAL and tool == return_tool.full_name:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: return step must not include 'await'."
            )

        deps: set[int] = set(
            extract_dependencies(obj=args, placeholder_pattern=_STEP_TOKEN)
        )

        try:
            return BlackboardSlot.from_dict(
                {
                    "step": step,
                    "tool": tool,
                    "args": args,
                    "status": "planned",
                    "step_dependencies": tuple(sorted(deps)),
                    "await_step": await_step,
                }
            )
        except Exception as exc:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: failed to construct blackboard slot "
                f"for {context} {step}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Subclass Hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _initialize_run_state(self, *, messages: list[dict[str, str]]) -> RS:
        """
        Initialize and return a run state for this invocation.

        Implementations should:
        - copy the incoming LLM-facing messages into run-local state
        - snapshot persisted blackboard entries if context is enabled
        - allocate an appropriate running blackboard for current-run tool calls
        - initialize execution bookkeeping such as executed_steps, prepared_steps,
        tool_calls_used, is_done, and return_value
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_next_batch(self, state: RS) -> RS:
        """
        Prepare exactly one executable batch for the next loop iteration.

        Implementations should:
        - decide which tool call(s) should execute next
        - validate tool names, step indices, dependencies, and placeholder legality
        - resolve placeholders with `_resolve_placeholders(...)`
        - populate `state.prepared_steps` with the running blackboard indices ready
        for execution
        - return the updated state

        The base ToolAgent loop will execute the prepared batch and handle return-tool
        completion.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of this ToolAgent.

        Extends the base Agent snapshot with ToolAgent-specific toolbox and blackboard
        diagnostics. The inherited `history` field remains a rendered compatibility
        view, while `turn_history` remains the canonical stored turn representation.
        """
        d = super().to_dict()
        d.update({
            "tool_calls_limit": self.tool_calls_limit,
            "peek_at_cache": self.peek_at_cache,
            "blackboard_preview_limit": self.blackboard_preview_limit,
            "tools": {
                name: tool.to_dict()
                for name, tool in self._toolbox.items()
            },
            "blackboard": self.blackboard_serialized(peek=False),
        })
        return d

# --------------------------------------------------------------------------- #
# PlanAct Agent
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PlanActRunState(ToolAgentRunState):
    """
    PlanActAgent-specific run state extending ToolAgentRunState.

    Adds planning-specific fields for managing pre-compiled concurrent batches.

    Fields
    ------
    batches : list[list[int]]
        Pre-compiled topologically-sorted batches. Each batch is a list of plan-local
        indices that can execute concurrently. Created during ``_initialize_run_state()``
        via ``_compile_batches_from_deps()``.

        Example: ``[[0, 1], [2, 3], [4]]`` means:
        - Batch 0: steps 0 and 1 execute together
        - Batch 1: steps 2 and 3 execute together (after batch 0)
        - Batch 2: step 4 executes (after batch 1; typically the return step)

    batch_index : int
        Cursor pointing to the next batch to prepare. Starts at 0 during initialization;
        incremented after each batch is prepared. Loop exits when batch_index >= len(batches).

    Workflow
    ~~~~~~~~
    1. ``_initialize_run_state()`` creates batches and sets batch_index=0
    2. Each iteration of the base loop:
       - ``_prepare_next_batch()`` reads batches[batch_index]
       - Resolves placeholders for that batch
       - Base ``_execute_prepared_batch()`` runs the batch concurrently
       - batch_index incremented for next iteration
    3. When batch_index >= len(batches), _prepare_next_batch() raises (loop exits)
    """
    batches: list[list[int]] = field(default_factory=list)
    batch_index: int = 0


class PlanActAgent(ToolAgent[PlanActRunState]):
    """
    One-shot planner agent: generates entire plan upfront, executes in batches.

    **Design**: PlanActAgent implements a **static planning** strategy:

    1. **Initialization** (``_initialize_run_state``)
       - LLM generates complete plan as a JSON array of steps (one-shot)
       - Each step: ``{"tool": "<name>", "args": {...}}``, optionally with "await"
       - Plan is normalized (return moved to end, added if missing)
       - Compiled into topologically-sorted batches using dependency analysis
       - Running blackboard allocated with slots for all planned steps

    2. **Compilation**
       - Each step's args are scanned for ``<<__sN__>>`` placeholders to extract
         plan-local dependencies
       - Topological sort produces concurrent batches (steps with identical dependency
         level execute together)
       - Return step is always isolated as the final batch

    3. **Execution**
       - ``_prepare_next_batch()`` reads next batch from ``state.batches[state.batch_index]``
       - Resolves placeholders in parallel-executable steps
       - Increments batch_index; loop continues until all batches consumed

    Advantages
    ~~~~~~~~~~
    - **No replanning**: Full plan is known upfront; no latency per iteration
    - **Concurrency-friendly**: Topological compilation enables maximal parallelism
    - **Deterministic**: Same inputs produce identical execution plan every time

    Limitations
    ~~~~~~~~~~~
    - **No adaptivity**: Cannot branch based on intermediate results
    - **Plan quality**: Entirely dependent on LLM's single planning turn
    - **Error recovery**: If a step fails, entire plan fails (no dynamic replanning)

    Parameters (construction)
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Same as ToolAgent, with ``tool_calls_limit`` being the max non-return steps
    in any single plan.
    """
    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        filter_extraneous_inputs: Optional[bool] = None,
        *,
        context_enabled: bool = False,
        tool_calls_limit: int | None = None,
        peek_at_cache: bool = False,
        response_preview_limit: Optional[int] = None,
        blackboard_preview_limit: Optional[int] = None,
        preview_limit: Optional[int] = None,
        pre_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        post_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        history_window: int | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            llm_engine=llm_engine,
            filter_extraneous_inputs=filter_extraneous_inputs,
            role_prompt=PLANNER_PROMPT,
            context_enabled=context_enabled,
            tool_calls_limit=tool_calls_limit,
            peek_at_cache=peek_at_cache,
            response_preview_limit=response_preview_limit,
            blackboard_preview_limit=blackboard_preview_limit,
            preview_limit=preview_limit,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            history_window=history_window,
        )

    # ------------------------------------------------------------------ #
    # Planning + initialization
    # ------------------------------------------------------------------ #
    def _normalize_planned_slots(
        self,
        planned_slots: list[BlackboardSlot],
    ) -> list[BlackboardSlot]:
        """
        Normalize a generated PlanAct slot list into final running-blackboard order.

        Normalization policy
        --------------------
        - The plan must contain at least one generated non-return or return slot.
        - At most one return slot may be present.
        - If a return slot is present, it is moved to the end.
        - If no return slot is present, `return(None)` is appended.
        - Final list positions become authoritative step indices.
        - The final return slot is forced to depend on all prior slots so completion
          represents the whole plan, not just the value in return args.
        - The final return slot cannot have await_step.

        Parameters
        ----------
        planned_slots : list[BlackboardSlot]
            Generated planned slots before return-position normalization.

        Returns
        -------
        list[BlackboardSlot]
            Normalized planned slots.

        Raises
        ------
        ToolAgentError
            If slots are empty, malformed, or contain multiple return slots.
        """
        if not isinstance(planned_slots, list) or not planned_slots:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: generated plan must contain at least one planned slot."
            )

        slots: list[BlackboardSlot] = []
        for i, slot in enumerate(planned_slots):
            if not isinstance(slot, BlackboardSlot):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: planned slot at index {i} must be a BlackboardSlot; "
                    f"got {type(slot).__name__!r}."
                )
            slots.append(slot.copy())

        return_name = return_tool.full_name
        return_positions = [
            i for i, slot in enumerate(slots)
            if slot.tool == return_name
        ]

        if len(return_positions) > 1:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan contains multiple return steps at positions "
                f"{return_positions!r}."
            )

        if len(return_positions) == 1:
            return_slot = slots.pop(return_positions[0])
            slots.append(return_slot)
        else:
            slots.append(
                BlackboardSlot(
                    step=len(slots),
                    tool=return_name,
                    args={"val": None},
                    resolved_args=NO_VAL,
                    result=NO_VAL,
                    error=NO_VAL,
                    status="planned",
                    step_dependencies=tuple(),
                    await_step=NO_VAL,
                )
            )

        for i, slot in enumerate(slots):
            slot.step = i
            slot.resolved_args = NO_VAL
            slot.result = NO_VAL
            slot.error = NO_VAL
            slot.status = "planned"

        return_idx = len(slots) - 1
        return_slot = slots[return_idx]

        if return_slot.tool != return_name:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: normalized plan does not end with return tool."
            )

        # Return is a synthetic finalization step, not a normal data-only step.
        # Force it to depend on every prior step so completion represents the whole plan.
        # This makes the blackboard invariant explicit even though batch compilation also
        # isolates return as the final batch.
        return_slot.step_dependencies = tuple(range(return_idx))
        return_slot.await_step = NO_VAL
        return_slot.status = "planned"

        return slots

    def _validate_planned_slots(
        self,
        *,
        planned_slots: list[BlackboardSlot],
        cache_blackboard: list[BlackboardSlot],
    ) -> None:
        """
        Validate a normalized PlanAct planned-slot list.

        This method validates the final internal representation of the generated
        plan after return normalization and step-index normalization.

        Parameters
        ----------
        planned_slots : list[BlackboardSlot]
            Normalized planned slots.

        cache_blackboard : list[BlackboardSlot]
            Runtime snapshot of persisted cache entries, used for cache-reference
            range validation.

        Raises
        ------
        ToolAgentError
            If the planned slot list violates PlanAct planning invariants.
        """
        if not isinstance(planned_slots, list) or not planned_slots:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: planned_slots must be a non-empty list."
            )

        if not isinstance(cache_blackboard, list):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cache_blackboard must be a list."
            )

        return_name = return_tool.full_name
        return_idx = len(planned_slots) - 1

        for i, slot in enumerate(planned_slots):
            if not isinstance(slot, BlackboardSlot):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: planned slot {i} must be a BlackboardSlot; "
                    f"got {type(slot).__name__!r}."
                )

            if slot.step != i:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: planned slot step mismatch at index {i}: "
                    f"slot.step={slot.step}."
                )

            if not slot.is_planned():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: planned slot {i} must have status='planned'; "
                    f"got {slot.status!r}."
                )

            if slot.tool is NO_VAL or not isinstance(slot.tool, str) or not slot.tool.strip():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: planned slot {i} has invalid tool name: "
                    f"{slot.tool!r}."
                )

            if not self.has_tool(slot.tool):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: unknown tool in plan at step {i}: "
                    f"{slot.tool!r}."
                )

            if i < return_idx and slot.tool == return_name:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: return tool must appear only as the final planned slot."
                )

        if planned_slots[return_idx].tool != return_name:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: final planned slot must be the return tool."
            )

        expected_return_deps = tuple(range(return_idx))
        if planned_slots[return_idx].step_dependencies != expected_return_deps:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: return step dependencies must be "
                f"{expected_return_deps!r}; got {planned_slots[return_idx].step_dependencies!r}."
            )

        limit = self.tool_calls_limit
        if limit is not None:
            non_return = sum(
                1 for slot in planned_slots
                if slot.tool != return_name
            )
            if non_return > limit:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan exceeds tool_calls_limit={limit} "
                    f"(non-return steps={non_return})."
                )

        cache_len = len(cache_blackboard)

        for i, slot in enumerate(planned_slots):
            cache_refs = extract_dependencies(slot.args, placeholder_pattern=_CACHE_TOKEN)
            bad_cache = [idx for idx in cache_refs if idx < 0 or idx >= cache_len]
            if bad_cache:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan step {i} references out-of-range cache indices "
                    f"{sorted(set(bad_cache))!r} (cache length={cache_len})."
                )

            bad_step_deps = [
                dep for dep in slot.step_dependencies
                if dep < 0 or dep >= i
            ]
            if bad_step_deps:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan step {i} has illegal deps "
                    f"{sorted(set(bad_step_deps))!r}; deps must be < {i}."
                )

            if slot.await_step is not NO_VAL:
                if type(slot.await_step) is not int or slot.await_step < 0:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: plan step {i} has invalid await_step "
                        f"{slot.await_step!r}."
                    )
                if slot.await_step >= i:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: plan step {i} has illegal await_step "
                        f"{slot.await_step!r}; await_step must be < {i}."
                    )

    def _generate_plan(
        self,
        *,
        messages: list[dict[str, str]],
        cache_blackboard: list[BlackboardSlot],
    ) -> list[BlackboardSlot]:
        """
        Generate, normalize, and validate a complete PlanAct running blackboard.

        This is the PlanAct generation hook. It is intentionally single-shot and
        fail-fast: no retry logic is performed here.

        Lifecycle
        ---------
        1. Generate raw LLM text from the provided messages.
        2. Extract the largest JSON array/object from the raw text.
        3. Validate that the extracted value is a non-empty list of mappings.
        4. Convert each mapping into a planned BlackboardSlot.
        5. Normalize the planned slot list.
        6. Validate the final planned slot list.
        7. Return the final list of planned slots.

        Parameters
        ----------
        messages : list[dict[str, str]]
            LLM-facing messages already built by the base Agent message pipeline.

        cache_blackboard : list[BlackboardSlot]
            Snapshot of persisted blackboard entries available to this invoke.
            Used for validating cache placeholder references.

        Returns
        -------
        list[BlackboardSlot]
            Fully normalized and validated planned slots for the running blackboard.

        Raises
        ------
        ToolAgentError
            If generation output cannot be parsed, converted, normalized, or validated.
        """
        if not messages:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: messages must be non-empty."
            )

        raw_plan = self._llm_engine.invoke({"messages": [dict(m) for m in messages]})
        parsed = self._extract_from_json_string(raw_plan)

        if not isinstance(parsed, list) or not parsed:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan output must be a non-empty JSON array."
            )

        planned_slots: list[BlackboardSlot] = []

        for i, item in enumerate(parsed):
            if not isinstance(item, Mapping):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan item at index {i} must be a JSON object; "
                    f"got {type(item).__name__!r}."
                )

            step_dict = self._validate_tool_step_dict(
                item,
                expected_step=i,
                allow_await=True,
                context="plan step",
            )
            slot = self._tool_step_dict_to_slot(
                step_dict,
                step=i,
                allow_await=True,
                context="plan step",
            )
            planned_slots.append(slot)

        planned_slots = self._normalize_planned_slots(planned_slots)

        self._validate_planned_slots(
            planned_slots=planned_slots,
            cache_blackboard=cache_blackboard,
        )

        return planned_slots

    def _initialize_run_state(self, *, messages: list[dict[str, str]]) -> PlanActRunState:
        """
        One-shot plan generation and compilation into concurrent batches.

        This PlanActAgent-specific initialization method performs the run-local setup
        for a complete planning cycle, delegates plan generation/normalization/validation
        to `_generate_plan(...)`, and pre-compiles the resulting planned slots into
        topologically-sorted concurrent batches.

        Execution Steps
        ~~~~~~~~~~~~~~~
        1. **Snapshot cache** from ``self._blackboard`` if ``context_enabled=True``
           (persisted results from prior invokes)

        2. **Plan generation**:
           - Delegates to ``_generate_plan(...)``
           - The LLM receives the conversation messages
           - Expected to emit a JSON array of step objects:
             ``[{"tool": "...", "args": {...}}, ...]``
           - ``step`` is optional/advisory; JSON array position is authoritative
           - Optionally, each non-return step can include an "await" field

        3. **Planned slot generation**:
           - ``_generate_plan(...)`` extracts JSON from the raw LLM text
           - Validates each raw step object
           - Converts each step into a ``BlackboardSlot`` with ``status="planned"``
           - Normalizes return placement and final step indices
           - Validates tools, cache references, dependencies, and budget

        4. **Batch compilation**:
           - Topologically sorts slots by ``step_dependencies``
           - Returns list of batches: each batch's indices execute concurrently
           - Return is always in its own final batch

        Parameters
        ----------
        messages : list[dict[str, str]]
            LLM conversation history to pass to the planner.

        Returns
        -------
        PlanActRunState
            Initialized state ready for the base template-method loop:
            - cache_blackboard populated with prior results when context_enabled=True
            - running_blackboard populated with planned BlackboardSlot objects
            - batches list with topologically-sorted concurrent batches
            - batch_index=0 (first batch)

        Raises
        ------
        ToolAgentError
            On any of:
            - Empty messages
            - Empty or invalid plan from LLM
            - Multiple return steps
            - Unknown tool references
            - Out-of-range placeholder references
            - Invalid plan dependencies
            - Budget exceeded
        """
        if not messages:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: messages must be non-empty."
            )

        working_messages = [dict(m) for m in messages]

        cache_blackboard: list[BlackboardSlot] = (
            self.blackboard if self.context_enabled else []
        )
        for i, slot in enumerate(cache_blackboard):
            slot.step = i

        planned_slots = self._generate_plan(
            messages=working_messages,
            cache_blackboard=cache_blackboard,
        )

        if not planned_slots:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: generated plan is empty."
            )

        return_idx = len(planned_slots) - 1
        if planned_slots[return_idx].tool != return_tool.full_name:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: generated plan does not end with return tool."
            )

        batches = self._compile_batches_from_deps(
            planned_slots=planned_slots,
            return_idx=return_idx,
        )

        return PlanActRunState(
            messages=working_messages,
            cache_blackboard=cache_blackboard,
            running_blackboard=planned_slots,
            executed_steps=set(),
            prepared_steps=[],
            tool_calls_used=0,
            is_done=False,
            return_value=NO_VAL,
            batches=batches,
            batch_index=0,
        )

    def _compile_batches_from_deps(
        self,
        *,
        planned_slots: list[BlackboardSlot],
        return_idx: int,
    ) -> list[list[int]]:
        """
        Compile concurrent batches from plan-local scheduling dependencies.

        For non-return step i:
          scheduling_deps[i] = step_dependencies + await_step if present
          level[i] = 0 if scheduling_deps are empty else
          1 + max(level[d] for d in scheduling_deps)

        step_dependencies represent data dependencies extracted from <<__sN__>>
        placeholders. await_step is an explicit scheduling barrier and is folded into
        dependencies only locally while compiling execution batches.

        Return step is always isolated as its own final batch [return_idx].
        """
        if not planned_slots:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: cannot compile empty plan.")

        if (
            return_idx != len(planned_slots) - 1
            or planned_slots[return_idx].tool != return_tool.full_name
        ):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: return_idx mismatch during batch compilation."
            )

        levels: dict[int, int] = {}

        # Non-return only.
        for i in range(return_idx):
            slot = planned_slots[i]

            scheduling_deps: set[int] = set(slot.step_dependencies)
            if slot.await_step is not NO_VAL:
                scheduling_deps.add(slot.await_step)

            if not scheduling_deps:
                levels[i] = 0
            else:
                levels[i] = 1 + max(levels[d] for d in scheduling_deps)

        buckets: dict[int, list[int]] = {}
        for i in range(return_idx):
            lvl = levels.get(i, 0)
            buckets.setdefault(lvl, []).append(i)

        batches: list[list[int]] = []
        for lvl in sorted(buckets):
            batch = sorted(buckets[lvl])
            if batch:
                batches.append(batch)

        batches.append([return_idx])
        return batches

    # ------------------------------------------------------------------ #
    # Prepare next batch
    # ------------------------------------------------------------------ #
    def _prepare_next_batch(self, state: PlanActRunState) -> PlanActRunState:
        """
        Prepare the next pre-compiled batch for execution.

        PlanActAgent uses pre-compiled batches created during initialization. This method
        reads the next batch indices, resolves placeholders, marks those slots prepared,
        and populates the prepared_steps list.

        Execution
        ~~~~~~~~~
        1. **Validate state**: prepared_steps must be empty
        2. **Read next batch**: Get batch indices from ``state.batches[state.batch_index]``
        3. **Validate non-empty**: Batch must have at least one step
        4. **For each step in batch**:
           - Validate bounds: index must be within running_blackboard
           - Validate not already executed or prepared
           - Validate slot is currently planned
           - Validate tool name is set
           - Call ``_resolve_placeholders(slot.args, state=state)``
           - Store resolved args in ``slot.resolved_args``
           - Mark slot ``status="prepared"``
        5. **Set prepared_steps**: List of all indices in this batch
        6. **Advance cursor**: Increment ``state.batch_index`` for next iteration

        Concurrency
        ~~~~~~~~~~~
        All steps in the batch can execute concurrently since the topological sort
        guarantee ensures no step in a batch depends on another step in the same batch.

        Parameters
        ----------
        state : PlanActRunState
            Current run state with initialized batches and batch_index cursor

        Returns
        -------
        PlanActRunState
            Updated state with prepared_steps populated, batch_index incremented

        Raises
        ------
        ToolAgentError
            On any of:
            - prepared_steps not empty
            - batch_index out of bounds
            - Batch validation failure
            - Placeholder resolution failure
        """
        if state.prepared_steps:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cannot prepare next batch while prepared_steps is non-empty."
            )

        if state.batch_index >= len(state.batches):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: no remaining batches to prepare (batch_index={state.batch_index})."
            )

        batch = state.batches[state.batch_index]
        if not batch:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: encountered empty batch at index {state.batch_index}."
            )

        board = state.running_blackboard
        board_len = len(board)

        # Resolve args for all steps in the batch; resolver enforces readiness.
        for i in batch:
            if not isinstance(i, int):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch contains non-int index: {i!r}."
                )
            if i < 0 or i >= board_len:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch index {i} out of range (plan length={board_len})."
                )

            slot = board[i]

            # Invariant: slot.step is plan-local index.
            if slot.step != i:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: running slot step mismatch at index {i}: slot.step={slot.step}."
                )

            if slot.is_executed():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch references already executed step {i}."
                )
            if slot.is_prepared():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch references already prepared step {i}."
                )
            if slot.is_failed():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch references failed step {i}."
                )
            if not slot.is_planned():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch references non-planned step {i} "
                    f"with status={slot.status!r}."
                )

            if slot.tool is NO_VAL or not isinstance(slot.tool, str) or not slot.tool.strip():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step {i} has invalid tool name in running_blackboard."
                )

            # Resolve placeholders using base resolver (checks cache + executed step readiness).
            slot.resolved_args = self._resolve_placeholders(slot.args, state=state)

            if slot.result is not NO_VAL:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: internal error: step {i} result was set during preparation."
                )

            slot.error = NO_VAL
            slot.status = "prepared"

        state.prepared_steps = sorted(batch)
        state.batch_index += 1
        logger.info(
            f"{self.full_name}: Prepared batch {state.batch_index}/{len(state.batches)} "
            f"with steps {state.prepared_steps}."
        )
        return state


# ───────────────────────────────────────────────────────────────────────────────
# Iterative Plan 'ReActAgent' class
# ───────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class ReActRunState(ToolAgentRunState):
    """
    ReActAgent-specific run state extending ToolAgentRunState.

    Tracks iteration-specific state for step-by-step reactive planning.

    Fields
    ------
    next_step_index : int
        Cursor for the next plan-local running_blackboard slot to fill. Starts at 0;
        incremented after each step is prepared.

        Dual role:
        1. **Allocation cursor**: Determines which slot index gets the next prepared step
        2. **Dependency cutoff**: Any <<__sN__>> placeholder in newly prepared args
           must satisfy N < next_step_index (cannot reference future steps)

        Example: If next_step_index=3, step being prepared cannot use <<__s3__>> or higher.

    latest_executed : list[int]
        Plan-local indices of the most recently *prepared* batch (always length 1 for
        ReAct). Used at the start of the next prepare() call to inject observation
        message into LLM context.

        Workflow:
        - After step N prepared: latest_executed = [N]
        - After step N executed: next iteration calls prepare() with latest_executed=[N]
        - prepare() injects \"Most recently executed: step N result=...\" into messages
        - LLM sees this observation and emits next step

        Enables reactive feedback loop: see result → emit next step.

    Workflow
    ~~~~~~~~
    Iteration k (k=1,2,...):

    1. prepare():
       - If latest_executed non-empty: inject observation into messages
       - Request next step from LLM (fresh turn)
       - Parse step object: {\"step\": idx, \"tool\": ..., \"args\": {...}}
       - Validate: idx == next_step_index (cursor enforcement)
       - Fill running_blackboard[idx]
       - Set prepared_steps=[idx], latest_executed=[idx]
       - Increment next_step_index

    2. execute():
       - Run step concurrently (trivial: length 1 batch)
       - Store result in running_blackboard[idx]

    3. emit control back to loop → goto iteration k+1
    """
    next_step_index: int = 0
    latest_executed: list[int] = field(default_factory=list)


class ReActAgent(ToolAgent[ReActRunState]):
    """
    Iterative agent with reactive step-by-step planning (ReAct-style architecture).

    **Design**: ReActAgent implements **dynamic iteration**: one step is emitted per LLM
    turn, with full visibility into prior results.

    1. **Initialization** (``_initialize_run_state``)
       - Pre-allocates a fixed-size running blackboard: ``tool_calls_limit + 1`` slots
         (to accommodate non-return tool calls + one return call)
       - Requires ``tool_calls_limit`` to be a concrete integer (not None)
       - Snapshots cached blackboard entries only when ``context_enabled=True``

    2. **Preparation** (``_prepare_next_batch`` – single step per turn)
       - **First turn**: LLM receives the original user query; emits first step as JSON
       - **Subsequent turns**: Injects "most recently executed step and result" into
         messages, then asks for the next step
       - Step must be a JSON object: ``{"step": <int>, "tool": "<name>", "args": {...}}``
       - Validates step index matches expected position (cursor enforcement)
       - Validates placeholder dependencies forward (step N can only ref steps < N)
       - Resolves placeholders; stores in ``running_blackboard[step_index]``
       - Sets ``prepared_steps = [step_index]`` (always length 1)

    3. **Execution**
       - Base loop executes the single step (trivial with thread pool)
       - Result is stored; loop returns to prepare next step

    4. **Termination**
       - When return tool is emitted and executed, loop exits
       - Running blackboard is persisted if ``context_enabled=True``

    Advantages
    ~~~~~~~~~~
    - **Fully adaptive**: Each step reacts to prior results; LLM can adjust dynamically
    - **Error recovery**: Failed steps don't invalidate entire plan; agent can replan
    - **Interpretability**: Clear step-by-step execution trace for auditing

    Limitations
    ~~~~~~~~~~~
    - **Higher latency**: One LLM call per step (vs. entire plan upfront)
    - **No concurrency**: Only one step executes per iteration
    - **Harder to optimize**: Difficult to parallelize cross-invocation

    Parameters (construction)
    ~~~~~~~~~~~~~~~~~~~~~~~~
    ``tool_calls_limit`` (int, REQUIRED): Must be a concrete integer >= 0.
    Determines pre-allocated blackboard size. Cannot be ``None``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        filter_extraneous_inputs: Optional[bool] = None,
        *,
        context_enabled: bool = False,
        tool_calls_limit: int = 25,
        peek_at_cache: bool = False,
        response_preview_limit: Optional[int] = None,
        blackboard_preview_limit: Optional[int] = None,
        preview_limit: Optional[int] = None,
        pre_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        post_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        history_window: int | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            llm_engine=llm_engine,
            filter_extraneous_inputs=filter_extraneous_inputs,
            role_prompt=ORCHESTRATOR_PROMPT,
            context_enabled=context_enabled,
            tool_calls_limit=tool_calls_limit,
            peek_at_cache=peek_at_cache,
            response_preview_limit=response_preview_limit,
            blackboard_preview_limit=blackboard_preview_limit,
            preview_limit=preview_limit,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            history_window=history_window,
        )

        # ReAct requires a concrete integer tool_calls_limit so that we can preallocate
        # a fixed-size running blackboard.
        if type(tool_calls_limit) is not int or tool_calls_limit < 0:
            raise ToolAgentError("ReActAgent requires tool_calls_limit to be an int >= 0.")

    # ------------------------------------------------------------------ #
    # Tool-Agent Hooks
    # ------------------------------------------------------------------ #
    @property
    def tool_calls_limit(self) -> int:
        """Max allowed non-return tool calls per invoke() run. Must be an int >= 0."""
        return self._tool_calls_limit

    @tool_calls_limit.setter
    def tool_calls_limit(self, value: int) -> None:
        if type(value) is not int or value < 0:
            raise ToolAgentError("ReActAgent requires tool_calls_limit to be an int >= 0.")
        self._tool_calls_limit = value

    # ------------------------------------------------------------------ #
    # Tool-Agent Hooks
    # ------------------------------------------------------------------ #
    def _initialize_run_state(self, *, messages: list[dict[str, str]]) -> ReActRunState:
        if not messages:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: messages must be non-empty.")

        if self._tool_calls_limit is None or type(self._tool_calls_limit) is not int or self._tool_calls_limit < 0:
            raise ToolAgentError("ReActAgent requires tool_calls_limit to be an int >= 0.")

        cache_blackboard = self.blackboard if self.context_enabled else []
        for i, slot in enumerate(cache_blackboard):
            slot.step = i

        working_messages = list(messages)

        # Preallocate fixed-size run blackboard: non-return calls + 1 return call.
        running_blackboard = [BlackboardSlot(step=i) for i in range(self._tool_calls_limit + 1)]

        return ReActRunState(
            messages=working_messages,
            cache_blackboard=cache_blackboard,
            running_blackboard=running_blackboard,
            executed_steps=set(),
            prepared_steps=[],
            tool_calls_used=0,
            is_done=False,
            return_value=NO_VAL,
            next_step_index=0,
            latest_executed=[],
        )

    def _generate_next_step(
        self,
        *,
        messages: list[dict[str, str]],
        cache_blackboard: list[BlackboardSlot],
        expected_step: int,
    ) -> BlackboardSlot:
        """
        Generate and validate one ReAct tool step as a planned BlackboardSlot.

        This is the ReAct generation hook. It is intentionally single-shot and
        fail-fast: no retry logic is performed here.

        Lifecycle
        ---------
        1. Generate raw LLM text from the provided messages.
        2. Extract the largest JSON array/object from the raw text.
        3. Validate that the extracted value is a mapping.
        4. Normalize the raw step dict using expected_step as authoritative.
        5. Convert the normalized mapping into a planned BlackboardSlot.
        6. Validate tool existence.
        7. Validate cache references against cache_blackboard.
        8. Validate step dependencies are prior-only.
        9. Return the planned slot.

        Parameters
        ----------
        messages : list[dict[str, str]]
            LLM-facing messages for this ReAct step.

        cache_blackboard : list[BlackboardSlot]
            Snapshot of persisted blackboard entries available to this invoke.
            Used for validating cache placeholder references.

        expected_step : int
            Authoritative plan-local step index for the generated slot. The raw
            LLM-produced "step" value is optional/advisory and is overwritten.

        Returns
        -------
        BlackboardSlot
            Planned single-step slot. The slot is not inserted into the running
            blackboard and placeholders are not resolved here.

        Raises
        ------
        ToolAgentError
            If generation output cannot be parsed, converted, or validated.
        """
        if not messages:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: messages must be non-empty."
            )

        if not isinstance(cache_blackboard, list):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cache_blackboard must be a list."
            )

        if type(expected_step) is not int or expected_step < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: expected_step must be an int >= 0; "
                f"got {expected_step!r}."
            )

        raw_text = self._llm_engine.invoke({"messages": [dict(m) for m in messages]})
        parsed = self._extract_from_json_string(raw_text)

        if not isinstance(parsed, Mapping):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next step output must be a JSON object; "
                f"got {type(parsed).__name__!r}."
            )

        step_dict = self._validate_tool_step_dict(
            parsed,
            expected_step=expected_step,
            allow_await=False,
            context="next step",
        )

        slot = self._tool_step_dict_to_slot(
            step_dict,
            step=expected_step,
            allow_await=False,
            context="next step",
        )

        # Validate tool exists before this slot is later stamped into run state.
        self.get_tool(slot.tool)

        cache_len = len(cache_blackboard)
        cache_refs = extract_dependencies(slot.args, placeholder_pattern=_CACHE_TOKEN)
        bad_cache = [idx for idx in cache_refs if idx < 0 or idx >= cache_len]
        if bad_cache:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next step references out-of-range cache indices "
                f"{sorted(set(bad_cache))!r} (cache length={cache_len})."
            )

        bad_step_deps = [
            dep for dep in slot.step_dependencies
            if dep < 0 or dep >= expected_step
        ]
        if bad_step_deps:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next step has illegal deps "
                f"{sorted(set(bad_step_deps))!r}; deps must be < {expected_step}."
            )

        return slot

    def _prepare_next_batch(self, state: ReActRunState) -> ReActRunState:
        """
        Prepare the next single-step batch.

        Semantics (single-step ReAct):
        - If state.latest_executed is non-empty, inject an assistant observation message
            describing the most recently executed step including result, followed by a
            small user request for the next step.
        - Call `_generate_next_step(...)`, which must return one planned BlackboardSlot.
        - Validate the generated slot against the current run cursor.
        - Fill exactly one preallocated slot in the running_blackboard.
        - Resolve placeholders against the current ToolAgent run state.
        - Mark the slot prepared.
        - prepared_steps is a list of exactly one index.
        - next_step_index advances by 1.
        - latest_executed is overwritten with the newly prepared index. By the next
            prepare call, the base execute loop must have executed it.
        """
        if not isinstance(state, ReActRunState):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: _prepare_next_batch requires a ReActRunState."
            )

        if state.prepared_steps:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cannot prepare next batch while prepared_steps is non-empty."
            )

        prefix_len = state.next_step_index
        if type(prefix_len) is not int or prefix_len < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next_step_index must be an int >= 0; "
                f"got {prefix_len!r}."
            )

        if prefix_len >= len(state.running_blackboard):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next_step_index exceeds run blackboard capacity "
                f"({prefix_len} >= {len(state.running_blackboard)})."
            )

        # ------------------------------------------------------------------ #
        # 1) Build run-local messages for this ReAct LLM turn
        # ------------------------------------------------------------------ #
        working_messages: list[dict[str, str]] = [dict(m) for m in state.messages]

        if state.latest_executed:
            if len(state.latest_executed) != 1:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: ReAct latest_executed must contain exactly one index; "
                    f"got {state.latest_executed!r}."
                )

            latest_idx = state.latest_executed[0]
            if type(latest_idx) is not int:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: latest_executed index must be int; "
                    f"got {type(latest_idx).__name__!r}."
                )
            if latest_idx < 0 or latest_idx >= len(state.running_blackboard):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: latest_executed index {latest_idx} out of range "
                    f"(running plan length={len(state.running_blackboard)})."
                )

            slot = state.running_blackboard[latest_idx]
            if not slot.is_executed():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: latest_executed step {latest_idx} is not executed "
                    f"(status={slot.status!r})."
                )

            obs_payload = {
                "step": slot.step,
                "tool": slot.tool,
                "args": slot.args,
                "result": self._preview_blackboard_result(slot.result),
            }
            obs_text = "Most recently executed steps and results:\n" + pprint.pformat(
                obs_payload, indent=2, sort_dicts=False
            )
            working_messages.append({"role": "assistant", "content": obs_text})

            working_messages.append(
                {
                    "role": "user",
                    "content": "Given the most recently executed steps and available CACHE (if provided) above, "
                            "produce the NEXT single step as ONE JSON object with keys {step, tool, args}.",
                }
            )

        # ------------------------------------------------------------------ #
        # 2) Generate and validate the next planned slot
        # ------------------------------------------------------------------ #
        generated_slot = self._generate_next_step(
            messages=working_messages,
            cache_blackboard=state.cache_blackboard,
            expected_step=prefix_len,
        )

        if not isinstance(generated_slot, BlackboardSlot):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: _generate_next_step must return a BlackboardSlot; "
                f"got {type(generated_slot).__name__!r}."
            )

        if generated_slot.step != prefix_len:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: generated step mismatch: "
                f"got {generated_slot.step}, expected {prefix_len}."
            )

        if not generated_slot.is_planned():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: generated slot must be planned; "
                f"got status={generated_slot.status!r}."
            )

        # Persist the run-local ReAct message transcript only after successful generation.
        state.messages = working_messages

        # ------------------------------------------------------------------ #
        # 3) Fill the next slot of the preallocated running blackboard
        # ------------------------------------------------------------------ #
        slot = state.running_blackboard[prefix_len]
        if slot.step != prefix_len:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: running slot step mismatch at index {prefix_len}: "
                f"slot.step={slot.step}."
            )

        if not slot.is_empty():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: attempted to prepare into non-empty slot {prefix_len}."
            )

        slot.tool = generated_slot.tool
        slot.args = generated_slot.args
        slot.result = NO_VAL
        slot.error = NO_VAL
        slot.step_dependencies = generated_slot.step_dependencies
        slot.await_step = generated_slot.await_step

        # Resolve placeholders after stamping the planned slot into the running state.
        # The base resolver validates that referenced cache/running slots are executed.
        slot.resolved_args = self._resolve_placeholders(slot.args, state=state)
        slot.status = "prepared"

        # prepared_steps is what the base execute() will run next (list-of-one).
        state.prepared_steps = [prefix_len]

        # Used for observation injection at the start of the next prepare() call.
        # The base ToolAgent loop executes this prepared step before preparing again.
        state.latest_executed = [prefix_len]

        # Advance cursor.
        state.next_step_index = prefix_len + 1

        return state
