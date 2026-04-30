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
from .data_classes import AgentTurn, ToolAgentTurn
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

def truncate_for_preview(obj: Any, limit: Optional[int]) -> Any:
    """
    Truncate an object's representation for display in preview messages.
    
    If limit is None, returns obj unchanged.
    If repr(obj) length <= limit, returns obj unchanged.
    Otherwise, intelligently truncates at collection boundaries and returns
    a formatted string: "(type_name)truncated_repr..." where the type prefix
    and ellipsis do not count toward the character limit.
    
    For collections (list, dict, set, tuple), truncation attempts to include
    complete items/pairs and ends with the appropriate closing bracket.
    
    Parameters
    ----------
    obj : Any
        Object to potentially truncate
    limit : Optional[int]
        Character limit for the representation. If None, no truncation.
        If a positive integer, truncation occurs only if repr exceeds this limit.
    
    Returns
    -------
    Any
        Either the original object unchanged, or a truncated string representation.
    """
    if limit is None:
        return obj
    
    try:
        repr_str = repr(obj)
    except Exception:
        repr_str = str(obj)
    
    if len(repr_str) <= limit:
        return obj
    
    # Need to truncate
    type_name = type(obj).__name__
    
    # Try intelligent truncation for collections
    if isinstance(obj, (list, tuple, set, dict)):
        # Choose bracket characters
        if isinstance(obj, list):
            open_br, close_br = "[", "]"
        elif isinstance(obj, tuple):
            open_br, close_br = "(", ")"
        elif isinstance(obj, set):
            open_br, close_br = "{", "}"
        else:
            open_br, close_br = "{", "}"

        # Build inner content up to the limit (limit counts ONLY inner characters)
        inner = ""
        if isinstance(obj, dict):
            iterator = obj.items()
            for i, (k, v) in enumerate(iterator):
                kv_repr = f"{repr(k)}: {repr(v)}"
                sep = ", " if i > 0 else ""
                if len(inner) + len(sep) + len(kv_repr) > limit:
                    break
                inner += sep + kv_repr
        else:
            iterator = obj
            for i, item in enumerate(iterator):
                item_repr = repr(item)
                sep = ", " if i > 0 else ""
                if len(inner) + len(sep) + len(item_repr) > limit:
                    break
                inner += sep + item_repr

        # If everything fit, return full representation without ellipsis
        full_inner = None
        try:
            full_inner = repr(obj)[len(open_br):-len(close_br)] if repr_str.startswith(open_br) and repr_str.endswith(close_br) else None
        except Exception:
            full_inner = None

        if inner and (len(inner) < len(repr_str) or full_inner is None and len(repr_str) > limit):
            # Truncated: place ellipsis INSIDE the brackets
            return f"({type_name}){open_br}{inner}...{close_br}"

        # If inner is empty but repr_str was longer than limit, fall back to simple truncation
        if not inner and len(repr_str) > limit:
            truncated = repr_str[:limit]
            return f"({type_name}){truncated}..."

        # Not truncated — return original representation (as string) to preserve formatting
        return f"({type_name}){repr_str}"

    else:
        # For other types, just truncate the repr directly
        truncated = repr_str[:limit]
        return f"({type_name}){truncated}..."

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
class BlackboardSlot:
    """
    One indexed slot in the run blackboard, representing a single tool invocation.

    Each slot tracks the complete lifecycle of a tool call from planning through execution.
    State transitions are **sentinel-driven** using the ``NO_VAL`` marker:

    State Lifecycle
    ~~~~~~~~~~~~~~~
    1. **Empty** (initial): ``tool=NO_VAL, resolved_args=NO_VAL, result=NO_VAL``
       - Slot allocated but not yet planned

    2. **Prepared**: ``tool≠NO_VAL, resolved_args≠NO_VAL, result=NO_VAL``
       - Slot assigned a tool and resolved arguments; ready for execution
       - Placeholder dependencies have been resolved to concrete values

    3. **Executed**: ``result≠NO_VAL`` (or ``error≠NO_VAL`` on failure)
       - Tool has been invoked; result (or exception) stored
       - Slot is now available for subsequent steps' placeholder resolution

    Fields
    ------
    step : int
        Global blackboard index (0-based). Always matches the slot's position in the
        containing blackboard list during planning. After persistence to cache, this
        index becomes globally unique (incremented from previous cache length).

    tool : str | NO_VAL
        Tool name (``Tool.full_name``). Set at prepare time; must reference a
        registered tool or invoke will raise.

    args : Any (typically dict)
        Raw, unresolved arguments. May contain placeholders (``<<__sN__>>``,
        ``<<__cN__>>``). Immutable after prepare time.

    resolved_args : Any (typically dict) | NO_VAL
        Arguments after placeholder resolution. Created at prepare time by
        ``_resolve_placeholders(args, state=...)``. Passed to ``tool.invoke()``.

    result : Any | NO_VAL
        Tool execution result. Set by ``_execute_prepared_batch()`` on success.
        Used for placeholder resolution in dependent steps.

    error : Any | NO_VAL
        Exception captured during execution (if any). Set only on failure.
        Result remains ``NO_VAL`` if error is set.
    """
    step: int

    tool: str | Any = NO_VAL
    args: Any = NO_VAL
    resolved_args: Any = NO_VAL
    result: Any = NO_VAL
    error: Any = NO_VAL

    def is_empty(self) -> bool:
        return self.tool is NO_VAL and self.result is NO_VAL and self.resolved_args is NO_VAL

    def is_prepared(self) -> bool:
        return self.tool is not NO_VAL and self.resolved_args is not NO_VAL and self.result is NO_VAL

    def is_executed(self) -> bool:
        return self.result is not NO_VAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tool": self.tool,
            "args": self.args,
            "resolved_args": self.resolved_args,
            "result": self.result,
            "error": self.error,
            "completed": bool(self.is_executed()),
        }

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
        ``cache_blackboard[N].result is not NO_VAL``.

    **Step Placeholder** (``<<__sN__>>``)
        Resolvable iff ``0 <= N < len(running_blackboard)`` AND
        ``running_blackboard[N].is_executed()``.

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

    @property
    def blackboard_serialized(self) -> list[dict[str, Any]]:
        """
        Read-only serialized view of the persisted blackboard.
        (Keeps backward-friendly dict shape.)
        """
        return [slot.to_dict() for slot in self._blackboard]
    
    @property
    def blackboard(self) -> list[BlackboardSlot]:
        return [BlackboardSlot(step = slot.step,
                               tool = slot.tool,
                               args = slot.args,
                               resolved_args = slot.resolved_args,
                               result = slot.result,
                               error = slot.error) for slot in self._blackboard]
    
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
        No other format fields are allowed.
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
                    "Only {TOOLS} and {TOOL_CALLS_LIMIT} are supported."
                )
            fields.add(field_name)

        required = {"TOOLS", "TOOL_CALLS_LIMIT"}
        missing = required - fields
        if missing:
            raise ToolAgentError(
                f"ToolAgent role_prompt template missing required placeholder(s): {', '.join(sorted(missing))}."
            )

        extra = fields - required
        if extra:
            raise ToolAgentError(
                f"ToolAgent role_prompt template contains unsupported placeholder(s): {', '.join(sorted(extra))}. "
                "Only {TOOLS} and {TOOL_CALLS_LIMIT} are supported."
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
    # Blackboard formatting
    # ------------------------------------------------------------------ #
    def blackboard_dumps(self, *, peek: bool = False, indent = 2, width = 160) -> str:
        """
        Pretty representation of the persisted blackboard.

        peek=False -> placeholder-view (args)
        peek=True  -> placeholder-view plus result previews

        Includes explicit `step` numbering to make global indexing unambiguous.
        If a slot's `step` is NO_VAL, the list position is used.
        """
        view: list[dict[str, Any]] = []

        for i, d in enumerate(self.blackboard_serialized):
            d["step"] = i
            d.pop("resolved_args", None)
            if d["error"] is NO_VAL:
                d.pop("error")
            if peek:
                d["result"] = self._preview_blackboard_result(d["result"])
            else:
                d.pop("result")
            view.append(d)

        try:
            return pprint.pformat(view, indent=indent, width=width, sort_dicts=False)
        except Exception:  # pragma: no cover
            return str(view)

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
        Before resolution, validates that all referenced slots are **executed** (result set):

        - Referenced slot must exist (index within bounds)
        - Referenced slot must have ``result != NO_VAL``

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
            if cache[idx].result is NO_VAL:
                raise ToolAgentError(f"Referenced cache {idx} is not executed (result is NO_VAL).")

        for idx in sorted(needed_steps):
            if idx < 0 or idx >= len(running):
                raise ToolAgentError(
                    f"Step reference {idx} out of range (running plan length={len(running)})."
                )
            if not running[idx].is_executed():
                raise ToolAgentError(f"Referenced step {idx} is not executed (result is NO_VAL).")

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
        5. **Is prepared**: Slot must have tool ≠ NO_VAL, resolved_args ≠ NO_VAL
        6. **Tool exists**: Tool name must be registered in toolbox
        7. **Budget enforcement**: Non-return calls don't exceed tool_calls_limit

        Execution Flow
        ~~~~~~~~~~~~~~
        1. Validate all preconditions (as above)
        2. Count non-return vs. return tool calls
        3. If tool_calls_limit set, check budget
        4. Execute concurrently via ``asyncio.gather(..., return_exceptions=True)``
        5. If any gathered result is an exception, identify the first such step,
        store the error on that slot, and raise
        6. Otherwise, store results in ``slot.result``
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
                    raise raw_error

                wrapped = ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool call failed at step {idx} for {board[idx].tool!r}: {raw_error}"
                )
                board[idx].error = wrapped
                raise wrapped from raw_error

            return [(idx, result) for idx, result in zip(indices, raw_results)]

        results = asyncio.run(run_batch())

        for idx, result in results:
            board[idx].result = result

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
                raise raw_error

            wrapped = ToolAgentError(
                f"{type(self).__name__}.{self.name}: tool call failed at index {idx} "
                f"for {board[idx].tool!r}: {raw_error}"
            )
            board[idx].error = wrapped
            raise wrapped from raw_error

        for idx, result in zip(indices, raw_results):
            board[idx].result = result
            board[idx].error = NO_VAL
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
        3. **Merge into cache**: Append rewritten slots to persist them
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
        base_cache: list[BlackboardSlot] = list(state.cache_blackboard)
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
            # Only persist slots that were actually planned (tool set) and executed (result set),
            # plus allow return slot (also executed).
            if slot.tool is NO_VAL and slot.result is NO_VAL and slot.resolved_args is NO_VAL:
                continue

            new_slot = BlackboardSlot(step=base_len + local_i)
            new_slot.tool = slot.tool
            new_slot.args = rewrite_step_to_cache_placeholders(slot.args)
            new_slot.resolved_args = slot.resolved_args
            new_slot.result = slot.result
            new_slot.error = slot.error
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

        return state.return_value, {
            "blackboard_start": blackboard_start,
            "blackboard_end": blackboard_end,
        }

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

        return state.return_value, {
            "blackboard_start": blackboard_start,
            "blackboard_end": blackboard_end,
        }

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
    # String to List of Objects helper
    # ------------------------------------------------------------------ #
    def _str_to_steps(self, raw_text: str) -> list[dict[str, Any]]:
        """
        Parse LLM plan output into a list[dict].

        Robustness:
        - Strips common markdown fences (``` / ```json).
        - Scans for JSON arrays using `re` to find candidate '[' starts.
        - Uses JSON decoding (nesting-aware) to select the *largest* decodable JSON array
        that is a non-empty list of dicts.
        - Raises ToolAgentError on failure.
        """
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: LLM returned empty plan output.")

        text = raw_text.strip()

        # Strip a single fenced block wrapper if present
        # Examples:
        # ```json
        # [...]
        # ```
        text = re.sub(r"^\s*```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text).strip()

        decoder = json.JSONDecoder()

        best_val: list[dict[str, Any]] | None = None
        best_span_len: int = -1

        # Candidate starts: every '[' in the text (cheap via regex)
        for m in re.finditer(r"\[", text):
            start = m.start()
            try:
                val, end_rel = decoder.raw_decode(text[start:])  # nesting-aware decode
            except json.JSONDecodeError:
                continue

            # Must be a non-empty list of dicts
            if not isinstance(val, list) or not val:
                continue
            if not all(isinstance(x, dict) for x in val):
                continue

            if end_rel > best_span_len:
                best_span_len = end_rel
                # Snapshot as plain dicts to ensure downstream mutation safety
                best_val = [dict(x) for x in val]

        if best_val is None:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: failed to find a valid JSON array of dicts in plan output."
            )

        return best_val

    def _str_to_dict(self, raw_text: str) -> dict[str, Any]:
        """
        Parse LLM output into a single dict[str, Any].

        Robustness:
        - Strips common markdown fences (``` / ```json).
        - Scans for JSON objects using `re` to find candidate '{' starts.
        - Uses JSON decoding (nesting-aware) to select the *largest* decodable JSON object
        that is a dict.
        - Raises ToolAgentError on failure.
        """
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: LLM returned empty output.")

        text = raw_text.strip()

        # Strip a single fenced block wrapper if present
        # Examples:
        # ```json
        # {...}
        # ```
        text = re.sub(r"^\s*```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text).strip()

        decoder = json.JSONDecoder()

        best_val: dict[str, Any] | None = None
        best_span_len: int = -1

        # Candidate starts: every '{' in the text (cheap via regex)
        for m in re.finditer(r"\{", text):
            start = m.start()
            try:
                val, end_rel = decoder.raw_decode(text[start:])  # nesting-aware decode
            except json.JSONDecodeError:
                continue

            if not isinstance(val, dict):
                continue

            if end_rel > best_span_len:
                best_span_len = end_rel
                best_val = dict(val)  # snapshot for downstream mutation safety

        if best_val is None:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: failed to find a valid JSON object in LLM output."
            )

        return best_val

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
            "blackboard": self.blackboard_serialized,
        })
        return d

# --------------------------------------------------------------------------- #
# PlanAct Agent
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PlannedStep:
    """
    Normalized, immutable representation of a planned tool invocation.

    Created from LLM plan output (JSON dict) and used internally during planning.
    Frozen dataclass ensures immutability; external API uses running blackboard slots.

    Dependency Extraction
    ~~~~~~~~~~~~~~~~~~~~~
    Dependencies are automatically extracted from args during construction:
    - Scans args for ``<<__sN__>>`` placeholders; extracts all referenced step indices
    - If \"await\" field is provided, adds it as an explicit dependency barrier

    This enables automatic topological sorting and concurrent batch compilation.

    Fields
    ------
    tool : str
        Tool name (``Tool.full_name``). Must exist in agent's toolbox when executed.

    args : Any (typically dict)
        Raw, unresolved tool arguments. May contain ``<<__sN__>>`` or ``<<__cN__>>``
        placeholders. Immutable; resolution happens at execution time.

    deps : frozenset[int]
        Plan-local dependencies (step indices this step waits for). Extracted from args
        and optionally from \"await\" field. Empty if no dependencies.
        Used for topological sorting into concurrent batches.

    is_return : bool
        ``True`` iff tool is the canonical ``return_tool``. Return steps always have
        special semantics (wait for all prior steps, trigger loop exit when executed).

    Construction Methods
    ~~~~~~~~~~~~~~~~~~~~
    **from_dict(data)** (classmethod):
        Parse LLM plan step output. Accepts keys: \"tool\", \"args\", \"await\" (optional),
        \"step\" (optional, ignored).

        Example:
        >>> step_dict = {\"tool\": \"search\", \"args\": {\"q\": \"<<__s0__>>\"}, \"await\": 0}
        >>> ps = PlannedStep.from_dict(step_dict)
        >>> ps.deps  # frozenset({0})

    **to_dict()** (instance):
        Returns minimal dict: {\"tool\", \"args\"}. Deps and is_return are not preserved
        (they are computed at serialization time).
    """
    tool: str
    args: Any
    deps: frozenset[int]
    is_return: bool

    def to_dict(self) -> dict[str, Any]:
        # NOTE: await is intentionally not preserved. Deps is canonical.
        return {"tool": self.tool, "args": self.args}

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> PlannedStep:
        if not isinstance(data, Mapping):
            raise ToolAgentError(
                f"PlannedStep.from_dict requires a mapping; got {type(data).__name__!r}."
            )

        allowed = {"tool", "args", "await", "step"}
        extra = set(data.keys()) - allowed
        if extra:
            raise ToolAgentError(f"Plan step contains unsupported keys: {sorted(extra)!r}.")

        if "tool" not in data:
            raise ToolAgentError("Plan step missing required key: 'tool'.")
        if "args" not in data:
            raise ToolAgentError("Plan step missing required key: 'args'.")

        tool = data["tool"]
        if not isinstance(tool, str) or not tool.strip():
            raise ToolAgentError("Plan step 'tool' must be a non-empty string.")

        args = data["args"]
        is_return = tool == return_tool.full_name

        deps: set[int] = set(extract_dependencies(obj=args, placeholder_pattern=_STEP_TOKEN))

        # Treat await as a dependency barrier if present.
        await_val = data.get("await", None)
        if await_val is not None:
            if not isinstance(await_val, int) or await_val < 0:
                raise ToolAgentError("Plan step 'await' must be null or an int >= 0.")
            # Return step: ignore await entirely (allowed).
            if not is_return:
                deps.add(await_val)

        return PlannedStep(tool=tool, args=args, deps=frozenset(deps), is_return=is_return)


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
    def _initialize_run_state(self, *, messages: list[dict[str, str]]) -> PlanActRunState:
        """
        One-shot plan generation and compilation into concurrent batches.

        This PlanActAgent-specific initialization method performs a complete planning
        cycle upfront, generating the entire execution plan from the LLM in a single
        turn, validating it, and pre-compiling it into topologically-sorted concurrent batches.

        Execution Steps
        ~~~~~~~~~~~~~~~
        1. **Snapshot cache** from ``self._blackboard`` if ``context_enabled=True``
           (persisted results from prior invokes)

        2. **LLM plan generation** (via PLANNER_PROMPT):
           - LLM receives the conversation messages
           - Expected to emit a JSON array of step objects: ``[{"tool": "...", "args": {...}}, ...]``
           - Optionally, each step can include an "await" field (dependency barrier)

        3. **Return normalization**:
           - Validates at most one return step; if multiple, raises
           - Always moves return to the END of the plan (even if already last)
           - If no return step, auto-appends one calling ``return(None)``

        4. **Conversion to PlannedStep**:
           - Each dict is converted to a ``PlannedStep`` (normalized representation)
           - Tool name is validated to exist in toolbox (early error detection)

        5. **Return dependency forcing**:
           - The return step's deps are set to ``frozenset(range(return_idx))``
           - This ensures return always waits for all prior steps to complete

        6. **Validation**:
           - Budget check: Non-return steps don't exceed ``tool_calls_limit``
           - Cache placeholder ranges: All ``<<__cN__>>`` must reference existing cache
           - Plan-local dependency ranges: All ``<<__sN__>>`` refs must be to prior steps

        7. **Batch compilation**:
           - Topologically sorts steps by dependency level (see ``_compile_batches_from_deps``)
           - Returns list of batches: each batch indices execute concurrently
           - Returns always in its own final batch

        8. **Blackboard pre-allocation**:
           - Creates running blackboard with exactly ``len(planned)`` slots
           - Pre-fills tool and args; result/resolved_args left as NO_VAL

        Parameters
        ----------
        messages : list[dict[str, str]]
            LLM conversation history to pass to the planner

        Returns
        -------
        PlanActRunState
            Initialized state ready for the base template-method loop:
            - cache_blackboard populated with prior results when context_enabled=True
            - running_blackboard pre-allocated and pre-filled with tools+args
            - batches list with topologically-sorted concurrent batches
            - batch_index=0 (first batch)

        Raises
        ------
        ToolAgentError
            On any of:
            - Empty or invalid plan from LLM
            - Multiple return steps
            - Unknown tool references
            - Out-of-range placeholder references
            - Budget exceeded
        """
        if not messages:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: messages must be non-empty.")

        working_messages = [dict(m) for m in messages]

        cache_blackboard: list[BlackboardSlot] = self.blackboard if self.context_enabled else []
        for i, slot in enumerate(cache_blackboard):
            slot.step = i

        raw_plan = self._llm_engine.invoke({"messages": working_messages})
        plan_dicts = self._str_to_steps(raw_plan)

        # ---- Normalize return: ensure exactly one return step and it is last. ----
        return_name = return_tool.full_name
        return_positions = [i for i, d in enumerate(plan_dicts) if d.get("tool") == return_name]

        if len(return_positions) > 1:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan contains multiple return steps at positions {return_positions!r}."
            )

        if len(return_positions) == 1:
            # ALWAYS remove return and append at the end (even if already last)
            pos = return_positions[0]
            ret = plan_dicts.pop(pos)
            plan_dicts.append(ret)
        else:
            plan_dicts.append({"tool": return_name, "args": {"val": None}})

        if not plan_dicts:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: internal error: empty plan after normalization.")

        # ---- Convert to PlannedStep and validate tool existence early. ----
        planned: list[PlannedStep] = []
        for i, d in enumerate(plan_dicts):
            ps = PlannedStep.from_dict(d)
            if not self.has_tool(ps.tool):
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool in plan: {ps.tool!r}.")
            planned.append(ps)

        # Force return to be final step with deps = all prior steps.
        return_idx = len(planned) - 1
        planned[return_idx] = PlannedStep(
            tool=planned[return_idx].tool,
            args=planned[return_idx].args,
            deps=frozenset(range(return_idx)),
            is_return=True,
        )

        # ---- Enforce budget (non-return only). ----
        limit = self.tool_calls_limit
        if limit is not None:
            non_return = sum(1 for ps in planned if not ps.is_return)
            if non_return > limit:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan exceeds tool_calls_limit={limit} "
                    f"(non-return steps={non_return})."
                )

        # ---- Validate placeholder ranges and plan-local deps. ----
        cache_len = len(cache_blackboard)

        for i, ps in enumerate(planned):
            # Validate cache references in args
            cache_refs = extract_dependencies(ps.args, placeholder_pattern=_CACHE_TOKEN)
            bad_cache = [k for k in cache_refs if k < 0 or k >= cache_len]
            if bad_cache:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan step {i} references out-of-range cache indices "
                    f"{sorted(set(bad_cache))!r} (cache length={cache_len})."
                )

            bad = [d for d in ps.deps if d < 0 or d >= i]
            if bad:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan step {i} has illegal deps {sorted(set(bad))!r}; "
                    f"deps must be < {i}."
                )

        # ---- Compile batches from deps; isolate return as final batch. ----
        batches = self._compile_batches_from_deps(planned_steps=planned, return_idx=return_idx)

        # ---- Allocate running blackboard plan-locally and prefill tool+args. ----
        running_blackboard = [BlackboardSlot(step=i) for i in range(len(planned))]
        for i, ps in enumerate(planned):
            running_blackboard[i].tool = ps.tool
            running_blackboard[i].args = ps.args

        return PlanActRunState(
            messages=working_messages,
            cache_blackboard=cache_blackboard,
            running_blackboard=running_blackboard,
            executed_steps=set(),
            prepared_steps=[],
            tool_calls_used=0,
            is_done=False,
            return_value=NO_VAL,
            batches=batches,
            batch_index=0,
        )

    def _compile_batches_from_deps(self, *, planned_steps: list[PlannedStep], return_idx: int) -> list[list[int]]:
        """
        Compile concurrent batches from plan-local deps.

        For non-return step i:
          level[i] = 0 if deps empty else 1 + max(level[d] for d in deps)

        Return step is always isolated as its own final batch [return_idx].
        """
        if not planned_steps:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: cannot compile empty plan.")

        if return_idx != len(planned_steps) - 1 or not planned_steps[return_idx].is_return:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: return_idx mismatch during batch compilation."
            )

        levels: dict[int, int] = {}

        # Non-return only
        for i in range(return_idx):
            deps = planned_steps[i].deps
            if not deps:
                levels[i] = 0
            else:
                levels[i] = 1 + max(levels[d] for d in deps)

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

        PlanActAgent uses pre-compiled batches (created during initialization). This method
        simply reads the next batch indices, resolves placeholders, and populates the
        prepared_steps list.

        Execution
        ~~~~~~~~~
        1. **Validate state**: prepared_steps must be empty (batch fully executed already)
        2. **Read next batch**: Get batch indices from ``state.batches[state.batch_index]``
        3. **Validate non-empty**: Batch must have at least one step (internal check)
        4. **For each step in batch**:
           - Validate bounds: index must be within running_blackboard
           - Validate not already executed or prepared
           - Validate tool name is set
           - Call ``_resolve_placeholders(slot.args, state=state)``
           - Store resolved args in ``slot.resolved_args``
        5. **Set prepared_steps**: List of all indices in this batch (ready for execution)
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
            - prepared_steps not empty (previous batch not executed)
            - batch_index out of bounds
            - Batch validation failure (bounds, prep state, tool name)
            - Placeholder resolution failure (out-of-range or unexecuted refs)
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

        # Resolve args for all steps in the batch; resolver enforces readiness (results exist).
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

        state.prepared_steps = sorted(batch)
        state.batch_index += 1
        logger.info(f"{self.full_name}: Prepared batch {state.batch_index}/{len(state.batches)} with steps {state.prepared_steps}.")
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

    def _prepare_next_batch(self, state: ReActRunState) -> ReActRunState:
        """
        Prepare the next single-step batch.

        Semantics (single-step ReAct):
        - If state.latest_executed is non-empty, inject an assistant observation message
            describing the most recently executed step(s) including results, followed by a
            small user request for the next step.
        - Call the LLM orchestrator, which must emit exactly ONE JSON object:
            {"step": <int>, "tool": "<Tool.full_name>", "args": {...}}
        - Validate strict schema + step index correctness.
        - Validate placeholder dependencies: any <<__sN__>> must satisfy N < step.
        - Fill exactly one slot in the running_blackboard.
        - prepared_steps is a list of exactly one index.
        - next_step_index advances by 1.
        - latest_executed is overwritten with the newly prepared index (it will be
            completed by the base execute loop before the next prepare call).
        """
        if not isinstance(state, ReActRunState):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: _prepare_next_batch requires a ReActRunState."
            )

        # ------------------------------------------------------------------ #
        # 1) Build messages for this LLM turn
        # ------------------------------------------------------------------ #
        working_messages: list[dict[str, str]] = list(state.messages)

        if state.latest_executed:
            slot = state.running_blackboard[state.latest_executed[0]]
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

            # Phase B: ONLY when latest_executed is non-empty
            working_messages.append(
                {
                    "role": "user",
                    "content": "Given the most recently executed steps and available CACHE (if provided) above, "
                            "produce the NEXT single step as ONE JSON object with keys {step, tool, args}.",
                }
            )

        # Persist the augmented message history for subsequent turns.
        state.messages = working_messages

        # ------------------------------------------------------------------ #
        # 2) Call LLM and parse a single step object
        # ------------------------------------------------------------------ #
        raw_text = self._llm_engine.invoke({"messages": working_messages})
        step_obj = self._str_to_dict(raw_text)

        # Strict schema: exactly step/tool/args
        allowed = {"step", "tool", "args"}
        extra = set(step_obj.keys()) - allowed
        missing = allowed - set(step_obj.keys())
        if missing:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: step object missing required keys: {sorted(missing)!r}."
            )
        if extra:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: step object contains unsupported keys: {sorted(extra)!r}."
            )

        step_index = step_obj.get("step")
        tool = step_obj.get("tool")
        args = step_obj.get("args")

        if not isinstance(step_index, int) or step_index < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: 'step' must be an int >= 0; got {step_index!r}."
            )
        if not isinstance(tool, str) or not tool.strip():
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: 'tool' must be a non-empty string.")
        if not isinstance(args, dict):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: 'args' must be a dict; got {type(args).__name__!r}."
            )

        # Enforce step index matches the run cursor exactly
        prefix_len = state.next_step_index
        if step_index != prefix_len:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: illegal step index: got {step_index}, expected {prefix_len}."
            )

        # ------------------------------------------------------------------ #
        # 3) Validate dependency legality for this single step
        # ------------------------------------------------------------------ #
        for dep in extract_dependencies(obj=args, placeholder_pattern=_STEP_TOKEN):
            if dep >= step_index:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: illegal dependency: "
                    f"step args reference <<__s{dep}__>> but current step is {step_index}."
                )

        # ------------------------------------------------------------------ #
        # 4) Fill the next slot of the preallocated running blackboard
        # ------------------------------------------------------------------ #
        if prefix_len >= len(state.running_blackboard):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next_step_index exceeds run blackboard capacity "
                f"({prefix_len} >= {len(state.running_blackboard)})."
            )

        slot = state.running_blackboard[prefix_len]
        if not slot.is_empty():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: attempted to prepare into non-empty slot {prefix_len}."
            )

        # Validate tool exists before stamping it into the slot.
        self.get_tool(tool)

        slot.tool = tool
        slot.args = args
        slot.resolved_args = self._resolve_placeholders(args, state=state)
        slot.result = NO_VAL
        slot.error = NO_VAL

        # prepared_steps is what the base execute() will run next (list-of-one).
        state.prepared_steps = [prefix_len]

        # Used for injection at the start of the next prepare() call.
        state.latest_executed = [prefix_len]

        # Advance cursor.
        state.next_step_index = prefix_len + 1

        return state
