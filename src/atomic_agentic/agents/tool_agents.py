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
   the persisted blackboard and the completed invocation is stored as a ToolAgentRecord

The canonical memory model separates storage from rendering:

- Agent memory is stored as memory records (`AgentRecord` / `ToolAgentRecord`)
- Tool execution results are stored as blackboard slots
- A ToolAgentRecord stores the half-open blackboard span produced by one invocation
- Future LLM-facing messages are rendered from records and their associated blackboard spans

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
from collections.abc import Collection
from datetime import datetime
import logging
import re
import json
from typing import (
    Any,
    Callable,
    Mapping,
    Optional,
    List,
)
import pprint

from .base import Agent
from .prompts import PLANNER_PROMPT, ORCHESTRATOR_PROMPT
from ..models.agents.prompts import PromptConfig
from ..constants.agents import (
    ARGS_FIELD,
    AWAIT_FIELD,
    BASE_STEP_FIELDS,
    DESCRIPTION_FIELD,
    DURATION_FIELD,
    PLAN_FIELDS,
    REACT_FIELDS,
    REQUIRED_PLAN_FIELDS,
    REQUIRED_REACT_FIELDS,
    RETURN_TOOL_FULL_NAME,
    RETURN_VALUE_FIELD,
    STEP_FIELD,
    TOOL_FIELD,
)

from ..models.agents.records import AgentRecord, LLMRecord, ToolAgentRecord
from ..models.results.agents import ToolAgentResult, ToolUsageRecord
from ..models.results import LLMModelData
from ..models.agents.blackboard_models import BlackboardSlot, ConstantSpec
from ..models.agents.runstates import ToolAgentRunState, PlanActRunState, ReActRunState, ReActStepMeta
from ..exceptions import (
    AgentError,
    ToolAgentError,
    ToolDefinitionError,
    ToolInvocationError,
    ToolRegistrationError,
)
from ..constants.core import IDENTIFIER_PATTERN_TEXT
from ..core.Invokable import AtomicInvokable
from ..constants.core import NO_VAL
from ..engines.LLMEngines import LLMEngine
from ..tools import Tool, toolify
from ..mcp import MCPClientHub
from ..a2a import PyA2AtomicClient
from ..utils.agents import extract_dependencies
from ..utils.core import run_coro_sync
from .tools import return_tool


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Base ToolAgent
# --------------------------------------------------------------------------- #
class ToolAgent(Agent, ABC):
    """
    Abstract base class implementing the template-method pattern for tool-using agents.

    This class owns the invariant iteration loop; subclasses provide domain-specific
    planning and batch preparation strategies. The architecture uses a blackboard slot
    system with sentinel-driven state and placeholder-based dependency management.

    Template-Method Loop
    --------------------
    The ``_invoke(turns, prompt)`` and ``_ainvoke(turns, prompt)`` methods are
    FINAL; subclasses should not override them. They receive the selected
    canonical turns and current prompt from the base ``Agent`` lifecycle, render a
    provider-facing message list once with ``build_messages(...)``, and then run
    the existing ToolAgent template loop::

    1. messages = build_messages(role_prompt, turns, prompt)
    2. state = _initialize_run_state(messages=messages)  [subclass hook]
    3. while not state.is_done:
        state = _prepare_next_batch(state)              [subclass hook]
        if prepared_steps is empty → continue           [cascade skip: entire batch was cascade-failed]
        state = _execute_prepared_batch(state)          [base implementation]
        [completion check: if return tool executed, is_done=True]
       (each LLM generation made along the way is captured as an LLMRecord
       and accumulated onto state.llm_records)
    4. blackboard_start = len(self._blackboard)
        state = update_blackboard(state)   [always; context_enabled only gates cache_blackboard]
        blackboard_end = len(self._blackboard)
    5. return a 2-tuple of a draft ToolAgentRecord (final_result=None) carrying
       state.return_value, blackboard_start, and blackboard_end; and a metadata
       dict with llm_records, llm_model_data, tool_usage, and exception_records.

    The draft is complete except for ``final_result``, which is set after
    ``make_result`` runs — ``invoke``/``async_invoke`` complete it via
    ``dataclasses.replace(draft, final_result=agent_result)``. Future LLM-facing
    messages are rendered from the stored ``ToolAgentRecord`` by
    ``render_turn(...)``.

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

    **Context Persistence**: The completed run blackboard is always merged into
        ``self._blackboard`` (``blackboard_start``/``blackboard_end`` always set on
        the ``ToolAgentRecord``). If ``context_enabled=True``, those prior slots are
        also fed into the LLM as context on the next invocation.

    Generic Type Parameter
    ~~~~~~~~~~~~~~~~~~~~~~
    ``RS`` is a TypeVar bound to ``ToolAgentRunState``. Subclasses provide a concrete
    runtime-specific state class such as ``PlanActRunState`` or ``ReActRunState``.
    """
    TOOLS_FIELD = "TOOLS"
    LIMIT_FIELD = "TOOL_CALLS_LIMIT"
    CONSTANTS_FIELD = "CONSTANTS"

    REQUIRED_PROMPT_FIELDS = frozenset(
        {
            TOOLS_FIELD,
            LIMIT_FIELD,
            CONSTANTS_FIELD,
        }
    )

    STEP_REF_PATTERN: re.Pattern[str] = re.compile(
    r"<<__s(\d+)__>>"
    )
    CACHE_REF_PATTERN: re.Pattern[str] = re.compile(
        r"<<__c(\d+)__>>"
    )
    CONST_REF_PATTERN: re.Pattern[str] = re.compile(
        rf"<<__k\.({IDENTIFIER_PATTERN_TEXT})__>>"
    )

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        tool_instructions: str | PromptConfig,
        filter_extraneous_inputs: Optional[bool] = None,
        context_enabled: bool = False,
        *,
        fail_fast: bool = True,
        tool_calls_limit: Optional[int] = None,
        peek_at_cache: bool = False,
        response_preview_limit: Optional[int] = None,
        blackboard_preview_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_result_key: Optional[str] = None,
        prompt_key: str = "tool_instructions",
        records_window: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        name : str
            Agent identity name. Frozen at construction.
        namespace : str
            Agent identity namespace. Frozen at construction.
        description : str
            Human-readable description of this agent's purpose.
        llm_engine : LLMEngine
            Provider-facing LLM engine used for all generation calls.
        tool_instructions : str | PromptConfig
            System prompt template for tool-calling instructions. Must contain
            the named placeholders ``{TOOLS}``, ``{TOOL_CALLS_LIMIT}``, and
            ``{CONSTANTS}``; may include additional simple named placeholders.
            Accepts a raw str (wrapped into a PromptConfig inline) or a
            pre-built PromptConfig. Validated and frozen at construction.
        filter_extraneous_inputs : bool | None
            When ``True``, inputs not declared in ``parameters`` are silently
            dropped before invocation. When ``False``, extraneous inputs raise.
            ``None`` inherits the base class default.
        context_enabled : bool
            When ``True``, prior blackboard steps are fed into each invocation
            as LLM context (``cache_blackboard`` is populated in run state).
            The blackboard is always persisted after each invoke regardless of
            this setting. Defaults to ``False``.
        fail_fast : bool
            When ``True`` (default), the first tool call failure immediately
            raises and aborts the run. When ``False``, failing slots are marked
            ``FAILED`` and the loop continues to execute independent steps;
            failures are collected in ``ToolAgentResult.exception_records``.
            Return-tool failures always raise regardless of this setting.
        tool_calls_limit : int | None
            Maximum number of non-return action calls per invoke run.
            ``None`` means unlimited. Must be ``>= 0`` if set.
        peek_at_cache : bool
            When ``True``, the persisted blackboard is rendered with raw
            result and resolved-args fields exposed (via
            ``blackboard_serialized(peek=True)``). Defaults to ``False``.
        response_preview_limit : int | None
            Character limit for assistant response previews in rendered turns.
            ``None`` means no truncation.
        blackboard_preview_limit : int | None
            Character limit for cached blackboard result previews. ``None``
            means no truncation.
        pre_invoke : AtomicInvokable | Callable | None
            Optional hook invoked before the main agent loop. Receives the
            same inputs as the agent.
        post_invoke : AtomicInvokable | Callable | None
            Optional hook invoked after the main agent loop. Receives the
            agent result.
        post_result_key : str | None
            Key under which the agent result is passed to ``post_invoke``
            when ``post_invoke`` is set.
        prompt_key : str
            Key used to store the tool instructions PromptConfig in
            ``_system_prompts``. Defaults to ``"tool_instructions"``.
        records_window : int | None
            Maximum number of prior ``AgentRecord`` turns rendered into LLM
            context. ``None`` means all records are rendered.
        """
        if isinstance(tool_instructions, str):
            if not tool_instructions.strip():
                raise ToolAgentError(
                    "ToolAgent tool_instructions must be a non-empty str template."
                )
            try:
                config = PromptConfig(
                    template=tool_instructions.strip(),
                    description="ToolAgent system instructions",
                )
            except (TypeError, ValueError) as exc:
                raise ToolAgentError(
                    f"Invalid ToolAgent tool_instructions template: {exc}"
                ) from exc
        elif isinstance(tool_instructions, PromptConfig):
            config = tool_instructions
        else:
            raise ToolAgentError(
                f"ToolAgent tool_instructions must be a str or PromptConfig; "
                f"got {type(tool_instructions).__name__!r}."
            )
        self._validate_tool_prompt_template(config)

        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            filter_extraneous_inputs=filter_extraneous_inputs,
            context_enabled=context_enabled,
            context_keys=None,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            records_window=records_window,
            response_preview_limit=response_preview_limit,
        )

        self._toolbox: dict[str, AtomicInvokable] = {}
        self._blackboard: list[BlackboardSlot] = []
        self._constants: list[ConstantSpec] = []

        if not isinstance(fail_fast, bool):
            raise ToolAgentError("fail_fast must be a bool.")
        self._fail_fast: bool = fail_fast

        if type(peek_at_cache) is not bool:
            raise ToolAgentError("peek_at_cache must be a boolean.")
        self._peek_at_cache = peek_at_cache

        if blackboard_preview_limit is None:
            self._blackboard_preview_limit = None
        elif type(blackboard_preview_limit) is not int or blackboard_preview_limit <= 0:
            raise ToolAgentError("blackboard_preview_limit must be None or a positive integer > 0.")
        else:
            self._blackboard_preview_limit = blackboard_preview_limit

        self._tool_calls_limit: Optional[int] = None
        self.tool_calls_limit = tool_calls_limit

        # Always include canonical return tool (avoid collisions by skipping).
        self.register(return_tool, name_collision_mode="skip")

        self._tool_prompt_key: str = prompt_key
        self._system_prompts[prompt_key] = config

    # ------------------------------------------------------------------ #
    # Agent Properties
    # ------------------------------------------------------------------ #
    @property
    def tool_instructions(self) -> str:
        """Tool instructions template string. Read-only."""
        return self._system_prompts[self._tool_prompt_key].template

    def update_prompt(self, key: str, config: PromptConfig) -> None:
        """Register or replace a system prompt. Raises AgentError if key matches the
        tool instructions key."""
        if isinstance(key, str) and key.strip() == self._tool_prompt_key:
            raise AgentError(
                f"{self._tool_prompt_key!r} is immutable on ToolAgent; "
                "construct a new ToolAgent to change the tool instructions."
            )
        super().update_prompt(key, config)

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

        ``peek=False`` (default): hides ``result`` and ``resolved_args`` fields.
        ``peek=True``: includes those fields; the ``result`` value is rendered
        through ``_preview_blackboard_result`` so ``blackboard_preview_limit``
        applies consistently with the ReAct observable rendering path.
        """
        if peek:
            result = []
            for slot in self._blackboard:
                d = slot.to_dict()
                if slot.is_executed():
                    d[BlackboardSlot.RESULT_FIELD] = self._preview_blackboard_result(
                        slot.result.result
                    )
                    d["run_id"] = slot.result.run_id
                result.append(d)
            return result
        else:
            result = []
            for slot in self._blackboard:
                d = slot.to_dict()
                d.pop(BlackboardSlot.RESOLVED_ARGS_FIELD)
                d.pop(BlackboardSlot.RESULT_FIELD)
                if slot.is_executed():
                    d["run_id"] = slot.result.run_id
                result.append(d)
            return result
    
    @property
    def blackboard(self) -> list[BlackboardSlot]:
        """Shallow copy of the persisted blackboard slots. Mutations to the
        returned list or its slots do not affect internal agent state."""
        return [slot.copy() for slot in self._blackboard]

    @property
    def fail_fast(self) -> bool:
        """When False, individual tool call failures are recorded rather than raised."""
        return self._fail_fast

    @property
    def peek_at_cache(self) -> bool:
        """Whether ``blackboard_serialized(peek=True)`` is used when building
        LLM context, exposing raw result and resolved-args fields."""
        return self._peek_at_cache

    @property
    def blackboard_preview_limit(self) -> Optional[int]:
        """Character limit for cached blackboard result previews. None means no truncation."""
        return self._blackboard_preview_limit

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
    def clear_memory(self) -> None:
        """Clear the stored turn history and the persisted blackboard."""
        super().clear_memory()
        self._blackboard.clear()
    # ------------------------------------------------------------------ #
    # Prompt Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_tool_prompt_template(config: PromptConfig) -> None:
        """Verify that config.parameters contains all required tool prompt fields."""
        param_names = {p.name for p in config.parameters}
        missing = ToolAgent.REQUIRED_PROMPT_FIELDS - param_names
        if missing:
            raise ToolAgentError(
                f"ToolAgent tool_instructions template missing required placeholder(s): "
                f"{', '.join(sorted(missing))}."
            )

    def _build_context(self, inputs: dict) -> tuple[dict, dict]:
        """Extend base context with tool-instruction render inputs from instance state."""
        context, remaining = super()._build_context(inputs)
        limit_text = (
            "unlimited" if self._tool_calls_limit is None else str(self._tool_calls_limit)
        )
        context[self.TOOLS_FIELD] = self.actions_context()
        context[self.LIMIT_FIELD] = limit_text
        context[self.CONSTANTS_FIELD] = self.constants_context()
        return context, remaining

    # ------------------------------------------------------------------ #
    # Toolbox Helpers
    # ------------------------------------------------------------------ #
    def actions_context(self) -> str:
        """String representation of all tools in the toolbox for prompt injection."""
        tools = list(self._toolbox.values())
        return "\n".join(f"-- {t}" for t in tools)

    def list_tools(self) -> dict[str, AtomicInvokable]:
        """Return a shallow copy of the toolbox mapping ``full_name → AtomicInvokable``."""
        return dict(self._toolbox)

    def has_tool(self, tool_full_name: str) -> bool:
        """Return ``True`` if a tool with the given ``full_name`` is registered."""
        return tool_full_name in self._toolbox

    def get_tool(self, tool_full_name: str) -> AtomicInvokable:
        """Return the registered invokable with the given ``full_name``.

        Raises
        ------
        ToolAgentError
            If no tool with ``tool_full_name`` is registered.
        """
        tool = self._toolbox.get(tool_full_name)
        if tool is None:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool {tool_full_name!r}.")
        return tool

    def remove_tool(self, tool_full_name: str) -> bool:
        """Remove the tool with the given ``full_name`` from the toolbox.

        Returns
        -------
        bool
            ``True`` if the tool was present and removed; ``False`` if it was
            not registered.
        """
        return self._toolbox.pop(tool_full_name, None) is not None

    def clear_tools(self) -> None:
        """Remove all registered tools from the toolbox."""
        self._toolbox.clear()

    # ------------------------------------------------------------------ #
    # Constants Helpers
    # ------------------------------------------------------------------ #
    @property
    def constants(self) -> list[ConstantSpec]:
        """Return a shallow copy of registered ToolAgent constants."""
        return list(self._constants)

    def register_constant(
        self,
        name: str,
        value: Any,
        description: str | None = None,
        inline_limit: int | None = None,
    ) -> str:
        """
        Register one named runtime constant on this ToolAgent.

        Constants are stored separately from tools. They are not executable and
        do not participate in tool registration, tool-call budgeting, or
        blackboard persistence.

        Parameters
        ----------
        name : str
            Constant name. Must be identifier-like: letters/underscore first,
            then letters/numbers/underscore.
        value : Any
            Runtime value to bind to the constant.
        description : str | None
            Optional human-readable description.
        inline_limit : int | None
            Optional character limit for future inline string substitution.

        Returns
        -------
        str
            The normalized registered constant name.

        Raises
        ------
        ToolAgentError
            If the spec is invalid or the name is already registered.
        """
        try:
            spec = ConstantSpec(
                name=name,
                value=value,
                description=description,
                inline_limit=inline_limit,
            )
        except Exception as exc:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: invalid constant spec: {exc}"
            ) from exc

        if any(existing.name == spec.name for existing in self._constants):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: constant already registered: {spec.name!r}."
            )

        self._constants.append(spec)
        return spec.name

    def batch_register_constants(
        self,
        **constants: tuple[Any, ...],
    ) -> list[str]:
        """
        Register constants from keyword arguments.

        Each keyword name becomes the constant name. Each keyword value must be
        a tuple with one, two, or three items:

        - ``NAME=(value,)``
        - ``NAME=(value, description)``
        - ``NAME=(value, description, inline_limit)``

        Tuple-valued constants must be wrapped as the first item of a one-item
        tuple, for example: ``COORDS=((1, 2),)``.

        This method validates the whole batch before mutating ``self._constants``.

        Returns
        -------
        list[str]
            Registered constant names in keyword insertion order.

        Raises
        ------
        ToolAgentError
            If any item is malformed, duplicated in the batch, or already
            registered on this ToolAgent.
        """
        if not constants:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: batch_register_constants expects at least one constant."
            )

        existing_names = {spec.name for spec in self._constants}
        candidate_names: set[str] = set()
        candidates: list[ConstantSpec] = []

        for raw_name, payload in constants.items():
            if not isinstance(payload, tuple):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: constant {raw_name!r} must be provided as a tuple "
                    "of (value,), (value, description), or (value, description, inline_limit)."
                )

            if len(payload) not in {1, 2, 3}:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: constant {raw_name!r} tuple must have length 1, 2, or 3; "
                    f"got {len(payload)}."
                )

            value = payload[0]
            description = payload[1] if len(payload) >= 2 else None
            inline_limit = payload[2] if len(payload) == 3 else None

            try:
                spec = ConstantSpec(
                    name=raw_name,
                    value=value,
                    description=description,
                    inline_limit=inline_limit,
                )
            except Exception as exc:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: invalid constant spec for {raw_name!r}: {exc}"
                ) from exc

            if spec.name in candidate_names:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: duplicate constant name in batch: {spec.name!r}."
                )

            if spec.name in existing_names:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: constant already registered: {spec.name!r}."
                )

            candidate_names.add(spec.name)
            candidates.append(spec)

        self._constants.extend(candidates)
        return [spec.name for spec in candidates]

    def has_constant(self, name: str) -> bool:
        """Return whether a constant with the given name is registered."""
        if not isinstance(name, str) or not name.strip():
            return False

        normalized_name = name.strip()
        return any(spec.name == normalized_name for spec in self._constants)

    def get_constant(self, name: str) -> ConstantSpec:
        """
        Return the registered constant with the given name.

        Raises ``ToolAgentError`` if no constant with that name exists.
        """
        if not isinstance(name, str) or not name.strip():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: constant name must be a non-empty string."
            )

        normalized_name = name.strip()
        for spec in self._constants:
            if spec.name == normalized_name:
                return spec

        raise ToolAgentError(
            f"{type(self).__name__}.{self.name}: unknown constant {normalized_name!r}."
        )

    def remove_constant(self, name: str) -> bool:
        """Remove a registered constant by name. Returns True if removed."""
        if not isinstance(name, str) or not name.strip():
            return False

        normalized_name = name.strip()
        for index, spec in enumerate(self._constants):
            if spec.name == normalized_name:
                del self._constants[index]
                return True

        return False

    def clear_constants(self) -> None:
        """Remove all registered constants from this ToolAgent."""
        self._constants.clear()

    def constants_context(self) -> str:
        """
        Render the dynamic constants list for future prompt injection.

        This method intentionally renders only the list body, not a section
        header or explanatory text. It also intentionally does not render raw
        constant values.
        """
        if not self._constants:
            return "No constants registered."

        rendered: list[str] = []
        for spec in self._constants:
            description = (
                spec.description
                if spec.description is not None
                else "No description provided."
            )
            rendered.append(
                f"- {spec.name}\n"
                f"  Type: {spec.type}\n"
                f"  Description: {description}"
            )

        return "\n\n".join(rendered)

    def register(
        self,
        component: AtomicInvokable | Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        *,
        name_collision_mode: str = "raise",
    ) -> str:
        """Register one invokable on this ToolAgent.

        AtomicInvokables are stored directly under their own ``full_name``;
        ``name`` and ``description`` overrides are not permitted for this route.
        Plain callables are normalized through ``toolify`` with this agent's
        ``name`` as their namespace; ``name`` and ``description`` may override
        the inferred values.

        Parameters
        ----------
        component : AtomicInvokable | Callable
            The item to register. AtomicInvokables are stored as-is; callables
            are wrapped via ``toolify(namespace=self.name)``.
        name : str | None
            Override the tool name (callable route only). ``None`` infers from
            ``component.__name__``.
        description : str | None
            Override the tool description (callable route only). ``None``
            infers from ``component.__doc__``.
        name_collision_mode : str
            Controls behavior when the resolved ``full_name`` is already
            registered. One of ``"raise"`` (default), ``"skip"``, or
            ``"replace"``.

        Returns
        -------
        str
            The registered invokable's ``full_name`` (``"Type.namespace.name"``).

        Raises
        ------
        ToolRegistrationError
            If ``name_collision_mode`` is invalid; if ``name`` or
            ``description`` overrides are supplied for an ``AtomicInvokable``
            component; if ``toolify`` fails; or if a collision is detected
            under ``"raise"`` mode.
        """
        name_collision_mode = name_collision_mode.lower().strip()
        if name_collision_mode not in ("raise", "skip", "replace"):
            raise ToolRegistrationError(
                "name_collision_mode must be one of: 'raise', 'skip', 'replace'."
            )

        # AtomicInvokable route — store directly under its own identity
        if isinstance(component, AtomicInvokable):
            if name is not None or description is not None:
                raise ToolRegistrationError(
                    f"{type(self).__name__}.{self.name}: name and description "
                    "overrides are not supported when registering an AtomicInvokable "
                    "directly. Use the component's own identity."
                )
            key = component.full_name
            invokable = component

        # Callable route — normalize via toolify with self.name as namespace
        elif callable(component):
            try:
                invokable = toolify(
                    component=component,
                    name=name or component.__name__,
                    description=description or component.__doc__,
                    namespace=self.name,
                )
            except Exception as e:
                raise ToolRegistrationError(
                    f"{type(self).__name__}.{self.name}: failed to toolify component: {e}"
                ) from e
            key = invokable.full_name

        else:
            raise ToolRegistrationError(
                f"{type(self).__name__}.{self.name}: unsupported component type "
                f"{type(component).__name__!r}. Expected AtomicInvokable or Callable."
            )

        if key in self._toolbox:
            if name_collision_mode == "raise":
                raise ToolRegistrationError(
                    f"{type(self).__name__}.{self.name}: tool already registered: {key}"
                )
            if name_collision_mode == "skip":
                return key

        self._toolbox[key] = invokable
        return key

    def batch_register(
        self,
        tools: list[AtomicInvokable | Callable] | None = None,
        client: PyA2AtomicClient | MCPClientHub | None = None,
        *,
        remote_names: list[str] | None = None,
        name_collision_mode: str = "raise",
        batch_filter_inputs: Optional[bool] = None,
    ) -> list[str]:
        """Register a batch of invokables on this ToolAgent.

        Accepts a local list, a remote client, or both. All items are expanded
        into ``(full_name, invokable)`` pairs before any toolbox mutation;
        duplicate full_names within the incoming batch always raise regardless
        of ``name_collision_mode``.

        Parameters
        ----------
        tools : list[AtomicInvokable | Callable] | None
            Local items to register. AtomicInvokables are stored as-is;
            callables are normalized via ``toolify(namespace=self.name)``.
        client : PyA2AtomicClient | MCPClientHub | None
            Remote client to enumerate and register tools from. Combined with
            ``tools`` in one registration pass when both are provided.
        remote_names : list[str] | None
            Whitelist of remote tool names to register from ``client``.
            ``None`` registers all available remote tools. Requires ``client``.
        name_collision_mode : str
            Per-item collision policy for toolbox conflicts. One of
            ``"raise"`` (default), ``"skip"``, or ``"replace"``. Does not
            affect intra-batch dedup, which always raises.
        batch_filter_inputs : bool | None
            ``filter_extraneous_inputs`` override applied uniformly to all
            callables and remote tools toolified in this batch. ``None``
            inherits each tool's own default.

        Returns
        -------
        list[str]
            ``full_name`` of every invokable newly registered. Skipped items
            (under ``"skip"`` mode) are excluded.

        Raises
        ------
        ValueError
            If both ``tools`` and ``client`` are ``None``; if ``tools`` is
            empty and no ``client`` is provided; if ``remote_names`` is
            supplied without a ``client``; or if ``remote_names`` is an empty
            list when a ``client`` is provided.
        ToolRegistrationError
            If ``name_collision_mode`` is invalid; if a duplicate full_name
            appears in the incoming batch; if toolification of any item fails;
            or if a toolbox collision is detected under ``"raise"`` mode.
        """
        name_collision_mode = name_collision_mode.lower().strip()
        if name_collision_mode not in ("raise", "skip", "replace"):
            raise ToolRegistrationError(
                "name_collision_mode must be one of: 'raise', 'skip', 'replace'."
            )

        # Validate argument combinations before any expansion
        if tools is None and client is None:
            raise ValueError(
                f"{type(self).__name__}.batch_register requires at least one of: "
                "tools list or client."
            )
        if tools is not None and len(tools) == 0 and client is None:
            raise ValueError(
                f"{type(self).__name__}.batch_register: tools list is empty and no "
                "client provided."
            )
        if remote_names is not None and client is None:
            raise ValueError(
                f"{type(self).__name__}.batch_register: remote_names requires a client."
            )
        if client is not None and remote_names is not None and len(remote_names) == 0:
            raise ValueError(
                f"{type(self).__name__}.batch_register: remote_names is an empty list; "
                "nothing to register from client."
            )

        # Expand all sources into (full_name, invokable) pairs
        combined: list[tuple[str, AtomicInvokable]] = []

        if tools is not None:
            for item in tools:
                if isinstance(item, AtomicInvokable):
                    combined.append((item.full_name, item))
                elif callable(item):
                    try:
                        t = toolify(
                            component=item,
                            namespace=self.name,
                            filter_extraneous_inputs=batch_filter_inputs,
                        )
                    except Exception as exc:
                        raise ToolRegistrationError(
                            f"{type(self).__name__}.{self.name}: failed to toolify "
                            f"{item!r}: {exc}"
                        ) from exc
                    combined.append((t.full_name, t))
                else:
                    raise ToolRegistrationError(
                        f"{type(self).__name__}.{self.name}: unsupported item type "
                        f"{type(item).__name__!r} in tools list."
                    )

        if client is not None:
            if isinstance(client, MCPClientHub):
                available = client.list_tools()
            else:
                available = client.list_invokables()

            names_to_register = (
                [n for n in available if n in remote_names]
                if remote_names is not None
                else available
            )

            for remote_name in names_to_register:
                try:
                    proxy = toolify(
                        component=client,
                        namespace=self.name,
                        remote_name=remote_name,
                        filter_extraneous_inputs=batch_filter_inputs,
                    )
                except Exception as exc:
                    raise ToolRegistrationError(
                        f"{type(self).__name__}.{self.name}: failed to toolify remote "
                        f"{remote_name!r}: {exc}"
                    ) from exc
                combined.append((proxy.full_name, proxy))

        # Intra-set dedup — always raise regardless of name_collision_mode
        seen: set[str] = set()
        for key, _ in combined:
            if key in seen:
                raise ToolRegistrationError(
                    f"{type(self).__name__}.{self.name}: duplicate full_name in "
                    f"incoming batch: {key!r}."
                )
            seen.add(key)

        # Register against toolbox — apply name_collision_mode per item
        registered: list[str] = []
        for key, invokable in combined:
            if key in self._toolbox:
                if name_collision_mode == "raise":
                    raise ToolRegistrationError(
                        f"{type(self).__name__}.{self.name}: already registered: {key}"
                    )
                if name_collision_mode == "skip":
                    continue
            self._toolbox[key] = invokable
            registered.append(key)

        return registered

    # ------------------------------------------------------------------ #
    # Placeholder resolution helpers (prepare-time)
    # ------------------------------------------------------------------ #
    def _resolve_placeholders(self, obj: Any, *, state: ToolAgentRunState) -> Any:
        """
        Resolve all placeholders in an object to their concrete values.

        This method recursively traverses the object structure and replaces placeholder
        references with their concrete runtime values. Placeholders can reference:

        - previously executed current-run steps,
        - previously persisted cache entries,
        - registered ToolAgent constants.

        Supported Placeholder Formats
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        - ``<<__sN__>>`` – Result from running step N (0-based, plan-local to this invoke)
        - ``<<__cN__>>`` – Result from cache entry N (0-based, from persisted blackboard)
        - ``<<__k.NAME__>>`` – Registered ToolAgent constant named NAME

        Two resolution modes apply depending on placeholder position:

        1. **Full-String Placeholder**:
           - Returns the referenced value as-is, preserving its type.
           - Examples:
             - ``"<<__s0__>>"`` returns step 0's result directly.
             - ``"<<__c0__>>"`` returns cache 0's result directly.
             - ``"<<__k.PI__>>"`` returns the registered constant value directly.

        2. **Inline Placeholder**:
           - Replaces the placeholder with ``repr(value)``, falling back to ``str(value)``.
           - For constants only, applies ``ConstantSpec.inline_limit`` if configured.
           - Example:
             - ``"Step returned: <<__s0__>>"`` becomes a string.
             - ``"Use constant: <<__k.NAME__>>"`` becomes a string, possibly truncated
               for that constant's inline representation.

        Readiness Validation
        ~~~~~~~~~~~~~~~~~~~~
        Before resolution, validates that all referenced step/cache slots are marked
        executed and that all referenced constants are registered.

        Parameters
        ----------
        obj : Any
            Object to resolve. Can be nested lists, tuples, sets, dicts, strings,
            or scalar values.
        state : ToolAgentRunState
            Execution state containing cache_blackboard and running_blackboard.

        Returns
        -------
        Any
            Resolved object with all placeholders replaced. Structure is preserved;
            only placeholder tokens are replaced. Step/cache placeholders always
            resolve to the unwrapped ``AtomicResult.result`` payload of the
            referenced slot — never the envelope itself — since readiness
            validation (above) guarantees those slots are executed before
            substitution runs.

        Raises
        ------
        ToolAgentError
            If a referenced step/cache placeholder is out of bounds or unexecuted,
            or if a referenced constant is not registered.
        """
        cache = state.cache_blackboard
        running = state.running_blackboard
        constants_by_name: dict[str, ConstantSpec] = {
            spec.name: spec
            for spec in self._constants
        }

        needed_cache: set[int] = set(
            extract_dependencies(obj, placeholder_pattern=self.CACHE_REF_PATTERN)
        )
        needed_steps: set[int] = set(
            extract_dependencies(obj, placeholder_pattern=self.STEP_REF_PATTERN)
        )
        needed_constants: set[str] = set()

        def collect_constant_refs(x: Any) -> None:
            if isinstance(x, str):
                for match in self.CONST_REF_PATTERN.finditer(x):
                    needed_constants.add(match.group(1))
                return
            if isinstance(x, dict):
                for key, value in x.items():
                    collect_constant_refs(key)
                    collect_constant_refs(value)
                return
            if isinstance(x, (list, tuple, set)):
                for value in x:
                    collect_constant_refs(value)
                return

        collect_constant_refs(obj)

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

        for name in sorted(needed_constants):
            if name not in constants_by_name:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: unknown constant reference {name!r}."
                )

        # ----------------------------
        # 3) Resolve recursively.
        # ----------------------------
        def render_inline(value: Any, *, inline_limit: int | None = None) -> str:
            try:
                text = repr(value)
            except Exception:
                text = str(value)

            if inline_limit is not None and len(text) > inline_limit:
                text = text[:inline_limit]

            return text

        def resolve_str(s: str) -> Any:
            # Exact placeholder -> preserve type
            m_cache = self.CACHE_REF_PATTERN.fullmatch(s)
            if m_cache:
                return cache[int(m_cache.group(1))].result.result

            m_step = self.STEP_REF_PATTERN.fullmatch(s)
            if m_step:
                return running[int(m_step.group(1))].result.result

            m_constant = self.CONST_REF_PATTERN.fullmatch(s)
            if m_constant:
                return constants_by_name[m_constant.group(1)].value

            # Inline substitution
            def repl_cache(m: re.Match[str]) -> str:
                idx = int(m.group(1))
                return render_inline(cache[idx].result.result)

            def repl_step(m: re.Match[str]) -> str:
                idx = int(m.group(1))
                return render_inline(running[idx].result.result)

            def repl_constant(m: re.Match[str]) -> str:
                spec = constants_by_name[m.group(1)]
                return render_inline(
                    spec.value,
                    inline_limit=spec.inline_limit,
                )

            out = self.CACHE_REF_PATTERN.sub(repl_cache, s)
            out = self.STEP_REF_PATTERN.sub(repl_step, out)
            out = self.CONST_REF_PATTERN.sub(repl_constant, out)
            return out

        def resolve(x: Any) -> Any:
            if isinstance(x, str):
                return resolve_str(x)
            if isinstance(x, list):
                return [resolve(v) for v in x]
            if isinstance(x, tuple):
                return tuple(resolve(v) for v in x)
            if isinstance(x, set):
                return set([resolve(v) for v in x])
            if isinstance(x, dict):
                return {resolve(k): resolve(v) for k, v in x.items()}
            return x

        return resolve(obj)
    # ------------------------------------------------------------------ #
    # Execution (base-owned)
    # ------------------------------------------------------------------ #
    def _execute_prepared_batch(self, state: ToolAgentRunState) -> ToolAgentRunState:
        """
        Execute all steps in the currently prepared batch concurrently.

        This is a core base-owned method (do not override). It executes all steps in
        ``state.prepared_steps`` using the tool async-invoke path, records results in
        the running blackboard, and handles termination if the return tool is executed.

        Batch Semantics
        ~~~~~~~~~~~~~~~
        - All steps in ``prepared_steps`` are **concurrent**
        - Multi-step batches use ``asyncio.gather(..., return_exceptions=True)``
        under a single ``run_coro_sync(...)``
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
        6. Otherwise, store each ``ToolResult`` envelope whole in ``slot.result``
        and mark slots executed
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
        - ``state.running_blackboard[idx].result`` is set to the whole
          ``ToolResult`` envelope (an ``AtomicResult``) for executed steps —
          preserved for richer tracing. Consumers that need the caller-facing
          value (placeholder resolution, previews, ``return_value``) read
          ``result.result``; those sites are always reached only after the
          slot is confirmed executed
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

            if tool_name == RETURN_TOOL_FULL_NAME:
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
            return list(zip(indices, raw_results))

        pairs = run_coro_sync(run_batch())

        failures = [(idx, raw) for idx, raw in pairs if isinstance(raw, BaseException)]
        successes = [(idx, raw) for idx, raw in pairs if not isinstance(raw, BaseException)]

        if failures:
            if self._fail_fast:
                # Existing behavior: mark first failure and raise immediately.
                idx, raw_error = failures[0]
                if isinstance(raw_error, ToolInvocationError):
                    board[idx].error = raw_error
                    board[idx].status = BlackboardSlot.FAILED
                    raise raw_error
                wrapped = ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool call failed at step {idx} "
                    f"for {board[idx].tool!r}: {raw_error}"
                )
                board[idx].error = wrapped
                board[idx].status = BlackboardSlot.FAILED
                raise wrapped from raw_error
            else:
                # fail_fast=False: mark all failures, then check for fatal return-tool failure.
                for idx, raw_error in failures:
                    if isinstance(raw_error, ToolInvocationError):
                        board[idx].error = raw_error
                    else:
                        board[idx].error = ToolAgentError(
                            f"{type(self).__name__}.{self.name}: tool call failed at step {idx} "
                            f"for {board[idx].tool!r}: {raw_error}"
                        )
                    board[idx].status = BlackboardSlot.FAILED

                # Return-tool failure is always fatal regardless of fail_fast.
                for idx, raw_error in failures:
                    if board[idx].tool == RETURN_TOOL_FULL_NAME:
                        err = board[idx].error
                        if isinstance(raw_error, ToolInvocationError):
                            raise err
                        raise err from raw_error

        for idx, tool_result in successes:
            board[idx].result = tool_result
            board[idx].error = NO_VAL
            board[idx].status = BlackboardSlot.EXECUTED
            state.executed_steps.add(idx)

        state.tool_calls_used += non_return_planned
        state.prepared_steps = []

        if return_indices:
            ret_idx = return_indices[0]
            if board[ret_idx].is_executed():
                state.return_value = board[ret_idx].result.result
                state.is_done = True
            # else: return slot failed; already raised above in fail_fast=False path

        return state

    async def _async_execute_prepared_batch(self, state: ToolAgentRunState) -> ToolAgentRunState:
        """
        Async analog of ``_execute_prepared_batch(...)``.

        Executes all currently prepared steps concurrently using each tool's
        ``async_invoke(...)`` path, stores each ``ToolResult`` envelope whole
        in the running blackboard, and updates return/completion bookkeeping.

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

            if tool_name == RETURN_TOOL_FULL_NAME:
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

        failures = [
            (idx, raw)
            for idx, raw in zip(indices, raw_results)
            if isinstance(raw, BaseException)
        ]
        successes = [
            (idx, raw)
            for idx, raw in zip(indices, raw_results)
            if not isinstance(raw, BaseException)
        ]

        if failures:
            if self._fail_fast:
                # Existing behavior: mark first failure and raise immediately.
                idx, raw_error = failures[0]
                if isinstance(raw_error, ToolInvocationError):
                    board[idx].error = raw_error
                    board[idx].status = BlackboardSlot.FAILED
                    raise raw_error
                wrapped = ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool call failed at index {idx} "
                    f"for {board[idx].tool!r}: {raw_error}"
                )
                board[idx].error = wrapped
                board[idx].status = BlackboardSlot.FAILED
                raise wrapped from raw_error
            else:
                # fail_fast=False: mark all failures, then check for fatal return-tool failure.
                for idx, raw_error in failures:
                    if isinstance(raw_error, ToolInvocationError):
                        board[idx].error = raw_error
                    else:
                        board[idx].error = ToolAgentError(
                            f"{type(self).__name__}.{self.name}: tool call failed at index {idx} "
                            f"for {board[idx].tool!r}: {raw_error}"
                        )
                    board[idx].status = BlackboardSlot.FAILED

                # Return-tool failure is always fatal regardless of fail_fast.
                for idx, raw_error in failures:
                    if board[idx].tool == RETURN_TOOL_FULL_NAME:
                        err = board[idx].error
                        if isinstance(raw_error, ToolInvocationError):
                            raise err
                        raise err from raw_error

        for idx, tool_result in successes:
            board[idx].result = tool_result
            board[idx].error = NO_VAL
            board[idx].status = BlackboardSlot.EXECUTED
            state.executed_steps.add(idx)

        state.tool_calls_used += non_return_planned
        state.prepared_steps = []

        if return_indices:
            ret_idx = return_indices[0]
            if board[ret_idx].is_executed():
                state.return_value = board[ret_idx].result.result
                state.is_done = True
            # else: return slot failed; already raised above in fail_fast=False path

        return state

    # ------------------------------------------------------------------ #
    # Finalization helpers
    # ------------------------------------------------------------------ #
    def update_blackboard(self, state: ToolAgentRunState) -> ToolAgentRunState:
        """
        Persist all non-empty run slots into the agent's persisted blackboard.

        Called unconditionally at the end of ``_invoke()`` and ``_ainvoke()`` —
        whether or not ``context_enabled`` is set. This mirrors how base ``Agent``
        always appends ``_records`` regardless of context settings. The
        ``context_enabled`` flag controls only whether ``cache_blackboard`` is
        populated in ``_initialize_run_state`` (i.e. whether the LLM sees prior
        steps as context).

        All non-empty slots (EXECUTED and FAILED) are persisted so that global
        blackboard indices remain contiguous and correct. FAILED slots are
        included for index continuity; ``render_turn`` controls whether they
        are surfaced to the LLM.

        Persistence Policy
        ~~~~~~~~~~~~~~~~~~
        1. **Trim empty/unplanned tail**: Remove trailing empty slots from running blackboard
           (slots with no tool assigned)
        2. **Rewrite placeholders**: All ``<<__sN__>>`` step references in appended slots'
           args are rewritten to ``<<__c{new_global_index}__>>`` cache references.
           Applied to both EXECUTED and FAILED slots.
        3. **Merge into cache**: Append all non-empty slots (EXECUTED and FAILED)
           preserving status. FAILED slots keep their ``error`` and no ``result``.
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

                return self.STEP_REF_PATTERN.sub(repl, obj)

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

        # 2) Append all non-empty running slots with rewritten placeholders and global indices.
        #    FAILED slots are included so local_i always equals the append offset (B3 fix).
        appended: list[BlackboardSlot] = []
        for local_i, slot in enumerate(running):
            if slot.is_empty():
                continue

            new_slot = BlackboardSlot(
                step=base_len + local_i,
                tool=slot.tool,
                args=rewrite_step_to_cache_placeholders(slot.args),
                resolved_args=slot.resolved_args,
                result=slot.result,
                error=slot.error,
                status=slot.status,
                step_dependencies=slot.step_dependencies,
                await_step=slot.await_step,
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
    def _invoke(self, turns: List[AgentRecord], prompt: str, context: dict) -> tuple[ToolAgentRecord, dict]:
        """
        FINAL sync ToolAgent template method.

        Receives selected canonical turns, the current prompt, and the assembled
        context dict from the base ``Agent.invoke(...)`` lifecycle. Renders the
        system prompt from context, builds the message list once, then runs the
        ToolAgent template loop. Returns a 2-tuple of a **draft** ``ToolAgentRecord``
        (``final_result`` is ``None``) and a metadata dict carrying ``llm_records``,
        ``llm_model_data``, and ``tool_usage``.

        The LLMRecord envelopes are accumulated on ``state.llm_records`` across
        the planning loop and transferred to the metadata dict at return time.
        ``invoke`` later completes the draft via
        ``dataclasses.replace(draft, final_result=agent_result)``.

        Subclasses should not override this method. They should implement:
        - ``_initialize_run_state(messages=...)``
        - ``_prepare_next_batch(state)``
        """
        system = self._system_prompts[self._tool_prompt_key].render(context)
        messages = self.build_messages(system, turns, prompt)

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

            # Empty prepared_steps means all steps in this batch were cascade-failed;
            # skip execution and loop to the next batch.
            if not state.prepared_steps:
                continue

            state = self._execute_prepared_batch(state)

        # Always persist run slots — mirrors how base Agent always appends records.
        # context_enabled only controls cache_blackboard in _initialize_run_state.
        blackboard_start = len(self._blackboard)
        state = self.update_blackboard(state)
        blackboard_end = len(self._blackboard)

        # Collect failures only when fail_fast=False (fail_fast=True would have raised).
        exception_records: tuple[tuple[int, Exception], ...] = ()
        if not self._fail_fast:
            exception_records = tuple(
                (slot.step, slot.error)
                for slot in self._blackboard[blackboard_start:blackboard_end]
                if slot.status == BlackboardSlot.FAILED and isinstance(slot.error, Exception)
            )

        # Derive per-tool call counts from this invocation's running blackboard.
        # Slots are iterated in execution order (first-call order preserved by dict).
        _counts: dict[str, int] = {}
        for _slot in state.running_blackboard:
            if (
                _slot.is_executed()
                and isinstance(_slot.tool, str)
                and _slot.tool != RETURN_TOOL_FULL_NAME
            ):
                _counts[_slot.tool] = _counts.get(_slot.tool, 0) + 1
        tool_usage = tuple(
            ToolUsageRecord(tool_name=name, call_count=count)
            for name, count in _counts.items()
        )

        draft = ToolAgentRecord(
            user_prompt=prompt,
            generated_response=state.return_value,
            blackboard_start=blackboard_start,
            blackboard_end=blackboard_end,
        )
        metadata: dict = {
            "llm_records": tuple(state.llm_records),
            "llm_model_data": state.llm_records[-1].llm_result.model_data,
            "tool_usage": tool_usage,
            "exception_records": exception_records,
        }
        return draft, metadata

    async def _ainvoke(
        self,
        turns: List[AgentRecord],
        prompt: str,
        context: dict,
    ) -> tuple[ToolAgentRecord, dict]:
        """
        FINAL async ToolAgent template method.

        Receives selected canonical turns, the current prompt, and the assembled
        context dict from the base ``Agent.async_invoke(...)`` lifecycle. Renders
        the system prompt from context, builds the message list once, then runs
        the ToolAgent template loop. Returns a 2-tuple mirroring the sync
        ``_invoke(...)`` contract — see its docstring for details on the
        draft-record and metadata dict contents.

        Mirrors the sync ``_invoke(...)`` loop, but offloads the current sync
        planning hooks to worker threads and awaits the async batch executor for
        tool execution.

        Subclasses should not override this method. They should implement:
        - ``_initialize_run_state(messages=...)``
        - ``_prepare_next_batch(state)``
        - ``_ainitialize_run_state(messages=...)`` (async; base default: asyncio.to_thread wrap)
        - ``_aprepare_next_batch(state)`` (async; base default: asyncio.to_thread wrap)
        """
        system = self._system_prompts[self._tool_prompt_key].render(context)
        messages = self.build_messages(system, turns, prompt)

        if not messages:
            raise ToolAgentError("ToolAgent._ainvoke requires a non-empty messages list.")

        state = await self._ainitialize_run_state(messages=messages)

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

            state = await self._aprepare_next_batch(state)

            # Empty prepared_steps means all steps in this batch were cascade-failed;
            # skip execution and loop to the next batch.
            if not state.prepared_steps:
                continue

            state = await self._async_execute_prepared_batch(state)

        # Always persist run slots — mirrors how base Agent always appends records.
        # context_enabled only controls cache_blackboard in _initialize_run_state.
        blackboard_start = len(self._blackboard)
        state = self.update_blackboard(state)
        blackboard_end = len(self._blackboard)

        # Collect failures only when fail_fast=False (fail_fast=True would have raised).
        exception_records: tuple[tuple[int, Exception], ...] = ()
        if not self._fail_fast:
            exception_records = tuple(
                (slot.step, slot.error)
                for slot in self._blackboard[blackboard_start:blackboard_end]
                if slot.status == BlackboardSlot.FAILED and isinstance(slot.error, Exception)
            )

        # Derive per-tool call counts from this invocation's running blackboard.
        _counts: dict[str, int] = {}
        for _slot in state.running_blackboard:
            if (
                _slot.is_executed()
                and isinstance(_slot.tool, str)
                and _slot.tool != RETURN_TOOL_FULL_NAME
            ):
                _counts[_slot.tool] = _counts.get(_slot.tool, 0) + 1
        tool_usage = tuple(
            ToolUsageRecord(tool_name=name, call_count=count)
            for name, count in _counts.items()
        )

        draft = ToolAgentRecord(
            user_prompt=prompt,
            generated_response=state.return_value,
            blackboard_start=blackboard_start,
            blackboard_end=blackboard_end,
        )
        metadata: dict = {
            "llm_records": tuple(state.llm_records),
            "llm_model_data": state.llm_records[-1].llm_result.model_data,
            "tool_usage": tool_usage,
            "exception_records": exception_records,
        }
        return draft, metadata

    def make_result(
        self,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> ToolAgentResult:
        """
        Construct this ToolAgent's ``ToolAgentResult`` envelope.

        Extends ``Agent.make_result`` with ``tool_usage`` and ``exception_records``
        from the metadata dict returned by ``_invoke``. No derivation occurs here —
        both fields are computed during the execution loop and passed through directly.
        """
        unexpected = set(result_kwargs) - {"llm_records", "llm_model_data", "tool_usage", "exception_records"}
        if unexpected:
            raise ToolAgentError(
                f"make_result: unexpected result kwarg(s): {sorted(unexpected)!r}."
            )

        llm_records = result_kwargs.get("llm_records")
        llm_model_data = result_kwargs.get("llm_model_data")
        tool_usage = result_kwargs.get("tool_usage", ())
        exception_records = result_kwargs.get("exception_records", ())

        if (
            not isinstance(llm_records, tuple)
            or not llm_records
            or not all(isinstance(r, LLMRecord) for r in llm_records)
        ):
            raise ToolAgentError(
                "ToolAgent.make_result: llm_records must be a non-empty tuple of LLMRecord instances."
            )

        if not isinstance(llm_model_data, LLMModelData):
            raise ToolAgentError(
                "ToolAgent.make_result: llm_model_data must be an LLMModelData instance."
            )

        llm_token_usage = tuple(
            r.llm_result.token_usage
            for r in llm_records
            if r.llm_result.token_usage is not None
        )

        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=ToolAgentResult,
            llm_token_usage=llm_token_usage,
            llm_model_data=llm_model_data,
            tool_usage=tool_usage,
            exception_records=exception_records,
        )

    def render_turn(self, turn: AgentRecord) -> list[dict[str, str]]:
        """Render one stored ToolAgentRecord into LLM-facing user/assistant messages.

        The base assistant response is rendered through `Agent.render_turn(...)`, preserving
        `assistant_response_source` and `response_preview_limit` behavior. If the turn has
        a non-empty blackboard span, this method appends a cached-step block containing
        each produced step's unresolved args. Result previews are included only when
        `peek_at_cache=True` and are bounded by `blackboard_preview_limit`.
        """
        if not isinstance(turn, ToolAgentRecord):
            raise ToolAgentError(
                f"render_turn expected ToolAgentRecord, got {type(turn)!r}"
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
                STEP_FIELD: slot.step,
                TOOL_FIELD: slot.tool,
                ARGS_FIELD: slot.args,
            }
            if slot.is_executed():
                step["run_id"] = slot.result.run_id
            if self.peek_at_cache:
                step["result"] = self._preview_blackboard_result(slot.result.result)
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
    @staticmethod
    def _normalize_step_field_set(
        fields: Collection[str],
        *,
        name: str,
        require_non_empty: bool,
    ) -> frozenset[str]:
        """
        Normalize and validate a ToolAgent step-field schema set.

        This helper treats schema arguments as programmer-supplied bounds, not
        LLM output. It rejects strings as a whole because a single string is
        technically a collection of characters, but never a valid field set.
        """
        if isinstance(fields, str) or not isinstance(fields, Collection):
            raise ToolAgentError(
                f"{name} must be a collection of field-name strings; "
                f"got {type(fields).__name__!r}."
            )

        normalized: set[str] = set()
        for field_name in fields:
            if not isinstance(field_name, str) or not field_name:
                raise ToolAgentError(
                    f"{name} must contain only non-empty strings; got {field_name!r}."
                )
            normalized.add(field_name)

        if require_non_empty and not normalized:
            raise ToolAgentError(f"{name} must not be empty.")

        return frozenset(normalized)

    def _validate_tool_step_dict(
        self,
        data: Mapping[str, Any],
        *,
        expected_step: int,
        allowed_fields: Collection[str],
        required_fields: Collection[str],
        context: str,
    ) -> dict[str, Any]:
        """
        Validate and normalize one raw LLM-produced ToolAgent step mapping.

        The caller provides explicit field bounds:
        - ``allowed_fields`` is the maximum allowed key set.
        - ``required_fields`` is the minimum required key set.

        ``context`` is only for error messages. This method does not infer PlanAct,
        ReAct, or base-step behavior from context.

        Runtime owns the authoritative step index. Any LLM-provided ``step`` value
        is advisory and is always overwritten with ``expected_step``.
        """
        if type(expected_step) is not int or expected_step < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} expected_step must be an int >= 0; "
                f"got {expected_step!r}."
            )

        if not isinstance(context, str) or not context.strip():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: context must be a non-empty string."
            )

        allowed = self._normalize_step_field_set(
            allowed_fields,
            name="allowed_fields",
            require_non_empty=True,
        )
        required = self._normalize_step_field_set(
            required_fields,
            name="required_fields",
            require_non_empty=False,
        )

        required_not_allowed = required - allowed
        if required_not_allowed:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: required_fields must be a subset of "
                f"allowed_fields; invalid required field(s): {sorted(required_not_allowed)!r}."
            )

        if not isinstance(data, Mapping):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} must be a mapping; "
                f"got {type(data).__name__!r}."
            )

        data_keys = set(data)

        extra = data_keys - allowed
        if extra:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} contains unsupported keys: "
                f"{sorted(extra)!r}."
            )

        missing = required - data_keys
        if missing:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} missing required keys: "
                f"{sorted(missing)!r}."
            )

        normalized = dict(data)

        # Advisory/fallback behavior:
        # - If the LLM omitted "step", fill it.
        # - If the LLM supplied a wrong/non-sequential "step", ignore it.
        # Runtime order is authoritative.
        normalized[STEP_FIELD] = expected_step

        tool = normalized.get(TOOL_FIELD, NO_VAL)
        if TOOL_FIELD in normalized or TOOL_FIELD in required:
            if not isinstance(tool, str) or not tool.strip():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: {context} {TOOL_FIELD!r} "
                    "must be a non-empty string."
                )

        args = normalized.get(ARGS_FIELD, NO_VAL)
        if ARGS_FIELD in normalized or ARGS_FIELD in required:
            if not isinstance(args, dict):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: {context} {ARGS_FIELD!r} "
                    f"must be a dict; got {type(args).__name__!r}."
                )

        if AWAIT_FIELD in normalized:
            await_step = normalized[AWAIT_FIELD]
            if type(await_step) is not int or await_step < 0:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: {context} {AWAIT_FIELD!r} "
                    "must be an int >= 0."
                )

            if tool == RETURN_TOOL_FULL_NAME:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: return step must not include "
                    f"{AWAIT_FIELD!r}."
                )

        return normalized

    def _tool_step_dict_to_slot(
        self,
        data: Mapping[str, Any],
        *,
        step: int,
        allowed_fields: Collection[str],
        context: str,
    ) -> BlackboardSlot:
        """
        Convert a normalized tool-step mapping into a planned BlackboardSlot.

        This method is a conversion guard, not the primary raw-LLM schema
        validator. It rejects fields outside ``allowed_fields`` and validates the
        executable slot fields needed to construct a BlackboardSlot.

        Required-field validation should already have happened in
        ``_validate_tool_step_dict(...)``.
        """
        if type(step) is not int or step < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} step must be an int >= 0; "
                f"got {step!r}."
            )

        if not isinstance(context, str) or not context.strip():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: context must be a non-empty string."
            )

        allowed = self._normalize_step_field_set(
            allowed_fields,
            name="allowed_fields",
            require_non_empty=True,
        )

        if not isinstance(data, Mapping):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} must be a mapping; "
                f"got {type(data).__name__!r}."
            )

        extra = set(data) - allowed
        if extra:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} contains unsupported keys: "
                f"{sorted(extra)!r}."
            )

        tool = data.get(TOOL_FIELD, NO_VAL)
        args = data.get(ARGS_FIELD, NO_VAL)

        if not isinstance(tool, str) or not tool.strip():
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} {TOOL_FIELD!r} "
                "must be a non-empty string."
            )

        if not isinstance(args, dict):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: {context} {ARGS_FIELD!r} "
                f"must be a dict; got {type(args).__name__!r}."
            )

        await_step = NO_VAL
        if AWAIT_FIELD in data:
            await_step = data[AWAIT_FIELD]
            if type(await_step) is not int or await_step < 0:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: {context} {AWAIT_FIELD!r} "
                    "must be an int >= 0."
                )

        if await_step is not NO_VAL and tool == RETURN_TOOL_FULL_NAME:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: return step must not include "
                f"{AWAIT_FIELD!r}."
            )

        deps: set[int] = set(
            extract_dependencies(obj=args, placeholder_pattern=self.STEP_REF_PATTERN)
        )

        try:
            return BlackboardSlot(
                step=step,
                tool=tool,
                args=args,
                resolved_args=NO_VAL,
                status=BlackboardSlot.PLANNED,
                step_dependencies=tuple(sorted(deps)),
                await_step=await_step,
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
    def _initialize_run_state(self, *, messages: list[dict[str, str]]) -> ToolAgentRunState:
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
    def _prepare_next_batch(self, state: ToolAgentRunState) -> ToolAgentRunState:
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

    async def _ainitialize_run_state(
        self,
        *,
        messages: list[dict[str, str]],
    ) -> ToolAgentRunState:
        """
        Async hook for run-state initialization.

        Default offloads the sync hook to a worker thread. Subclasses override
        when the hook contains blocking I/O (e.g. a planning LLM call).
        """
        return await asyncio.to_thread(self._initialize_run_state, messages=messages)

    async def _aprepare_next_batch(
        self,
        state: ToolAgentRunState,
    ) -> ToolAgentRunState:
        """
        Async hook for batch preparation.

        Default offloads the sync hook to a worker thread. Subclasses override
        when the hook contains blocking I/O (e.g. a per-step LLM call).
        """
        return await asyncio.to_thread(self._prepare_next_batch, state)

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of this ToolAgent.

        Extends the base Agent snapshot with ToolAgent-specific toolbox and blackboard
        diagnostics.
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
class PlanActAgent(ToolAgent):
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
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        filter_extraneous_inputs: Optional[bool] = None,
        *,
        context_enabled: bool = False,
        tool_calls_limit: int | None = None,
        fail_fast: bool = True,
        peek_at_cache: bool = False,
        response_preview_limit: Optional[int] = None,
        blackboard_preview_limit: Optional[int] = None,
        pre_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        post_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        post_result_key: Optional[str] = None,
        records_window: int | None = None,
    ) -> None:
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            tool_instructions=PLANNER_PROMPT,
            filter_extraneous_inputs=filter_extraneous_inputs,
            context_enabled=context_enabled,
            tool_calls_limit=tool_calls_limit,
            fail_fast=fail_fast,
            peek_at_cache=peek_at_cache,
            response_preview_limit=response_preview_limit,
            blackboard_preview_limit=blackboard_preview_limit,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            prompt_key="plan_first",
            records_window=records_window,
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

        return_name = RETURN_TOOL_FULL_NAME
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
                    args={RETURN_VALUE_FIELD: None},
                    resolved_args=NO_VAL,
                    result=NO_VAL,
                    error=NO_VAL,
                    status=BlackboardSlot.PLANNED,
                    step_dependencies=tuple(),
                    await_step=NO_VAL,
                )
            )

        for i, slot in enumerate(slots):
            slot.step = i
            slot.resolved_args = NO_VAL
            slot.result = NO_VAL
            slot.error = NO_VAL
            slot.status = BlackboardSlot.PLANNED

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
        return_slot.status = BlackboardSlot.PLANNED

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

        return_name = RETURN_TOOL_FULL_NAME
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
            cache_refs = extract_dependencies(slot.args, placeholder_pattern=self.CACHE_REF_PATTERN)
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

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #
    def _setup_plan_init(
        self,
        *,
        messages: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], list[BlackboardSlot]]:
        """Validate messages, copy to working list, snapshot cache blackboard."""
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
        return working_messages, cache_blackboard

    def _process_plan_output(
        self,
        *,
        engine_result: Any,
        messages: list[dict[str, str]],
        cache_blackboard: list[BlackboardSlot],
    ) -> tuple[list[BlackboardSlot], LLMRecord]:
        """
        Parse, validate, and normalize raw LLM plan output into planned slots.

        Constructs the LLMRecord, then applies the full
        parse → validate → convert → normalize → validate pipeline.
        """
        llm_record = LLMRecord(
            messages=[messages[-1]],
            llm_result=engine_result,
            system_prompt_name=self._tool_prompt_key,
        )
        parsed = self._extract_from_json_string(engine_result.result)

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
                allowed_fields=PLAN_FIELDS,
                required_fields=REQUIRED_PLAN_FIELDS,
                context="plan step",
            )
            slot = self._tool_step_dict_to_slot(
                step_dict,
                step=i,
                allowed_fields=PLAN_FIELDS,
                context="plan step",
            )
            planned_slots.append(slot)

        planned_slots = self._normalize_planned_slots(planned_slots)

        self._validate_planned_slots(
            planned_slots=planned_slots,
            cache_blackboard=cache_blackboard,
        )

        return planned_slots, llm_record

    def _build_planact_run_state(
        self,
        *,
        planned_slots: list[BlackboardSlot],
        working_messages: list[dict[str, str]],
        cache_blackboard: list[BlackboardSlot],
        llm_record: LLMRecord,
    ) -> PlanActRunState:
        """Validate plan structure, compile batches, and construct PlanActRunState."""
        if not planned_slots:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: generated plan is empty."
            )

        return_idx = len(planned_slots) - 1
        if planned_slots[return_idx].tool != RETURN_TOOL_FULL_NAME:
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
            llm_records=[llm_record],
            is_done=False,
            return_value=NO_VAL,
            batches=batches,
            batch_index=0,
        )

    def _generate_plan(
        self,
        *,
        messages: list[dict[str, str]],
        cache_blackboard: list[BlackboardSlot],
    ) -> tuple[list[BlackboardSlot], LLMRecord]:
        """
        Generate, normalize, and validate a complete PlanAct running blackboard.

        This is the PlanAct generation hook. It is intentionally single-shot and
        fail-fast: no retry logic is performed here.

        Lifecycle
        ---------
        1. Generate raw LLM output from the provided messages, capturing the full
           LLMResult and wrapping it (with the prompt that produced it) in an
           LLMRecord for the run's accumulated LLM history.
        2. Extract the largest JSON array/object from the generated text.
        3. Validate that the extracted value is a non-empty list of mappings.
        4. Convert each mapping into a planned BlackboardSlot.
        5. Normalize the planned slot list.
        6. Validate the final planned slot list.
        7. Return the final list of planned slots together with the LLMRecord.

        Parameters
        ----------
        messages : list[dict[str, str]]
            LLM-facing messages already built by the base Agent message pipeline.

        cache_blackboard : list[BlackboardSlot]
            Snapshot of persisted blackboard entries available to this invoke.
            Used for validating cache placeholder references.

        Returns
        -------
        tuple[list[BlackboardSlot], LLMRecord]
            Fully normalized and validated planned slots for the running
            blackboard, plus the LLMRecord capturing the single generation that
            produced them.

        Raises
        ------
        ToolAgentError
            If generation output cannot be parsed, converted, normalized, or validated.

        PlanAct raw step schema:
        - allowed: PLAN_FIELDS
        - required: REQUIRED_PLAN_FIELDS

        The LLM-provided ``step`` field is advisory. JSON array position is
        authoritative and is passed as ``expected_step``.
        """
        if not messages:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: messages must be non-empty."
            )

        engine_result = self._llm_engine.invoke({"messages": [dict(m) for m in messages]})
        return self._process_plan_output(
            engine_result=engine_result,
            messages=messages,
            cache_blackboard=cache_blackboard,
        )

    async def _agenerate_plan(
        self,
        *,
        messages: list[dict[str, str]],
        cache_blackboard: list[BlackboardSlot],
    ) -> tuple[list[BlackboardSlot], LLMRecord]:
        """Async mirror of ``_generate_plan``: uses ``async_invoke`` for the LLM call."""
        if not messages:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: messages must be non-empty."
            )

        engine_result = await self._llm_engine.async_invoke({"messages": [dict(m) for m in messages]})
        return self._process_plan_output(
            engine_result=engine_result,
            messages=messages,
            cache_blackboard=cache_blackboard,
        )

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
        working_messages, cache_blackboard = self._setup_plan_init(messages=messages)
        planned_slots, llm_record = self._generate_plan(
            messages=working_messages,
            cache_blackboard=cache_blackboard,
        )
        return self._build_planact_run_state(
            planned_slots=planned_slots,
            working_messages=working_messages,
            cache_blackboard=cache_blackboard,
            llm_record=llm_record,
        )

    async def _ainitialize_run_state(
        self,
        *,
        messages: list[dict[str, str]],
    ) -> PlanActRunState:
        """
        Async override: uses ``_agenerate_plan`` so the planning LLM call
        goes through ``async_invoke`` rather than a worker thread.
        """
        working_messages, cache_blackboard = self._setup_plan_init(messages=messages)
        planned_slots, llm_record = await self._agenerate_plan(
            messages=working_messages,
            cache_blackboard=cache_blackboard,
        )
        return self._build_planact_run_state(
            planned_slots=planned_slots,
            working_messages=working_messages,
            cache_blackboard=cache_blackboard,
            llm_record=llm_record,
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
            or planned_slots[return_idx].tool != RETURN_TOOL_FULL_NAME
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
           - **Cascade check** (``fail_fast=False`` only): if any ``step_dependencies``
             entry is FAILED in the running blackboard, the return tool raises immediately;
             non-return steps are marked FAILED and skipped (not added to prepared_steps)
           - Call ``_resolve_placeholders(slot.args, state=state)``
           - Store resolved args in ``slot.resolved_args``
           - Mark slot ``status="prepared"``
        5. **Set prepared_steps**: Indices of steps that passed the cascade check and were
           prepared; may be empty if all steps in the batch were cascade-failed
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
        prepared_in_batch: list[int] = []  # excludes cascade-failed steps; may yield empty prepared_steps
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

            # Cascade-fail: when fail_fast=False, propagate failures through arg dependencies.
            # Use extract_dependencies(slot.args) rather than slot.step_dependencies because
            # the return slot's step_dependencies is forced to include ALL prior steps for
            # scheduling; only steps actually referenced in args need to resolve successfully.
            if not self._fail_fast:
                failed_arg_deps = sorted(
                    d
                    for d in extract_dependencies(slot.args, placeholder_pattern=self.STEP_REF_PATTERN)
                    if board[d].is_failed()
                )
                if failed_arg_deps:
                    dep_str = ", ".join(str(d) for d in failed_arg_deps)
                    if slot.tool == RETURN_TOOL_FULL_NAME:
                        raise ToolAgentError(
                            f"{type(self).__name__}.{self.name}: return step {i} cannot execute; "
                            f"dependency step(s) {dep_str} failed."
                        )
                    slot.error = ToolAgentError(
                        f"{type(self).__name__}.{self.name}: step {i} skipped — "
                        f"dependency step(s) {dep_str} failed."
                    )
                    slot.status = BlackboardSlot.FAILED
                    continue

            # Resolve placeholders using base resolver (checks cache + executed step readiness).
            slot.resolved_args = self._resolve_placeholders(slot.args, state=state)

            if slot.result is not NO_VAL:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: internal error: step {i} result was set during preparation."
                )

            slot.error = NO_VAL
            slot.status = BlackboardSlot.PREPARED
            prepared_in_batch.append(i)

        state.prepared_steps = sorted(prepared_in_batch)  # empty when all steps cascade-failed
        state.batch_index += 1
        logger.info(
            f"{self.full_name}: Prepared batch {state.batch_index}/{len(state.batches)} "
            f"with steps {state.prepared_steps}."
        )
        return state


# ───────────────────────────────────────────────────────────────────────────────
# Iterative Plan 'ReActAgent' class
# ───────────────────────────────────────────────────────────────────────────────
class ReActAgent(ToolAgent):
    """
    Iterative agent with reactive step-by-step planning (ReAct-style architecture).

    **Design**: ReActAgent implements dynamic iteration: one step is emitted per LLM
    turn from a compact running-plan snapshot. Result references are always visible
    by placeholder; raw result previews are shown only while their observability
    duration remains active. Generated step descriptions preserve semantic intent
    across turns without exposing raw results.

    1. **Initialization** (``_initialize_run_state``)
       - Pre-allocates a fixed-size running blackboard: ``tool_calls_limit + 1`` slots
         to accommodate non-return tool calls plus one return call.
       - Requires ``tool_calls_limit`` to be a concrete integer.
       - Snapshots cached blackboard entries only when ``context_enabled=True``.
       - Initializes ReAct-specific per-step metadata:
         observability counters and generated step descriptions.

    2. **Preparation** (``_prepare_next_batch`` – single step per turn)
       - Builds a fresh temporary message list from static base messages.
       - Appends a compact running-plan snapshot containing executed steps,
         descriptions, unresolved args, result_ref placeholders, and any currently
         observable_result values.
       - Asks the LLM for the next single step.
       - Step generation returns a planned slot, observability duration, and
         one-sentence description.
       - Validates step index matches the expected cursor position.
       - Validates placeholder dependencies are prior-only.
       - Resolves placeholders into concrete tool args.
       - Stores the prepared slot in ``running_blackboard[step_index]``.
       - Stores duration and description in ReAct run state.
       - Sets ``prepared_steps = [step_index]``.

    3. **Execution**
       - Base loop executes the single prepared step.
       - Result is stored; loop returns to preparation for the next step.

    4. **Termination**
       - When the return tool is emitted and executed, loop exits.
       - Running blackboard is persisted if ``context_enabled=True``.

    Advantages
    ~~~~~~~~~~
    - Fully adaptive: each step can react to prior tool results.
    - Context-hygienic: running-plan messages are rebuilt fresh each turn.
    - Interpretable: descriptions, placeholders, and optional observations provide
      a compact execution trace.

    Limitations
    ~~~~~~~~~~~
    - Higher latency: one LLM call per step.
    - No concurrency: only one step executes per iteration.
    - Step quality depends on the model's ability to select the next best action.

    Parameters (construction)
    ~~~~~~~~~~~~~~~~~~~~~~~~
    ``tool_calls_limit`` (int, REQUIRED): Must be a concrete integer >= 0.
    Determines pre-allocated blackboard size. Cannot be ``None``.
    """

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        filter_extraneous_inputs: Optional[bool] = None,
        *,
        context_enabled: bool = False,
        tool_calls_limit: int = 25,
        fail_fast: bool = True,
        peek_at_cache: bool = False,
        response_preview_limit: Optional[int] = None,
        blackboard_preview_limit: Optional[int] = None,
        pre_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        post_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        post_result_key: Optional[str] = None,
        records_window: int | None = None,
    ) -> None:
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            tool_instructions=ORCHESTRATOR_PROMPT,
            filter_extraneous_inputs=filter_extraneous_inputs,
            context_enabled=context_enabled,
            tool_calls_limit=tool_calls_limit,
            fail_fast=fail_fast,
            peek_at_cache=peek_at_cache,
            response_preview_limit=response_preview_limit,
            blackboard_preview_limit=blackboard_preview_limit,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            prompt_key="reason_then_act",
            records_window=records_window,
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
    # Private helpers
    # ------------------------------------------------------------------ #
    def _validate_react_prepare_state(self, state: ReActRunState) -> None:
        """
        Validate cursor bounds, prior-step execution, and step_meta length.

        Replaces the former ``latest_executed`` validation block: the invariant
        that the previous step was executed is derived from ``next_step_index - 1``
        and the slot's status rather than a separate bookkeeping field.
        """
        prefix_len = state.next_step_index
        if type(prefix_len) is not int or prefix_len < 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next_step_index must be an "
                f"int >= 0; got {prefix_len!r}."
            )
        if prefix_len >= len(state.running_blackboard):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next_step_index exceeds run "
                f"blackboard capacity ({prefix_len} >= {len(state.running_blackboard)})."
            )
        if prefix_len > 0:
            prev = state.running_blackboard[prefix_len - 1]
            # With fail_fast=False a previous step may be FAILED rather than EXECUTED;
            # both count as "processed" and allow generation of the next step.
            prev_processed = prev.is_executed() or (not self._fail_fast and prev.is_failed())
            if not prev_processed:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step {prefix_len - 1} was "
                    f"not executed before the next prepare call "
                    f"(status={prev.status!r})."
                )
        if len(state.step_meta) != len(state.running_blackboard):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: step_meta length must match "
                f"running_blackboard length "
                f"({len(state.step_meta)} != {len(state.running_blackboard)})."
            )

    def _build_react_messages(
        self,
        state: ReActRunState,
        prefix_len: int,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """
        Build the working message list and delta for one ReAct step generation.

        Returns ``(working_messages, delta)`` where ``working_messages`` is the
        full message list to pass to the LLM and ``delta`` is the three-element
        list used to construct the LLMRecord.
        """
        working_messages: list[dict[str, str]] = [dict(m) for m in state.messages]

        running_records: list[dict[str, Any]] = []
        for idx in range(prefix_len):
            slot = state.running_blackboard[idx]
            if not slot.is_executed():
                continue

            record: dict[str, Any] = {
                STEP_FIELD: slot.step,
                DESCRIPTION_FIELD: state.step_meta[idx].description,
                TOOL_FIELD: slot.tool,
                ARGS_FIELD: slot.args,
                "result_ref": f"<<__s{idx}__>>",
                "run_id": slot.result.run_id,
            }

            if state.step_meta[idx].observable > 0:
                record["observable_result"] = self._preview_blackboard_result(slot.result.result)

            running_records.append(record)

        if running_records:
            running_text = (
                f"RUNNING PLAN STEPS 0-{prefix_len - 1} SO FAR:\n"
                "These are the executed steps for the current user task.\n"
                "Use descriptions to understand what each executed step was intended to do.\n"
                "Use result_ref placeholders when a new arg needs a prior step value.\n"
                "observable_result fields are for OBSERVATION ONLY: use them only to choose the next tool or branch.\n"
                "Do not copy observable_result values into new args.\n\n"
                + pprint.pformat(running_records, indent=2, width=160, sort_dicts=False)
            )
        else:
            running_text = (
                "RUNNING PLAN STEPS SO FAR:\n"
                "No steps executed yet.\n"
                "When steps execute, their results will be available by result_ref "
                "placeholders like <<__s0__>>."
            )

        max_duration = len(state.running_blackboard) - prefix_len - 1

        working_messages.append({"role": "assistant", "content": running_text})
        working_messages.append(
            {
                "role": "user",
                "content": (
                    "Produce the NEXT BEST single tool call for the current task. "
                    "Pick the return tool if the running plan has completed all needed work. "
                    "Output exactly one JSON object with keys {step, tool, args, duration, description}. "
                    "Preserve symbolic dataflow with quoted placeholders; do not copy observable_result values into args. "
                    f"For this output step, duration must be an int from 0 to {max_duration}."
                ),
            }
        )

        delta = [state.messages[-1], working_messages[-2], working_messages[-1]]
        return working_messages, delta

    def _process_next_step_output(
        self,
        *,
        engine_result: Any,
        delta: list[dict[str, str]],
        expected_step: int,
        cache_blackboard: list[BlackboardSlot],
        max_duration: int,
    ) -> tuple[BlackboardSlot, int, str, LLMRecord]:
        """
        Parse and validate one ReAct step from raw LLM output.

        Constructs the LLMRecord, parses the JSON payload, and applies the full
        validation pipeline extracted from ``_generate_next_step``.
        """
        llm_record = LLMRecord(
            messages=delta,
            llm_result=engine_result,
            system_prompt_name=self._tool_prompt_key,
        )
        parsed = self._extract_from_json_string(engine_result.result)

        if not isinstance(parsed, Mapping):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next step output must be a JSON object; "
                f"got {type(parsed).__name__!r}."
            )

        raw_payload = dict(parsed)

        extra = set(raw_payload) - REACT_FIELDS
        if extra:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next step contains unsupported keys: "
                f"{sorted(extra)!r}."
            )

        if DURATION_FIELD not in raw_payload:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next step missing required key {DURATION_FIELD!r}."
            )

        if DESCRIPTION_FIELD not in raw_payload:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next step missing required key {DESCRIPTION_FIELD!r}."
            )

        step_payload = self._validate_tool_step_dict(
            raw_payload,
            expected_step=expected_step,
            allowed_fields=REACT_FIELDS,
            required_fields=REQUIRED_REACT_FIELDS,
            context="next step",
        )

        duration = step_payload.pop(DURATION_FIELD)
        if type(duration) is not int or duration < 0 or duration > max_duration:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next step {DURATION_FIELD!r} must be an int in "
                f"[0, {max_duration}] for expected_step={expected_step}; got {duration!r}."
            )

        description = step_payload.pop(DESCRIPTION_FIELD)
        if type(description) is not str:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next step {DESCRIPTION_FIELD!r} must be a string; "
                f"got {type(description).__name__!r}."
            )

        description = description.strip()
        if not description:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: next step {DESCRIPTION_FIELD!r} cannot be empty."
            )

        slot = self._tool_step_dict_to_slot(
            step_payload,
            step=expected_step,
            allowed_fields=BASE_STEP_FIELDS,
            context="next step",
        )

        self.get_tool(slot.tool)

        if slot.tool == RETURN_TOOL_FULL_NAME and duration != 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: return tool must use {DURATION_FIELD!r} 0; "
                f"got {duration!r}."
            )

        cache_len = len(cache_blackboard)
        cache_refs = extract_dependencies(slot.args, placeholder_pattern=self.CACHE_REF_PATTERN)
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

        return slot, duration, description, llm_record

    def _apply_react_step_result(
        self,
        state: ReActRunState,
        prefix_len: int,
        generated_slot: BlackboardSlot,
        observe_duration: int,
        description: str,
        llm_record: LLMRecord,
    ) -> ReActRunState:
        """
        Apply one validated ReAct step generation result to the run state.

        Validates the returned tuple fields, decrements observable counters,
        fills the preallocated running-blackboard slot, then either cascade-fails
        the slot (``fail_fast=False`` only) or resolves placeholders and marks it
        prepared. Advances the cursor and writes ``step_meta``.

        **Cascade path** (``fail_fast=False``): if any ``step_dependencies`` entry
        is FAILED in the running blackboard, the return tool raises immediately;
        non-return slots are marked FAILED and the method returns early with
        ``prepared_steps`` left empty — the ``_invoke`` loop will skip execution
        and continue to the next generation turn.
        """
        max_duration = len(state.running_blackboard) - prefix_len - 1

        if not isinstance(generated_slot, BlackboardSlot):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: _generate_next_step must return "
                f"(BlackboardSlot, observe_duration, description, LLMRecord); got first "
                f"item {type(generated_slot).__name__!r}."
            )

        state.llm_records.append(llm_record)

        if type(observe_duration) is not int or observe_duration < 0 or observe_duration > max_duration:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: observe_duration must be an int in "
                f"[0, {max_duration}]; got {observe_duration!r}."
            )

        if generated_slot.tool == RETURN_TOOL_FULL_NAME and observe_duration != 0:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: return tool must use duration 0; "
                f"got {observe_duration!r}."
            )

        if type(description) is not str:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: description must be a string; "
                f"got {type(description).__name__!r}."
            )

        description = description.strip()

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

        # A successful generation turn consumed any raw results that were visible.
        for meta in state.step_meta:
            if meta.observable > 0:
                meta.observable -= 1

        # Fill the preallocated running-blackboard slot.
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

        # Cascade-fail: when fail_fast=False, propagate failures through arg dependencies.
        # Use extract_dependencies(slot.args) for the same reason as PlanActAgent: only
        # steps actually referenced in args need to resolve successfully.
        if not self._fail_fast:
            board = state.running_blackboard
            failed_arg_deps = sorted(
                d
                for d in extract_dependencies(slot.args, placeholder_pattern=self.STEP_REF_PATTERN)
                if board[d].is_failed()
            )
            if failed_arg_deps:
                dep_str = ", ".join(str(d) for d in failed_arg_deps)
                if slot.tool == RETURN_TOOL_FULL_NAME:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: return step cannot execute; "
                        f"dependency step(s) {dep_str} failed."
                    )
                slot.error = ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step {prefix_len} skipped — "
                    f"dependency step(s) {dep_str} failed."
                )
                slot.status = BlackboardSlot.FAILED
                state.step_meta[prefix_len].description = description
                state.next_step_index = prefix_len + 1
                return state  # prepared_steps stays []; _invoke loop will skip execute and continue

        # Resolve placeholders after stamping the planned slot into the running state.
        slot.resolved_args = self._resolve_placeholders(slot.args, state=state)
        slot.status = BlackboardSlot.PREPARED

        # Write per-slot metadata.
        state.step_meta[prefix_len].observable = observe_duration
        state.step_meta[prefix_len].description = description

        state.prepared_steps = [prefix_len]
        state.next_step_index = prefix_len + 1

        return state

    # ------------------------------------------------------------------ #
    # Tool-Agent Hooks
    # ------------------------------------------------------------------ #
    def _initialize_run_state(self, *, messages: list[dict[str, str]]) -> ReActRunState:
        """
        Initialize run state for a single ReAct invocation.

        Satisfies the ``ToolAgent._initialize_run_state`` abstract hook. Unlike
        ``PlanActAgent``, ReAct performs no LLM call at initialization — the
        running blackboard is pre-allocated and step planning is deferred to
        ``_prepare_next_batch`` / ``_aprepare_next_batch``.

        Steps
        -----
        1. Validate ``messages`` is non-empty.
        2. Snapshot the persisted blackboard as ``cache_blackboard`` (empty list
           when ``context_enabled=False``).
        3. Copy ``messages`` into a mutable working list.
        4. Pre-allocate a fixed-size ``running_blackboard`` of
           ``tool_calls_limit + 1`` slots — one slot per allowed non-return
           call plus one slot for the mandatory return call.
        5. Initialize ``step_meta`` as a list of ``ReActStepMeta()`` instances
           of the same length as ``running_blackboard``.
        6. Construct and return ``ReActRunState`` with empty ``llm_records``
           (records are appended by each ``_prepare_next_batch`` call).

        Returns
        -------
        ReActRunState

        Raises
        ------
        ToolAgentError
            If ``messages`` is empty.
        """
        if not messages:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: messages must be non-empty.")

        cache_blackboard = self.blackboard if self.context_enabled else []
        for i, slot in enumerate(cache_blackboard):
            slot.step = i

        working_messages = [dict(m) for m in messages]

        # Preallocate fixed-size run blackboard: non-return calls + 1 return call.
        running_blackboard = [BlackboardSlot(step=i) for i in range(self._tool_calls_limit + 1)]

        return ReActRunState(
            messages=working_messages,
            cache_blackboard=cache_blackboard,
            running_blackboard=running_blackboard,
            executed_steps=set(),
            prepared_steps=[],
            tool_calls_used=0,
            llm_records=[],
            is_done=False,
            return_value=NO_VAL,
            next_step_index=0,
            step_meta=[ReActStepMeta() for _ in running_blackboard],
        )

    def _generate_next_step(
        self,
        *,
        messages: list[dict[str, str]],
        cache_blackboard: list[BlackboardSlot],
        expected_step: int,
        delta: list[dict[str, str]],
    ) -> tuple[BlackboardSlot, int, str, LLMRecord]:
        """
        Generate and validate one ReAct tool step as a planned BlackboardSlot plus
        observation duration and description.

        This is the ReAct generation hook. It is intentionally single-shot and
        fail-fast: no retry logic is performed here.

        Lifecycle
        ---------
        1. Generate raw LLM output from the provided messages, capturing the full
           LLMResult and wrapping it (with the prompt that produced it) in an
           LLMRecord for the run's accumulated LLM history.
        2. Extract the largest JSON array/object from the generated text.
        3. Validate that the extracted value is a mapping.
        4. Extract and validate "duration" as an int within the remaining future
           step-generation capacity.
        5. Extract and validate "description" as a non-empty string.
        6. Normalize the remaining raw step dict using expected_step as authoritative.
        7. Convert the normalized mapping into a planned BlackboardSlot.
        8. Validate tool existence.
        9. Validate return-tool duration is 0.
        10. Validate cache references against cache_blackboard.
        11. Validate step dependencies are prior-only.
        12. Return the planned slot, duration, description, and LLMRecord.

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

        delta : list[dict[str, str]]
            The messages that are new for this specific step call, pre-computed
            by the caller. Used to construct the LLMRecord for this generation.
            Typically three messages: the original user task, the running-plan
            snapshot (assistant), and the step-request stub (user).

        Returns
        -------
        tuple[BlackboardSlot, int, str, LLMRecord]
            Planned single-step slot, observation duration, step description, and
            the LLMRecord capturing the single generation that produced them.
            The slot is not inserted into the running blackboard and placeholders
            are not resolved here. The duration controls how many future successful
            step-generation turns may render this step's raw result as observable_result.

        Raises
        ------
        ToolAgentError
            If generation output cannot be parsed, converted, or validated.


        ReAct raw step schema:
        - allowed: REACT_FIELDS
        - required: REQUIRED_REACT_FIELDS

        ``step`` is allowed but advisory. Runtime normalizes it to
        ``expected_step``.
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

        max_duration = max(0, self._tool_calls_limit - expected_step)
        engine_result = self._llm_engine.invoke({"messages": [dict(m) for m in messages]})
        return self._process_next_step_output(
            engine_result=engine_result,
            delta=delta,
            expected_step=expected_step,
            cache_blackboard=cache_blackboard,
            max_duration=max_duration,
        )

    async def _agenerate_next_step(
        self,
        *,
        messages: list[dict[str, str]],
        cache_blackboard: list[BlackboardSlot],
        expected_step: int,
        delta: list[dict[str, str]],
    ) -> tuple[BlackboardSlot, int, str, LLMRecord]:
        """Async mirror of ``_generate_next_step``: uses ``async_invoke`` for the LLM call."""
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

        max_duration = max(0, self._tool_calls_limit - expected_step)
        engine_result = await self._llm_engine.async_invoke(
            {"messages": [dict(m) for m in messages]}
        )
        return self._process_next_step_output(
            engine_result=engine_result,
            delta=delta,
            expected_step=expected_step,
            cache_blackboard=cache_blackboard,
            max_duration=max_duration,
        )

    def _prepare_next_batch(self, state: ReActRunState) -> ReActRunState:
        """
        Prepare the next single-step batch.

        Semantics (single-step ReAct):
        - Build a fresh, temporary LLM message list from the static base messages.
        - Append one assistant message containing the current running-plan snapshot.
        - Append one small user request for the next step.
        - Call `_generate_next_step(...)`, which returns one planned BlackboardSlot,
          a duration for future raw-result observability, and a step description.
        - Validate the generated slot against the current run cursor.
        - Decrement existing non-zero observability durations after successful generation.
        - Fill exactly one preallocated slot in the running_blackboard.
        - Resolve placeholders against the current ToolAgent run state.
        - Store the generated description for future running-plan rendering.
        - Mark the slot prepared.
        - prepared_steps is a list of exactly one index.
        - next_step_index advances by 1.
        - step_meta[idx] is updated with the new observable duration and description.

        The temporary running-plan messages do not persist between turns; state.messages
        remains the static base message list for this invoke.
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
        self._validate_react_prepare_state(state)
        working_messages, delta = self._build_react_messages(state, prefix_len)

        generated_slot, observe_duration, description, llm_record = self._generate_next_step(
            messages=working_messages,
            cache_blackboard=state.cache_blackboard,
            expected_step=prefix_len,
            delta=delta,
        )
        return self._apply_react_step_result(
            state, prefix_len, generated_slot, observe_duration, description, llm_record
        )

    async def _aprepare_next_batch(self, state: ReActRunState) -> ReActRunState:
        """
        Async override: uses ``_agenerate_next_step`` so the per-step LLM call
        goes through ``async_invoke`` rather than a worker thread.
        """
        if not isinstance(state, ReActRunState):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: _aprepare_next_batch requires a ReActRunState."
            )

        if state.prepared_steps:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cannot prepare next batch while prepared_steps is non-empty."
            )

        prefix_len = state.next_step_index
        self._validate_react_prepare_state(state)
        working_messages, delta = self._build_react_messages(state, prefix_len)

        generated_slot, observe_duration, description, llm_record = await self._agenerate_next_step(
            messages=working_messages,
            cache_blackboard=state.cache_blackboard,
            expected_step=prefix_len,
            delta=delta,
        )
        return self._apply_react_step_result(
            state, prefix_len, generated_slot, observe_duration, description, llm_record
        )
