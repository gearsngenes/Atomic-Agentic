"""
ToolAgents: LLM-Driven Iterative Tool Calling with Persistent Blackboard Memory

This module provides an extensible framework for building intelligent agents that
use **Large Language Models (LLMs)** to decide which tools to invoke, observe
results, and either plan or react accordingly.

Core Concept
------------
Rather than executing a fixed sequence of operations, ToolAgents maintain an
interactive execution loop:

1. **LLM decides**: The LLM examines the current task and decides which tools to invoke
2. **Tools execute**: Selected tools run and produce results
3. **Task updates**: Results are stored in the invocation's running blackboard
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

  - ``|STEP.N|`` – result from running step N (current invoke, 0-based index)
  - ``|CACHE.N|`` – result from cache entry N (prior invokes, 0-based index)

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
``ToolAgent`` drives every subclass through the shared
``think`` → ``prepare`` → ``act`` lifecycle (``agents/base.py``); ``act`` is
concrete and final here. Subclasses implement:

**_initialize_task(*, turns, prompt, inputs)** → ``ToolAgentTask`` (or subclass)
  Build and return a task for this invocation:
  - Stamp the system prompt name
  - Snapshot prior cached results if context is enabled
  - Allocate running blackboard slots for current-run tool calls

**think(task)** / **async_think(task)** → ``ToolAgentTask`` (or subclass)
  Make this round's real decision via the LLM: render, call the engine,
  parse/validate the output, store the decision onto ``task``.

**prepare(task)** / **async_prepare(task)** → ``ToolAgentTask`` (or subclass)
  Turn that decision into something executable, no further LLM calls:
  validate/resolve placeholders, cascade-check dependencies, populate
  `task.prepared_steps`.

**_render_task_messages(task)** → ``list[dict[str, str]]``
  The one piece of ``Agent.render_task``'s shared pipeline (``base.py``)
  each subclass still owns — built on the shared ``_render_task_banner``
  helper below (the pipeline's own ``_render_system_message`` is a sibling
  step, not something ``_render_task_messages`` itself calls).

The task is extensible: subclasses carry domain-specific fields (batches,
cursors, planning metadata) on their own ``ToolAgentTask`` subclass
(``PlanActTask``, ``ReActTask``).

Concrete Subclasses
-------------------
- **PlanActAgent** (``agents/planact.py``): One-shot planner; queries LLM once to
  generate an entire plan, then executes in concurrent batches. Fast, deterministic,
  no replanning.
- **ReActAgent** (``agents/react.py``): Iterative actor; queries LLM once per step,
  reacts to each result. Fully adaptive, but requires more LLM turns and sequential
  execution.
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
    Literal,
    Mapping,
    Optional,
)
import pprint

from .base import Agent
from ..constants.agents import (
    ARGS_FIELD,
    AWAIT_FIELD,
    RETURN_TOOL_FULL_NAME,
    RETURN_VALUE_FIELD,
    STEP_FIELD,
    TOOL_FIELD,
    EVENT_TAG,
    RETURN_TAG,
    RESULT_TAG,
    ERROR_TAG,
    RUN_ID_TAG,
)

from ..models.agents.records import AgentRecord, LLMRecord, ToolAgentRecord
from ..models.results.agents import ToolAgentResult, ToolUsageRecord
from ..models.agents.blackboard_models import BlackboardSlot, ConstantSpec
from ..models.agents.tasks import ToolAgentTask
from ..exceptions import (
    ToolAgentError,
    ToolInvocationError,
    ToolRegistrationError,
)
from ..constants.core import IDENTIFIER_PATTERN_TEXT
from ..core.Invokable import AtomicInvokable
from ..constants.core import NO_VAL
from ..llm.base import LLMEngine
from ..tools import toolify
from ..mcp import MCPClientHub
from ..a2a import A2AClientHub, PyA2AtomicClient
from ..utils.agents import (
    extract_dependencies,
    extract_json_object,
    extract_regex_steps,
    format_generation_issues,
)
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

    Task-Oriented Lifecycle
    ------------------------
    ``Agent.invoke()``/``async_invoke()`` drive every ``ToolAgent`` through
    the shared base lifecycle::

    1. task = _initialize_task(turns=turns, prompt=prompt, inputs=inputs)   [subclass hook]
    2. while not task.complete:
        task = think(task)                                                  [subclass hook]
        task = prepare(task)                                                [subclass hook]
        task = act(task)                                                    [ToolAgent, final]
    3. record = _build_record_from_task(task, turns)                        [ToolAgent override]
    4. post_invoke -> build_result_from_record(record, ...)                 [ToolAgent override]

    ``act`` (final; subclasses should not override) trusts ``prepare``'s
    (and, for tool existence/budget, ``think``'s) contract completely — it
    does not re-validate indices, tool names, or budget, only whether
    there's anything to run::

        if not task.prepared_steps: return task       # cascade-skip: entire round produced nothing
        <execute task.prepared_steps concurrently>     [ToolAgent, final]

    Each LLM generation made along the way is captured as an ``LLMRecord``
    and accumulated onto ``task.llm_records``. ``_build_record_from_task``
    persists the completed run's blackboard slots via ``update_blackboard``
    — always, regardless of ``context_enabled`` (that flag only gates
    ``valid_cache_indices``/``failed_cache_indices`` in ``_initialize_task``)
    — and captures the resulting span as ``blackboard_start``/
    ``blackboard_end`` on the returned ``ToolAgentRecord``.
    ``build_result_from_record`` re-derives ``tool_usage``/
    ``exception_records`` from that persisted span. Future LLM-facing
    messages are rendered from the stored ``ToolAgentRecord`` by
    ``render_turn(...)``.

    Subclass Responsibilities
    -------------------------
    Subclasses implement:

    **_initialize_task(*, turns, prompt, inputs)** → ``ToolAgentTask`` (or subclass)
        Build and return a task for this invocation. Must:
        - Stamp ``system_prompt_name`` so ``render_task`` can render the
          active system prompt on demand (never pre-rendered/cached as text)
        - Compute ``valid_cache_indices``/``failed_cache_indices`` via
          ``_compute_cache_index_sets(turns)``
        - Create an appropriate running blackboard
        - Initialize ``executed_steps``, ``prepared_steps``, and completion state

    **think(task)** / **async_think(task)** → ``ToolAgentTask`` (or subclass)
        Make this round's real decision via the LLM (render via
        ``self.render_task(task)``, call the engine, parse/validate the
        output, store the decision onto ``task``). Responsible for
        guaranteeing tool names are registered and the decision respects
        ``tool_calls_limit`` — neither is re-checked downstream.

    **prepare(task)** / **async_prepare(task)** → ``ToolAgentTask`` (or subclass)
        Turn that decision into something executable, no further LLM calls:
        - Cascade-check dependencies (``self._check_cascade_failure``)
        - Resolve placeholders with ``self._resolve_placeholders(...)``
        - Fill ``task.prepared_steps`` with indices ready for execution
        - Return the updated task

    **_render_task_messages(task)** → ``list[dict[str, str]]``
        The one piece of ``Agent.render_task``'s shared pipeline still
        owned per-family — built on ``_render_task_banner`` below
        (``_render_system_message`` is a sibling pipeline step, not
        something this hook itself calls).

    Key Features
    ~~~~~~~~~~~~
    **Concurrent Execution**: Each prepared batch runs through async tool invocation and
    gather-based result collection.

    **Placeholder Resolution**: Supported syntaxes:
        - ``|STEP.N|`` – reference to running step N
        - ``|CACHE.N|`` – reference to cache entry N
        - ``|K.NAME|`` – reference to a registered constant (NAME is case-sensitive)
        Discriminator word is case-insensitive. Full-string placeholders preserve
        types; inline placeholders render via ``repr()``.

    **Return Semantics**: The canonical ``return_tool`` is registered automatically.
        When return executes, ``task.generated_response`` is set and ``task.complete``
        becomes ``True``, ending the think→prepare→act loop.

    **Budget Enforcement**: If ``tool_calls_limit`` is set, ``think`` is responsible for
        keeping non-return tool calls within it — validated once at decision time,
        not re-checked when ``act`` executes.

    **Context Persistence**: The completed run blackboard is always merged into
        ``self._blackboard`` (``blackboard_start``/``blackboard_end`` always set on
        the ``ToolAgentRecord``). If ``context_enabled=True``, those prior slots are
        also fed into the LLM as context on the next invocation.
    """
    TOOLS_FIELD = "TOOLS"
    LIMIT_FIELD = "TOOL_CALLS_LIMIT"
    CONSTANTS_FIELD = "CONSTANTS"

    STEP_REF_PATTERN: re.Pattern[str] = re.compile(
        r"\|STEP\.(\d+)\|", re.IGNORECASE
    )
    CACHE_REF_PATTERN: re.Pattern[str] = re.compile(
        r"\|CACHE\.(\d+)\|", re.IGNORECASE
    )
    CONST_REF_PATTERN: re.Pattern[str] = re.compile(
        rf"\|K\.({IDENTIFIER_PATTERN_TEXT})\|", re.IGNORECASE
    )

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        fail_fast: bool = True,
        generation_retries: int = 0,
        generation_format: Literal["json", "regex"] = "json",
        tool_calls_limit: Optional[int] = None,
        peek_at_cache: bool = False,
        response_preview_limit: Optional[int] = None,
        blackboard_preview_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_result_key: Optional[str] = None,
        records_window: Optional[int] = None,
        assistant_response_source: Literal["raw", "final"] = "raw",
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
        context_enabled : bool
            When ``True``, prior blackboard steps are fed into each invocation
            as LLM context (``valid_cache_indices``/``failed_cache_indices``
            become non-empty in ``_initialize_task``). The blackboard is
            always persisted after each invoke regardless of this setting.
            Defaults to ``False``.
        fail_fast : bool
            When ``True`` (default), the first tool call failure immediately
            raises and aborts the run. When ``False``, failing slots are marked
            ``FAILED`` and the loop continues to execute independent steps;
            failures are collected in ``ToolAgentResult.exception_records``.
            Return-tool failures always raise regardless of this setting.
        generation_retries : int
            Number of additional LLM generation attempts to make when the plan
            output cannot be parsed or fails spec validation. ``0`` (default)
            means a single attempt with no retries; ``N`` means up to ``N``
            extra attempts beyond the first. Must be a non-negative ``int``.
        generation_format : "json" | "regex"
            Wire format for generation output. ``"json"`` (default)
            preserves all current behavior. ``"regex"`` selects the
            free-form, tag-delimited format. Frozen at construction --
            fixed-topology, no setter.
        tool_calls_limit : int | None
            Maximum number of non-return action calls per invoke run.
            ``None`` means unlimited. Must be ``>= 0`` if set.
        peek_at_cache : bool
            When ``True``, ``render_turn()`` includes each executed cached
            step's raw ``result`` value in the LLM-facing history it
            renders. Defaults to ``False``. (Distinct from
            ``blackboard_serialized``'s own ``peek`` parameter, which has no
            LLM-facing caller -- its only caller is ``to_dict()``, always
            with ``peek=False``.)
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
        records_window : int | None
            Maximum number of prior ``AgentRecord`` turns rendered into LLM
            context. ``None`` means all records are rendered.
        assistant_response_source : "raw" | "final"
            Whether rendered assistant history uses the raw generated
            response or the final post-``post_invoke`` result. Defaults to
            ``"raw"``.
        """
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            context_enabled=context_enabled,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            records_window=records_window,
            response_preview_limit=response_preview_limit,
            assistant_response_source=assistant_response_source,
        )

        self._toolbox: dict[str, AtomicInvokable] = {}
        self._blackboard: list[BlackboardSlot] = []
        self._constants: list[ConstantSpec] = []

        if not isinstance(fail_fast, bool):
            raise ToolAgentError("fail_fast must be a bool.")
        self._fail_fast: bool = fail_fast

        if type(generation_retries) is not int or generation_retries < 0:
            raise ToolAgentError("generation_retries must be a non-negative int.")
        self._generation_retries: int = generation_retries

        if generation_format not in ("json", "regex"):
            raise ToolAgentError(
                f"generation_format must be 'json' or 'regex'; got {generation_format!r}."
            )
        self._generation_format: Literal["json", "regex"] = generation_format

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
    def generation_retries(self) -> int:
        """Number of extra generation attempts allowed beyond the first. Read-only."""
        return self._generation_retries

    @property
    def generation_format(self) -> Literal["json", "regex"]:
        """Wire format used for generation output ('json' or 'regex').
        Frozen at construction -- no setter."""
        return self._generation_format

    @property
    def peek_at_cache(self) -> bool:
        """Whether ``render_turn()`` includes each executed cached step's
        raw result value in the LLM-facing history it renders."""
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
    # Toolbox Helpers
    # ------------------------------------------------------------------ #
    def actions_context(self) -> str:
        """String representation of all tools in the toolbox for prompt injection.

        Each tool renders as its own block: the tool's signature line, then
        its description with every non-blank line indented two spaces (blank
        lines are preserved as fully blank, not indent-padded). Blocks are
        separated by "\\n---\\n".

        In regex-mode (``generation_format == "regex"``), the return tool is
        excluded from this listing entirely -- it is reached via the
        ``[RETURN]`` block, never invoked as a named tool call. Applies to
        every ``ToolAgent`` subclass uniformly (both ``PlanActAgent`` and
        ``ReActAgent`` use ``[RETURN]`` sugar in regex-mode).
        """
        tools = [
            t for name, t in self._toolbox.items()
            if self._generation_format != "regex" or name != RETURN_TOOL_FULL_NAME
        ]
        blocks: list[str] = []
        for t in tools:
            indented_lines = [
                line if not line.strip() else f"  {line}"
                for line in t.description.splitlines()
            ]
            blocks.append(f"{t.signature}\n" + "\n".join(indented_lines))
        return "\n---\n".join(blocks)

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
        """Remove all registered tools from the toolbox except the mandatory return tool."""
        self._toolbox.clear()
        self.register(return_tool, name_collision_mode="skip")

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

        AtomicInvokables are stored directly under their own ``full_name``
        when no ``name``/``description`` override is requested. When either
        is supplied, the AtomicInvokable is routed through ``toolify``
        instead, producing a new Tool that delegates to it by reference
        under the requested identity — the original is never mutated. Plain
        callables always go through ``toolify`` with this agent's ``name``
        as their namespace.

        Parameters
        ----------
        component : AtomicInvokable | Callable
            The item to register. AtomicInvokables are stored as-is unless
            an override is requested; callables always go through
            ``toolify(namespace=self.name)``.
        name : str | None
            Override the tool name. For callables, ``None`` infers from
            ``component.__name__``. For AtomicInvokables, supplying this
            routes registration through ``toolify`` instead of direct store.
        description : str | None
            Override the tool description. For callables, ``None`` infers
            from ``component.__doc__``. For AtomicInvokables, supplying this
            routes registration through ``toolify`` instead of direct store.
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
            If ``name_collision_mode`` is invalid; if ``toolify`` fails; or
            if a collision is detected under ``"raise"`` mode.
        """
        name_collision_mode = name_collision_mode.lower().strip()
        if name_collision_mode not in ("raise", "skip", "replace"):
            raise ToolRegistrationError(
                "name_collision_mode must be one of: 'raise', 'skip', 'replace'."
            )

        # AtomicInvokable route — store directly under its own identity,
        # unless an override is requested, in which case toolify it.
        if isinstance(component, AtomicInvokable):
            if name is None and description is None:
                key = component.full_name
                invokable = component
            else:
                try:
                    invokable = toolify(
                        component=component,
                        name=name,
                        description=description,
                        namespace=self.name,
                    )
                except Exception as e:
                    raise ToolRegistrationError(
                        f"{type(self).__name__}.{self.name}: failed to toolify component: {e}"
                    ) from e
                key = invokable.full_name

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
        client: PyA2AtomicClient | MCPClientHub | A2AClientHub | None = None,
        *,
        remote_names: list[str] | None = None,
        name_collision_mode: str = "raise",
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
        client : PyA2AtomicClient | MCPClientHub | A2AClientHub | None
            Remote client to enumerate and register tools from. Combined with
            ``tools`` in one registration pass when both are provided. For an
            ``A2AClientHub``, every discovered Atomic skill is registered in
            skill mode, plus one generic-mode tool registered unconditionally
            (not filtered by ``remote_names`` -- it isn't a discoverable
            skill, it's the hub's baseline reachability path).
        remote_names : list[str] | None
            Whitelist of remote tool names to register from ``client``.
            ``None`` registers all available remote tools. Requires ``client``.
            For an ``A2AClientHub``, filters among skill ids only -- has no
            effect on the always-registered generic tool.
        name_collision_mode : str
            Per-item collision policy for toolbox conflicts. One of
            ``"raise"`` (default), ``"skip"``, or ``"replace"``. Does not
            affect intra-batch dedup, which always raises.

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
            Also raised if ``remote_names`` contains entries not present in the
            client's available tool list.
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
            elif isinstance(client, A2AClientHub):
                available = list(client.get_atomic_skills())
            else:
                available = client.list_invokables()

            if remote_names is not None:
                available_set = set(available)
                missing = [n for n in remote_names if n not in available_set]
                if missing:
                    raise ToolRegistrationError(
                        f"{type(self).__name__}.{self.name}: remote_names entries not found "
                        f"on client: {sorted(missing)!r}."
                    )
                names_to_register = [n for n in available if n in remote_names]
            else:
                names_to_register = available

            for remote_name in names_to_register:
                try:
                    proxy = toolify(
                        component=client,
                        namespace=self.name,
                        remote_name=remote_name,
                    )
                except Exception as exc:
                    raise ToolRegistrationError(
                        f"{type(self).__name__}.{self.name}: failed to toolify remote "
                        f"{remote_name!r}: {exc}"
                    ) from exc
                combined.append((proxy.full_name, proxy))

            # A2A generic tool: always registered when client is an
            # A2AClientHub, unconditionally -- not filtered by remote_names,
            # since it isn't a discoverable skill. A same-named skill
            # colliding with it on full_name is caught by the intra-batch
            # dedup check below, same as any other collision.
            if isinstance(client, A2AClientHub):
                try:
                    generic_proxy = toolify(component=client, namespace=self.name)
                except Exception as exc:
                    raise ToolRegistrationError(
                        f"{type(self).__name__}.{self.name}: failed to toolify "
                        f"generic A2A tool: {exc}"
                    ) from exc
                combined.append((generic_proxy.full_name, generic_proxy))

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
    def _resolve_placeholders(self, obj: Any, *, task: ToolAgentTask) -> Any:
        """
        Resolve all placeholders in an object to their concrete values.

        This method recursively traverses the object structure and replaces placeholder
        references with their concrete runtime values. Placeholders can reference:

        - previously executed current-run steps,
        - previously persisted cache entries,
        - registered ToolAgent constants.

        Supported Placeholder Formats
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        - ``|STEP.N|`` – Result from running step N (0-based, plan-local to this invoke)
        - ``|CACHE.N|`` – Result from cache entry N (0-based, from persisted blackboard)
        - ``|K.NAME|`` – Registered ToolAgent constant named NAME

        The discriminator word (``STEP``/``CACHE``/``K``) is matched
        case-insensitively. ``NAME`` in ``|K.NAME|`` is matched
        case-sensitively against the registered constant name -- ``|K.pi|``
        does NOT resolve a constant registered as ``"PI"``.

        Two resolution modes apply depending on placeholder position:

        1. **Full-String Placeholder**:
           - Returns the referenced value as-is, preserving its type.
           - Examples:
             - ``"|STEP.0|"`` returns step 0's result directly.
             - ``"|CACHE.0|"`` returns cache 0's result directly.
             - ``"|K.PI|"`` returns the registered constant value directly.

        2. **Inline Placeholder**:
           - Replaces the placeholder with ``repr(value)``, falling back to ``str(value)``.
           - For constants only, applies ``ConstantSpec.inline_limit`` if configured.
           - Example:
             - ``"Step returned: |STEP.0|"`` becomes a string.
             - ``"Use constant: |K.NAME|"`` becomes a string, possibly truncated
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
        task : ToolAgentTask
            Execution task containing running_blackboard. Cache references
            are resolved directly against the agent-level, always-persisted
            ``self._blackboard`` (see ``update_blackboard``) — never a
            task-local snapshot.

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
        cache = self._blackboard
        running = task.running_blackboard
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
                status_note = "permanently FAILED" if cache[idx].is_failed() else "not executed"
                raise ToolAgentError(
                    f"Referenced cache {idx} is {status_note} and cannot be resolved."
                )

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
    # Batch preparation helpers (prepare-time, shared by subclasses)
    # ------------------------------------------------------------------ #
    def _compile_batches_from_deps(
        self,
        *,
        planned_slots: list[BlackboardSlot],
        return_idx: int | None = None,
    ) -> list[list[int]]:
        """
        Compile concurrent execution batches from scheduling dependencies.

        For non-return step i: scheduling_deps[i] = step_dependencies plus
        await_step if present; level[i] = 0 if scheduling_deps is empty,
        else 1 + max(level[d] for d in scheduling_deps). Steps at the same
        level share no dependency and may execute concurrently.

        ``return_idx`` is optional so a partial step list with no
        terminating return step (e.g. one round of a future adaptive-batch
        subclass's generation) can be compiled the same way a full plan
        can:

        - Given: isolates it as its own final batch — the caller
          (``PlanActAgent.think()``/``async_think()``) is responsible for
          having already validated it points at a real return-tool slot at
          the final position; this method trusts that contract rather than
          re-checking it.
        - ``None``: every given slot participates in leveling; no isolated
          final batch.

        This method does not itself guard against ``planned_slots`` being
        empty — callers must ensure non-emptiness before calling.
        """
        # Only indices in [0, level_span) participate in leveling; when return_idx
        # is given, that final index is isolated below rather than leveled.
        level_span = return_idx if return_idx is not None else len(planned_slots)

        levels: dict[int, int] = {}
        for i in range(level_span):
            slot = planned_slots[i]

            scheduling_deps: set[int] = set(slot.step_dependencies)
            if slot.await_step is not NO_VAL:
                scheduling_deps.add(slot.await_step)

            if not scheduling_deps:
                levels[i] = 0
            else:
                levels[i] = 1 + max(levels[d] for d in scheduling_deps)

        buckets: dict[int, list[int]] = {}
        for i in range(level_span):
            lvl = levels.get(i, 0)
            buckets.setdefault(lvl, []).append(i)

        batches: list[list[int]] = []
        for lvl in sorted(buckets):
            batch = sorted(buckets[lvl])
            if batch:
                batches.append(batch)

        if return_idx is not None:
            batches.append([return_idx])

        return batches

    def _check_cascade_failure(
        self,
        slot: BlackboardSlot,
        board: list[BlackboardSlot],
    ) -> bool:
        """
        Check one slot's argument-dependencies for already-failed steps and
        mark or raise accordingly.

        Shared by ``PlanActAgent``, ``ReActAgent``, and future ``ToolAgent``
        subclasses. Callers are responsible for only invoking this when
        ``not self._fail_fast`` — this method does not check ``fail_fast``
        itself.

        Uses ``extract_dependencies(slot.args, ...)`` rather than
        ``slot.step_dependencies`` because a return slot's
        ``step_dependencies`` may be forced to include every prior step for
        scheduling purposes; only steps actually referenced in ``args``
        need to resolve successfully.

        Returns
        -------
        bool
            ``True`` if ``slot`` was marked ``FAILED`` due to a failed
            dependency (caller should skip executing it). ``False`` if no
            cascade failure was found.

        Raises
        ------
        ToolAgentError
            If ``slot`` is the return tool and depends on a failed step — a
            return that cannot execute ends the run regardless of
            ``fail_fast``.
        """
        failed_deps = sorted(
            d
            for d in extract_dependencies(slot.args, placeholder_pattern=self.STEP_REF_PATTERN)
            if board[d].is_failed()
        )
        if not failed_deps:
            return False

        dep_str = ", ".join(str(d) for d in failed_deps)

        if slot.tool == RETURN_TOOL_FULL_NAME:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: return step {slot.step} "
                f"cannot execute; dependency step(s) {dep_str} failed."
            )

        slot.error = ToolAgentError(
            f"{type(self).__name__}.{self.name}: step {slot.step} skipped — "
            f"dependency step(s) {dep_str} failed."
        )
        slot.status = BlackboardSlot.FAILED
        return True

    # ------------------------------------------------------------------ #
    # Shared generation-retry loop (PlanActAgent/ReActAgent think() bodies)
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _validate_generation_output(self, raw_output: str, *, task: ToolAgentTask) -> Any | str:
        """
        Re-abstracted per family (same pattern as ``think``/``prepare``).
        Takes the raw LLM string for one generation attempt and owns the
        entire decode-and-validate pipeline itself, including the JSON
        extraction step -- there is no separate loop-level decode phase.

        Returns
        -------
        Any
            The validated, ready-to-use result on success
            (``list[BlackboardSlot]`` for ``PlanActAgent``,
            ``tuple[BlackboardSlot, int]`` for ``ReActAgent``).
        str
            A single, fully-formatted feedback message (via
            ``format_generation_issues``) describing every problem found
            with this attempt -- ready to inject as the next retry turn's
            user message verbatim, no further formatting by the caller.

        No async mirror -- pure computation, no I/O.
        """
        ...
    def _run_generation_retry_loop(self, *, task: ToolAgentTask) -> Any:
        """
        Shared retry loop for one-shot/per-step LLM generation: render, call
        the engine, record the attempt, delegate to the family's own
        ``_validate_generation_output`` hook for decoding *and* validation,
        and retry with the hook's own feedback text until success or the
        retry budget (``self._generation_retries``, tracked via
        ``task.retries_used``) is exhausted.

        Steps
        -----
        1. ``additional_messages`` starts empty.
        2. Loop:
           a. Render this attempt's send payload via ``render_task``.
           b. Call the LLM engine; capture ``engine_result``.
           c. Append an ``LLMRecord`` (``messages=list(task.task_messages)``)
              to ``task.llm_records``.
           d. ``result = self._validate_generation_output(engine_result.result, task=task)``.
           e. If ``result`` is a ``str`` (the hook's own fully-formatted
              feedback, covering every problem found with this attempt):
              budget-check (raise if exhausted); else inject
              ``{"role": "assistant", "content": raw_output}``,
              ``{"role": "user", "content": result}``, increment
              ``task.retries_used``, continue.
           f. Else: return ``result``.

        Raises
        ------
        ToolAgentError
            If the retry budget is exhausted.
        """
        additional_messages: list[dict[str, str]] = []

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)
            engine_result = self._llm_engine.invoke({"messages": messages})
            raw_output: str = engine_result.result

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
                system_prompt_name=task.system_prompt_name,
            ))

            result = self._validate_generation_output(raw_output, task=task)
            if isinstance(result, str):
                if task.retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error(s): {result}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": result},
                ]
                task.retries_used += 1
                continue

            return result

    async def _arun_generation_retry_loop(self, *, task: ToolAgentTask) -> Any:
        """Async mirror of ``_run_generation_retry_loop``: uses
        ``async_invoke`` for the engine call; identical retry logic and
        feedback injection otherwise."""
        additional_messages: list[dict[str, str]] = []

        while True:
            messages = self.render_task(task, additional_messages=additional_messages)
            engine_result = await self._llm_engine.async_invoke({"messages": messages})
            raw_output: str = engine_result.result

            task.llm_records.append(LLMRecord(
                messages=list(task.task_messages),
                llm_result=engine_result,
                system_prompt_name=task.system_prompt_name,
            ))

            result = self._validate_generation_output(raw_output, task=task)
            if isinstance(result, str):
                if task.retries_used >= self._generation_retries:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: generation retry budget exhausted "
                        f"after {task.retries_used + 1} attempt(s). Last error(s): {result}"
                    )
                additional_messages = [
                    {"role": "assistant", "content": raw_output},
                    {"role": "user", "content": result},
                ]
                task.retries_used += 1
                continue

            return result

    # ------------------------------------------------------------------ #
    # Finalization helpers
    # ------------------------------------------------------------------ #
    def update_blackboard(self, task: ToolAgentTask) -> ToolAgentTask:
        """
        Persist all non-empty run slots into the agent's persisted blackboard.

        Called unconditionally by ``_build_record_from_task`` — whether or
        not ``context_enabled`` is set. Always appends onto the *live*
        ``self._blackboard``, read fresh right here rather than from a
        stale invoke-start snapshot. This method already runs under
        ``self._invoke_lock`` (via ``_commit_emit``), so reading live here
        is the actual fix for the mutation-safety concern flagged during
        1a's brainstorming: a stale snapshot as the append base could
        otherwise silently clobber a concurrently-committed invocation's
        own persisted results under the narrower 1a lock scope.
        ``context_enabled`` controls only whether
        ``valid_cache_indices``/``failed_cache_indices`` are populated in
        ``_initialize_task`` (i.e. whether the LLM sees prior steps as
        context) — never whether history is persisted.

        All non-empty slots (EXECUTED and FAILED) are persisted so that global
        blackboard indices remain contiguous and correct. FAILED slots are
        included for index continuity; ``render_turn`` controls whether they
        are surfaced to the LLM.

        Persistence Policy
        ~~~~~~~~~~~~~~~~~~
        1. **Trim empty/unplanned tail**: Remove trailing empty slots from running blackboard
           (slots with no tool assigned)
        2. **Rewrite placeholders**: All ``|STEP.N|`` step references in appended slots'
           args are rewritten to ``|CACHE.{new_global_index}|`` cache references.
           Applied to both EXECUTED and FAILED slots.
        3. **Merge into cache**: Append all non-empty slots (EXECUTED and FAILED)
           preserving status and ``reason``. FAILED slots keep their ``error``
           and no ``result``.
        4. **Trim cache tail**: Remove trailing empty slots from final cache

        Placeholder Rewriting Example
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Initial state:
        - self._blackboard has 5 entries (indices 0-4)
        - running_blackboard has 3 executed entries (indices 0-2)

        Rewriting in appended slots:
        - Step 0's args contain ``|STEP.1|`` → rewritten to ``|CACHE.6|`` (5 + 1)
        - Step 1's args contain ``|STEP.0|`` → rewritten to ``|CACHE.5|`` (5 + 0)
        - Step 2's args contain ``|STEP.1|`` → rewritten to ``|CACHE.6|`` (5 + 1)

        After persistence:
        - self._blackboard now has 8 entries
        - Future invokes can use ``|CACHE.5|``, ``|CACHE.6|``, ``|CACHE.7|``
          to reference steps 0, 1, 2 respectively

        Parameters
        ----------
        task : ToolAgentTask
            Task with executed steps in running_blackboard.

        Returns
        -------
        ToolAgentTask
            Updated task after blackboard persistence.

        Side Effects
        ~~~~~~~~~~~~
        - ``self._blackboard`` is replaced with merged cache + appended slots
        """
        base_cache: list[BlackboardSlot] = [slot.copy() for slot in self._blackboard]
        base_len = len(base_cache)

        running: list[BlackboardSlot] = list(task.running_blackboard)

        # 1) Trim empty/unplanned tail from running plan to avoid caching unused slots.
        last = len(running) - 1
        while last >= 0 and running[last].is_empty():
            last -= 1
        running = running[: last + 1]

        def rewrite_step_to_cache_placeholders(obj: Any) -> Any:
            """
            Rewrite |STEP.j| -> |CACHE.{base_len + j}| recursively.
            Leaves |CACHE.k| unchanged.
            """
            if isinstance(obj, str):
                # exact placeholder: still rewrite as a string placeholder (we are rewriting args,
                # not resolving)
                def repl(m: re.Match[str]) -> str:
                    j = int(m.group(1))
                    return f"|CACHE.{base_len + j}|"

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
        #    FAILED slots are included so local_i always equals the append offset.
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
                reason=slot.reason,
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
        return task

    def _build_record_from_task(
        self,
        task: ToolAgentTask,
        turns: list[AgentRecord],
    ) -> ToolAgentRecord:
        """
        Assemble a completed ToolAgentRecord from a finished ToolAgentTask.

        Persists the run's blackboard slots via ``update_blackboard`` and
        captures the half-open span those slots landed in — mirrors how
        base ``Agent`` always appends ``_records`` regardless of
        ``context_enabled`` (that flag only gates
        ``valid_cache_indices``/``failed_cache_indices`` in
        ``_initialize_task``).
        """
        prev = turns[-1] if turns else None
        blackboard_start = len(self._blackboard)
        task = self.update_blackboard(task)
        blackboard_end = len(self._blackboard)
        return ToolAgentRecord(
            user_prompt=task.user_prompt,
            generated_response=task.generated_response,
            inputs=task.inputs,
            llm_records=tuple(task.llm_records),
            prev=prev,
            blackboard_start=blackboard_start,
            blackboard_end=blackboard_end,
        )

    def build_result_from_record(
        self,
        record: ToolAgentRecord,
        *,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
    ) -> ToolAgentResult:
        """
        Construct this ToolAgent's ToolAgentResult envelope directly from a
        completed ToolAgentRecord.

        Extends ``Agent.build_result_from_record``: re-derives
        ``tool_usage``/``exception_records`` from the record's persisted
        blackboard span (``record.blackboard_start:record.blackboard_end``)
        rather than from task-local bookkeeping — ``update_blackboard``
        preserves each slot's ``status``/``tool`` verbatim and rewrites
        ``step`` to its final global index, so tallying over the persisted
        span gives results identical to tallying over
        ``task.running_blackboard`` directly.
        """
        llm_token_usage = tuple(r.llm_result.token_usage for r in record.llm_records)
        llm_model_data = record.llm_records[-1].llm_result.model_data

        span = self._blackboard[record.blackboard_start:record.blackboard_end]

        # Collect failures only when fail_fast=False (fail_fast=True would have raised).
        exception_records: tuple[tuple[int, Exception], ...] = ()
        if not self._fail_fast:
            exception_records = tuple(
                (slot.step, slot.error)
                for slot in span
                if slot.status == BlackboardSlot.FAILED and isinstance(slot.error, Exception)
            )

        # Derive per-tool call counts from the persisted span.
        _counts: dict[str, int] = {}
        for slot in span:
            if (
                slot.is_executed()
                and isinstance(slot.tool, str)
                and slot.tool != RETURN_TOOL_FULL_NAME
            ):
                _counts[slot.tool] = _counts.get(slot.tool, 0) + 1
        tool_usage = tuple(
            ToolUsageRecord(tool_name=name, call_count=count)
            for name, count in _counts.items()
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

    def _compute_cache_index_sets(
        self,
        turns: list[AgentRecord],
    ) -> tuple[frozenset[int], frozenset[int]]:
        """
        Derive valid and failed cache index sets from the current conversation turns.

        Walks every ``ToolAgentRecord`` in ``turns`` and collects the half-open
        blackboard span ``[blackboard_start, blackboard_end)``. Executed slots
        within those spans go into ``valid``; failed slots go into ``failed``.
        When ``context_enabled=False`` the caller passes an empty turns list
        (or turns with no blackboard spans), so both sets are empty — the LLM
        sees no cache context even though ``self._blackboard`` itself may be
        non-empty (persistence is unconditional; only visibility is gated).

        Steps
        -----
        1. Iterate ``turns``.
        2. Skip non-``ToolAgentRecord`` entries and entries where either boundary
           is ``None``.
        3. For each index in ``range(blackboard_start, blackboard_end)``:
           - If the slot is FAILED → add to ``failed``.
           - If the slot is EXECUTED → add to ``valid``.
        4. Return ``(frozenset(valid), frozenset(failed))``.
        """
        valid: set[int] = set()
        failed: set[int] = set()
        for turn in turns:
            if not isinstance(turn, ToolAgentRecord):
                continue
            if turn.blackboard_start is None or turn.blackboard_end is None:
                continue
            for idx in range(turn.blackboard_start, turn.blackboard_end):
                if idx >= len(self._blackboard):
                    continue
                slot = self._blackboard[idx]
                if slot.is_failed():
                    failed.add(idx)
                elif slot.is_executed():
                    valid.add(idx)
                else:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: internal error: persisted blackboard "
                        f"slot {idx} has unexpected status {slot.status!r}; expected EXECUTED or FAILED."
                    )
        return frozenset(valid), frozenset(failed)

    def render_turn(self, turn: AgentRecord) -> list[dict[str, str]]:
        """Render one stored ToolAgentRecord into LLM-facing user/assistant messages.

        The base assistant response is rendered through `Agent.render_turn(...)`, preserving
        `assistant_response_source` and `response_preview_limit` behavior. Everything after
        that point branches on ``self.generation_format``:

        - ``"json"`` (default): if the turn has a non-empty blackboard span and all slots
          executed, this method appends a cached-step block (``CACHED STEPS`` section) with
          each produced step's unresolved args and ``run_id``. When some slots are FAILED
          (``fail_fast=False``), the output splits into a ``CACHED STEPS`` section for
          executed slots and a ``FAILED STEPS`` section for failed slots; failed entries
          include step index, tool name, and truncated error string — no args. Result
          previews are included only when `peek_at_cache=True` and are bounded by
          `blackboard_preview_limit`.
        - ``"regex"``: see ``_render_regex_turn_body`` for the tag-formatted equivalent.
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

        if self._generation_format == "json":
            assistant_content = self._render_json_turn_body(start, end, assistant_response)
        else:
            assistant_content = self._render_regex_turn_body(start, end, assistant_response)

        return [
            user_message,
            {"role": "assistant", "content": assistant_content},
        ]

    def _render_json_turn_body(self, start: int, end: int, assistant_response: str) -> str:
        """
        json-mode ``render_turn`` body — unchanged from the pre-regex-mode
        implementation, extracted verbatim into its own method so
        ``render_turn`` can branch on ``generation_format`` cleanly.
        """
        executed: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        for slot in self._blackboard[start:end]:
            if slot.is_executed():
                entry: dict[str, Any] = {
                    STEP_FIELD: slot.step,
                    TOOL_FIELD: slot.tool,
                    ARGS_FIELD: slot.args,
                    "run_id": slot.result.run_id,
                }
                if self.peek_at_cache:
                    entry["result"] = self._preview_blackboard_result(slot.result.result)
                executed.append(entry)
            elif slot.is_failed():
                err_str = str(slot.error)
                if self.blackboard_preview_limit is not None:
                    err_str = err_str[:self.blackboard_preview_limit]
                failed.append({
                    STEP_FIELD: slot.step,
                    TOOL_FIELD: slot.tool,
                    "error": err_str,
                })
            # Other statuses (PLANNED, PREPARED, EMPTY) cannot appear in a persisted
            # blackboard span — silently skipped if present.

        if not failed:
            # All-executed path: format unchanged.
            dump = pprint.pformat(executed, indent=2, width=160, sort_dicts=False)
            return (
                f"RESPONSE:\n{assistant_response}\n\n"
                f"CACHED STEPS {list(range(start, end))} PRODUCED:\n\n{dump}"
            )

        # Mixed path: two-section output.
        parts = [f"RESPONSE:\n{assistant_response}"]
        if executed:
            ex_indices = [e[STEP_FIELD] for e in executed]
            parts.append(
                f"CACHED STEPS {ex_indices} PRODUCED:\n\n"
                + pprint.pformat(executed, indent=2, width=160, sort_dicts=False)
            )
        fa_indices = [f[STEP_FIELD] for f in failed]
        parts.append(
            f"FAILED STEPS {fa_indices}:\n\n"
            + pprint.pformat(failed, indent=2, width=160, sort_dicts=False)
        )
        return "\n\n".join(parts)

    def _render_regex_turn_body(self, start: int, end: int, assistant_response: str) -> str:
        """
        regex-mode ``render_turn`` body — tag-formatted equivalent of
        ``_render_json_turn_body``, sharing no code with it (the two
        formats' shapes diverge too much for a shared helper to pay off).

        1. ``response_section`` is ``f"POST-PROCESSED RESPONSE:\\n{assistant_response}"``
           when ``self.assistant_response_source == "final"``; ``None`` when
           ``"raw"`` (the default) — regex-mode never echoes the raw
           generated text back as a "RESPONSE:" section, since that text is
           already the same tag vocabulary the structured entries below
           re-present.
        2. ``args_for(slot)`` returns ``slot.resolved_args`` when
           ``self.peek_at_cache`` is true and ``slot.resolved_args is not
           NO_VAL``, else ``slot.args`` — ``peek_at_cache`` governs both
           this raw-vs-resolved selection and (for ordinary slots)
           whether ``[RESULT]`` is shown at all.
        3. For each slot in ``self._blackboard[start:end]``:
           - Executed, ``slot.tool == RETURN_TOOL_FULL_NAME``: block is
             ``"** |CACHE.{slot.step}| RUN_ID={slot.result.run_id} **"``
             then ``"[RETURN] {args_for(slot)[RETURN_VALUE_FIELD]!r}"`` —
             the bare value only, never the wrapping dict. Appended to
             ``executed_blocks``.
           - Executed, otherwise: same header, then one fused
             ``"[EVENT] {slot.tool}({kwargs_str})"`` line (``kwargs_str``
             joins ``args_for(slot)`` as ``key=value!r`` pairs — mirrors the
             generation-side call-expression shape, not a bare name +
             separate dict), and, only if ``self.peek_at_cache``,
             ``"[RESULT] {preview}"`` (via ``_preview_blackboard_result``,
             already a possibly-truncated ``str`` — no extra ``repr()``).
             Appended to ``executed_blocks``.
           - Failed: ``"** FAILED |CACHE.{slot.step}| **"`` (no ``RUN_ID`` —
             no result object exists), the same fused
             ``"[EVENT] {slot.tool}({kwargs_str})"`` line, ``"[ERROR]
             {err_str}"`` (``err_str`` computed identically to the json
             body's failed entries). Never special-cased into ``[RETURN]``
             shape even if this was the return tool. Appended to
             ``failed_blocks``.
           - Other statuses: silently skipped (same invariant as json-mode).
        4. ``banner`` is ``f"** CACHE ENTRIES {start}-{end - 1} PRODUCED **"``
           when ``executed_blocks`` is non-empty, else ``None``.
        5. Assemble ``parts`` in order, skipping ``None``/empty entries, and
           join with ``"\\n\\n"``:
           - ``response_section``.
           - ``banner + "\\n\\n" + "\\n\\n".join(executed_blocks)`` if
             ``banner`` is not ``None``.
           - ``"\\n\\n".join(failed_blocks)`` if non-empty — always
             individually headered, never grouped under a shared banner.
        6. Return the joined ``parts``.
        """
        def args_for(slot: BlackboardSlot) -> Any:
            if self.peek_at_cache and slot.resolved_args is not NO_VAL:
                return slot.resolved_args
            return slot.args

        executed_blocks: list[str] = []
        failed_blocks: list[str] = []

        for slot in self._blackboard[start:end]:
            if slot.is_executed():
                header = f"** |CACHE.{slot.step}| {RUN_ID_TAG}={slot.result.run_id} **"
                if slot.tool == RETURN_TOOL_FULL_NAME:
                    lines = [
                        header,
                        f"[{RETURN_TAG}] {args_for(slot)[RETURN_VALUE_FIELD]!r}",
                    ]
                else:
                    kwargs_str = ", ".join(f"{k}={v!r}" for k, v in args_for(slot).items())
                    lines = [
                        header,
                        f"[{EVENT_TAG}] {slot.tool}({kwargs_str})",
                    ]
                    if self.peek_at_cache:
                        lines.append(
                            f"[{RESULT_TAG}] {self._preview_blackboard_result(slot.result.result)}"
                        )
                executed_blocks.append("\n".join(lines))
            elif slot.is_failed():
                err_str = str(slot.error)
                if self.blackboard_preview_limit is not None:
                    err_str = err_str[:self.blackboard_preview_limit]
                kwargs_str = ", ".join(f"{k}={v!r}" for k, v in args_for(slot).items())
                lines = [
                    f"** FAILED |CACHE.{slot.step}| **",
                    f"[{EVENT_TAG}] {slot.tool}({kwargs_str})",
                    f"[{ERROR_TAG}] {err_str}",
                ]
                failed_blocks.append("\n".join(lines))
            # Other statuses (PLANNED, PREPARED, EMPTY) cannot appear in a persisted
            # blackboard span — silently skipped if present.

        parts: list[str] = []
        if self.assistant_response_source == "final":
            parts.append(f"POST-PROCESSED RESPONSE:\n{assistant_response}")

        if executed_blocks:
            banner = f"** CACHE ENTRIES {start}-{end - 1} PRODUCED **"
            parts.append(banner + "\n\n" + "\n\n".join(executed_blocks))

        if failed_blocks:
            parts.append("\n\n".join(failed_blocks))

        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # String to JSON Objects helper
    # ------------------------------------------------------------------ #
    def _extract_from_json_string(self, raw_text: str) -> Any:
        """
        Extract the largest decodable JSON array/object from a possibly noisy string.

        Thin delegating wrapper — see ``utils.agents.extract_json_object``
        for the full contract (parsing logic lives there now, shared with
        any other caller that needs to pull structured output out of
        free-form LLM text).

        Raises
        ------
        TypeError
            If ``raw_text`` is not a string (engine contract violation).
            Changed from ``ToolAgentError`` when the parsing logic was
            promoted to ``extract_json_object`` — a ``ToolAgent``-specific
            exception type no longer fit a shared utility.
        json.JSONDecodeError
            If ``raw_text`` is empty or contains no decodable JSON array/object.
        """
        return extract_json_object(raw_text, source_label=f"{type(self).__name__}.{self.name}")

    def _extract_from_regex_string(self, raw_text: str) -> tuple[list[dict[str, Any]], list[str]]:
        """
        Thin delegating wrapper — see ``utils.agents.extract_regex_steps``
        for the full contract. Mirrors ``_extract_from_json_string``'s role
        exactly.

        Raises
        ------
        TypeError
            If ``raw_text`` is not a string (engine contract violation).
        """
        return extract_regex_steps(raw_text, source_label=f"{type(self).__name__}.{self.name}")

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
    ) -> dict[str, Any] | list[str]:
        """
        Validate and normalize one raw LLM-produced ToolAgent step mapping.

        The caller provides explicit field bounds:
        - ``allowed_fields`` is the maximum allowed key set.
        - ``required_fields`` is the minimum required key set.

        ``context`` is only for error messages. This method does not infer PlanAct,
        ReAct, or base-step behavior from context.

        Runtime owns the authoritative step index. Any LLM-provided ``step`` value
        is advisory and is always overwritten with ``expected_step``.

        Every applicable rule is checked independently and accumulated,
        rather than returning at the first violation -- lets a caller
        report every problem with this one step in a single retry turn.

        Returns
        -------
        dict[str, Any]
            Validated and normalized step mapping on success.
        list[str]
            LLM-facing feedback strings, one per violation found. No
            class/name prefix. Returned (not raised) so the caller decides
            whether to retry.
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

        data_keys = set(data)
        issues: list[str] = []

        extra = data_keys - allowed
        if extra:
            issues.append(f"{context} {expected_step} contains unsupported keys: {sorted(extra)!r}.")

        missing = required - data_keys
        if missing:
            issues.append(f"{context} {expected_step} is missing required keys: {sorted(missing)!r}.")

        normalized = dict(data)

        # Advisory/fallback behavior:
        # - If the LLM omitted "step", fill it.
        # - If the LLM supplied a wrong/non-sequential "step", ignore it.
        # Runtime order is authoritative.
        normalized[STEP_FIELD] = expected_step

        # Type checks below only run when the field is actually present --
        # a field already reported missing above is never also type-checked,
        # which would just re-flag the same root cause a second time.
        tool = normalized.get(TOOL_FIELD, NO_VAL)
        if TOOL_FIELD in normalized:
            if not isinstance(tool, str) or not tool.strip():
                issues.append(f"{context} {expected_step} 'tool' must be a non-empty string.")

        args = normalized.get(ARGS_FIELD, NO_VAL)
        if ARGS_FIELD in normalized:
            if not isinstance(args, dict):
                issues.append(f"{context} {expected_step} 'args' must be a dict; got {type(args).__name__!r}.")

        if AWAIT_FIELD in normalized:
            await_step = normalized[AWAIT_FIELD]
            if type(await_step) is not int or await_step < 0:
                issues.append(f"{context} {expected_step} 'await' must be an int >= 0.")
            if tool == RETURN_TOOL_FULL_NAME:
                issues.append(f"{context} {expected_step} is a return step and must not include 'await'.")

        if issues:
            return issues

        return normalized

    def _validate_cache_refs(
        self,
        *,
        args: dict[str, Any],
        context: str,
        cache_blackboard: list[BlackboardSlot],
        valid_cache_indices: frozenset[int],
        failed_cache_indices: frozenset[int],
    ) -> list[str]:
        """
        Validate ``|CACHE.N|`` references in ``args`` against three
        independent categories: out-of-range, failed-in-this-conversation,
        and in-range-but-not-part-of-this-conversation.

        ``context`` is a pre-formatted label (e.g. ``"plan step 3"`` or
        ``"next step"``) used verbatim in every issue string -- unifies
        ``PlanActAgent``'s per-slot and ``ReActAgent``'s per-round versions
        of this check, which were word-for-word identical except for this
        label.
        """
        issues: list[str] = []
        cache_len = len(cache_blackboard)
        cache_refs = extract_dependencies(args, placeholder_pattern=self.CACHE_REF_PATTERN)

        out_of_range = [idx for idx in cache_refs if idx < 0 or idx >= cache_len]
        if out_of_range:
            issues.append(
                f"{context} references cache indices that do not exist: "
                f"{sorted(set(out_of_range))!r} (cache has {cache_len} entries)."
            )

        failed_in_conv = [idx for idx in cache_refs if idx in failed_cache_indices]
        if failed_in_conv:
            details = "; ".join(
                f"entry {idx} ({cache_blackboard[idx].tool}): {cache_blackboard[idx].error}"
                for idx in sorted(set(failed_in_conv))
            )
            issues.append(
                f"{context} references cache entries that failed in this "
                f"conversation and cannot be used: {details}."
            )

        out_of_conv = [
            idx for idx in cache_refs
            if 0 <= idx < cache_len
            and idx not in valid_cache_indices
            and idx not in failed_cache_indices
        ]
        if out_of_conv:
            issues.append(
                f"{context} references cache indices not part of this "
                f"conversation: {sorted(set(out_of_conv))!r}."
            )

        return issues

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

        This method is a converter, not the primary raw-LLM schema validator.
        ``allowed_fields`` is validated as a well-formed programmer-supplied set
        (via ``_normalize_step_field_set``). It does not filter ``data`` — callers
        are responsible for ensuring ``data`` only contains expected keys before
        calling this method. Required-field validation should already have happened
        in ``_validate_tool_step_dict(...)``.
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

        self._normalize_step_field_set(
            allowed_fields,
            name="allowed_fields",
            require_non_empty=True,
        )

        tool = data.get(TOOL_FIELD, NO_VAL)
        args = data.get(ARGS_FIELD, NO_VAL)

        await_step = NO_VAL
        if AWAIT_FIELD in data:
            await_step = data[AWAIT_FIELD]

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
    # Task-lifecycle hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _initialize_task(
        self,
        *,
        turns: list[AgentRecord],
        prompt: str,
        inputs: dict,
    ) -> ToolAgentTask:
        """
        Re-declared abstract at this level: base ``Agent``'s concrete
        implementation returns a bare ``AgentTask``, which isn't sufficient
        here — only ``PlanActAgent``/``ReActAgent`` know how to build their
        own richer ``ToolAgentTask``-family subclass.

        Same three base parameters as ``Agent``'s own hook — no
        ``valid_cache_indices``/``failed_cache_indices``; the real
        implementation computes those internally via
        ``_compute_cache_index_sets``.
        """
        ...

    @abstractmethod
    def think(self, task: ToolAgentTask) -> ToolAgentTask:
        """
        Make this round's real decision via the LLM.

        Hard-abstract — every ``ToolAgent`` family has genuine generation
        work here, overriding base ``Agent``'s no-op default. A concrete
        override renders via ``self.render_task(task)``, calls the engine,
        parses/validates the raw output against this family's schema, and
        stores the validated decision onto ``task``. Responsible for
        guaranteeing, before any slot ever reaches ``prepare``: the tool
        name is registered (``self.has_tool(...)``) and the decision as a
        whole respects ``tool_calls_limit`` — neither is re-checked
        downstream. May no-op on a later round once there's nothing
        further to decide (e.g. a one-shot planner with an
        already-compiled plan).
        """
        ...

    @abstractmethod
    async def async_think(self, task: ToolAgentTask) -> ToolAgentTask:
        """Async mirror of ``think``. No default — same rationale as base
        ``Agent.async_act``: every family that reaches this hook performs
        real I/O of its own."""
        ...

    @abstractmethod
    def prepare(self, task: ToolAgentTask) -> ToolAgentTask:
        """
        Turn this round's decision into something ``act`` can run, with no
        further LLM calls.

        Hard-abstract — each family's resolve/cascade logic differs enough
        (batch-cursor vs. step-cursor) that no shared body fits both. A
        concrete override cascade-checks dependencies
        (``self._check_cascade_failure``), resolves placeholders
        (``self._resolve_placeholders``), marks surviving slots
        ``PREPARED``, and populates ``task.prepared_steps`` — advancing
        whatever cursor this family uses. ``act`` trusts this hook
        completely: every index landing in ``task.prepared_steps`` must be
        in-bounds, unique, not already executed, and (batch semantics)
        contain at most one return call. An empty ``task.prepared_steps``
        on return (a fully cascade-skipped round) is legal, not an error.
        """
        ...

    @abstractmethod
    async def async_prepare(self, task: ToolAgentTask) -> ToolAgentTask:
        """Async mirror of ``prepare``. No default — a family with a real
        per-step generation call folded in elsewhere may need this to be
        genuinely async; no shared body fits both families."""
        ...

    def _render_system_message(self, task: ToolAgentTask) -> list[dict[str, str]]:
        """
        Render this ``ToolAgent``'s active system prompt with tool/constant
        context injected.

        Overrides base ``Agent``'s ``task.inputs``-only rendering — neither
        ``PLANNER_PROMPT`` nor ``ORCHESTRATOR_PROMPT`` ever uses an
        input-derived placeholder, only ``{TOOLS}``/``{TOOL_CALLS_LIMIT}``/
        ``{CONSTANTS}`` — so this builds that context directly instead of
        merging with ``task.inputs``. Shared by every ``ToolAgent``
        subclass; eliminates the identical ``render_context`` dict each one
        built independently before this sub-pass.
        """
        if task.system_prompt_name is None:
            return []
        limit_text = "unlimited" if self._tool_calls_limit is None else str(self._tool_calls_limit)
        render_context = {
            self.TOOLS_FIELD: self.actions_context(),
            self.LIMIT_FIELD: limit_text,
            self.CONSTANTS_FIELD: self.constants_context(),
        }
        rendered = self._system_prompts[task.system_prompt_name].render(render_context)
        return [{"role": "system", "content": rendered}]

    def _render_task_banner(self, task: ToolAgentTask) -> dict[str, str]:
        """
        Return the "what is the task" user message shared by every
        ``ToolAgent`` subclass's ``_render_task_messages``.

        Deduplicates the identical ``===== CURRENT TASK =====`` banner
        ``planact.py``/``react.py`` each built independently before this
        sub-pass. No caller in this file — 1d/1e's own
        ``_render_task_messages`` consumes it (directly as its own message,
        or with an instruction appended onto its ``content``).
        """
        return {
            "role": "user",
            "content": f"===== CURRENT TASK =====\n{task.user_prompt}\n===== END TASK =====",
        }

    def _apply_batch_results(
        self,
        task: ToolAgentTask,
        indices: list[int],
        board: list[BlackboardSlot],
        raw_results: list[Any],
    ) -> ToolAgentTask:
        """
        Shared post-gather bookkeeping for ``act``/``async_act``: partition
        ``raw_results`` (paired positionally with ``indices``) into
        failures/successes, apply ``fail_fast`` handling, mutate slots, and
        check for run completion. Both callers differ only in *how*
        ``raw_results`` was gathered (``run_coro_sync``-wrapped vs. awaited
        directly) — everything after that point is identical, so it lives
        here once instead of twice.

        Failure handling: ``fail_fast=True`` (default) marks and raises on
        the first failed step; ``fail_fast=False`` marks every failed slot
        but only raises if the return tool itself failed, otherwise falling
        through to record the surviving successes.
        """
        non_return_planned = 0
        return_indices: list[int] = []
        for idx in indices:
            if board[idx].tool == RETURN_TOOL_FULL_NAME:
                return_indices.append(idx)
            else:
                non_return_planned += 1

        pairs = list(zip(indices, raw_results))
        failures = [(idx, raw) for idx, raw in pairs if isinstance(raw, BaseException)]
        successes = [(idx, raw) for idx, raw in pairs if not isinstance(raw, BaseException)]

        if failures:
            if self._fail_fast:
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
                for idx, raw_error in failures:
                    if isinstance(raw_error, ToolInvocationError):
                        board[idx].error = raw_error
                    else:
                        board[idx].error = ToolAgentError(
                            f"{type(self).__name__}.{self.name}: tool call failed at step {idx} "
                            f"for {board[idx].tool!r}: {raw_error}"
                        )
                    board[idx].status = BlackboardSlot.FAILED

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
            task.executed_steps.add(idx)

        task.tool_calls_used += non_return_planned
        task.prepared_steps = []

        if return_indices:
            ret_idx = return_indices[0]
            if board[ret_idx].is_executed():
                task.generated_response = board[ret_idx].result.result
                task.complete = True

        return task

    def act(self, task: ToolAgentTask) -> ToolAgentTask:
        """
        Execute the currently prepared batch concurrently, or no-op if
        ``prepare`` produced nothing to run this round.

        Concrete and final — no ``ToolAgent`` subclass overrides this.
        Trusts ``prepare``'s (and, for tool existence/budget, ``think``'s)
        contract completely: every index in ``task.prepared_steps`` is
        assumed in-bounds, unique, not yet executed, marked ``PREPARED``,
        naming a registered tool, and within ``tool_calls_limit`` — none of
        that is re-validated here. A cascade-skipped round (``prepare``
        left ``task.prepared_steps`` empty) is not an error: this method
        just returns ``task`` unchanged and the outer loop tries again next
        round.

        Gathers results via a ``run_coro_sync``-wrapped coroutine (this
        method itself is sync); ``_apply_batch_results`` does everything
        after that, shared verbatim with ``async_act``.
        """
        logger.debug(f"{type(self).__name__}.{self.name} has made {task.tool_calls_used} this run")
        if not task.prepared_steps:
            return task

        indices = list(task.prepared_steps)
        board = task.running_blackboard

        async def run_batch() -> list[Any]:
            # Resolve every tool in the batch before constructing any
            # coroutine -- a later index naming an unregistered tool must
            # not leave an earlier index's already-created (and now
            # unawaited) coroutine behind.
            tools = [self.get_tool(board[idx].tool) for idx in indices]
            coros: list[Any] = []
            for idx, tool in zip(indices, tools):
                slot = board[idx]
                logger.debug(
                    f"{type(self).__name__}.{self.name}:\nTool: {slot.tool}\nArgs: {slot.args}\n\n"
                )
                coros.append(tool.async_invoke(slot.resolved_args))
            return await asyncio.gather(*coros, return_exceptions=True)

        raw_results = run_coro_sync(run_batch())
        return self._apply_batch_results(task, indices, board, raw_results)

    async def async_act(self, task: ToolAgentTask) -> ToolAgentTask:
        """Async mirror of ``act``; same trust/trim contract, awaits each
        tool's ``async_invoke`` directly rather than wrapping in
        ``run_coro_sync``. ``_apply_batch_results`` does everything past the
        gather step, shared verbatim with ``act``."""
        logger.debug(f"{type(self).__name__}.{self.name} has made {task.tool_calls_used} this run")
        if not task.prepared_steps:
            return task

        indices = list(task.prepared_steps)
        board = task.running_blackboard

        # Resolve every tool in the batch before constructing any coroutine
        # -- a later index naming an unregistered tool must not leave an
        # earlier index's already-created (and now unawaited) coroutine
        # behind.
        tools = [self.get_tool(board[idx].tool) for idx in indices]
        coros: list[Any] = []
        for idx, tool in zip(indices, tools):
            slot = board[idx]
            logger.debug(
                f"{type(self).__name__}.{self.name}:\nTool: {slot.tool}\nArgs: {slot.args}\n\n"
            )
            coros.append(tool.async_invoke(slot.resolved_args))

        raw_results = await asyncio.gather(*coros, return_exceptions=True)
        return self._apply_batch_results(task, indices, board, raw_results)

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of this ToolAgent.

        Extends the base Agent snapshot with ToolAgent-specific toolbox and blackboard
        diagnostics.
        """
        d = super().to_dict()
        d.update({
            "tool_calls_limit": self.tool_calls_limit,
            "fail_fast": self._fail_fast,
            "generation_retries": self._generation_retries,
            "generation_format": self._generation_format,
            "peek_at_cache": self.peek_at_cache,
            "blackboard_preview_limit": self.blackboard_preview_limit,
            "tools": {
                name: tool.to_dict()
                for name, tool in self._toolbox.items()
            },
            "blackboard": self.blackboard_serialized(peek=False),
        })
        return d
