"""
ToolAgent2: LLM-Driven Tool Calling on the Agent2 think/prepare/execute Lifecycle

Reworks the abstract ``ToolAgent`` base (``agents/toolagent.py``) onto
``Agent2``'s shared task-oriented lifecycle
(``_initialize_task`` -> ``while not task.complete: think -> prepare ->
execute``) instead of the retired ``_progress`` model. Toolbox/constants
management, placeholder resolution, and batch-compilation helpers are
carried over from v1 ``ToolAgent`` unchanged -- none of that is coupled to
the lifecycle shape. Thinking becomes an optional, gateable lifecycle
feature shared uniformly with every other ``Agent2`` subclass, rather than a
separate agent family.

Two real fixes land alongside the lifecycle rework, found while designing
it:

- **Blackboard persistence.** v1 ``update_blackboard`` used a task-local
  snapshot (``cache_blackboard``, populated only when ``context_enabled``)
  as its append base, which silently *replaced* ``self._blackboard`` with
  just the current run's own slots whenever ``context_enabled=False`` --
  discarding prior history and, worse, corrupting ``blackboard_start``/
  ``blackboard_end`` bookkeeping on the second such invoke onward.
  ``ToolAgent2.update_blackboard`` always appends onto the live
  ``self._blackboard`` instead, regardless of ``context_enabled`` -- that
  flag now only gates whether prior results are fed to the LLM as context,
  never whether history is retained.
- **Lock scope.** ``Agent2.invoke``/``async_invoke`` now lock only the final
  ``_commit_emit`` call (see ``base2.py``), not the whole invocation -- safe
  once persistence is unconditional, since every other read of
  ``self._blackboard``/``self._records`` is stale-safe by construction
  (append-only, immutable-once-persisted).

See ``.claude/brainstorms/toolagent2-lifecycle.md`` for the full design
record and ``.claude/specs/toolagent2-base.md`` for this file's spec.

Subclass Responsibilities
--------------------------
Concrete subclasses (``PlanActAgent2``, ``ReActAgent2`` -- not yet built,
out of scope for this pass) must implement:

**_initialize_task(*, turns, prompt, inputs)** -> ``ToolAgentTask`` (or
subclass)
  Build and return a task for this invocation: stamp
  ``system_prompt_name="think"`` (matching ``Agent2._initialize_task``'s own
  seeding convention -- the subclass transitions away from "think" during
  its own ``_prepare_next_batch``, not here); compute
  ``valid_cache_indices``/``failed_cache_indices`` via
  ``_compute_cache_index_sets(turns)``; allocate a running blackboard; seed
  ``task.keep_thinking = self._thinking_rounds != 0``.

**_prepare_next_batch(task)** -> ``ToolAgentTask`` (or subclass)
  Prepare exactly one executable batch: decide which tool call(s) to invoke,
  validate names/dependencies/budget, resolve placeholders via
  ``self._resolve_placeholders(...)``, populate ``task.prepared_steps``.
  Also owns this subclass's own phase-transition stamp (system prompt name
  + ``task_messages`` clear) -- the two families register different
  action-phase prompt names and clear ``task_messages`` on different
  cadences, so this can't be hoisted onto the shared ``prepare()`` below.

**_render_task_messages(task)** -> ``list[dict[str, str]]`` (inherited
abstract from ``Agent2``, not re-declared here)
  Every ``ToolAgent2`` subclass fully implements this, delegating its
  "think" branch to ``self._render_thinking_task_messages(task)`` (already
  generic, no ``ToolAgent2``-specific dispatch needed).
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

from .base2 import Agent2
from .prompts import TOOL_THINKING_PROMPT
from .tools import return_tool
from ..constants.agents import (
    ARGS_FIELD,
    AWAIT_FIELD,
    RETURN_TOOL_FULL_NAME,
    STEP_FIELD,
    TOOL_FIELD,
)
from ..models.agents.records import AgentRecord, ToolAgentRecord
from ..models.results.agents import ToolAgentResult, ToolUsageRecord
from ..models.agents.blackboard_models import BlackboardSlot, ConstantSpec
from ..models.agents.prompts import PromptConfig
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
from ..a2a import PyA2AtomicClient
from ..utils.agents import extract_dependencies, extract_json_object
from ..utils.core import run_coro_sync


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Base ToolAgent2
# --------------------------------------------------------------------------- #
class ToolAgent2(Agent2, ABC):
    """
    Abstract base for tool-using agents on the Agent2 lifecycle.

    Owns the blackboard slot system, toolbox/constants registries, and
    placeholder-resolution/batch-compilation machinery -- identical to v1
    ``ToolAgent`` in every respect not coupled to the retired ``_progress``
    loop shape. ``prepare``/``execute`` (and their async mirrors) are
    concrete and final here, each gated by ``task.keep_thinking`` unless
    ``self._permits_unbounded_thinking`` (a future ``ReActAgent2`` sets this
    to decouple its action phase from an active thinking phase entirely).
    ``_initialize_task``/``_prepare_next_batch``/``_render_task_messages``
    stay abstract -- real per-family bodies are each concrete subclass's own
    job, same division of labor v1 ``ToolAgent`` already established for
    ``_initialize_task``/``_prepare_next_batch``/``render_task``.
    """
    TOOLS_FIELD = "TOOLS"
    LIMIT_FIELD = "TOOL_CALLS_LIMIT"
    CONSTANTS_FIELD = "CONSTANTS"

    STEP_REF_PATTERN: re.Pattern[str] = re.compile(
        r"<<__s(\d+)__>>"
    )
    CACHE_REF_PATTERN: re.Pattern[str] = re.compile(
        r"<<__c(\d+)__>>"
    )
    CONST_REF_PATTERN: re.Pattern[str] = re.compile(
        rf"<<__k\.({IDENTIFIER_PATTERN_TEXT})__>>"
    )

    # _permits_unbounded_thinking inherited from Agent2 (False, not
    # overridden here) -- a future ReActAgent2 overrides it via plain
    # class-body reassignment, mirroring _THINK_PROMPT's own override
    # convention below.
    _THINK_PROMPT: PromptConfig = TOOL_THINKING_PROMPT

    def __init__(
        self,
        name: str,
        namespace: str,
        description: str,
        llm_engine: LLMEngine,
        filter_extraneous_inputs: Optional[bool] = None,
        context_enabled: bool = False,
        fail_fast: bool = True,
        generation_retries: int = 0,
        tool_calls_limit: Optional[int] = None,
        peek_at_cache: bool = False,
        response_preview_limit: Optional[int] = None,
        blackboard_preview_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_result_key: Optional[str] = None,
        records_window: Optional[int] = None,
        assistant_response_source: Literal["raw", "final"] = "raw",
        thinking_rounds: int | None = 0,
        thoughts_per_round: int = 1,
        thoughts_window: int | None = 0,
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
        filter_extraneous_inputs : bool | None
            When ``True``, inputs not declared in ``parameters`` are silently
            dropped before invocation. When ``False``, extraneous inputs raise.
            ``None`` inherits the base class default.
        context_enabled : bool
            When ``True``, prior blackboard steps are fed into each invocation
            as LLM context (``valid_cache_indices``/``failed_cache_indices``
            become non-empty). The blackboard is always persisted after each
            invoke regardless of this setting. Defaults to ``False``.
        fail_fast : bool
            When ``True`` (default), the first tool call failure immediately
            raises and aborts the run. When ``False``, failing slots are marked
            ``FAILED`` and the loop continues to execute independent steps;
            failures are collected in ``ToolAgentResult.exception_records``.
            Return-tool failures always raise regardless of this setting.
        generation_retries : int
            Number of additional LLM generation attempts to make when a
            subclass's own generation output cannot be parsed or fails spec
            validation. ``0`` (default) means a single attempt with no
            retries; ``N`` means up to ``N`` extra attempts beyond the first.
            Must be a non-negative ``int``.
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
        records_window : int | None
            Maximum number of prior ``AgentRecord`` turns rendered into LLM
            context. ``None`` means all records are rendered.
        assistant_response_source : Literal["raw", "final"]
            Whether rendered assistant history uses raw or final turn
            responses. Re-exposed here (v1 ``ToolAgent`` never surfaced it,
            always implicitly ``"raw"``) for parity with ``BasicAgent2``.
        thinking_rounds : int | None
            Thinking-round hard cap. ``0`` (default) disables thinking
            entirely. ``None`` requires ``self._permits_unbounded_thinking``
            (``False`` here, inherited from ``Agent2`` -- a future
            ``ReActAgent2`` opts in).
        thoughts_per_round : int
            Max thoughts kept per thinking round; extras silently dropped.
        thoughts_window : int | None
            Trailing-record window governing how many past records' thoughts
            get replayed into history. ``0`` means none, ``None`` means all.

        Notes
        -----
        ``thinking_instructions`` is deliberately not exposed here. Tools
        already carry their own descriptions/usage rules, and the thinking
        phase is already grounded in this agent's actual
        tools/constants/call-limit via ``_render_system_prompt`` below -- a
        separate caller-supplied "how to think" block would be a second,
        potentially-conflicting source of guidance. ``Agent2``'s own
        "invisible when empty" splice already hides the corresponding
        template section with zero further work, since this class never
        forwards a non-``None`` value up.
        """
        super().__init__(
            name=name,
            namespace=namespace,
            description=description,
            llm_engine=llm_engine,
            filter_extraneous_inputs=filter_extraneous_inputs,
            context_enabled=context_enabled,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            post_result_key=post_result_key,
            records_window=records_window,
            response_preview_limit=response_preview_limit,
            assistant_response_source=assistant_response_source,
            thinking_rounds=thinking_rounds,
            thoughts_per_round=thoughts_per_round,
            thoughts_window=thoughts_window,
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
        applies consistently with rendered-turn observable rendering.
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
        """Clear the stored turn history, thought history, and the persisted
        blackboard. ``super().clear_memory()`` now clears both
        ``self._records`` and ``self._thoughts`` (``Agent2``'s own body) --
        v1 ``ToolAgent``'s ``super()`` only cleared ``self._records``, since
        thinking didn't exist for this family yet."""
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
        """
        tools = list(self._toolbox.values())
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
        """Return a shallow copy of registered ToolAgent2 constants."""
        return list(self._constants)

    def register_constant(
        self,
        name: str,
        value: Any,
        description: str | None = None,
        inline_limit: int | None = None,
    ) -> str:
        """
        Register one named runtime constant on this ToolAgent2.

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
            registered on this ToolAgent2.
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
        """Remove all registered constants from this ToolAgent2."""
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
        """Register one invokable on this ToolAgent2.

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
        """Register a batch of invokables on this ToolAgent2.

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
    def _resolve_placeholders(self, obj: Any, *, task: ToolAgentTask) -> Any:
        """
        Resolve all placeholders in an object to their concrete values.

        This method recursively traverses the object structure and replaces placeholder
        references with their concrete runtime values. Placeholders can reference:

        - previously executed current-run steps,
        - previously persisted blackboard entries,
        - registered ToolAgent2 constants.

        Supported Placeholder Formats
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        - ``<<__sN__>>`` – Result from running step N (0-based, plan-local to this invoke)
        - ``<<__cN__>>`` – Result from cache entry N (0-based, from persisted blackboard)
        - ``<<__k.NAME__>>`` – Registered ToolAgent2 constant named NAME

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
        task : ToolAgentTask
            Execution task containing running_blackboard (cache references are
            resolved directly against the agent-level, always-persisted
            ``self._blackboard`` -- see ``update_blackboard``).

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
        terminating return step (e.g. one round of an adaptive-batch
        subclass's generation) can be compiled the same way a full plan
        can:

        - Given: validates it points at a real return-tool slot at the
          final position, raises on mismatch, and isolates it as its own
          final batch.
        - ``None``: every given slot participates in leveling; no isolated
          final batch, no return-shaped validation at all.

        This method does not itself guard against ``planned_slots`` being
        empty — callers must ensure non-emptiness before calling.

        Raises
        ------
        ToolAgentError
            If ``return_idx`` is given but does not point at a real
            return-tool slot at the final position.
        """
        if return_idx is not None and (
            return_idx != len(planned_slots) - 1
            or planned_slots[return_idx].tool != RETURN_TOOL_FULL_NAME
        ):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: return_idx mismatch during batch compilation."
            )

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

        Shared by ``PlanActAgent2``, ``ReActAgent2``, and future ``ToolAgent2``
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
    # Finalization helpers
    # ------------------------------------------------------------------ #
    def update_blackboard(self, task: ToolAgentTask) -> ToolAgentTask:
        """
        Persist all non-empty run slots into the agent's persisted blackboard.

        Called unconditionally by ``_build_record_from_task`` — whether or
        not ``context_enabled`` is set. Always appends onto the *live*
        ``self._blackboard`` as its base, regardless of ``context_enabled``
        — fixing a v1 bug where a task-local snapshot (``cache_blackboard``,
        empty whenever ``context_enabled=False``) was used as the append
        base instead, silently replacing ``self._blackboard`` with just the
        current run's own slots and corrupting ``blackboard_start``/
        ``blackboard_end`` bookkeeping on the next such invoke.
        ``context_enabled`` now controls only whether prior steps are fed to
        the LLM as context (``valid_cache_indices``/``failed_cache_indices``
        in ``_initialize_task``), never whether history is persisted.

        Persistence Policy
        ~~~~~~~~~~~~~~~~~~
        1. **Trim empty/unplanned tail**: Remove trailing empty slots from running blackboard
           (slots with no tool assigned)
        2. **Rewrite placeholders**: All ``<<__sN__>>`` step references in appended slots'
           args are rewritten to ``<<__c{new_global_index}__>>`` cache references.
           Applied to both EXECUTED and FAILED slots.
        3. **Merge into blackboard**: Append all non-empty slots (EXECUTED and FAILED)
           preserving status. FAILED slots keep their ``error`` and no ``result``.
        4. **Trim tail**: Remove trailing empty slots from the final merged blackboard

        Placeholder Rewriting Example
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Initial state:
        - self._blackboard has 5 entries (indices 0-4)
        - running_blackboard has 3 executed entries (indices 0-2)

        Rewriting in appended slots:
        - Step 0's args contain ``<<__s1__>>`` → rewritten to ``<<__c6__>>`` (5 + 1)
        - Step 1's args contain ``<<__s0__>>`` → rewritten to ``<<__c5__>>`` (5 + 0)
        - Step 2's args contain ``<<__s1__>>`` → rewritten to ``<<__c6__>>`` (5 + 1)

        After persistence:
        - self._blackboard now has 8 entries
        - Future invokes can use ``<<__c5__>>``, ``<<__c6__>>``, ``<<__c7__>>``
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
        - ``self._blackboard`` is replaced with merged live-blackboard + appended slots
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
            )
            appended.append(new_slot)

        combined = base_cache + appended

        # 3) Trim empty tail from combined.
        if combined:
            last2 = len(combined) - 1
            while last2 >= 0 and combined[last2].is_empty():
                last2 -= 1
            combined = combined[: last2 + 1]

        self._blackboard = combined
        return task

    def _build_record_from_task(self, task: ToolAgentTask) -> ToolAgentRecord:
        """
        Assemble a completed ToolAgentRecord from a finished ToolAgentTask.

        New ``(task)``-only signature, matching ``Agent2._build_record_from_task``'s
        own simplification (``task.turns`` already holds the same list
        ``_initialize_task`` was given). Combines two bookkeeping steps that
        used to live in separate families: thoughts (new to ``ToolAgent2``,
        mirroring ``Agent2._build_record_from_task``'s own body) and
        blackboard persistence (v1 ``ToolAgent``'s own body, fixed per
        ``update_blackboard`` above).
        """
        prev = task.turns[-1] if task.turns else None

        thoughts_start = len(self._thoughts)
        self._thoughts.extend(task.thoughts)
        thoughts_end = len(self._thoughts)

        blackboard_start = len(self._blackboard)
        task = self.update_blackboard(task)
        blackboard_end = len(self._blackboard)

        return ToolAgentRecord(
            user_prompt=task.user_prompt,
            generated_response=task.generated_response,
            inputs=task.inputs,
            llm_records=tuple(task.llm_records),
            prev=prev,
            thoughts_start=thoughts_start,
            thoughts_end=thoughts_end,
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
        Construct this ToolAgent2's ToolAgentResult envelope directly from a
        completed ToolAgentRecord.

        Extends ``Agent2.build_result_from_record``: re-derives
        ``tool_usage``/``exception_records`` from the record's persisted
        blackboard span (``record.blackboard_start:record.blackboard_end``)
        rather than from task-local bookkeeping — ``update_blackboard``
        preserves each slot's ``status``/``tool`` verbatim and rewrites
        ``step`` to its final global index, so tallying over the persisted
        span gives results identical to tallying over
        ``task.running_blackboard`` directly. ``llm_model_data`` is sourced
        from ``self._llm_engine._get_model_data()`` directly, matching
        ``Agent2.build_result_from_record``'s own robustness fix — don't
        assume ``record.llm_records`` is non-empty or that its last entry is
        the canonical engine identity. ``thoughts_start``/``thoughts_end``
        passed through, new to this family.
        """
        llm_token_usage = tuple(r.llm_result.token_usage for r in record.llm_records)
        llm_model_data = self._llm_engine._get_model_data()

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
            thoughts_start=record.thoughts_start,
            thoughts_end=record.thoughts_end,
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
        When ``context_enabled=False`` the caller passes an empty turns list,
        so both sets are empty — this remains the actual LLM-facing visibility
        gate for ``<<__cN__>>`` references, independent of whether
        ``self._blackboard`` itself is empty.

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

    def render_turn(
        self,
        turn: AgentRecord,
        *,
        include_thoughts: bool = False,
    ) -> list[dict[str, str]]:
        """Render one stored ToolAgentRecord into LLM-facing user/assistant messages.

        Gains ``include_thoughts`` to match ``Agent2.render_turn``'s own
        signature. The base response (and, when ``include_thoughts=True``
        and the record has a nonzero thoughts span, a leading
        ``THOUGHTS:``/``RESPONSE:`` splice) is rendered through
        ``Agent2.render_turn(turn, include_thoughts=include_thoughts)`` and
        treated opaquely from here on — this method just appends the
        blackboard-span content on top of whatever comes back, exactly as
        v1 appended it onto the plain response. If the turn has a non-empty
        blackboard span and all slots executed, this method appends a
        cached-step block (``CACHED STEPS`` section) with each produced step's
        unresolved args and ``run_id``. When some slots are FAILED
        (``fail_fast=False``), the output splits into a ``CACHED STEPS``
        section for executed slots and a ``FAILED STEPS`` section for failed
        slots; failed entries include step index, tool name, and truncated
        error string — no args. Result previews are included only when
        ``peek_at_cache=True`` and are bounded by ``blackboard_preview_limit``.
        """
        if not isinstance(turn, ToolAgentRecord):
            raise ToolAgentError(
                f"render_turn expected ToolAgentRecord, got {type(turn)!r}"
            )

        messages = super().render_turn(turn, include_thoughts=include_thoughts)
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
            assistant_content = (
                f"RESPONSE:\n{assistant_response}\n\n"
                f"CACHED STEPS {list(range(start, end))} PRODUCED:\n\n{dump}"
            )
        else:
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
            assistant_content = "\n\n".join(parts)

        return [
            user_message,
            {"role": "assistant", "content": assistant_content},
        ]

    def _render_system_prompt(
        self,
        task: ToolAgentTask,
        render_context: dict[str, Any],
    ) -> str:
        """Augment ``render_context`` with this agent's tool surface before
        rendering, then delegate to ``Agent2``'s own low-level primitive.

        Fires for every phase a subclass renders through
        ``Agent2._render_system_prompt`` — the "think" phase (via base
        ``Agent2._render_system_message``) and whatever action-phase prompt
        a subclass registers (via that subclass's own
        ``_render_system_message`` override) both get
        ``TOOLS``/``CONSTANTS``/``TOOL_CALLS_LIMIT`` injected through this
        one override point, since neither branches on which template is
        active.
        """
        limit_text = "unlimited" if self._tool_calls_limit is None else str(self._tool_calls_limit)
        augmented = {
            **render_context,
            self.TOOLS_FIELD: self.actions_context(),
            self.CONSTANTS_FIELD: self.constants_context(),
            self.LIMIT_FIELD: limit_text,
        }
        return super()._render_system_prompt(task, augmented)

    # ------------------------------------------------------------------ #
    # String to JSON Objects helper
    # ------------------------------------------------------------------ #
    def _extract_from_json_string(self, raw_text: str) -> Any:
        """
        Extract the largest decodable JSON array/object from a possibly noisy string.

        Thin delegating wrapper — see ``utils.agents.extract_json_object``
        for the full contract (parsing logic lives there, shared with any
        other caller that needs to pull structured output out of free-form
        LLM text).

        Raises
        ------
        TypeError
            If ``raw_text`` is not a string (engine contract violation).
        json.JSONDecodeError
            If ``raw_text`` is empty or contains no decodable JSON array/object.
        """
        return extract_json_object(raw_text, source_label=f"{type(self).__name__}.{self.name}")

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
        Normalize and validate a ToolAgent2 step-field schema set.

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
    ) -> dict[str, Any] | str:
        """
        Validate and normalize one raw LLM-produced ToolAgent2 step mapping.

        The caller provides explicit field bounds:
        - ``allowed_fields`` is the maximum allowed key set.
        - ``required_fields`` is the minimum required key set.

        ``context`` is only for error messages. This method does not infer
        family-specific behavior from context.

        Runtime owns the authoritative step index. Any LLM-provided ``step`` value
        is advisory and is always overwritten with ``expected_step``.

        Returns
        -------
        dict[str, Any]
            Validated and normalized step mapping on success.
        str
            LLM-facing feedback string describing the schema violation. No
            class/name prefix. Returned (not raised) so the caller decides whether
            to retry.
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

        extra = data_keys - allowed
        if extra:
            return f"plan step {expected_step} contains unsupported keys: {sorted(extra)!r}."

        missing = required - data_keys
        if missing:
            return f"plan step {expected_step} is missing required keys: {sorted(missing)!r}."

        normalized = dict(data)

        # Advisory/fallback behavior:
        # - If the LLM omitted "step", fill it.
        # - If the LLM supplied a wrong/non-sequential "step", ignore it.
        # Runtime order is authoritative.
        normalized[STEP_FIELD] = expected_step

        tool = normalized.get(TOOL_FIELD, NO_VAL)
        if TOOL_FIELD in normalized or TOOL_FIELD in required:
            if not isinstance(tool, str) or not tool.strip():
                return f"plan step {expected_step} 'tool' must be a non-empty string."

        args = normalized.get(ARGS_FIELD, NO_VAL)
        if ARGS_FIELD in normalized or ARGS_FIELD in required:
            if not isinstance(args, dict):
                return f"plan step {expected_step} 'args' must be a dict; got {type(args).__name__!r}."

        if AWAIT_FIELD in normalized:
            await_step = normalized[AWAIT_FIELD]
            if type(await_step) is not int or await_step < 0:
                return f"plan step {expected_step} 'await' must be an int >= 0."

            if tool == RETURN_TOOL_FULL_NAME:
                return f"plan step {expected_step} is a return step and must not include 'await_step'."

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
        Re-declared abstract at this level: ``Agent2``'s own concrete
        implementation returns a bare ``AgentTask``, which isn't sufficient
        here — only ``PlanActAgent2``/``ReActAgent2`` know how to build their
        own richer ``ToolAgentTask``-family subclass.

        Same three base parameters as ``Agent2``'s own hook — no
        ``valid_cache_indices``/``failed_cache_indices``; the real
        implementation computes those internally via
        ``_compute_cache_index_sets``.
        """
        ...

    def prepare(self, task: ToolAgentTask) -> ToolAgentTask:
        """Advance ``task``'s preparation phase by one round.

        Concrete, final. No-ops while thinking still gates this family
        (``task.keep_thinking`` is ``True`` and
        ``self._permits_unbounded_thinking`` is ``False``); otherwise
        delegates to abstract ``_prepare_next_batch``, which also owns the
        phase-transition stamp (system prompt name + ``task_messages``
        clear) once thinking allows it to run.
        """
        if task.keep_thinking and not self._permits_unbounded_thinking:
            return task
        return self._prepare_next_batch(task)

    async def async_prepare(self, task: ToolAgentTask) -> ToolAgentTask:
        """Async mirror of ``prepare``, delegating to
        ``_aprepare_next_batch``."""
        if task.keep_thinking and not self._permits_unbounded_thinking:
            return task
        return await self._aprepare_next_batch(task)

    @abstractmethod
    def _prepare_next_batch(self, task: ToolAgentTask) -> ToolAgentTask:
        """
        Prepare exactly one executable batch for the next loop iteration.

        Implementations should:
        - own this subclass's own phase-transition stamp (system prompt
          name + task_messages clear/cadence) once thinking has concluded
          (or immediately, for a subclass that decouples this hook from
          thinking entirely)
        - decide which tool call(s) should execute next
        - validate tool names, step indices, dependencies, and placeholder legality
        - resolve placeholders with `_resolve_placeholders(...)`
        - populate `task.prepared_steps` with the running blackboard indices ready
        for execution
        - return the updated task

        ``prepare()`` handles the thinking gate; ``execute()`` will run the
        prepared batch and handle return-tool completion.
        """
        raise NotImplementedError

    async def _aprepare_next_batch(
        self,
        task: ToolAgentTask,
    ) -> ToolAgentTask:
        """
        Async hook for batch preparation.

        Default offloads the sync hook to a worker thread. Subclasses override
        when the hook contains blocking I/O (e.g. a per-step LLM call).
        """
        return await asyncio.to_thread(self._prepare_next_batch, task)

    def execute(self, task: ToolAgentTask) -> ToolAgentTask:
        """
        Execute all steps in the currently prepared batch concurrently.

        Concrete, final — do not override. No-ops under the same thinking
        gate as ``prepare()``; otherwise executes ``task.prepared_steps``
        using the tool async-invoke path, records results in the running
        blackboard, and handles termination if the return tool is executed.

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
        5. Handle gathered failures, branching on ``fail_fast``:
           - ``fail_fast=True`` (default): identify the first failed step,
             store the error on that slot, mark it failed, and raise
             immediately -- no further slots are stored this round.
           - ``fail_fast=False``: mark **every** failed slot (error +
             ``FAILED`` status), but only raise if the return tool itself
             is among the failures (a failed return always ends the run
             regardless of ``fail_fast``); otherwise fall through to step 6.
        6. Store each successful ``ToolResult`` envelope whole in
           ``slot.result`` and mark those slots executed (runs after step 5
           in both branches -- reached unconditionally under
           ``fail_fast=True`` only when there were no failures at all, and
           always reached under ``fail_fast=False`` for the surviving
           successes).
        7. If return tool executed: set ``task.generated_response`` and ``task.complete = True``
        8. Update ``task.executed_steps``, ``task.tool_calls_used``
        9. Clear ``prepared_steps`` (consumed)

        Parameters
        ----------
        task : ToolAgentTask
            Task with prepared_steps populated. After execution, updated with
            results and completion flags.

        Returns
        -------
        ToolAgentTask
            Updated task with results recorded and completion status set.

        Raises
        ------
        ToolAgentError
            On any validation failure (preconditions, budget, tool not found, etc.)
            or if any tool invocation raises.

        Side Effects
        ~~~~~~~~~~~~
        - ``task.running_blackboard[idx].result`` is set to the whole
          ``ToolResult`` envelope (an ``AtomicResult``) for executed steps —
          preserved for richer tracing. Consumers that need the caller-facing
          value (placeholder resolution, previews, ``generated_response``) read
          ``result.result``; those sites are always reached only after the
          slot is confirmed executed
        - ``task.running_blackboard[idx].error`` is set on failure
        - ``task.running_blackboard[idx].status`` is updated to executed/failed
        - ``task.executed_steps`` is updated with executed indices
        - ``task.tool_calls_used`` incremented by non-return call count
        - ``task.prepared_steps`` is cleared
        - ``task.complete`` and ``task.generated_response`` set if return tool executed
        """
        if task.keep_thinking and not self._permits_unbounded_thinking:
            return task

        indices = list(task.prepared_steps)
        if not indices:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: no prepared steps to execute (prepared_steps is empty)."
            )

        if len(indices) != len(set(indices)):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: prepared_steps contains duplicates: {indices!r}."
            )

        board = task.running_blackboard
        board_len = len(board)

        non_return_planned = 0
        return_indices: list[int] = []

        for idx in indices:
            if idx < 0 or idx >= board_len:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepared step index {idx} out of range "
                    f"(running plan length={board_len})."
                )

            slot = board[idx]

            if slot.step != idx:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: running slot step mismatch at index {idx}: slot.step={slot.step}."
                )

            if slot.is_executed() or idx in task.executed_steps:
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
            if task.tool_calls_used + non_return_planned > self._tool_calls_limit:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool_calls_limit exceeded "
                    f"(limit={self._tool_calls_limit}, used={task.tool_calls_used}, planned={non_return_planned})."
                )

        # Validate tool existence early — same pre-gather guard as async path.
        for idx in indices:
            tool_name = board[idx].tool
            if not self.has_tool(tool_name):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepared step {idx} "
                    f"references unknown tool {tool_name!r}."
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
            task.executed_steps.add(idx)

        task.tool_calls_used += non_return_planned
        task.prepared_steps = []

        if return_indices:
            ret_idx = return_indices[0]
            if board[ret_idx].is_executed():
                task.generated_response = board[ret_idx].result.result
                task.complete = True
            # else: return slot failed; already raised above in fail_fast=False path

        return task

    async def async_execute(self, task: ToolAgentTask) -> ToolAgentTask:
        """
        Async analog of ``execute(...)``.

        Executes all currently prepared steps concurrently using each tool's
        ``async_invoke(...)`` path, stores each ``ToolResult`` envelope whole
        in the running blackboard, and updates return/completion bookkeeping.
        Same thinking gate as ``execute()``. Failure handling mirrors
        ``execute`` exactly: under ``fail_fast=True`` (default), the first
        failed step is marked and raised immediately; under
        ``fail_fast=False``, every failed slot is marked but the round only
        raises if the return tool itself failed -- otherwise execution falls
        through and records the surviving successes.
        """
        if task.keep_thinking and not self._permits_unbounded_thinking:
            return task

        indices = list(task.prepared_steps)
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

        board = task.running_blackboard
        board_len = len(board)

        non_return_planned = 0
        return_indices: list[int] = []

        for idx in indices:
            if idx < 0 or idx >= board_len:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepared step index {idx} out of range "
                    f"(running plan length={board_len})."
                )

            slot = board[idx]

            if slot.step != idx:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: running slot step mismatch at index {idx}: "
                    f"slot.step={slot.step}."
                )

            if slot.is_executed() or idx in task.executed_steps:
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
            if task.tool_calls_used + non_return_planned > self._tool_calls_limit:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool_calls_limit exceeded "
                    f"(limit={self._tool_calls_limit}, used={task.tool_calls_used}, "
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
            task.executed_steps.add(idx)

        task.tool_calls_used += non_return_planned
        task.prepared_steps = []

        if return_indices:
            ret_idx = return_indices[0]
            if board[ret_idx].is_executed():
                task.generated_response = board[ret_idx].result.result
                task.complete = True
            # else: return slot failed; already raised above in fail_fast=False path

        return task

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic snapshot of this ToolAgent2.

        Extends the base Agent2 snapshot (now including thinking-related
        fields automatically) with ToolAgent2-specific toolbox and blackboard
        diagnostics.
        """
        d = super().to_dict()
        d.update({
            "tool_calls_limit": self.tool_calls_limit,
            "fail_fast": self._fail_fast,
            "generation_retries": self._generation_retries,
            "peek_at_cache": self.peek_at_cache,
            "blackboard_preview_limit": self.blackboard_preview_limit,
            "tools": {
                name: tool.to_dict()
                for name, tool in self._toolbox.items()
            },
            "blackboard": self.blackboard_serialized(peek=False),
        })
        return d
