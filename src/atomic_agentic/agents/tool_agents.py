"""
ToolAgents

This module defines the abstract base ToolAgent.

ToolAgent is a template-method runtime that:
- owns a toolbox of Tool instances
- runs an iterative "prepare -> execute" loop against a per-invoke RunState
- supports global blackboard placeholder references: <<__step__N>>
- enforces a resolvable cutoff: placeholders must satisfy N <= resolvable_cutoff
- executes one prepared batch per loop iteration (A1)
- resolves placeholders at prepare time and stores resolved_args in the blackboard slots
- executes concurrently, fail-fast, recording results into the same indexed slots
- terminates when the canonical return tool executes, storing return_value on state
- optionally persists the trimmed (no empty tail) run blackboard back onto the agent blackboard

Subclasses must implement:
- _initialize_run_state(...)
- _prepare_next_batch(...)

The run state is extensible: subclasses may return a dataclass inheriting from ToolAgentRunState.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import logging
import re
import string
import json
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, Optional, Sequence, TypeVar, Iterable
import pprint

from .base import Agent
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
from ..core.Prompts import PLANNER_PROMPT
from ..core.Exceptions import ToolAgentError


logger = logging.getLogger(__name__)

# Canonical step placeholder: <<__step__N>> where N is a 0-based index.
_STEP_TOKEN: re.Pattern[str] = re.compile(r"<<__step__(\d+)>>")


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
    One indexed slot in the run blackboard.

    Lifecycle (sentinel-driven):
      - Empty: tool is NO_VAL
      - Prepared: tool != NO_VAL and resolved_args != NO_VAL and result is NO_VAL
      - Executed: result != NO_VAL

    Note: error is optional for debugging; execution is fail-fast.

    `step`:
      - If left as NO_VAL, callers may compute an index from list position.
    """
    step: int | Any = NO_VAL

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
        """
        Serialization-friendly view.
        - Keeps sentinel NO_VAL values as-is (caller can post-process if desired).
        """
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
    Base run state contract required by ToolAgent.

    Subclasses may extend this dataclass (inherit) to add fields.
    """
    messages: list[dict[str, str]]
    blackboard: list[BlackboardSlot]

    # Inclusive cutoffs:
    # - resolvable_cutoff: max index whose result is available and legal to reference
    # - planned_cutoff: max index that has been planned (prepared) into blackboard
    resolvable_cutoff: int
    planned_cutoff: int

    # Optional but useful bookkeeping:
    tool_calls_used: int = 0  # non-return calls executed in this run
    return_value: Any = NO_VAL  # set once return tool executes


RS = TypeVar("RS", bound=ToolAgentRunState)


@dataclass(frozen=True, slots=True)
class StepCall:
    """
    Optional structured representation of a planned step (tool + args),
    if a subclass prefers to build these before writing into the blackboard.
    """
    batch: int
    tool_name: str
    args: Any


# --------------------------------------------------------------------------- #
# Base ToolAgent
# --------------------------------------------------------------------------- #
class ToolAgent(Agent, ABC, Generic[RS]):
    """
    Abstract base class for tool-using agents.

    The base class owns the invariant runtime loop in `_invoke(messages=...)`:

      initialize state -> repeat:
        prepare one batch (subclass) [must resolve placeholders + update planned_cutoff]
        execute prepared batch (base) [concurrent, fail-fast]
        completion check (base) [return_value set]
      finalize (base) [trim tail + optional persistence]

    Subclasses implement:
      - _initialize_run_state(...) -> RS
      - _prepare_next_batch(state: RS) -> RS
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: str,
        context_enabled: bool = False,
        *,
        tool_calls_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        history_window: Optional[int] = None,
    ) -> None:
        template = self._validate_role_prompt_template(role_prompt)

        super().__init__(
            name=name,
            description=description,
            llm_engine=llm_engine,
            role_prompt=template,
            context_enabled=context_enabled,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            history_window=history_window,
        )

        self._toolbox: dict[str, Tool] = {}
        self._blackboard: list[BlackboardSlot] = []

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
        if not isinstance(value, int) or value < 0:
            raise ToolAgentError("tool_calls_limit must be None or an int >= 0.")
        self._tool_calls_limit = value

    @property
    def blackboard(self) -> list[dict[str, Any]]:
        """
        Read-only serialized view of the persisted blackboard.
        (Keeps backward-friendly dict shape.)
        """
        return [slot.to_dict() for slot in self._blackboard]

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
        component: Any,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        namespace: Optional[str] = None,
        remote_protocol: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        name_collision_mode: str = "raise",  # raise|skip|replace
    ) -> str:
        if name_collision_mode not in ("raise", "skip", "replace"):
            raise ToolRegistrationError("name_collision_mode must be one of: 'raise', 'skip', 'replace'.")

        try:
            tool = toolify(
                component,
                name=name,
                description=description,
                namespace=namespace or self.name,
                remote_protocol=remote_protocol,
                headers=headers,
            )
        except ToolDefinitionError:
            raise
        except Exception as exc:  # pragma: no cover
            raise ToolRegistrationError(f"toolify failed for {component!r}: {exc}") from exc

        key = tool.full_name
        if key in self._toolbox:
            if name_collision_mode == "raise":
                raise ToolRegistrationError(
                    f"{type(self).__name__}.{self.name}: tool already registered: {key}"
                )
            if name_collision_mode == "skip":
                return key
        self._toolbox[key] = tool
        return key

    def batch_register(
        self,
        tools: Sequence[Any] = (),
        mcp_servers: Sequence[tuple[str, Any]] = (),
        a2a_servers: Sequence[tuple[str, Any]] = (),
        *,
        name_collision_mode: str = "raise",
    ) -> list[str]:
        if name_collision_mode not in ("raise", "skip", "replace"):
            raise ToolRegistrationError("name_collision_mode must be one of: 'raise', 'skip', 'replace'.")

        try:
            tool_list = batch_toolify(
                executable_components=list(tools),
                a2a_servers=list(a2a_servers),
                mcp_servers=list(mcp_servers),
                batch_namespace=self.name,
            )
        except ToolDefinitionError:
            raise

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
    def blackboard_dumps(self, *, peek: bool = False) -> str:
        """
        Pretty representation of the persisted blackboard.

        peek=False -> placeholder-view (args)
        peek=True  -> resolved-view (resolved_args)

        Includes explicit `step` numbering to make global indexing unambiguous.
        If a slot's `step` is NO_VAL, the list position is used.
        """
        key = "resolved_args" if peek else "args"
        view: list[dict[str, Any]] = []

        for i, slot in enumerate(self._blackboard):
            d = slot.to_dict()

            step_val = d.get("step", i)
            if step_val is NO_VAL:
                step_val = i

            view.append(
                {
                    "step": step_val,
                    "tool": d.get("tool"),
                    "args": d.get(key),
                }
            )

        try:
            return pprint.pformat(view, indent=2)
        except Exception:  # pragma: no cover
            return str(view)
    # ------------------------------------------------------------------ #
    # Placeholder resolution helpers (prepare-time)
    # ------------------------------------------------------------------ #
    def _step_result_by_index(self, idx: int, *, board: Sequence[BlackboardSlot], resolvable_cutoff: int) -> Any:
        if not isinstance(idx, int):
            raise ToolAgentError(f"Step reference must be an int; got {type(idx).__name__!r}.")
        if idx < 0:
            raise ToolAgentError(f"Step reference must be >= 0; got {idx}.")
        if idx > resolvable_cutoff:
            raise ToolAgentError(
                f"Step reference {idx} exceeds resolvable_cutoff={resolvable_cutoff}."
            )
        if idx >= len(board):
            raise ToolAgentError(
                f"Step reference {idx} out of range (blackboard length={len(board)})."
            )
        slot = board[idx]
        if not slot.is_executed():
            # With the cutoff contract this should be unreachable, but keep it explicit.
            raise ToolAgentError(f"Referenced step {idx} has no executed result.")
        return slot.result

    def _resolve_placeholders(self, obj: Any, *, state: ToolAgentRunState) -> Any:
        """
        Resolve <<__step__N>> placeholders recursively.

        Semantics (as agreed):
        - Full-string "<<__step__N>>" returns the referenced result as-is (preserves type)
        - Inline occurrences inside larger strings are replaced with repr(result)
        """
        board = state.blackboard
        cutoff = state.resolvable_cutoff

        if isinstance(obj, str):
            m = _STEP_TOKEN.fullmatch(obj)
            if m:
                return self._step_result_by_index(int(m.group(1)), board=board, resolvable_cutoff=cutoff)

            def repl(match: re.Match[str]) -> str:
                result = self._step_result_by_index(int(match.group(1)), board=board, resolvable_cutoff=cutoff)
                try:
                    return repr(result)
                except Exception:
                    return str(result)

            return _STEP_TOKEN.sub(repl, obj)

        if isinstance(obj, list):
            return [self._resolve_placeholders(v, state=state) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._resolve_placeholders(v, state=state) for v in obj)
        if isinstance(obj, set):
            return {self._resolve_placeholders(v, state=state) for v in obj}
        if isinstance(obj, dict):
            return {k: self._resolve_placeholders(v, state=state) for k, v in obj.items()}

        # Scalars / unknown objects pass through unchanged.
        return obj

    # ------------------------------------------------------------------ #
    # Execution (base-owned)
    # ------------------------------------------------------------------ #
    def _execute_prepared_batch(self, state: RS) -> RS:
        """
        Execute the currently prepared (planned-but-not-executed) slice:

          start = resolvable_cutoff + 1
          end   = planned_cutoff

        Concurrency: thread pool; fail-fast.
        On success: resolvable_cutoff is advanced to planned_cutoff.
        On tool error: sets slot.error and raises including failing index.
        """
        start = state.resolvable_cutoff + 1
        end = state.planned_cutoff

        if end < start:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: no prepared steps to execute "
                f"(resolvable_cutoff={state.resolvable_cutoff}, planned_cutoff={state.planned_cutoff})."
            )

        if end >= len(state.blackboard):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: planned_cutoff={end} out of range for blackboard length={len(state.blackboard)}."
            )

        steps_to_run = list(range(start, end + 1))

        # Budget enforcement (non-return only).
        if self._tool_calls_limit is not None:
            non_return_planned = 0
            for idx in steps_to_run:
                tool_name = state.blackboard[idx].tool
                if tool_name is NO_VAL:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: slot {idx} is not prepared (tool is NO_VAL)."
                    )
                if tool_name != return_tool.full_name:
                    non_return_planned += 1
            if state.tool_calls_used + non_return_planned > self._tool_calls_limit:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool_calls_limit exceeded "
                    f"(limit={self._tool_calls_limit}, used={state.tool_calls_used}, planned={non_return_planned})."
                )

        def run_one(idx: int) -> tuple[int, Any]:
            slot = state.blackboard[idx]
            if not slot.is_prepared():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: slot {idx} is not prepared for execution."
                )
            tool_name = slot.tool
            tool = self.get_tool(tool_name)

            try:
                result = tool.invoke(slot.resolved_args)
            except ToolInvocationError:
                raise
            except Exception as exc:  # pragma: no cover
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool call failed at index {idx} for {tool_name!r}: {exc}"
                ) from exc
            return idx, result

        if len(steps_to_run) == 1:
            idx = steps_to_run[0]
            try:
                _idx, result = run_one(idx)
            except Exception as exc:
                state.blackboard[idx].error = exc
                raise
            state.blackboard[idx].result = result
        else:
            executor = ThreadPoolExecutor()
            try:
                futures = {executor.submit(run_one, idx): idx for idx in steps_to_run}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        _idx, result = fut.result()
                    except Exception as exc:
                        state.blackboard[idx].error = exc
                        # Fail-fast: raise immediately. Outstanding futures are canceled on shutdown.
                        raise
                    state.blackboard[idx].result = result
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        # Post-execution bookkeeping (fail-fast means we only reach here if all succeeded)
        state.tool_calls_used += sum(
            1 for sc in steps_to_run if state.blackboard[sc].tool != return_tool.full_name)
        for idx in steps_to_run[::-1]:
            tool_name = state.blackboard[idx].tool
            if tool_name == return_tool.full_name:
                # Return tracking: only one return allowed per run.
                state.return_value = state.blackboard[idx].result
                break
        state.resolvable_cutoff = state.planned_cutoff
        return state

    # ------------------------------------------------------------------ #
    # Finalization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _trim_empty_tail(board: list[BlackboardSlot]) -> list[BlackboardSlot]:
        """
        Drop trailing empty/unplanned slots to avoid memory bloat.
        """
        if not board:
            return []
        last = len(board) - 1
        while last >= 0 and board[last].is_empty():
            last -= 1
        return board[: last + 1]

    # ------------------------------------------------------------------ #
    # Template Method (FINAL)
    # ------------------------------------------------------------------ #
    def _invoke(self, *, messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], Any]:
        """
        FINAL template method (do not override in subclasses).

        Requires subclasses to implement:
          - _initialize_run_state(...)
          - _prepare_next_batch(...)
        """
        if not messages:
            raise ToolAgentError("ToolAgent._invoke requires a non-empty messages list.")

        # Snapshot original user message for history preservation.
        original_user_msg = dict(messages[-1])

        state = self._initialize_run_state(messages=messages)

        # Base invariants: state must satisfy required fields.
        if not isinstance(state, ToolAgentRunState):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: _initialize_run_state must return a ToolAgentRunState (or subclass)."
            )

        while True:
            # A1 invariant: before preparing a new batch, there must be no pending planned steps.
            if state.planned_cutoff != state.resolvable_cutoff:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: violation: planned_cutoff={state.planned_cutoff} "
                    f"!= resolvable_cutoff={state.resolvable_cutoff}. Prepare must be followed by execute before next prepare."
                )

            state = self._prepare_next_batch(state)

            if state.planned_cutoff == state.resolvable_cutoff:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepare produced an empty batch (planned_cutoff did not advance)."
                )

            state = self._execute_prepared_batch(state)

            # Completion check: return_value set iff return executed.
            if state.return_value is not NO_VAL:
                # Persist only meaningful blackboard entries if context is enabled.
                if self.context_enabled:
                    trimmed = self._trim_empty_tail(state.blackboard)
                    self._blackboard = trimmed
                try:
                    assistant = repr(state.return_value)
                except Exception:  # pragma: no cover
                    assistant = str(state.return_value)
                newest_history = [original_user_msg, {"role": "assistant", "content": assistant}]

                return newest_history, state.return_value

    # ------------------------------------------------------------------ #
    # Subclass Hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _initialize_run_state(self, *, messages: list[dict[str, str]]) -> RS:
        """
        Must:
          - snapshot messages and any persisted blackboard prefix (if context enabled)
          - append (tool_calls_limit + 1) empty slots, or a fixed cap if unlimited
          - set resolvable_cutoff = prefix_len - 1
          - set planned_cutoff = resolvable_cutoff
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_next_batch(self, state: RS) -> RS:
        """
        Must:
          - plan exactly one batch per loop iteration
          - validate placeholders against state.resolvable_cutoff (raise on violation)
          - fill slots [planned_cutoff+1 .. new_planned_cutoff] contiguously:
              slot.tool, slot.args, slot.resolved_args
            where slot.resolved_args is produced by resolving placeholders now
            (use self._resolve_placeholders(obj, state=state))
          - advance state.planned_cutoff
          - may plan a return tool call; base will detect it after execution
        """
        raise NotImplementedError


@dataclass(slots=True)
class PlanActRunState(ToolAgentRunState):
    """
    PlanAct extends the base run state with:
      - plan_batches: the entire plan, grouped into executable concurrent batches
      - batch_index: which batch to load next during prepare

    All other loop mechanics (cutoffs, execution, fail-fast, return_value) are handled by ToolAgent.
    """
    plan_batches: list[list[StepCall]] = field(default_factory=list)
    batch_index: int = 0


BLACKBOARD_INJECTION_TEMPLATE = """\
# BLACKBOARD CONTEXT (READ-ONLY)
The following blackboard shows the arguments and tool calls made for the
completed steps 0 to {RESOLVABLE_CUTOFF}.

Note that the NEW plan you are making is starting at step {NEW_START}.
Refer to the below steps to reuse any results needed for your current
or future tasks.

{BLACKBOARD_CONTEXT}
"""

class PlanActAgent(ToolAgent[PlanActRunState]):
    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        *,
        context_enabled: bool = False,
        tool_calls_limit: int | None = None,
        pre_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        post_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        history_window: int | None = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            llm_engine=llm_engine,
            role_prompt=PLANNER_PROMPT,
            context_enabled=context_enabled,
            tool_calls_limit=tool_calls_limit,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            history_window=history_window,
        )

    # ------------------------------------------------------------------ #
    # Hook: initialize (one-shot planning)
    # ------------------------------------------------------------------ #
    def _initialize_run_state(self, *, messages: list[dict[str, str]]) -> PlanActRunState:
        if not messages:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: messages must be non-empty.")

        prefix: list[BlackboardSlot] = list(self._blackboard) if self.context_enabled else []
        prefix_len = len(prefix)

        # NEW: ensure persisted prefix slots carry explicit global indices.
        # This is safe even though prefix is a shallow copy (it updates the persisted slots too).
        for i, slot in enumerate(prefix):
            slot.step = i

        working_messages = messages
        if self.context_enabled and len(self._blackboard) > 0:
            working_messages = self.inject_blackboard(working_messages)

        raw = self._llm_engine.invoke(working_messages)

        plan = self._parse_plan_json(raw)
        stepcalls = self._to_stepcalls(plan)

        self._validate_return_is_final(stepcalls)
        self._enforce_budget(stepcalls)

        # Validate intra-batch dependency constraints (prefix refs allowed).
        self._validate_intra_batch_deps(stepcalls, prefix_len=prefix_len)

        plan_batches = self._group_by_batch(stepcalls)

        resolvable_cutoff = prefix_len - 1
        planned_cutoff = resolvable_cutoff

        reserve = (len(stepcalls) if self._tool_calls_limit is None else self._tool_calls_limit + 1)

        # NEW: allocate reserve slots with explicit global step indices.
        blackboard = prefix + [BlackboardSlot(step=prefix_len + i) for i in range(reserve)]

        return PlanActRunState(
            messages=working_messages,
            blackboard=blackboard,
            resolvable_cutoff=resolvable_cutoff,
            planned_cutoff=planned_cutoff,
            tool_calls_used=0,
            return_value=NO_VAL,
            plan_batches=plan_batches,
            batch_index=0,
        )


    # ------------------------------------------------------------------ #
    # Hook: prepare (load next batch)
    # ------------------------------------------------------------------ #
    def _prepare_next_batch(self, state: PlanActRunState) -> PlanActRunState:
        if state.batch_index >= len(state.plan_batches):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: no remaining batches "
                f"(batch_index={state.batch_index}, total={len(state.plan_batches)})."
            )

        batch = state.plan_batches[state.batch_index]
        if not batch:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: empty batch at index {state.batch_index}.")

        start = state.planned_cutoff + 1
        end = start + len(batch) - 1
        if end >= len(state.blackboard):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: insufficient blackboard capacity for batch {state.batch_index} "
                f"(need end={end}, len={len(state.blackboard)})."
            )

        for offset, step in enumerate(batch):
            idx = start + offset
            slot = state.blackboard[idx]

            # NEW: defensive step index assignment (should already be set by _initialize_run_state()).
            if slot.step is NO_VAL:
                slot.step = idx

            slot.tool = step.tool_name
            slot.args = step.args
            slot.resolved_args = self._resolve_placeholders(step.args, state=state)

        state.planned_cutoff = end
        state.batch_index += 1
        return state

    # ------------------------------------------------------------------ #
    # Planning helpers
    # ------------------------------------------------------------------ #
    def inject_blackboard(
        self,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:

        out = [dict(m) for m in messages]
        prefix_len = len(self._blackboard)
        blackboard_context = self.blackboard_dumps(peek=False)
        hint = BLACKBOARD_INJECTION_TEMPLATE.format(RESOLVABLE_CUTOFF=prefix_len -1,
                                                    NEW_START=prefix_len,
                                                    BLACKBOARD_CONTEXT=blackboard_context)
        msg, out = out[-1], out[:-1]
        msg["content"] += "\n\n" + hint
        out += [msg]
        return out

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        s = (text or "").strip()
        if s.startswith("```"):
            s = s.split("\n", 1)[1] if "\n" in s else ""
            if s.rstrip().endswith("```"):
                s = s.rstrip()[:-3]
        return s.strip()

    def _parse_plan_json(self, raw: Any) -> list[dict[str, Any]]:
        text = raw if isinstance(raw, str) else str(raw)
        cleaned = self._strip_json_fences(text)
        if not cleaned:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: planner returned empty output.")

        try:
            parsed = json.loads(cleaned)
        except Exception as exc:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: planner output is not valid JSON: {exc}") from exc

        if not isinstance(parsed, list) or not parsed:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: planner output must be a non-empty JSON array.")

        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: plan element {i} must be an object.")
            if set(item.keys()) != {"tool", "args", "batch"}:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan element {i} must have exactly keys "
                    f"{{'tool','args','batch'}} (got {sorted(item.keys())})."
                )
        return parsed  # type: ignore[return-value]

    def _to_stepcalls(self, plan: list[dict[str, Any]]) -> list[StepCall]:
        out: list[StepCall] = []
        for i, step in enumerate(plan):
            tool_name = step.get("tool")
            args = step.get("args")
            batch = step.get("batch")

            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: step {i} has invalid 'tool'.")
            if not isinstance(args, dict):
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: step {i} 'args' must be an object/dict.")
            if not isinstance(batch, int) or batch < 0:
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: step {i} 'batch' must be int >= 0.")
            if not self.has_tool(tool_name):
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool {tool_name!r} in step {i}.")

            out.append(StepCall(batch=batch, tool_name=tool_name, args=dict(args)))
        return out

    def _validate_return_is_final(self, stepcalls: list[StepCall]) -> None:
        ret_name = return_tool.full_name
        ret_positions = [i for i, sc in enumerate(stepcalls) if sc.tool_name == ret_name]
        if len(ret_positions) != 1:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan must contain exactly one return step; found {len(ret_positions)}."
            )
        if ret_positions[0] != len(stepcalls) - 1:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: return step must be the final step "
                f"(found at index {ret_positions[0]} of {len(stepcalls) - 1})."
            )

    def _enforce_budget(self, stepcalls: list[StepCall]) -> None:
        if self._tool_calls_limit is None:
            return
        non_return = sum(1 for sc in stepcalls if sc.tool_name != return_tool.full_name)
        if non_return > self._tool_calls_limit:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan exceeds tool_calls_limit "
                f"(limit={self._tool_calls_limit}, non_return_steps={non_return})."
            )

    @staticmethod
    def _group_by_batch(stepcalls: list[StepCall]) -> list[list[StepCall]]:
        buckets: dict[int, list[StepCall]] = {}
        for sc in stepcalls:
            buckets.setdefault(sc.batch, []).append(sc)
        return [buckets[b] for b in sorted(buckets)]

    def _validate_intra_batch_deps(self, stepcalls: list[StepCall], *, prefix_len: int) -> None:
        """
        Enforce: if a placeholder references a NEW step in this plan, the referenced step must be
        in a strictly smaller batch. Prefix references are always allowed.
        """
        n = len(stepcalls)
        # New steps occupy global indices [prefix_len .. prefix_len+n-1] in plan order.
        batch_by_global: dict[int, int] = {prefix_len + i: stepcalls[i].batch for i in range(n)}

        for i, sc in enumerate(stepcalls):
            cur_global = prefix_len + i
            cur_batch = sc.batch
            for ref in self._extract_step_refs_json(sc.args):
                if ref < 0:
                    raise ToolAgentError(f"{type(self).__name__}.{self.name}: negative step reference {ref}.")
                if ref < prefix_len:
                    continue  # prefix always allowed
                if ref >= prefix_len + n:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: step {cur_global} references out-of-range step {ref} "
                        f"(prefix_len={prefix_len}, plan_steps={n})."
                    )
                dep_batch = batch_by_global[ref]
                if dep_batch >= cur_batch:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: invalid dependency by batch: "
                        f"step {cur_global} (batch={cur_batch}) references step {ref} (batch={dep_batch})."
                    )

    @staticmethod
    def _extract_step_refs_json(obj: Any) -> set[int]:
        refs: set[int] = set()

        def walk(v: Any) -> None:
            if isinstance(v, dict):
                for vv in v.values():
                    walk(vv)
                return
            if isinstance(v, list):
                for vv in v:
                    walk(vv)
                return
            if isinstance(v, str):
                for m in _STEP_TOKEN.finditer(v):
                    refs.add(int(m.group(1)))
                return

        walk(obj)
        return refs

# ───────────────────────────────────────────────────────────────────────────────
# Iterative Plan 'ReActAgent' class
# ───────────────────────────────────────────────────────────────────────────────
class ReActAgent(ToolAgent):
    """
    Dynamic, step-by-step tool-using agent (ReAct-style).

    Message choreography (per-loop):
      1) call LLM on messages_snapshot -> step_text
      2) parse a SINGLE step object -> (tool_name, raw_args)
      3) append that step as an assistant message (canonical JSON)
      4) execute tool, record completed blackboard entry, update running_result
      5) append a user message containing:
           - NEW STEPS THIS RUN (only new_steps)
           - OBSERVATION (running_result preview)
           - request for the next single-step dict
    """
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        tool_calls_limit: Optional[int] = 25,
        preview_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        history_window: Optional[int] = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            llm_engine=llm_engine,
            role_prompt=ORCHESTRATOR_PROMPT,
            context_enabled=context_enabled,
            tool_calls_limit=tool_calls_limit,
            pre_invoke=pre_invoke,
            post_invoke=post_invoke,
            history_window=history_window,
        )

        self._preview_limit: Optional[int] = None
        self.preview_limit = preview_limit

    # ------------------------------------------------------------------ #
    # Tool-Agent Properties
    # ------------------------------------------------------------------ #    
    @property
    def tool_calls_limit(self) -> Optional[int]:
        """Max allowed tool calls per invoke() run. None means unlimited."""
        return self._tool_calls_limit

    @tool_calls_limit.setter
    def tool_calls_limit(self, value: Optional[int]) -> None:
        if value is None or not isinstance(value, int) or value <= 0:
            raise ToolAgentError("ReActAgent requires a tool_calls_limit >= 0.")
        self._tool_calls_limit = value

    # ------------------------------------------------------------------ #
    # ReAct-Agent Properties
    # ------------------------------------------------------------------ #
    @property
    def preview_limit(self) -> Optional[int]:
        """
        Max characters allowed in the observation preview of `running_result`.

        - None: no truncation (entire stringified result is shown)
        - positive int: truncate to at most that many characters
        """
        return self._preview_limit

    @preview_limit.setter
    def preview_limit(self, value: Optional[int]) -> None:
        if value is None:
            self._preview_limit = None
            return
        if not isinstance(value, int) or value <= 0:
            raise ToolAgentError("preview_limit must be None or a positive int.")
        self._preview_limit = value

    # ------------------------------------------------------------------ #
    # ReAct-Agent Helpers
    # ------------------------------------------------------------------ #
    def _tool_calls_made_snapshot(self) -> int:
        with self._tool_calls_lock:
            return int(self._tool_calls_made)

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^\s*```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _safe_stringify(value: Any) -> str:
        try:
            return repr(value)
        except Exception:
            return str(value)

    def _truncate_preview(self, text: str) -> str:
        limit = self._preview_limit
        if limit is None:
            return text
        if len(text) <= limit:
            return text
        return text[:limit] + "… [truncated]"

    def _parse_single_step(self, text: str) -> tuple[str, Dict[str, Any]]:
        """
        Parse exactly one step dict from the LLM output.

        Expected schema:
            {"tool": "<Tool.full_name>", "args": {...}}

        Robustness:
        - Strips markdown fences.
        - If extra text exists, extracts the LAST JSON object matching the schema.
        """
        cleaned = self._strip_markdown_fences(text).strip()
        logger.debug("%s.%s: LLM raw step output (preview): %r", type(self).__name__, self.name, cleaned[:1500])

        if not cleaned:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: LLM returned empty step output.")

        if cleaned.lstrip().startswith("["):
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: expected a single JSON object, got an array.")

        def validate(obj: Any) -> tuple[str, Dict[str, Any]] | None:
            if not isinstance(obj, dict):
                return None
            if set(obj.keys()) != {"tool", "args"}:
                return None
            tool_name = obj.get("tool")
            args = obj.get("args")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: 'tool' must be a non-empty string.")
            if not isinstance(args, dict):
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: 'args' must be a JSON object/dict.")
            return tool_name, args

        # Fast path: pure JSON object
        if cleaned.lstrip().startswith("{"):
            try:
                obj = json.loads(cleaned)
                out = validate(obj)
                if out is not None:
                    return out
            except Exception:
                pass

        # Tolerant path: scan for step-like JSON objects and pick the last one.
        decoder = json.JSONDecoder()
        candidates: list[tuple[int, str, Dict[str, Any]]] = []

        for m in re.finditer(r"{", cleaned):
            start = m.start()
            try:
                parsed, _end = decoder.raw_decode(cleaned, start)
            except Exception:
                continue
            out = validate(parsed)
            if out is None:
                continue
            tool_name, args = out
            candidates.append((start, tool_name, dict(args)))

        if not candidates:
            preview = cleaned if len(cleaned) <= 800 else cleaned[:800] + "…"
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: could not find a valid step object in LLM output. "
                f"Output preview: {preview!r}"
            )

        _, tool_name, args = max(candidates, key=lambda t: t[0])
        if len(candidates) > 1:
            logger.warning(
                "%s.%s: multiple step-like JSON objects found; using the last one.",
                type(self).__name__,
                self.name,
            )
        return tool_name, args

    def _run(
        self,
        *,
        messages: List[Dict[str, str]],
    ) -> tuple[List[BlackboardEntry], Any]:
        persisted: List[BlackboardEntry] = self.blackboard
        base_len = len(persisted)

        new_steps: List[BlackboardEntry] = []
        running_result: Any = "No intermediate results produced yet"
        last_tool_called: str = ""

        messages_snapshot: List[Dict[str, str]] = list(messages)

        while True:
            # If return already executed, we're done.
            if last_tool_called == return_tool.full_name:
                return new_steps, running_result

            # Auto-return if the (non-return) budget is exhausted.
            limit = self.tool_calls_limit
            made = self._tool_calls_made_snapshot()
            if limit is not None and made >= limit:
                last_non_return_local_idx: Optional[int] = None
                for i in range(len(new_steps) - 1, -1, -1):
                    if new_steps[i].get("tool") != return_tool.full_name and bool(new_steps[i].get("completed", False)):
                        last_non_return_local_idx = i
                        break

                if last_non_return_local_idx is None:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: tool usage limit reached ({made}/{limit}) "
                        "but no completed non-return step exists to return."
                    )

                last_global_idx = base_len + last_non_return_local_idx
                raw_args = {"val": f"<<__step__{last_global_idx}>>"}

                board = list(persisted) + list(new_steps)
                result = self._call_tool(return_tool.full_name, raw_args, board=board)

                resolved_any = self._resolve_step_refs(dict(raw_args), board=board)
                if not isinstance(resolved_any, dict):
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: resolved args must be a dict; got {type(resolved_any).__name__!r}."
                    )

                new_steps.append(
                    {
                        "tool": return_tool.full_name,
                        "args": dict(raw_args),
                        "resolved_args": dict(resolved_any),
                        "completed": True,
                        "result": result,
                    }
                )
                return new_steps, result

            # 1) Ask LLM for next step (based on current snapshot).
            step_text = self._llm_engine.invoke(messages_snapshot)
            tool_name, raw_args_any = self._parse_single_step(step_text)

            # 2) Save the step as an assistant message (canonical JSON).
            canonical_step = json.dumps({"tool": tool_name, "args": raw_args_any}, indent=2)
            messages_snapshot.append({"role": "assistant", "content": canonical_step})

            # 3) Execute and record.
            
            # 3.a) Validate tool exists.
            if not self.has_tool(tool_name):
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool {tool_name!r}.")

            raw_args: Dict[str, Any] = dict(raw_args_any)
            board_for_step = list(persisted) + list(new_steps)

            # 3.b) Call tool via ToolAgent gateway.
            result = self._call_tool(tool_name, raw_args, board=board_for_step)

            # 3.c) Record completed step with resolved args.
            resolved_any = self._resolve_step_refs(dict(raw_args), board=board_for_step)
            if not isinstance(resolved_any, dict):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: resolved args must be a dict; got {type(resolved_any).__name__!r}."
                )

            new_steps.append(
                {
                    "tool": tool_name,
                    "raw_args": dict(raw_args),
                    "resolved_args": dict(resolved_any),
                    "completed": True,
                    "result": result,
                }
            )

            running_result = result
            last_tool_called = tool_name

            # If return tool executed, stop immediately.
            if last_tool_called == return_tool.full_name:
                return new_steps, running_result

            # 4) Append observation + request as the next user message.
            global_idx = base_len + len(new_steps) - 1
            preview_text = self._truncate_preview(self._safe_stringify(running_result))
            new_user_msg = (
                f"STEP {global_idx} EXECUTED TOOL {tool_name!r}.\n"
                f"OBSERVED RESULT: {preview_text}\n\n"
                "Now emit EXACTLY ONE JSON object for the next step and no other object."
            )
            messages_snapshot.append({"role": "user", "content": new_user_msg})
