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
@dataclass(frozen=True, slots=True)
class StepCall:
    """
    Canonical representation of a single tool step.

    Maps 1:1 to a global blackboard slot:
      - index: authoritative blackboard index for storage and placeholder references
      - tool_name: must match a registered Tool.full_name in the agent toolbox
      - args: literal-ish structure that may contain placeholders <<__step__N>>
      - await_index: optional dependency metadata (base ToolAgent does not interpret it)
    """
    index: int
    tool_name: str
    args: Any
    await_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialization-friendly view matching planner output keys."""
        return {
            "index": self.index,
            "tool": self.tool_name,
            "args": self.args,
            "await": self.await_index,
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "StepCall":
        if not isinstance(data, Mapping):
            raise ToolAgentError(
                f"StepCall.from_dict requires a mapping; got {type(data).__name__!r}."
            )

        if "index" not in data:
            raise ToolAgentError("StepCall.from_dict missing required key: 'index'.")
        if "tool" not in data:
            raise ToolAgentError("StepCall.from_dict missing required key: 'tool'.")
        if "args" not in data:
            raise ToolAgentError("StepCall.from_dict missing required key: 'args'.")

        index = data["index"]
        tool = data["tool"]
        args = data["args"]
        await_val = data.get("await", None)

        if not isinstance(index, int) or index < 0:
            raise ToolAgentError("StepCall.index must be an int >= 0.")
        if not isinstance(tool, str) or not tool.strip():
            raise ToolAgentError("StepCall.tool must be a non-empty str.")
        if await_val is not None and (not isinstance(await_val, int) or await_val < 0):
            raise ToolAgentError("StepCall.await must be None or an int >= 0.")

        return StepCall(index=index, tool_name=tool, args=args, await_index=await_val)

@dataclass(slots=True)
class BlackboardSlot:
    """
    One indexed slot in the run blackboard.

    Lifecycle (sentinel-driven):
      - Empty: tool is NO_VAL
      - Prepared: tool != NO_VAL and resolved_args != NO_VAL and result is NO_VAL
      - Executed: result != NO_VAL

    `step` is always the global blackboard index for this slot (0-based).
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
    Base run state contract required by ToolAgent.

    Plan-agnostic: the base ToolAgent does not store or interpret any "plan".

    Placeholder resolvability (<<__step__N>>):

      A placeholder index N is resolvable iff:
        - N < prefix_blackboard_len   (injected / persisted prefix context), OR
        - N in executed_indices       (a non-prefix step that has executed)
    """
    messages: list[dict[str, str]]
    blackboard: list[BlackboardSlot]

    # Prefix length for injected/persisted context.
    prefix_blackboard_len: int

    # Indices (typically >= prefix_blackboard_len) whose results are available.
    executed_indices: set[int] = field(default_factory=set)

    # Exact indices to execute next; must be set by _prepare_next_batch.
    prepared_indices: list[int] = field(default_factory=list)

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
            return pprint.pformat(view, indent=2, width=160)
        except Exception:  # pragma: no cover
            return str(view)

    # ------------------------------------------------------------------ #
    # Placeholder resolution helpers (prepare-time)
    # ------------------------------------------------------------------ #
    def _resolve_placeholders(self, obj: Any, *, state: ToolAgentRunState) -> Any:
        """
        Resolve <<__step__N>> placeholders recursively.

        Readiness rule (as agreed):
        - A placeholder resolves iff state.blackboard[N].result is not NO_VAL.
        - If not executed (result is NO_VAL), raise.

        Semantics (as agreed):
        - Full-string "<<__step__N>>" returns the referenced result as-is (preserves type)
        - Inline occurrences inside larger strings are replaced with repr(result)
        """
        board = state.blackboard

        # ----------------------------
        # 1) Collect all dependencies.
        # ----------------------------
        needed: set[int] = set()

        def collect(x: Any) -> None:
            if isinstance(x, str):
                for m in _STEP_TOKEN.finditer(x):
                    needed.add(int(m.group(1)))
                return
            if isinstance(x, dict):
                for k, v in x.items():
                    collect(k)
                    collect(v)
                return
            if isinstance(x, (list, tuple, set)):
                for v in x:
                    collect(v)
                return
            # scalars / unknown objects: no dependencies

        collect(obj)

        # ----------------------------
        # 2) Validate readiness.
        # ----------------------------
        board_len = len(board)
        for idx in sorted(needed):
            if idx < 0 or idx >= board_len:
                raise ToolAgentError(
                    f"Step reference {idx} out of range (blackboard length={board_len})."
                )
            if not board[idx].is_executed():
                raise ToolAgentError(f"Referenced step {idx} is not executed (result is NO_VAL).")

        # ----------------------------
        # 3) Resolve recursively.
        # ----------------------------
        def resolve(x: Any) -> Any:
            if isinstance(x, str):
                full = _STEP_TOKEN.fullmatch(x)
                if full:
                    idx = int(full.group(1))
                    # readiness already validated
                    return board[idx].result

                def repl(m: re.Match[str]) -> str:
                    idx = int(m.group(1))
                    val = board[idx].result
                    try:
                        return repr(val)
                    except Exception:
                        return str(val)

                return _STEP_TOKEN.sub(repl, x)

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

    def extract_dependencies(self, obj: Any) -> set[int]:
        """
        Return the set of placeholder indices (N) referenced by <<__step__N>> anywhere in obj.

        This does NOT validate readiness; it is purely structural and intended for dependency analysis.
        """
        deps: set[int] = set()

        def walk(x: Any) -> None:
            if isinstance(x, str):
                for m in _STEP_TOKEN.finditer(x):
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
            # scalars / unknown objects: no dependencies

        walk(obj)
        return deps

    # ------------------------------------------------------------------ #
    # Execution (base-owned)
    # ------------------------------------------------------------------ #
    def _execute_prepared_batch(self, state: RS) -> RS:
        """
        Execute the currently prepared batch described by state.prepared_indices.

        Concurrency: thread pool; fail-fast.
        On success:
        - records results into the same blackboard slots
        - adds indices to executed_indices
        - increments tool_calls_used (non-return only)
        - clears prepared_indices
        - sets is_done/return_value if the return tool executed
        On failure:
        - records the exception into slot.error for the failing index
        - raises immediately (fail-fast)
        """
        indices = list(state.prepared_indices)
        if not indices:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: no prepared steps to execute (prepared_indices is empty)."
            )

        # Validate uniqueness early (double-execution would be catastrophic).
        if len(indices) != len(set(indices)):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: prepared_indices contains duplicates: {indices!r}."
            )

        board = state.blackboard
        board_len = len(board)

        # Pre-validate slots + compute planned non-return count and return count.
        non_return_planned = 0
        return_indices: list[int] = []

        for idx in indices:
            if not isinstance(idx, int):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepared index must be int; got {type(idx).__name__!r}."
                )
            if idx < 0 or idx >= board_len:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepared index {idx} out of range for blackboard length={board_len}."
                )

            slot = board[idx]

            # Optional consistency check: slot.step should match its index.
            if isinstance(slot.step, int) and slot.step != idx:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: blackboard slot step mismatch at index {idx}: slot.step={slot.step}."
                )

            if slot.is_executed() or idx in state.executed_indices:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepared index {idx} is already executed."
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

        def run_one(idx: int) -> tuple[int, Any]:
            slot = board[idx]
            tool_name = slot.tool  # already validated above
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

        # Execute concurrently (fail-fast).
        if len(indices) == 1:
            idx = indices[0]
            try:
                _idx, result = run_one(idx)
            except Exception as exc:
                board[idx].error = exc
                raise
            board[idx].result = result
        else:
            executor = ThreadPoolExecutor()
            try:
                futures = {executor.submit(run_one, idx): idx for idx in indices}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        _idx, result = fut.result()
                    except Exception as exc:
                        board[idx].error = exc
                        raise
                    board[idx].result = result
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        # Post-execution bookkeeping (only reached if all succeeded).
        for idx in indices:
            state.executed_indices.add(idx)

        state.tool_calls_used += non_return_planned
        state.prepared_indices.clear()

        if return_indices:
            ret_idx = return_indices[0]
            state.return_value = board[ret_idx].result
            state.is_done = True

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

        original_user_msg = dict(messages[-1])

        state = self._initialize_run_state(messages=messages)

        if not isinstance(state, ToolAgentRunState):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: _initialize_run_state must return a ToolAgentRunState (or subclass)."
            )

        while not state.is_done:
            # Invariant: prepare must not be called with a pending prepared batch.
            if state.prepared_indices:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: violation: prepared_indices is non-empty before prepare. "
                    f"Execute must follow prepare before preparing again."
                )

            state = self._prepare_next_batch(state)

            # Inline empty-batch check (per design): raise here, not inside helpers.
            if not state.prepared_indices:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: prepare produced an empty batch (prepared_indices is empty)."
                )

            state = self._execute_prepared_batch(state)

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


BLACKBOARD_INJECTION_TEMPLATE = """\
# BLACKBOARD CONTEXT (READ-ONLY)
The following blackboard shows the arguments and tool calls made for the
completed steps 0 to {RESOLVABLE_CUTOFF}.

Note that the NEW plan you are making is starting at step {NEW_START}.
Refer to the below steps to reuse any results needed for your current
or future tasks.

{BLACKBOARD_CONTEXT}
"""

@dataclass(slots=True)
class PlanActRunState(ToolAgentRunState):
    """
    PlanAct run state.

    Extends ToolAgentRunState with:
      - plan_batches: compiled batches of StepCall objects (concurrent executable groups)
      - batch_index: next batch to prepare
    """
    plan_batches: list[list[StepCall]] = field(default_factory=list)
    batch_index: int = 0

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
    # Subclass Hook: Initialize Run State + Helpers
    # ------------------------------------------------------------------ #
    def _initialize_run_state(self, *, messages: list[dict[str, str]]) -> PlanActRunState:
        """
        One-shot planning + compilation into executable batches.

        Pipeline:
          0) create minimal state inputs (messages copy, prefix blackboard)
          1) inject blackboard context into messages (if enabled)
          2) call LLM to get raw plan
          3) parse raw -> list[dict]
          4) convert dicts -> canonical StepCall list (global indices)
          5) validate deps + await constraints
          6) compile flat StepCalls -> list[list[StepCall]] batches
          7) init batch tracker and allocate run blackboard slots
        """
        if not messages:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: messages must be non-empty.")

        # 0) Minimal state inputs: copy messages + prefix blackboard.
        working_messages = [dict(m) for m in messages]

        prefix: list[BlackboardSlot] = list(self._blackboard) if self.context_enabled else []
        prefix_len = len(prefix)

        # Normalize prefix slot indices (global 0..prefix_len-1).
        for i, slot in enumerate(prefix):
            slot.step = i

        # 1) Inject blackboard context into the messages (if enabled).
        if self.context_enabled and prefix_len > 0:
            # Helper exists (or will be updated) to augment the final user message.
            working_messages = self.inject_blackboard(working_messages)

        # 2) Generate raw plan from LLM.
        raw_plan = self._llm_engine.invoke(working_messages)

        # 3) Parse raw plan -> list[dict].
        plan_dicts = self._parse_plan_json(raw_plan)

        # 4) Convert dicts -> canonical StepCalls (global indices).
        #
        # Surface-level helper (NOT implemented here):
        #   - assigns StepCall.index = prefix_len + i
        #   - pulls tool/args/await into tool_name/args/await_index
        #   - does tool existence checks (or defers to validation)
        step_calls = self._plan_dicts_to_stepcalls(plan_dicts, prefix_len=prefix_len)

        # Enforce return tool policy as part of plan structure (surface-level helper).
        # Surface-level helper (NOT implemented here):
        #   - at most one return
        #   - if present must be last
        #   - if missing, append return(None) as a final step
        step_calls = self._ensure_and_validate_return(step_calls, prefix_len=prefix_len)

        # 5) Validate argument deps + await deps (surface-level helper).
        # Surface-level helper (NOT implemented here):
        #   - uses extract_dependencies(step.args) to find placeholder deps
        #   - enforces:
        #       * no forward/self refs (dep < step.index)
        #       * await rules you specified (null await => deps prefix-only, await covers max non-prefix dep, etc.)
        self._validate_stepcalls(step_calls, prefix_len=prefix_len)

        # Optional budget enforcement at plan-time (surface-level helper).
        # Surface-level helper (NOT implemented here):
        #   - counts non-return StepCalls and compares to self._tool_calls_limit (if not None)
        self._enforce_budget_stepcalls(step_calls)

        # 6) Compile flat StepCalls -> concurrent batches (surface-level helper).
        # Surface-level helper (NOT implemented here):
        #   - computes batch levels from (deps ∪ await_index), ignoring prefix predecessors
        #   - guarantees return step is isolated as its own final batch
        plan_batches = self._compile_to_batches(step_calls, prefix_len=prefix_len)

        # 7) Allocate run blackboard slots and initialize trackers.
        #
        # Allocate exactly enough slots so that every StepCall.index is a valid slot.
        # We assume StepCall.index values are global and >= prefix_len.
        max_index = max((sc.index for sc in step_calls), default=(prefix_len - 1))
        total_len = max_index + 1

        # Build full run blackboard:
        #   - prefix slots as-is
        #   - then empty slots for the new planned region
        blackboard = prefix + [BlackboardSlot(step=i) for i in range(prefix_len, total_len)]

        # Executed indices are those slots with a realized result (result != NO_VAL).
        executed_indices = {i for i, slot in enumerate(blackboard) if slot.result is not NO_VAL}

        return PlanActRunState(
            messages=working_messages,
            blackboard=blackboard,
            prefix_blackboard_len=prefix_len,
            executed_indices=executed_indices,
            prepared_indices=[],
            tool_calls_used=0,
            is_done=False,
            return_value=NO_VAL,
            plan_batches=plan_batches,
            batch_index=0,
        )
    
    def inject_blackboard(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Inject persisted blackboard context into messages.

        Uses BLACKBOARD_INJECTION_TEMPLATE and the agent's persisted blackboard.

        Contract:
        - Does NOT mutate the input list in-place (returns a new list).
        - Only injects if context_enabled and there is at least one prefix slot.
        """
        if not self.context_enabled or not self._blackboard:
            return list(messages)

        out = [dict(m) for m in messages]
        prefix_len = len(self._blackboard)
        resolvable_cutoff = prefix_len - 1
        new_start = prefix_len

        ctx = self.blackboard_dumps(peek=False)

        injection = BLACKBOARD_INJECTION_TEMPLATE.format(
            RESOLVABLE_CUTOFF=resolvable_cutoff,
            NEW_START=new_start,
            BLACKBOARD_CONTEXT=ctx,
        )

        # Prefer augmenting the last user message if present; otherwise append a new user message.
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                out[i] = {
                    "role": "user",
                    "content": f"{out[i].get('content','')}\n\n{injection}",
                }
                return out

        out.append({"role": "user", "content": injection})
        return out

    def _parse_plan_json(self, raw_text: str) -> list[dict[str, Any]]:
        """
        Parse LLM plan output into a list of dicts.

        Expected canonical schema per element:
            {"tool": "<Tool.full_name>", "args": <any JSON value>, "await": <int|null>}

        Robustness:
        - Strips markdown fences.
        - Accepts either a JSON array directly, or extracts the first JSON array found.
        - Raises ToolAgentError on failure.
        """
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: LLM returned empty plan output.")

        text = raw_text.strip()

        # Strip markdown fences if present.
        text = re.sub(r"^\s*```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text).strip()

        # Fast path: direct JSON array.
        if text.lstrip().startswith("["):
            try:
                parsed = json.loads(text)
            except Exception as exc:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: failed to parse plan JSON array: {exc}"
                ) from exc
        else:
            # Tolerant path: find the first JSON array in the text.
            decoder = json.JSONDecoder()
            parsed = None
            for m in re.finditer(r"\[", text):
                start = m.start()
                try:
                    obj, _end = decoder.raw_decode(text, start)
                except Exception:
                    continue
                if isinstance(obj, list):
                    parsed = obj
                    break
            if parsed is None:
                preview = text if len(text) <= 1200 else text[:1200] + "…"
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: could not locate a JSON array plan in LLM output. "
                    f"Output preview: {preview!r}"
                )

        if not isinstance(parsed, list):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan must be a JSON array; got {type(parsed).__name__!r}."
            )

        plan: list[dict[str, Any]] = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan element {i} must be an object; got {type(item).__name__!r}."
                )
            plan.append(dict(item))

        return plan

    def _plan_dicts_to_stepcalls(self, plan: list[dict[str, Any]], *, prefix_len: int) -> list[StepCall]:
        """
        Convert parsed plan dicts into canonical StepCall objects with global indices.

        Global index assignment:
            StepCall.index = prefix_len + local_step_index
        """
        if not isinstance(prefix_len, int) or prefix_len < 0:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: prefix_len must be int >= 0.")

        step_calls: list[StepCall] = []
        for i, d in enumerate(plan):
            tool_name = d.get("tool")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan step {i} missing/invalid 'tool'."
                )

            # Validate tool exists early (helps catch planner drift).
            if not self.has_tool(tool_name):
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool in plan: {tool_name!r}.")

            args = d.get("args", {})
            await_val = d.get("await", None)

            if await_val is not None and (not isinstance(await_val, int) or await_val < 0):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan step {i} has invalid 'await' (must be int>=0 or null)."
                )

            step_calls.append(
                StepCall(
                    index=prefix_len + i,
                    tool_name=tool_name,
                    args=args,
                    await_index=await_val,
                )
            )

        return step_calls

    def _ensure_and_validate_return(self, step_calls: list[StepCall], *, prefix_len: int) -> list[StepCall]:
        """
        Enforce return-step policy:

        - At most one return tool call is allowed.
        - If present, it must be the last step.
        - If absent, append `return(val=None)` as the final step.

        Returns the (possibly extended) list.
        """
        if not step_calls:
            # If the planner emitted nothing, we still return None.
            return [
                StepCall(index=prefix_len, tool_name=return_tool.full_name, args={"val": None}, await_index=None)
            ]

        return_positions = [i for i, sc in enumerate(step_calls) if sc.tool_name == return_tool.full_name]
        if len(return_positions) > 1:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan contains multiple return steps at positions {return_positions!r}."
            )
        if len(return_positions) == 1:
            pos = return_positions[0]
            if pos != len(step_calls) - 1:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: return step must be last; found at position {pos}."
                )
            return step_calls

        # No return: append one.
        next_index = step_calls[-1].index + 1
        return step_calls + [
            StepCall(index=next_index, tool_name=return_tool.full_name, args={"val": None}, await_index=None)
        ]

    def _validate_stepcalls(self, step_calls: list[StepCall], *, prefix_len: int) -> None:
        """
        Validate dependency structure according to current rules:

        Let deps(step) be placeholder references found in step.args.

        Rules:
        1) For any dep d in deps: 0 <= d < total_slots and d < step.index (no forward/self refs).
        2) If await_index is None:
             all deps must be from prefix (d < prefix_len).
        3) If await_index is not None:
             - await_index < step.index
             - await_index must be >= max(non-prefix deps), if any
        4) Special normalization rule:
             If await_index is not None and await_index < prefix_len and ALL deps are prefix,
             then await_index is treated as None (we normalize by rebuilding that StepCall in-place).
        """
        if not step_calls:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: internal error: empty step_calls after parsing.")

        total_slots = max(sc.index for sc in step_calls) + 1

        for i, sc in enumerate(step_calls):
            deps = self.extract_dependencies(sc.args)

            for d in deps:
                if d < 0 or d >= total_slots:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: step index {sc.index} depends on out-of-range step {d} "
                        f"(total_slots={total_slots})."
                    )
                if d >= sc.index:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: step index {sc.index} has illegal forward/self dependency "
                        f"on step {d}."
                    )

            await_idx = sc.await_index

            # Normalization rule: await inside prefix becomes None if deps are prefix-only.
            if await_idx is not None and await_idx < prefix_len:
                if all(d < prefix_len for d in deps):
                    step_calls[i] = StepCall(
                        index=sc.index,
                        tool_name=sc.tool_name,
                        args=sc.args,
                        await_index=None,
                    )
                    await_idx = None

            if await_idx is None:
                # Null await means: may only depend on prefix.
                if any(d >= prefix_len for d in deps):
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: step index {sc.index} has await=null but depends on "
                        f"non-prefix steps {sorted(d for d in deps if d >= prefix_len)!r}."
                    )
                continue

            # Non-null await validation.
            if await_idx < 0 or await_idx >= total_slots:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step index {sc.index} has out-of-range await={await_idx}."
                )
            if await_idx >= sc.index:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step index {sc.index} has invalid await={await_idx} "
                    f"(must be < step.index)."
                )

            non_prefix_deps = [d for d in deps if d >= prefix_len]
            if non_prefix_deps:
                required = max(non_prefix_deps)
                if await_idx < required:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: step index {sc.index} has await={await_idx} but depends on "
                        f"step {required}; await must be >= max non-prefix dependency."
                    )

    def _enforce_budget_stepcalls(self, step_calls: list[StepCall]) -> None:
        """
        Optional plan-time budget enforcement.

        Counts only non-return steps and compares to tool_calls_limit (None => unlimited).
        """
        limit = self.tool_calls_limit
        if limit is None:
            return

        non_return = sum(1 for sc in step_calls if sc.tool_name != return_tool.full_name)
        if non_return > limit:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan exceeds tool_calls_limit={limit} "
                f"(non-return steps={non_return})."
            )

    def _compile_to_batches(self, step_calls: list[StepCall], *, prefix_len: int) -> list[list[StepCall]]:
        """
        Compile a flat list of StepCalls into concurrent executable batches.

        Levelization algorithm:
        - prerequisites(step) = deps(step.args) ∪ ({await_index} if not None else ∅)
        - prerequisites in prefix (< prefix_len) are considered already satisfied
        - level(step) = 0 if no planned prerequisites
                      else 1 + max(level(prereq)) over planned prerequisites

        Return tool is always isolated as its own final batch (policy).
        """
        if not step_calls:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: cannot compile empty step_calls.")

        # Return is guaranteed last by _ensure_and_validate_return.
        return_step = step_calls[-1]
        if return_step.tool_name != return_tool.full_name:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: last step is not return after validation."
            )

        planned_steps = step_calls[:-1]
        planned_by_index: dict[int, StepCall] = {sc.index: sc for sc in planned_steps}

        # Compute levels in increasing index order (deps are backward-only by validation).
        level: dict[int, int] = {}
        for sc in sorted(planned_steps, key=lambda s: s.index):
            deps = self.extract_dependencies(sc.args)
            prereqs = set(deps)
            if sc.await_index is not None:
                prereqs.add(sc.await_index)

            planned_prereqs = [p for p in prereqs if p >= prefix_len]
            if not planned_prereqs:
                level[sc.index] = 0
                continue

            missing = [p for p in planned_prereqs if p not in planned_by_index]
            if missing:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step index {sc.index} depends on missing planned steps "
                    f"{sorted(missing)!r}."
                )

            level[sc.index] = 1 + max(level[p] for p in planned_prereqs)

        # Group by level.
        buckets: dict[int, list[StepCall]] = {}
        for sc in planned_steps:
            lvl = level.get(sc.index, 0)
            buckets.setdefault(lvl, []).append(sc)

        batches: list[list[StepCall]] = []
        for lvl in sorted(buckets):
            batch = sorted(buckets[lvl], key=lambda s: s.index)
            if batch:
                batches.append(batch)

        # Always isolate return as a final batch all its own.
        batches.append([return_step])
        return batches

    # ------------------------------------------------------------------ #
    # Subclass Hook: Prepare Next Batch
    # ------------------------------------------------------------------ #
    def _prepare_next_batch(self, state: PlanActRunState) -> PlanActRunState:
        """
        Prepare the next executable batch.

        Responsibilities:
        1) Fetch the next batch from state.plan_batches using state.batch_index
        2) Populate the corresponding blackboard slots:
            - slot.tool
            - slot.args
            - slot.resolved_args (via _resolve_placeholders; enforces deps are executed)
        3) Set state.prepared_indices to the indices for this batch
        4) Increment state.batch_index

        This method must NOT execute tools or update executed_indices.
        """
        if state.prepared_indices:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: cannot prepare next batch while prepared_indices is non-empty."
            )

        if state.batch_index >= len(state.plan_batches):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: no remaining plan batches to prepare (batch_index={state.batch_index})."
            )

        batch = state.plan_batches[state.batch_index]
        if not batch:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: encountered empty plan batch at index {state.batch_index}."
            )

        indices = [sc.index for sc in batch]

        # Validate uniqueness early (double prepare is a correctness bug).
        if len(indices) != len(set(indices)):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan batch {state.batch_index} contains duplicate indices: {indices!r}."
            )

        board = state.blackboard
        board_len = len(board)

        for sc in batch:
            idx = sc.index
            if not isinstance(idx, int):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan batch {state.batch_index} contains non-int index: {idx!r}."
                )
            if idx < 0 or idx >= board_len:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan batch {state.batch_index} index {idx} out of range "
                    f"(blackboard length={board_len})."
                )

            slot = board[idx]

            # Keep the global index invariant tight.
            if isinstance(slot.step, int) and slot.step != idx:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: blackboard slot step mismatch at index {idx}: slot.step={slot.step}."
                )

            # Batch compiler should never schedule these; catch stale/ghost state early.
            if slot.is_executed():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan batch {state.batch_index} references already executed slot {idx}."
                )
            if slot.is_prepared():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan batch {state.batch_index} references already prepared slot {idx}."
                )

            # Populate slot fields required by _execute_prepared_batch().
            slot.tool = sc.tool_name
            slot.args = sc.args
            slot.resolved_args = self._resolve_placeholders(sc.args, state=state)

            # Guard against accidental mutation by placeholder resolution or stale state.
            if slot.result is not NO_VAL:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: internal error: slot {idx} result was set during preparation."
                )

        # Publish the batch as executable and advance.
        state.prepared_indices = sorted(indices)
        state.batch_index += 1
        return state


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
