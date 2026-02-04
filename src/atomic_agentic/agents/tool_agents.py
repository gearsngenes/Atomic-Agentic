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
from typing import Any, Callable, Dict, Generic, Mapping, Optional, Sequence, List, TypeVar, Iterable
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

# Canonical placeholders:
#   <<__step_i__>>  : result of plan-local step i (0-based within the running plan)
#   <<__cache_i__>> : result of cache entry i (0-based within persisted cache)
_STEP_TOKEN: re.Pattern[str] = re.compile(r"<<__step_(\d+)__>>")
_CACHE_TOKEN: re.Pattern[str] = re.compile(r"<<__cache_(\d+)__>>")



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

    - cache_blackboard: snapshot of persisted cache (read-only for this run)
    - running_blackboard: plan-local slots (0..N-1), populated/prepared/executed during this run

    Placeholder resolvability:

      <<__cache_i__>> is resolvable iff:
        - 0 <= i < len(cache_blackboard)
        - cache_blackboard[i].result is not NO_VAL

      <<__step_i__>> is resolvable iff:
        - 0 <= i < len(running_blackboard)
        - running_blackboard[i].result is not NO_VAL
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
        Resolve placeholders recursively.

        Supported:
        - <<__step_i__>>  : plan-local running step i
        - <<__cache_i__>> : cache entry i

        Readiness rules:
        - referenced slot must exist
        - referenced slot must be executed (result != NO_VAL)

        Semantics:
        - Full-string placeholder returns referenced result as-is (preserves type)
        - Inline occurrences inside larger strings are replaced with repr(result) (fallback to str)
        """
        cache = state.cache_blackboard
        running = state.running_blackboard

        needed_cache: set[int] = set()
        needed_steps: set[int] = set()

        # ----------------------------
        # 1) Collect all dependencies.
        # ----------------------------
        def collect(x: Any) -> None:
            if isinstance(x, str):
                for m in _CACHE_TOKEN.finditer(x):
                    needed_cache.add(int(m.group(1)))
                for m in _STEP_TOKEN.finditer(x):
                    needed_steps.add(int(m.group(1)))
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
            if running[idx].result is NO_VAL:
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

    def extract_step_dependencies(self, obj: Any) -> set[int]:
        """
        Return the set of plan-local step indices referenced by <<__step_i__>> anywhere in obj.

        This does NOT validate readiness; purely structural.
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

        walk(obj)
        return deps

    def extract_cache_dependencies(self, obj: Any) -> set[int]:
        """
        Return the set of cache indices referenced by <<__cache_i__>> anywhere in obj.

        This does NOT validate readiness; purely structural.
        """
        deps: set[int] = set()

        def walk(x: Any) -> None:
            if isinstance(x, str):
                for m in _CACHE_TOKEN.finditer(x):
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

    # ------------------------------------------------------------------ #
    # Execution (base-owned)
    # ------------------------------------------------------------------ #
    def _execute_prepared_batch(self, state: RS) -> RS:
        """
        Execute the currently prepared batch described by state.prepared_steps (plan-local indices).

        Records results into state.running_blackboard.
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

        def run_one(idx: int) -> tuple[int, Any]:
            slot = board[idx]
            tool_name = slot.tool
            tool = self.get_tool(tool_name)
            try:
                result = tool.invoke(slot.resolved_args)
            except ToolInvocationError:
                raise
            except Exception as exc:  # pragma: no cover
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool call failed at step {idx} for {tool_name!r}: {exc}"
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


    # ------------------------------------------------------------------ #
    # Finalization helpers
    # ------------------------------------------------------------------ #
    def update_blackboard(self, state: ToolAgentRunState) -> None:
        """
        Persist this run into the agent's cache blackboard (self._blackboard).

        Policy:
        - Cache is state.cache_blackboard (snapshot of self._blackboard at invoke start)
        - Append executed running plan slots (trim empty/unplanned tail first)
        - Rewrite any <<__step_j__>> placeholders inside appended slot.args into
        <<__cache_{base_len + j}__>> so cached args never contain step-local placeholders.

        Also trims any empty tail slots from the final persisted cache.
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
            Rewrite <<__step_j__>> -> <<__cache_{base_len + j}__>> recursively.
            Leaves <<__cache_k__>> unchanged.
            """
            if isinstance(obj, str):
                # exact placeholder: still rewrite as a string placeholder (we are rewriting args,
                # not resolving)
                def repl(m: re.Match[str]) -> str:
                    j = int(m.group(1))
                    return f"<<__cache_{base_len + j}__>>"

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

        # Persist run outputs into cache if context is enabled.
        if self.context_enabled:
            self.update_blackboard(state)

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

# --------------------------------------------------------------------------- #
# PlanAct: cache injection + planned-step normalization
# --------------------------------------------------------------------------- #

CACHE_INJECTION_TEMPLATE = """\
# CACHE (READ-ONLY)
You may reuse results from the following CACHE entries using ONLY these placeholders:
- <<__cache_i__>> : result of cache entry i (0-based)

For the NEW plan you are about to create:
- Step indices start at 0 and use <<__step_i__>>.
- <<__step_i__>> may ONLY reference earlier steps in THIS plan (no forward refs).
- <<__cache_i__>> may ONLY reference CACHE entries.

CACHE:
{CACHE_CONTEXT}
"""


@dataclass(frozen=True, slots=True)
class PlannedStep:
    """
    Minimal, normalized plan-step representation.

    - tool: Tool.full_name (must exist in toolbox)
    - args: raw/unresolved args (may contain <<__step_i__>> / <<__cache_i__>> placeholders)
    - deps: plan-local prerequisite indices (includes placeholder step deps + await barrier if provided)
    - is_return: whether this is the canonical return tool step
    """
    tool: str
    args: Any
    deps: frozenset[int]
    is_return: bool

    def to_dict(self) -> dict[str, Any]:
        # NOTE: await is intentionally not preserved. Deps is canonical.
        return {"tool": self.tool, "args": self.args}

    @staticmethod
    def from_dict(
        data: Mapping[str, Any],
        *,
        extract_step_deps: Callable[[Any], set[int]],
    ) -> "PlannedStep":
        if not isinstance(data, Mapping):
            raise ToolAgentError(
                f"PlannedStep.from_dict requires a mapping; got {type(data).__name__!r}."
            )

        allowed = {"tool", "args", "await"}
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

        deps: set[int] = set(extract_step_deps(args))

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
    PlanAct run state extends ToolAgentRunState with:
      - batches: compiled executable batches as plan-local indices
      - batch_index: next batch to prepare
    """
    batches: list[list[int]] = field(default_factory=list)
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
    # Cache injection
    # ------------------------------------------------------------------ #
    def inject_blackboard(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Inject persisted CACHE context into messages.

        Cache indices are 0-based and correspond to <<__cache_i__>>.
        Does not mutate the input list in-place.
        """
        if not self.context_enabled or not self._blackboard:
            return list(messages)

        out = [dict(m) for m in messages]

        # Ensure cache slot indices are coherent with list positions.
        # (Base resolver uses list indices, so we normalize here for clarity.)
        for i, slot in enumerate(self._blackboard):
            slot.step = i

        cache_ctx = self.blackboard_dumps(peek=False)
        injection = CACHE_INJECTION_TEMPLATE.format(CACHE_CONTEXT=cache_ctx)

        # Prefer augmenting the last user message; otherwise append a new user message.
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                out[i] = {
                    "role": "user",
                    "content": f"{out[i].get('content','')}\n\n{injection}",
                }
                return out

        out.append({"role": "user", "content": injection})
        return out

    # ------------------------------------------------------------------ #
    # Planning + initialization
    # ------------------------------------------------------------------ #
    def _initialize_run_state(self, *, messages: list[dict[str, str]]) -> PlanActRunState:
        """
        One-shot plan generation + compilation into executable batches.

        Algorithm:
          1) snapshot cache_blackboard (if context enabled)
          2) inject cache context into messages (if enabled)
          3) call LLM to produce plan JSON array
          4) parse JSON -> list[dict]
          5) normalize return: ensure exactly one return dict and it is last
             - if missing, append return(None)
             - if misplaced, move to end
             - if return has 'await', ignore it (allowed)
          6) dict -> PlannedStep (deps = placeholder deps + await barrier)
          7) validate tools, deps ranges, cache placeholder ranges, budget
             - return deps: force deps = {0..return_idx-1} (ignore await)
          8) compile batches from deps (return isolated final batch)
          9) allocate running_blackboard plan-locally and pre-fill tool+args
        """
        if not messages:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: messages must be non-empty.")

        working_messages = [dict(m) for m in messages]

        cache_blackboard: list[BlackboardSlot] = list(self._blackboard) if self.context_enabled else []
        for i, slot in enumerate(cache_blackboard):
            slot.step = i

        if self.context_enabled and cache_blackboard:
            working_messages = self.inject_blackboard(working_messages)

        raw_plan = self._llm_engine.invoke(working_messages)
        plan_dicts = self._parse_plan_json(raw_plan)

        # ---- Normalize return: ensure exactly one return step and it is last. ----
        return_name = return_tool.full_name
        return_positions = [i for i, d in enumerate(plan_dicts) if d.get("tool") == return_name]

        if len(return_positions) > 1:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan contains multiple return steps at positions {return_positions!r}."
            )

        if len(return_positions) == 1:
            pos = return_positions[0]
            if pos != len(plan_dicts) - 1:
                ret = plan_dicts.pop(pos)
                plan_dicts.append(ret)
        else:
            plan_dicts.append({"tool": return_name, "args": {"val": None}})

        if not plan_dicts:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: internal error: empty plan after normalization.")

        # ---- Convert to PlannedStep and validate tool existence early. ----
        planned: list[PlannedStep] = []
        for i, d in enumerate(plan_dicts):
            ps = PlannedStep.from_dict(
                d,
                extract_step_deps=self.extract_step_dependencies,
            )

            if not self.has_tool(ps.tool):
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool in plan: {ps.tool!r}.")
            planned.append(ps)

        plan_len = len(planned)
        return_idx = plan_len - 1
        if not planned[return_idx].is_return:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: internal error: last step is not return after normalization."
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
            cache_refs = self.extract_cache_dependencies(ps.args)
            bad_cache = [k for k in cache_refs if k < 0 or k >= cache_len]
            if bad_cache:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan step {i} references out-of-range cache indices "
                    f"{sorted(set(bad_cache))!r} (cache length={cache_len})."
                )

            # Validate plan-local deps range (deps were derived from <<__step_j__>> and await)
            # Return is special-cased below.
            if not ps.is_return:
                bad = [d for d in ps.deps if d < 0 or d >= i]
                if bad:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: plan step {i} has illegal deps {sorted(set(bad))!r}; "
                        f"deps must be < {i}."
                    )
            else:
                # Return args may contain <<__step_j__>>; they must be < return_idx.
                step_refs = self.extract_step_dependencies(ps.args)
                bad_refs = [d for d in step_refs if d < 0 or d >= return_idx]
                if bad_refs:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: return step references out-of-range plan steps "
                        f"{sorted(set(bad_refs))!r} (return index={return_idx})."
                    )

        # ---- Force return deps to be all prior steps (ignore await entirely). ----
        forced_return_deps = frozenset(range(return_idx))
        planned[return_idx] = PlannedStep(
            tool=planned[return_idx].tool,
            args=planned[return_idx].args,
            deps=forced_return_deps,
            is_return=True,
        )

        # ---- Compile batches from deps; isolate return as final batch. ----
        batches = self._compile_batches_from_deps(planned_steps=planned, return_idx=return_idx)

        # ---- Allocate running blackboard plan-locally and prefill tool+args. ----
        running_blackboard = [BlackboardSlot(step=i) for i in range(plan_len)]
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

    # ------------------------------------------------------------------ #
    # Prepare next batch
    # ------------------------------------------------------------------ #
    def _prepare_next_batch(self, state: PlanActRunState) -> PlanActRunState:
        """
        Prepare exactly one batch:
          - reads next batch indices from state.batches[state.batch_index]
          - resolves args for those indices (writes slot.resolved_args)
          - sets state.prepared_steps
          - increments batch_index
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
