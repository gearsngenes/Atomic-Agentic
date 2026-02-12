"""
ToolAgents

This module defines the abstract base ToolAgent.

ToolAgent is a template-method runtime that:
- owns a toolbox of Tool instances
- runs an iterative "prepare -> execute" loop against a per-invoke RunState
- supports global blackboard placeholder references: <<__sN__>>
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
from ..core.Prompts import PLANNER_PROMPT, ORCHESTRATOR_PROMPT
from ..core.Exceptions import ToolAgentError


logger = logging.getLogger(__name__)

# Canonical placeholders:
#   <<__si__>>  : result of plan-local step i (0-based within the running plan)
#   <<__ci__>> : result of cache entry i (0-based within persisted cache)
_STEP_TOKEN: re.Pattern[str] = re.compile(r"<<__s(\d+)__>>")
_CACHE_TOKEN: re.Pattern[str] = re.compile(r"<<__c(\d+)__>>")

def extract_dependencies(obj: Any, placeholder_pattern: re.Pattern[str]) -> set[int]:
        """
        Return the set of step/cache indices referenced by placeholder_pattern anywhere in obj.

        placeholder_pattern: <<__ci__>> or <<__si__>>
        
        This does NOT validate readiness; purely structural.
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

      <<__ci__>> is resolvable iff:
        - 0 <= i < len(cache_blackboard)
        - cache_blackboard[i].result is not NO_VAL

      <<__si__>> is resolvable iff:
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


CACHE_INJECTION_TEMPLATE = """\
# CACHE (READ-ONLY)
You may reuse results from the following CACHE entries using ONLY these placeholders:
- <<__ci__>> : result of cache entry i (0-based)

For the NEW LIST OF STEPS you are about to create:
- If you are starting a NEW TASK, step indices start at 0
- If you are continuing an ONGOING TASK with N existing steps (i = 0 to N - 1), continue from step N 
- <<__si__>> may ONLY reference earlier steps in THIS plan (no forward refs).
- <<__ci__>> may ONLY reference CACHE entries.

CACHE:
{CACHE_CONTEXT}
"""

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
    def blackboard_dumps(self, *, peek: bool = False, indent = 2, width = 160) -> str:
        """
        Pretty representation of the persisted blackboard.

        peek=False -> placeholder-view (args)
        peek=True  -> resolved-view (resolved_args)

        Includes explicit `step` numbering to make global indexing unambiguous.
        If a slot's `step` is NO_VAL, the list position is used.
        """
        key = "resolved_args" if peek else "args"
        view: list[dict[str, Any]] = []

        for i, d in enumerate(self.blackboard_serialized):
            d["step"] = i
            d.pop("resolved_args")
            if d["error"] is NO_VAL:
                d.pop("error")
            if not peek:
                d.pop("result")
            view.append(d)

        try:
            return pprint.pformat(view, indent=indent, width=width)
        except Exception:  # pragma: no cover
            return str(view)

    def inject_blackboard(self, messages: list[dict[str, str]], peek = False) -> list[dict[str, str]]:
        """
        Inject persisted CACHE context into messages.

        Cache indices are 0-based and correspond to <<__ci__>>.
        Does not mutate the input list in-place.
        """
        if not self.context_enabled or not self._blackboard:
            return list(messages)

        out = [dict(m) for m in messages]

        # Ensure cache slot indices are coherent with list positions.
        # (Base resolver uses list indices, so we normalize here for clarity.)
        for i, slot in enumerate(self._blackboard):
            slot.step = i

        cache_ctx = self.blackboard_dumps(peek=peek)
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
    # Placeholder resolution helpers (prepare-time)
    # ------------------------------------------------------------------ #
    def _resolve_placeholders(self, obj: Any, *, state: ToolAgentRunState) -> Any:
        """
        Resolve placeholders recursively.

        Supported:
        - <<__si__>>  : plan-local running step i
        - <<__ci__>> : cache entry i

        Readiness rules:
        - referenced slot must exist
        - referenced slot must be executed (result != NO_VAL)

        Semantics:
        - Full-string placeholder returns referenced result as-is (preserves type)
        - Inline occurrences inside larger strings are replaced with repr(result) (fallback to str)
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
    def update_blackboard(self, state: ToolAgentRunState) -> ToolAgentRunState:
        """
        Persist this run into the agent's cache blackboard (self._blackboard).

        Policy:
        - Cache is state.cache_blackboard (snapshot of self._blackboard at invoke start)
        - Append executed running plan slots (trim empty/unplanned tail first)
        - Rewrite any <<__sj__>> placeholders inside appended slot.args into
        <<__c{base_len + j}__>> so cached args never contain step-local placeholders.

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

        extracted = [{"step":slot.step, "tool":slot.tool, "args": slot.args} for slot in appended]
        newest_dump = ",".join([f"\n  {str(step)}" for step in extracted])
        newest_dump = f"CACHE STEPS #{appended[0].step}-{appended[-1].step} PRODUCED:\n\n[{newest_dump}\n]"
        state.messages.append({"role":"assistant", "content": newest_dump})

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
            state = self.update_blackboard(state)

        newest_history = [original_user_msg, state.messages[-1]]
        return newest_history, state.return_value

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


# --------------------------------------------------------------------------- #
# PlanAct Agent
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PlannedStep:
    """
    Minimal, normalized plan-step representation.

    - tool: Tool.full_name (must exist in toolbox)
    - args: raw/unresolved args (may contain <<__si__>> / <<__ci__>> placeholders)
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
    def from_dict(data: Mapping[str, Any]) -> PlannedStep:
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

        raw_plan = self._llm_engine.invoke(working_messages)
        plan_dicts = self._str_to_steps(raw_plan)

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
            # Convert to PlannedStep object
            ps = PlannedStep.from_dict(d)
            # Validate tool's existence
            if not self.has_tool(ps.tool):
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool in plan: {ps.tool!r}.")
            # Add to planned steps
            planned.append(ps)

        # Validate that return is the final step in the plan
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
            cache_refs = extract_dependencies(ps.args, placeholder_pattern=_CACHE_TOKEN)
            bad_cache = [k for k in cache_refs if k < 0 or k >= cache_len]
            if bad_cache:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: plan step {i} references out-of-range cache indices "
                    f"{sorted(set(bad_cache))!r} (cache length={cache_len})."
                )

            # Validate plan-local deps range (deps were derived from <<__sj__>> and await)
            # Return is special-cased below.
            if not ps.is_return:
                bad = [d for d in ps.deps if d < 0 or d >= i]
                if bad:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: plan step {i} has illegal deps {sorted(set(bad))!r}; "
                        f"deps must be < {i}."
                    )
            else:
                # Return args may contain <<__sj__>>; they must be < return_idx.
                step_refs = extract_dependencies(ps.args, placeholder_pattern=_STEP_TOKEN)
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
@dataclass(slots=True)
class ReActRunState(ToolAgentRunState):
    """
    ReAct run state extends ToolAgentRunState with:

    - next_step_index:
        The next plan-local running_blackboard index to fill during preparation.
        This is also the "prefix length" cutoff for dependency legality: any <<__sN__>>
        placeholder in a newly prepared batch must satisfy N < next_step_index.

    - latest_executed:
        Plan-local indices for the most recently *prepared* batch. This value is used
        at the start of the next prepare() call to inject "Most recently executed steps
        and results" into the LLM messages. It is overwritten at the end of every
        prepare() call with the newly emitted indices.
    """
    next_step_index: int = 0
    latest_executed: list[int] = field(default_factory=list)


class ReActAgent(ToolAgent[ReActRunState]):
    """
    Dynamic, iterative tool-using agent (ReAct-style), but with v3 semantics:

    - Each LLM turn emits the NEXT CONCURRENT BATCH as a JSON array of steps.
    - Each step must be a dict with:
        {"tool": "<Tool.full_name>", "args": { ... } }

    The base ToolAgent owns the invariant prepare -> execute loop. This subclass only:
    - initializes a fixed-size run blackboard (tool_calls_limit + 1, including return)
    - prepares the next batch by:
        * injecting the most recently executed steps/results (if available)
        * asking for the next executable steps
        * parsing + validating
        * filling the next contiguous segment of the run blackboard
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        *,
        context_enabled: bool = False,
        tool_calls_limit: int = 25,
        pre_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        post_invoke: AtomicInvokable | Callable[..., Any] | None = None,
        history_window: int | None = None,
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

        # ReAct requires a concrete integer tool_calls_limit so that we can preallocate
        # a fixed-size running blackboard.
        if not isinstance(tool_calls_limit, int) or tool_calls_limit < 0:
            raise ToolAgentError("ReActAgent requires tool_calls_limit to be an int >= 0.")

    # ------------------------------------------------------------------ #
    # Tool-Agent Hooks
    # ------------------------------------------------------------------ #
    @property
    def tool_calls_limit(self) -> int:
        """Max allowed non-return tool calls per invoke() run. None means unlimited."""
        return self._tool_calls_limit

    @tool_calls_limit.setter
    def tool_calls_limit(self, value: int) -> None:
        if not isinstance(value, int) or value < 0:
            raise ToolAgentError("tool_calls_limit must be None or an int >= 0.")
        self._tool_calls_limit = value

    # ------------------------------------------------------------------ #
    # Tool-Agent Hooks
    # ------------------------------------------------------------------ #
    def _initialize_run_state(self, *, messages: list[dict[str, str]]) -> ReActRunState:
        if not messages:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: messages must be non-empty.")

        if self._tool_calls_limit is None or not isinstance(self._tool_calls_limit, int) or self._tool_calls_limit < 0:
            raise ToolAgentError("ReActAgent requires tool_calls_limit to be an int >= 0.")

        cache_blackboard = list(self._blackboard)

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
        Prepare the next concurrent batch.

        Uses the run state's message history, optionally injecting a single assistant
        message describing the most recently executed steps/results, followed by a
        simple user request for the next executable steps.

        Then:
          - parses the JSON array of steps (tool + args dict)
          - moves return to the end if present (at most one)
          - validates no intra-batch dependencies via <<__sN__>> where N >= next_step_index
          - fills the next contiguous segment of the preallocated running_blackboard
          - sets prepared_steps and overwrites latest_executed to those indices
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
            obs_payload: list[dict[str, Any]] = []
            for idx in state.latest_executed:
                if idx < 0 or idx >= len(state.running_blackboard):
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: latest_executed contains out-of-range index {idx}."
                    )
                slot = state.running_blackboard[idx]
                obs_payload.append(
                    {
                        "step": slot.step,
                        "tool": slot.tool,
                        "args": slot.args,
                        "result": slot.result,
                    }
                )

            obs_text = "Most recently executed steps and results:\n" + pprint.pformat(
                obs_payload, indent=2, sort_dicts=False
            )
            working_messages.append({"role": "assistant", "content": obs_text})

            working_messages.append(
                {
                    "role": "user",
                    "content": "Given the most recently executed steps and available CACHE (if provided) above, "
                            "strategize and produce the next set of executable steps as a single JSON array of "
                            "step dicts.",
                }
            )

        # Persist the augmented message history for subsequent turns.
        state.messages = working_messages

        # ------------------------------------------------------------------ #
        # 2) Call LLM and parse steps
        # ------------------------------------------------------------------ #
        raw_text = self._llm_engine.invoke(working_messages)
        step_dicts = self._str_to_steps(raw_text)

        # Strict: tool + args only; args must be a dict.
        planned_steps: list[PlannedStep] = []
        for d in step_dicts:
            if not isinstance(d, dict):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: each step must be an object; got {type(d).__name__!r}."
                )
            tool = d.get("tool", None)
            args = d.get("args", None)
            if not isinstance(tool, str) or not tool.strip():
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: step 'tool' must be a non-empty string.")
            if not isinstance(args, dict):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step 'args' must be a dict; got {type(args).__name__!r}."
                )

            # Enforce no extra keys early (PlannedStep.from_dict also validates).
            allowed = {"tool", "args"}
            extra = set(d.keys()) - allowed
            if extra:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step contains unsupported keys: {sorted(extra)!r}."
                )

            planned_steps.append(PlannedStep.from_dict({"tool": tool, "args": args}))

        if not planned_steps:
            # Base loop will also reject empty prepared_steps, but this gives clearer context.
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: LLM produced an empty batch.")

        # ------------------------------------------------------------------ #
        # 3) Normalize return placement (at most one; move to end if misplaced)
        # ------------------------------------------------------------------ #
        return_name = return_tool.full_name
        return_positions = [i for i, ps in enumerate(planned_steps) if ps.tool == return_name]
        if len(return_positions) > 1:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: batch contains multiple return steps.")
        if return_positions:
            rpos = return_positions[0]
            if rpos != len(planned_steps) - 1:
                ret = planned_steps.pop(rpos)
                planned_steps.append(ret)

        # ------------------------------------------------------------------ #
        # 4) Validate single-batch dependency legality
        # ------------------------------------------------------------------ #
        prefix_len = state.next_step_index
        for ps in planned_steps:
            for dep in extract_dependencies(obj=ps.args, placeholder_pattern=_STEP_TOKEN):
                if dep >= prefix_len:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: illegal intra-batch dependency: "
                        f"step args reference <<__s{dep}__>> but prefix cutoff is {prefix_len}."
                    )

        # ------------------------------------------------------------------ #
        # 5) Fill the next contiguous segment of the preallocated running blackboard
        # ------------------------------------------------------------------ #
        end = prefix_len + len(planned_steps)
        if end > len(state.running_blackboard):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: batch would exceed run blackboard capacity "
                f"({end} > {len(state.running_blackboard)})."
            )

        new_indices: list[int] = []
        for i, ps in enumerate(planned_steps):
            idx = prefix_len + i
            slot = state.running_blackboard[idx]
            if not slot.is_empty():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: attempted to prepare into non-empty slot {idx}."
                )

            # Validate tool exists before we stamp it into the slot.
            self.get_tool(ps.tool)

            slot.tool = ps.tool
            slot.args = ps.args
            slot.resolved_args = self._resolve_placeholders(ps.args, state=state)
            slot.result = NO_VAL
            slot.error = NO_VAL
            new_indices.append(idx)

        # prepared_steps is what the base execute() will run next.
        state.prepared_steps = list(new_indices)

        # Advance cursor.
        state.next_step_index = end

        # Per approved semantics: overwrite latest_executed with the newly emitted indices.
        # This will be used for injection at the start of the next prepare() call.
        state.latest_executed = list(new_indices)

        return state
