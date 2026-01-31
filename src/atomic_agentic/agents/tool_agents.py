"""Agents

This module contains agent implementations built on top of the core
:class:`~atomic_agentic.Primitives.Agent` primitive.

The current focus is the refactored :class:`ToolAgent`, an abstract base class
for *tool-using* agents (e.g. Planner/Orchestrator) that:

- owns a toolbox of :class:`~atomic_agentic.Tools.Tool` instances
- executes tools via a single gateway (:meth:`ToolAgent._call_tool`)
- records executed steps in a blackboard (a list of minimal step dicts)

ToolAgent uses a Template Method pattern:

* The base :meth:`ToolAgent._invoke` is **final** and implements the common
  lifecycle (reset counters, manage blackboard context, commit history).
* Subclasses implement :meth:`ToolAgent._run` to decide what tools to call and
  what value to return.

Concurrency / thread-safety
---------------------------
ToolAgent serializes *external* concurrent calls to :meth:`invoke` (and other
public mutators like :meth:`register`) via a single per-instance lock
(``self._invoke_lock``). This keeps the external API simple while still allowing
a subclass to run *intra-invoke* tool calls concurrently (e.g. via threads)
because tool execution does **not** require holding ``self._invoke_lock``.

If a subclass does parallelize tool calls within a single run, the tool-call
budget remains correct because it is enforced by ``self._tool_calls_lock``.

Blackboard rules
----------------
- ``self._blackboard`` is *persisted* across runs only when ``context_enabled``
  is True.
- Subclasses **must not** mutate ``self._blackboard`` inside :meth:`_run`.
  Instead, they should build/execute NEW steps and return them from :meth:`_run`.
- The base class commits the returned steps to ``self._blackboard`` *after*
  :meth:`_run` completes.

Step placeholders
----------------
The canonical placeholder format is ``<<__step__N>>`` where N is a 0-based step
index into the relevant board. Resolution rules:

- Full-string ``<<__step__N>>`` returns the referenced result as-is (preserves type)
- Inline occurrences inside larger strings are replaced with ``repr(result)``

By default, placeholders are resolved against the persisted blackboard from
prior runs. If you need resolution against “persisted steps + new-steps-so-far”
during a run, pass an explicit ``board=`` to :meth:`_call_tool` /
:meth:`_resolve_step_refs`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import ast
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Iterable,)
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
import json
import pprint
import string

from .base import Agent
from ..core.Exceptions import (
    ToolAgentError,
    ToolDefinitionError,
    ToolInvocationError,
    ToolRegistrationError,
)
from ..core.Invokable import AtomicInvokable
from ..engines.LLMEngines import LLMEngine
from ..tools import Tool, toolify, batch_toolify
from ..core.Prompts import PLANNER_PROMPT, ORCHESTRATOR_PROMPT

logger = logging.getLogger(__name__)

# Canonical step placeholder: <<__step__N>> where N is a 0-based step index.
_STEP_TOKEN: re.Pattern[str] = re.compile(r"<<__step__(\d+)>>")


class BlackboardEntry(TypedDict):
    """Minimal required blackboard entry for ToolAgent."""
    tool: str
    args: Any
    resolved_args: Any
    completed: bool
    result: Any


@dataclass(frozen=True, slots=True)
class StepCall:
    """Planned step call emitted by a planner/orchestrator (may include placeholders)."""
    index: int
    tool_name: str
    args: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResolvedStepCall:
    """Execution-ready step call (placeholders resolved into raw_args)."""
    index: int
    tool_name: str
    args: Any          # placeholder-view
    resolved_args: Any      # resolved


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
        "Returns the passed-in value. This method should ONLY OCCUR ONCE as the "
        "FINAL STEP of any plan."
    ),
)

# ───────────────────────────────────────────────────────────────────────────────
# Base 'ToolAgent' class
# ───────────────────────────────────────────────────────────────────────────────
class ToolAgent(Agent, ABC):
    """
    Abstract base class for tool-using agents.

    The base class owns the invariant runtime loop in `_invoke(messages=...)`:

    - per-run mutable run-state (dict) with copies of messages + blackboard
    - iterative planning (subclass hook) -> batch of StepCall
    - base validation + placeholder resolution -> list[ResolvedStepCall]
    - base budget enforcement (non-return only)
    - base concurrent execution (fail-fast) -> list[BlackboardEntry]
    - commit blackboard entries to the run-state blackboard in batch list order
    - terminate when return tool executes; persist compact history + blackboard

    Subclasses implement only:
    - `_initialize_run_state(...) -> dict`
    - `_prepare_next_steps_batch(state) -> (state, list[StepCall])`
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: str,
        context_enabled: bool = False,
        *,
        tool_calls_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable] = None,
        post_invoke: Optional[AtomicInvokable | Callable] = None,
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
        self._blackboard: List[BlackboardEntry] = []

        self._tool_calls_limit: Optional[int] = None
        self.tool_calls_limit = tool_calls_limit

        # Always include the canonical return tool (avoid collisions by skipping).
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
        tool_calls_limit_text = "unlimited" if self._tool_calls_limit is None else str(self._tool_calls_limit)
        try:
            return template.format(
                TOOLS=self.actions_context(),
                TOOL_CALLS_LIMIT=tool_calls_limit_text,
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
    def blackboard(self) -> List[BlackboardEntry]:
        """Read-only view (shallow copy) of the persisted blackboard."""
        with self._invoke_lock:
            return list(self._blackboard)

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

    def list_tools(self) -> Dict[str, Tool]:
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
        mcp_servers: Sequence[Tuple[str, Any]] = (),
        a2a_servers: Sequence[Tuple[str, Any]] = (),
        *,
        name_collision_mode: str = "raise",
    ) -> List[str]:
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

        registered: List[str] = []
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
    def blackboard_dumps(
        self,
        obj: Optional[List[BlackboardEntry]] = None,
        *,
        peek: bool = False,
    ) -> str:
        if obj is None:
            obj = self.blackboard

        args_key = "resolved_args" if peek else "args"
        view = [{"tool": step.get("tool"), "args": step.get(args_key, None)} for step in obj]

        try:
            return pprint.pformat(view, indent=2) #json.dumps(view, indent=indent, default=str)
        except Exception:  # pragma: no cover
            return str(view)

    # ------------------------------------------------------------------ #
    # Placeholder resolution (supports literal-eval types)
    # ------------------------------------------------------------------ #
    def _step_result_by_index(self, idx: int, *, board: Sequence[BlackboardEntry]) -> Any:
        if not isinstance(idx, int):
            raise ToolAgentError(f"Step reference must be an int; got {type(idx).__name__!r}.")
        if idx < 0:
            raise ToolAgentError(f"Step reference must be >= 0; got {idx}.")
        if idx >= len(board):
            raise ToolAgentError(f"Step reference {idx} out of range (blackboard length={len(board)}).")

        step = board[idx]
        if not bool(step.get("completed", False)):
            raise ToolAgentError(f"Referenced step {idx} has not been completed yet.")
        if "result" not in step:
            raise ToolAgentError("Blackboard entry missing 'result'.")
        return step["result"]

    def _resolve_step_refs(self, obj: Any, *, board: Sequence[BlackboardEntry]) -> Any:
        """
        Resolve `<<__step__N>>` placeholders recursively.

        Important semantics:
        - If the entire string is exactly `<<__step__N>>`, return the referenced
          result **as-is**, preserving its Python type (even if not a builtin).
        - If the placeholder appears inside a larger string, substitute it with
          `repr(result)` (fallback to `str(result)`).
        """
        # Strings: full-token returns object as-is, inline gets repr-substitution.
        if isinstance(obj, str):
            m = _STEP_TOKEN.fullmatch(obj)
            if m:
                return self._step_result_by_index(int(m.group(1)), board=board)

            def repl(match: re.Match[str]) -> str:
                result = self._step_result_by_index(int(match.group(1)), board=board)
                try:
                    return repr(result)
                except Exception:
                    return str(result)

            return _STEP_TOKEN.sub(repl, obj)

        # Containers (ast.literal_eval compatible)
        if isinstance(obj, list):
            return [self._resolve_step_refs(v, board=board) for v in obj]

        if isinstance(obj, tuple):
            return tuple(self._resolve_step_refs(v, board=board) for v in obj)

        if isinstance(obj, set):
            return {self._resolve_step_refs(v, board=board) for v in obj}

        if isinstance(obj, dict):
            return {k: self._resolve_step_refs(v, board=board) for k, v in obj.items()}

        # Scalars / other: bool, None, numbers, bytes, etc. (return as-is)
        return obj

    # ------------------------------------------------------------------ #
    # Base invariant resolve/validate/execute
    # ------------------------------------------------------------------ #
    def _resolve_steps_batch_args(
        self,
        *,
        state: Dict[str, Any],
        batch: List[StepCall],
    ) -> List[ResolvedStepCall]:
        """
        Base-owned phase: validate + resolve planned StepCalls into ResolvedStepCalls.

        Enforced here:
        - StepCall structural correctness
        - Tool existence
        - Return-step constraints (<= 1 return per batch)
        - Placeholder legality: ALL refs must be to already executed steps
          (i.e., indices < len(pre_batch_blackboard))
        - Placeholder resolution (snapshot-based) into raw_args
        """
        if not isinstance(state, dict):
            raise ToolAgentError("run-state must be a dict.")
        if "blackboard" not in state:
            raise ToolAgentError("run-state missing required key 'blackboard'.")
        board_snapshot = state["blackboard"]

        # Tool existence + structure checks + return constraint.
        return_count = 0

        resolved: List[ResolvedStepCall] = []

        for i, step in enumerate(batch):
            if not isinstance(step, StepCall):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch item {i} is not a StepCall "
                    f"(got {type(step).__name__!r})."
                )

            if not isinstance(step.index, int) or step.index < 0:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step.index must be int >= 0; got {step.index!r}."
                )
            if not isinstance(step.tool_name, str) or not step.tool_name.strip():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: step.tool_name must be a non-empty str; got {step.tool_name!r}."
                )

            # Validate tool exists.
            self.get_tool(step.tool_name)

            if step.tool_name == return_tool.full_name:
                return_count += 1
                if return_count > 1:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: a batch may contain at most one return step."
                    )

            # Placeholder legality: all refs must be to already executed steps.
            # Resolve placeholders into execution args (may return non-dict args; tool decides).
            raw_args = self._resolve_step_refs(step.args, board=board_snapshot)

            resolved.append(
                ResolvedStepCall(
                    index=step.index,
                    tool_name=step.tool_name,
                    args=step.args,
                    resolved_args=raw_args,
                )
            )

        return resolved

    def _execute_steps_batch(self, *, resolved_batch: List[ResolvedStepCall]) -> List[BlackboardEntry]:
        """
        Base-owned phase: execute tools concurrently (fail-fast) and return blackboard entries
        in *step-index order* (ascending by `ResolvedStepCall.index`).

        Key invariants enforced here (to keep blackboard/placeholder semantics sane):
        - Each step.index in the batch must be unique.
        - The returned list is ordered by step.index (NOT by completion order, and not necessarily
        by the incoming `resolved_batch` order if a planner emits them out of order).
        - This method does NOT mutate the run-state blackboard; `_invoke` appends entries.

        Notes:
        - We intentionally do not “fill gaps” for missing indices; if a batch contains
        {index: 10, index: 12} we return two entries ordered [10, 12]. If your runtime
        requires contiguous indices, enforce that earlier (e.g., in batch validation).
        """
        if not resolved_batch:
            raise ToolAgentError("resolved_batch must be non-empty.")

        # ---- Enforce and prepare index ordering ----
        indices: List[int] = [step.index for step in resolved_batch]
        if any((not isinstance(i, int)) for i in indices):
            raise ToolAgentError("All resolved step indices must be ints.")
        if any(i < 0 for i in indices):
            raise ToolAgentError("All resolved step indices must be >= 0.")

        # Uniqueness is critical: duplicate indices would make ordering ambiguous and
        # would corrupt placeholder semantics (which rely on stable step positions).
        if len(set(indices)) != len(indices):
            # Keep a helpful error message that pinpoints duplicates.
            seen: set[int] = set()
            dups: List[int] = []
            for i in indices:
                if i in seen and i not in dups:
                    dups.append(i)
                seen.add(i)
            raise ToolAgentError(
                f"resolved_batch contains duplicate step index/indices: {sorted(dups)}. "
                "Each step in a batch must have a unique index."
            )

        # We'll execute in the provided order, but we will *emit* results in index order.
        # Sorting here gives us a deterministic final order and a deterministic mapping
        # from step.index -> output position.
        steps_sorted: List[ResolvedStepCall] = sorted(resolved_batch, key=lambda s: s.index)

        # Preallocate by sorted position so we can fill from futures without re-sorting entries.
        results_by_sorted_pos: List[Optional[BlackboardEntry]] = [None] * len(steps_sorted)

        def run_one(step: ResolvedStepCall) -> Tuple[int, BlackboardEntry]:
            """
            Execute a single tool call and return (step.index, entry).
            Returning the index lets the caller place results deterministically regardless
            of completion order.
            """
            tool = self.get_tool(step.tool_name)
            try:
                # Tools generally expect Mapping[str, Any], but we allow richer literal types
                # as args; a tool may normalize/validate internally.
                result = tool.invoke(step.resolved_args)
            except ToolInvocationError:
                raise
            except Exception as exc:  # pragma: no cover
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool call failed for {step.tool_name!r}: {exc}"
                ) from exc

            entry: BlackboardEntry = {
                "tool": step.tool_name,
                "raw_args": step.args,
                "resolved_args": step.resolved_args,
                "completed": True,
                "result": result,
            }
            return step.index, entry

        # Fast path for single-step batches.
        if len(steps_sorted) == 1:
            _idx, entry = run_one(steps_sorted[0])
            return [entry]

        # Map index -> sorted position for O(1) placement.
        index_to_sorted_pos: Dict[int, int] = {step.index: pos for pos, step in enumerate(steps_sorted)}

        # Execute concurrently, fail-fast on the first exception.
        # (The executor is shut down in a finally-block below; futures may still run briefly,
        # but we stop collecting results as soon as an exception is raised.)
        executor = ThreadPoolExecutor()
        try:
            future_to_index = {executor.submit(run_one, step): step.index for step in steps_sorted}
            for fut in as_completed(future_to_index):
                idx = future_to_index[fut]
                # fut.result() raises if the tool call raised -> fail-fast
                _idx, entry = fut.result()
                pos = index_to_sorted_pos[idx]
                results_by_sorted_pos[pos] = entry
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        # Guarantee all entries produced and return in ascending step.index order.
        out: List[BlackboardEntry] = []
        for pos, entry in enumerate(results_by_sorted_pos):
            if entry is None:  # pragma: no cover
                # If this happens, something went wrong in internal accounting.
                missing_step = steps_sorted[pos]
                raise ToolAgentError(
                    f"Internal error: missing executed entry for step index {missing_step.index}."
                )
            out.append(entry)

        return out

    # ------------------------------------------------------------------ #
    # Template Method (FINAL)
    # ------------------------------------------------------------------ #
    def _invoke(self, *, messages: List[Dict[str, str]]) -> Tuple[List[Dict[str,str]], Any]:
        """
        FINAL template method (do not override in subclasses).

        Subclasses provide planning hooks:
          - _initialize_run_state(...)
          - _prepare_next_steps_batch(...)
        """
        if not messages:
            raise ToolAgentError("ToolAgent._invoke requires a non-empty messages list.")

        # Preserve the original user message for persistence (pre-mutation).
        original_user_msg = dict(messages[-1])

        # Runtime-only counter (non-return calls only).
        non_return_tool_call_count = 0
        tool_calls_limit = self.tool_calls_limit  # None => unlimited

        # Initialize per-run mutable state.
        saved_blackboard = self.blackboard  # persisted board snapshot
        state = self._initialize_run_state(messages=list(messages), saved_blackboard=saved_blackboard)
        if not isinstance(state, dict):
            raise ToolAgentError("_initialize_run_state must return a dict run-state.")
        if "messages" not in state or "blackboard" not in state:
            raise ToolAgentError("run-state must include 'messages' and 'blackboard' keys.")

        while True:
            # 1) Plan next batch (planning only; StepCalls: index, tool_name, args)
            state, batch = self._prepare_next_steps_batch(state=state)
            if not isinstance(state, dict):
                raise ToolAgentError("_prepare_next_steps_batch must return a dict run-state.")
            if not batch:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: planned an empty batch; "
                    "each iteration must invoke at least one tool."
                )

            # 2) Resolve + validate batch (base-owned)
            resolved_batch = self._resolve_steps_batch_args(state=state, batch=batch)

            # Return constraints already enforced in resolve; compute budget inline.
            return_in_batch = sum(1 for s in resolved_batch if s.tool_name == return_tool.full_name)
            non_return_in_batch = len(resolved_batch) - return_in_batch

            if tool_calls_limit is not None and (non_return_tool_call_count + non_return_in_batch) > tool_calls_limit:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool call limit exceeded "
                    f"({non_return_tool_call_count}+{non_return_in_batch} > {tool_calls_limit})."
                )
            non_return_tool_call_count += non_return_in_batch

            # 3) Execute concurrently (fail-fast) -> blackboard entries (base-owned)
            entries = self._execute_steps_batch(resolved_batch=resolved_batch)

            # 4) Commit results to the running blackboard in batch order.
            run_board = state.get("blackboard")
            if not isinstance(run_board, list):
                raise ToolAgentError("run-state['blackboard'] must be a list.")
            run_board.extend(entries)

            # 5) Termination: if return step executed, return its value.
            if return_in_batch == 1:
                # Find the return entry (should exist exactly once).
                for entry in entries:
                    if entry.get("tool") == return_tool.full_name:
                        return_value = entry.get("result")
                        break
                else:  # pragma: no cover
                    raise ToolAgentError("Internal error: return step executed but no return entry found.")

                # Persist context only after final result is established.
                if self.context_enabled:
                    self._blackboard.clear()
                    self._blackboard.extend(run_board)

                # Persist compact history: [latest original user msg] + [assistant(final result)]
                try:
                    assistant_text = repr(return_value)
                except Exception:  # pragma: no cover
                    assistant_text = str(return_value)

                newest_history = [original_user_msg, {"role": "assistant", "content": assistant_text}]
                return newest_history, return_value

    # ------------------------------------------------------------------ #
    # Required planning hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _initialize_run_state(
        self,
        *,
        messages: List[Dict[str, str]],
        saved_blackboard: List[BlackboardEntry],
    ) -> Dict[str, Any]:
        """
        Create a per-run mutable dict run-state.

        Must include:
          - state["messages"]: mutable list[dict[str,str]] copy
          - state["blackboard"]: mutable list[BlackboardEntry] copy
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_next_steps_batch(
        self,
        *,
        state: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[StepCall]]:
        """
        Planning hook: produce the next batch of StepCall objects.

        StepCall is minimal:
          - index
          - tool_name (full Tool.full_name)
          - args (placeholder-view, literal-eval compatible)

        Returns (state, batch). Batch must be non-empty.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Memory + Serialization
    # ------------------------------------------------------------------ #
    def clear_memory(self) -> None:
        super().clear_memory()
        self._blackboard.clear()

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["toolbox"] = [t.to_dict() for t in self._toolbox.values()]
        d["blackboard"] = self._blackboard.copy()
        d["tool_calls_limit"] = self._tool_calls_limit
        return d


def _strip_code_fences(text: str) -> str:
    """Strip a single pair of markdown code fences if present."""
    s = text.strip()
    s = re.sub(r"^\s*```[a-zA-Z0-9]*\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _extract_outermost_list_literal(text: str) -> str:
    """
    Extract the outermost list literal using a simple heuristic:
    - If the cleaned text starts with '[' and ends with ']', return it.
    - Otherwise slice from first '[' to last ']' and return that.
    """
    s = text.strip()
    if s.startswith("[") and s.endswith("]"):
        return s

    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ToolAgentError("Planner output did not contain a bracketed list literal.")
    return s[start : end + 1].strip()


def _collect_step_refs(value: Any) -> List[int]:
    """
    Collect all <<__step__N>> references found anywhere inside a Python-literal structure.
    We scan strings (including embedded placeholders) and recurse through containers.
    """
    out: List[int] = []

    if isinstance(value, str):
        for m in _STEP_TOKEN.finditer(value):
            out.append(int(m.group(1)))
        return out

    if isinstance(value, dict):
        for k, v in value.items():
            out.extend(_collect_step_refs(k))
            out.extend(_collect_step_refs(v))
        return out

    if isinstance(value, (list, tuple, set)):
        for item in value:
            out.extend(_collect_step_refs(item))
        return out

    return out


# ───────────────────────────────────────────────────────────────────────────────
# Plan-first 'PlanActAgent' class
# ───────────────────────────────────────────────────────────────────────────────
class PlanActAgent(ToolAgent):
    """
    Plan-first-then-execute ToolAgent.

    Contract:
    - `_initialize_run_state` performs the one-shot planner LLM call and parses a single
      Python literal: List[List[{"tool": str, "args": dict}]].
    - Each inner list is a batch; steps within a batch must be independent (no intra-batch refs).
    - `_prepare_next_steps_batch` yields the next batch by cursor (O(1)).
    - ToolAgent owns placeholder resolution, budgeting during execution, tool execution,
      blackboard mutation, and termination (return tool).
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        tool_calls_limit: Optional[int] = None,
        pre_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        post_invoke: Optional[AtomicInvokable | Callable[..., Any]] = None,
        history_window: Optional[int] = None,
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

    def _initialize_run_state(
        self,
        *,
        messages: List[Dict[str, str]],
        saved_blackboard: List[BlackboardEntry],
    ) -> Dict[str, Any]:
        if not messages:
            raise ToolAgentError("PlanActAgent requires a non-empty messages list.")

        # Copy inputs (do not mutate caller messages).
        working_messages: List[Dict[str, str]] = [dict(m) for m in messages]
        run_board: List[BlackboardEntry] = list(saved_blackboard)

        # Inject blackboard into the latest message string if context enabled and blackboard present.
        if self.context_enabled and run_board:
            last = dict(working_messages[-1])
            content = last.get("content")
            if not isinstance(content, str):
                raise ToolAgentError(
                    "Last message must include a string 'content' field for blackboard injection."
                )
            injection = "\n\n---\nBLACKBOARD OF PRIOR STEPS\n" + self.blackboard_dumps(run_board, peek=True)
            last["content"] = content + injection
            working_messages[-1] = last

        # One-shot plan call (must emit one Python literal).
        plan_text = self._llm_engine.invoke(working_messages)
        if not isinstance(plan_text, str):
            plan_text = str(plan_text)

        plan_text = _strip_code_fences(plan_text)
        plan_literal = _extract_outermost_list_literal(plan_text)

        try:
            parsed = ast.literal_eval(plan_literal)
        except Exception as exc:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: failed to parse planner output via ast.literal_eval: {exc}"
            ) from exc

        # Validate top-level structure: List[List[dict]]
        if not isinstance(parsed, list) or not parsed:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: plan must be a non-empty list of batches.")

        return_name = return_tool.full_name
        last_batch_idx = len(parsed) - 1

        return_count = 0
        return_pos: Optional[Tuple[int, int]] = None
        non_return_count = 0

        # Validate each batch and each step (shape + tool existence + args dict).
        for bi, batch in enumerate(parsed):
            if not isinstance(batch, list) or not batch:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: batch {bi} must be a non-empty list of step dicts."
                )

            for si, step in enumerate(batch):
                if not isinstance(step, dict):
                    raise ToolAgentError(f"{type(self).__name__}.{self.name}: step ({bi},{si}) must be a dict.")

                if set(step.keys()) != {"tool", "args"}:
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: step ({bi},{si}) must have exactly keys "
                        f"{{'tool','args'}} (got {sorted(step.keys())})."
                    )

                tool_name = step.get("tool")
                if not isinstance(tool_name, str) or not tool_name.strip():
                    raise ToolAgentError(f"{type(self).__name__}.{self.name}: step ({bi},{si}) has invalid 'tool'.")

                if not self.has_tool(tool_name):
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: unknown tool {tool_name!r} in step ({bi},{si})."
                    )

                args = step.get("args")
                if not isinstance(args, dict):
                    raise ToolAgentError(f"{type(self).__name__}.{self.name}: step ({bi},{si}) 'args' must be a dict.")

                if tool_name == return_name:
                    return_count += 1
                    return_pos = (bi, si)
                else:
                    non_return_count += 1

        # Return must occur exactly once.
        if return_count != 1 or return_pos is None:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan must contain exactly one return step; found {return_count}."
            )

        # Return must be only in last batch and last step of that batch.
        ret_bi, ret_si = return_pos
        if ret_bi != last_batch_idx:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: return step must occur only in the last batch "
                f"(found in batch {ret_bi}, last batch is {last_batch_idx})."
            )
        if ret_si != (len(parsed[ret_bi]) - 1):
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: return step must be the last step in the plan.")

        # Early tool-call budget check (return does not count).
        if self.tool_calls_limit is not None and non_return_count > self.tool_calls_limit:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan requires {non_return_count} non-return tool calls "
                f"but tool_calls_limit={self.tool_calls_limit}."
            )

        # Normalize into StepCalls with deterministic global indices (PlanAct owns indices).
        base_len = len(run_board)
        global_counter = 0
        plan_batches: List[List[StepCall]] = []

        for batch in parsed:
            batch_calls: List[StepCall] = []
            for step in batch:
                idx = base_len + global_counter
                global_counter += 1
                batch_calls.append(StepCall(index=idx, tool_name=step["tool"], args=step["args"]))
            plan_batches.append(batch_calls)

        return {
            "messages": working_messages,
            "blackboard": run_board,
            "plan_batches": plan_batches,
            "next_batch_pos": 0,
        }

    def _prepare_next_steps_batch(self, *, state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[StepCall]]:
        plan_batches = state.get("plan_batches")
        if not isinstance(plan_batches, list):
            raise ToolAgentError("run-state missing 'plan_batches' (internal error).")

        pos = state.get("next_batch_pos", 0)
        if not isinstance(pos, int) or pos < 0:
            raise ToolAgentError("run-state['next_batch_pos'] must be an int >= 0 (internal error).")

        if pos >= len(plan_batches):
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: plan exhausted (no remaining batches).")

        batch = plan_batches[pos]
        if not isinstance(batch, list) or not batch:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: planned batch at position {pos} is invalid/empty.")

        state["next_batch_pos"] = pos + 1
        return state, batch

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
