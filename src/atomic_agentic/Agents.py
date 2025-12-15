"""
Agents
======

(Existing module docstring retained.)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Callable, Union, TypedDict, Sequence
import logging
from collections import OrderedDict
from abc import abstractmethod, ABC
import re
import json
import string
import threading

# Local imports (adjust the module paths if your project structure differs)
from .LLMEngines import LLMEngine
from .Tools import Tool
from .Exceptions import (
    ToolAgentError,
    ToolDefinitionError,
    ToolInvocationError,
    ToolRegistrationError,
)
from .Primitives import Agent
from .Toolify import toolify

__all__ = ["Agent", "ToolAgent", "BlackboardEntry", "return_tool"]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Canonical final-return tool (should appear once as the last step of a plan)
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

# Canonical step placeholder: <<__step__N>> where N is a 0-based step index.
_STEP_TOKEN: re.Pattern[str] = re.compile(r"<<__step__(\d+)>>")

class BlackboardEntry(TypedDict):
    """
    Minimal required blackboard entry for ToolAgent.

    NOTE: This is intentionally minimal. You can expand later (e.g., add step id,
    timestamps, error info, tool output schema, etc.). ToolAgent only requires
    `completed` and `result` to support step-resolution.
    """
    # tool full name (Tool.full_name)
    tool: str
    # the step arguments retrieved directly from json.loads() or model output
    raw_args: Dict[str, Any]
    # the step arguments after resolving <<__step__N>> placeholders
    resolved_args: Dict[str, Any]
    # whether the step is completed
    completed: bool
    # the completed step result (any python object)
    result: Any


class ToolAgent(Agent, ABC):
    """
    Abstract base class for tool-using agents (Planner/Orchestrator/etc.).

    Key contracts (per your refactor decisions)
    -------------------------------------------
    - Thread-safety:
        * Concurrent calls to `invoke()` are serialized by the base `Agent`'s
          per-instance invoke lock.
        * ToolAgent does NOT mutate `self._blackboard` during `_run()`.
        * If a subclass parallelizes tool calls inside a single run, the tool-call
          budget counter remains atomic via `_tool_calls_lock`.

    - Blackboard lifecycle:
        * `self._blackboard` is committed ONLY by this base class in `_invoke()`.
        * `_run()` must treat the blackboard as read-only reference data.
        * `_run()` returns a list of NEW executed steps which are appended (only
          if `context_enabled=True`).

    - Placeholders:
        * Canonical step placeholder format is `<<__step__N>>` (0-based index).
        * Placeholder resolution defaults to the frozen snapshot of the blackboard
          taken at the start of `_invoke()` (so resolution is stable during `_run()`).
        * Subclasses that need “existing + new-steps-so-far” resolution should pass
          an explicit `board=` to `_call_tool(...)` / `_resolve_step_refs(...)`.
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
        pre_invoke: Optional[Tool | Callable[..., Any]] = None,
        post_invoke: Optional[Tool | Callable[..., Any]] = None,
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

        self._toolbox: OrderedDict[str, Tool] = OrderedDict()
        self._blackboard: List[BlackboardEntry] = []

        # Frozen snapshot used for stable placeholder resolution during _run().
        self._blackboard_snapshot: tuple[BlackboardEntry, ...] = tuple()

        # Tool-call budgeting (keep lock to support potential intra-run parallelism)
        self._tool_calls_lock = threading.Lock()
        self._tool_calls_made: int = 0
        self._tool_calls_limit: Optional[int] = None
        self.tool_calls_limit = tool_calls_limit

        # Always include the canonical return tool (avoid collisions by skipping).
        self.register(return_tool, name_collision_mode="skip")

    # ------------------------------------------------------------------ #
    # Role prompt templating
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_role_prompt_template(template: Any) -> str:
        """
        ToolAgent requires a non-empty role prompt *template* containing:
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

    @property
    def role_prompt(self) -> str:
        """
        Formatted role prompt for this run.

        IMPORTANT: self._role_prompt stores the *template* string.
        """
        template = self._role_prompt or ""
        if not isinstance(template, str) or not template.strip():
            raise ToolAgentError("ToolAgent has an invalid role_prompt template stored on _role_prompt.")

        tool_calls_limit_text = "unlimited" if self._tool_calls_limit is None else str(self._tool_calls_limit)
        try:
            return template.format(
                TOOLS=self.actions_context(),
                TOOL_CALLS_LIMIT=tool_calls_limit_text,
            )
        except Exception as exc:  # pragma: no cover
            raise ToolAgentError(f"Failed to format ToolAgent role_prompt template: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def tool_calls_limit(self) -> Optional[int]:
        """Max allowed tool calls per invoke() run. None means unlimited."""
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
        # Consistent snapshot for external reads (base Agent serializes invoke).
        with self._invoke_lock:
            return list(self._blackboard)

    # ------------------------------------------------------------------ #
    # Toolbox helpers (public reads are lock-protected; internal calls use unlocked helper)
    # ------------------------------------------------------------------ #
    def list_tools(self) -> OrderedDict[str, Tool]:
        with self._invoke_lock:
            return OrderedDict(self._toolbox)

    def has_tool(self, tool_full_name: str) -> bool:
        with self._invoke_lock:
            return tool_full_name in self._toolbox

    def get_tool(self, tool_full_name: str) -> Tool:
        with self._invoke_lock:
            tool = self._toolbox.get(tool_full_name)
        if tool is None:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool {tool_full_name!r}.")
        return tool

    def _get_tool_unlocked(self, tool_full_name: str) -> Tool:
        # Used by _call_tool to avoid deadlocks if subclasses parallelize tool calls.
        tool = self._toolbox.get(tool_full_name)
        if tool is None:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool {tool_full_name!r}.")
        return tool

    def remove_tool(self, tool_full_name: str) -> bool:
        with self._invoke_lock:
            return self._toolbox.pop(tool_full_name, None) is not None

    def clear_tools(self) -> None:
        with self._invoke_lock:
            self._toolbox.clear()

    # ------------------------------------------------------------------ #
    # Prompt helpers
    # ------------------------------------------------------------------ #
    def actions_context(self) -> str:
        with self._invoke_lock:
            tools = list(self._toolbox.values())
        return "\n".join(f"-- {t}" for t in tools)

    def blackboard_view(
        self,
        obj: Optional[List[BlackboardEntry]] = None,
        *,
        raw_results: bool = False,
        indent: int = 2,
    ) -> str:
        """
        Convert the blackboard into a human-readable string as a list of dicts,
        each containing exactly:
          - "tool": tool full name
          - "args": resolved args (placeholders replaced)

        This is intended for prompt injection / debugging, not strict serialization.
        """
        if obj is None:
            obj = self.blackboard

        args_key = "raw_args" if raw_results else "resolved_args"
        view = [{"tool": s.get("tool"), "args": s.get(args_key, {})} for s in obj]

        try:
            return json.dumps(view, indent=indent, default=str)
        except Exception:
            return str(view)

    # ------------------------------------------------------------------ #
    # Tool Registration (mutations are serialized against invoke via base invoke lock)
    # ------------------------------------------------------------------ #
    def register(
        self,
        component: Callable[..., Any] | Tool | Agent | str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        namespace: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        name_collision_mode: str = "raise",  # raise|skip|replace
    ) -> List[str]:
        if name_collision_mode not in ("raise", "skip", "replace"):
            raise ToolRegistrationError("name_collision_mode must be one of: 'raise', 'skip', 'replace'.")

        # Local import is safe even if module wiring changes again.
        try:
            tools = toolify(
                component,
                name=name,
                description=description,
                namespace=namespace,
                headers=headers,
                include=include,
                exclude=exclude,
            )
        except ToolDefinitionError:
            raise
        except Exception as exc:  # pragma: no cover
            raise ToolRegistrationError(f"toolify failed for {component!r}: {exc}") from exc

        registered: List[str] = []
        with self._invoke_lock:
            for tool in tools:
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

    def batch_register(
        self,
        tools: Sequence[Callable[..., Any] | Tool | Agent],
        name_collision_mode: str = "raise",
    ) -> List[str]:
        registered: List[str] = []
        for obj in tools:
            if isinstance(obj, (Tool, Agent)):
                nm, desc = obj.name, obj.description
            elif callable(obj):
                nm, desc = obj.__name__, obj.__doc__
            else:
                raise ToolRegistrationError(
                    f"batch_register expected Tool, Agent, or callable; got {type(obj).__name__!r}."
                )

            registered.extend(
                self.register(
                    obj,
                    name=nm,
                    description=desc,
                    namespace=self.name,
                    name_collision_mode=name_collision_mode,
                )
            )
        return registered

    # ------------------------------------------------------------------ #
    # Tool call accounting + gateway
    # ------------------------------------------------------------------ #
    def _reset_tool_calls_made(self) -> None:
        with self._tool_calls_lock:
            self._tool_calls_made = 0

    def _reserve_tool_call_or_raise(self) -> None:
        with self._tool_calls_lock:
            limit = self._tool_calls_limit
            if limit is not None and self._tool_calls_made >= limit:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool usage limit exceeded "
                    f"({self._tool_calls_made}/{limit})."
                )
            self._tool_calls_made += 1

    def _call_tool(
        self,
        tool_full_name: str,
        inputs: Mapping[str, Any],
        *,
        board: Optional[Sequence[BlackboardEntry]] = None,
    ) -> Any:
        if not isinstance(tool_full_name, str) or not tool_full_name.strip():
            raise ToolAgentError("_call_tool requires a non-empty tool_full_name string.")
        if not isinstance(inputs, Mapping):
            raise ToolAgentError("_call_tool requires inputs to be a Mapping[str, Any].")

        tool = self._get_tool_unlocked(tool_full_name)
        self._reserve_tool_call_or_raise()

        resolved_inputs = self._resolve_step_refs(dict(inputs), board=board)

        try:
            return tool.invoke(resolved_inputs)
        except ToolInvocationError:
            raise
        except Exception as exc:  # pragma: no cover
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: tool call failed for {tool_full_name!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Step placeholder resolution (defaults to frozen snapshot)
    # ------------------------------------------------------------------ #
    def _resolve_step_refs(self, obj: Any, *, board: Optional[Sequence[BlackboardEntry]] = None) -> Any:
        if board is None:
            board = self._blackboard_snapshot

        if isinstance(obj, str):
            # Canonical full match => typed result
            m = _STEP_TOKEN.fullmatch(obj)
            if m:
                return self._step_result_by_index(int(m.group(1)), board=board)

            def repl(match: re.Match[str]) -> str:
                result = self._step_result_by_index(int(match.group(1)), board=board)
                try: return repr(result)
                except Exception: return str(result)

            return _STEP_TOKEN.sub(repl, obj)

        if isinstance(obj, list):
            return [self._resolve_step_refs(v, board=board) for v in obj]

        if isinstance(obj, dict):
            return {k: self._resolve_step_refs(v, board=board) for k, v in obj.items()}

        return obj

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

    # ------------------------------------------------------------------ #
    # Template method: subclasses implement _run(...)
    # ------------------------------------------------------------------ #
    def _invoke(self, *, messages: List[Dict[str, str]]) -> Any:
        """
        FINAL template method (do not override).

        Subclasses implement:
          _run(messages=...) -> (new_blackboard_steps, return_value)

        `_run()` must not mutate `self._blackboard`. It should create/execute NEW
        steps and return them as a list.
        """
        # base Agent.invoke serializes calls; we're inside that lock here.
        self._reset_tool_calls_made()

        if not self.context_enabled:
            self._blackboard.clear()

        # Freeze blackboard view for stable placeholder resolution during _run().
        self._blackboard_snapshot = tuple(self._blackboard)

        new_blackboard_steps, return_value = self._run(messages=messages)

        if not isinstance(new_blackboard_steps, list):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}._run must return (list[BlackboardEntry], Any); "
                f"got {type(new_blackboard_steps).__name__!r} for first element."
            )

        # Commit blackboard AFTER _run (never during).
        if self.context_enabled:
            self._blackboard.extend(new_blackboard_steps)
            assistant_bb_text = self.blackboard_view()
        else:
            assistant_bb_text = self.blackboard_view(new_blackboard_steps)
            self._blackboard.clear()

        user_msg = {"role": "user", "content": messages[-1]["content"]}
        self._newest_history.append(user_msg)
        self._newest_history.append({"role": "assistant", "content": assistant_bb_text})

        return return_value

    @abstractmethod
    def _run(
        self,
        *,
        messages: List[Dict[str, str]],
    ) -> tuple[List[BlackboardEntry], Any]:
        """
        Subclass hook.

        Must return:
          1) new_blackboard_steps: NEW steps that were executed for this run
          2) return_value: the final output for this run

        IMPORTANT:
        - Must NOT mutate `self._blackboard`.
        - If you need placeholder resolution that can reference “existing steps
          + new-steps-so-far”, build a local board like:
              local_board = list(self._blackboard_snapshot) + new_steps_so_far
          and call:
              self._call_tool(..., board=local_board)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Memory lifecycle
    # ------------------------------------------------------------------ #
    def clear_memory(self) -> None:
        with self._invoke_lock:
            super().clear_memory()
            self._blackboard.clear()
            self._blackboard_snapshot = tuple()
            self._reset_tool_calls_made()
