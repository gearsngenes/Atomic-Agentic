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

    Core responsibilities:
      1) toolbox: registry of Tools (keyed by Tool.full_name)
      2) blackboard: list of step entries (persisted across runs iff context_enabled)
      3) tool-call budgeting: per-run limit + concurrency-safe counter
      4) shared step placeholder resolver: <<__step__N>> (0-based)
      5) Template Method: _invoke(messages) delegates to _run(...)
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
        # Validate and store the *template* on self._role_prompt via base Agent init.
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

        # Toolbox + lock (safe registry access if other threads inspect/register tools)
        self._toolbox: OrderedDict[str, Tool] = OrderedDict()
        self._toolbox_lock = threading.RLock()

        # Blackboard + lock (base exposes a safe read/copy + can guard resolver reads)
        self._blackboard: List[BlackboardEntry] = []
        self._blackboard_lock = threading.RLock()

        # Tool-call budgeting (per-run) — MUST be concurrency-safe
        self._tool_calls_lock = threading.Lock()
        self._tool_calls_made: int = 0
        self._tool_calls_limit: Optional[int] = None
        self.tool_calls_limit = tool_calls_limit  # validated

        # Always include the canonical return tool (avoid collisions by skipping).
        self.register(return_tool, name_collision_mode="skip")

    # ------------------------------------------------------------------ #
    # Role prompt templating: template is stored in _role_prompt; getter formats
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
        """Read-only view (shallow copy) of the persisted blackboard."""
        with self._blackboard_lock:
            return list(self._blackboard)

    # ------------------------------------------------------------------ #
    # Toolbox helpers
    # ------------------------------------------------------------------ #
    def list_tools(self) -> OrderedDict[str, Tool]:
        with self._toolbox_lock:
            return OrderedDict(self._toolbox)

    def has_tool(self, tool_full_name: str) -> bool:
        with self._toolbox_lock:
            return tool_full_name in self._toolbox

    def get_tool(self, tool_full_name: str) -> Tool:
        with self._toolbox_lock:
            tool = self._toolbox.get(tool_full_name)
        if tool is None:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool {tool_full_name!r}.")
        return tool

    def remove_tool(self, tool_full_name: str) -> bool:
        with self._toolbox_lock:
            return self._toolbox.pop(tool_full_name, None) is not None

    def clear_tools(self) -> None:
        with self._toolbox_lock:
            self._toolbox.clear()

    # ------------------------------------------------------------------ #
    # Prompt helpers
    # ------------------------------------------------------------------ #
    def actions_context(self) -> str:
        """
        String representation of all tools in the toolbox for prompt injection.
        """
        with self._toolbox_lock:
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

        args = "raw_args" if raw_results else "resolved_args"

        view = [{"tool": s.get("tool"), "args": s.get(args, {})} for s in obj]

        try:
            return json.dumps(view, indent = indent, default = str)
        except Exception:
            return str(view)

    # ------------------------------------------------------------------ #
    # Tool Registration
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
                headers=headers,  # MCP URL contract: key presence required when component is str
                include=include,
                exclude=exclude,
            )
        except ToolDefinitionError:
            raise
        except Exception as exc:  # pragma: no cover
            raise ToolRegistrationError(f"toolify failed for {component!r}: {exc}") from exc

        registered: List[str] = []
        with self._toolbox_lock:
            for tool in tools:
                key = tool.full_name
                if key in self._toolbox:
                    if name_collision_mode == "raise":
                        raise ToolRegistrationError(
                            f"{type(self).__name__}.{self.name}: tool already registered: {key}"
                        )
                    if name_collision_mode == "skip":
                        continue
                self._toolbox[key] = tool  # replace or new
                registered.append(key)

        return registered

    def batch_register(
        self,
        tools: List[Callable|Tool|Agent],
        name_collision_mode: str = "raise",
    ) -> List[str]:
        registered: List[str] = []
        for obj in tools:
            if type(obj) in (Tool, Agent):
                name, description = obj.name, obj.description
            elif callable(obj):
                name, description = obj.__name__, obj.__doc__
            else:
                raise ToolRegistrationError(
                    f"batch_register expected Tool, Agent, or callable; got {type(obj).__name__!r}."
                )
            registered.extend(
                self.register(
                    obj,
                    name=name,
                    description=description,
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
        """
        Atomically check the limit and increment tool_calls_made.

        Reserve-first semantics are intentional:
          - Correct under concurrency (prevents overshooting limit).
          - Counts attempts (failures consume budget) to avoid runaway retries.
        """
        with self._tool_calls_lock:
            limit = self._tool_calls_limit
            if limit is not None and self._tool_calls_made >= limit:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: tool usage limit exceeded "
                    f"({self._tool_calls_made}/{limit})."
                )
            self._tool_calls_made += 1

    def _call_tool(self, tool_full_name: str, inputs: Mapping[str, Any]) -> Any:
        """
        Execute a tool by full name with tool-call budget enforcement.

        This method:
          1) Reserves a budget slot (atomic check+increment)
          2) Resolves step placeholders in inputs using the current self._blackboard
          3) Invokes the tool
        """
        if not isinstance(tool_full_name, str) or not tool_full_name.strip():
            raise ToolAgentError("_call_tool requires a non-empty tool_full_name string.")
        if not isinstance(inputs, Mapping):
            raise ToolAgentError("_call_tool requires inputs to be a Mapping[str, Any].")

        tool = self.get_tool(tool_full_name)

        # Enforce limit correctly under concurrency.
        self._reserve_tool_call_or_raise()

        resolved_inputs = self._resolve_step_refs(dict(inputs))

        try:
            return tool.invoke(resolved_inputs)
        except ToolInvocationError:
            raise
        except Exception as exc:  # pragma: no cover
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: tool call failed for {tool_full_name!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Step placeholder resolution (uses ONLY self._blackboard)
    # ------------------------------------------------------------------ #
    def _resolve_step_refs(self, obj: Any) -> Any:
        """
        Resolve step placeholders against the current self._blackboard.

        Canonical:
          - <<__step__N>> where N is 0-based index
        Optional legacy (if enabled):
          - {stepN}
        """
        if isinstance(obj, str):
            # Canonical full match => typed result
            m = _STEP_TOKEN.fullmatch(obj)
            if m:
                return self._step_result_by_index(int(m.group(1)))

            # Canonical inline substitution => str(result)
            def repl(match: re.Match[str]) -> str:
                result = self._step_result_by_index(int(match.group(1)))
                try: stringified = repr(result)
                except: stringified = str(result)
                return stringified

            text = _STEP_TOKEN.sub(repl, obj)

            return text

        if isinstance(obj, list):
            return [self._resolve_step_refs(v) for v in obj]

        if isinstance(obj, dict):
            return {k: self._resolve_step_refs(v) for k, v in obj.items()}

        return obj

    def _step_result_by_index(self, idx: int) -> Any:
        if not isinstance(idx, int):
            raise ToolAgentError(f"Step reference must be an int; got {type(idx).__name__!r}.")
        if idx < 0:
            raise ToolAgentError(f"Step reference must be >= 0; got {idx}.")

        with self._blackboard_lock:
            if idx >= len(self._blackboard):
                raise ToolAgentError(
                    f"Step reference {idx} out of range (blackboard length={len(self._blackboard)})."
                )
            step = self._blackboard[idx]

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
        FINAL template method (do not override in subclasses).

        Subclasses implement:
          _run(messages=..., blackboard=...) -> (loaded_blackboard, new_step_strings, return_value)
        """
        user_msg = {"role":"user", "content": messages[-1]["content"]}
        self._reset_tool_calls_made()

        # Stateless runs must not leak prior runs into this run.
        if not self.context_enabled:
            with self._blackboard_lock:
                self._blackboard.clear()

        new_blackboard_steps, return_value = self._run(
            messages=messages,
            blackboard=self._blackboard,
        )

        if not isinstance(new_blackboard_steps, list):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}._run must return a list of new steps for the blackboard; "
                f"got {type(new_blackboard_steps).__name__!r}."
            )

        if self.context_enabled:
            with self._blackboard_lock:
                self._blackboard.extend(new_blackboard_steps)
        else:
            with self._blackboard_lock:
                self._blackboard.clear()
        self._newest_history.append(user_msg)
        self._newest_history.append({"role" : "assistant", "content" : self.blackboard_view()})
        return return_value

    @abstractmethod
    def _run(
        self,
        *,
        messages: List[Dict[str, str]],
        blackboard: List[BlackboardEntry],
    ) -> tuple[List[BlackboardEntry], Any]:
        """
        Subclass hook.

        Must return:
          1) new_blackboard_steps: new steps generated steps that were run for a given task
          2) return_value: final output value for this run

        Per your architecture: _run() should populate self._newest_history.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Memory lifecycle
    # ------------------------------------------------------------------ #
    def clear_memory(self) -> None:
        super().clear_memory()
        with self._blackboard_lock:
            self._blackboard.clear()
        self._reset_tool_calls_made()
