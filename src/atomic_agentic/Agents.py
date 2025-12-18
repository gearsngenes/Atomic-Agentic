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
from collections import OrderedDict
import json
import logging
import re
import string
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from python_a2a import (
    A2AServer, run_server, agent,
    Message, MessageRole, FunctionResponseContent,
    TextContent
)

from .Exceptions import (
    ToolAgentError,
    ToolDefinitionError,
    ToolInvocationError,
    ToolRegistrationError,
)
from .LLMEngines import LLMEngine
from .Primitives import Agent
from .Toolify import toolify
from .Tools import Tool
from .Prompts import PLANNER_PROMPT, ORCHESTRATOR_PROMPT


__all__ = ["Agent", "ToolAgent", "BlackboardEntry", "return_tool", "A2AgentHost"]

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
    """Minimal required blackboard entry for ToolAgent."""

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

# ───────────────────────────────────────────────────────────────────────────────
# Base 'ToolAgent' class
# ───────────────────────────────────────────────────────────────────────────────
class ToolAgent(Agent, ABC):
    """
    Abstract base class for tool-using agents (Planner/Orchestrator/etc.).

    Key contracts (per your refactor decisions)
    -------------------------------------------
    - Thread-safety:
        * External concurrent calls to `invoke()` are serialized by ToolAgent's
          per-instance `_invoke_lock`.
        * ToolAgent does NOT mutate `self._blackboard` during `_run()`.
        * If a subclass parallelizes tool calls inside a single run, the tool-call
          budget counter remains atomic via `_tool_calls_lock`.

    - Blackboard lifecycle:
        * `self._blackboard` is committed ONLY by this base class in `_invoke()`.
        * `_run()` must treat the blackboard as read-only reference data.
        * `_run()` returns a list of NEW executed steps which are appended only
          when `context_enabled=True`.

    - Placeholders:
        * Canonical step placeholder format is `<<__step__N>>` (0-based index).
        * Placeholder resolution defaults to the persisted blackboard (prior steps).
        * Subclasses needing “existing + new-steps-so-far” resolution should pass
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

        IMPORTANT: ``self._role_prompt`` stores the *template* string, not the final
        formatted prompt.
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
        with self._invoke_lock:
            return list(self._blackboard)

    # ------------------------------------------------------------------ #
    # Toolbox helpers
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
        """Get a tool without taking `_invoke_lock` (for intra-run parallel tool calls)."""
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
        """String representation of all tools in the toolbox for prompt injection."""
        with self._invoke_lock:
            tools = list(self._toolbox.values())
        return "\n".join(f"-- {t}" for t in tools)

    def blackboard_dumps(
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
        view = [{"tool": step.get("tool"), "args": step.get(args_key, {})} for step in obj]

        try:
            return json.dumps(view, indent=indent, default=str)
        except Exception:  # pragma: no cover
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
        """
        Register a component into the toolbox via :func:`~atomic_agentic.Toolify.toolify`.

        Notes
        -----
        - This method is serialized against :meth:`invoke` via `_invoke_lock`.
        - MCP bulk registration will return multiple full_names.
        """
        if name_collision_mode not in ("raise", "skip", "replace"):
            raise ToolRegistrationError("name_collision_mode must be one of: 'raise', 'skip', 'replace'.")

        try:
            tools = toolify(
                component,
                name=name,
                description=description,
                namespace=namespace or self.name,
                headers=headers,  # MCP URL contract: key presence required when component is str
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
                # replace or new
                self._toolbox[key] = tool
                registered.append(key)

        return registered

    def batch_register(
        self,
        tools: Sequence[Callable[..., Any] | Tool | Agent],
        *,
        name_collision_mode: str = "raise",
    ) -> List[str]:
        """Register a batch of components."""
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
        """Atomically check the limit and increment tool_calls_made."""
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
        """
        Execute a tool by full name with tool-call budget enforcement.

        Parameters
        ----------
        tool_full_name:
            The Tool.full_name to execute.
        inputs:
            Input mapping for the tool.
        board:
            Optional board to resolve placeholders against. If omitted, resolves
            against the persisted `self._blackboard` (prior steps).
        """
        if not isinstance(tool_full_name, str) or not tool_full_name.strip():
            raise ToolAgentError("_call_tool requires a non-empty tool_full_name string.")
        if not isinstance(inputs, Mapping):
            raise ToolAgentError("_call_tool requires inputs to be a Mapping[str, Any].")

        tool = self._get_tool_unlocked(tool_full_name)

        # Enforce limit correctly under concurrency.
        if tool.full_name != return_tool.full_name:
            self._reserve_tool_call_or_raise()
        logger.info(f"[{self.name}.TOOL]: {tool_full_name!r}, args: {json.dumps(inputs)}.")
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
    # Step placeholder resolution
    # ------------------------------------------------------------------ #
    def _resolve_step_refs(self, obj: Any, *, board: Optional[Sequence[BlackboardEntry]] = None) -> Any:
        """Resolve ``<<__step__N>>`` placeholders recursively."""
        if board is None:
            # Local snapshot of the persisted board for stable resolution.
            with self._invoke_lock:
                board = list(self._blackboard)

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
        FINAL template method (do not override in subclasses).

        Subclasses implement:
            _run(messages=...) -> (new_blackboard_steps, return_value)

        `_run()` must not mutate `self._blackboard`.
        """
        # Reset per-invoke tool call counter.
        self._reset_tool_calls_made()
        # Save the latest message for newest history
        prompt = messages[-1]["content"]
        user_msg = {"role": "user", "content": prompt}
        # if context_enabled, inject blackboard + indexing info into last user msg
        base_len = len(self.blackboard)
        run_messages = list(messages)
        if self.context_enabled:
            bb_view = self.blackboard_dumps(obj=None, raw_results=True, indent=2)
            last = dict(run_messages[-1])
            hi = base_len - 1
            last["content"] = (
                f"{last.get('content', '')}\n\n"
                f"PREVIOUS STEPS (global indices 0..{hi if hi >= 0 else -1}):\n"
                f"{bb_view}\n\n"
                f"INDEXING:\n"
                f"- The view above corresponds to indices 0..{hi if hi >= 0 else -1}.\n"
                f"- New steps you emit MUST continue from index {base_len}.\n"
                f"- Use placeholders like \"<<__step__N>>\" with these GLOBAL indices.\n"
            )
            run_messages[-1] = last
        # Call _run() to get newly executed steps + return value
        new_blackboard_steps, return_value = self._run(messages=run_messages)

        if not isinstance(new_blackboard_steps, list):
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}._run must return (list[BlackboardEntry], Any); "
                f"got {type(new_blackboard_steps).__name__!r} for first element."
            )

        # Commit blackboard AFTER _run (never during).
        if self.context_enabled:
            self._blackboard.extend(new_blackboard_steps)
        else:
            pass
        try: str_res = repr(return_value)
        except: str_res = str(return_value)
        # Update newest history
        assistant_text = f"Generated steps:\n{self.blackboard_dumps(new_blackboard_steps, raw_results=True)}\nResult produced:\n{str_res}"
        self._newest_history.append(user_msg)
        self._newest_history.append({"role": "assistant", "content": assistant_text})
        # Return final value
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
          1) new_blackboard_steps: NEW executed steps for this run
          2) return_value: the final output value for this run

        IMPORTANT:
        - Must NOT mutate `self._blackboard`.
        - If you need placeholder resolution that can reference “existing steps
          + new-steps-so-far”, build a local board like:

              local_board = list(self._blackboard) + new_steps_so_far

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
            self._reset_tool_calls_made()
    
    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> OrderedDict[str, Any]:
        d = super().to_dict()
        with self._invoke_lock:
            d["toolbox"] = [t.to_dict() for t in self._toolbox.values()]
            d["blackboard"] = self._blackboard.copy()
            d["tool_calls_limit"] = self._tool_calls_limit
            d["tool_calls_made"] = self._tool_calls_made
        return d

# ───────────────────────────────────────────────────────────────────────────────
# Plan-first 'PlanActAgent' class
# ───────────────────────────────────────────────────────────────────────────────
class PlanActAgent(ToolAgent):
    """
    Plan-first-then-execute (ReWOO-style) ToolAgent.

    - Exactly ONE LLM call produces a JSON plan: List[{"tool": str, "args": dict}, ...]
    - Executes steps sequentially (default) or concurrently by dependency "waves"
      when run_concurrent=True.
    - Uses ToolAgent's existing placeholder resolution and tool gateway.
    - Fail-fast: any parse/validation error or tool failure raises immediately.
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        tool_calls_limit: Optional[int] = None,
        pre_invoke: Optional[Tool | Callable[..., Any]] = None,
        post_invoke: Optional[Tool | Callable[..., Any]] = None,
        history_window: Optional[int] = None,
        run_concurrent: bool = False,
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
        self._run_concurrent: bool = bool(run_concurrent)

    @property
    def run_concurrent(self) -> bool:
        return self._run_concurrent

    @run_concurrent.setter
    def run_concurrent(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ToolAgentError("run_concurrent must be a bool.")
        self._run_concurrent = value

    def _run(
        self,
        *,
        messages: List[Dict[str, str]],
    ) -> tuple[List[BlackboardEntry], Any]:
        # Persisted board snapshot (read-only for this run)
        persisted: List[BlackboardEntry] = self.blackboard
        base_len = len(persisted)

        # Single LLM call to produce plan string
        plan_text = self._llm_engine.invoke(messages).strip()
        plan_text = re.sub(r"^\s*```[a-zA-Z0-9]*\s*|\s*```\s*$", "", plan_text).strip()

        # Parse and load steps; ensure return tool is last
        try:
            parsed = json.loads(plan_text)
        except Exception as exc:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: failed to parse plan JSON: {exc}"
            ) from exc

        if not isinstance(parsed, list):
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: plan must be a JSON array (list).")

        plan: List[Dict[str, Any]] = []
        for i, step in enumerate(parsed):
            if not isinstance(step, dict):
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: plan step {i} must be an object/dict.")
            tool_name = step.get("tool")
            args = step.get("args")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: step {i} missing non-empty 'tool' string.")
            if not isinstance(args, dict):
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: step {i} missing 'args' object/dict.")
            if not self.has_tool(tool_name):
                raise ToolAgentError(f"{type(self).__name__}.{self.name}: unknown tool {tool_name!r} in step {i}.")
            plan.append({"tool": tool_name, "args": args})

        if not plan:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: plan cannot be empty.")

        return_name = return_tool.full_name
        return_positions = [i for i, s in enumerate(plan) if s["tool"] == return_name]
        if len(return_positions) > 1:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: plan may contain return tool only once.")
        if return_positions and return_positions[0] != len(plan) - 1:
            raise ToolAgentError(f"{type(self).__name__}.{self.name}: return tool must be the final step.")
        if not return_positions:
            # Auto-append return: return the last step's result using GLOBAL index
            last_global_idx = base_len + (len(plan) - 1)
            plan.append({"tool": return_name, "args": {"val": f"<<__step__{last_global_idx}>>"}})

        # 3.d) If non-return steps exceed tool call limit, raise (return does not count)
        non_return_count = sum(1 for s in plan if s["tool"] != return_name)
        if self.tool_calls_limit is not None and non_return_count > self.tool_calls_limit:
            raise ToolAgentError(
                f"{type(self).__name__}.{self.name}: plan requires {non_return_count} tool calls "
                f"but tool_calls_limit={self.tool_calls_limit}."
            )

        # 3.e) Pre-allocate NEW (uncompleted) blackboard entries to update in-place
        new_steps: List[BlackboardEntry] = []
        for step in plan:
            new_steps.append(
                {
                    "tool": step["tool"],
                    "raw_args": dict(step["args"]),
                    "resolved_args": {},
                    "completed": False,
                    "result": None,
                }
            )

        # Global board used for dependency checks / resolution (persisted + new)
        board: List[BlackboardEntry] = persisted + new_steps
        uncompleted: set[int] = set(range(len(new_steps)))

        # 3.e/f) Execute steps in waves
        while uncompleted:
            batch = self._collect_step_dependencies(
                uncompleted=uncompleted,
                board=board,
                base_len=base_len,
            )
            if not batch:
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: no runnable steps remain (deadlock/cycle/forward-ref)."
                )

            # Snapshot board for this batch (read-only for the duration of batch execution)
            board_snapshot: List[BlackboardEntry] = list(persisted) + list(new_steps)

            results = self._run_steps_batch(
                step_idxs=batch,
                board_snapshot=board_snapshot,
                base_len=base_len,
            )

            for local_idx, result, resolved_args in results:
                entry = new_steps[local_idx]
                entry["resolved_args"] = resolved_args
                entry["result"] = result
                entry["completed"] = True
                uncompleted.discard(local_idx)

        # 3.f) Return final result + new steps (ToolAgent expects (new_steps, return_value))
        return_value = new_steps[-1]["result"]
        return new_steps, return_value

    def _collect_step_dependencies(
        self,
        *,
        uncompleted: set[int],
        board: Sequence[BlackboardEntry],
        base_len: int,
    ) -> List[int]:
        """
        Return the next runnable step indices (LOCAL indices into the NEW plan steps).

        - If run_concurrent is False: returns exactly ONE step, in plan order.
        - If run_concurrent is True: returns ALL ready steps (dependency wave).
        """
        if not uncompleted:
            return []

        # Sequential mode: always next step in order of appearance
        if not self.run_concurrent:
            return [min(uncompleted)]

        def extract_refs(obj: Any) -> set[int]:
            found: set[int] = set()

            def walk(x: Any) -> None:
                if isinstance(x, str):
                    for m in _STEP_TOKEN.finditer(x):
                        found.add(int(m.group(1)))
                elif isinstance(x, list):
                    for v in x:
                        walk(v)
                elif isinstance(x, dict):
                    for v in x.values():
                        walk(v)

            walk(obj)
            return found

        ready: List[int] = []
        for local_idx in sorted(uncompleted):
            global_idx = base_len + local_idx
            raw_args = board[global_idx].get("raw_args", {})

            refs = extract_refs(raw_args)
            runnable = True
            for ref_idx in refs:
                if ref_idx < 0 or ref_idx >= len(board):
                    raise ToolAgentError(
                        f"{type(self).__name__}.{self.name}: invalid placeholder <<__step__{ref_idx}>> "
                        f"(board size={len(board)})."
                    )
                dep = board[ref_idx]
                if not bool(dep.get("completed", False)):
                    # If referencing persisted steps, that's an invariant violation (they should be completed)
                    if ref_idx < base_len:
                        raise ToolAgentError(
                            f"{type(self).__name__}.{self.name}: referenced prior step {ref_idx} is not completed."
                        )
                    runnable = False
                    break

            if runnable:
                ready.append(local_idx)

        return ready

    def _run_steps_batch(
        self,
        *,
        step_idxs: List[int],
        board_snapshot: Sequence[BlackboardEntry],
        base_len: int,
    ) -> List[tuple[int, Any, Dict[str, Any]]]:
        """
        Execute a batch of LOCAL step indices.

        Returns a list of tuples:
            (local_step_idx, result, resolved_args)
        """
        if not step_idxs:
            return []

        def run_one(local_idx: int) -> tuple[int, Any, Dict[str, Any]]:
            entry = board_snapshot[base_len + local_idx]

            tool_name = entry.get("tool")
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: invalid tool name at local step {local_idx}: {tool_name!r}"
                )

            raw_args_obj = entry.get("raw_args", {})
            if not isinstance(raw_args_obj, dict):
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: raw_args must be a dict at local step {local_idx}; "
                    f"got {type(raw_args_obj).__name__!r}"
                )

            # Execute via ToolAgent gateway:
            # - reserves tool-call budget
            # - resolves placeholders against `board_snapshot`
            # - invokes the tool
            result = self._call_tool(tool_name, raw_args_obj, board=board_snapshot)

            # Only after a successful tool call do we compute resolved_args for bookkeeping.
            resolved_args_obj = self._resolve_step_refs(raw_args_obj, board=board_snapshot)
            if not isinstance(resolved_args_obj, dict):
                # Defensive: raw_args is a dict, so resolved should remain a dict; if not, fail loudly.
                raise ToolAgentError(
                    f"{type(self).__name__}.{self.name}: resolved_args must be a dict at local step {local_idx}; "
                    f"got {type(resolved_args_obj).__name__!r}"
                )

            return local_idx, result, dict(resolved_args_obj)

        # Sequential mode: caller passes one idx per wave.
        if (not self.run_concurrent) or len(step_idxs) == 1:
            return [run_one(step_idxs[0])]

        # Concurrent: execute dependency wave in parallel.
        results: List[tuple[int, Any, Dict[str, Any]]] = []
        executor = ThreadPoolExecutor()
        try:
            futures = [executor.submit(run_one, i) for i in step_idxs]
            for fut in as_completed(futures):
                results.append(fut.result())
        finally:
            # Best-effort: cancel any not-yet-started work on failure; running threads can’t be force-stopped. :contentReference[oaicite:4]{index=4}
            executor.shutdown(wait=True, cancel_futures=True)

        # Deterministic return ordering (useful for tests / logs).
        results.sort(key=lambda t: t[0])
        return results

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

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        tool_calls_limit: Optional[int] = 25,
        preview_limit: Optional[int] = None,
        pre_invoke: Optional[Tool | Callable[..., Any]] = None,
        post_invoke: Optional[Tool | Callable[..., Any]] = None,
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
    # Properties
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
    # Internal helpers
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

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #
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
                        "raw_args": dict(raw_args),
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


A2A_RESULT_KEY = "__py_A2A_result__"

# ───────────────────────────────────────────────────────────────────────────────
# A2AgentHost wrapper class
# ───────────────────────────────────────────────────────────────────────────────
class A2AgentHost:
    """
    A2AgentHost
    -----------
    Wraps a local :class:`~atomic_agentic.Primitives.Agent` as a python-a2a server
    using the message-level function-calling pattern.

    Exposed function names:
      - "invoke":          payload: Mapping[str, Any] -> {__py_A2A_result__: <agent.invoke(payload)>}
      - "agent_metadata":  no params                  -> {arguments_map: <agent.arguments_map>, return_type: <agent.post_invoke.return_type>}
    """

    def __init__(
        self,
        seed_agent: Agent,
        version: str = "1.0.0",
        host: str = "localhost",
        port: int = 5000,
    ) -> None:
        if not isinstance(seed_agent, Agent):
            raise TypeError("A2AgentHost requires a seed Agent.")
        self._seed_agent = seed_agent
        self._version = version
        self._host = host
        self._port = port

        outer = self

        @agent(
            name=seed_agent.name,
            description=seed_agent.description,
            version=version,
            url=f"http://{host}:{port}",
        )
        class _Server(A2AServer):
            """Per-instance A2A server wrapper around a local Agent."""

            def handle_message(self, message: Message) -> Message:
                content = message.content
                ctype = content.type

                # Text: brief help
                if ctype == "text":
                    return Message(
                        content=TextContent(
                            text="Call as function_call: name in {'invoke','agent_metadata'}."
                        ),
                        role=MessageRole.AGENT,
                        parent_message_id=message.message_id,
                        conversation_id=message.conversation_id,
                    )

                # Function call dispatch
                if ctype == "function_call":
                    fn = content.name
                    params = {p.name: p.value for p in (content.parameters or [])}

                    try:
                        if fn == "invoke":
                            payload = params.get("payload", {})
                            if not isinstance(payload, Mapping):
                                raise TypeError("invoke expects 'payload' to be a mapping")
                            result = outer._seed_agent.invoke(payload)  # returns Any
                            return Message(
                                content=FunctionResponseContent(
                                    name="invoke",
                                    response={A2A_RESULT_KEY: result},
                                ),
                                role=MessageRole.AGENT,
                                parent_message_id=message.message_id,
                                conversation_id=message.conversation_id,
                            )

                        if fn == "agent_metadata":
                            meta = {
                                "arguments_map": outer._seed_agent.arguments_map,
                                "return_type": outer._seed_agent.post_invoke.return_type,
                            }
                            return Message(
                                content=FunctionResponseContent(
                                    name="agent_metadata",
                                    response=meta,
                                ),
                                role=MessageRole.AGENT,
                                parent_message_id=message.message_id,
                                conversation_id=message.conversation_id,
                            )

                        # Unknown function
                        return Message(
                            content=FunctionResponseContent(
                                name=fn,
                                response={"error": f"Unknown function '{fn}'."},
                            ),
                            role=MessageRole.AGENT,
                            parent_message_id=message.message_id,
                            conversation_id=message.conversation_id,
                        )

                    except Exception as e:
                        return Message(
                            content=FunctionResponseContent(
                                name=fn,
                                response={"error": f"{type(e).__name__}: {e}"},
                            ),
                            role=MessageRole.AGENT,
                            parent_message_id=message.message_id,
                            conversation_id=message.conversation_id,
                        )

                # Fallback
                return Message(
                    content=TextContent(text="Unsupported content type."),
                    role=MessageRole.AGENT,
                    parent_message_id=message.message_id,
                    conversation_id=message.conversation_id,
                )

        self._server = _Server(url=f"http://{self._host}:{self._port}")

    @property
    def seed_agent(self) -> Agent:
        """The wrapped Agent instance."""
        return self._seed_agent

    @property
    def host(self) -> str:
        """The host address the server will bind to."""
        return self._host

    @property
    def port(self) -> int:
        """The port number the server will listen on."""
        return self._port

    def run(self, *, debug: bool = False) -> None:
        """Start the underlying A2A server."""
        run_server(self._server, host=self._host, port=self._port, debug=debug)
