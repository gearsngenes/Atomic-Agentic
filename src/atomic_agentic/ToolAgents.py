# modules/ToolAgents.py
from __future__ import annotations

# ──────────────────────────── Imports (consolidated) ──────────────────────────
import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Iterable, List, Optional, Union, Dict, Mapping

from . import Prompts
from .Agents import *
from .LLMEngines import LLMEngine
from .Toolify import toolify
from .Tools import Tool
from ._exceptions import ToolAgentError, ToolRegistrationError

# ──────────────────────────────────────────────────────────────────────────────
__all__ = ["PlannerAgent", "OrchestratorAgent",]

logger = logging.getLogger(__name__)

# Precompiled token pattern for {stepN} references
_STEP_TOKEN = re.compile(r"\{step(\d+)\}")

# Canonical final-return tool (must be used once as the last step of any plan)
def _return(val: Any) -> Any:
    return val

return_tool = Tool(
    func=_return,
    name="_return",
    description=(
        "Returns the passed-in value. This method should ONLY OCCUR ONCE as the "
        "FINAL STEP of any plan."
    ),
)

RETURN_KEY = return_tool.full_name

# ──────────────────────────────────────────────────────────────────────────────
# ToolAgent (abstract)
# ──────────────────────────────────────────────────────────────────────────────

class ToolAgent(Agent, ABC):
    """
    Schema-driven base for *tool-using* agents.

    Inherits from `Agent` and manages a toolbox of `Tool` objects produced by `toolify`.
    This class does **not** prescribe planning/execution; concrete subclasses
    (e.g., `PlannerAgent`) implement orchestration by overriding `_invoke(prompt)`.

    Core behaviors
    --------------
    • Constructor mirrors `Agent.__init__` and delegates to it.
    • `register(...)` accepts a single item (Tool|Agent|callable|str) and normalizes it
      via `toolify` into one or more `Tool` instances. Collisions are handled with
      `name_collision_mode = {"raise","skip","replace"}`.
    • `batch_register(...)` accepts an **Iterable** of (Tool|Agent|str); raw callables
      are intentionally **not** allowed in batch mode.
    • `list_tools()` returns an `OrderedDict[str, Tool]` (schema-first).
    • `actions_context()` returns a **human-readable** block: tool ids,
      signatures, required keys; suitable for inclusion in prompts.

    Thread-safety
    -------------
    • Registration/removal are guarded by a re-entrant lock. Read paths return copies.

    Subclass hook
    -------------
    • Override `_invoke(self, prompt: str) -> Any` to implement planning/execution.
      The base `Agent.invoke()` will have already validated the mapping and produced
      the `prompt` string via `pre_invoke`. Do not re-validate `prompt` here.
    """

    # ----------------------------- construction ------------------------------

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        role_prompt: Optional[str] = None,
        context_enabled: bool = True,
        *,
        pre_invoke: Optional[Tool] = None,
        history_window: int = 50,
    ) -> None:
        # Delegate validation and defaults to Agent
        super().__init__(
            name=name,
            description=description,
            llm_engine=llm_engine,
            role_prompt=role_prompt,
            context_enabled=context_enabled,
            pre_invoke=pre_invoke,
            history_window=history_window,
        )
        self._toolbox: OrderedDict[str, Tool] = OrderedDict()
        self._mcp_server_tag: int = 0  # unique tag seed for MCP server ids
        self._previous_steps: List[Dict[str, Any]] = []

        # Always provide the canonical final-return tool
        self.register(return_tool)

    # ------------------------------ helpers ----------------------------------

    def _resolve(self, obj: Any) -> Any:
        """
        Recursively resolve `{stepN}` placeholders in strings, lists, and dicts.

        Rules
        -----
        - Full-string "{stepN}" → returns the referenced result (preserves type).
        - Inline occurrences inside strings are replaced with str(previous_result).
        - Lists and dicts are traversed recursively.
        - Raises RuntimeError if a referenced step index is out of range or not completed.
        """
        if isinstance(obj, str):
            # Full-string token
            m = _STEP_TOKEN.fullmatch(obj)
            if m:
                idx = int(m.group(1))
                if idx >= len(self._previous_steps) or not self._previous_steps[idx]["completed"]:
                    raise RuntimeError(f"Step {idx} has not been completed yet.")
                return self._previous_steps[idx]["result"]

            # Inline replacements (strict raise on incomplete)
            def _repl(match: re.Match[str]) -> str:
                idx = int(match.group(1))
                if idx >= len(self._previous_steps) or not self._previous_steps[idx]["completed"]:
                    raise RuntimeError(f"Step {idx} has not been completed yet.")
                return str(self._previous_steps[idx]["result"])

            return _STEP_TOKEN.sub(_repl, obj)

        if isinstance(obj, list):
            return [self._resolve(v) for v in obj]

        if isinstance(obj, dict):
            return {k: self._resolve(v) for k, v in obj.items()}

        return obj

    # --------------------------- abstract contract ---------------------------

    @abstractmethod
    def _invoke(self, prompt: str) -> Any:
        """
        Subclass hook invoked by `Agent.invoke()` **after** the input mapping was
        validated and converted to a string prompt via `pre_invoke`.

        Implement your orchestration here (e.g., plan + tool execution) and return
        the final result (shape defined by the subclass).
        """
        raise NotImplementedError

    # ------------------------------ registration -----------------------------

    def register(
        self,
        item: Union[Tool, Agent, str, Callable],
        *,
        # ---- explicit toolify knobs (callable / agent / MCP) ----
        name: Optional[str] = None,
        description: Optional[str] = None,
        source: Optional[str] = None,
        # Name-collision handling
        name_collision_mode: str = "raise",
        # MCP-only (handled by toolify when items are MCP endpoints/descriptors)
        server_name: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Register a single input (Tool|Agent|callable|str) into the toolbox via `toolify`,
        using **explicit**, typed adapter parameters (no generic **kwargs).

        Acceptable `item` values:
        - `Tool` (passthrough)
        - `Agent` (wrapped by `toolify` as an AgentTool)
        - Python `callable` (requires both `name` and `description` here)
        - MCP endpoint URL string (requires `server_name`; optional include/exclude/headers)

        Parameters
        ----------
        item : Tool | Agent | str | callable
            One input accepted by `toolify`.
        name_collision_mode : {"raise","skip","replace"}, default "raise"
            Name-collision policy for identical `Tool.full_name`.

        Adapter parameters (explicit)
        -----------------------------
        name : Optional[str]
            Canonical name to assign (e.g., for callables).
        description : Optional[str]
            Human-readable description for prompts.
        source : Optional[str]
            Source label (e.g., "local", "mcp", "agent").
        server_name : Optional[str]
            MCP server identifier used by the adapter when `item` is an MCP endpoint.
        include : Optional[List[str]]
            MCP tool name whitelist.
        exclude : Optional[List[str]]
            MCP tool name blacklist.
        headers : Optional[dict[str,str]]
            Optional HTTP headers for MCP adapters.

        Returns
        -------
        List[str]
            The list of `full_name`s that ended up registered (inserted or replaced).

        Raises
        ------
        ToolRegistrationError
            On invalid `name_collision_mode` or name conflict under `"raise"`.
        ToolDefinitionError
            Propagated from `toolify` for unsupported or malformed inputs.
        """
        if name_collision_mode not in ("raise", "skip", "replace"):
            raise ToolRegistrationError(
                "name_collision_mode must be one of: 'raise', 'skip', 'replace'"
            )

        # Guard for raw callables: require non-empty name/description
        if isinstance(item, Callable):
            if not (isinstance(name, str) and isinstance(description, str) and name.strip() and description.strip()):
                raise ToolRegistrationError(
                    "ToolAgent.register: callables require non-empty 'name' and 'description'"
                )

        # Build explicit kwargs for toolify — clear and type-stable.
        toolify_params = {
            "name": name,
            "description": description,
            "source": source,
            "server_name": server_name,
            "include": include,
            "exclude": exclude,
            "headers": headers,
        }

        tools = toolify(item, **toolify_params)

        # Mutate toolbox atomically per tool under a lock
        registered: List[str] = []
        for tool in tools:
            full_name = tool.full_name
            exists = full_name in self._toolbox

            if exists:
                if name_collision_mode == "raise":
                    raise ToolRegistrationError(
                        f"Duplicate tool name: {full_name!r}. "
                        f"Use name_collision_mode='skip' or 'replace' to override."
                    )
                if name_collision_mode == "skip":
                    logger.info("ToolAgent.register: skipping duplicate %s", full_name)
                    continue
                # name_collision_mode == "replace"
                logger.info("ToolAgent.register: replacing %s", full_name)
                self._toolbox[full_name] = tool
                registered.append(full_name)
            else:
                self._toolbox[full_name] = tool
                registered.append(full_name)

        return registered

    def batch_register(
        self,
        items: Iterable[Union[Tool, Agent, str]],
        *,
        name_collision_mode: str = "raise",
    ) -> List[str]:
        """
        Register **multiple** inputs via `toolify`. This method intentionally excludes
        raw callables; register functions individually via `register(...)`.

        Acceptable `items`:
        - `Tool` (passthrough)
        - `Agent` (wrapped by `toolify` as an AgentTool)
        - `str` MCP endpoint URL (requires `server_name` to be inferred or provided per item)

        Parameters
        ----------
        items : Iterable[Tool | Agent | str]
            Iterable collection of inputs accepted by `toolify`.
        name_collision_mode : {"raise","skip","replace"}, default "raise"
            Name-collision policy for identical `Tool.full_name`.

        Returns
        -------
        List[str]
            The list of `full_name`s that ended up registered (inserted or replaced).

        Raises
        ------
        ToolRegistrationError
            On invalid `name_collision_mode`, non-iterable inputs, or raw callables in the list.
        ToolDefinitionError
            Propagated from `toolify` for unsupported or malformed inputs.
        """
        if name_collision_mode not in ("raise", "skip", "replace"):
            raise ToolRegistrationError(
                "name_collision_mode must be one of: 'raise', 'skip', 'replace'"
            )
        if not isinstance(items, Iterable):
            raise ToolRegistrationError(
                f"Expected an Iterable collection of items, but got {type(items)}"
            )
        if any(not isinstance(item, (Tool, Agent, str)) for item in items):
            raise ToolRegistrationError(
                "When registering multiple items, raw functions are not permitted"
            )

        produced: List[Tool] = []
        for obj in items:
            toolify_params = {
                "name": obj.name if not isinstance(obj, str) else None,
                "description": obj.description if not isinstance(obj, str) else None,
                "source": obj.name if isinstance(obj, Agent) else (
                    obj.source if isinstance(obj, Tool) else None
                ),
                "server_name": None if not isinstance(obj, str) else f"{self.name}_mcpserver_{self._mcp_server_tag}",
                "include": None,
                "exclude": None,
                "headers": None,
            }
            if isinstance(obj, str):
                self._mcp_server_tag += 1
            tools = toolify(obj, **toolify_params)
            # Contract: toolify returns only Tool instances
            if any(not isinstance(t, Tool) for t in tools):
                raise ToolRegistrationError("toolify produced a non-Tool instance.")
            produced.extend(tools)

        # Mutate toolbox atomically per tool under a lock
        registered: List[str] = []
        for tool in produced:
            full_name = tool.full_name
            exists = full_name in self._toolbox

            if exists:
                if name_collision_mode == "raise":
                    raise ToolRegistrationError(
                        f"Duplicate tool name: {full_name!r}. "
                        f"Use name_collision_mode='skip' or 'replace' to override."
                    )
                if name_collision_mode == "skip":
                    logger.info("ToolAgent.batch_register: skipping duplicate %s", full_name)
                    continue
                # name_collision_mode == "replace"
                logger.info("ToolAgent.batch_register: replacing %s", full_name)
                self._toolbox[full_name] = tool
                registered.append(full_name)
            else:
                self._toolbox[full_name] = tool
                registered.append(full_name)

        return registered

    # ------------------------------ management -------------------------------

    def has_tool(self, full_name: str) -> bool:
        """Return True if a tool with `full_name` is registered."""
        return full_name in self._toolbox

    def remove_tool(self, full_name: str) -> bool:
        """Remove tool by `full_name`. Returns True if it existed and was removed."""
        return self._toolbox.pop(full_name, None) is not None

    def clear_tools(self) -> None:
        """Remove all tools from the toolbox."""
        self._toolbox.clear()

    def list_tools(self) -> OrderedDict[str, Tool]:
        """
        Return a shallow copy of the toolbox mapping.

        Keys are canonical `Tool.full_name` strings; values are `Tool` instances.
        """
        return OrderedDict(self._toolbox)

    # ------------------------------ introspection -----------------------------

    def actions_context(self) -> str:
        """
        Build a **human-readable** context block of available actions for prompts.

        Returns
        -------
        str
            A newline-joined context block suitable for inclusion in prompts.
        """
        lines: List[str] = []
        for full_name, tool in self._toolbox.items():
            signature = tool.signature
            required = list(tool.required_names)
            desc = tool.description
            lines.append(f"- {signature}")
            if required:
                lines.append(f"  required: [{', '.join(required)}]")
            if desc:
                lines.append(f"  {desc}")
        return "\n".join(lines).strip()

    # ------------------------------ serialization ----------------------------

    def to_dict(self) -> OrderedDict[str, Any]:
        """Summarize this ToolAgent and its toolbox (safe for logs/telemetry)."""
        _my_dict = Agent.to_dict(self)
        _my_dict.update(OrderedDict(tools = [tool.to_dict() for _, tool in self._toolbox.items()]))
        return _my_dict


# ──────────────────────────────────────────────────────────────────────────────
# PlannerAgent
# ──────────────────────────────────────────────────────────────────────────────

class PlannerAgent(ToolAgent):
    """
    Single-shot planner that generates a complete step list then executes it.

    Behavior
    --------
    • Uses `Prompts.PLANNER_PROMPT` as the system prompt, with `{TOOLS}` replaced by the
      **human-readable** `actions_context()` list (ids, signatures, required keys).
    • Calls the LLM **once** to produce a JSON array of steps; stores the raw string in
      history (assistant turn) before execution.
    • Ensures the final step is the canonical `_return` tool.
    • Executes steps either synchronously or, if `run_concurrent=True`, with async fan-out
      for steps whose args do not reference incomplete `{stepN}` results.
      If an event loop is **already running** (e.g., Jupyter/Streamlit/FastAPI),
      the planner **falls back to synchronous** execution to avoid nested-loop errors.

    Parameters
    ----------
    run_concurrent : bool, default False
        When True, `.execute()` attempts concurrent execution (event-loop safe fallback).
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = False,
        *,
        pre_invoke: Optional[Tool] = None,
        history_window: int = 50,
        run_concurrent: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            llm_engine=llm_engine,
            role_prompt=Prompts.PLANNER_PROMPT,
            context_enabled=context_enabled,
            pre_invoke=pre_invoke,
            history_window=history_window,
        )
        self._run_concurrent = run_concurrent

    # -- public flag -----------------------------------------------------------

    @property
    def role_prompt(self):
        return self._role_prompt.format(TOOLS = self.actions_context())
    
    @property
    def run_concurrent(self) -> bool:
        """When True, `.execute()` attempts async execution; loop-safe fallback to sync."""
        return self._run_concurrent

    @run_concurrent.setter
    def run_concurrent(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ToolAgentError("run_concurrent must be a bool.")
        self._run_concurrent = value

    # -- planning & execution --------------------------------------------------

    def strategize(self, prompt: str) -> str:
        """
        Ask the LLM for a **full plan** (JSON array of steps) using the available tools.

        The system prompt contains the formatted AVAILABLE METHODS; the user prompt carries
        the task. `strategize` then returns the cleaned, raw JSON string to be parsed.
        """
        # 1) Format the user request into a json-decomposition task
        user_prompt = f"Decompose the following task into a JSON plan:\n{prompt}\n\n"

        # 2) Ask the LLM for a full plan (array of steps)
        logger.debug(f"{self.name}.strategize: Building messages")
        messages = [
            {"role": "system", "content": self.role_prompt},
        ]
        if self.context_enabled and self._history:
            messages += self._history[-2*self._history_window:]
        messages.append({"role": "user", "content": user_prompt})

        # Use positional signature to match base Agent consistency
        logger.debug(f"{self.name}.strategize: Invoking LLM")
        raw = self._llm_engine.invoke(messages)

        # Strip markdown fences (common LLM formatting)
        logger.debug(f"{self.name}.strategize: Cleaning LLM text")
        raw = re.sub(r"^```[a-zA-Z]*|```$", "", raw)
        
        return raw
    
    def load_steps(self, raw_plan: str)->list[dict]:
        """Converts raw plan JSON string into a python list"""
        logger.debug(f"{self.name}.load_steps: Parsing steps")
        steps: List[dict] = list(json.loads(raw_plan))
        if not steps or steps[-1].get("function") != RETURN_KEY:
            steps.append({"function": RETURN_KEY, "args": {"val": None}})

        return steps

    def execute(self, plan: List[dict]) -> Any:
        """
        Execute the provided plan and return the final step's result.

        Uses synchronous or asynchronous execution based on `run_concurrent`.
        Tracks each step's `result` and `completed` status in `_previous_steps`.
        """
        logger.debug(f"{self.name}.execute: Executing steps")
        self._previous_steps = [{"result": None, "completed": False} for _ in plan]

        if self._run_concurrent:
            # Loop-safe: avoid asyncio.run() if a loop is already running (e.g., Jupyter)
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # Graceful fallback to synchronous execution
                    return self._execute_sync(plan)
            except RuntimeError:
                # No running loop; OK to use asyncio.run
                pass
            return asyncio.run(self._execute_async(plan))
        else:
            return self._execute_sync(plan)

    def _execute_sync(self, steps: List[dict]) -> Any:
        """Synchronous step runner; rejects async callables with a clear error."""
        tools = self.list_tools()
        for i, step in enumerate(steps):
            step_tool = tools[step["function"]]
            logger.info(f"{type(self).__name__}.{self.name}._execute_sync: Running step {i} - {step['function']}, args: {step.get('args', {})}")
            args = self._resolve(step.get("args", {}))
            result = step_tool.invoke(inputs=args)
            self._previous_steps[i]["result"] = result
            self._previous_steps[i]["completed"] = True
        return self._previous_steps[-1]["result"]

    async def _execute_async(self, steps: List[dict]) -> Any:
        """
        Asynchronous step runner with simple dependency gating via `{stepN}` usage.
        A step is 'ready' when none of its args reference an incomplete step.
        """
        tools = self.list_tools()

        def get_deps(i: int) -> set[int]:
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

            walk(steps[i].get("args", {}))
            return found

        async def run_step(i: int) -> Any:
            step = steps[i]
            step_tool = tools[step["function"]]
            logger.info(f"{type(self).__name__}.{self.name}._execute_async: Running step {i} - {step['function']}, args: {step.get('args', {})}")
            args = self._resolve(step.get("args", {}))
            # Offload blocking tool invocation to a thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: step_tool.invoke(inputs=args))

        remaining = set(range(len(steps)))
        completed: set[int] = set()

        while remaining:
            ready = [i for i in remaining if get_deps(i) <= completed]
            if not ready:
                raise RuntimeError("Circular dependency in plan.")
            logger.info(f"{self.name}._execute_async: Gathered steps: {ready}")
            results = await asyncio.gather(*(run_step(i) for i in ready))
            for i, result in zip(ready, results):
                self._previous_steps[i]["result"] = result
                self._previous_steps[i]["completed"] = True
                completed.add(i)
                remaining.remove(i)

        return self._previous_steps[-1]["result"]

    # Hook from Agent.invoke(): prompt is already validated as str there
    def _invoke(self, prompt: str) -> Any:
        """
        High-level entry: generate a plan, execute it, and return the final value.

        - Emits progress via logging for visibility.
        - Stores the raw plan text (already added by `strategize`) and appends a summary
          turn after execution if `context_enabled`.
        """
        # Reset step history for this run
        self._previous_steps = []

        # Plan generation (stores raw plan as an assistant turn)
        plan_raw = self.strategize(prompt)
        # Parse plan string
        plan = self.load_steps(plan_raw)
        # Execute plan
        result = self.execute(plan)

        # Save user+assistant turn (summary) if context is enabled
        self._newest_history.append({
            "role":"user",
            "content": prompt
        })
        self._newest_history.append({
                "role": "assistant",
                "content": f"Generated plan:\n{plan_raw}\nExecuted Result: {str(result)}",
            })

        # Clear step tracker and return
        self._previous_steps = []
        return result
    
    def to_dict(self)-> OrderedDict[str, Any]:
        data = super().to_dict()
        data.update(OrderedDict(run_concurrent = self.run_concurrent))
        return data


class OrchestratorAgent(ToolAgent):
    """
    Iterative, schema-driven tool orchestrator.

    Behavior
    --------
    • System persona: `Prompts.ORCHESTRATOR_PROMPT` with `{TOOLS}` filled by `actions_context()`.
    • Each iteration, the LLM must return exactly ONE JSON object:
        {
          "step_call": { "function": "<type>.<source>.<name>", "args": {...} },
          "explanation": "why this step",
          "status": "CONTINUE" | "COMPLETE"
        }
      No prose, no markdown fences.
    • Placeholders:
        - Full-string "{stepN}" returns the referenced result with its original type.
        - Inline "{stepN}" inside a larger string stringifies the referenced result.
    • Execution:
        - Tool lookup is schema-first via canonical `Tool.full_name`.
        - Arg shape checks are delegated to `Tool.invoke` (intentional).
        - Unknown tool / placeholder resolution errors are recorded and allow repair,
          bounded by `max_failures`.
        - If the model returns "COMPLETE" for a non-`_return` step, a soft
          `"Tool.default._return"` is appended with the latest result.
    • Guards:
        - `max_steps` hard-stops after N executed steps.
        - `max_failures` hard-stops after M failed iterations.
    • History policy:
        - During iteration, a compact "Previous Steps" section is injected into the
          current user message; it is **not** persisted as separate turns.
        - On finish (success or failure), if `context_enabled=True`, exactly two stored
          turns are appended:
            1) user: the original prompt,
            2) assistant: a formatted summary of executed steps and the final result.
        - Stored history is never trimmed; `history_window` only limits what is sent.

    Parameters
    ----------
    name, description, llm_engine, role_prompt, context_enabled, pre_invoke, history_window
        See `Agent` / `ToolAgent`. `role_prompt` defaults to `Prompts.ORCHESTRATOR_PROMPT`.
    max_steps : int, default 25
        Max executed steps before aborting with failure summary.
    max_failures : int, default 5
        Max failed iterations (parse/lookup/resolve/tool failures) before abort.
    context_budget_chars : int, default 4000
        Approximate character budget for the inline "Previous Steps" section per turn.

    Returns
    -------
    Any
        Final value from the canonical `_return` tool, or the latest result on failure.
    """

    def __init__(
        self,
        name: str,
        description: str,
        llm_engine: LLMEngine,
        context_enabled: bool = True,
        *,
        pre_invoke: Optional[Tool] = None,
        history_window: int = 50,
        max_steps: int = 25,
        max_failures: int = 5,
        context_budget_chars: int = 4000,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            llm_engine=llm_engine,
            role_prompt=Prompts.ORCHESTRATOR_PROMPT,
            context_enabled=context_enabled,
            pre_invoke=pre_invoke,
            history_window=history_window,
        )
        self._max_steps = int(max_steps)
        self._max_failures = int(max_failures)
        self._context_budget_chars = int(context_budget_chars)
    
    @property
    def role_prompt(self):
        return self._role_prompt.format(TOOLS=self.actions_context(), MAX_EXPLAIN_WORDS = 30)

    # ------------------------------ public API --------------------------------

    def strategize_next(self, prompt: str) -> str:
        """
        Ask the LLM for the NEXT single step JSON object (as a raw string).

        The system message embeds available tools; the user message includes:
        - the current task,
        - a compact "Previous Steps" block,
        - and a final reminder to emit one JSON object using single-brace `{stepN}` placeholders.
        """
        messages = [
            {"role": "system", "content": self.role_prompt},
        ]

        # Windowed prior history (if any)
        if self.context_enabled and self._history:
            if self._history_window > 0:
                prior = self._history[-(self._history_window * 2):]
            else:
                prior = []
            messages += prior

        # Compose the current user prompt with compact previous steps
        prev_block = self._format_previous_steps(self._previous_steps, self._context_budget_chars)
        user_prompt = (
            f"USER TASK:\n{prompt}\n\n"
            f"Previous Steps (compact):\n{prev_block}\n\n"
            "Emit exactly ONE JSON object for the next step. "
            "Use single-brace placeholders like {step0} if needed."
        )
        messages.append({"role": "user", "content": user_prompt})

        # Engine call
        raw = self._llm_engine.invoke(messages)

        # Strip common markdown fences (if any)
        raw = re.sub(r"^```[a-zA-Z]*|```$", "", raw).strip()
        return raw

    def _invoke(self, prompt: str) -> Any:
        """
        Iteratively request and execute one step at a time until COMPLETE or guards trip.
        Returns the final value (or latest successful result on failure).
        """
        steps_executed = 0
        failures = 0
        final_result: Any = None

        # Ensure a clean step tracker for this run
        self._previous_steps = []

        tools = self.list_tools()

        while True:
            if steps_executed >= self._max_steps:
                logger.warning("%s._invoke: max_steps (%d) reached; aborting.", self.name, self._max_steps)
                break
            if failures >= self._max_failures:
                logger.warning("%s._invoke: max_failures (%d) reached; aborting.", self.name, self._max_failures)
                break

            # Ask for the next step
            try:
                raw = self.strategize_next(prompt)
                obj:dict = json.loads(raw)
            except Exception as e:
                # Parsing failure -> counted as iteration failure
                self._previous_steps.append({
                    "function": "<parse-error>",
                    "args": {},
                    "explanation": "Failed to parse single-step JSON.",
                    "result": f"ERROR: {e}",
                    "ok": False,
                    "completed": True,
                })
                failures += 1
                continue

            # Validate minimal shape (step_call/explanation/status)
            step_call = obj.get("step_call")
            explanation = obj.get("explanation", "")
            status = str(obj.get("status", "CONTINUE")).upper()

            if not isinstance(step_call, dict):
                self._previous_steps.append({
                    "function": "<shape-error>",
                    "args": {},
                    "explanation": "Missing or invalid 'step_call' object.",
                    "result": "ERROR: invalid schema",
                    "ok": False,
                    "completed": True,
                })
                failures += 1
                continue

            function = step_call.get("function")
            args = step_call.get("args", {})

            # Prepare a step slot early (so placeholder resolution can target it)
            step_index = len(self._previous_steps)
            self._previous_steps.append({
                "function": function if isinstance(function, str) else "<invalid>",
                "args": args if isinstance(args, (dict, list, str, int, float, bool, type(None))) else {},
                "explanation": explanation if isinstance(explanation, str) else "",
                "result": None,
                "ok": False,
                "completed": False,
            })

            # Lookup tool
            if not isinstance(function, str) or function not in tools:
                self._previous_steps[step_index].update({
                    "result": f"ERROR: unknown tool {function!r}",
                    "ok": False,
                    "completed": True,
                })
                failures += 1
                continue

            tool = tools[function]

            # Resolve placeholders (single-brace policy)
            try:
                resolved_args = self._resolve(args)
            except Exception as e:
                self._previous_steps[step_index].update({
                    "result": f"ERROR: placeholder resolution failed: {e}",
                    "ok": False,
                    "completed": True,
                })
                failures += 1
                continue

            # Execute the tool (arg validation is handled by the Tool itself)
            try:
                logger.info(f"{self.name}._invoke: executing step {step_index}: {tool.full_name}")
                result = tool.invoke(inputs=resolved_args)
            except Exception as e:
                self._previous_steps[step_index].update({
                    "result": f"ERROR: tool invocation failed: {e}",
                    "ok": False,
                    "completed": True,
                })
                failures += 1
                continue

            # Success
            self._previous_steps[step_index].update({
                "result": result,
                "ok": True,
                "completed": True,
            })
            final_result = result
            steps_executed += 1

            # Termination handling
            if status == "COMPLETE":
                if function != RETURN_KEY:
                    # Soft-append a final return step for parity with Planner
                    try:
                        ret = tools[RETURN_KEY].invoke(inputs={"val": final_result})
                    except Exception as e:
                        # This should not realistically fail, but capture if it does
                        self._previous_steps.append({
                            "function": RETURN_KEY,
                            "args": {"val": "<unavailable>"},
                            "explanation": "Soft finalization via _return",
                            "result": f"ERROR: return failed: {e}",
                            "ok": False,
                            "completed": True,
                        })
                        break

                    self._previous_steps.append({
                        "function": RETURN_KEY,
                        "args": {"val": final_result},
                        "explanation": "Soft finalization via _return",
                        "result": ret,
                        "ok": True,
                        "completed": True,
                    })
                    final_result = ret
                break  # Completed

        self._newest_history.append({"role": "user", "content": prompt})
        self._newest_history.append({
            "role": "assistant",
            "content": self._format_final_summary(self._previous_steps, final_result),
        })

        # Reset internal step tracker for next call
        self._previous_steps = []
        return final_result

    # ------------------------------ helpers ----------------------------------

    @staticmethod
    def _truncate(s: str, max_len: int) -> str:
        if len(s) <= max_len:
            return s
        if max_len <= 3:
            return s[:max_len]
        return s[: max_len - 3] + "..."

    def _format_previous_steps(self, steps: List[dict], budget: int) -> str:
        """
        Compact single-line summaries per step, truncated to a character budget.
        Example line:
          0) plugin.Math.mul(args={a:2,b:3}) → 6
          1) function.default._return(args={val:{step0}}) → 6
        Failed steps show: → ERROR: <message>
        """
        if not steps:
            return "(none)"
        parts: List[str] = []
        used = 0
        for i, st in enumerate(steps):
            fn = str(st.get("function", "<?>"))
            args = st.get("args", {})
            res = st.get("result", None)
            ok = bool(st.get("ok", False))

            # Compact arg/result stringifications
            try:
                args_s = json.dumps(args, ensure_ascii=False)
            except Exception:
                args_s = str(args)
            args_s = self._truncate(args_s, 256)

            if ok:
                try:
                    res_s = json.dumps(res, ensure_ascii=False)
                except Exception:
                    res_s = str(res)
            else:
                res_s = str(res)
            res_s = self._truncate(res_s, 256)

            line = f"{i}) {fn}(args={args_s}) → {'OK ' if ok else ''}{res_s if res_s is not None else 'null'}"
            if used + len(line) + 1 > budget:
                parts.append("… (truncated)")
                break
            parts.append(line)
            used += len(line) + 1

        return "\n".join(parts)

    def _format_final_summary(self, steps: List[dict], final_result: Any) -> str:
        header = "Executed Steps:"
        lines = [header]
        for i, st in enumerate(steps):
            fn = str(st.get("function", "<?>"))
            try:
                args_s = json.dumps(st.get("args", {}), ensure_ascii=False)
            except Exception:
                args_s = str(st.get("args", {}))
            ok = bool(st.get("ok", False))
            try:
                res_s = json.dumps(st.get("result"), ensure_ascii=False)
            except Exception:
                res_s = str(st.get("result"))
            lines.append(f"{i}) {fn}(args={args_s}) → {'OK ' if ok else ''}{res_s}")
        try:
            final_s = json.dumps(final_result, ensure_ascii=False)
        except Exception:
            final_s = str(final_result)
        lines.append(f"\nFinal Result: {final_s}")
        return "\n".join(lines)
    
    def to_dict(self) -> OrderedDict[str,Any]:
        base = super().to_dict()
        base.update({
            "max_steps": self._max_steps,
            "max_failures": self._max_failures,
            "context_budge_chars" : self._context_budget_chars,
        })
        return base
