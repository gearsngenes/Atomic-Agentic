# modules/ToolAgents.py
from __future__ import annotations

import logging
import re
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Iterable, List, Mapping, Optional, Union, Callable

from .Agents import Agent
from .LLMEngines import LLMEngine
from .Tools import Tool, ToolDefinitionError
from .ToolAdapters import toolify

__all__ = [
    "ToolAgent",
    "ToolAgentError",
    "ToolRegistrationError",
]

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────────
# Errors
# ───────────────────────────────────────────────────────────────────────────────

class ToolAgentError(RuntimeError):
    """Base exception for ToolAgent-related errors."""


class ToolRegistrationError(ToolAgentError):
    """Raised when registering tools fails due to collisions or bad inputs."""


# ───────────────────────────────────────────────────────────────────────────────
# ToolAgent (abstract)
# ───────────────────────────────────────────────────────────────────────────────

def _return(val: Any): return val

return_tool = Tool(
    func= _return,
    name= "_return",
    description= "Returns the passed-in value. This method should ONLY OCCUR ONCE as the FINAL STEP of any plan.")

class ToolAgent(Agent, ABC):
    """
    Schema-driven base for tool-using agents.

    Inherits from `Agent` and `ABC`. Manages a toolbox of `Tool` objects
    produced by `toolify`. This class does **not** implement planning/execution;
    concrete subclasses (e.g., Planner/Orchestrator) must implement `invoke`.

    Key behaviors
    -------------
    • Constructor mirrors `Agent.__init__` and delegates to it.
    • `register(...)` accepts one object or a list of objects; each is normalized
      via `toolify` into one or more `Tool` instances. Collisions are handled
      with `mode={"raise","skip","replace"}`.
    • `list_tools()` returns `Tool` objects (schema-first).
    • `actions_context(with_schemas=True)` includes canonical signatures and
      required keys (from `Tool`), not free-form text.

    Not supported
    -------------
    • No plugin inputs. Registration accepts only: `Tool`, `Agent`, Python `callable`,
      and MCP endpoint URL strings (as supported by `toolify`).

    Thread-safety
    -------------
    • Registration/removal are guarded by a re-entrant lock. Read paths return copies.

    Notes for subclass authors
    --------------------------
    • Implement `invoke(self, inputs: Mapping[str, Any]) -> Any`.
    • If you need a prompt string, call `self.pre_invoke.invoke(inputs)` and validate
      its type/shape before using it.
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
        self._lock = threading.RLock()
        self._mcp_server_tag:int = 0
        self._previous_steps: list[dict] = []
        self.register(return_tool)

    def _resolve(self, obj: Any) -> Any:
        """Recursively resolve {stepN} placeholders in strings, lists, and dicts.

        Rules
        -----
        - Full-string "{stepN}" → returns the referenced result (preserves type).
        - Inline occurrences inside strings are replaced with str(previous_result).
        - Lists and dicts are traversed recursively.
        - Raises RuntimeError if a referenced step index is out of range or incomplete.
        """
        import re
        if isinstance(obj, str):
            # Full-string token
            m = re.fullmatch(r"\{step(\d+)\}", obj)
            if m:
                idx = int(m.group(1))
                if idx >= len(self._previous_steps) or not self._previous_steps[idx]["completed"]:
                    raise RuntimeError(f"Step {idx} has not been completed yet.")
                return self._previous_steps[idx]["result"]

            # Inline replacements with strict raise on incomplete
            def _repl(match: re.Match[str]) -> str:
                idx = int(match.group(1))
                if idx >= len(self._previous_steps) or not self._previous_steps[idx]["completed"]:
                    raise RuntimeError(f"Step {idx} has not been completed yet.")
                return str(self._previous_steps[idx]["result"])

            return re.sub(r"\{step(\d+)\}", _repl, obj)

        if isinstance(obj, list):
            return [self._resolve(v) for v in obj]

        if isinstance(obj, dict):
            return {k: self._resolve(v) for k, v in obj.items()}

        return obj

    # --------------------------- abstract contract ---------------------------

    @abstractmethod
    def _invoke(self, prompt: str) -> Any:
        """
        Invoke the tool-using agent with a single **input mapping**.

        Subclasses must:
        1) Validate `inputs` is a Mapping.
        2) Optionally call `self.pre_invoke.invoke(inputs)` and validate the output.
        3) Execute their plan/orchestration using `Tool` objects from `list_tools()`.
        4) Return the final result (shape defined by the subclass).
        """
        raise NotImplementedError

    # ------------------------------ registration -----------------------------

    def register(
        self,
        item: Union[Tool,Agent,str,Callable],
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
        headers: Optional[dict[str, str]] = None,
    ) -> List[str]:
        """
        Register one or many inputs into the toolbox via `toolify`, using explicit,
        named parameters for adapter configuration (no generic **kwargs).

        Acceptable `items`:
        - `Tool` (passthrough)
        - `Agent` (wrapped as an AgentTool by toolify)
        - Python `callable` (requires `name` and `description` here)
        - MCP endpoint URL string (requires `server_name`; optional include/exclude/headers)

        Parameters
        ----------
        items : Tool | Agent | str | callable | Iterable[...]
            Single input or an iterable of inputs accepted by `toolify`.
        mode : {"raise","skip","replace"}, default "raise"
            Name-collision policy for identical `Tool.full_name`.

        Adapter parameters (explicit)
        -----------------------------
        name : Optional[str]
            Canonical name to assign (e.g., for callables).
        description : Optional[str]
            Human-readable description for prompts.
        source : Optional[str]
            Source label (e.g., "local", "mcp", "agent").
        tool_type : Optional[str]
            Logical type tag (e.g., "function", "mcp", "agent").
        server_name : Optional[str]
            MCP server identifier used by the adapter when `items` are MCP endpoints.
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
            On invalid `mode` or name conflict under `"raise"`.
        ToolDefinitionError
            Propagated from `toolify` for unsupported or malformed inputs.
        """
        if name_collision_mode not in ("raise", "skip", "replace"):
            raise ToolRegistrationError("mode must be one of: 'raise', 'skip', 'replace'")
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
        if isinstance(item, Callable) and not (isinstance(name, str) and isinstance(description, str)) and not (name.strip() and description.strip()):
            raise ToolRegistrationError("ToolAgent.register: method missing 'name' and/or 'description'")

        # Toolify OUTSIDE the lock (may raise ToolDefinitionError)
        tools = toolify(item, **toolify_params)
        # Mutate toolbox atomically per tool under a lock
        registered: List[str] = []
        with self._lock:
            for tool in tools:
                full_name = tool.full_name
                exists = full_name in self._toolbox

                if exists:
                    if name_collision_mode == "raise":
                        raise ToolRegistrationError(
                            f"Duplicate tool name: {full_name!r}. "
                            f"Use mode='skip' or mode='replace' to override."
                        )
                    if name_collision_mode == "skip":
                        logger.debug("ToolAgent.register: skipping duplicate %s", full_name)
                        continue
                    # mode == "replace"
                    logger.debug("ToolAgent.register: replacing %s", full_name)
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
        Register one or many inputs into the toolbox via `toolify`, using explicit,
        named parameters for adapter configuration (no generic **kwargs).

        Acceptable `items`:
        - `Tool` (passthrough)
        - `Agent` (wrapped as an AgentTool by toolify)
        - Python `callable` (requires `name` and `description` here)
        - MCP endpoint URL string (requires `server_name`; optional include/exclude/headers)

        Parameters
        ----------
        items : Tool | Agent | str | callable | Iterable[...]
            Single input or an iterable of inputs accepted by `toolify`.
        mode : {"raise","skip","replace"}, default "raise"
            Name-collision policy for identical `Tool.full_name`.

        Adapter parameters (explicit)
        -----------------------------
        name : Optional[str]
            Canonical name to assign (e.g., for callables).
        description : Optional[str]
            Human-readable description for prompts.
        source : Optional[str]
            Source label (e.g., "local", "mcp", "agent").
        tool_type : Optional[str]
            Logical type tag (e.g., "function", "mcp", "agent").
        server_name : Optional[str]
            MCP server identifier used by the adapter when `items` are MCP endpoints.
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
            On invalid `mode` or name conflict under `"raise"`.
        ToolDefinitionError
            Propagated from `toolify` for unsupported or malformed inputs.
        """
        if name_collision_mode not in ("raise", "skip", "replace"):
            raise ToolRegistrationError("mode must be one of: 'raise', 'skip', 'replace'")
        # Normalize to list without mutating toolbox yet
        if not isinstance(items, Iterable):
            raise ToolRegistrationError(f"Expected an Iterable collection of items, but got {type(items)}")
        if any(not isinstance(item, (Tool, Agent, str)) for item in items):
            raise ToolRegistrationError("When registering multiple items, raw functions are not permitted")
        # Toolify OUTSIDE the lock (may raise ToolDefinitionError)
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
            if isinstance(obj, str): self._mcp_server_tag += 1
            tools = toolify(obj, **toolify_params)
            # Contract: toolify returns only Tool instances
            if any(not isinstance(t, Tool) for t in tools):
                raise ToolRegistrationError("toolify produced a non-Tool instance.")
            produced.extend(tools)

        # Mutate toolbox atomically per tool under a lock
        registered: List[str] = []
        with self._lock:
            for tool in produced:
                full_name = tool.full_name
                exists = full_name in self._toolbox

                if exists:
                    if name_collision_mode == "raise":
                        raise ToolRegistrationError(
                            f"Duplicate tool name: {full_name!r}. "
                            f"Use mode='skip' or mode='replace' to override."
                        )
                    if name_collision_mode == "skip":
                        logger.debug("ToolAgent.register: skipping duplicate %s", full_name)
                        continue
                    # mode == "replace"
                    logger.debug("ToolAgent.register: replacing %s", full_name)
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
        with self._lock:
            return self._toolbox.pop(full_name, None) is not None

    def clear_tools(self) -> None:
        """Remove all tools from the toolbox."""
        with self._lock:
            self._toolbox.clear()

    def list_tools(self) -> OrderedDict[str, Tool]:
        """
        Return a shallow copy of the toolbox mapping.

        Keys are canonical `Tool.full_name` strings; values are `Tool` instances.
        """
        return OrderedDict(self._toolbox)

    # ------------------------------ introspection -----------------------------

    def actions_context(self, *, with_schemas: bool = True) -> str:
        """
        Build a human-readable context block of available actions.

        Parameters
        ----------
        with_schemas : bool, default True
            If True, include each Tool's canonical `signature` and its required keys.
            If False, include only `full_name` and description.

        Returns
        -------
        str
            A newline-joined context block suitable for inclusion in prompts.
        """
        lines: List[str] = []
        for full_name, tool in self._toolbox.items():
            if with_schemas:
                # Use Tool's concrete properties (no attribute probing).
                signature = tool.signature
                required = sorted(tool.required_names)
                desc = tool.description
                lines.append(f"- {full_name}: {signature}")
                if required:
                    lines.append(f"  required: [{', '.join(required)}]")
                if desc:
                    lines.append(f"  {desc}")
            else:
                desc = tool.description
                lines.append(f"- {full_name}: {desc}" if desc else f"- {full_name}")
        return "\n".join(lines).strip()

    # ------------------------------ serialization -----------------------------

    def to_dict(self) -> dict:
        """Summarize this ToolAgent and its toolbox (safe for logs/telemetry)."""
        tools_block: List[dict] = []
        for full_name, tool in self._toolbox.items():
            # Use Tool's public properties directly.
            rt = tool.return_type
            rt_str = getattr(rt, "__name__", repr(rt))
            tools_block.append({
                "full_name": full_name,
                "signature": tool.signature,
                "return_type": rt_str,
                "type": tool.type,
                "source": tool.source,
                "name": tool.name,
                "description": tool.description,
                "required": sorted(tool.required_names),
            })

        return {
            "name": self.name,
            "description": self.description,
            "role_prompt": bool(self.role_prompt),
            "context_enabled": self.context_enabled,
            "history_window": self.history_window,
            "attachments_count": len(self.attachments),
            "pre_invoke": self.pre_invoke.name,
            "tool_count": len(self._toolbox),
            "tools": tools_block,
        }

from . import Prompts
import json
import asyncio

class PlannerAgent(ToolAgent):
    """Single-shot planner that generates a complete step list then executes it.

    - Uses `Prompts.PLANNER_PROMPT` as the system prompt.
    - Stores the raw LLM plan output in history (assistant turn) before execution.
    - Appends a canonical `function.default._return` step when missing.
    - `is_async` determines whether execution uses `execute_async`.
    """

    def __init__(self,
                 name: str,
                 description: str,
                 llm_engine: LLMEngine,
                 context_enabled = False,
                 *,
                 pre_invoke: Tool|None = None,
                 history_window: int = 50,
                 run_concurrent=False):
        """Initialize planner state and behavior flags."""
        super().__init__(name = name,
                         description=description,
                         llm_engine=llm_engine,
                         role_prompt=Prompts.PLANNER_PROMPT,
                         context_enabled = context_enabled,
                         pre_invoke=pre_invoke,
                         history_window=history_window)
        self._run_concurrent = run_concurrent

    @property
    def run_concurrent(self): 
        """bool: When True, `.execute()` dispatches to `.execute_async()`."""
        return self._run_concurrent

    @run_concurrent.setter
    def run_concurrent(self, value: bool): 
        """Enable/disable async execution mode."""
        self._run_concurrent = value

    @property
    def description(self):
        """Decorated description presented to users/planners (read-only facade)."""
        return f"~~Planner Agent {self.name}~~\nThis agent decomposes tasks into a list of tool calls.\nDescription: {self._description}"

    @description.setter
    def description(self, val): 
        """Internal storage for human-authored description (used in facade)."""
        self._description = val

    def strategize(self, prompt: str) -> dict:
        """Ask the LLM for a **full plan** (JSON array of steps) using the available tools.

        The system prompt contains the formatted AVAILABLE METHODS; the user prompt carries
        the task and the key matching rule. The raw JSON string is stored in history.
        The final plan is returned as a Python list, ensuring a trailing return step.
        """
        # 1) Build the AVAILABLE METHODS block from Tool objects
        block = self.actions_context()

        user_prompt = (
            f"Decompose the following task into a JSON plan:\n{prompt}\n\n"
        )

        # 2) Ask the LLM for a full plan (array of steps)
        messages = [
            {"role": "system", "content": Prompts.PLANNER_PROMPT.format(TOOLS = block)},
        ]
        if self.context_enabled:
            messages += self._history
        messages.append({"role": "user", "content": user_prompt})
        raw = self._llm_engine.invoke(messages = messages, file_paths = self._attachments)
        raw = re.sub(r'^```[a-zA-Z]*|```$', '', raw)
        self._history.append({"role": "assistant", "content": raw})
        steps: list[dict] = list(json.loads(raw))
        # 3) Ensure last step is the canonical return tool
        if not steps or steps[-1].get('function') != 'function.default._return':
            steps.append({"function": "function.default._return", "args": {"val": None}})

        return steps

    def execute(self, plan: list[dict]) -> Any:
        """Execute the provided plan in order; returns the final step's result.

        Uses synchronous or asynchronous execution based on `is_async`. Tracks each
        step's `result` and `completed` status in `_previous_steps`.
        """
        self._previous_steps = [{"result": None, "completed": False} for _ in plan]
        return asyncio.run(self._execute_async(plan)) if self._run_concurrent else self._execute_sync(plan)

    def _execute_sync(self, steps: list[dict]) -> Any:
        """Synchronous step runner; rejects async callables with a clear error."""
        tools = self.list_tools()
        for i, step in enumerate(steps):
            step_tool = tools[step["function"]]
            logging.debug(f"[TOOL] {step['function']} args: {step.get('args', {})}")
            args = self._resolve(step.get("args", {}))
            result = step_tool.invoke(inputs = args)
            self._previous_steps[i]["result"] = result
            self._previous_steps[i]["completed"] = True
        return self._previous_steps[-1]["result"]

    async def _execute_async(self, steps: list[dict]) -> Any:
        """Asynchronous step runner with simple dependency gating via {{stepN}} use."""
        tools = self.list_tools()

        def get_deps(i):
            import re
            pat = re.compile(r"\{step(\d+)\}")
            found = set()
            def walk(x):
                if isinstance(x, str):
                    for m in pat.finditer(x):
                        found.add(int(m.group(1)))
                elif isinstance(x, list):
                    for v in x: walk(v)
                elif isinstance(x, dict):
                    for v in x.values(): walk(v)
            walk(steps[i].get("args", {}))
            return found


        async def run_step(i: int):
            step = steps[i]
            step_tool = tools[step["function"]]
            logging.info(f"[TOOL] {step['function']} args: {step.get('args', {})}")
            args = self._resolve(step.get("args", {}))
            return await asyncio.get_running_loop().run_in_executor(None, lambda: step_tool.invoke(inputs = args))

        remaining = set(range(len(steps)))
        completed = set()

        while remaining:
            ready = [i for i in remaining if get_deps(i) <= completed]
            if not ready:
                raise RuntimeError("Circular dependency in plan.")
            results = await asyncio.gather(*(run_step(i) for i in ready))
            for i, result in zip(ready, results):
                self._previous_steps[i]["result"] = result
                self._previous_steps[i]["completed"] = True
                completed.add(i)
                remaining.remove(i)

        return self._previous_steps[-1]["result"]

    def _invoke(self, prompt: str)-> Any:
        """High-level entry point: generate a plan, execute it, and return the final value.

        - Emits start/finish banners via logging for visibility.
        - Stores the raw plan text as the latest assistant turn only when `context_enabled`.
        """
        if not isinstance(prompt, str):
            raise RuntimeError(f"{self.name}.pre_invoke must be a tool that returns a string when invoked")
        # reset step history
        self._previous_steps = []
        # generate json-formatted plan
        plan = self.strategize(prompt)
        # retrieve raw string output
        plan_raw = self._history[-1]["content"] if self._history else ""
        self._history = self._history[:-1]
        logging.debug(f"{self.name} created plan with {len(plan)} steps")
        # execute plan
        result = self.execute(plan)
        # save history
        if self.context_enabled:
            self._history.append({"role": "user", "content": prompt})
            self._history.append({"role": "assistant", "content": f"Generated plan:\n{plan_raw}\nExecuted Result: {str(result)}"})
        self._previous_steps = []
        return result
