# Tools.py
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, get_type_hints, TypedDict

# External integrations (MCP) + local modules
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from modules.Agents import Agent
from modules.Plugins import *  # Provides Plugin-shaped dicts (see TypedDict below)


# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
__all__ = ["Tool", "ToolFactory"]

# ───────────────────────────────────────────────────────────────────────────────
# Tool
# ───────────────────────────────────────────────────────────────────────────────

class Tool:
    """
    A thin, immutable wrapper around a Python callable (or an agent/plugin/MCP façade),
    exposing a stable, fully-qualified name and a user-facing signature string.

    Fully-qualified naming convention (stable key for planners/orchestrators):
        <type>.<source>.<name>

    Examples:
        function.default.add
        plugin.MathPlugin.multiply
        agent.Researcher.invoke
        mcp.fastmcp_server.mcp_summarize

    Parameters
    ----------
    name : str
        The short method name (right-most segment of the key).
        Must be stable and human-readable for docs/prompts.

    func : Callable[..., Any]
        The underlying Python callable to execute. May be a sync function only
        for ToolAgent/Planner/Orchestrator usage; async callables should be
        wrapped upstream where appropriate.

    type : str, default "function"
        Logical category: "function" | "plugin" | "agent" | "mcp".
        Drives the <type> segment in `full_name` and grouping in prompts.

    source : str, default "default"
        Namespace for the tool (e.g., plugin name, agent name, MCP server name).

    description : str, default ""
        Short, task-oriented doc used in AVAILABLE METHODS prompts. Keep
        parameter details short; Tool.signature already includes arg names.

    clear_mem_func : Optional[Callable[[], None]]
        Optional callback invoked by `clear_memory()` to reset any internal
        state the wrapped callable keeps (e.g., caches). No-op if None.

    Attributes
    ----------
    signature : str
        Human-readable signature: "<full_key>(a: int, b: int = 0) → str".
        Built lazily from Python type hints if present; falls back to `Any`.

    Notes
    -----
    - This class does NOT validate or coerce arguments. Callers (e.g., ToolAgent)
      should resolve placeholders and pass kwargs positionally/nominally as needed.
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        type: str = "function",
        source: str = "default",
        description: str = "",
        clear_mem_func: Optional[Callable[[], None]] = None,
    ) -> None:
        self._type: str = type
        self._source: str = source
        self._name: str = name
        self._func: Callable[..., Any] = func
        self._description: str = description
        self.signature: str = Tool._build_signature(self.full_name, func)
        self._clear_mem: Optional[Callable[[], None]] = clear_mem_func

    # ── Read-only properties (stable surface used in planners/orchestrators) ──
    @property
    def type(self) -> str:
        """Logical category: 'function' | 'plugin' | 'agent' | 'mcp'."""
        return self._type

    @property
    def source(self) -> str:
        """Namespace for the method (plugin name, agent name, or MCP server key)."""
        return self._source

    @property
    def name(self) -> str:
        """Short method name (right-most segment)."""
        return self._name

    @property
    def full_name(self) -> str:
        """Fully-qualified tool key: '<type>.<source>.<name>'."""
        return f"{self._type}.{self._source}.{self._name}"

    @property
    def description(self) -> str:
        """User-facing help text for prompts and docs."""
        full_description = f"{self.signature}: {self._description}"
        return full_description#self._description
    @description.setter
    def description(self, val: str):
        """User-facing description setter"""
        self._description = val

    @property
    def func(self) -> Callable[..., Any]:
        """The underlying callable to execute."""
        return self._func

    # ── Control ────────────────────────────────────────────────────────────────
    def clear_memory(self) -> None:
        """
        Invoke the optional 'clear_mem_func' to reset any internal state/caches.
        No-op if no callback was provided.
        """
        if self._clear_mem is not None:
            self._clear_mem()

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the underlying callable.

        Notes
        -----
        - Planner/Orchestrator infrastructure already resolves placeholders into
          concrete Python objects before calling this.
        - If the callable is async, it should be awaited by an async-aware runner;
          the default Planner/Orchestrator paths assume sync callables.
        """
        return self._func(*args, **kwargs)

    # ── Helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_signature(key: str, func: Callable[..., Any]) -> str:
        """
        Build a human-readable, single-line signature using Python type hints.

        Example:
            "plugin.MathPlugin.add(a: int, b: int) → int"
        """
        sig = inspect.signature(func)
        hints = get_type_hints(func)

        params: List[str] = []
        for n, p in sig.parameters.items():
            if n == "self":
                continue
            annotated = hints.get(n, Any)
            ann_name = getattr(annotated, "__name__", str(annotated))
            default_str = f" = {p.default!r}" if p.default is not inspect._empty else ""
            params.append(f"{n}: {ann_name}{default_str}")

        rtype = hints.get("return", Any)
        rtype_name = getattr(rtype, "__name__", str(rtype))
        return f"{key}({', '.join(params)}) → {rtype_name}"

    # ── Debugging niceties ─────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"Tool<{self.full_name}>"


# ───────────────────────────────────────────────────────────────────────────────
# ToolFactory
# ───────────────────────────────────────────────────────────────────────────────

class ToolFactory:
    """
    Factory helpers to convert common objects into `Tool` instances.

    Supported inputs (polymorphic):
      • Plain function          → one Tool ("function.default.<fn_name>")
      • Agent                   → four Tools (invoke/attach/detach/clear_memory) under type="agent"
      • Plugin-dict             → N Tools (one per method) under type="plugin"
      • MCP server (string URL) → N Tools (one per remote tool) under type="mcp"

    All generated `Tool` objects follow the same fully-qualified key convention
    and expose `.signature` and `.description` suitable for LLM prompting.
    """

    # ── Functions → Tools ──────────────────────────────────────────────────────
    @staticmethod
    def toolify_function(
        func: Callable[..., Any],
        type: str = "function",
        source: str = "default",
        description: str = "",
    ) -> List[Tool]:
        """
        Wrap a named Python function as a single Tool.

        Raises
        ------
        ValueError
            If func.__name__ == '<lambda>' (anonymous lambdas are unstable keys).
        """
        if func.__name__ == "<lambda>":
            raise ValueError("Lambda functions must be given proper names.")
        return [Tool(name=func.__name__, func=func, type=type, source=source, description=description)]

    # ── Agents → Tools ─────────────────────────────────────────────────────────
    @staticmethod
    def toolify_agent(agent: Agent) -> List[Tool]:
        """
        Expose an Agent as four tools under type='agent' and source=<agent.name>:

          agent.<name>.invoke(prompt: str) → str
              Invoke the agent with a prompt. Uses the agent's configured LLM engine.

          agent.<name>.attach(path: str) → bool
              Attach a local file path to the agent's internal attachment list.

          agent.<name>.detach(path: str) → bool
              Detach a local file path from the agent.

          agent.<name>.clear_memory() → None
              Clear only the agent's conversation history (attachments remain).
        """
        invoke_tool = ToolFactory.toolify_function(
            func=agent.invoke,
            type="agent",
            source=agent.name,
            description=f"Invoke the {agent.name} agent. Agent description: {agent.description}",
        )
        attach_tool = ToolFactory.toolify_function(
            func=agent.attach,
            type="agent",
            source=agent.name,
            description=f"Attach a local file path to {agent.name}'s attachments list. "
                        f"Use only when a specific path is explicitly required.",
        )
        detach_tool = ToolFactory.toolify_function(
            func=agent.detach,
            type="agent",
            source=agent.name,
            description=f"Detach a local file path from {agent.name}'s attachments list.",
        )
        clear_tool = ToolFactory.toolify_function(
            func=agent.clear_memory,
            type="agent",
            source=agent.name,
            description=f"Clear {agent.name}'s conversation history (attachments unaffected).",
        )
        return invoke_tool + attach_tool + detach_tool + clear_tool

    # ── Plugin dicts → Tools ───────────────────────────────────────────────────
    @staticmethod
    def toolify_plugin(plugin: Plugin) -> List[Tool]:
        """
        Convert a plugin dict into Tools (one per method in method_map).

        Each generated Tool is:
          type="plugin", source=<plugin['name']>, name=<method_name>

        The callable's __name__ is normalized to the method_name if it was a
        lambda (stable, readable keys for planners/orchestrators).
        """
        tools: List[Tool] = []
        source: str = plugin.get("name", "unknown")
        tool_map: Dict[str, Dict[str, Any]] = plugin.get("method_map", {})

        for method_name, method_info in tool_map.items():
            func = method_info.get("callable")
            description = method_info.get("description", "")
            if not callable(func):
                continue
            if getattr(func, "__name__", "<lambda>") == "<lambda>":
                # Stabilize the callable name to keep a readable fully-qualified key
                try:
                    func.__name__ = method_name  # type: ignore[attr-defined]
                except Exception:
                    # Fallback: best-effort; signature still contains parameter names
                    pass
            tools.append(
                Tool(
                    name=method_name,
                    func=func,
                    type="plugin",
                    source=source,
                    description=description,
                )
            )
        return tools

    # ── MCP server URL → Tools ─────────────────────────────────────────────────
    @staticmethod
    def toolify_mcp_server(name: str, url_or_base: str) -> List[Tool]:
        """
        Introspect a native MCP server and produce Tool wrappers for each remote tool.

        Parameters
        ----------
        name : str
            Logical server name (becomes `source` in '<type>.<source>.<name>').
            Must be stable—this is the key your plans will use.

        url_or_base : str
            Either a full MCP endpoint (endswith '/mcp') or a base URL.
            If base is provided, '/mcp' is appended.

        Returns
        -------
        list[Tool]
            One Tool per remote MCP tool. Each generated Tool is:
              type="mcp", source=<name>, name="mcp_<remote_tool_name>"

        Description format
        ------------------
        The Tool.description includes remote tool description and a compact
        "Args:" block listing schema properties and which are required.

        Invocation
        ----------
        The generated functions accept keyword arguments only (kwargs), matching
        the MCP tool's JSON schema. If a plan passes {"kwargs": {...}} it is
        unwrapped to raw keyword args for convenience.

        Notes
        -----
        - Calls are executed synchronously via `asyncio.run` to preserve
          compatibility with existing Planner/Orchestrator runners.
        - The wrapper attempts to return the first text part from MCP's result;
          if none exists, it falls back to `model_dump()` or the raw result.
        """
        if not name or not isinstance(name, str):
            raise ValueError("toolify_mcp_server requires a non-empty server name.")

        base = url_or_base.rstrip("/")
        mcp_url = base if base.endswith("/mcp") else f"{base}/mcp"

        async def _list_tools(u: str) -> List[Dict[str, Any]]:
            async with streamablehttp_client(u) as (r, w, _sid):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    resp = await session.list_tools()
                    out: List[Dict[str, Any]] = []
                    for t in resp.tools:
                        out.append(
                            {
                                "name": t.name,
                                "description": getattr(t, "description", "") or t.name,
                                "schema": getattr(t, "input_schema", None)
                                or getattr(t, "inputSchema", None)
                                or {},
                            }
                        )
                    return out

        specs = asyncio.run(_list_tools(mcp_url))

        def _schema_props(schema: Dict[str, Any]) -> Tuple[List[str], List[str]]:
            props: List[str] = []
            req: List[str] = []
            if isinstance(schema, dict):
                props = list((schema.get("properties") or {}).keys())
                req = schema.get("required") or []
            return props, req

        def _make_wrapper(u: str, tool_name: str) -> Callable[..., Any]:
            async def _acall(**payload: Any) -> Any:
                # Allow {"kwargs": {...}} shape from planners
                if "kwargs" in payload and isinstance(payload["kwargs"], dict) and len(payload) == 1:
                    payload = payload["kwargs"]

                async with streamablehttp_client(u) as (r, w, _sid):
                    async with ClientSession(r, w) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments=payload)
                        # Prefer plain text when available

                        # Fallback to dict-like form
                        try:
                            result_dict = result.model_dump()
                            if "structuredContent" in result_dict and result_dict["structuredContent"] is not None:
                                struct_cont = result_dict["structuredContent"]
                                if isinstance(struct_cont, dict) and "result" in struct_cont:
                                    return struct_cont.get("result")
                                else:
                                    return struct_cont
                            if getattr(result, "content", None):
                                texts = [getattr(c, "text", None) for c in result.content if getattr(c, "text", None)]
                                if texts:
                                    return texts[0]
                        except Exception:
                            return result

            def _sync(**payload: Any) -> Any:
                return asyncio.run(_acall(**payload))

            _sync.__name__ = f"mcp_{tool_name}"
            return _sync

        tools: List[Tool] = []
        for spec in specs:
            tool_name = spec["name"]
            desc = spec.get("description", "") or tool_name
            props, req = _schema_props(spec.get("schema", {}))
            fn = _make_wrapper(mcp_url, tool_name)

            # Include a compact arg summary in description
            if props:
                arg_lines = [f"- {p}" + (" (required)" if p in req else " (optional)") for p in props]
                arg_block = "\nArgs:\n" + "\n".join(arg_lines)
            else:
                arg_block = ""

            tools.append(
                Tool(
                    name=f"mcp_{tool_name}",
                    func=fn,
                    type="mcp",
                    source=name,
                    description=f"Calls MCP tool '{tool_name}' at {mcp_url}. {desc}{arg_block}",
                )
            )
        return tools

    # ── Polymorphic entrypoint → Tools ─────────────────────────────────────────
    @staticmethod
    def toolify(object: Any, name: Optional[str] = None, description: str = "") -> List[Tool]:
        """
        Polymorphic converter: return a list of Tools from a supported input.

        Accepted inputs
        ---------------
        - function:                returns [Tool] with type="function", source="default"
        - Agent:                   returns [Tool x4] (invoke/attach/detach/clear_memory) with type="agent"
        - Plugin-like dict:        returns [Tool xN] with type="plugin"
        - MCP server URL (str):    returns [Tool xN] with type="mcp" (requires `name`)

        Parameters
        ----------
        object : Any
            One of the accepted input types above.

        name : Optional[str]
            For MCP server URLs, this is REQUIRED and becomes the 'source' part.

        description : str
            Optional description forwarded to function-wrapped Tools.

        Raises
        ------
        ValueError
            If the input type is unsupported or if MCP URL is provided without a name.
        """
        if inspect.isfunction(object):
            return ToolFactory.toolify_function(func=object, description=description)

        if isinstance(object, Agent):
            return ToolFactory.toolify_agent(agent=object)

        if isinstance(object, dict) and "method_map" in object and "name" in object:
            # Treat as Plugin dict
            return ToolFactory.toolify_plugin(plugin=object)  # type: ignore[arg-type]

        if isinstance(object, str) and object.endswith("/mcp"):
            if not name:
                raise ValueError("toolify(object='/mcp', ...) requires a non-empty 'name' to use as source.")
            return ToolFactory.toolify_mcp_server(name=name, url_or_base=object)

        raise ValueError("Unsupported object type for toolification.")
