# Tools.py
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, get_type_hints, get_origin, get_args, TypedDict
from collections import OrderedDict

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
import inspect
from collections import OrderedDict
from typing import Any, Callable, Optional, get_args, get_origin, get_type_hints


class Tool:
    """
    Wrapper around a callable providing:
      • Stable identity (type, source, name → full_name)
      • Human-readable signature string (for prompts/UI)
      • Optional memory clear hook
      • Simple invoke() dispatcher

    Internal detail (compat preserved):
      • Structured signature is stored as:
          _sig_types: OrderedDict[str, type]
          _return_type: type
        .signature (str) is rendered from these.

    Conventions:
      • Full name IS type-prefixed: '{type}.{source}.{name}'
      • Drop conventional receiver params ('self'/'cls') from signatures
      • Render *args/**kwargs as '*args: Any' / '**kwargs: Any'
    """

    # ----------------------------
    # Construction & Introspection
    # ----------------------------

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
        self._clear_mem: Callable[[], None] = clear_mem_func

        # Structured signature storage (authoritative)
        self._sig_types: OrderedDict[str, Any] = OrderedDict()
        self._sig_defaults: OrderedDict[str, Any] = OrderedDict()
        self._return_type: Any = Any
        self._build_signature_map(func)

    # ----------------------------
    # Identity & Metadata
    # ----------------------------

    @property
    def type(self) -> str:
        """Logical tool category (e.g., 'function', 'plugin', 'api')."""
        return self._type

    @property
    def source(self) -> str:
        """Namespace/owner; contributes to fully-qualified name."""
        return self._source

    @property
    def name(self) -> str:
        """Unqualified tool name."""
        return self._name

    @property
    def full_name(self) -> str:
        """Fully-qualified, type-prefixed name: '{type}.{source}.{name}'."""
        return f"{self._type}.{self._source}.{self._name}"

    # ----------------------------
    # Description & Signature
    # ----------------------------

    @property
    def description(self) -> str:
        """
        Human-readable description prefixed with the formatted signature.
        Example:
            "function.plugin_math.add(a: int, b: int) → int: Add two integers."
        """
        return f"{self.signature}: {self._description}" if self._description else self.signature

    @description.setter
    def description(self, val: str) -> None:
        self._description = val

    @property
    def signature(self) -> str:
        """Human-readable signature string for prompts/UI."""
        return self._format_signature(self.full_name, self._sig_types, self._return_type, self._sig_defaults)

    # Optional: expose structured signature read-only (useful for tooling).
    @property
    def signature_map(self) -> OrderedDict[str, Any]:
        """Ordered mapping of parameter names to their (annotation) types."""
        return OrderedDict(self._sig_types)

    @property
    def return_type(self) -> Any:
        """Return annotation/type if present, else `typing.Any`."""
        return self._return_type

    @property
    def default_map(self) -> "OrderedDict[str, Any]":
        """Ordered mapping of parameter names to their default values (if any)."""
        return OrderedDict(self._sig_defaults)

    # ----------------------------
    # Execution
    # ----------------------------

    @property
    def func(self) -> Callable[..., Any]:
        """Underlying callable."""
        return self._func

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Call the underlying function."""
        return self._func(*args, **kwargs)

    def clear_memory(self) -> None:
        """Invoke the optional clear-memory hook, if provided."""
        if self._clear_mem is not None:
            self._clear_mem()

    # ----------------------------
    # Dunder
    # ----------------------------

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Tool<{self.signature}>"

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _build_signature_map(self, func: Callable[..., Any]) -> None:
        """
        Populate `_sig_types` (param -> annotation), `_sig_defaults` (param -> default),
        and `_return_type`. Drops conventional receiver params.
        """
        sig = inspect.signature(func)
        hints = get_type_hints(func)

        params = list(sig.parameters.items())
        for idx, (name, p) in enumerate(params):
            # Drop conventional receiver ('self'/'cls') if present as first positional
            if (
                idx == 0
                and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and name in {"self", "cls"}
            ):
                continue

            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                self._sig_types["*args"] = Any
                # no default for var-positional
            elif p.kind == inspect.Parameter.VAR_KEYWORD:
                self._sig_types["**kwargs"] = Any
                # no default for var-keyword
            else:
                self._sig_types[name] = hints.get(name, Any)
                if p.default is not inspect._empty:
                    self._sig_defaults[name] = p.default  # record default

        self._return_type = hints.get("return", Any)

    @staticmethod
    def _format_signature(
        key: str,
        sig_types: OrderedDict[str, Any],
        rtype: Any,
        sig_defaults: OrderedDict[str, Any] | None = None,   # <-- NEW param
    ) -> str:
        """
        Render the structured signature to a concise single line:
            "{key}(a: int = 3, b: list[str] = None) → bool"
        """

        def _ann_name(t: Any) -> str:
            origin = get_origin(t)
            if origin is None:
                # Builtins/typing.Any/ForwardRef fallback
                return getattr(t, "__name__", str(t))
            args = get_args(t)
            base = getattr(origin, "__name__", str(origin))
            if args:
                inner = ", ".join(_ann_name(a) for a in args)
                return f"{base}[{inner}]"
            return base

        def _default_repr(v: Any) -> str:
            # readable, bounded-length repr for prompts
            if isinstance(v, str):
                s = v if len(v) <= 40 else (v[:37] + "…")
                return repr(s)
            if v is None or isinstance(v, (int, float, bool)):
                return repr(v)
            r = repr(v)
            return r if len(r) <= 40 else (r[:37] + "…")

        sig_defaults = sig_defaults or OrderedDict()
        parts = []
        for n, ann in sig_types.items():
            piece = f"{n}: {_ann_name(ann)}"
            if n in sig_defaults:
                piece += f" = {_default_repr(sig_defaults[n])}"
            parts.append(piece)

        params_str = ", ".join(parts)
        rtype_str = _ann_name(rtype)
        return f"{key}({params_str}) \u2192 {rtype_str}"


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
