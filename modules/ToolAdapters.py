# modules/ToolAdapters.py
from __future__ import annotations

# Base imports
from typing import Any, Mapping, Dict, List, Tuple, Optional, get_origin, get_args
import inspect
import asyncio
import threading
from queue import Queue
from collections import OrderedDict
from urllib.parse import urlparse, urlunparse

# MCP tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Atomic Agentic Imports
from modules.Tools import Tool, ToolDefinitionError
from modules.Agents import Agent

# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
__all__ = ["AgentTool", "MCPProxyTool", "toolify"]


# ---------- JSON Schema -> Python type (best-effort) ----------
_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


# ---------- URL helpers ----------
def _is_http_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


# ───────────────────────────────────────────────────────────────────────────────
# Agent -> Tool (invoke)
# ───────────────────────────────────────────────────────────────────────────────
class AgentTool(Tool):
    """
    Adapter that exposes an Agent as a Tool with schema-driven introspection.

    Metadata
    --------
    - type   = "agent"
    - source = agent.name
    - name   = "invoke"
    - description = agent.description

    Schema exposure
    ---------------
    Mirrors the agent's `pre_invoke` Tool call-plan (arguments map, required sets,
    varargs flags, etc.) so planners see the exact input keys & binding rules.

    Signature string
    ----------------
    The base Tool builds a canonical, schema-derived signature of the form:
        "agent.<agent_name>.invoke(p1:Type, p2?:Type) -> str"
    We set return_type to "str" and call `_rebuild_signature_str()` after
    overriding the plan to reflect the agent’s actual output.
    """

    def __init__(self, agent: Agent) -> None:
        if not isinstance(agent, Agent):
            raise ToolDefinitionError("AgentTool requires an Agents.Agent instance.")

        pre = agent.pre_invoke
        if not isinstance(pre, Tool):
            raise ToolDefinitionError("AgentTool requires agent.pre_invoke to be a Tools.Tool.")

        self.agent = agent
        self.pre = pre

        posonly_names: List[str] = list(pre.posonly_order)

        def _agent_wrapper(*_args: Any, **_kwargs: Any) -> Any:
            inputs: Dict[str, Any] = {}
            # Map positional-only prefix back into named keys deterministically
            for i, pname in enumerate(posonly_names):
                if i < len(_args):
                    inputs[pname] = _args[i]
            inputs.update(_kwargs)
            return self.agent.invoke(inputs)

        # Initialize base Tool with wrapper and agent metadata
        super().__init__(
            func=_agent_wrapper,
            name="invoke",
            description=agent.description,
            type="agent",
            source=agent.name,
        )

        # ---- Mirror the pre_invoke call-plan (shallow copies to avoid aliasing) ----
        self._arguments_map = pre.arguments_map
        self.posonly_order = list(pre.posonly_order)
        self.p_or_kw_names = list(pre.p_or_kw_names)
        self.kw_only_names = list(pre.kw_only_names)
        self.required_names = set(pre.required_names)
        self.has_varargs = bool(pre.has_varargs)
        self.varargs_name = pre.varargs_name
        self.has_varkw = bool(pre.has_varkw)
        self.varkw_name = pre.varkw_name

        # Explicit return type for AgentTool (agent responses are strings)
        self._return_type = "str"
        self._rebuild_signature_str()


# ───────────────────────────────────────────────────────────────────────────────
# Helpers (self-contained; ensure there is ONLY ONE definition of each)
# ───────────────────────────────────────────────────────────────────────────────
def _run_sync(coro):
    """
    Run a coroutine synchronously with robust loop handling:
      • If no loop or the current loop is CLOSED → create a fresh loop, run, shutdown, close.
      • If a loop is RUNNING → execute in a worker thread via asyncio.run().
      • Else → run_until_complete on the existing idle loop.

    This avoids 'Event loop is closed' after earlier asyncio.run(...) usage.
    """
    import asyncio
    from queue import Queue
    import threading

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    # No loop or closed loop → make a temporary one
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    # Loop is running in this thread → run in a worker thread
    if loop.is_running():
        q: "Queue[Tuple[str, Any]]" = Queue()

        def _worker():
            try:
                q.put(("ok", asyncio.run(coro)))
            except BaseException as exc:  # surface original exception to caller
                q.put(("err", exc))

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        status, payload = q.get()
        t.join()
        if status == "err":
            raise payload
        return payload

    # Loop exists and is idle → run directly
    return loop.run_until_complete(coro)


def _normalize_url(u: str) -> str:
    """
    Ensure a valid MCP streamable-HTTP URL. If no path is provided, default to '/mcp'.
    """
    from urllib.parse import urlparse, urlunparse
    parts = urlparse(u.strip())
    if not parts.scheme or not parts.netloc:
        raise ToolDefinitionError(f"Invalid MCP URL: {u!r}")
    if not parts.path or parts.path == "/":
        parts = parts._replace(path="/mcp")
    return urlunparse(parts)


def _extract_rw(transport) -> Tuple[Any, Any]:
    """
    Obtain (read, write) callables from the transport. Some SDK variants return
    a tuple; others expose .read/.write attributes.
    """
    r = getattr(transport, "read", None)
    w = getattr(transport, "write", None)
    if r is not None and w is not None:
        return r, w
    if isinstance(transport, (tuple, list)) and len(transport) >= 2:
        return transport[0], transport[1]
    raise ToolDefinitionError("MCPProxyTool: could not extract (read, write) from transport.")


def _extract_structured_or_text(result: Any) -> Optional[Any]:
    """
    OLD behavior, restored and streamlined:

    1) If `structuredContent` exists, return it.
       • If it's exactly {'result': X}, unwrap to X.
    2) Else if `content` exists, concatenate its 'text' blocks.
    3) Else return None (caller can fall back to model_dump()/raw).
    """
    # Prefer structuredContent (attribute or mapping)
    sc = getattr(result, "structuredContent", None)
    if sc is None and isinstance(result, dict):
        sc = result.get("structuredContent")
    if sc is not None:
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc

    # Fallback to unstructured content blocks (join text)
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if isinstance(content, list):
        texts = [p.get("text") for p in content if isinstance(p, dict) and "text" in p]
        if texts:
            return "".join(texts)

    return None


def _to_plain(result: Any) -> Any:
    """Last resort: pydantic v2 models → dict; otherwise pass result through."""
    try:
        if hasattr(result, "model_dump") and callable(result.model_dump):
            return result.model_dump()
    except Exception:
        pass
    return result

# ---------- MCP discovery (list tool names) ----------
async def _async_discover_mcp_tool_names(url: str, headers: Optional[dict]) -> List[str]:
    async with streamablehttp_client(url=url, headers=headers or None) as transport:
        read, write = _extract_rw(transport)
        async with ClientSession(read, write) as session:
            # REQUIRED handshake
            await session.initialize()
            tools_resp = await session.list_tools()
            tool_objs = getattr(tools_resp, "tools", tools_resp)
            return [t.name for t in tool_objs]

def _discover_mcp_tool_names(url: str, headers: Optional[dict]) -> List[str]:
    return _run_sync(_async_discover_mcp_tool_names(url, headers))


# ───────────────────────────────────────────────────────────────────────────────
# MCP Proxy Tool
# ───────────────────────────────────────────────────────────────────────────────
class MCPProxyTool(Tool):
    """
    Proxy a single MCP server tool as a normal dict-first Tool.

    • __init__: open short-lived session, `initialize`, `list_tools`, extract the
      tool’s `inputSchema` + description, close. Build a *keyword-only* signature
      in server property order (for display only; JSON objects are unordered by spec).

    • invoke(): open short-lived session, `initialize`, `call_tool`, then return
      **structured content** (or joined text fallback), close. No background loop,
      so scripts terminate cleanly.
    """

    def __init__(
        self,
        *,
        server_url: str,
        server_name: str,
        tool_name: str,
        headers: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        self._server_url = _normalize_url(server_url)
        self._server_name = str(server_name).strip()
        self._tool_name = str(tool_name).strip()
        self._headers = dict(headers or {})

        if not self._server_name:
            raise ToolDefinitionError("MCPProxyTool: 'server_name' cannot be empty.")
        if not self._tool_name:
            raise ToolDefinitionError("MCPProxyTool: 'tool_name' cannot be empty.")

        # 1) Discover schema (short-lived session)
        params_spec, required_names, remote_desc = self._discover_schema()

        # 2) Build a KW-ONLY wrapper whose signature mirrors the remote schema
        def _wrapper(**inputs: Any) -> Any:
            # Base Tool.invoke has already validated keys/requireds (unknown keys rejected).
            return self._call_remote_once(inputs)

        _wrapper.__name__ = self._tool_name
        _wrapper.__doc__ = description or remote_desc or f"MCP proxy to {self._server_name}.{self._tool_name}"

        parameters: List[inspect.Parameter] = []
        for p in params_spec:
            default = inspect._empty if not p["has_default"] else p["default"]
            parameters.append(
                inspect.Parameter(
                    name=p["name"],
                    kind=inspect.Parameter.KEYWORD_ONLY,  # MCP uses a single JSON object (keyword-only)
                    default=default,
                    annotation=p["py_type"],
                )
            )
        _wrapper.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            parameters=tuple(parameters),
            return_annotation=Any,
        )

        # 3) Initialize as a normal Tool (base builds arguments_map/signature string)
        super().__init__(
            func=_wrapper,
            name=self._tool_name,
            description=_wrapper.__doc__ or "",
            type="mcp_tool",
            source=self._server_name,
        )

        # Enforce named-only semantics (no varargs/kwargs)
        self.posonly_order = []
        self.has_varargs = False
        self.varargs_name = None
        self.has_varkw = False
        self.varkw_name = None
        self.required_names = set(required_names)
        self._return_type = "any"  # MCP doesn’t guarantee static return types

    # ── schema discovery (short-lived session) ──────────────────────────────────
    def _discover_schema(self) -> Tuple[List[Dict[str, Any]], List[str], str]:
        """
        Connect → initialize → list_tools → extract tool → close.

        Returns:
          - params_spec: list of {"name","py_type","has_default","default"} in server property order
          - required_names: list of required property names
          - description: tool description string (or "")
        """
        async def _fetch():
            async with streamablehttp_client(url=self._server_url, headers=self._headers or None) as transport:
                read, write = _extract_rw(transport)
                async with ClientSession(read, write) as sess:
                    await sess.initialize()
                    tools_resp = await sess.list_tools()
            return tools_resp

        tools_resp = _run_sync(_fetch())
        tools = getattr(tools_resp, "tools", tools_resp)

        target = None
        for t in tools:
            nm = getattr(t, "name", None) or (isinstance(t, dict) and t.get("name"))
            if nm == self._tool_name:
                target = t
                break
        if target is None:
            names = [getattr(t, "name", None) or (isinstance(t, dict) and t.get("name")) for t in tools]
            raise ToolDefinitionError(
                f"MCP tool '{self._tool_name}' not found on server '{self._server_name}' @ {self._server_url}; "
                f"available: {sorted(n for n in names if n)}"
            )

        desc = getattr(target, "description", None) or (isinstance(target, dict) and target.get("description")) or ""

        schema = (
            getattr(target, "inputSchema", None)
            or (isinstance(target, dict) and target.get("inputSchema"))
            or {}
        )
        props: Mapping[str, Any] = schema.get("properties") or {}
        required_list: List[str] = schema.get("required") or []

        params_spec: List[Dict[str, Any]] = []
        # Preserve server-reported property order for display/signature only (JSON objects are unordered by spec).
        for name, meta in (props.items() if isinstance(props, Mapping) else []):
            meta = meta or {}
            py_type = _JSON_TO_PY.get(meta.get("type"), Any)
            has_default = "default" in meta
            default = meta.get("default", inspect._empty)
            params_spec.append(
                {"name": name, "py_type": py_type, "has_default": has_default, "default": default}
            )

        return params_spec, list(required_list), desc

    # ── one-shot invoke (short-lived session) ───────────────────────────────────
    def _call_remote_once(self, inputs: Mapping[str, Any]) -> Any:
        """
        initialize → call_tool → extract structured/text → close.
        """
        async def _do():
            async with streamablehttp_client(url=self._server_url, headers=self._headers or None) as transport:
                read, write = _extract_rw(transport)
                async with ClientSession(read, write) as sess:
                    await sess.initialize()
                    return await sess.call_tool(self._tool_name, dict(inputs))

        raw = _run_sync(_do())

        # Structured-first, then text fallback, else model_dump/raw.
        val = _extract_structured_or_text(raw)
        if val is not None:
            return val
        return _to_plain(raw)


# ───────────────────────────────────────────────────────────────────────────────
# Public factory
# ───────────────────────────────────────────────────────────────────────────────
def toolify(
    obj: Any,
    *,
    # callable-specific
    name: Optional[str] = None,
    description: Optional[str] = None,
    source: Optional[str] = None,
    # MCP-specific (obj is an MCP URL)
    server_name: Optional[str] = None,
    headers: Optional[dict] = None,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    fail_if_empty: bool = True,
) -> List[Tool]:
    """
    Create packaged Tools from one of: Tool | Agent | callable | MCP URL (str).

    Returns:
        list[Tool]

    Raises:
        ToolDefinitionError for invalid inputs or unmet requirements.

    Notes:
        - MCP metadata and schemas are JSON-defined; we discover tools then proxy them. :contentReference[oaicite:6]{index=6}
        - JSON wire formats must be JSON-encodable (no raw Python-only objects). :contentReference[oaicite:7]{index=7}
    """

    # 1) Passthrough if already a Tool
    if isinstance(obj, Tool):
        return [obj]

    # 2) Agent -> AgentTool
    if isinstance(obj, Agent):
        return [AgentTool(obj)]

    # 3) MCP URL -> list of MCPProxyTool
    if isinstance(obj, str) and _is_http_url(obj):
        if not server_name:
            raise ToolDefinitionError("ToolFactory: 'server_name' is required when toolifying an MCP URL.")
        url = _normalize_url(obj)
        names = _discover_mcp_tool_names(url, headers)
        if include:
            inc = set(include)
            names = [n for n in names if n in inc]
        if exclude:
            exc = set(exclude)
            names = [n for n in names if n not in exc]
        if not names and fail_if_empty:
            raise ToolDefinitionError("ToolFactory: no MCP tools found after applying filters.")
        return [
            MCPProxyTool(tool_name=n, server_name=server_name, server_url=url, headers=headers)
            for n in names
        ]

    # 4) callable -> Tool
    if callable(obj):
        if not name or not isinstance(name, str):
            raise ToolDefinitionError("ToolFactory: 'name' (str) is required for callables.")
        if description is not None and not isinstance(description, str):
            raise ToolDefinitionError("ToolFactory: 'description' expects a string value for callables.")
        return [
            Tool(
                func=obj,
                name=name,
                description= (description or obj.__doc__) or "",
                type="function",
                source=source or "default",
            )
        ]

    # 5) Unsupported
    raise ToolDefinitionError(
        "ToolFactory: unsupported input. Expected Tool | Agent | callable | MCP URL string."
    )
