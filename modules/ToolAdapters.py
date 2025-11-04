# modules/ToolAdapters.py
from __future__ import annotations
# Base imports
from typing import Any, Mapping, Dict, List, Tuple, Optional
import inspect
from collections import OrderedDict
import asyncio
import threading
from queue import Queue
from collections import OrderedDict
from urllib.parse import urlparse, urlunparse
# MCP tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
# Atomic Agentic Imports
from modules.Tools import Tool, ToolDefinitionError, NO_DEFAULT
from modules.Agents import Agent

# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
__all__ = ["AgentTool", "MCPProxyTool", "toolify"]


# =========================
# Agent -> Tool (invoke)
# =========================

class AgentTool(Tool):
    """
    Wrap an Agent as a Tool.

    Metadata
    --------
    - type   = "agent"
    - source = agent.name
    - name   = "invoke"
    - description = agent.description

    Schema exposure
    ---------------
    Mirrors the agent.pre_invoke Tool's call-plan so planners see the correct keys.

    Execution
    ---------
    Accepts a flat mapping via Tool.invoke(...) and forwards it to agent.invoke(inputs).
    """

    def __init__(self, agent: Agent) -> None:
        if not isinstance(agent, Agent):
            raise ToolDefinitionError("AgentInvokeTool requires an Agents.Agent instance.")
        self.pre: Tool = agent.pre_invoke
        self.agent = agent
        if not isinstance(self.pre, Tool):
            raise ToolDefinitionError("AgentInvokeTool requires agent.pre_invoke to be a Tools.Tool.")

        # Use the pre_invoke call-plan to define our binding behavior.
        posonly_names: List[str] = list(self.pre.posonly_order)

        def _agent_wrapper(*_args: Any, **_kwargs: Any) -> Any:
            # Rebuild the flat mapping expected by Agent.invoke
            inputs: Dict[str, Any] = {}
            for i, pname in enumerate(posonly_names):
                if i < len(_args):
                    inputs[pname] = _args[i]
            inputs.update(_kwargs)
            return self.agent.invoke(inputs)

        super().__init__(
            func=_agent_wrapper,
            name="invoke",
            description=agent.description,
            type="agent",
            source=agent.name,
        )

        # Mirror pre_invoke Tool's call-plan (strict; no envelopes)
        self._arguments_map = self.pre.arguments_map
        self.posonly_order = list(self.pre.posonly_order)
        self.p_or_kw_names = list(self.pre.p_or_kw_names)
        self.kw_only_names = list(self.pre.kw_only_names)
        self.required_names = set(self.pre.required_names)
        self.has_varargs = bool(self.pre.has_varargs)
        self.varargs_name = self.pre.varargs_name
        self.has_varkw = bool(self.pre.has_varkw)
        self.varkw_name = self.pre.varkw_name



# ---------- JSON Schema -> Python annotation (best-effort) ----------

_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}

def _ann_from_json_type(t: Any, items: Any = None) -> Any:
    # anyOf/oneOf/allOf -> Any (limitation)
    if isinstance(t, list):
        return Any
    if t == "array":
        if isinstance(items, Mapping) and "type" in items:
            base = _JSON_TO_PY.get(items["type"], Any)
            try:
                return list[base]  # py3.9+ typing operator form
            except TypeError:
                return list  # fallback; purely cosmetic
        return list
    return _JSON_TO_PY.get(t, Any)


# ---------- sync runner with event-loop fallback ----------

def _run_sync(coro):
    """
    Run an async coroutine in sync context.
    If an event loop is active, run in a separate thread.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        # Likely "asyncio.run() cannot be called from a running event loop"
        if "running event loop" not in str(e):
            raise
        q: Queue = Queue()

        def _target():
            try:
                q.put(("ok", asyncio.run(coro)))
            except BaseException as exc:
                q.put(("err", exc))

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        status, payload = q.get()
        t.join()
        if status == "err":
            raise payload
        return payload


# ---------- JSON Schema -> Python annotation (best-effort) ----------
_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}

def _ann_from_json_type(t: Any, items: Any = None) -> Any:
    # anyOf/oneOf/allOf -> Any (limitation)
    if isinstance(t, list):
        return Any
    if t == "array":
        if isinstance(items, Mapping) and "type" in items:
            base = _JSON_TO_PY.get(items["type"], Any)
            try:
                return list[base]  # py3.9+ pretty form
            except TypeError:
                return list
        return list
    return _JSON_TO_PY.get(t, Any)


# ---------- sync runner with event-loop fallback ----------
def _run_sync(coro):
    """
    Run an async coroutine in sync context.
    If an event loop is active, run in a separate thread.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "running event loop" not in str(e):
            raise
        q: Queue = Queue()

        def _target():
            try:
                q.put(("ok", asyncio.run(coro)))
            except BaseException as exc:
                q.put(("err", exc))

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        status, payload = q.get()
        t.join()
        if status == "err":
            raise payload
        return payload


def _normalize_url(u: str) -> str:
    parts = urlparse(u)
    if not parts.path or parts.path == "/":
        # Most Streamable-HTTP servers mount under /mcp; append if missing.
        parts = parts._replace(path="/mcp")
    return urlunparse(parts)


def _extract_rw(transport) -> Tuple[Any, Any]:
    """
    Robustly get (read, write) from the streamable transport.

    Some SDKs return a transport object with .read/.write; others return a tuple,
    sometimes including a session_id function. We accept all.
    """
    # Object with attributes?
    r = getattr(transport, "read", None)
    w = getattr(transport, "write", None)
    if r is not None and w is not None:
        return r, w
    # Tuple / list form
    if isinstance(transport, (tuple, list)) and len(transport) >= 2:
        return transport[0], transport[1]
    raise ToolDefinitionError("MCPProxyTool: could not extract (read, write) from transport.")


class MCPProxyTool(Tool):
    """
    Strict, schema-driven proxy around a single MCP remote tool.

    Construction (sync; uses asyncio under the hood):
        MCPProxyTool(tool_name: str, server_name: str, server_url: str, headers: Optional[dict] = None)

    Metadata:
        - type   = "mcp"
        - source = server_name
        - name   = tool_name
        - description = remote description (raw)

    Binding plan:
        - Derived from remote inputSchema properties/required/default (strict, named-only)
        - Unknown keys rejected by Tool.invoke BEFORE the network call.

    Invocation:
        - Each invoke opens a short-lived client session to server_url,
          **initializes** the session, then calls `call_tool(tool_name, inputs)`,
          and returns normalized text (if present) or the raw result.
    """

    def __init__(self, tool_name: str, server_name: str, server_url: str, headers: Optional[dict] = None) -> None:
        if not isinstance(tool_name, str) or not tool_name:
            raise ToolDefinitionError("MCPProxyTool: 'tool_name' (str) is required.")
        if not isinstance(server_name, str) or not server_name:
            raise ToolDefinitionError("MCPProxyTool: 'server_name' (str) is required.")
        if not isinstance(server_url, str) or not server_url:
            raise ToolDefinitionError("MCPProxyTool: 'server_url' (str) is required.")

        self._server_url = _normalize_url(server_url)
        self._headers = dict(headers or {})

        # 1) Fetch remote description + input schema synchronously (with proper initialize())
        description, input_schema = _run_sync(self._fetch_remote_tool_meta(self._server_url, tool_name, self._headers))

        # 2) Build strict call plan from schema
        arguments_map, p_or_kw_names, required_names = self._build_from_schema(input_schema)

        # 3) Build the wrapper that will call the remote tool synchronously
        def _mcp_wrapper(**inputs: Any) -> Any:
            result = _run_sync(self._call_remote(self._server_url, tool_name, inputs, self._headers))
            result = result.model_dump()
            import json
            if "structuredContent" in result:
                if "result" in result["structuredContent"]: return result["structuredContent"]["result"]
                else: return result["structuredContent"]
            else:
                try:
                    content = result["content"]
                    texts = [c["text"] for c in content]
                    return "".join(texts)
                except:
                    pass
                return result
            # # Prefer text content if present (typical MCP result shape)
            # try:
            #     content = getattr(result, "content", None)
            #     if isinstance(content, list) and content:
            #         texts = [getattr(c, "text", "") for c in content if hasattr(c, "text")]
            #         joined = "".join(texts).strip()
            #         if joined:
            #             return joined
            # except Exception:
            #     pass
            # return result

        # 4) Initialize base Tool, then override the call plan
        super().__init__(
            func=_mcp_wrapper,
            name=tool_name,
            description=description or "",
            type="mcp",
            source=server_name,
        )

        # Apply strict plan (named-only)
        self._arguments_map = arguments_map
        self.posonly_order = []
        self.p_or_kw_names = p_or_kw_names
        self.kw_only_names = []
        self.required_names = required_names
        self.has_varargs = False
        self.varargs_name = None
        self.has_varkw = False
        self.varkw_name = None

    # ----- async helpers -----

    @staticmethod
    async def _fetch_remote_tool_meta(server_url: str, tool_name: str, headers: dict) -> Tuple[str, Mapping[str, Any]]:
        """
        Connect to the MCP server, initialize the session, ensure the tool exists,
        and return (description, input_schema).
        """
        async with streamablehttp_client(url=server_url, headers=headers or None) as transport:
            read, write = _extract_rw(transport)
            async with ClientSession(read, write) as session:
                # REQUIRED handshake before any request.
                await session.initialize()  # Establish connection/session. :contentReference[oaicite:5]{index=5}
                tools = await session.list_tools()   # Discover available tools. :contentReference[oaicite:6]{index=6}
                tool_list = getattr(tools, "tools", tools)

                names = []
                target = None
                for t in tool_list:
                    nm = getattr(t, "name", None)
                    if nm:
                        names.append(nm)
                    if nm == tool_name:
                        target = t
                        break
                if target is None:
                    raise ToolDefinitionError(
                        f"MCPProxyTool: tool '{tool_name}' not found on server; available: {sorted(names)}"
                    )

                schema = getattr(target, "input_schema", None) or getattr(target, "inputSchema", None)
                if not isinstance(schema, Mapping):
                    raise ToolDefinitionError(
                        f"MCPProxyTool: tool '{tool_name}' has no valid input schema; got: {type(schema).__name__}"
                    )
                desc = getattr(target, "description", "") or ""
                return desc, schema

    @staticmethod
    async def _call_remote(server_url: str, tool_name: str, inputs: Mapping[str, Any], headers: dict) -> Any:
        """
        Execute the remote tool and return server result (text normalized if available).
        """
        async with streamablehttp_client(url=server_url, headers=headers or None) as transport:
            read, write = _extract_rw(transport)
            async with ClientSession(read, write) as session:
                await session.initialize()  # REQUIRED before call_tool. :contentReference[oaicite:7]{index=7}
                return await session.call_tool(tool_name, dict(inputs))  # Execute. :contentReference[oaicite:8]{index=8}

    # ----- schema-to-plan -----

    @staticmethod
    def _build_from_schema(schema: Mapping[str, Any]) -> Tuple["OrderedDict[str, Dict[str, Any]]", List[str], set]:
        """
        Convert a JSON Schema object into (arguments_map, p_or_kw_names, required_names).
        Supported:
          - 'properties' (ordered), optional 'required'
          - per-property: 'type', optional 'items.type' for arrays, optional 'default'
        Complex constructs (anyOf/oneOf/allOf/$ref) -> annotate as 'Any' but still
        respect 'required' and 'default'.
        """
        props = schema.get("properties")
        if not isinstance(props, Mapping):
            raise ToolDefinitionError("MCPProxyTool: input_schema.properties must be a mapping.")
        required = schema.get("required") or []
        if not isinstance(required, (list, tuple)):
            raise ToolDefinitionError("MCPProxyTool: input_schema.required must be a list if present.")

        arguments_map: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        p_or_kw_names: List[str] = []
        required_names: set = set()

        for idx, (pname, pspec) in enumerate(props.items()):
            if not isinstance(pspec, Mapping):
                raise ToolDefinitionError(f"MCPProxyTool: property '{pname}' must be a mapping.")

            ptype = pspec.get("type")
            ann = _ann_from_json_type(ptype, pspec.get("items"))

            has_default = "default" in pspec
            default_val = pspec.get("default", NO_DEFAULT)

            arguments_map[pname] = {
                "index": idx,
                "kind": inspect.Parameter.POSITIONAL_OR_KEYWORD,
                "ann": ann,
                "has_default": bool(has_default),
                "default": default_val if has_default else NO_DEFAULT,
            }
            p_or_kw_names.append(pname)

        for n in required:
            if n in arguments_map:
                required_names.add(n)

        return arguments_map, p_or_kw_names, required_names


# ---------- URL helpers ----------
def _is_http_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def _normalize_url(u: str) -> str:
    p = urlparse(u)
    if not p.path or p.path == "/":
        p = p._replace(path="/mcp")
    return urlunparse(p)


# ---------- MCP discovery (list tool names) ----------
async def _async_discover_mcp_tool_names(url: str, headers: Optional[dict]) -> List[str]:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except Exception as e:
        raise ToolDefinitionError(
            "ToolFactory: MCP client libraries not available. Install the MCP client or remove the MCP URL input."
        ) from e

    async with streamablehttp_client(url=url, headers=headers or None) as transport:
        # robustly extract (read, write) for both transport-object and tuple forms
        read = getattr(transport, "read", None)
        write = getattr(transport, "write", None)
        if read is None or write is None:
            read, write = transport[0], transport[1]

        async with ClientSession(read, write) as session:
            # REQUIRED handshake
            await session.initialize()
            tools_resp = await session.list_tools()
            tool_objs = getattr(tools_resp, "tools", tools_resp)
            return [t.name for t in tool_objs]

def _discover_mcp_tool_names(url: str, headers: Optional[dict]) -> List[str]:
    return _run_sync(_async_discover_mcp_tool_names(url, headers))


# ---------- filter helpers ----------
def _filter_names(names: List[str],
                  include: Optional[List[str]],
                  exclude: Optional[List[str]]) -> List[str]:
    s = list(names)
    if include:
        inc = set(include)
        s = [n for n in s if n in inc]
    if exclude:
        exc = set(exclude)
        s = [n for n in s if n not in exc]
    return s


# ---------- Public API ----------

def toolify(
    obj: Any,
    *,
    # callable-specific
    name: Optional[str] = None,
    description: Optional[str] = None,
    tool_type: str = "python",
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
        names = _filter_names(names, include, exclude)
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
        if not description or not isinstance(description, str):
            raise ToolDefinitionError("ToolFactory: 'description' (str) is required for callables.")
        return [
            Tool(
                func=obj,
                name=name,
                description=description,
                type=tool_type,
                source=source or "local",
            )
        ]

    # 5) Unsupported
    raise ToolDefinitionError(
        "ToolFactory: unsupported input. Expected Tool | Agent | callable | MCP URL string."
    )
