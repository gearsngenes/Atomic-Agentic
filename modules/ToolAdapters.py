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
from modules.Tools import Tool, ToolDefinitionError, NO_DEFAULT
from modules.Agents import Agent

# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
__all__ = ["AgentTool", "MCPProxyTool", "toolify"]

# ───────────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────────

def _run_sync(coro):
    """
    Run an async coroutine in sync context.
    If an event loop is active, run in a separate thread.

    Ref: asyncio loop rules; json & inspect behaviors require JSON-safe payloads and
    Python-only enums should not be sent over the wire. :contentReference[oaicite:2]{index=2}
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
    """
    Normalize Streamable-HTTP MCP URLs to include a mount path if missing.
    MCP tools are JSON-schema driven (name/description/inputSchema). :contentReference[oaicite:3]{index=3}
    """
    parts = urlparse(u)
    if not parts.path or parts.path == "/":
        parts = parts._replace(path="/mcp")
    return urlunparse(parts)


def _extract_rw(transport) -> Tuple[Any, Any]:
    """
    Robustly get (read, write) from the streamable transport.

    Some SDKs return a transport object with .read/.write; others return a tuple.
    """
    r = getattr(transport, "read", None)
    w = getattr(transport, "write", None)
    if r is not None and w is not None:
        return r, w
    if isinstance(transport, (tuple, list)) and len(transport) >= 2:
        return transport[0], transport[1]
    raise ToolDefinitionError("MCPProxyTool: could not extract (read, write) from transport.")


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

def _ann_from_json_type(t: Any, items: Any = None) -> Any:
    """
    Map JSON Schema 'type' (+ 'items' for arrays) into a Python typing-ish object.
    This is only a staging shape; we will turn it into a canonical string below.
    """
    # anyOf/oneOf/allOf -> Any (limitation)
    if isinstance(t, list):
        return Any
    if t == "array":
        if isinstance(items, Mapping) and "type" in items:
            base = _JSON_TO_PY.get(items["type"], Any)
            try:
                return list[base]  # py3.9+ generic alias
            except TypeError:
                return list
        return list
    return _JSON_TO_PY.get(t, Any)

def _canon_str_from_python(t: Any) -> str:
    """
    Convert a Python/typing annotation-ish object to a canonical string
    that is stable and JSON/wire-friendly.

    We keep this intentionally small to avoid importing private internals:
    list[T], dict[K, V], tuple[...], union (X|Y) → 'union[...]', NoneType → 'none',
    Any → 'any', classes → lower-cased __name__.
    """
    # Textual input
    if isinstance(t, str):
        return t.strip().lower() or "object"

    # None / Any
    if t in (type(None), None):
        return "none"
    if t is Any:
        return "any"

    # PEP 604 union (X|Y) or typing.Union
    origin = get_origin(t)
    if origin in (getattr(__import__('typing'), 'Union', None), getattr(__import__('types'), 'UnionType', None)):
        args = [ _canon_str_from_python(a) for a in (get_args(t) or ()) ]
        args = sorted(args)
        non_none = [a for a in args if a != "none"]
        if len(args) == 2 and len(non_none) == 1 and "none" in args:
            return f"optional[{non_none[0]}]"
        return f"union[{', '.join(args)}]"

    # Containers
    if origin is list:
        (arg,) = get_args(t) or (Any,)
        return f"list[{_canon_str_from_python(arg)}]"
    if origin is dict:
        k, v = (get_args(t) + (Any, Any))[:2]
        return f"dict[{_canon_str_from_python(k)}, {_canon_str_from_python(v)}]"
    if origin is tuple:
        args = get_args(t) or ()
        inner = ", ".join(_canon_str_from_python(a) for a in args) if args else ""
        return f"tuple[{inner}]" if inner else "tuple"

    # Bare Python classes
    if isinstance(t, type):
        return t.__name__.lower()

    # Fallback
    return "object"


# ---------- URL helpers ----------
def _is_http_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


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
# MCP Proxy Tool
# ───────────────────────────────────────────────────────────────────────────────
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

    Signature exposure:
        - After mirroring the schema into `arguments_map`, we synthesize an
          plan so `to_dict()["signature"]` shows `(a: str, b?: int)` rather than `(**inputs)`.

    Invocation:
        - Each `invoke` opens a short-lived client session to `server_url`,
          **initializes** the session, calls `call_tool(tool_name, inputs)`,
          and returns normalized text (if present) or the raw result.

    MCP tools are JSON-schema driven and expect JSON-safe payloads. :contentReference[oaicite:4]{index=4}
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
        description, input_schema = _run_sync(
            self._fetch_remote_tool_meta(self._server_url, tool_name, self._headers)
        )

        # 2) Build strict call plan from schema (named-only)
        arguments_map, p_or_kw_names, required_names = self._build_from_schema(input_schema)

        # 3) Wrapper that calls the MCP tool synchronously
        def _mcp_wrapper(**inputs: Any) -> Any:
            result = _run_sync(self._call_remote(self._server_url, tool_name, inputs, self._headers))
            # The client session commonly returns a dataclass-like object with .model_dump() (pydantic) or dict
            try:
                if hasattr(result, "model_dump"):
                    result = result.model_dump()  # pydantic v2 dict :contentReference[oaicite:5]{index=5}
            except Exception:
                pass

            # Try to return concise structured value if server provided one
            try:
                if isinstance(result, Mapping) and "structuredContent" in result:
                    sc = result["structuredContent"]
                    if isinstance(sc, dict) and "result" in sc and len(sc) == 1:
                        return sc["result"]
                    return sc
                # Fallback to concatenated text content
                if isinstance(result, Mapping):
                    content = result.get("content", []) or []
                    texts = [c["text"] for c in content if isinstance(c, dict) and "text" in c]
                    if texts:
                        return "".join(texts)
            except Exception:
                pass
            return result

        # 4) Initialize base Tool, then override the call-plan with our schema
        super().__init__(
            func=_mcp_wrapper,
            name=tool_name,
            description=description or "",
            type="mcp",
            source=server_name,
        )

        # ---- Mirror schema-driven plan (strict named-only) ----
        self._arguments_map = OrderedDict((k, dict(v)) for k, v in arguments_map.items())
        self.posonly_order = []
        self.p_or_kw_names = list(p_or_kw_names)
        self.kw_only_names = []
        self.required_names = set(required_names)
        self.has_varargs = False
        self.varargs_name = None
        self.has_varkw = False
        self.varkw_name = None
        self._rebuild_signature_str()

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
                # REQUIRED handshake before any request
                await session.initialize()

                tools = await session.list_tools()
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
                await session.initialize()
                return await session.call_tool(tool_name, inputs)

    # ----- schema → plan -----

    @staticmethod
    def _build_from_schema(schema: Mapping[str, Any]) -> Tuple["OrderedDict[str, Dict[str, Any]]", List[str], set]:
        """
        Build an arguments_map and binding lists from a JSON Schema object.
        Only 'properties', 'required', and 'default' are honored.

        Returns:
            (arguments_map, p_or_kw_names, required_names)
        """
        props = schema.get("properties") or {}
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
            ann_py = _ann_from_json_type(ptype, pspec.get("items"))
            ann_str = _canon_str_from_python(ann_py)

            has_default = "default" in pspec

            entry: Dict[str, Any] = {
                "index": idx,
                "kind_name": inspect.Parameter.POSITIONAL_OR_KEYWORD.name,
                "mode": "pos_or_kw",  # JSON/wire safe token
                "ann": ann_str,       # canonical string
                "has_default": bool(has_default),
            }
            if has_default:
                # JSON Schema defaults must already be JSON-encodable
                entry["default"] = pspec["default"]

            arguments_map[pname] = entry
            p_or_kw_names.append(pname)

        for n in required:
            if n in arguments_map:
                required_names.add(n)

        return arguments_map, p_or_kw_names, required_names


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
        if not description or not isinstance(description, str):
            raise ToolDefinitionError("ToolFactory: 'description' (str) is required for callables.")
        return [
            Tool(
                func=obj,
                name=name,
                description=description,
                type="function",
                source=source or "default",
            )
        ]

    # 5) Unsupported
    raise ToolDefinitionError(
        "ToolFactory: unsupported input. Expected Tool | Agent | callable | MCP URL string."
    )
