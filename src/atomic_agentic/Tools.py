# Tools.py
from __future__ import annotations
import inspect
from collections import OrderedDict 
import inspect
from collections import OrderedDict
from typing import (
    Any,
    Mapping,
    Callable,
    List,
    Optional,
    Tuple,
    Dict
)
import logging
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .Exceptions import ToolDefinitionError, ToolInvocationError
from .Primitives import Tool

from .__utils__ import (
    _canonize_annotation,
    _jsonify_default,
    _run_sync,
    _extract_rw,
    _extract_structured_or_text,
    _to_plain,
    _normalize_url,
    _JSON_TO_PY,
    KIND_TO_MODE,
)

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
__all__ = ["Tool", "MCPProxyTool"]

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
        server_url: str,
        server_name: str,
        tool_name: str,
        headers: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> None:
        self._server_url = _normalize_url(server_url)
        self._source = str(server_name).strip()
        self._name = str(tool_name).strip()
        self._headers = dict(headers or {})
        self.module = None
        self.qualname = None

        if not self._source:
            raise ToolDefinitionError("MCPProxyTool: 'server_name' cannot be empty.")
        if not self._name:
            raise ToolDefinitionError("MCPProxyTool: 'tool_name' cannot be empty.")

        # 1) Discover schema (short-lived session)
        params_spec, required_names, remote_desc = self._discover_schema()
        self._description = (description or remote_desc) or ""

        # 2) Build a KW-ONLY wrapper whose signature mirrors the remote schema
        def _wrapper(**inputs: Any) -> Any:
            # Base Tool.invoke has already validated keys/requireds (unknown keys rejected).
            return self._call_remote_once(inputs)

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
        self._func = _wrapper
        # Build call plan once (unwrap to reach original if decorated)
        try:
            sig = inspect.signature(inspect.unwrap(self._func))  # :contentReference[oaicite:9]{index=9}
        except Exception as e:
            raise ToolDefinitionError(f"{self._name}: could not inspect callable: {e}") from e

        (
            self._arguments_map,
            self.posonly_order,
            self.p_or_kw_names,
            self.kw_only_names,
            self.required_names,
            self.has_varargs,
            self.varargs_name,
            self.has_varkw,
            self.varkw_name,
        ) = self._build_arguments_map_and_plan(sig)

        self._return_type = "Any"  # MCP doesn’t guarantee static return types

        self._sig_str: str = ""
        self._rebuild_signature_str()
        
    def invoke(self, inputs: Mapping[str, Any]) -> Any:
        """
        Invoke the remote MCP tool with validated inputs.

        Raises:
            ToolInvocationError for invocation errors.
        """
        logger.debug(f"[{self.full_name}.invoke started]")
        try:
            result = self._call_remote_once(inputs)
        except Exception as e:
            raise ToolInvocationError(f"MCPProxyTool.invoke error: {e}") from e
        logger.debug(f"[{self.full_name}.invoke started]")
        return result
    def to_dict(self)-> OrderedDict[str, Any]:
        dict_data = super().to_dict()
        dict_data["mcp_url"] = self._server_url
        dict_data["headers"] = self._headers
        return dict_data

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
            if nm == self._name:
                target = t
                break
        if target is None:
            names = [getattr(t, "name", None) or (isinstance(t, dict) and t.get("name")) for t in tools]
            raise ToolDefinitionError(
                f"MCP tool '{self._name}' not found on server '{self._source}' @ {self._server_url}; "
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
                    logger.debug(f"{self.full_name} starting a session with '{self._server_url}'")
                    await sess.initialize()
                    return await sess.call_tool(self._name, dict(inputs))

        raw = _run_sync(_do())

        # Structured-first, then text fallback, else model_dump/raw.
        logger.debug(f"{self.full_name} extracting the structured content from MCP response")
        val = _extract_structured_or_text(raw)
        if val is not None:
            return val
        return _to_plain(raw)
