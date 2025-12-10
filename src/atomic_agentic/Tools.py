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
# Tool
# ───────────────────────────────────────────────────────────────────────────────
# class Tool:
#     """
#     Stateless wrapper around a Python callable with a dict-first invocation API.

#     Construction builds:
#       • arguments_map (name → {index, mode/kind, ann, ann_meta?, has_default, default? [JSON-safe]})
#       • call-plan (posonly / p_or_kw / kw_only / required / varargs / varkw)
#       • return_type (canonical string)
#       • signature (canonical, display-only)

#     Parameter kinds follow Python's calling convention (pos-only, pos-or-kw, kw-only, *args, **kwargs).  :contentReference[oaicite:8]{index=8}
#     """

#     # -------------------------
#     # Construction
#     # -------------------------
#     def __init__(
#         self,
#         func: Callable,
#         name: str,
#         description: str = "",
#         source: str = "default",
#     ) -> None:
#         self._func: Callable = func
#         self._name: str = (name or func.__name__) or "unnamed_callable"
#         self._description: str = (description or func.__doc__) or ""
#         self._source: str = source
        
#         self.module = getattr(self.func, "__module__", None)
#         self.qualname = getattr(self.func, "__qualname__", None)

#         if self.module is None or self.qualname is None:
#             logger.warning(f"{func.__name__} lacks either a module ({self.module}) or qualname ({self.qualname}) load from in future reference")

#         # Build call plan once (unwrap to reach original if decorated)
#         try:
#             sig = inspect.signature(inspect.unwrap(func))  # :contentReference[oaicite:9]{index=9}
#         except Exception as e:
#             raise ToolDefinitionError(f"{name}: could not inspect callable: {e}") from e

#         (
#             self._arguments_map,
#             self.posonly_order,
#             self.p_or_kw_names,
#             self.kw_only_names,
#             self.required_names,
#             self.has_varargs,
#             self.varargs_name,
#             self.has_varkw,
#             self.varkw_name,
#         ) = self._build_arguments_map_and_plan(sig)

#         # Derive canonical return type string
#         raw_ret = getattr(self._func, "__annotations__", {}).get("return", inspect._empty)
#         self._return_type: str = _canonize_annotation(raw_ret)[0]

#         # Build signature string
#         self._sig_str: str = ""
#         self._rebuild_signature_str()

#     # -------------------------
#     # Read-only tags and doc
#     # -------------------------
#     @property
#     def name(self) -> str:
#         return self._name

#     @property
#     def description(self) -> str:
#         return self._description

#     @property
#     def source(self) -> str:
#         return self._source

#     @property
#     def full_name(self) -> str:
#         return f"{type(self).__name__}.{self._source}.{self._name}"

#     @property
#     def func(self) -> Callable:
#         return self._func

#     @property
#     def arguments_map(self) -> OrderedDict[str, Any]:
#         return self._arguments_map

#     @property
#     def return_type(self) -> str:
#         return self._return_type

#     @property
#     def signature(self) -> str:
#         return self._sig_str

#     # -------------------------
#     # Core invocation
#     # -------------------------
#     def invoke(self, inputs: Mapping[str, Any]) -> Any:
#         """
#         Invoke the underlying callable using a dict of inputs.
#         Rules:
#         - Top-level keys must match known parameter names; extras go under '_kwargs' if **varkw exists.
#         - Positional-only params must form a contiguous prefix (we read them by name, in order).
#         - Optional reserved keys:
#             _args: list/tuple for extra *args (if function declares VAR_POSITIONAL)
#             _kwargs: mapping for extra **kwargs (if function declares VAR_KEYWORD)
#         """
#         logger.debug(f"[{self.full_name}.invoke started]")
#         if not isinstance(inputs, Mapping):
#             raise ToolInvocationError(f"{self._name}: inputs must be a mapping")

#         ARGS_KEY = "_args"
#         KWARGS_KEY = "_kwargs"

#         # Validate reserved keys early
#         if ARGS_KEY in inputs and not isinstance(inputs[ARGS_KEY], (list, tuple)):
#             raise ToolInvocationError(f"{self._name}: '{ARGS_KEY}' must be a list or tuple")
#         if KWARGS_KEY in inputs and not isinstance(inputs[KWARGS_KEY], Mapping):
#             raise ToolInvocationError(f"{self._name}: '{KWARGS_KEY}' must be a Mapping")

#         # Unknown top-level keys are not allowed; extras must go in _kwargs if **varkw exists
#         provided_names = set(inputs.keys()) - {ARGS_KEY, KWARGS_KEY}
#         known_names = set(self._arguments_map.keys())
#         unknown = sorted(provided_names - known_names)
#         if unknown:
#             if not self.has_varkw:
#                 raise ToolInvocationError(f"{self._name}: unexpected keys: {unknown}")
#             raise ToolInvocationError(
#                 f"{self._name}: unexpected keys {unknown}; place extras under '{KWARGS_KEY}' because function accepts **kwargs"
#             )

#         # Required named parameters (no default) among p_or_kw + kw_only
#         missing = sorted(self.required_names - provided_names)
#         if missing:
#             raise ToolInvocationError(f"{self._name}: missing required keys: {missing}")

#         # Build args (positional-only), gap-safe
#         args: List[Any] = []
#         seen_gap = False
#         for pname in self.posonly_order:
#             present = pname in inputs
#             if not present:
#                 seen_gap = True
#                 continue
#             if seen_gap:
#                 raise ToolInvocationError(
#                     f"{self._name}: positional-only parameters must be a contiguous prefix; missing a value before '{pname}'"
#                 )
#             args.append(inputs[pname])

#         # Extra *args if declared
#         if self.has_varargs and ARGS_KEY in inputs:
#             extra_args = inputs[ARGS_KEY]
#             args.extend(list(extra_args))

#         # Build kwargs
#         kwargs: OrderedDict[str, Any] = OrderedDict()
#         for pname in (self.p_or_kw_names + self.kw_only_names):
#             if pname in inputs:
#                 kwargs[pname] = inputs[pname]

#         # Extra **kwargs if declared
#         if self.has_varkw and KWARGS_KEY in inputs:
#             extra_kwargs = inputs[KWARGS_KEY]
#             dupes = set(kwargs.keys()) & set(extra_kwargs.keys())
#             if dupes:
#                 raise ToolInvocationError(
#                     f"{self._name}: duplicate keys supplied both as named inputs and in '{KWARGS_KEY}': {dupes}"
#                 )
#             kwargs.update(extra_kwargs)  # type: ignore[arg-type]

#         # Final call
#         try:
#             result = self._func(*args, **kwargs)
#         except ToolInvocationError:
#             raise
#         except Exception as e:
#             raise ToolInvocationError(f"{self._name}: invocation failed: {e}") from e
#         logger.debug(f"[{self.full_name}.invoke finished]")
#         return result

#     # -------------------------
#     # Introspection & serialization
#     # -------------------------
#     def to_dict(self) -> OrderedDict[str, Any]:
#         """
#         Serialize Tool metadata and signature plan for registries and UIs.

#         Notes
#         -----
#         - 'ann' is a canonical Python-flavored string (not repr of type objects).
#         - 'ann_meta' is included for complex heads ('callable', 'annotated', 'literal').
#         - 'default' is included only when a parameter has a default; it is JSON-safe already.
#         - 'mode' is a JSON-stable parameter kind token; 'kind_name' is for human display.
#         """

#         return OrderedDict(
#             # Tool Type
#             tool_type = type(self).__name__,
#             # Top-level metadata
#             name=self._name,
#             description=self._description,
#             source=self._source,
#             signature=self.signature,
#             return_type=self._return_type,
#             # Reconstructive metadata
#             module=self.module,
#             qualname=self.qualname,
#             # Arguments metadata
#             arguments_map=self.arguments_map,
#         )

#     # -------------------------
#     # Internal: build arg map + call plan (JSON-safe at creation)
#     # -------------------------
#     def _build_arguments_map_and_plan(
#         self, sig: inspect.Signature
#     ) -> Tuple[
#         OrderedDict[str, OrderedDict[str, Any]],      # arguments_map
#         List[str],                                    # posonly_order
#         List[str],                                    # p_or_kw_names
#         List[str],                                    # kw_only_names
#         set,                                          # required_names
#         bool, Optional[str],                          # has_varargs, varargs_name
#         bool, Optional[str],                          # has_varkw, varkw_name
#     ]:
#         arguments_map: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict()
#         posonly_order: List[str] = []
#         p_or_kw_names: List[str] = []
#         kw_only_names: List[str] = []
#         required_names: set = set()
#         has_varargs: bool = False
#         varargs_name: Optional[str] = None
#         has_varkw: bool = False
#         varkw_name: Optional[str] = None

#         index = 0
#         for pname, p in sig.parameters.items():
#             if p.kind is p.VAR_POSITIONAL:
#                 has_varargs = True
#                 varargs_name = pname
#                 continue
#             if p.kind is p.VAR_KEYWORD:
#                 has_varkw = True
#                 varkw_name = pname
#                 continue

#             raw_ann = p.annotation if p.annotation is not inspect._empty else inspect._empty
#             ann_str, ann_meta = _canonize_annotation(raw_ann)
#             kind_enum = p.kind  # exact enum
#             mode = KIND_TO_MODE.get(kind_enum, "pos_or_kw")

#             has_def = p.default is not inspect._empty
#             default_json = _jsonify_default(p.default) if has_def else None

#             if kind_enum is p.POSITIONAL_ONLY:
#                 posonly_order.append(pname)
#             elif kind_enum is p.POSITIONAL_OR_KEYWORD:
#                 p_or_kw_names.append(pname)
#             elif kind_enum is p.KEYWORD_ONLY:
#                 kw_only_names.append(pname)
#             else:
#                 raise ToolDefinitionError(f"Unexpected parameter kind for {pname}: {kind_enum!r}")

#             entry: OrderedDict[str, Any] = OrderedDict(
#                 index=index,
#                 kind=kind_enum,      # internal convenience
#                 mode=mode,           # JSON-stable token
#                 ann=ann_str,
#                 has_default=has_def,
#             )
#             if ann_meta is not None:
#                 entry["ann_meta"] = ann_meta
#             if has_def:
#                 entry["default"] = default_json  # JSON-safe now

#             arguments_map[pname] = entry
#             index += 1

#         # Required named parameters (no default) among p_or_kw + kw_only
#         for pname in (p_or_kw_names + kw_only_names):
#             if not arguments_map[pname]["has_default"]:
#                 required_names.add(pname)

#         return (
#             arguments_map,
#             posonly_order,
#             p_or_kw_names,
#             kw_only_names,
#             required_names,
#             has_varargs,
#             varargs_name,
#             has_varkw,
#             varkw_name,
#         )

#     # -------------------------
#     # Internal: signature-string builder (schema-derived)
#     # -------------------------
#     def _rebuild_signature_str(self) -> None:
#         """Refresh the canonical signature string from the current plan + return_type."""
#         ordered = sorted(self._arguments_map.items(), key=lambda kv: kv[1]["index"])
#         parts: List[str] = []

#         for pname, spec in ordered:
#             ann_str = spec.get("ann", "any")
#             has_default = bool(spec.get("has_default", False))
#             token: str
#             if not has_default:
#                 token = f"{pname}: {ann_str}"
#             else:
#                 token = f"{pname}?: {ann_str}"
#                 if "default" in spec:
#                     try:
#                         token += f" = {repr(spec['default'])}"
#                     except Exception:
#                         token += " = <default>"
#             parts.append(token)

#         if self.has_varargs:
#             parts.append(f"*{self.varargs_name or 'args'}")
#         if self.has_varkw:
#             parts.append(f"**{self.varkw_name or 'kwargs'}")

#         self._sig_str = f"{self.full_name}({', '.join(parts)}) -> {self._return_type}"


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
