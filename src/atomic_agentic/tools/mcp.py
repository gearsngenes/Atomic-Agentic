from __future__ import annotations
import functools
import asyncio
import threading
from typing import (
    Any,
    Mapping,
    Callable,
    List,
    Optional,
    Dict,
    Awaitable,
    TypeVar,
)

import logging
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from urllib.parse import urlparse, urlunparse

from ..core.Exceptions import ToolDefinitionError, ToolInvocationError
from ..core.Invokable import ArgumentMap
from ..core.Parameters import ParamSpec
from ..core.sentinels import NO_VAL
from .base import Tool
logger = logging.getLogger(__name__)


def _normalize_mcp_url(url: str) -> str:
        """Normalize an MCP server URL so that the path is `/mcp` when the
        provided URL has an empty or root path.

        Examples:
            - "http://localhost:8000" -> "http://localhost:8000/mcp"
            - "http://localhost:8000/" -> "http://localhost:8000/mcp"
            - "http://localhost:8000/mcp" -> unchanged
        """
        parts = urlparse(str(url))
        if not parts.path or parts.path == "/":
                parts = parts._replace(path="/mcp")
        return urlunparse(parts)

# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
__all__ = ["MCPProxyTool", "list_mcp_tools",]

# ───────────────────────────────────────────────────────────────────────────────
# MCP helper functions (generic, class-independent)
# ───────────────────────────────────────────────────────────────────────────────
T = TypeVar("T")

def _run_coro_sync(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine from sync code, even if we're already inside
    an event loop.

    - If no loop is running in this thread, uses asyncio.run(coro).
    - If a loop *is* running, spins up a fresh event loop in a worker
      thread, runs the coroutine there, and returns the result.

    This avoids the common "Cannot run the event loop while another loop
    is running" error you get in notebooks / async apps.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread → safe to use asyncio.run.
        return asyncio.run(coro)

    # Already inside a running loop → run coro in a separate thread.
    result_box: List[T] = []
    error_box: List[BaseException] = []

    def runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            result_box.append(result)
        except BaseException as exc:  # noqa: BLE001
            error_box.append(exc)
        finally:
            loop.close()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if error_box:
        raise error_box[0]
    if not result_box:
        raise RuntimeError("Coroutine completed without result")

    return result_box[0]

def list_mcp_tools(
    server_url: str,
    headers: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    List tools exposed by an MCP server over streamable HTTP.

    This is a *synchronous* helper that:

    - Opens a short-lived MCP session to `server_url`,
    - Initializes the session,
    - Calls `list_tools`,
    - Returns a mapping of tool name → metadata dict with:
        - "name"          : str
        - "description"   : str
        - "input_schema"  : dict | None     (JSON Schema)
        - "output_schema" : dict | None     (JSON Schema, if provided)
        - "raw"           : the original Tool object (for advanced use)
    """

    # Normalize the provided URL to point at the MCP mount path when needed
    server_url = _normalize_mcp_url(server_url)

    async def _do() -> Dict[str, Dict[str, Any]]:
        headers_dict: Optional[Dict[str, str]] = dict(headers) if headers else None

        # Connect to the MCP server over streamable HTTP.
        async with streamablehttp_client(server_url, headers=headers_dict) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_resp = await session.list_tools()

        # Newer SDKs return a ListToolsResult with `.tools`
        tools = getattr(tools_resp, "tools", tools_resp)
        result: Dict[str, Dict[str, Any]] = {}
        if not tools:
            return result

        for tool in tools:
            # Name
            name = getattr(tool, "name", None)
            if name is None and isinstance(tool, Mapping):
                name = tool.get("name")
            if not name:
                continue
            name_str = str(name)

            # Description
            description = getattr(tool, "description", None)
            if description is None and isinstance(tool, Mapping):
                description = tool.get("description")
            description_str = str(description) if description is not None else ""

            # Input schema (JSON Schema) if present
            input_schema = getattr(tool, "inputSchema", None)
            if input_schema is None and isinstance(tool, Mapping):
                input_schema = tool.get("inputSchema")

            # Output schema if present (structured output) – newer MCP servers
            output_schema = getattr(tool, "outputSchema", None)
            if output_schema is None and isinstance(tool, Mapping):
                output_schema = tool.get("outputSchema")

            result[name_str] = {
                "name": name_str,
                "description": description_str,
                "input_schema": input_schema,
                "output_schema": output_schema,
                "raw": tool,
            }
        return result
    return _run_coro_sync(_do())

def call_mcp_tool_once(
    server_url: str,
    tool_name: str,
    inputs: Mapping[str, Any],
    headers: Optional[Mapping[str, str]] = None,
) -> Any:
    """
    Call a single MCP tool exactly once, synchronously.

    This is designed to be partially applied with `server_url`, `tool_name`,
    and `headers` so that the resulting callable has the shape:

        fn(inputs: Mapping[str, Any]) -> Any

    which can be used as the underlying function for a dict-first Tool.

    Parameters
    ----------
    server_url:
        Streamable HTTP MCP endpoint (e.g. "http://localhost:8000/mcp").
    tool_name:
        Name of the tool on that server.
    inputs:
        Dict-like payload to send as the tool's arguments.
    headers:
        Optional HTTP headers (e.g. auth) for the MCP transport.

    Returns
    -------
    Any
        Structured result if available (from `structuredContent` or dict),
        otherwise a best-effort text/string representation, otherwise the
        raw CallToolResult object.
    """

    async def _do() -> Any:
        headers_dict: Optional[Dict[str, str]] = dict(headers) if headers else None

        try:
            async with streamablehttp_client(server_url, headers=headers_dict) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    raw = await session.call_tool(
                        tool_name,
                        arguments=dict(inputs),
                    )
        except Exception as exc:  # noqa: BLE001
            # Let MCP/network errors become ToolInvocationError so callers
            # see a consistent error type.
            raise ToolInvocationError(
                f"Error calling MCP tool '{tool_name}' at '{server_url}': {exc}"
            ) from exc

        return raw #_normalize_mcp_call_result()

    return _run_coro_sync(_do())


# ───────────────────────────────────────────────────────────────────────────────
# MCP-Proxy Tool
# ───────────────────────────────────────────────────────────────────────────────
class MCPProxyTool(Tool):
    """
    Proxy a single MCP server tool as a normal dict-first Tool.

    Construction:
    - Discovers the tool via `list_mcp_tools(server_url, headers)`.
    - Extracts JSON Schema `input_schema` / `output_schema` and description.
    - Binds a dict-first function that calls the MCP tool exactly once.
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        server_url: str,
        tool_name: str,
        namespace: str = None,
        description: str = "",
        filter_extraneous_inputs: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self._server_url = _normalize_mcp_url(str(server_url))
        self._headers: Dict[str, str] = dict(headers or {})

        # Discover tools once and stash the schema for this tool
        try:
            all_tools = list_mcp_tools(server_url=self._server_url, headers=self._headers)
        except Exception as e:
            raise ValueError(f"Failed to connect to server on '{server_url}'. Got error: {e}")
        if tool_name not in all_tools:
            raise ToolDefinitionError(
                f"MCPProxyTool: tool {tool_name!r} not found on MCP server {self._server_url!r}"
            )

        self._mcpdata: Dict[str, Any] = all_tools[tool_name]
        name = tool_name

        # Prefer MCP description; fall back to explicit description, then a stub
        effective_description = (
            (self._mcpdata.get("description") or description).strip()
            or "undescribed MCP tool"
        )

        # Underlying function: dict-first → MCP call
        function = functools.partial(
            call_mcp_tool_once,
            server_url=self._server_url,
            tool_name=name,
            headers=self._headers,
        )

        super().__init__(
            function=function,
            name=name,
            namespace=namespace or "mcp",
            description=effective_description,
            filter_extraneous_inputs=filter_extraneous_inputs,
        )

    # ------------------------------------------------------------------ #
    # Atomic-Invokable Properties
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return self._name

    @name.setter  # override base setter
    def name(self, value: str) -> None:
        """
        When the MCP tool name changes, update metadata and function binding
        to point to the new tool on the same server.
        """
        new_name = str(value).strip()
        if not new_name:
            raise ToolDefinitionError("MCPProxyTool.name cannot be empty.")

        # handles edge case of initializing self._name for the first time
        if getattr(self, "_name", None) == new_name:
            return  # no-op

        all_tools = list_mcp_tools(server_url=self._server_url, headers=self._headers)
        if new_name not in all_tools:
            raise ToolDefinitionError(
                f"{self.full_name}: tool {new_name!r} not found on MCP server {self._server_url!r}"
            )

        self._name = new_name
        self._mcpdata = all_tools[new_name]
        new_function = functools.partial(
            call_mcp_tool_once,
            server_url=self._server_url,
            tool_name=self._name,
            headers=self._headers,
        )
        self._function = new_function
        self._module, self._qualname = self._get_mod_qual(new_function)
        
        # Rebuild schema using template method
        parameters, return_type = self._build_tool_signature()
        self._parameters = parameters
        self._return_type = return_type

    # ------------------------------------------------------------------ #
    # Tool Properties
    # ------------------------------------------------------------------ #
    @property
    def function(self) -> Callable:
        return self._function

    # ------------------------------------------------------------------ #
    # MCP-Proxy-Tool Properties
    # ------------------------------------------------------------------ #
    @property
    def server_url(self) -> str:
        return self._server_url
    
    @server_url.setter
    def server_url(self, value: str):
        value = _normalize_mcp_url(value)
        try:
            all_tools = list_mcp_tools(value, headers = self._headers)
        except Exception as e:
            raise ValueError(f"Failed to connect to server on '{value}'. Got error: {e}")
        if self.name not in all_tools:
            raise ToolDefinitionError(f"'{self.name}' is not present in this new MCP server's list")
        self._mcpdata = all_tools[self.name]
        self._server_url = value
        new_function = functools.partial(
            call_mcp_tool_once,
            server_url=self._server_url,
            tool_name=self._name,
            headers=self._headers,
        )
        self._function = new_function
        self._module, self._qualname = self._get_mod_qual(new_function)
        
        # Rebuild schema using template method
        parameters, return_type = self._build_tool_signature()
        self._parameters = parameters
        self._return_type = return_type

    # ------------------------------------------------------------------ #
    # Signature Building (Template Method)
    # ------------------------------------------------------------------ #
    def _build_tool_signature(self) -> tuple[list[ParamSpec], str]:
        """Build tool signature from MCP `input_schema` and `output_schema`.

        - All parameters are KEYWORD_ONLY (MCP sends a single JSON object).
        - Required parameters come from `input_schema["required"]`.
        - Optional parameters are marked as having a default (if provided by
          the schema; otherwise we give them a placeholder default of None).
        - Return type is derived from `output_schema` if provided; otherwise "Any".
        """
        parameters: list[ParamSpec] = []

        # ----- Inputs: JSON Schema → list[ParamSpec] -----
        input_schema = self._mcpdata.get("input_schema")
        if isinstance(input_schema, Mapping):
            props = input_schema.get("properties") or {}
            if not isinstance(props, Mapping):
                props = {}

            required = input_schema.get("required") or []
            if not isinstance(required, (list, tuple)):
                required = []

            for index, (raw_name, raw_meta) in enumerate(props.items()):
                meta_schema = raw_meta or {}
                if not isinstance(meta_schema, Mapping):
                    meta_schema = {}
                name = str(raw_name)
                kind = "KEYWORD_ONLY"
                type_str = self._json_schema_type_to_str(meta_schema)
                default = NO_VAL
                if "default" in meta_schema:
                    default = meta_schema.get("default")
                
                parameters.append(ParamSpec(
                    name=name,
                    index=index,
                    kind=kind,
                    type=type_str,
                    default=default
                ))

        # ----- Output: JSON Schema → return_type string -----
        return_type = "Any"
        output_schema = self._mcpdata.get("output_schema")
        if isinstance(output_schema, Mapping):
            return_type = self._json_schema_type_to_str(output_schema)

        return parameters, return_type

    # ------------------------------------------------------------------ #
    # Tool Helpers
    # ------------------------------------------------------------------ #
    def _get_mod_qual(
        self,
        function: Callable[..., Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """
        MCP-backed tools don't map to a stable Python import path.
        We explicitly opt-out of import-based identity.
        """
        return call_mcp_tool_once.__module__, call_mcp_tool_once.__qualname__

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """
        Execute the underlying MCP call using the dict-only function.
        """
        if args:
            raise ToolInvocationError(
                f"{self.full_name}: MCP tools do not accept positional arguments; got {args!r}"
            )
        raw = self._function(inputs = kwargs)
        return self._normalize_mcp_result(raw)

    # ------------------------------------------------------------------ #
    # MCP-Proxy-Tool Helpers
    # ------------------------------------------------------------------ #
    def _normalize_mcp_result(self, raw: Any):
        # 1) Direct CallToolResult: prefer structuredContent if present
        structured = getattr(raw, "structuredContent", None)
        structured_return = None
        if structured not in (None, [], {}):
            structured_return = structured

        # 2) Tuple (content, structured_data)
        if isinstance(raw, tuple) and len(raw) == 2:
            contents, structured_data = raw
            if structured_data not in (None, [], {}):
                structured_return = structured_data
            raw = contents  # fall through to content handling
        
        # 3) Dict ({"content":..., "structuredContent":...})
        if isinstance(raw, Mapping) and "structuredContent" in raw:
            structured_return = raw["structuredContent"]
        
        # if structured_return != None
        if structured_return is not None:
            if isinstance(structured_return, Mapping):
                if len(structured_return) == 1 and "result" in structured_return:
                    return structured_return["result"]
            return structured_return
        
        if isinstance(raw, Mapping):
            return dict(raw)

        # 4) Content-only result
        contents = getattr(raw, "content", raw)
        if isinstance(contents, (list, tuple)):
            texts: List[str] = []
            for item in contents:
                # Text-like content blocks (e.g. mcp.types.TextContent)
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    texts.append(text)
                elif isinstance(item, str):
                    texts.append(item)
            if texts:
                return "\n".join(texts)

        # 5) Fallback: return as-is
        return raw
    
    @staticmethod
    def _json_primitive_to_str(json_type: str, schema: Mapping[str, Any]) -> str:
        """
        Map a single JSON Schema primitive 'type' value to a Python-ish type string.

        For arrays, try to inspect `items.type` to refine the inner type if available.
        """
        mapping: Dict[str, str] = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "object": "Dict[str, Any]",
            "array": "List[Any]",
            "null": "None",
        }

        if json_type == "array":
            items = schema.get("items")
            if isinstance(items, Mapping):
                inner = MCPProxyTool._json_schema_type_to_str(items)
                return f"List[{inner}]"

        return mapping.get(json_type, "Any")

    @staticmethod
    def _json_schema_type_to_str(schema: Mapping[str, Any]) -> str:
        """
        Best-effort conversion from a JSON Schema fragment to a type string.

        Handles:
        - `type: "string" | "integer" | "number" | "boolean" | "object" | "array" | "null"`
        - union types such as `["string", "null"]`
        - falls back to "Any" when type is missing or unknown.
        """
        if not isinstance(schema, Mapping):
            return "Any"

        t = schema.get("type")

        # Union types expressed as a list, e.g. ["string", "null"]
        if isinstance(t, (list, tuple)):
            parts: List[str] = []
            for item in t:
                if isinstance(item, str):
                    parts.append(MCPProxyTool._json_primitive_to_str(item, schema))
            parts = [p for p in parts if p]
            return " | ".join(parts) if parts else "Any"

        # Simple primitive type
        if isinstance(t, str):
            return MCPProxyTool._json_primitive_to_str(t, schema)

        # No explicit "type": we don't attempt to infer from enum/format/etc.
        return "Any"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def refresh(self, headers: Any) -> None:
        """
        Re-fetch this tool's definition from the MCP server and rebuild
        its argument/return schemas and function binding.
        """
        self._headers = headers
        all_tools = list_mcp_tools(server_url=self._server_url, headers=self._headers)
        if self._name not in all_tools:
            raise ToolDefinitionError(
                f"{self.full_name}: tool {self._name!r} no longer exists on MCP server {self._server_url!r}"
            )

        self._mcpdata = all_tools[self._name]
        new_function = functools.partial(
            call_mcp_tool_once,
            server_url=self._server_url,
            tool_name=self._name,
            headers=self._headers,
        )
        # This will rebuild argument map, return type, and persistibility flag.
        self._function = new_function
        self._module, self._qualname = self._get_mod_qual(new_function)
        
        # Rebuild schema using template method
        parameters, return_type = self._build_tool_signature()
        self._parameters = parameters
        self._return_type = return_type

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """
        Extend the base Tool serialization with MCP connection details.

        Note: we deliberately do **not** include header values to avoid
        leaking credentials; only their presence/keys.
        """
        d = super().to_dict()
        d.update({"server_url": self._server_url,})
        return d

