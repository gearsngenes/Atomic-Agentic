# Tools.py
from __future__ import annotations
from collections import OrderedDict
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

from .Exceptions import ToolDefinitionError, ToolInvocationError
from .Primitives import Tool, Agent, ArgumentMap

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────
__all__ = ["Tool", "AgentTool", "MCPProxyTool"]

# ───────────────────────────────────────────────────────────────────────────────
# Agent Tool
# ───────────────────────────────────────────────────────────────────────────────
class AgentTool(Tool):
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, agent: Agent):
        # extract tool creation inputs
        function = agent.invoke
        name = "invoke"
        namespace = agent.name
        description = agent.description
        # set private variable
        self._agent = agent
        super().__init__(function, name, namespace, description)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def agent(self) -> Agent:
        return self._agent
    
    @agent.setter
    def agent(self, value: Agent)-> None:
        self._agent = value
        self._function = self._agent.invoke
        self._namespace = value.name
        self._description = value.description
        # Identity in import space (may be overridden by subclasses)
        self._module, self._qualname = self._get_mod_qual(self.function)
        # Build argument schema and return type from the current function.
        self._arguments_map, self._return_type = self._build_io_schemas()
        # Persistibility flag exposed as a public property.
        self._is_persistible_internal: bool = self._compute_is_persistible()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def namespace(self) -> str:
        return self._namespace
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def function(self) -> Callable:
        return self._function
    
    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_io_schemas(self) -> tuple[ArgumentMap, str]:
        """Construct ``arguments_map`` and ``return_type`` from the wrapped
        callable's signature.

        Rules:
        - If an annotation is present, it *always* defines the type string.
        - If no annotation but a default value exists, the type string is
          derived from ``type(default)``.
        - If neither is present, the type string is 'Any'.
        """
        return self.agent.pre_invoke.arguments_map, self.agent.post_invoke.return_type

    def _compute_is_persistible(self) -> bool:
        """Default persistibility check for callable-based tools.

        A Tool is considered persistible if its function has both ``__module__``
        and ``__qualname__`` and does not appear to be a local/helper function.
        Subclasses can override this with their own criteria.
        """
        self.agent.pre_invoke.is_persistible and self.agent.post_invoke.is_persistible

    def to_arg_kwarg(self, inputs: Mapping[str, Any]) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        """Default implementation for mapping input dicts to ``(*args, **kwargs)``.

        The base policy is:

        - Required parameters (those without ``default`` and not VAR_*) must be present.
        - Unknown keys raise if there is no VAR_KEYWORD parameter; otherwise they
          are accepted and passed through in ``**kwargs``.
        - POSITIONAL_ONLY parameters are always passed positionally.
        - POSITIONAL_OR_KEYWORD and KEYWORD_ONLY parameters are passed as
          keywords (Python accepts this for both kinds).
        - VAR_POSITIONAL expects the mapping to contain the parameter name with
          a sequence value; these are appended to ``*args``.
        - VAR_KEYWORD collects all remaining unknown keys into ``**kwargs``.
        """
        return tuple([]), dict(inputs)

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute the underlying callable.

        Subclasses may override this to change *how* a tool is executed (for
        example, by making a remote MCP call or invoking an Agent), but should
        not change the high-level semantics.
        """
        try:
            result = self._function(kwargs) # function = self.agent.invoke()
        except Exception as e:  # pragma: no cover - thin wrapper
            raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e
        return result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self)-> OrderedDict[str, Any]:
        base = super().to_dict()
        base.update(OrderedDict(
            agent = self.agent.to_dict()
        ))
        return base


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

    It does *not* depend on any __utils__ helpers and is safe to call
    from both sync code and from within a running event loop via
    `_run_coro_sync`.
    """

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
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self._server_url = str(server_url)
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
        )

    # ------------------------------------------------------------------ #
    # Properties
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
        self._arguments_map, self._return_type = self._build_io_schemas()
        self._is_persistible_internal = self._compute_is_persistible()

    @property
    def server_url(self) -> str:
        return self._server_url
    
    @server_url.setter
    def server_url(self, value: str):
        try:
            all_tools = list_mcp_tools(value, headers = self._headers)
        except Exception as e:
            raise ValueError(f"Failed to connect to server on '{value}'. Got error: {e}")
        if self.name not in all_tools:
            raise ToolDefinitionError(f"'{self.name}' is not present in this new MCP serverl's list")
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
        self._arguments_map, self._return_type = self._build_io_schemas()
        self._is_persistible_internal = self._compute_is_persistible()
    
    @property
    def function(self) -> Callable:
        return self._function
    
    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _get_mod_qual(
        self,
        function: Callable[..., Any],
    ) -> tuple[Optional[str], Optional[str]]:
        """
        MCP-backed tools don't map to a stable Python import path.
        We explicitly opt-out of import-based identity.
        """
        return None, None

    def _compute_is_persistible(self) -> bool:
        """
        Consider an MCPProxyTool persistible if:
        - it has a server URL, namespace, and tool name,
        - and the named tool still exists on the server.

        Network errors are treated as non-persistible.
        """
        if not (self._server_url and self._name):
            return False

        try:
            tools = list_mcp_tools(self._server_url, self._headers)
        except Exception:
            return False

        return self._name in tools

    def _build_io_schemas(self) -> tuple[ArgumentMap, str]:
        """
        Construct `arguments_map` and `return_type` from MCP `input_schema`
        and `output_schema`.

        - All parameters are KEYWORD_ONLY (MCP sends a single JSON object).
        - Required parameters come from `input_schema["required"]`.
        - Optional parameters are marked as having a default (if provided by
          the schema; otherwise we give them a placeholder default of None).
        - Return type is derived from `output_schema` if provided; otherwise "Any".
        """

        arg_map: ArgumentMap = ArgumentMap()

        # ----- Inputs: JSON Schema → ArgumentMap -----
        input_schema = self._mcpdata.get("input_schema")
        if isinstance(input_schema, Mapping):
            props = input_schema.get("properties") or {}
            if not isinstance(props, Mapping):
                props = {}

            required = input_schema.get("required") or []
            if not isinstance(required, (list, tuple)):
                required = []
            required_set = {str(name) for name in required}

            for index, (raw_name, raw_meta) in enumerate(props.items()):
                name = str(raw_name)
                meta_schema = raw_meta or {}
                if not isinstance(meta_schema, Mapping):
                    meta_schema = {}

                type_str = self._json_schema_type_to_str(meta_schema)
                param_meta: Dict[str, Any] = {
                    "index": index,
                    "kind": "KEYWORD_ONLY",
                    "type": type_str,
                }

                if name in required_set:
                    # Required param: no 'default' key so Tool.to_arg_kwarg treats it as required.
                    pass
                else:
                    # Optional param: mark as having a default so base Tool logic
                    # doesn't consider it required. If the schema provides a
                    # default, surface it; otherwise use None as a placeholder.
                    if "default" in meta_schema:
                        param_meta["default"] = meta_schema.get("default")
                arg_map[name] = param_meta

        # ----- Output: JSON Schema → return_type string -----
        return_type = "Any"
        output_schema = self._mcpdata.get("output_schema")
        if isinstance(output_schema, Mapping):
            return_type = self._json_schema_type_to_str(output_schema)

        return arg_map, return_type

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
    # Class specific helpers
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
        self._arguments_map, self._return_type = self._build_io_schemas()
        self._is_persistible_internal = self._compute_is_persistible()

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """
        Extend the base Tool serialization with MCP connection details.

        Note: we deliberately do **not** include header values to avoid
        leaking credentials; only their presence/keys.
        """
        base = super().to_dict()
        base.update(
            {
                "server_url": self._server_url,
            }
        )
        return base
