from .__utils__ import (
    _normalize_url,
    _discover_mcp_tool_names,
    _is_http_url
)
from .Tools import *
from .Agents import *
from ._exceptions import ToolDefinitionError
from typing import List, Optional, Union, Callable
def toolify(
        obj: Union[Tool, Agent, str, Callable],
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
                MCPProxyTool(
                    tool_name=n,
                    server_name=server_name,
                    server_url=url,
                    headers=headers)
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
                    source=source or "default",
                )
            ]

        # 5) Unsupported
        raise ToolDefinitionError(
            "ToolFactory: unsupported input. Expected Tool | Agent | callable | MCP URL string."
        )
