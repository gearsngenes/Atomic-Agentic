from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence, Union

from .Tools import Tool, AgentTool, MCPProxyTool, list_mcp_tools
from .Agents import Agent
from .Exceptions import ToolDefinitionError

__all__ = ["toolify"]

def toolify(
    component: Union[Tool, Agent, str, Callable[..., Any]],
    **kwargs: Any,
) -> List[Tool]:
    """
    Normalize a single component into a list of Tool instances.

    Parameters
    ----------
    component:
        One of:
        - Tool       → returned as `[component]` (passthrough).
        - Agent      → wrapped as `AgentTool` and returned in a list.
        - callable   → wrapped as a plain `Tool` using `name`/`description`/`namespace`.
        - str        → treated as an MCP HTTP endpoint URL, materialised as one
                       or more `MCPProxyTool` instances.

    Keyword-only configuration
    --------------------------
    name : Optional[str]
        For callables: required logical name.
        For MCP URLs: optional remote tool name. If omitted, all tools on the
        server are instantiated (subject to `include`/`exclude`).
    description : Optional[str]
        Human-readable description. For MCP tools, this is only a fallback:
        the remote description wins when present.
    namespace : Optional[str]
        Logical namespace for the Tool. Defaults to:
        - For callables: `source` if provided, else `"default"`.
        - For MCP URLs: `server_name` if provided, else `source`, else `"mcp"`.
    source : Optional[str]
        Backwards-compatibility alias for `namespace` (legacy API).
    server_name : Optional[str]
        Backwards-compatibility alias used as the MCP namespace label when
        `component` is an MCP URL string.
    headers : Optional[Mapping[str, str]]
        HTTP headers for MCP transport (auth, etc.). The *presence* of this key
        is required when `component` is a string; the value may be `None` if
        no headers are needed.
    include : Optional[Sequence[str]]
        When toolifying an MCP URL with no explicit `name`, restrict to this
        whitelist of remote tool names.
    exclude : Optional[Sequence[str]]
        When toolifying an MCP URL with no explicit `name`, drop these tool
        names after any `include` filter is applied.

    Returns
    -------
    List[Tool]
        One or more Tool instances derived from `component`.

    Raises
    ------
    ToolDefinitionError
        For invalid inputs, missing required metadata, or MCP discovery issues.
    """

    # Keep an original view so we can detect presence of keys (e.g. headers).
    original_kwargs = dict(kwargs)

    # Extract supported kwargs with backwards-compatible aliases
    name = kwargs.pop("name", None)
    description = kwargs.pop("description", None)
    namespace_kw = kwargs.pop("namespace", None)
    headers = kwargs.pop("headers", None)
    include = kwargs.pop("include", None)
    exclude = kwargs.pop("exclude", None)

    # If any unexpected kwargs remain, surface them early (fail-fast contract)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise ToolDefinitionError(
            f"toolify: unexpected keyword argument(s): {unexpected}"
        )

    # 1) Passthrough if already a Tool
    if isinstance(component, Tool):
        return [component]

    # 2) Agent → AgentTool
    if isinstance(component, Agent):
        return [AgentTool(component)]

    # 3) MCP URL → MCPProxyTool(s)
    if isinstance(component, str):
        url = component.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ToolDefinitionError(
                "toolify: when `component` is a string it must be an MCP HTTP URL "
                "(e.g. 'http://localhost:8000/mcp')."
            )

        # Enforce your contract: headers key must be present (value may be None).
        if "headers" not in original_kwargs:
            raise ToolDefinitionError(
                "toolify: 'headers' keyword must be provided (it may be None) "
                "when toolifying an MCP URL string."
            )

        # Determine namespace label for MCP tools
        namespace = namespace_kw

        # (a) Explicit remote tool name → single MCPProxyTool
        if name:
            if not isinstance(name, str):
                raise ToolDefinitionError("toolify: 'name' must be a string.")
            effective_description = (description or "").strip()
            return [
                MCPProxyTool(
                    server_url=url,
                    tool_name=name,
                    namespace=namespace,
                    description=effective_description,
                    headers=headers,
                )
            ]

        # (b) No explicit name → discover all remote tools and filter
        try:
            all_tools = list_mcp_tools(server_url=url, headers=headers)
        except Exception as exc:  # noqa: BLE001
            raise ToolDefinitionError(
                f"toolify: failed to discover MCP tools at {url!r}: {exc}"
            ) from exc

        names = list(all_tools.keys())

        if include:
            include_set = {str(n) for n in include}
            names = [n for n in names if n in include_set]

        if exclude:
            exclude_set = {str(n) for n in exclude}
            names = [n for n in names if n not in exclude_set]

        if not names:
            raise ToolDefinitionError(
                    f"toolify: no MCP tools discovered for {url!r} after filtering."
                )

        tools: List[Tool] = [
            MCPProxyTool(
                server_url=url,
                tool_name=tool_name,
                namespace=namespace,
                headers=headers,
            )
            for tool_name in names
        ]
        return tools

    # 4) Raw callable → Tool
    if callable(component):
        if not name or not isinstance(name, str):
            raise ToolDefinitionError(
                "toolify: 'name' (str) is required when toolifying a callable."
            )
        if description is not None and not isinstance(description, str):
            raise ToolDefinitionError(
                "toolify: 'description' must be a string when provided for callables."
            )

        namespace = namespace_kw
        effective_description = (description or component.__doc__ or "").strip()

        return [
            Tool(
                function=component,
                name=name,
                namespace=namespace,
                description=effective_description,
            )
        ]

    # 5) Unsupported type
    raise ToolDefinitionError(
        "toolify: unsupported input type. Expected Tool | Agent | callable | MCP URL string."
    )
