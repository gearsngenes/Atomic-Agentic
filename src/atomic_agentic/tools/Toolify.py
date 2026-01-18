from __future__ import annotations

from typing import Any, Callable, List, Mapping, Union, Tuple

from ..core.Exceptions import ToolDefinitionError
from ..core.Invokable import AtomicInvokable
from .base import Tool
from .a2a import A2AProxyTool
from .adapter import AdapterTool
from .mcp import MCPProxyTool, list_mcp_tools

__all__ = ["toolify"]

# ───────────────────────────────────────────────────────────────────────────────
# toolify
# ───────────────────────────────────────────────────────────────────────────────
def toolify(
    component: Union[AtomicInvokable, str, Callable[..., Any]],
    *,
    name: str | None = None,
    description: str | None = None,
    namespace: str | None = None,
    remote_protocol: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> Tool:
    """
    Normalize a single component into a single Tool instance.

    Parameters
    ----------
    component:
        One of:
        - Tool            → returned unchanged (passthrough).
        - AtomicInvokable → wrapped as `AdapterTool`.
        - callable        → wrapped as a plain `Tool`.
        - str (URL)       → treated as a remote endpoint URL. Toolify will attempt:
                            1) MCP discovery (strict): requires explicit `name` parameter.
                            2) A2A fallback (lenient): auto-discovers name from agent metadata.

    Keyword-only parameters
    -----------------------
    name : Optional[str]
        For callables: required logical name (inferred from function __name__ if omitted).
        For MCP URLs: REQUIRED when using an MCP server endpoint.
        For A2A URLs: ignored (auto-discovered from remote agent card).
    description : Optional[str]
        Human-readable description. For callables and tools, used as fallback when not provided.
        For MCP tools, the remote description is preferred if available.
    namespace : Optional[str]
        Logical namespace for the Tool.
    remote_protocol : Optional[str] (must be "mcp" or "a2a" if toolifying from URL)
        When `component` is a URL string, specifies which remote protocol to use:
    headers : Optional[Mapping[str, str]]
        Transport headers for MCP/A2A requests (auth, etc.).

    Returns
    -------
    Tool
        A single Tool instance derived from `component`.

    Raises
    ------
    ToolDefinitionError
        For invalid inputs, missing required metadata, or remote discovery issues.
    """

    # 1) Passthrough if already a Tool
    if isinstance(component, Tool):
        return component

    # 2) AtomicInvokable → AdapterTool
    if isinstance(component, AtomicInvokable):
        return AdapterTool(component, namespace=namespace)

    # 3) String → MCP (strict, requires name) or A2A (lenient, auto-discovers)
    if isinstance(component, str):
        # Validate remote_protocol
        if remote_protocol not in ("mcp", "a2a"):
            raise ToolDefinitionError(
                "toolify: 'remote_protocol' must be either 'mcp' or 'a2a' when toolifying from a URL string."
            )
        # Extract & sanitize URL
        url = component.strip()
        # Both MCP and A2A implementations are HTTP-based.
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ToolDefinitionError(
                "toolify: when `component` is a string it must be an HTTP(S) URL "
                "(e.g. 'http://localhost:8000' for MCP or A2A)."
            )

        if remote_protocol == "mcp":
            return MCPProxyTool(
                server_url=url,
                tool_name=name,
                namespace=namespace,
                description=description,
                headers=headers,
            )
        else:
            return A2AProxyTool(url=url, headers=headers)

    # 4) Raw callable → Tool
    if callable(component):
        if not name:
            try:
                name = component.__name__
            except Exception as exc:  # noqa: BLE001
                raise ToolDefinitionError(
                    "toolify: 'name' (str) is required when toolifying a callable "
                    "and could not be inferred from __name__."
                ) from exc

        if not isinstance(name, str) or not name:
            raise ToolDefinitionError(
                "toolify: 'name' (str) is required when toolifying a callable."
            )

        if description is not None and not isinstance(description, str):
            raise ToolDefinitionError(
                "toolify: 'description' must be a string when provided for callables."
            )

        effective_description = (description or component.__doc__ or "undescribed").strip()

        return Tool(
            function=component,
            name=name,
            namespace=namespace,
            description=effective_description,
        )

    # 5) Unsupported type
    raise ToolDefinitionError(
        "toolify: unsupported input type. Expected AtomicInvokable | Callable | A2A/MCP endpoint URL string."
    )

def batch_toolify(
    executable_components: List[Union[AtomicInvokable, Callable[..., Any]]] = [],
    *,
    a2a_servers: List[Tuple[str, Any]] = [],
    mcp_servers: List[Tuple[str, Any]] = [],
    batch_namespace: str = "default",
) -> List[Tool]:
    """
    Normalize a batch of components into Tool instances.

    Parameters
    ----------
    executable_components : List[AtomicInvokable | Callable[..., Any]]
        List of local callables or AtomicInvokable instances to toolify.
    a2a_servers : List[Tuple[str, Any]]
        List of (A2A endpoint URL, headers) tuples to toolify all tools from.
    mcp_servers : List[Tuple[str, Any]]
        List of (MCP server URL, headers) tuples to toolify all tools from.
    batch_namespace : str
        Namespace to assign to all toolified local components.

    Returns
    -------
    List[Tool]
        List of Tool instances derived from the provided components and remote servers.
    """
    tools: List[Tool] = []

    # Toolify local components
    for component in executable_components:
        tool = toolify(component, namespace=batch_namespace)
        tools.append(tool)

    # Toolify all tools from A2A servers
    for url, headers in a2a_servers:
        a2a_tool = toolify(
            url,
            remote_protocol="a2a",
            headers=headers,
            namespace=batch_namespace,
        )
        tools.append(a2a_tool)

    # Toolify all tools from MCP servers
    for url, headers in mcp_servers:
        # Discover available MCP tools
        mcp_tool_names = list_mcp_tools(url, headers=headers)
        for tool_name in mcp_tool_names:
            mcp_tool = toolify(
                url,
                name=tool_name,
                remote_protocol="mcp",
                headers=headers,
                namespace=batch_namespace,
            )
            tools.append(mcp_tool)

    return tools
