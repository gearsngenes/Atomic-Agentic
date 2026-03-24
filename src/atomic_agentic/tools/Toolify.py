from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Union, Tuple

from ..core.Exceptions import ToolDefinitionError
from ..core.Invokable import AtomicInvokable
from .base import Tool
from .a2a import A2AProxyTool
from .adapter import AdapterTool
from .mcp import MCPProxyTool
from ..mcp import MCPClientHub

__all__ = ["toolify"]

# ───────────────────────────────────────────────────────────────────────────────
# toolify
# ───────────────────────────────────────────────────────────────────────────────
def toolify(
    component: AtomicInvokable | Callable[..., Any] | MCPClientHub | None,
    *,
    name: str | None = None,
    namespace: str | None = None,
    description: str | None = None,
    filter_extraneous_inputs: bool | None = None,
    remote_name: str | None = None,
    a2a_endpoint: str | None = None,
    a2a_headers: Mapping[str, str] | None = None,
) -> Tool:
    """
    Normalize a single component into a single Tool instance.

    Routing rules
    -------------
    1) If a non-empty ``a2a_endpoint`` is provided:
       - ``component`` must be ``None``.
       - construct and return ``A2AProxyTool``.

    2) If ``component`` is already a ``Tool``:
       - mutate only the explicitly provided fields
         (``name``, ``namespace``, ``description``, ``filter_extraneous_inputs``)
       - return the same instance.

    3) If ``component`` is a non-tool ``AtomicInvokable``:
       - wrap it in ``AdapterTool`` using wrapper-layer overrides.

    4) If ``component`` is an ``MCPClientHub``:
       - ``remote_name`` is required
       - construct and return ``MCPProxyTool``.

    5) If ``component`` is a plain callable:
       - construct and return a plain ``Tool``.

    Raises
    ------
    ToolDefinitionError
        For invalid routing combinations, missing required metadata, or
        unsupported component types.
    """
    a2a_url = str(a2a_endpoint or "").strip()
    filter_flag = (
        filter_extraneous_inputs
        if filter_extraneous_inputs is not None
        else True
    )

    # 1) A2A branch
    if a2a_url:
        if component is not None:
            raise ToolDefinitionError(
                "toolify: `component` must be None when a non-empty `a2a_endpoint` is provided."
            )
        return A2AProxyTool(
            url=a2a_url,
            name=name,
            namespace=namespace,
            description=description,
            headers=a2a_headers,
            filter_extraneous_inputs=filter_flag,
        )

    # If no A2A endpoint is being used, a component is required.
    if component is None:
        raise ToolDefinitionError(
            "toolify: expected either a local `component` or a non-empty `a2a_endpoint`."
        )

    # 2) Existing Tool -> mutate in place and return same instance
    if isinstance(component, Tool):
        if name is not None:
            component.name = name
        if namespace is not None:
            component.namespace = namespace
        if description is not None:
            component.description = description
        if filter_extraneous_inputs is not None:
            component.filter_extraneous_inputs = filter_extraneous_inputs
        return component

    # 3) AtomicInvokable -> AdapterTool
    if isinstance(component, AtomicInvokable):
        return AdapterTool(
            component,
            name=name,
            namespace=namespace,
            description=description,
            filter_extraneous_inputs=filter_flag,
        )

    # 4) MCP client hub -> MCPProxyTool
    if isinstance(component, MCPClientHub):
        resolved_remote_name = str(remote_name or "").strip()
        if not resolved_remote_name:
            raise ToolDefinitionError(
                "toolify: `remote_name` is required when `component` is an MCPClientHub."
            )

        return MCPProxyTool(
            remote_name=resolved_remote_name,
            name=name,
            namespace=namespace,
            description=description or "",
            client_hub=component,
            filter_extraneous_inputs=filter_flag,
        )

    # 5) Raw callable -> Tool
    if callable(component):
        resolved_name = name
        if not resolved_name:
            try:
                resolved_name = component.__name__
            except Exception as exc:  # noqa: BLE001
                raise ToolDefinitionError(
                    "toolify: `name` is required when toolifying a callable and could not be inferred from __name__."
                ) from exc

        if not isinstance(resolved_name, str) or not resolved_name:
            raise ToolDefinitionError(
                "toolify: `name` must resolve to a non-empty string when toolifying a callable."
            )

        if description is not None and not isinstance(description, str):
            raise ToolDefinitionError(
                "toolify: `description` must be a string when provided for callables."
            )

        effective_description = (
            description or component.__doc__ or "undescribed"
        ).strip() or "undescribed"

        return Tool(
            function=component,
            name=resolved_name,
            namespace=namespace,
            description=effective_description,
            filter_extraneous_inputs=filter_flag,
        )

    raise ToolDefinitionError(
        "toolify: unsupported `component` type. Expected Tool | AtomicInvokable | Callable | MCPClientHub, "
        "or use `a2a_endpoint` for A2A proxy construction."
    )

def batch_toolify(
    executable_components: List[Union[AtomicInvokable, Callable[..., Any]]] = [],
    *,
    a2a_servers: List[Tuple[str, Any]] = [],
    mcp_servers: List[Tuple[str, Any]] = [],
    batch_namespace: str = "default",
    batch_filter: Optional[bool] = None,
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
    batch_filter : Optional[bool]
    Returns
    -------
    List[Tool]
        List of Tool instances derived from the provided components and remote servers.
    """
    flag = batch_filter if batch_filter is not None else True
    tools: List[Tool] = []

    # Toolify local components
    for component in executable_components:
        tool = toolify(component, namespace=batch_namespace, filter_extraneous_inputs=flag)
        tools.append(tool)

    # Toolify all tools from A2A servers
    for url, headers in a2a_servers:
        a2a_tool = toolify(url, remote_protocol="a2a", headers=headers, namespace=batch_namespace, filter_extraneous_inputs=flag)
        tools.append(a2a_tool)

    # Toolify all tools from MCP servers
    for url, headers in mcp_servers:
        # Discover available MCP tools
        mcp_tool_names = {}#list_mcp_tools(url, headers=headers)
        for tool_name in mcp_tool_names:
            mcp_tool = toolify(
                url,
                name=tool_name,
                remote_protocol="mcp",
                headers=headers,
                namespace=batch_namespace,
                filter_extraneous_inputs=flag,
            )
            tools.append(mcp_tool)

    return tools
