from __future__ import annotations

from typing import Any, Callable, Optional

from ..core.Exceptions import ToolDefinitionError
from ..core.Invokable import AtomicInvokable
from ..mcp import MCPClientHub
from ..a2a import PyA2AtomicClient
from .base import Tool
from .a2a import PyA2AtomicTool
from .mcp import MCPProxyTool

__all__ = ["toolify", "batch_toolify"]


# ───────────────────────────────────────────────────────────────────────────────
# toolify
# ───────────────────────────────────────────────────────────────────────────────
def toolify(
    component: AtomicInvokable | Callable[..., Any] | MCPClientHub | PyA2AtomicClient,
    *,
    name: str | None = None,
    namespace: str | None = None,
    description: str | None = None,
    filter_extraneous_inputs: bool | None = None,
    remote_name: str | None = None,
) -> Tool:
    """
    Normalize a single component into a single Tool instance.

    Routing rules
    -------------
    1) If ``component`` is already a ``Tool``:
       - return it unchanged when no local wrapper overrides are provided;
       - if ``name``, ``namespace``, ``description``, or
         ``filter_extraneous_inputs`` is provided, return a new ``Tool`` wrapping
         the existing Tool;
       - never mutate the existing Tool in place.

    2) If ``component`` is a non-tool ``AtomicInvokable``:
       - wrap it directly in ``Tool`` using the invokable's declared schema and
         dict-first execution path;
       - if ``filter_extraneous_inputs`` is omitted, inherit
         ``component.filter_extraneous_inputs``.

    3) If ``component`` is an ``MCPClientHub``:
       - ``remote_name`` is required;
       - construct and return ``MCPProxyTool``.

    4) If ``component`` is a ``PyA2AtomicClient``:
       - ``remote_name`` is required;
       - construct and return ``PyA2AtomicTool``.

    5) If ``component`` is a plain callable:
       - construct and return a plain ``Tool``.

    ``remote_name`` is only valid for remote-client routes.

    Raises
    ------
    ToolDefinitionError
        For invalid routing combinations, missing required metadata, or
        unsupported component types.
    """
    if component is None:
        raise ToolDefinitionError(
            "toolify: expected a non-empty `component`."
        )

    # Default filtering for routes that do not inherit an existing invokable's
    # filtering policy. Local AtomicInvokable wrapping handles inheritance in
    # its own branch below.
    default_filter_flag = (
        filter_extraneous_inputs
        if filter_extraneous_inputs is not None
        else True
    )

    # Local wrapper overrides are identity/filter settings for local Tool
    # wrapping. Passing any of these with an existing Tool means the caller wants
    # a new wrapper Tool, not mutation of the original Tool.
    local_tool_override_requested = (
        name is not None
        or namespace is not None
        or description is not None
        or filter_extraneous_inputs is not None
    )

    # Treat None, "", and whitespace as "not provided" for local-route rejection.
    # Remote routes still perform their own required-name validation.
    remote_name_provided = remote_name is not None and bool(str(remote_name).strip())

    # 1) Existing Tool.
    #
    # A Tool is already normalized. With no local wrapper overrides, return it
    # unchanged. With local wrapper overrides, create a new Tool wrapping the
    # existing Tool so the original identity and filter policy are not mutated.
    if isinstance(component, Tool):
        if remote_name_provided:
            raise ToolDefinitionError(
                "toolify: `remote_name` is only valid for MCPClientHub or "
                "PyA2AtomicClient components; got `remote_name` for an existing Tool."
            )

        if not local_tool_override_requested:
            return component

        resolved_filter = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else component.filter_extraneous_inputs
        )

        return Tool(
            function=component,
            name=name if name is not None else component.name,
            namespace=namespace if namespace is not None else component.namespace,
            description=description if description is not None else component.description,
            filter_extraneous_inputs=resolved_filter,
        )

    # 2) Non-tool AtomicInvokable.
    #
    # Base Tool now owns the old AdapterTool job: reuse the invokable's declared
    # schema and call through the invokable's dict-first invoke/async_invoke path.
    if isinstance(component, AtomicInvokable):
        if remote_name_provided:
            raise ToolDefinitionError(
                "toolify: `remote_name` is only valid for MCPClientHub or "
                "PyA2AtomicClient components; got `remote_name` for a local AtomicInvokable."
            )

        resolved_filter = (
            filter_extraneous_inputs
            if filter_extraneous_inputs is not None
            else component.filter_extraneous_inputs
        )

        return Tool(
            function=component,
            name=name,
            namespace=namespace,
            description=description,
            filter_extraneous_inputs=resolved_filter,
        )

    # 3) MCP client hub -> MCPProxyTool.
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
            filter_extraneous_inputs=default_filter_flag,
        )

    # 4) PyA2AtomicClient -> PyA2AtomicTool.
    if isinstance(component, PyA2AtomicClient):
        resolved_remote_name = str(remote_name or "").strip()
        if not resolved_remote_name:
            raise ToolDefinitionError(
                "toolify: `remote_name` is required when `component` is a PyA2AtomicClient."
            )

        return PyA2AtomicTool(
            remote_name=resolved_remote_name,
            name=name,
            namespace=namespace,
            description=description,
            client=component,
            filter_extraneous_inputs=default_filter_flag,
        )

    # 5) Raw callable -> Tool.
    if callable(component):
        if remote_name_provided:
            raise ToolDefinitionError(
                "toolify: `remote_name` is only valid for MCPClientHub or "
                "PyA2AtomicClient components; got `remote_name` for a plain callable."
            )

        resolved_name = name
        if not resolved_name:
            try:
                resolved_name = component.__name__
            except Exception as exc:  # noqa: BLE001
                raise ToolDefinitionError(
                    "toolify: `name` is required when toolifying a callable and "
                    "could not be inferred from __name__."
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
            filter_extraneous_inputs=default_filter_flag,
        )

    raise ToolDefinitionError(
        "toolify: unsupported `component` type. Expected Tool | AtomicInvokable | "
        "Callable | MCPClientHub | PyA2AtomicClient."
    )


def batch_toolify(
    sources: list[AtomicInvokable | Callable[..., Any] | MCPClientHub | PyA2AtomicClient] | None = None,
    *,
    batch_namespace: str | None = None,
    batch_filter_inputs: Optional[bool] = None,
) -> list[Tool]:
    """
    Toolify every provided source into Tool objects.

    Behavior
    --------
    - Local AtomicInvokables and callables are each toolified into one Tool.
    - Existing Tool instances are returned unchanged unless batch-level wrapper
      overrides are provided, in which case a new Tool wrapper is created.
    - Each MCPClientHub is expanded by listing all remote tools and toolifying
      each one into its own MCPProxyTool.
    - Each PyA2AtomicClient is expanded by listing all remote invokables and
      toolifying each one into its own PyA2AtomicTool.

    Parameters
    ----------
    sources : list[AtomicInvokable | Callable[..., Any] | MCPClientHub | PyA2AtomicClient]
        Mixed list of local invokables, callables, tools, and/or remote client
        hubs.
    batch_namespace : str | None
        Namespace override passed to each produced local or remote tool.
    batch_filter_inputs : Optional[bool]
        Filter flag override passed to each produced tool.

    Returns
    -------
    list[Tool]
        Flat list of all produced tools, in source order, with MCP hubs expanded
        in the order returned by each hub's list_tools() and PyA2Atomic clients
        expanded in the order returned by each client's list_invokables().
    """
    tools: list[Tool] = []

    for source in sources or []:
        if isinstance(source, MCPClientHub):
            remote_tools = source.list_tools()
            hub_tools: list[Tool] = []

            for remote_name in remote_tools:
                hub_tools.append(
                    toolify(
                        source,
                        remote_name=remote_name,
                        namespace=batch_namespace,
                        filter_extraneous_inputs=batch_filter_inputs,
                    )
                )

            tools.extend(hub_tools)
            continue

        if isinstance(source, PyA2AtomicClient):
            client_tools: list[Tool] = []

            for remote_name in source.list_invokables():
                client_tools.append(
                    toolify(
                        source,
                        remote_name=remote_name,
                        namespace=batch_namespace,
                        filter_extraneous_inputs=batch_filter_inputs,
                    )
                )

            tools.extend(client_tools)
            continue

        tools.append(
            toolify(
                source,
                namespace=batch_namespace,
                filter_extraneous_inputs=batch_filter_inputs,
            )
        )

    return tools