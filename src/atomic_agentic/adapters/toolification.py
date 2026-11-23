from __future__ import annotations

from typing import Any, Callable

from ..a2a import A2AClientHub
from ..exceptions import ToolDefinitionError
from ..invokables.atomic_invokable import AtomicInvokable
from ..mcp import MCPClientHub
from ..tools.a2a_sdk import A2AProxyTool
from ..tools.base import Tool
from ..tools.mcp import MCPProxyTool

__all__ = ["to_tool", "to_tools"]


# ───────────────────────────────────────────────────────────────────────────────
# to_tool
# ───────────────────────────────────────────────────────────────────────────────
def to_tool(
    component: AtomicInvokable | Callable[..., Any] | MCPClientHub | A2AClientHub,
    name: str | None = None,
    description: str | None = None,
) -> Tool:
    """
    Normalize a single component into a single Tool instance.

    Routing rules
    -------------
    1) If ``component`` is already a ``Tool``:
       - return it unchanged when no local wrapper overrides are provided;
       - if ``name`` or ``description`` is provided, return
         a new ``Tool`` that delegates to ``component`` by reference (calls
         its real ``invoke``/``async_invoke`` under the hood) under the
         requested identity — the original is never mutated. Works
         uniformly for plain ``Tool`` and ``Tool`` subclasses
         (``MCPProxyTool``, ``A2AProxyTool``) alike.

    2) If ``component`` is a non-tool ``AtomicInvokable``:
       - wrap it the same way — a new ``Tool`` delegating to ``component``
         by reference under the requested (or inferred) identity.

    3) If ``component`` is an ``MCPClientHub``:
       - ``name`` is required and selects the remote tool;
       - construct and return ``MCPProxyTool``; ``description``, when given,
         replaces the remote description through the proxy's setter.

    4) If ``component`` is an ``A2AClientHub``:
       - ``name`` is OPTIONAL here — the one deliberate asymmetry vs. route 3.
         Present & non-blank selects skill mode (bound to that skill);
         absent/blank selects generic mode;
       - construct and return ``A2AProxyTool``, with ``description`` applied
         as in route 3. Its own ``ToolDefinitionError`` (unknown skill)
         propagates unwrapped.

    5) If ``component`` is a plain callable:
       - construct and return a plain ``Tool``; the name defaults to
         ``__name__`` and the description to the docstring (inference and the
         nameless-callable error are the ``Tool`` constructor's).

    For a hub, ``name`` selects the remote tool or skill and the returned proxy
    is named ``<hub.name>_<name>`` (the raw name for an unnamed hub); for every
    other component ``name`` names the tool itself.

    Raises
    ------
    ToolDefinitionError
        For invalid routing combinations, missing required metadata, or
        unsupported component types.
    """
    if component is None:
        raise ToolDefinitionError(
            "to_tool: expected a non-empty `component`."
        )

    # Local wrapper overrides are identity settings for local Tool wrapping.
    # Passing any of these with an existing Tool means the caller wants a new
    # wrapper Tool, not mutation of the original Tool.
    local_tool_override_requested = (
        name is not None
        or description is not None
    )

    # 1) Existing Tool.
    #
    # A Tool is already normalized. With no local wrapper overrides, return it
    # unchanged. With local wrapper overrides, create a new Tool wrapping the
    # existing Tool so the original identity is not mutated.
    if isinstance(component, Tool):
        if not local_tool_override_requested:
            return component

        return Tool(component, name, description)

    # 2) Non-tool AtomicInvokable.
    #
    # Same treatment as route 1's override case: a new Tool delegating to
    # component by reference.
    if isinstance(component, AtomicInvokable):
        return Tool(component, name, description)

    # 3) MCP client hub -> MCPProxyTool.
    if isinstance(component, MCPClientHub):
        remote = str(name or "").strip()
        if not remote:
            raise ToolDefinitionError(
                "to_tool: `name` is required when `component` is an MCPClientHub."
            )

        proxy = MCPProxyTool(component, remote)
        if description is not None:
            proxy.description = description  # setter validates
        return proxy

    # 4) A2AClientHub -> A2AProxyTool. Unlike route 3, name is optional:
    # absent/blank selects generic mode, a deliberate outcome.
    if isinstance(component, A2AClientHub):
        remote = str(name).strip() if name is not None and str(name).strip() else None

        proxy = A2AProxyTool(component, remote)
        if description is not None:
            proxy.description = description
        return proxy

    # 5) Raw callable -> Tool.
    if callable(component):
        if name is not None and not isinstance(name, str):
            raise ToolDefinitionError(
                "to_tool: `name` must be a string when provided for callables."
            )

        if description is not None and not isinstance(description, str):
            raise ToolDefinitionError(
                "to_tool: `description` must be a string when provided for callables."
            )

        # Name and description inference (and the nameless-callable error)
        # belong to the Tool constructor.
        return Tool(component, name, description)

    raise ToolDefinitionError(
        "to_tool: unsupported `component` type. Expected Tool | AtomicInvokable | "
        "Callable | MCPClientHub | A2AClientHub."
    )


def to_tools(
    sources: list[AtomicInvokable | Callable[..., Any] | MCPClientHub | A2AClientHub] | None = None,
) -> list[Tool]:
    """
    Convert every provided source into Tool objects.

    Behavior
    --------
    - Callables and AtomicInvokables are each converted into one Tool
      (AtomicInvokable-backed Tools delegate to the original by reference).
    - Existing Tool instances are returned unchanged.
    - Each MCPClientHub is expanded by listing all remote tools and converting
      each one into its own MCPProxyTool.
    - No collision check is made here; duplicate names across hubs are caught
      at registration.
    - Each A2AClientHub is expanded into one skill-mode A2AProxyTool per
      discovered Atomic skill, plus exactly one trailing generic-mode
      A2AProxyTool -- skills first, generic last, unconditionally (a
      zero-skill hub still contributes its one generic tool).

    Parameters
    ----------
    sources : list[AtomicInvokable | Callable[..., Any] | MCPClientHub | A2AClientHub]
        Mixed list of local invokables, callables, Tools, and/or remote
        client hubs.

    Returns
    -------
    list[Tool]
        Flat list of all produced tools, in source order, with MCP hubs expanded
        in the order returned by each hub's list_tools(), and A2A client hubs
        expanded in list_skills() order followed by the
        one generic tool.
    """
    tools: list[Tool] = []

    for source in sources or []:
        if isinstance(source, MCPClientHub):
            remote_tools = source.list_tools()
            hub_tools: list[Tool] = []

            for remote_name in remote_tools:
                hub_tools.append(to_tool(source, name=remote_name))

            tools.extend(hub_tools)
            continue

        if isinstance(source, A2AClientHub):
            skill_tools = [
                to_tool(source, name=skill)
                for skill in source.list_skills()
            ]
            generic_tool = to_tool(source)

            tools.extend([*skill_tools, generic_tool])
            continue

        tools.append(
            to_tool(
                source,
            )
        )

    return tools
