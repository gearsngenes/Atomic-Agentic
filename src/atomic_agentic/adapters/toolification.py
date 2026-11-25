from __future__ import annotations

from typing import Any, Callable

from ..a2a import A2AClientHub
from ..constants.core import NO_VAL
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
    component: AtomicInvokable | Callable[..., Any] | MCPClientHub | A2AClientHub = NO_VAL,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
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

    3) If ``component`` is an ``MCPClientHub`` or ``A2AClientHub``:
       - MCP: ``name`` selects the remote tool from the hub's stored catalog;
         a missing, blank or unknown name raises the hub's ``MCPToolError``
         showing what was given;
       - A2A: a given ``name`` selects skill mode (bound to that skill; an
         unknown or blank one raises the hub's ``A2AProxyError``); only
         ``None`` selects generic mode;
       - construct and return the proxy; ``description``, when given,
         replaces the remote description through the proxy's setter.

    4) If ``component`` is a plain callable:
       - construct and return a plain ``Tool``; the name defaults to
         ``__name__`` and the description to the docstring (inference and the
         nameless-callable error are the ``Tool`` constructor's).

    Decorator form
    --------------
    5) If ``component`` is omitted, return a decorator that builds the Tool
       from the function it receives, exactly as ``to_tool(fn, name,
       description)``. So ``@to_tool``, ``@to_tool(name=..., description=...)``
       and ``@to_tool()`` all work. ``to_tool(None)`` is still an error: only
       omission selects the decorator.

    For a hub, ``name`` selects the remote tool or skill and the returned proxy
    is named ``<hub.name>_<name>`` (the raw name for an unnamed hub); for every
    other component ``name`` names the tool itself.

    Raises
    ------
    ToolDefinitionError
        For invalid routing combinations, missing required metadata, or
        unsupported component types.
    """
    # 5) Decorator form: configure now, build when the function arrives.
    if component is NO_VAL:
        def decorator(fn: Callable[..., Any]) -> Tool:
            return to_tool(fn, name, description)

        return decorator

    if component is None:
        raise ToolDefinitionError(
            "to_tool: expected a non-empty `component`."
        )

    # A positional string is almost certainly @to_tool("name").
    if isinstance(component, str) and name is None and description is None:
        raise ToolDefinitionError(
            "to_tool: keyword arguments only for the decorator form; use "
            "@to_tool(name=...), not a positional value."
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

    # 3) Client hub -> proxy tool. `name` goes to the proxy untouched: only
    # None means generic mode (A2A); any other bad name fails in the hub's
    # get_tool/get_skill showing what was typed.
    if isinstance(component, (MCPClientHub, A2AClientHub)):
        proxy_cls = MCPProxyTool if isinstance(component, MCPClientHub) else A2AProxyTool
        proxy = proxy_cls(component, name)
        if description is not None:
            proxy.description = description  # setter validates
        return proxy

    # 4) Raw callable -> Tool.
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
    - Each MCPClientHub is expanded from its stored catalog (no network)
      into one MCPProxyTool per remote tool.
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
            tools.extend(MCPProxyTool(source, n) for n in source.list_tools())
            continue

        if isinstance(source, A2AClientHub):
            tools.extend(A2AProxyTool(source, s) for s in source.list_skills())
            tools.append(A2AProxyTool(source))
            continue

        tools.append(
            to_tool(
                source,
            )
        )

    return tools
