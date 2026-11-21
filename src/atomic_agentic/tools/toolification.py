from __future__ import annotations

from typing import Any, Callable

from ..exceptions import ToolDefinitionError
from ..invokables.atomic_invokable import AtomicInvokable
from ..invokables.prompt_template import PromptTemplate
from ..mcp import MCPClientHub
from ..a2a import A2AClientHub
from .base import Tool
from .mcp import MCPProxyTool
from .a2a_sdk import A2AProxyTool

__all__ = ["toolify", "batch_toolify", "to_invokable"]


# ───────────────────────────────────────────────────────────────────────────────
# toolify
# ───────────────────────────────────────────────────────────────────────────────
def toolify(
    component: AtomicInvokable | Callable[..., Any] | MCPClientHub | A2AClientHub,
    name: str | None = None,
    description: str | None = None,
    *,
    remote_name: str | None = None,
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
       - ``remote_name`` is required;
       - construct and return ``MCPProxyTool``.

    4) If ``component`` is an ``A2AClientHub``:
       - ``remote_name`` is OPTIONAL here — the one deliberate asymmetry vs.
         route 3. Present & non-blank selects skill mode (bound to that
         skill id); absent/blank selects generic mode (``skill_id=None``);
       - construct and return ``A2AProxyTool``. Its own ``ToolDefinitionError``
         (unknown ``skill_id``) propagates unwrapped.

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

    # Local wrapper overrides are identity settings for local Tool wrapping.
    # Passing any of these with an existing Tool means the caller wants a new
    # wrapper Tool, not mutation of the original Tool.
    local_tool_override_requested = (
        name is not None
        or description is not None
    )

    # Treat None, "", and whitespace as "not provided" for local-route rejection.
    # Remote routes still perform their own required-name validation.
    remote_name_provided = remote_name is not None and bool(str(remote_name).strip())

    # 1) Existing Tool.
    #
    # A Tool is already normalized. With no local wrapper overrides, return it
    # unchanged. With local wrapper overrides, create a new Tool wrapping the
    # existing Tool so the original identity is not mutated.
    if isinstance(component, Tool):
        if remote_name_provided:
            raise ToolDefinitionError(
                "toolify: `remote_name` is only valid for MCPClientHub "
                "or A2AClientHub components; got `remote_name` "
                "for an existing Tool."
            )

        if not local_tool_override_requested:
            return component

        return Tool(
            function=component,
            name=name if name is not None else component.name,
            description=description if description is not None else component._description,
        )

    # 2) Non-tool AtomicInvokable.
    #
    # Same treatment as route 1's override case: a new Tool delegating to
    # component by reference.
    if isinstance(component, AtomicInvokable):
        if remote_name_provided:
            raise ToolDefinitionError(
                "toolify: `remote_name` is only valid for MCPClientHub "
                "or A2AClientHub components; got `remote_name` "
                "for a local AtomicInvokable."
            )

        return Tool(
            function=component,
            name=name if name is not None else component.name,
            description=description if description is not None else component._description,
        )

    # 3) MCP client hub -> MCPProxyTool.
    if isinstance(component, MCPClientHub):
        resolved_remote_name = str(remote_name or "").strip()
        if not resolved_remote_name:
            raise ToolDefinitionError(
                "toolify: `remote_name` is required when `component` is an MCPClientHub."
            )

        return MCPProxyTool(
            component,
            resolved_remote_name,
            name=name,
            description=description,
        )

    # 4) A2AClientHub -> A2AProxyTool. Unlike route 3, remote_name is
    # optional here: present selects skill mode (skill_id=remote_name),
    # absent/blank selects generic mode (skill_id=None) -- both are valid,
    # deliberate outcomes, not a missing-required-arg error.
    if isinstance(component, A2AClientHub):
        resolved_skill_id = str(remote_name).strip() if remote_name_provided else None

        return A2AProxyTool(
            component,
            resolved_skill_id,
            name=name,
            description=description,
        )

    # 5) Raw callable -> Tool.
    if callable(component):
        if remote_name_provided:
            raise ToolDefinitionError(
                "toolify: `remote_name` is only valid for MCPClientHub "
                "or A2AClientHub components; got `remote_name` "
                "for a plain callable."
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
            description=effective_description,
        )

    raise ToolDefinitionError(
        "toolify: unsupported `component` type. Expected Tool | AtomicInvokable | "
        "Callable | MCPClientHub | A2AClientHub."
    )


def batch_toolify(
    sources: list[AtomicInvokable | Callable[..., Any] | MCPClientHub | A2AClientHub] | None = None,
) -> list[Tool]:
    """
    Toolify every provided source into Tool objects.

    Behavior
    --------
    - Callables and AtomicInvokables are each toolified into one Tool
      (AtomicInvokable-backed Tools delegate to the original by reference).
    - Existing Tool instances are returned unchanged.
    - Each MCPClientHub is expanded by listing all remote tools and toolifying
      each one into its own MCPProxyTool.
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
        expanded in get_atomic_skills() order followed by the
        one generic tool.
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
                    )
                )

            tools.extend(hub_tools)
            continue

        if isinstance(source, A2AClientHub):
            skill_tools = [
                toolify(source, remote_name=skill_id)
                for skill_id in source.get_atomic_skills()
            ]
            generic_tool = toolify(source)

            tools.extend([*skill_tools, generic_tool])
            continue

        tools.append(
            toolify(
                source,
            )
        )

    return tools


# ───────────────────────────────────────────────────────────────────────────────
# to_invokable
# ───────────────────────────────────────────────────────────────────────────────
def to_invokable(
    obj: str | Callable[..., Any] | AtomicInvokable,
    *,
    name: str,
    description: str,
) -> AtomicInvokable:
    """
    Coerce one construction-time value into an invokable.

    1. An ``AtomicInvokable`` is returned unchanged (no wrapper, even for a
       ``Tool``; ``name``/``description`` are ignored).
    2. A ``str`` becomes a ``PromptTemplate`` with discovered parameters. The
       string is used verbatim (no ``.strip()``); a blank string raises
       ``PromptTemplate``'s ``ValueError``.
    3. Any other callable goes through ``toolify`` with ``name`` and
       ``description``.
    4. Anything else (``None`` included) raises ``TypeError`` naming ``name``
       and the type.

    Invokables are callable, so they are tested first; ``str`` is not callable.
    Used for instructions and the pre/post stages; an agent is an invokable, so
    recursion is allowed and unguarded.
    """
    if isinstance(obj, AtomicInvokable):
        return obj
    if isinstance(obj, str):
        return PromptTemplate(template=obj, name=name, description=description)
    if callable(obj):
        return toolify(obj, name=name, description=description)
    raise TypeError(
        f"{name} must be a str, callable or AtomicInvokable; got {type(obj).__name__}."
    )