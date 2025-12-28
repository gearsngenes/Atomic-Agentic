from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence, Union

from ..core.Exceptions import ToolDefinitionError
from ..core.Invokable import AtomicInvokable
from .base import Tool
from .a2a import A2AgentTool
from .adapter import AdapterTool
from .mcp import MCPProxyTool, list_mcp_tools

__all__ = ["toolify"]


def toolify(
    component: Union[AtomicInvokable, str, Callable[..., Any]],
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
        - str        → treated as a remote endpoint URL. Toolify will attempt:
                       1) MCP discovery (default): list tools via MCP and build `MCPProxyTool` instances.
                       2) A2A fallback: if MCP discovery fails, build a single `A2AgentTool`.

    Keyword-only configuration
    --------------------------
    name : Optional[str]
        For callables: required logical name (falls back to function __name__ when omitted).
        For MCP URLs: optional remote tool name. If omitted and MCP discovery succeeds, all tools on the
        server are instantiated (subject to `include`/`exclude`).
        NOTE: If MCP discovery fails and Toolify falls back to A2A, `name` is ignored.
    description : Optional[str]
        Human-readable description. For MCP tools, this is only a fallback:
        the remote description wins when present.
    namespace : Optional[str]
        Logical namespace for the Tool.
    source : Optional[str]
        Backwards-compatibility alias for `namespace` (legacy API).
    headers : Optional[Mapping[str, str]]
        Transport headers for MCP/A2A (auth, etc.). The *presence* of this key
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
        For invalid inputs, missing required metadata, or remote discovery issues.
    """

    # Keep an original view so we can detect presence of keys (e.g. headers).
    original_kwargs = dict(kwargs)

    # Extract supported kwargs
    name = kwargs.pop("name", None)
    description = kwargs.pop("description", None)
    namespace_kw = kwargs.pop("namespace", None)
    headers = kwargs.pop("headers", None)
    include = kwargs.pop("include", None)
    exclude = kwargs.pop("exclude", None)

    # If any unexpected kwargs remain, surface them early (fail-fast contract)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise ToolDefinitionError(f"toolify: unexpected keyword argument(s): {unexpected}")

    # 1) Passthrough if already a Tool
    if isinstance(component, Tool):
        return [component]

    # 2) Agent → AgentTool
    if isinstance(component, AtomicInvokable):
        return [AdapterTool(component)]

    # 3) String → MCP-by-default, then A2A fallback
    if isinstance(component, str):
        url = component.strip()

        # Enforce contract: headers key must be present (value may be None).
        if "headers" not in original_kwargs:
            raise ToolDefinitionError(
                "toolify: 'headers' keyword must be provided (it may be None) "
                "when toolifying a remote endpoint string."
            )

        # Both MCP and A2A implementations here are HTTP-based.
        if not (url.startswith("http://") or url.startswith("https://")):
            raise ToolDefinitionError(
                "toolify: when `component` is a string it must be an HTTP(S) URL "
                "(e.g. 'http://localhost:8000/mcp' for MCP or 'http://localhost:8001' for A2A)."
            )

        namespace = namespace_kw

        # (a) MCP probe: attempt discovery first to distinguish MCP vs A2A.
        mcp_exc: Exception | None = None
        all_tools: Mapping[str, Any] | None = None
        try:
            all_tools = list_mcp_tools(server_url=url, headers=headers)
        except Exception as exc:  # noqa: BLE001
            mcp_exc = exc

        if all_tools is not None:
            # MCP path (discovery succeeded): preserve existing semantics.
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

            return [
                MCPProxyTool(
                    server_url=url,
                    tool_name=tool_name,
                    namespace=namespace,
                    headers=headers,
                )
                for tool_name in names
            ]

        # (b) A2A fallback: MCP discovery failed, so attempt to build an A2AgentTool.
        a2a_exc: Exception | None = None
        try:
            # Instantiation fetches the agent card; failures here strongly indicate
            # this is not an A2A agent endpoint (or auth/network is invalid).
            return [A2AgentTool(url=url, headers=headers)]
        except Exception as exc:  # noqa: BLE001
            a2a_exc = exc

        # (c) Neither protocol matched.
        raise ToolDefinitionError(
            "toolify: failed to toolify remote endpoint string as either MCP or A2A.\n"
            f"- MCP discovery error: {mcp_exc!r}\n"
            f"- A2A agent error: {a2a_exc!r}"
        )

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

        namespace = namespace_kw
        effective_description = (description or component.__doc__ or "undescribed").strip()

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
        "toolify: unsupported input type. Expected Tool | Agent | callable | endpoint URL string."
    )
