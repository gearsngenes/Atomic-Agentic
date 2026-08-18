from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .atomic import AtomicResult

__all__ = ["ToolResult", "MCPToolResult", "PyA2AtomicToolResult", "A2AProxyToolResult"]


@dataclass(frozen=True, slots=True)
class ToolResult(AtomicResult):
    """
    Successful Tool invocation result.

    ``ToolResult.result`` is the caller-facing payload produced by the Tool.
    This subclass intentionally adds no Tool-specific fields yet.
    """


@dataclass(frozen=True, slots=True)
class MCPToolResult(ToolResult):
    """
    ToolResult carrying MCP transport/identity metadata.

    Fields
    ------
    transport_mode:
        Transport kind — one of ``"stdio"``, ``"sse"``, or
        ``"streamable_http"``.
    remote_name:
        The MCP-side tool name this proxy is bound to.
    endpoint:
        Non-None for SSE/HTTP transports; ``None`` for stdio.
    command:
        Non-None for stdio transport; ``None`` for SSE/HTTP.
    """

    transport_mode: str
    remote_name: str
    endpoint: str | None = None
    command: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transport_mode, str) or not self.transport_mode.strip():
            raise TypeError("transport_mode must be a non-empty string.")
        if not isinstance(self.remote_name, str) or not self.remote_name.strip():
            raise TypeError("remote_name must be a non-empty string.")
        if self.endpoint is not None:
            if not isinstance(self.endpoint, str) or not self.endpoint.strip():
                raise TypeError("endpoint must be a non-empty string when provided.")
            object.__setattr__(self, "endpoint", self.endpoint.strip())
        if self.command is not None:
            if not isinstance(self.command, str) or not self.command.strip():
                raise TypeError("command must be a non-empty string when provided.")
            object.__setattr__(self, "command", self.command.strip())
        object.__setattr__(self, "transport_mode", self.transport_mode.strip())
        object.__setattr__(self, "remote_name", self.remote_name.strip())
        AtomicResult.__post_init__(self)

    def to_dict(self) -> dict[str, Any]:
        data = AtomicResult.to_dict(self)
        data.update(
            {
                "transport_mode": self.transport_mode,
                "remote_name": self.remote_name,
                "endpoint": self.endpoint,
                "command": self.command,
            }
        )
        return data


@dataclass(frozen=True, slots=True)
class PyA2AtomicToolResult(ToolResult):
    """
    ToolResult carrying A2A transport/identity metadata.

    Fields
    ------
    url:
        The A2A host URL this proxy is bound to.
    remote_name:
        The invokable name registered on the remote host.
    invokable_type:
        Class name of the remote invokable as reported by the host
        (e.g. ``"Tool"``, ``"Agent"``, ``"ToolAgent"``). Reflects the
        metadata snapshot at the time of this invocation.
    """

    url: str
    remote_name: str
    invokable_type: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("url", self.url),
            ("remote_name", self.remote_name),
            ("invokable_type", self.invokable_type),
        ):
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{field_name} must be a non-empty string.")
            object.__setattr__(self, field_name, value.strip())
        AtomicResult.__post_init__(self)

    def to_dict(self) -> dict[str, Any]:
        data = AtomicResult.to_dict(self)
        data.update(
            {
                "url": self.url,
                "remote_name": self.remote_name,
                "invokable_type": self.invokable_type,
            }
        )
        return data


@dataclass(frozen=True, slots=True)
class A2AProxyToolResult(ToolResult):
    """
    ToolResult carrying a2a-sdk transport/identity metadata.

    Fields
    ------
    transport_mode:
        One of A2AClientHub's VALID_TRANSPORT_MODES ("JSONRPC", "HTTP+JSON",
        "GRPC").
    base_url:
        The remote A2A server's base_url this proxy is bound to.
    persistent:
        Whether the backing A2AClientHub holds a persistent connection.
    skill_id:
        Non-None in AA-skill mode (the bound remote skill name); None in
        generic mode.
    """

    transport_mode: str
    base_url: str
    persistent: bool
    skill_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transport_mode, str) or not self.transport_mode.strip():
            raise TypeError("transport_mode must be a non-empty string.")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise TypeError("base_url must be a non-empty string.")
        if not isinstance(self.persistent, bool):
            raise TypeError("persistent must be a bool.")
        if self.skill_id is not None:
            if not isinstance(self.skill_id, str) or not self.skill_id.strip():
                raise TypeError("skill_id must be a non-empty string when provided.")
            object.__setattr__(self, "skill_id", self.skill_id.strip())
        object.__setattr__(self, "transport_mode", self.transport_mode.strip())
        object.__setattr__(self, "base_url", self.base_url.strip())
        AtomicResult.__post_init__(self)

    def to_dict(self) -> dict[str, Any]:
        data = AtomicResult.to_dict(self)
        data.update(
            {
                "transport_mode": self.transport_mode,
                "base_url": self.base_url,
                "persistent": self.persistent,
                "skill_id": self.skill_id,
            }
        )
        return data
