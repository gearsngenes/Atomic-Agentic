from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .atomic import AtomicResult

__all__ = ["ToolResult", "MCPToolResult", "PyA2AtomicToolResult"]


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
