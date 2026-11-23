from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .atomic import AtomicResult

__all__ = ["ToolResult", "MCPToolResult", "A2AProxyToolResult"]


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
    location:
        The server command for stdio, the URL for SSE and streamable HTTP.
    """

    transport_mode: str
    remote_name: str
    location: str

    def __post_init__(self) -> None:
        if not isinstance(self.transport_mode, str) or not self.transport_mode.strip():
            raise TypeError("transport_mode must be a non-empty string.")
        if not isinstance(self.remote_name, str) or not self.remote_name.strip():
            raise TypeError("remote_name must be a non-empty string.")
        if not isinstance(self.location, str) or not self.location.strip():
            raise TypeError("location must be a non-empty string.")
        object.__setattr__(self, "transport_mode", self.transport_mode.strip())
        object.__setattr__(self, "remote_name", self.remote_name.strip())
        object.__setattr__(self, "location", self.location.strip())
        AtomicResult.__post_init__(self)

    def to_dict(self) -> dict[str, Any]:
        data = AtomicResult.to_dict(self)
        data.update(
            {
                "transport_mode": self.transport_mode,
                "remote_name": self.remote_name,
                "location": self.location,
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
    location:
        The remote A2A server's URL this proxy is bound to.
    remote_name:
        The bound skill's name in skill mode; None in generic mode.
    """

    transport_mode: str
    location: str
    remote_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transport_mode, str) or not self.transport_mode.strip():
            raise TypeError("transport_mode must be a non-empty string.")
        if not isinstance(self.location, str) or not self.location.strip():
            raise TypeError("location must be a non-empty string.")
        if self.remote_name is not None:
            if not isinstance(self.remote_name, str) or not self.remote_name.strip():
                raise TypeError("remote_name must be a non-empty string when provided.")
            object.__setattr__(self, "remote_name", self.remote_name.strip())
        object.__setattr__(self, "transport_mode", self.transport_mode.strip())
        object.__setattr__(self, "location", self.location.strip())
        AtomicResult.__post_init__(self)

    def to_dict(self) -> dict[str, Any]:
        data = AtomicResult.to_dict(self)
        data.update(
            {
                "transport_mode": self.transport_mode,
                "location": self.location,
                "remote_name": self.remote_name,
            }
        )
        return data
