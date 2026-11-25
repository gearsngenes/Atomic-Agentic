import pytest

from atomic_agentic.a2a import A2AClientHub
from atomic_agentic.mcp import MCPClientHub


@pytest.mark.parametrize("bad", [0, -3, True, "10", float("nan")])
def test_a2a_hub_rejects_bad_timeout(bad) -> None:
    with pytest.raises(ValueError, match="timeout_seconds must be a positive number"):
        A2AClientHub("http://localhost:1", "JSONRPC", timeout_seconds=bad)


@pytest.mark.parametrize("bad", ["", "  ", 5])
def test_a2a_hub_rejects_bad_relative_card_path(bad) -> None:
    with pytest.raises(ValueError, match="relative_card_path"):
        A2AClientHub("http://localhost:1", "JSONRPC", relative_card_path=bad)


@pytest.mark.parametrize("bad", [0, -1, True, "5"])
def test_mcp_hub_rejects_bad_read_timeout(bad) -> None:
    with pytest.raises(ValueError, match="read_timeout_seconds must be a positive number"):
        MCPClientHub("python", "stdio", read_timeout_seconds=bad)
