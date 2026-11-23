from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from atomic_agentic.models.results.tools import A2AProxyToolResult, MCPToolResult


def _envelope() -> dict:
    started_at = datetime.now(timezone.utc)
    return dict(
        result="payload",
        invoker_id="MCPProxyTool.search",
        started_at=started_at,
        ended_at=started_at + timedelta(seconds=1),
    )


def make_mcp(**overrides) -> MCPToolResult:
    kwargs = dict(_envelope(), transport_mode="stdio", remote_name="search", location="python")
    kwargs.update(overrides)
    return MCPToolResult(**kwargs)


def make_a2a(**overrides) -> A2AProxyToolResult:
    kwargs = dict(
        _envelope(), transport_mode="JSONRPC", location="http://example.test", persistent=False
    )
    kwargs.update(overrides)
    return A2AProxyToolResult(**kwargs)


class TestMCPToolResult:
    def test_fields_are_stripped(self) -> None:
        result = make_mcp(location="  python  ", remote_name=" search ", transport_mode=" stdio ")

        assert (result.transport_mode, result.remote_name, result.location) == (
            "stdio",
            "search",
            "python",
        )

    def test_location_is_required(self) -> None:
        kwargs = dict(_envelope(), transport_mode="stdio", remote_name="search")

        with pytest.raises(TypeError):
            MCPToolResult(**kwargs)

    @pytest.mark.parametrize("field", ["transport_mode", "remote_name", "location"])
    @pytest.mark.parametrize("value", ["   ", "", 5, None])
    def test_blank_or_non_str_rejected(self, field: str, value) -> None:
        with pytest.raises(TypeError, match=field):
            make_mcp(**{field: value})

    def test_to_dict_keys(self) -> None:
        data = make_mcp().to_dict()

        assert data["transport_mode"] == "stdio"
        assert data["remote_name"] == "search"
        assert data["location"] == "python"
        assert "endpoint" not in data
        assert "command" not in data


class TestA2AProxyToolResult:
    def test_remote_name_optional(self) -> None:
        assert make_a2a().remote_name is None
        assert make_a2a(remote_name=" add ").remote_name == "add"

    def test_location_is_required(self) -> None:
        kwargs = dict(_envelope(), transport_mode="JSONRPC", persistent=False)

        with pytest.raises(TypeError):
            A2AProxyToolResult(**kwargs)

    @pytest.mark.parametrize("field", ["transport_mode", "location"])
    @pytest.mark.parametrize("value", ["   ", "", 5, None])
    def test_blank_or_non_str_rejected(self, field: str, value) -> None:
        with pytest.raises(TypeError, match=field):
            make_a2a(**{field: value})

    @pytest.mark.parametrize("value", ["  ", 5])
    def test_bad_remote_name_rejected(self, value) -> None:
        with pytest.raises(TypeError, match="remote_name"):
            make_a2a(remote_name=value)

    def test_persistent_must_be_bool(self) -> None:
        with pytest.raises(TypeError, match="persistent"):
            make_a2a(persistent="no")

    def test_to_dict_keys(self) -> None:
        data = make_a2a(remote_name="add").to_dict()

        assert data["transport_mode"] == "JSONRPC"
        assert data["location"] == "http://example.test"
        assert data["persistent"] is False
        assert data["remote_name"] == "add"
        assert "base_url" not in data
        assert "skill_id" not in data
