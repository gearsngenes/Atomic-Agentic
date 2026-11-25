from __future__ import annotations

import sys
from pathlib import Path

import pytest

from atomic_agentic.exceptions import ToolRegistrationError
from atomic_agentic.mcp.mcp_client_hub import MCPClientHub

from ..agents.conftest import make_stub_tool_agent

_FIXTURE_SERVER = str(Path(__file__).parent / "_stdio_fixture_server.py")


class TestMCPClientHubRealStdioPersistence:
    """Regression guard for the anyio same-*task* open/close constraint
    (see MCPClientHub._run_persistent_session's docstring): a persistent
    stdio hub's underlying ClientSession/transport are anyio task-group-
    owning resources whose cancel scopes may only be exited by the same
    asyncio Task that entered them. test_client_hub.py's persistent-mode
    tests are all monkeypatched fakes that never exercise a real anyio task
    group, so they could not have caught -- and can't guard against a
    regression of -- the bug this class targets (confirmed by direct
    reproduction during the Pass 5 implementation: dispatching "open" and
    "close" as two independently-dispatched coroutines raised "Attempted to
    exit cancel scope in a different task than it was entered in" on the
    first live close()). Spawns a real subprocess end to end, no mocks.
    """

    def test_construct_call_close_does_not_raise(self) -> None:
        hub = MCPClientHub(
            sys.executable,
            "stdio",
            persistent=True,
            args=[_FIXTURE_SERVER],
        )
        try:
            result = hub.call_tool("echo", {"value": "hello"})
            assert result["isError"] is False
            assert hub.is_connected is True
        finally:
            hub.close()  # The exact call that raised RuntimeError pre-fix.

        assert hub.is_connected is False

    def test_close_is_idempotent(self) -> None:
        hub = MCPClientHub(
            sys.executable,
            "stdio",
            persistent=True,
            args=[_FIXTURE_SERVER],
        )

        hub.close()
        hub.close()  # Second call must be a safe no-op, not a re-raise.

        assert hub.is_connected is False

    def test_refresh_respawns_process_and_remains_usable(self) -> None:
        """refresh() retires the old holder task through the same
        same-task-close mechanism close() uses (_do_refresh -> old
        close_event.set() + await old_holder_task) -- a second real path
        into the exact constraint this file guards."""
        hub = MCPClientHub(
            sys.executable,
            "stdio",
            persistent=True,
            args=[_FIXTURE_SERVER],
        )
        try:
            hub.call_tool("echo", {"value": "before"})

            hub.refresh(client_kwargs={})

            result = hub.call_tool("echo", {"value": "after"})
            assert result["isError"] is False
        finally:
            hub.close()


class TestRealHubRegistrationNaming:
    """Hub tools register under ``<hub.name>_<remote name>`` (raw name for an
    unnamed hub) against a real stdio fixture server."""

    def test_named_hub_tool_registers_prefixed(self) -> None:
        hub = MCPClientHub(sys.executable, "stdio", args=[_FIXTURE_SERVER], name="fx")
        try:
            agent = make_stub_tool_agent()
            agent.register_tools([hub])
            assert agent.has_tool("fx_echo")
            assert not agent.has_tool("echo")
            assert agent.list_tools()["fx_echo"].remote_name == "echo"
        finally:
            hub.close()

    def test_unnamed_hub_tool_registers_raw(self) -> None:
        hub = MCPClientHub(sys.executable, "stdio", args=[_FIXTURE_SERVER])
        try:
            agent = make_stub_tool_agent()
            agent.register_tools([hub])
            assert agent.has_tool("echo")
        finally:
            hub.close()

    def test_two_hubs_same_remote_names_prefix_apart(self) -> None:
        hub_a = MCPClientHub(sys.executable, "stdio", args=[_FIXTURE_SERVER], name="a")
        hub_b = MCPClientHub(sys.executable, "stdio", args=[_FIXTURE_SERVER], name="b")
        try:
            agent = make_stub_tool_agent()
            agent.register_tools([hub_a, hub_b])
            assert agent.has_tool("a_echo")
            assert agent.has_tool("b_echo")
        finally:
            hub_a.close()
            hub_b.close()

    def test_two_unnamed_hubs_collide_in_batch(self) -> None:
        hub_a = MCPClientHub(sys.executable, "stdio", args=[_FIXTURE_SERVER])
        hub_b = MCPClientHub(sys.executable, "stdio", args=[_FIXTURE_SERVER])
        try:
            agent = make_stub_tool_agent()
            with pytest.raises(ToolRegistrationError):
                agent.register_tools([hub_a, hub_b])
        finally:
            hub_a.close()
            hub_b.close()


class TestRealHubReadTimeout:
    """The read timeout must be handed to the SDK as float seconds (mcp 2.x);
    the unit tests fake ClientSession, so only a real session proves it."""

    def test_read_timeout_with_real_session(self) -> None:
        hub = MCPClientHub(
            sys.executable,
            "stdio",
            args=[_FIXTURE_SERVER],
            read_timeout_seconds=30.0,
        )
        assert list(hub.list_tools()) == ["echo"]
        assert hub.call_tool("echo", {"value": "hi"})["isError"] is False
