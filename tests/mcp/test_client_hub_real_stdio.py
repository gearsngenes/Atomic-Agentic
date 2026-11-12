from __future__ import annotations

import sys
from pathlib import Path

from atomic_agentic.mcp.MCPClientHub import MCPClientHub

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
            transport_mode="stdio",
            persistent=True,
            command=sys.executable,
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
            transport_mode="stdio",
            persistent=True,
            command=sys.executable,
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
            transport_mode="stdio",
            persistent=True,
            command=sys.executable,
            args=[_FIXTURE_SERVER],
        )
        try:
            hub.call_tool("echo", {"value": "before"})

            hub.refresh(client_kwargs={})

            result = hub.call_tool("echo", {"value": "after"})
            assert result["isError"] is False
        finally:
            hub.close()
