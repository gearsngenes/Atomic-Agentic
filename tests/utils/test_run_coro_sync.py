from __future__ import annotations

import asyncio

from atomic_agentic.utils.core import run_coro_sync


class TestRunCoroSync:
    def test_run_coro_sync_returns_result_without_running_loop(self) -> None:
        async def sample() -> int:
            return 42

        assert run_coro_sync(sample()) == 42

    def test_run_coro_sync_works_when_event_loop_is_already_running(self) -> None:
        async def outer() -> str:
            async def inner() -> str:
                return "ok"

            return run_coro_sync(inner())

        assert asyncio.run(outer()) == "ok"
