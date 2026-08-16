from __future__ import annotations

import asyncio

import pytest

from atomic_agentic.utils.core import run_coro_sync, start_background_loop, stop_background_loop


class TestStartBackgroundLoop:
    def test_thread_is_a_daemon(self) -> None:
        loop, thread = start_background_loop()
        try:
            assert thread.daemon is True
        finally:
            stop_background_loop(loop, thread)

    def test_loop_accepts_dispatched_work_immediately(self) -> None:
        loop, thread = start_background_loop()
        try:
            async def sample() -> int:
                return 42

            assert run_coro_sync(sample(), loop=loop) == 42
        finally:
            stop_background_loop(loop, thread)


class TestStopBackgroundLoop:
    def test_stops_the_loop_and_joins_the_thread(self) -> None:
        loop, thread = start_background_loop()

        stop_background_loop(loop, thread)

        assert not thread.is_alive()

    def test_dispatching_after_stop_raises(self) -> None:
        loop, thread = start_background_loop()
        stop_background_loop(loop, thread)

        async def sample() -> int:
            return 1

        with pytest.raises(RuntimeError):
            asyncio.run_coroutine_threadsafe(sample(), loop)
