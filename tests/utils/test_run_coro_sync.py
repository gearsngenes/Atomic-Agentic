from __future__ import annotations

import asyncio
import threading

from atomic_agentic.utils.core import run_coro_sync, start_background_loop, stop_background_loop


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


class TestRunCoroSyncWithLoop:
    def test_dispatches_onto_the_given_loop_instead_of_a_fresh_one(self) -> None:
        loop, thread = start_background_loop()
        try:
            async def current_thread_name() -> str:
                return threading.current_thread().name

            assert run_coro_sync(current_thread_name(), loop=loop) == thread.name
        finally:
            stop_background_loop(loop, thread)

    def test_blocks_the_calling_thread_until_completion(self) -> None:
        loop, thread = start_background_loop()
        try:
            async def sample() -> int:
                await asyncio.sleep(0.05)
                return 7

            assert run_coro_sync(sample(), loop=loop) == 7
        finally:
            stop_background_loop(loop, thread)

    def test_propagates_exceptions_from_the_dispatched_coroutine(self) -> None:
        loop, thread = start_background_loop()
        try:
            async def failing() -> None:
                raise ValueError("boom")

            try:
                run_coro_sync(failing(), loop=loop)
            except ValueError as exc:
                assert str(exc) == "boom"
            else:
                raise AssertionError("expected ValueError to propagate")
        finally:
            stop_background_loop(loop, thread)
