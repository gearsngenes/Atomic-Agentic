from __future__ import annotations

import asyncio

import pytest

from atomic_agentic.utils.core import run_coro_async, start_background_loop, stop_background_loop


class TestRunCoroAsync:
    def test_dispatches_onto_a_different_loop_without_blocking_the_caller(self) -> None:
        loop, thread = start_background_loop()
        try:
            progress: list[str] = []

            async def on_background_loop() -> str:
                await asyncio.sleep(0.1)
                return "background-done"

            async def caller() -> None:
                async def ticker() -> None:
                    for _ in range(5):
                        progress.append("tick")
                        await asyncio.sleep(0.02)

                ticker_task = asyncio.ensure_future(ticker())
                result = await run_coro_async(on_background_loop(), loop=loop)
                assert result == "background-done"
                await ticker_task

            asyncio.run(caller())

            # The caller's own loop kept making progress on the ticker
            # while waiting on the background dispatch -- not blocked.
            assert len(progress) == 5
        finally:
            stop_background_loop(loop, thread)

    def test_propagates_exceptions_from_the_dispatched_coroutine(self) -> None:
        loop, thread = start_background_loop()
        try:
            async def failing() -> None:
                raise ValueError("boom")

            async def caller() -> None:
                await run_coro_async(failing(), loop=loop)

            with pytest.raises(ValueError, match="boom"):
                asyncio.run(caller())
        finally:
            stop_background_loop(loop, thread)
