from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Mapping
from dataclasses import fields
from types import MappingProxyType
from typing import Any

from ..constants.core import HeaderValue, T

__all__ = [
    "dataclass_record_to_dict",
    "normalize_headers",
    "run_coro_async",
    "run_coro_sync",
    "start_background_loop",
    "stop_background_loop",
]


def dataclass_record_to_dict(record: Any) -> dict[str, Any]:
    """
    Serialize a dataclass record into a plain dict, tagged with its concrete
    class name.

    Generic leaf serializer for frozen dataclass records across the package
    (e.g. token-usage and model-identity records). Assumes ``record`` is a
    dataclass instance; this is not a validation boundary.
    """
    data = {field.name: getattr(record, field.name) for field in fields(record)}
    return {"type": type(record).__name__, **data}


async def _await_value(awaitable: Awaitable[T]) -> T:
    """Await an arbitrary awaitable through a coroutine wrapper."""
    return await awaitable


def run_coro_sync(awaitable: Awaitable[T], loop: asyncio.AbstractEventLoop | None = None) -> T:
    """
    Run an awaitable to completion from synchronous code.

    Contract
    --------
    - If ``loop`` is given, dispatch the awaitable onto it via
      ``asyncio.run_coroutine_threadsafe`` and block the calling thread
      until it completes. Intended for a loop started by
      ``start_background_loop`` and kept alive across multiple calls (e.g.
      A2AClientHub's persistent mode) -- some asyncio-native resources
      (a grpc.aio.Channel in particular) break when reused across
      independently-created event loops, so anything meant to outlive one
      call must always dispatch onto the *same* loop.
    - If ``loop`` is ``None`` (default): if no event loop is running in the
      current thread, run the awaitable with ``asyncio.run(...)``. If one
      is already running, run it inside a temporary event loop on a worker
      thread.
    - Propagate exceptions raised by the awaitable.
    - Return the awaitable's result.

    Notes
    -----
    This helper is intended for sync public APIs that need to bridge into async
    implementation code, such as Tool execution or remote client/session calls.

    The awaitable should not be an ``asyncio.Task`` or ``asyncio.Future`` already
    bound to another running event loop. Passing raw coroutine objects is the
    expected use case.
    """
    if loop is not None:
        return asyncio.run_coroutine_threadsafe(_await_value(awaitable), loop).result()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_value(awaitable))

    result_box: list[T] = []
    error_box: list[BaseException] = []

    def runner() -> None:
        worker_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(worker_loop)
            result_box.append(worker_loop.run_until_complete(_await_value(awaitable)))
        except BaseException as exc:  # noqa: BLE001
            error_box.append(exc)
        finally:
            asyncio.set_event_loop(None)
            worker_loop.close()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if error_box:
        raise error_box[0]

    if not result_box:
        raise RuntimeError("Awaitable completed without producing a result.")

    return result_box[0]


async def run_coro_async(awaitable: Awaitable[T], loop: asyncio.AbstractEventLoop) -> T:
    """
    Dispatch a coroutine onto a specific other loop from async code, without
    blocking the caller's own currently-running loop.

    ``loop`` is required, not optional: this exists specifically to target a
    *different* loop than the one the caller is currently running on (e.g.
    A2AClientHub persistent mode's dedicated background loop). A caller with
    no other loop to target should just await the awaitable directly --
    there is nothing to bridge cross-thread in that case.

    ``asyncio.wrap_future`` integrates the cross-thread
    ``concurrent.futures.Future`` into the caller's own running event loop;
    the real work still runs on ``loop``, a different loop entirely.
    """
    return await asyncio.wrap_future(
        asyncio.run_coroutine_threadsafe(_await_value(awaitable), loop)
    )


def normalize_headers(
    value: Mapping[str, HeaderValue] | None,
) -> Mapping[str, str] | None:
    """
    Normalize optional transport headers into an immutable string mapping.

    Contract
    --------
    - ``None`` returns ``None``.
    - Non-mapping inputs raise ``ValueError``.
    - Header names must be strings.
    - Header names must be non-empty after stripping.
    - Header values may be strings, ints, floats, bools, bytes, or bytearrays.
    - Bytes and bytearray values must be ASCII-decodable.
    - ``None``, mappings, collections, and arbitrary objects are rejected as values.
    - Header names and normalized values may not contain CR, LF, or NUL characters.
    - Returned headers are copied into a read-only ``MappingProxyType``.

    Notes
    -----
    This helper performs limited, explicit scalar normalization. It does not
    serialize cookies, auth objects, structured headers, nested dictionaries,
    collections, or arbitrary custom objects. Those should be serialized by the
    caller before entering this header-normalization boundary.
    """
    if value is None:
        return None

    if not isinstance(value, Mapping):
        raise ValueError("headers must be a mapping.")

    normalized: dict[str, str] = {}

    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError("header names must be strings.")

        if _contains_forbidden_header_char(raw_key):
            raise ValueError(
                f"header name {raw_key!r} contains a forbidden character."
            )

        key = raw_key.strip()
        if not key:
            raise ValueError("header names must be non-empty strings.")

        header_value = _normalize_header_value(raw_value, key=key)

        if _contains_forbidden_header_char(header_value):
            raise ValueError(f"header value for {key!r} contains a forbidden character.")

        normalized[key] = header_value

    return MappingProxyType(normalized)


def _normalize_header_value(value: Any, *, key: str) -> str:
    """Normalize one explicitly supported scalar header value to a string."""
    if isinstance(value, str):
        return value

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)

    if isinstance(value, float):
        return str(value)

    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"header value for {key!r} must be ASCII-decodable bytes."
            ) from exc

    if value is None:
        raise ValueError(f"header value for {key!r} must not be None.")

    if isinstance(value, Mapping):
        raise ValueError(
            f"header value for {key!r} must not be a mapping; serialize structured "
            "headers, cookies, or auth config before passing headers."
        )

    if isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(
            f"header value for {key!r} must not be a collection; serialize it before "
            "passing headers."
        )

    raise ValueError(
        f"header value for {key!r} must be str, int, float, bool, bytes, or bytearray; "
        f"got {type(value).__name__!r}."
    )


def _contains_forbidden_header_char(value: str) -> bool:
    """Return whether a header name/value contains CR, LF, or NUL."""
    return "\r" in value or "\n" in value or "\x00" in value


def start_background_loop() -> tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """
    Start a fresh asyncio event loop on its own daemon thread.

    The loop is running and ready to accept coroutines dispatched via
    run_coro_sync/run_coro_async's ``loop`` param by the time this returns.
    Daemon so a caller that never calls stop_background_loop doesn't hang
    process exit -- it only leaks the thread/loop/whatever was dispatched
    onto it for the rest of the process's life. Needed by anything holding
    an asyncio-native resource that must stay bound to the same loop for
    its whole lifetime -- confirmed necessary because grpc.aio.Channel
    breaks when reused across independently-created event loops (a channel
    built and used inside one asyncio.run() call raises "attached to a
    different loop" on the very next call from a separate asyncio.run() --
    reproduced directly).
    """
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return loop, thread


def stop_background_loop(loop: asyncio.AbstractEventLoop, thread: threading.Thread) -> None:
    """Stop a loop started by start_background_loop and join its thread."""
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.close()
