from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Mapping
from types import MappingProxyType
from typing import Any

from ..constants.core import HeaderValue, T

__all__ = [
    "normalize_headers",
    "run_coro_sync",
]


async def _await_value(awaitable: Awaitable[T]) -> T:
    """Await an arbitrary awaitable through a coroutine wrapper."""
    return await awaitable


def run_coro_sync(awaitable: Awaitable[T]) -> T:
    """
    Run an awaitable to completion from synchronous code.

    Contract
    --------
    - If no event loop is running in the current thread, run the awaitable with
      ``asyncio.run(...)``.
    - If an event loop is already running in the current thread, run the awaitable
      inside a temporary event loop on a worker thread.
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
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_value(awaitable))

    result_box: list[T] = []
    error_box: list[BaseException] = []

    def runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result_box.append(loop.run_until_complete(_await_value(awaitable)))
        except BaseException as exc:  # noqa: BLE001
            error_box.append(exc)
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if error_box:
        raise error_box[0]

    if not result_box:
        raise RuntimeError("Awaitable completed without producing a result.")

    return result_box[0]


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
