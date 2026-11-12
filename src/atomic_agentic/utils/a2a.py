"""Pure Part<->dict conversion helpers for the a2a-sdk-backed A2A track.

Mirrors ``utils/mcp.py``'s role: protocol value-conversion helpers live here,
not inlined on the Tool class (``tools/a2a_sdk.py :: A2AProxyTool``). Both
functions raise plain ``ValueError``/``TypeError`` on malformed input --
matching how ``A2AClientHub.send_parts`` itself already raises plain
``ValueError``/``TypeError`` for malformed input -- and leave wrapping into
``ToolInvocationError`` to the Tool layer.
"""

from __future__ import annotations

import base64
from typing import Any, Mapping

from a2a.helpers import new_data_part, new_raw_part, new_text_part, new_url_part
from a2a.types import Part
from google.protobuf.json_format import MessageToDict

from ..constants.a2a_sdk import (
    PART_CONTENT_KEYS,
    PART_DATA_KEY,
    PART_FILENAME_KEY,
    PART_MEDIA_TYPE_KEY,
    PART_RAW_B64_KEY,
    PART_TEXT_KEY,
    PART_URL_KEY,
)

__all__ = ["_dict_to_part", "_part_to_dict"]


def _dict_to_part(d: Mapping[str, Any]) -> Part:
    """
    Normalize one plain dict into an a2a-sdk Part.

    Exactly one of PART_CONTENT_KEYS must be present with a non-None value.
    filename is only valid alongside raw_b64/url. raw_b64 accepts either a
    base64-encoded str or literal bytes.
    """
    if not isinstance(d, Mapping):
        raise TypeError(f"_dict_to_part: expected a Mapping, got {type(d)!r}")

    # Step 1: locate the one set content key.
    present = [key for key in PART_CONTENT_KEYS if d.get(key) is not None]
    if len(present) != 1:
        raise ValueError(
            f"_dict_to_part: exactly one of {PART_CONTENT_KEYS!r} must be "
            f"set; found {present!r}"
        )
    content_key = present[0]
    value = d[content_key]

    # Step 2: validate/normalize the shared filename/media_type fields.
    filename = d.get(PART_FILENAME_KEY)
    if filename is not None and not isinstance(filename, str):
        raise TypeError(f"_dict_to_part: {PART_FILENAME_KEY!r} must be a str or None.")

    media_type = d.get(PART_MEDIA_TYPE_KEY)
    if media_type is not None and not isinstance(media_type, str):
        raise TypeError(f"_dict_to_part: {PART_MEDIA_TYPE_KEY!r} must be a str or None.")

    if filename is not None and content_key not in (PART_RAW_B64_KEY, PART_URL_KEY):
        raise ValueError(
            f"_dict_to_part: {PART_FILENAME_KEY!r} is only valid alongside "
            f"{PART_RAW_B64_KEY!r} or {PART_URL_KEY!r}, not {content_key!r}."
        )

    # Step 3: dispatch on the one set content key.
    if content_key == PART_TEXT_KEY:
        if not isinstance(value, str):
            raise TypeError(f"_dict_to_part: {PART_TEXT_KEY!r} must be a str.")
        return new_text_part(value, media_type=media_type)

    if content_key == PART_DATA_KEY:
        return new_data_part(value, media_type=media_type)

    if content_key == PART_RAW_B64_KEY:
        if isinstance(value, bytes):
            raw_bytes = value
        elif isinstance(value, str):
            raw_bytes = base64.b64decode(value, validate=True)
        else:
            raise TypeError(
                f"_dict_to_part: {PART_RAW_B64_KEY!r} must be a str (base64) "
                f"or bytes, got {type(value)!r}."
            )
        return new_raw_part(raw_bytes, media_type=media_type, filename=filename)

    # content_key == PART_URL_KEY
    if not isinstance(value, str):
        raise TypeError(f"_dict_to_part: {PART_URL_KEY!r} must be a str.")
    return new_url_part(value, media_type=media_type, filename=filename)


def _part_to_dict(part: Part) -> dict[str, Any]:
    """
    Normalize one a2a-sdk Part back into a plain dict.

    Always returns all six keys (PART_TEXT_KEY/PART_DATA_KEY/
    PART_RAW_B64_KEY/PART_URL_KEY/PART_FILENAME_KEY/PART_MEDIA_TYPE_KEY),
    unset ones as None. raw content always comes back as a base64 str, never
    bare bytes, so the output stays uniformly JSON-safe.
    """
    if not isinstance(part, Part):
        raise TypeError(f"_part_to_dict: expected a Part, got {type(part)!r}")

    present = [
        key for key, field in (
            (PART_TEXT_KEY, "text"),
            (PART_DATA_KEY, "data"),
            (PART_RAW_B64_KEY, "raw"),
            (PART_URL_KEY, "url"),
        )
        if part.HasField(field)
    ]
    if len(present) != 1:
        raise ValueError(
            f"_part_to_dict: expected exactly one content field set on Part, "
            f"found {present!r}"
        )
    content_key = present[0]

    d: dict[str, Any] = {
        PART_TEXT_KEY: None,
        PART_DATA_KEY: None,
        PART_RAW_B64_KEY: None,
        PART_URL_KEY: None,
        PART_FILENAME_KEY: None,
        PART_MEDIA_TYPE_KEY: None,
    }

    if content_key == PART_TEXT_KEY:
        d[PART_TEXT_KEY] = part.text
    elif content_key == PART_DATA_KEY:
        d[PART_DATA_KEY] = MessageToDict(part.data)
    elif content_key == PART_RAW_B64_KEY:
        d[PART_RAW_B64_KEY] = base64.b64encode(part.raw).decode("ascii")
    else:  # PART_URL_KEY
        d[PART_URL_KEY] = part.url

    d[PART_FILENAME_KEY] = part.filename or None
    d[PART_MEDIA_TYPE_KEY] = part.media_type or None
    return d
