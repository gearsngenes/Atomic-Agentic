"""Attachment validation helpers for LLM engine adapters."""

import mimetypes
import os


def validate_attachment_path(
    path: str,
    illegal_exts: set[str],
    allowed_exts: set[str] | None,
    illegal_mime_prefixes: tuple[str, ...],
) -> None:
    """
    Validate a local file path for engine attachment.

    Raises ``ValueError`` on any violation. Engine subclasses call this from
    their ``_validate_attachment_path`` overrides and wrap ``ValueError`` into
    ``LLMEngineError`` at the engine boundary.

    Validation order:
    1. File must exist and be a regular file.
    2. Extension must not be in ``illegal_exts``.
    3. If ``allowed_exts`` is given, extension must be in it.
    4. If ``illegal_mime_prefixes`` is non-empty, MIME type must not match.
    """
    if not os.path.isfile(path):
        raise ValueError(f"path does not exist or is not a file: {path!r}")

    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext and ext in illegal_exts:
        raise ValueError(f"extension {ext!r} is not permitted")

    if allowed_exts is not None and ext not in allowed_exts:
        raise ValueError(f"extension {ext!r} is not supported")

    if illegal_mime_prefixes:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        if any(mime.startswith(prefix) for prefix in illegal_mime_prefixes):
            raise ValueError(f"MIME type {mime!r} is not permitted")
