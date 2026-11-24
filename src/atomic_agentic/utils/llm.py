"""Pure helpers shared by the LLM engine adapters (messages, attachments)."""

import mimetypes
import os
from typing import Any, Iterable, Mapping


def collect_system_text(messages: Iterable[Mapping[str, Any]]) -> str | None:
    """
    Join the non-blank ``system`` message contents with blank lines.

    Returns the stripped result, or ``None`` when no system message has
    non-blank content.
    """
    parts = [
        m["content"].strip()
        for m in messages
        if m.get("role") == "system" and m["content"].strip()
    ]
    return "\n\n".join(parts) or None


def classify_path(path: str, image_exts: Iterable[str]) -> str:
    """
    Bucket ``path`` as ``"pdf"``, ``"image"`` or ``"text"``.

    A path is a PDF by extension or MIME type, an image when its extension is
    in ``image_exts`` or its MIME type starts with ``image/``, and text
    otherwise.
    """
    mime, _ = mimetypes.guess_type(path)
    mime = mime or ""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf" or mime == "application/pdf":
        return "pdf"
    if ext in image_exts or mime.startswith("image/"):
        return "image"
    return "text"


def read_text_file(path: str) -> str:
    """Read ``path`` as UTF-8, replacing undecodable bytes. IO errors propagate."""
    with open(path, "rb") as fh:
        return fh.read().decode("utf-8", errors="replace")


def inlined_file_block(path: str, text: str) -> str:
    """Return ``text`` under an ``[Inlined file: <name>]`` header line."""
    return f"\n[Inlined file: {os.path.basename(path)}]\n{text}"


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
