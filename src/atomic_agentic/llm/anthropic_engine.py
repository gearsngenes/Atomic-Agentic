from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
from typing import Any, ClassVar, Dict, List, Mapping, Optional

try:
    from anthropic import Anthropic, AsyncAnthropic
    _SDK_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:
    Anthropic = None            # type: ignore[assignment,misc]
    AsyncAnthropic = None       # type: ignore[assignment,misc]
    _SDK_IMPORT_ERROR = _exc

from .base import LLMEngine
from ..constants.llm import (
    ILLEGAL_ATTACHMENT_EXTS,
    ENGINE_ILLEGAL_MIME_PREFIXES,
    ANTHROPIC_IMAGE_EXTS,
    ANTHROPIC_ALLOWED_EXTS,
)
from ..exceptions import LLMEngineError
from ..models.results.llm import AnthropicTokenUsage, RemoteLLMModelData
from ..utils.llm import (
    classify_path,
    collect_system_text,
    read_text_file,
    validate_attachment_path,
)
from ..utils.core import format_missing_dependency, run_coro_sync

__all__ = ["AnthropicEngine"]

_MIME_MAP: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".pdf":  "application/pdf",
}


class AnthropicEngine(LLMEngine):
    """
    Anthropic Claude adapter using the Messages API.

    Client model
    ------------
    A single ``_client`` attribute holds either an ``Anthropic`` or
    ``AsyncAnthropic`` instance.  Sync/async routing is determined at call
    time via ``isinstance(self._client, AsyncAnthropic)``:

    - ``AsyncAnthropic`` injected  → ``_call_provider`` bridges via
      ``run_coro_sync``; ``_call_provider_async`` awaits natively.
    - ``Anthropic`` injected (or auto-built from ``**client_kwargs``) →
      ``_call_provider`` calls directly; ``_call_provider_async`` offloads
      via ``asyncio.to_thread``.

    File policy
    -----------
    Inline-only attachment support (no Files API):

    - Images (PNG, JPG, JPEG, GIF, WEBP) → base64 ``image`` content blocks.
    - PDFs → base64 ``document`` content blocks.
    - Text / code files → UTF-8 ``document/text`` content blocks
      (no base64 required).

    ``attach`` stores light metadata (``kind``, ``mime``, ``ext``) in
    ``attachments`` and keeps the content block privately; the blocks are
    appended to the last user turn in ``_build_provider_payload``.
    ``attach``/``detach``/``clear_attachments`` are this engine's own API;
    detach has no provider cleanup.

    Invocation
    ----------
    System messages are joined and forwarded as the top-level ``system=``
    parameter; the rest form ``messages=``. ``max_tokens`` is always sent
    (required by Anthropic): ``_REQUIRED_RUNTIME_DEFAULTS`` supplies
    ``max_tokens=8192`` under the runtime parameters, so the store starts empty
    and removing ``max_tokens`` falls back to that default. Every other
    generation setting (``stop_sequences``, ``thinking``, ``extra_body``, ...)
    is a runtime parameter under Anthropic's own name, forwarded verbatim:
    ``engine.update_runtime_parameters(thinking={"type": "adaptive"})``. Which
    fields the API accepts depends on the SDK version (anthropic 2.x rejects
    ``temperature``, which can travel in ``extra_body``); AA does not validate
    them. Timeout and retry are the SDK client's own settings through
    ``client_kwargs``. ``block_separator`` (default ``""``) is the string
    inserted between multiple response ``text`` blocks (e.g. with citations);
    it is a validated, settable attribute, not a runtime parameter. Anthropic
    always reports usage, so ``require_usage`` has no effect here.

    Result extraction
    -----------------
    ``_extract_result`` joins the ``text`` blocks with ``block_separator``
    (empty string on refusal or no text) and, on a structured call, parses
    them with ``json.loads`` (raw text on a parse failure).

    Structured-output schema policy
    -------------------------------
    ``output_structure`` is forwarded unmodified to ``output_config.format.schema``;
    Anthropic rejects what it does not accept natively (``additionalProperties``
    must be ``false`` on every object).

    A runtime ``output_config`` (e.g. ``effort``) is allowed: on a structured
    call the engine copies it and sets only its ``format`` key; a plain call
    passes it through untouched.
    """

    _RESERVED_RUNTIME_KEYS = frozenset({"model", "messages", "system", "stream"})

    # Fields the API requires on every call; runtime parameters override them.
    _REQUIRED_RUNTIME_DEFAULTS: ClassVar[Mapping[str, Any]] = {"max_tokens": 8192}

    def __init__(
        self,
        model: str,
        name: str | None = None,
        description: str | None = None,
        client: Anthropic | AsyncAnthropic | None = None,
        **client_kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Anthropic model identifier (e.g. ``"claude-sonnet-4-6"``).
        name:
            Optional human-friendly engine name; defaults to the class name.
        description:
            Human-readable description for this engine instance.
        client:
            Pre-built ``Anthropic`` or ``AsyncAnthropic`` client. When
            provided, used directly (combining it with ``client_kwargs``
            raises); when ``None``, a sync ``Anthropic(**client_kwargs)`` is
            built.
        **client_kwargs:
            Keyword arguments forwarded unchanged to ``Anthropic(...)``. Common
            uses: ``api_key``, ``base_url``, and the SDK's own ``timeout`` and
            ``max_retries``.

        Lifecycle: SDK presence check; reject ``client`` combined with
        ``client_kwargs``; base init; build or adopt the client;
        ``block_separator`` starts as ``""``.
        """
        # 1. SDK presence check
        if Anthropic is None:
            raise LLMEngineError(
                "AnthropicEngine "
                + format_missing_dependency("anthropic", "anthropic", _SDK_IMPORT_ERROR)
            ) from _SDK_IMPORT_ERROR

        # 2. Exactly one client source
        if client is not None and client_kwargs:
            raise LLMEngineError(
                "AnthropicEngine: pass either 'client' or client kwargs, not both."
            )

        # 3. Base init, then engine state
        super().__init__(
            name=name,
            description=description or "Anthropic Claude language model engine.",
        )
        self._attachments: Dict[str, Dict[str, Any]] = {}
        self._blocks: Dict[str, Dict[str, Any]] = {}
        self._client: Anthropic | AsyncAnthropic = (
            client if client is not None else Anthropic(**client_kwargs)
        )
        self.model = model
        self.block_separator = ""

    @property
    def block_separator(self) -> str:
        """String inserted between response text blocks when they are joined."""
        return self._block_separator

    @block_separator.setter
    def block_separator(self, value: str) -> None:
        if not isinstance(value, str):
            raise LLMEngineError(
                f"AnthropicEngine.block_separator must be a str, got {type(value).__name__}."
            )
        self._block_separator = value

    # ------------------------------------------------------------------ #
    # Attachments
    # ------------------------------------------------------------------ #
    @property
    def attachments(self) -> Mapping[str, Mapping[str, Any]]:
        """Copy of attached local path -> metadata (``kind``, ``mime``, ``ext``)."""
        return {path: dict(meta) for path, meta in self._attachments.items()}

    def attach(self, path: str) -> Mapping[str, Any]:
        """
        Attach a local file for every later call; returns its metadata copy
        (never the content block).

        1. Reject a blank or non-string path.
        2. Return the stored metadata when ``path`` is already attached.
        3. Validate the path, build the metadata and content block, store both.
        """
        if not isinstance(path, str) or not path:
            raise LLMEngineError("AnthropicEngine.attach: path must be a non-empty string")
        if path in self._attachments:
            return dict(self._attachments[path])

        self._validate_attachment_path(path)
        meta, block = self._prepare_attachment(path)
        self._attachments[path] = meta
        self._blocks[path] = block
        return dict(meta)

    def detach(self, path: str) -> bool:
        """Detach ``path``; ``True`` when something was removed (no provider cleanup)."""
        self._blocks.pop(path, None)
        return self._attachments.pop(path, None) is not None

    def clear_attachments(self) -> None:
        """Detach every attached path."""
        for path in list(self._attachments):
            self.detach(path)

    # ------------------------------------------------------------------ #
    # Client routing
    # ------------------------------------------------------------------ #

    def _call_provider(self, payload: Any) -> Any:
        """Synchronous Messages API call; bridges via ``run_coro_sync`` when an async client is held."""
        if isinstance(self._client, AsyncAnthropic):
            return run_coro_sync(self._client.messages.create(**payload))
        return self._client.messages.create(**payload)

    async def _call_provider_async(self, payload: Any) -> Any:
        """Native async Messages API call; offloads to a thread when only a sync client is held."""
        if isinstance(self._client, AsyncAnthropic):
            return await self._client.messages.create(**payload)
        return await asyncio.to_thread(self._call_provider, payload)

    # ------------------------------------------------------------------ #
    # Payload construction
    # ------------------------------------------------------------------ #

    def _build_provider_payload(
        self,
        messages: List[Dict[str, Any]],
        output_structure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the ``messages.create`` payload from normalized messages and a
        snapshot of the engine's attachments.

        1. Join the system messages (blank ones dropped) for ``system``.
        2. Convert the other messages to content-list format.
        3. Append the attachment blocks to the last user turn.
        4. Start from the required defaults, then the runtime parameters, then
           the engine-owned fields (``model``, ``messages``, ``system``), so
           the engine-owned fields win.
        5. Structured call: set ``output_config.format`` on a copy of any
           runtime ``output_config`` (the caller's raw schema, unmodified).
        """
        # 1-2. System text and conversation turns
        system = collect_system_text(messages)
        anthropic_msgs: list[dict[str, Any]] = [
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in messages
            if m["role"] != "system"
        ]

        # 3. Append attachment blocks to the last user turn
        blocks = list(self._blocks.values())
        if blocks:
            last_user = next(m for m in reversed(anthropic_msgs) if m["role"] == "user")
            last_user["content"] = last_user["content"] + blocks

        # 4. Assemble the call kwargs
        payload: dict[str, Any] = dict(self._REQUIRED_RUNTIME_DEFAULTS)
        payload.update(self._runtime_parameters)
        payload.update(model=self.model, messages=anthropic_msgs)
        if system:
            payload.update(system=system)

        if output_structure is not None:
            output_config = dict(payload.get("output_config") or {})
            output_config["format"] = {"type": "json_schema", "schema": output_structure}
            payload.update(output_config=output_config)
        return payload

    # ------------------------------------------------------------------ #
    # Response extraction
    # ------------------------------------------------------------------ #

    def _extract_result(
        self, response: Any, requested_structured: bool
    ) -> str | int | float | bool | list[Any] | dict[str, Any] | None:
        """
        Join the ``text`` blocks with ``block_separator`` (empty string on
        refusal or no text); on a structured call, ``json.loads`` the joined
        text (raw text on a parse failure, including the empty refusal).
        """
        text = self.block_separator.join(
            block.text for block in response.content if block.type == "text"
        )
        if not requested_structured:
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _extract_token_usage(self, response: Any) -> AnthropicTokenUsage:
        """
        Map ``response.usage`` to ``AnthropicTokenUsage``.

        ``total_tokens`` is derived as ``input_tokens + output_tokens`` (Anthropic
        does not provide a pre-summed total). ``response_tokens`` is derived as
        ``output_tokens`` minus any thinking-token subset. Cache and thinking
        fields are read from ``usage`` attributes; absent fields default to
        ``None``.
        """
        u = response.usage
        inp = u.input_tokens
        gen = u.output_tokens

        # `output_tokens_details` is either `None` or a real
        # `anthropic.OutputTokensDetails` Pydantic object (never a dict) —
        # `getattr` avoids assuming dict-style `.get()` access, which crashes
        # on the real object whenever extended thinking actually populates it.
        details = getattr(u, "output_tokens_details", None)
        thinking_tokens = (
            getattr(details, "thinking_tokens", None) if details is not None else None
        )
        response_tokens = gen - (thinking_tokens or 0)

        return AnthropicTokenUsage(
            input_tokens=inp,
            generated_tokens=gen,
            total_tokens=inp + gen,
            response_tokens=response_tokens,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", None),
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", None),
            thinking_tokens=thinking_tokens,
        )

    def _get_model_data(self) -> RemoteLLMModelData:
        """Return configured Anthropic model identity for this engine."""
        return RemoteLLMModelData(provider="anthropic", model_name=self.model)

    # ------------------------------------------------------------------ #
    # Attachment pipeline (inline only)
    # ------------------------------------------------------------------ #

    def _validate_attachment_path(self, path: str) -> None:
        """Validate ``path`` against the Anthropic allow-list and shared blacklists; raises ``LLMEngineError`` on rejection."""
        try:
            validate_attachment_path(
                path,
                illegal_exts=ILLEGAL_ATTACHMENT_EXTS,
                allowed_exts=ANTHROPIC_ALLOWED_EXTS,
                illegal_mime_prefixes=ENGINE_ILLEGAL_MIME_PREFIXES,
            )
        except ValueError as exc:
            raise LLMEngineError(str(exc)) from exc

    def _prepare_attachment(self, path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Return ``(metadata, content block)`` for ``path`` (no upload).

        Images become a base64 ``image`` block, PDFs a base64 ``document``
        block, text/code a ``document`` block with a ``text`` source (whole
        file, UTF-8 with replacement). The metadata is ``kind``
        (``"image"``/``"pdf"``/``"text"``), ``mime`` and ``ext``.
        """
        try:
            kind = classify_path(path, ANTHROPIC_IMAGE_EXTS)
            mime, _ = mimetypes.guess_type(path)
            ext = os.path.splitext(path)[1].lower()
            meta = {"kind": kind, "mime": mime or "", "ext": ext}

            if kind in ("image", "pdf"):
                with open(path, "rb") as fh:
                    data = base64.b64encode(fh.read()).decode()
                block_type = "image" if kind == "image" else "document"
                source = {"type": "base64", "media_type": _MIME_MAP[ext], "data": data}
            else:
                block_type = "document"
                source = {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": read_text_file(path),
                }
            return meta, {"type": block_type, "source": source}
        except LLMEngineError:
            raise
        except Exception as exc:
            raise LLMEngineError(
                f"AnthropicEngine._prepare_attachment failed for {path!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "model": self.model,
            "block_separator": self.block_separator,
        })
        return d
