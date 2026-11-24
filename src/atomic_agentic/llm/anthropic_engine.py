from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any, Dict, List, Mapping, Optional

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
    ANTHROPIC_DOCUMENT_EXTS,
    ANTHROPIC_ALLOWED_EXTS,
)
from ..exceptions import LLMEngineError
from ..models.results.llm import AnthropicTokenUsage, RemoteLLMModelData
from ..utils.llm import validate_attachment_path
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
    Inline-only attachment support (Anthropic Files API is still in beta —
    deferred to a future pass):

    - Images (PNG, JPG, JPEG, GIF, WEBP) → base64 ``image`` content blocks.
    - PDFs → base64 ``document`` content blocks.
    - Text / code files → UTF-8 ``document/text`` content blocks
      (no base64 required).

    Attachment blocks are appended to the last user-turn ``content`` list
    in ``_build_provider_payload``. ``attach``/``detach``/``clear_attachments``
    are this engine's own API; detach has no provider cleanup.

    Invocation
    ----------
    ``role="system"`` messages are extracted from the normalized input list
    and forwarded as the top-level ``system=`` parameter (joined with
    ``"\\n\\n"`` when multiple system messages are present).  The remaining
    messages form the ``messages=`` list.  ``max_tokens`` is always sent
    (required by Anthropic).  ``stop_sequences``, ``thinking`` and
    ``extra_body`` (sampling fields such as ``temperature``) are omitted
    when ``None``.
    ``block_separator`` (default ``""``) controls how multiple response text
    blocks are rejoined into the returned string — relevant when a response
    contains more than one ``text`` block (e.g. Anthropic citations, which
    can split a response into interleaved cited/uncited blocks). Defaults to
    no inserted separator, matching Anthropic's own text-reconstruction
    convention (the streaming SDK's ``text_stream`` concatenates raw text
    deltas across content-block boundaries with nothing inserted).

    Result extraction
    ------------------
    ``_extract_result`` reuses the same ``TextBlock``-joining logic as
    before (``block_separator``-joined; empty string on refusal or no text
    blocks). When the base class signals ``requested_structured=True``, the
    joined text is parsed via ``json.loads``; a parse failure (including the
    empty-string refusal case) falls back to returning the raw text rather
    than raising, matching the base ``LLMEngine`` fallback contract
    (``structured-generation`` Pass 3). No Anthropic-specific detection is
    needed — the response shape is identical whether or not structured
    output was requested.

    Structured-output schema policy
    ---------------------------------
    ``output_structure``, when given, is forwarded to
    ``output_config.format.schema`` completely unmodified — this engine
    performs no schema pruning (structured-generation Pass 8). Keywords
    Anthropic's JSON-Schema mode rejects unconditionally (confirmed live,
    2026-08-19: ``minimum``, ``maximum``, ``multipleOf``, ``maxItems``) now
    surface as Anthropic's own API error rather than being silently
    stripped. ``oneOf`` and an over-constrained ``minItems`` are
    structurally risky (can collapse a field to an unconstrained ``{}``)
    and are not validated here; ``additionalProperties`` must be set to
    exactly ``false`` on every object schema, the caller's own
    responsibility.
    """

    def __init__(
        self,
        model: str,
        name: str | None = None,
        description: str = "Anthropic Claude language model engine.",
        client: Anthropic | AsyncAnthropic | None = None,
        max_output_tokens: int = 8192,
        stop_sequences: list[str] | None = None,
        thinking_config: dict[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        block_separator: str = "",
        *,
        timeout_seconds: float = 600.0,
        **client_kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Anthropic model identifier (e.g. ``"claude-sonnet-4-6"``).
        name:
            Optional human-friendly engine name; defaults to
            ``anthropic_{model}`` with non-identifier characters replaced by
            underscores.
        description:
            Human-readable description for this engine instance.
        client:
            Pre-built ``Anthropic`` or ``AsyncAnthropic`` client. When
            provided, used directly; when ``None``, a sync
            ``Anthropic(**client_kwargs)`` is built.
        max_output_tokens:
            Always sent as ``max_tokens=`` (required by the Anthropic API).
        stop_sequences:
            Optional list of stop strings; omitted when ``None``.
        thinking_config:
            Optional extended-thinking configuration dict (e.g.
            ``{"type": "adaptive"}``); sent as ``thinking=``, omitted when
            ``None``.
        extra_body:
            Extra request fields forwarded verbatim through the SDK's
            ``extra_body`` passthrough, e.g. ``{"temperature": 0.3}`` or
            ``{"top_k": 5}``. Which fields the API accepts depends on the
            API version the SDK speaks (anthropic 2.x rejects
            ``temperature``); AA does not validate them. Omitted when ``None``.
        block_separator:
            Controls how multiple response text blocks are rejoined into the
            returned string (relevant for citation-bearing responses).
            Defaults to ``""`` (no inserted separator).
        timeout_seconds:
            Per-call timeout inserted into the client's ``timeout`` option.
        **client_kwargs:
            Additional keyword arguments forwarded verbatim to
            ``Anthropic(...)`` during client construction. Common uses:
            ``api_key``, ``base_url``, and the SDK's own ``max_retries``. Not
            forwarded when ``client`` is injected directly.
        """
        # 1. Name sanitization + super init
        sanitized = (name or f"anthropic_{model}").replace(":", "_").replace(
            "-", "_").replace(" ", "_").replace(".", "_")
        super().__init__(
            name=sanitized,
            description=description or "Anthropic Claude language model engine.",
            timeout_seconds=timeout_seconds,
        )
        self._attachments: Dict[str, Mapping[str, Any]] = {}

        # 2. SDK presence check
        if Anthropic is None:
            raise LLMEngineError(
                "AnthropicEngine "
                + format_missing_dependency("anthropic", "anthropic", _SDK_IMPORT_ERROR)
            ) from _SDK_IMPORT_ERROR

        # 3. Build client kwargs; seed timeout default
        _ckw = dict(client_kwargs)
        _ckw.setdefault("timeout", self._timeout_seconds)

        # 4. Assign or build client
        self._client: Anthropic | AsyncAnthropic = (
            client if client is not None else Anthropic(**_ckw)
        )

        # 5. Store request-level params
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.stop_sequences = stop_sequences
        self.thinking_config = thinking_config
        self.extra_body = dict(extra_body) if extra_body is not None else None
        self.block_separator = block_separator

    # ------------------------------------------------------------------ #
    # Attachments
    # ------------------------------------------------------------------ #
    @property
    def attachments(self) -> Mapping[str, Mapping[str, Any]]:
        """Read-only copy of attached local path -> provider metadata."""
        return dict(self._attachments)

    def attach(self, path: str) -> Mapping[str, Any]:
        """
        Attach a local file for every later call; returns its metadata.

        1. Reject a blank or non-string path.
        2. Return the stored metadata when ``path`` is already attached.
        3. Validate the path, prepare the provider metadata, store it.
        """
        if not isinstance(path, str) or not path:
            raise LLMEngineError("AnthropicEngine.attach: path must be a non-empty string")
        if path in self._attachments:
            return self._attachments[path]

        self._validate_attachment_path(path)
        meta = self._prepare_attachment(path)
        self._attachments[path] = meta
        return meta

    def detach(self, path: str) -> bool:
        """Detach ``path``; ``True`` when something was removed (no provider cleanup)."""
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

        Lifecycle: (1) partition system vs. conversation; (2) convert to
        Anthropic content-list format; (3) append attachment blocks to the
        last user turn; (4) assemble final payload dict with optional
        generation params; (5) if ``output_structure`` is given, wire it into
        ``output_config.format`` (the caller's raw schema, forwarded
        unmodified).
        """
        # 1. Partition system vs. conversation messages
        system_parts = [
            m["content"] for m in messages if m.get("role") == "system"
        ]
        conv = [m for m in messages if m.get("role") != "system"]

        # 2. Convert to Anthropic format — content always a list
        anthropic_msgs: list[dict[str, Any]] = []
        for m in conv:
            content = m["content"]
            if not isinstance(content, list):
                content = [{"type": "text", "text": content}]
            anthropic_msgs.append({"role": m["role"], "content": content})

        # 3. Append attachment blocks to the last user turn
        blocks = list(self._attachments.values())
        if blocks:
            idx = next(
                (i for i in range(len(anthropic_msgs) - 1, -1, -1)
                 if anthropic_msgs[i]["role"] == "user"),
                None,
            )
            if idx is not None:
                anthropic_msgs[idx]["content"] = anthropic_msgs[idx]["content"] + blocks
            else:
                anthropic_msgs.append({"role": "user", "content": blocks})

        # 4. Assemble payload
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "messages": anthropic_msgs,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        # Sampling params (temperature, top_p, top_k, ...) are no longer fixed
        # engine knobs: newer Anthropic API versions (SDK 2.x) reject some of
        # them, so they travel through the SDK's extra_body passthrough exactly
        # as the caller supplied them.
        if self.extra_body:
            payload["extra_body"] = dict(self.extra_body)

        if self.stop_sequences is not None:
            payload["stop_sequences"] = self.stop_sequences
        if self.thinking_config is not None:
            payload["thinking"] = self.thinking_config
        if output_structure is not None:
            payload["output_config"] = {
                "format": {"type": "json_schema", "schema": output_structure}
            }
        return payload

    # ------------------------------------------------------------------ #
    # Response extraction
    # ------------------------------------------------------------------ #

    def _extract_result(
        self, response: Any, requested_structured: bool
    ) -> str | int | float | bool | list[Any] | dict[str, Any] | None:
        """
        Extract the assistant's textual or structured reply from a Messages
        API response object.

        Joins all ``TextBlock`` parts with ``block_separator`` (empty string
        on refusal or no text blocks) -- unchanged regardless of
        ``requested_structured``. When not requested, returns that text
        as-is. When requested, attempts ``json.loads`` on the joined text; a
        parse failure (including the empty-string refusal case) falls back
        to returning the raw text instead of raising.
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

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        """
        Build an inline Anthropic content block for ``path`` (no upload).

        Images → base64 ``image`` block; PDFs → base64 ``document`` block;
        text/code → UTF-8 ``document/text`` block (whole file).
        """
        try:
            ext = os.path.splitext(path)[1].lower()

            if ext in ANTHROPIC_IMAGE_EXTS:
                with open(path, "rb") as fh:
                    data = base64.b64encode(fh.read()).decode()
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _MIME_MAP[ext],
                        "data": data,
                    },
                }

            if ext in ANTHROPIC_DOCUMENT_EXTS:
                with open(path, "rb") as fh:
                    data = base64.b64encode(fh.read()).decode()
                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": _MIME_MAP[ext],
                        "data": data,
                    },
                }

            # Text / code — document/text source (no base64 needed)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            return {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": text,
                },
            }
        except LLMEngineError:
            raise
        except Exception as exc:
            raise LLMEngineError(
                f"AnthropicEngine._prepare_attachment failed for {path!r}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "stop_sequences": self.stop_sequences,
            "thinking_config": self.thinking_config,
            "extra_body": self.extra_body,
            "block_separator": self.block_separator,
        })
        return d
