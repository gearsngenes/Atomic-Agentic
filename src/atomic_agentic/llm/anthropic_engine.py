from __future__ import annotations

import asyncio
import base64
import os
from typing import Any, Dict, List, Mapping

try:
    from anthropic import Anthropic, AsyncAnthropic, APIConnectionError, APIStatusError
except ImportError:
    Anthropic = None            # type: ignore[assignment,misc]
    AsyncAnthropic = None       # type: ignore[assignment,misc]
    APIConnectionError = None   # type: ignore[assignment,misc]
    APIStatusError = None       # type: ignore[assignment,misc]

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
from ..utils.core import run_coro_sync

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
    in ``_build_provider_payload``.  ``_on_detach`` is a no-op.

    Invocation
    ----------
    ``role="system"`` messages are extracted from the normalized input list
    and forwarded as the top-level ``system=`` parameter (joined with
    ``"\\n\\n"`` when multiple system messages are present).  The remaining
    messages form the ``messages=`` list.  ``max_tokens`` is always sent
    (required by Anthropic).  ``temperature``, ``top_p``, ``top_k``,
    ``stop_sequences``, and ``thinking`` are omitted when ``None``.
    ``block_separator`` (default ``""``) controls how multiple response text
    blocks are rejoined into the returned string — relevant when a response
    contains more than one ``text`` block (e.g. Anthropic citations, which
    can split a response into interleaved cited/uncited blocks). Defaults to
    no inserted separator, matching Anthropic's own text-reconstruction
    convention (the streaming SDK's ``text_stream`` concatenates raw text
    deltas across content-block boundaries with nothing inserted).
    """

    def __init__(
        self,
        model: str,
        name: str | None = None,
        namespace: str = "llm",
        description: str = "Anthropic Claude language model engine.",
        client: Anthropic | AsyncAnthropic | None = None,
        max_output_tokens: int = 8192,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: list[str] | None = None,
        thinking_config: dict[str, Any] | None = None,
        block_separator: str = "",
        inline_cutoff_chars: int = 200_000,
        *,
        timeout_seconds: float = 600.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 8.0,
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
        namespace:
            Grouping label; defaults to ``"llm"``.
        description:
            Human-readable description for this engine instance.
        client:
            Pre-built ``Anthropic`` or ``AsyncAnthropic`` client. When
            provided, used directly; when ``None``, a sync
            ``Anthropic(**client_kwargs)`` is built.
        max_output_tokens:
            Always sent as ``max_tokens=`` (required by the Anthropic API).
        temperature:
            Sampling temperature. ``None`` omits the parameter entirely.
        top_p, top_k:
            Optional sampling parameters; omitted when ``None``.
        stop_sequences:
            Optional list of stop strings; omitted when ``None``.
        thinking_config:
            Optional extended-thinking configuration dict (e.g.
            ``{"type": "adaptive"}``); sent as ``thinking=``, omitted when
            ``None``.
        block_separator:
            Controls how multiple response text blocks are rejoined into the
            returned string (relevant for citation-bearing responses).
            Defaults to ``""`` (no inserted separator).
        inline_cutoff_chars:
            Maximum characters to inline from text/code attachments.
        timeout_seconds:
            Per-call timeout inserted into the client's ``timeout`` option.
        max_retries, retry_backoff_base, retry_backoff_max:
            Shared ``LLMEngine`` retry/backoff configuration.
        **client_kwargs:
            Additional keyword arguments forwarded verbatim to
            ``Anthropic(...)`` during client construction. Common uses:
            ``api_key``, ``base_url``. Not forwarded when ``client`` is
            injected directly.
        """
        # 1. Name sanitization + super init
        sanitized = (name or f"anthropic_{model}").replace(":", "_").replace(
            "-", "_").replace(" ", "_").replace(".", "_")
        super().__init__(
            name=sanitized,
            namespace=namespace,
            description=description or "Anthropic Claude language model engine.",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        # 2. SDK presence check
        if Anthropic is None:
            raise LLMEngineError(
                "AnthropicEngine requires the `anthropic` package: "
                "pip install anthropic"
            )

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
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.stop_sequences = stop_sequences
        self.thinking_config = thinking_config
        self.block_separator = block_separator
        self._inline_cutoff_chars = int(inline_cutoff_chars)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def inline_cutoff_chars(self) -> int:
        """Max characters inlined from text attachments; fixed at construction."""
        return self._inline_cutoff_chars

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

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        """
        Retry on Anthropic connection/timeout errors and on retryable HTTP
        status codes (429, 500, 502, 503, 504) from ``APIStatusError``.
        """
        if attempt > self._max_retries:
            return False
        if isinstance(exc, APIConnectionError):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code in {429, 500, 502, 503, 504}
        return False

    # ------------------------------------------------------------------ #
    # Payload construction
    # ------------------------------------------------------------------ #

    def _build_provider_payload(
        self,
        messages: List[Dict[str, Any]],
        attachments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the ``messages.create`` payload from normalized messages and the
        current attachments snapshot.

        Lifecycle: (1) partition system vs. conversation; (2) convert to
        Anthropic content-list format; (3) append attachment blocks to the
        last user turn; (4) assemble final payload dict with optional
        generation params.
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
        blocks = list(attachments.values())
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
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.stop_sequences is not None:
            payload["stop_sequences"] = self.stop_sequences
        if self.thinking_config is not None:
            payload["thinking"] = self.thinking_config
        return payload

    # ------------------------------------------------------------------ #
    # Response extraction
    # ------------------------------------------------------------------ #

    def _extract_text(self, response: Any) -> str:
        """Return generated text by joining all ``TextBlock`` parts with ``block_separator``; empty string on refusal."""
        return self.block_separator.join(
            block.text for block in response.content if block.type == "text"
        )

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
        text/code → UTF-8 ``document/text`` block, truncated at
        ``inline_cutoff_chars``.
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
            if len(text) > self._inline_cutoff_chars:
                text = text[: self._inline_cutoff_chars] + "\n…[truncated]\n"
            return {
                "type": "document",
                "source": {
                    "type": "text",
                    "data": text,
                },
            }
        except LLMEngineError:
            raise
        except Exception as exc:
            raise LLMEngineError(
                f"AnthropicEngine._prepare_attachment failed for {path!r}"
            ) from exc

    def _on_detach(self, meta: Any) -> None:
        """No-op — inline attachments leave no server-side state to clean up."""

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop_sequences": self.stop_sequences,
            "thinking_config": self.thinking_config,
            "block_separator": self.block_separator,
            "inline_cutoff_chars": self._inline_cutoff_chars,
        })
        return d
