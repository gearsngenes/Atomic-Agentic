from __future__ import annotations

import mimetypes
try:
    from mistralai.client import Mistral
    from mistralai.client import errors as mistral_errors
except ImportError:
    Mistral = None
    mistral_errors = None
import os
from typing import (
    Any,
    Dict,
    List,
    Mapping,
)

from .base import LLMEngine
from ..constants.llm import (
    ILLEGAL_ATTACHMENT_EXTS,
    ENGINE_ILLEGAL_MIME_PREFIXES,
    MISTRAL_IMAGE_EXTS,
)
from ..exceptions import LLMEngineError
from ..models.results.llm import (
    TokenUsage,
    MistralTokenUsage,
    LLMModelData,
    RemoteLLMModelData
)
from ..utils.llm import validate_attachment_path

__all__ = ["MistralEngine"]

# ── MISTRAL ─────────────────────────────
class MistralEngine(LLMEngine):
    """
    Mistral adapter using: **upload → sign → attach URL parts**.

    Flow per call
    -------------
    1) Attachments are prepared via `attach(path)`:
       - PDFs  → upload + sign → `{ "kind": "pdf", "signed_url": ... }`
       - Images → upload + sign → `{ "kind": "image", "signed_url": ... }`
       - Text/code → read + inline → `{ "kind": "text", "inlined_text": ... }`
    2) `invoke({'messages': ...})` (from the base `LLMEngine`) will:
       - normalize chat messages (role/content strings),
       - snapshot current attachments,
       - call `_build_provider_payload` to:
         * convert messages into Mistral's chat schema
         * ensure the last user message has a `content` parts array
         * append inline text and signed URL parts to that last user turn
       - call `_call_with_retries` → `_call_provider`
       - call `_extract_text` to normalize the response.
    3) `detach(path)` triggers best-effort deletion of uploaded files via
       `_on_detach`, which calls `client.files.delete(file_id=...)`.
    """

    def __init__(
            self,
            model: str,
            name: str | None = None,
            namespace: str = "llm",
            description: str = "Mistral LLM Engine",
            client: Mistral | None = None,
            temperature: float | None = 0.1,
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
            Mistral model identifier (e.g. ``"mistral-small-latest"``).
        name:
            Optional human-friendly engine name; defaults to ``mistral_{model}``
            with non-identifier characters replaced by underscores.
        namespace:
            Grouping label; inherited by the base engine as ``"llm"`` by default.
        description:
            Human-readable description for this engine instance.
        client:
            Pre-built ``Mistral`` client. When provided, used directly and no
            client is constructed from ``client_kwargs``. ``Mistral`` exposes
            both sync (``.chat.complete``) and async (``.chat.complete_async``)
            paths on the same object — no isinstance routing is needed.
        temperature:
            Sampling temperature for text generation. ``None`` omits the kwarg
            entirely, letting the SDK apply its own default sentinel.
        inline_cutoff_chars:
            Maximum characters to inline from text/code attachments.
        timeout_seconds, max_retries,
        retry_backoff_base, retry_backoff_max:
            Shared ``LLMEngine`` configuration (see base class).
        **client_kwargs:
            Additional keyword arguments forwarded to ``Mistral(...)`` during
            client construction. Common uses: ``api_key``, ``server_url``,
            ``timeout_ms``. A default ``timeout_ms`` is seeded from
            ``timeout_seconds`` unless supplied explicitly. Not forwarded when
            ``client`` is injected.
        """
        # 1. Name sanitization + super init.
        sanitized_name = (
            (name or f"mistral_{model}")
            .replace(":", "_").replace("-", "_")
            .replace(" ", "_").replace(".", "_")
        )
        super().__init__(
            name=sanitized_name,
            namespace=namespace,
            description=description or "Mistral LLM Engine",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        # 2. SDK presence check.
        if Mistral is None:
            raise LLMEngineError(
                "MistralEngine requires the `mistralai` package to be installed."
            )

        # 3. Build _ckw + seed timeout_ms default from the base-engine knob.
        _ckw = dict(client_kwargs)
        _ckw.setdefault("timeout_ms", int(self._timeout_seconds * 1000))

        # 4. Single client: injected as-is or built from kwargs.
        self._client: Mistral = client if client is not None else Mistral(**_ckw)

        # 5. Store model and generation config.
        self.model = model
        self.temperature = temperature
        self._inline_cutoff_chars = int(inline_cutoff_chars)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def inline_cutoff_chars(self) -> int:
        """Max characters inlined from text attachments; fixed at construction."""
        return self._inline_cutoff_chars

    # ------------------------------------------------------------------ #
    # Attachment validation & preparation
    # ------------------------------------------------------------------ #

    def _validate_attachment_path(self, path: str) -> None:
        """
        Validate ``path`` against the shared illegal-ext set and MIME-prefix rules.

        Delegates to ``validate_attachment_path`` (blacklist-only policy; no
        positive allow-list). Converts ``ValueError`` to ``LLMEngineError``.
        """
        try:
            validate_attachment_path(
                path,
                illegal_exts=ILLEGAL_ATTACHMENT_EXTS,
                allowed_exts=None,
                illegal_mime_prefixes=ENGINE_ILLEGAL_MIME_PREFIXES,
            )
        except ValueError as exc:
            raise LLMEngineError(str(exc)) from exc

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        """
        Prepare a local path for Mistral: upload/sign or inline as text/code.

        Metadata shapes:

        - PDFs/images → upload + sign; store `file_id` + `signed_url`:

            {
                "kind": "pdf" | "image",
                "mime": <str>,
                "ext": <str>,
                "uploaded": True,
                "file_id": <str>,
                "signed_url": <str>,
            }

        - Text/code → read & inline, respecting `inline_cutoff_chars` as a
          per-file safety cap (the global cap is enforced in `_build_provider_payload`):

            {
                "kind": "text",
                "mime": <str>,
                "ext": <str>,
                "inlined": True,
                "inlined_text": <str>,
            }
        """
        try:
            kind = self._classify_kind(path)
            mime, _ = mimetypes.guess_type(path)
            mime = mime or ""
            ext = os.path.splitext(path)[1].lower()

            if kind in ("pdf", "image"):
                file_id = self._upload_file(path)
                signed_url = self._sign_file(file_id)
                return {
                    "kind": kind,
                    "mime": mime,
                    "ext": ext,
                    "uploaded": True,
                    "file_id": file_id,
                    "signed_url": signed_url,
                }

            # text/code → inline
            text = self._read_text_file(path)
            if len(text) > self._inline_cutoff_chars:
                text = text[: self._inline_cutoff_chars] + "\n…[truncated]\n"
            return {
                "kind": "text",
                "mime": mime,
                "ext": ext,
                "inlined": True,
                "inlined_text": text,
            }
        except LLMEngineError:
            raise
        except Exception as exc:
            raise LLMEngineError(
                f"MistralEngine._prepare_attachment failed for {path!r}"
            ) from exc

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        """
        Delete Mistral file resource if present (best-effort).

        `meta` is the metadata previously returned by `_prepare_attachment`.
        Errors are swallowed; the base engine logs detach errors at debug level.
        """
        file_id = meta.get("file_id")
        if not file_id:
            return
        try:
            self._client.files.delete(file_id=file_id)
        except Exception:
            # Best-effort cleanup only.
            return

    # ------------------------------------------------------------------ #
    # Template hooks for invocation
    # ------------------------------------------------------------------ #

    def _build_provider_payload(
            self,
            messages: List[Dict[str, str]],
            attachments: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
        Map normalized messages + prepared attachments to Mistral chat schema.

        - `messages` are already validated and have lowercase `role` and
          string `content`.
        - `attachments` is a snapshot of `self._attachments` at invoke time.

        Inline text from attachments and signed URLs are appended as parts to
        the **last user message** so they are clearly associated with the most
        recent user query.
        """
        # Start from normalized messages.
        chat_messages: List[Dict[str, Any]] = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        # Ensure the last user message is a parts array we can extend.
        user_idx = self._ensure_user_parts(chat_messages)
        parts = chat_messages[user_idx]["content"]

        # Inline text/code with a global cutoff across all attachments.
        total_inlined = 0
        cutoff_marker = "\n[Inline cutoff reached; additional text files omitted]\n"
        for path, meta in attachments.items():
            kind = meta.get("kind")
            inlined_text = meta.get("inlined_text")
            if kind == "text" and inlined_text:
                budget = self._inline_cutoff_chars - total_inlined
                if budget <= 0:
                    if total_inlined == self._inline_cutoff_chars:
                        parts.append({"type": "text", "text": cutoff_marker})
                        total_inlined += len(cutoff_marker)
                    continue

                text = inlined_text
                if len(text) > budget:
                    text = text[:budget] + "\n…[truncated]\n"

                header = f"\n[Inlined file: {os.path.basename(path)}]\n"
                parts.append({"type": "text", "text": header + text})
                # We treat `inline_cutoff_chars` as a soft cap, so this is approximate.
                total_inlined += len(text)

        # Attach signed URLs for PDFs & images from persistent attachments.
        for _, meta in attachments.items():
            kind = meta.get("kind")
            signed_url = meta.get("signed_url")
            if kind == "pdf" and signed_url:
                parts.append({"type": "document_url", "document_url": signed_url})
            elif kind == "image" and signed_url:
                parts.append({"type": "image_url", "image_url": signed_url})

        return {"messages": chat_messages}

    def _call_provider(self, payload: Any) -> Any:
        """
        Single synchronous Mistral chat completion call.

        Retries and error-wrapping are handled by the shared ``LLMEngine`` template.
        Temperature is omitted when ``None`` so the SDK applies its own default.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": payload["messages"],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return self._client.chat.complete(**kwargs)

    async def _call_provider_async(self, payload: Any) -> Any:
        """
        Native async Mistral chat completion call via ``chat.complete_async``.

        ``Mistral`` (v2.6.0+) exposes both sync and async paths on the same
        object — no thread offload or isinstance routing needed. Temperature is
        omitted when ``None`` so the SDK applies its own default.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": payload["messages"],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return await self._client.chat.complete_async(**kwargs)

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        """
        Retry on ``NoResponseError`` (no HTTP response received at all) and
        on retryable HTTP status codes (429, 500, 502, 503, 504) from
        ``SDKError.raw_response``.
        """
        if attempt > self._max_retries:
            return False
        if isinstance(exc, mistral_errors.NoResponseError):
            return True
        if isinstance(exc, mistral_errors.SDKError):
            status = getattr(getattr(exc, "raw_response", None), "status_code", None)
            return status in {429, 500, 502, 503, 504}
        return False

    def _extract_text(self, response: Any) -> str:
        """
        Extract assistant text from a Mistral chat completion response.

        `response.choices[0].message.content` may be a string or a list of chunks.
        """
        msg = getattr(response.choices[0].message, "content", "")
        if isinstance(msg, list):
            msg = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in msg
            )
        return (msg or "").strip()

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        """
        Extract Mistral chat-completion token usage into a MistralTokenUsage record.

        Mistral chat-completion usage reports:

        - prompt_tokens: prompt/input-side tokens
        - completion_tokens: generated response tokens
        - total_tokens: total prompt + completion tokens
        - prompt_tokens_details.cached_tokens: cached prompt-token subset, when present
        """
        usage = response.usage
        if usage is None:
            raise LLMEngineError("Mistral response did not include usage.")

        if usage.prompt_tokens is None:
            raise LLMEngineError("Mistral usage missing prompt_tokens.")
        if usage.completion_tokens is None:
            raise LLMEngineError("Mistral usage missing completion_tokens.")
        if usage.total_tokens is None:
            raise LLMEngineError("Mistral usage missing total_tokens.")

        cached_tokens = None
        # The real mistralai.UsageInfo model never declares
        # prompt_tokens_details as an attribute at all (verified empirically
        # against the installed SDK) — plain attribute access raises
        # AttributeError on every real response. getattr with a default
        # makes this genuinely optional instead of assumed-present.
        prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_tokens_details is not None:
            if isinstance(prompt_tokens_details, Mapping):
                cached_tokens = prompt_tokens_details.get("cached_tokens")
            else:
                cached_tokens = prompt_tokens_details.cached_tokens

        return MistralTokenUsage(
            input_tokens=usage.prompt_tokens,
            generated_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            response_tokens=usage.completion_tokens,
            cached_tokens=cached_tokens,
        )

    def _get_model_data(self) -> LLMModelData:
        """
        Return configured Mistral model identity data for this engine.

        Model data is derived from engine configuration, not from the provider
        response object.
        """
        return RemoteLLMModelData(
            provider="mistral",
            model_name=self.model,
        )

    # ------------------------------------------------------------------ #
    # Mistral-specific helpers (not part of the template surface)
    # ------------------------------------------------------------------ #

    def _classify_kind(self, path: str) -> str:
        """
        Classify `path` into 'pdf' | 'image' | 'text'.

        Base `_validate_attachment_path` has already checked existence and coarse
        extension policy; here we only bucket by type.
        """
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf" or mime == "application/pdf":
            return "pdf"
        if ext in MISTRAL_IMAGE_EXTS or mime.startswith("image/"):
            return "image"
        # Fallback: treat as text/code and attempt to inline.
        return "text"

    def _ensure_user_parts(self, messages: List[Dict[str, Any]]) -> int:
        """
        Ensure there is a user message with `content` as a parts list.

        - If none exists: append an empty user turn.
        - If it's a string: convert to `[{\"type\": \"text\", \"text\": ...}]`.
        """
        idx = next(
            (
                i
                for i in range(len(messages) - 1, -1, -1)
                if (messages[i].get("role") or "").lower() == "user"
            ),
            None,
        )
        if idx is None:
            messages.append({"role": "user", "content": []})
            idx = len(messages) - 1

        content = messages[idx].get("content", "")
        if isinstance(content, list):
            # already parts
            return idx

        parts: List[Dict[str, Any]] = []
        if isinstance(content, str) and content:
            parts.append({"type": "text", "text": content})
        messages[idx]["content"] = parts
        return idx

    def _read_text_file(self, path: str) -> str:
        """Read a local file as UTF-8 (with replacement)."""
        try:
            with open(path, "rb") as f:
                raw = f.read()
            return raw.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - defensive
            return f"[Error reading file '{os.path.basename(path)}': {exc}]"

    def _upload_file(self, path: str) -> str:
        """Upload to Mistral Files; return file handle ID."""
        with open(path, "rb") as f:
            up = self._client.files.upload(
                file={"file_name": os.path.basename(path), "content": f},
                purpose="ocr",  # suitable for PDFs/images; used for doc/image understanding
            )
        return up.id  # string handle

    def _sign_file(self, file_id: str) -> str:
        """
        Obtain a signed URL for an uploaded file.

        If this call fails, the attachment fails; no additional retries are
        performed here (chat-level retries are handled by the base engine).
        """
        try:
            signed = self._client.files.get_signed_url(file_id=file_id)
            return signed.url
        except Exception as exc:
            raise LLMEngineError(
                f"MistralEngine._sign_file failed for file_id {file_id!r}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """
        Diagnostic snapshot for MistralEngine: provider + requested knobs.

        Includes non-secret configuration only.
        """
        base = super().to_dict()
        base.update({
            "model": self.model,
            "temperature": self.temperature,
            "inline_cutoff_chars": self._inline_cutoff_chars,
        })
        return base
