from __future__ import annotations

import json
import mimetypes
try:
    from mistralai.client import Mistral
    _SDK_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:
    Mistral = None
    _SDK_IMPORT_ERROR = _exc
import os
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
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
from ..utils.core import format_missing_dependency
from ..utils.llm import (
    classify_path,
    inlined_file_block,
    read_text_file,
    validate_attachment_path,
)

__all__ = ["MistralEngine"]


# ── MISTRAL ─────────────────────────────
class MistralEngine(LLMEngine):
    """
    Mistral adapter using: **upload → sign → attach URL parts**.

    Attachments
    -----------
    ``attach(path)`` prepares a file once:

    - PDFs and images: upload and sign, stored as ``{"kind": "pdf" | "image",
      "signed_url": ...}`` and sent as ``document_url`` / ``image_url`` parts.
    - Text/code: read and stored as ``{"kind": "text", "inlined_text": ...}``
      and sent as an inlined text part.

    The parts are appended to the last user message (converted to a parts
    list only when attachments exist). ``detach(path)`` deletes an uploaded
    file best-effort.

    Result extraction
    -----------------
    ``_extract_result`` joins ``choices[0].message.content`` (a string or a
    list of chunks) and, on a structured call, parses it with ``json.loads``
    (raw text on a parse failure).

    Structured-output schema policy
    -------------------------------
    ``output_structure`` is forwarded unmodified in ``response_format``
    (``json_schema`` with ``strict=self.strict``); Mistral rejects what it
    does not accept natively. ``strict=False`` accepts more keywords but
    enforces less (for example ``additionalProperties: false`` goes
    unenforced).

    Runtime parameters and configuration
    ------------------------------------
    Generation settings (`temperature`, `max_tokens`, ...) are `chat.complete`
    call parameters under Mistral's own names, held in the base engine's
    runtime-parameter store and forwarded verbatim:
    `engine.update_runtime_parameters(temperature=0.2)`. Nothing is set by
    default, so Mistral's own defaults apply. `strict` is the one AA behavior
    attribute: a validating, settable bool re-read on every structured-output
    call, mirroring `OpenAIEngine.strict`. Timeout and retry are the SDK
    client's own settings (`timeout_ms`, `retry_config`) through
    `client_kwargs`. `require_usage` (base) controls a response with no usage
    object.
    """

    _RESERVED_RUNTIME_KEYS = frozenset({"model", "messages", "response_format", "stream"})

    def __init__(
            self,
            model: str,
            name: str | None = None,
            description: str | None = None,
            client: Mistral | None = None,
            **client_kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Mistral model identifier (e.g. ``"mistral-small-latest"``).
        name:
            Optional human-friendly engine name; defaults to the class name.
        description:
            Human-readable description for this engine instance.
        client:
            Pre-built ``Mistral`` client. When provided, used directly
            (combining it with ``client_kwargs`` raises). ``Mistral`` exposes
            both sync (``.chat.complete``) and async (``.chat.complete_async``)
            paths on the same object — no isinstance routing is needed.
        **client_kwargs:
            Keyword arguments forwarded unchanged to ``Mistral(...)``. Common
            uses: ``api_key``, ``server_url``, ``timeout_ms``, ``retry_config``.

        ``strict`` starts ``True`` (Mistral's strict JSON-Schema mode).
        """
        # 1. SDK presence check.
        if Mistral is None:
            raise LLMEngineError(
                "MistralEngine "
                + format_missing_dependency("mistralai", "mistral", _SDK_IMPORT_ERROR)
            ) from _SDK_IMPORT_ERROR

        # 2. Exactly one client source.
        if client is not None and client_kwargs:
            raise LLMEngineError(
                "MistralEngine: pass either 'client' or client kwargs, not both."
            )

        # 3. Base init, then engine state.
        super().__init__(name=name, description=description or "Mistral LLM Engine")
        self._attachments: Dict[str, Mapping[str, Any]] = {}
        self._client: Mistral = client if client is not None else Mistral(**client_kwargs)
        self.model = model
        self.strict = True

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def strict(self) -> bool:
        """Whether structured-output requests use strict schema conformance."""
        return self._strict

    @strict.setter
    def strict(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise LLMEngineError(
                f"MistralEngine.strict must be a bool, got {type(value).__name__}."
            )
        self._strict = value

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
            raise LLMEngineError("MistralEngine.attach: path must be a non-empty string")
        if path in self._attachments:
            return self._attachments[path]

        self._validate_attachment_path(path)
        meta = self._prepare_attachment(path)
        self._attachments[path] = meta
        return meta

    def detach(self, path: str) -> bool:
        """
        Detach ``path``; ``True`` when something was removed. The uploaded file
        is deleted best-effort (a failed delete is ignored).
        """
        meta = self._attachments.pop(path, None)
        if meta is None:
            return False
        file_id = meta.get("file_id")
        if file_id:
            try:
                self._client.files.delete(file_id=file_id)
            except Exception:
                pass
        return True

    def clear_attachments(self) -> None:
        """Detach every attached path."""
        for path in list(self._attachments):
            self.detach(path)

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

        - Text/code → read & inline whole:

            {
                "kind": "text",
                "mime": <str>,
                "ext": <str>,
                "inlined": True,
                "inlined_text": <str>,
            }
        """
        try:
            kind = classify_path(path, MISTRAL_IMAGE_EXTS)
            mime, _ = mimetypes.guess_type(path)
            mime = mime or ""
            ext = os.path.splitext(path)[1].lower()

            if kind in ("pdf", "image"):
                file_id = self._upload_file(path)
                try:
                    signed_url = self._sign_file(file_id)
                except Exception:
                    # Don't leave an orphaned upload behind.
                    try:
                        self._client.files.delete(file_id=file_id)
                    except Exception:
                        pass
                    raise
                return {
                    "kind": kind,
                    "mime": mime,
                    "ext": ext,
                    "uploaded": True,
                    "file_id": file_id,
                    "signed_url": signed_url,
                }

            # text/code → inline
            text = read_text_file(path)
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
                f"MistralEngine._prepare_attachment failed for {path!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Template hooks for invocation
    # ------------------------------------------------------------------ #

    def _build_provider_payload(
            self,
            messages: List[Dict[str, str]],
            output_structure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Map normalized messages + prepared attachments (+ optionally the
        caller's raw structured-output template) to Mistral chat schema.

        - `messages` are already validated and have lowercase `role` and
          string `content`.
        - `output_structure`, when given, is the caller's raw schema,
          forwarded unmodified and wired into `response_format={"type":
          "json_schema", "json_schema": {"name": "output_structure",
          "schema": output_structure, "strict": self.strict}}`.

        When attachments exist, inline text and signed URLs are appended as
        parts to the last user message (converted to a parts list). Without
        attachments, message content passes through as plain strings.
        """
        attachments = dict(self._attachments)

        # Start from normalized messages.
        chat_messages: List[Dict[str, Any]] = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        # Attachments need the last user message as a parts array; without
        # them the message content passes through unchanged.
        if attachments:
            user_idx = self._ensure_user_parts(chat_messages)
            parts = chat_messages[user_idx]["content"]

            # Inline text/code attachments whole.
            for path, meta in attachments.items():
                inlined_text = meta.get("inlined_text")
                if meta.get("kind") == "text" and inlined_text:
                    parts.append(
                        {"type": "text", "text": inlined_file_block(path, inlined_text)}
                    )

            # Attach signed URLs for PDFs & images from persistent attachments.
            for _, meta in attachments.items():
                kind = meta.get("kind")
                signed_url = meta.get("signed_url")
                if kind == "pdf" and signed_url:
                    parts.append({"type": "document_url", "document_url": signed_url})
                elif kind == "image" and signed_url:
                    parts.append({"type": "image_url", "image_url": signed_url})

        payload: Dict[str, Any] = dict(self._runtime_parameters)
        payload.update(model=self.model, messages=chat_messages)
        if output_structure is not None:
            payload.update(response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "output_structure",
                    "schema": output_structure,
                    "strict": self.strict,
                },
            })
        return payload

    def _call_provider(self, payload: Any) -> Any:
        """
        Single synchronous Mistral chat completion call; the payload is the
        call's kwargs. Error-wrapping is handled by the shared ``LLMEngine``
        template.
        """
        return self._client.chat.complete(**payload)

    async def _call_provider_async(self, payload: Any) -> Any:
        """
        Native async Mistral chat completion call via ``chat.complete_async``.

        ``Mistral`` (v2.6.0+) exposes both sync and async paths on the same
        object — no thread offload or isinstance routing needed.
        """
        return await self._client.chat.complete_async(**payload)

    def _extract_result(
        self, response: Any, requested_structured: bool
    ) -> str | int | float | bool | list[Any] | dict[str, Any] | None:
        """
        Join `response.choices[0].message.content` (a string or a list of
        chunks); on a structured call, its `json.loads` value (raw text on a
        parse failure).
        """
        msg = getattr(response.choices[0].message, "content", "")
        if isinstance(msg, list):
            msg = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in msg
            )
        text = msg or ""
        if not requested_structured:
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        """
        Extract Mistral chat-completion token usage into a MistralTokenUsage record.

        Mistral chat-completion usage reports:

        - prompt_tokens: prompt/input-side tokens
        - completion_tokens: generated response tokens
        - total_tokens: total prompt + completion tokens
        - prompt_tokens_details.cached_tokens: cached prompt-token subset, when present

        A response with no usage object raises unless ``require_usage`` is
        ``False``, which yields an all-zero record.
        """
        usage = response.usage
        if usage is None:
            if not self.require_usage:
                return MistralTokenUsage(
                    input_tokens=0, generated_tokens=0, total_tokens=0, response_tokens=0
                )
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
        """Return configured Mistral model identity (from ``self.model``)."""
        return RemoteLLMModelData(
            provider="mistral",
            model_name=self.model,
        )

    # ------------------------------------------------------------------ #
    # Mistral-specific helpers (not part of the template surface)
    # ------------------------------------------------------------------ #

    def _ensure_user_parts(self, messages: List[Dict[str, Any]]) -> int:
        """
        Convert the last user message's string content into a parts list
        (``[]`` for empty content) and return its index.
        """
        idx = next(i for i in reversed(range(len(messages))) if messages[i]["role"] == "user")
        content = messages[idx]["content"]
        messages[idx]["content"] = [{"type": "text", "text": content}] if content else []
        return idx

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

        If this call fails, the attachment fails.
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
        Diagnostic snapshot for MistralEngine: provider + model + strict (the
        base adds the runtime parameters).

        Includes non-secret configuration only.
        """
        base = super().to_dict()
        base.update({
            "model": self.model,
            "strict": self.strict,
        })
        return base
