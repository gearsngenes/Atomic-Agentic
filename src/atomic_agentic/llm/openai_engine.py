from __future__ import annotations

import asyncio
import json
import mimetypes
try:
    from openai import AsyncOpenAI, OpenAI, APIConnectionError, APIStatusError
except ImportError:
    AsyncOpenAI = None
    OpenAI = None
    APIConnectionError = None
    APIStatusError = None
import os
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    ClassVar,
)

from .base import LLMEngine
from ..constants.llm import (
    ILLEGAL_ATTACHMENT_EXTS,
    ENGINE_ILLEGAL_MIME_PREFIXES,
    OPENAI_IMAGE_EXTS,
    OPENAI_ALLOWED_EXTS,
    OPENAI_STRUCTURE_OMITTED_KEYS,
)
from ..exceptions import LLMEngineError
from ..models.results.llm import (
    TokenUsage,
    OpenAITokenUsage,
    LLMModelData,
    RemoteLLMModelData
)
from ..utils.llm import validate_attachment_path
from ..utils.core import run_coro_sync

__all__ = ["OpenAIEngine"]

# ── OPENAI (Responses API) ─────────────────────────────────────────────────────
class OpenAIEngine(LLMEngine):
    """
    OpenAI adapter using the Responses API.

    Client model
    ------------
    A single ``_client`` attribute holds either an ``OpenAI`` or
    ``AsyncOpenAI`` instance. Sync/async routing is determined at call time
    via ``isinstance(self._client, AsyncOpenAI)``:

    - ``AsyncOpenAI`` injected: ``_call_provider`` bridges to the async path
      via ``run_coro_sync``; ``_call_provider_async`` awaits natively.
    - ``OpenAI`` injected (or built from kwargs): ``_call_provider`` calls
      directly; ``_call_provider_async`` offloads to a thread via
      ``asyncio.to_thread``.

    ``_upload_file`` and ``_on_detach`` also use ``isinstance`` to route
    file-API calls, wrapping async file operations in ``run_coro_sync``
    when needed.

    File policy
    -----------
    Attachments are persistent engine state:

    - PDFs    → uploaded once via Files API; attached as
      ``{"type": "input_file", "file_id": ...}``
    - Images  → uploaded once via Files API; attached as
      ``{"type": "input_image", "file_id": ...}``
    - Text/Code → read and inlined as
      ``{"type": "input_text", "text": ...}``
      with a configurable character cutoff.

    Unsupported file classes such as audio, video, archives, executables,
    databases, model weights, and obviously binary files are rejected at
    ``attach`` time.

    Invocation
    ----------
    This engine uses ``client.responses.create(...)`` in ``_call_provider``.
    System messages are carried via the Responses API ``instructions`` field;
    non-system messages are encoded as ``input_text`` or ``output_text`` blocks.

    Result extraction
    -----------------
    ``_extract_result`` reads the generated assistant reply from the Responses
    API response. For a plain-text call (``requested_structured=False``) it
    returns ``response.output_text`` unchanged. For a structured call, it
    attempts ``json.loads(response.output_text)`` and falls back to
    ``response.output_text`` on a parse failure rather than raising —
    detection is the base-supplied ``requested_structured`` flag, not
    response introspection (Pass 3 revision; previously inspected
    ``response.text.format.type`` and raised uncaught on parse failure).
    Renamed from ``_extract_text`` (structured-generation Pass 1 base
    contract widening).
    ``_extract_token_usage`` maps Responses API usage fields into an
    ``OpenAITokenUsage`` record. ``_get_model_data`` returns configured model
    identity from ``self.model``.

    Mutable configuration
    ----------------------
    ``temperature``, ``max_output_tokens``, ``reasoning``, ``truncation``, and
    ``strict`` are call-time Responses API knobs, re-read fresh on every call —
    each is a ``@property`` with a validating setter, freely settable after
    construction. ``inline_cutoff_chars`` and (base) ``timeout_seconds`` stay
    read-only instead: both are baked into already-constructed state
    (attachment metadata / the SDK client's own timeout option respectively)
    at the moment they're used, so mutating them later wouldn't propagate.
    """

    structure_omitted_keys:ClassVar[frozenset[str]] = OPENAI_STRUCTURE_OMITTED_KEYS

    def __init__(
            self,
            model: str,
            name: str | None = None,
            namespace: str = "llm",
            description: str = "OpenAI LLM Engine",
            client: OpenAI | AsyncOpenAI | None = None,
            temperature: float | None = None,
            max_output_tokens: int | None = None,
            reasoning: dict[str, str] | None = None,
            truncation: str | None = None,
            strict: bool = True,
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
            OpenAI model identifier (e.g. ``"gpt-4.1"``, ``"gpt-4o-mini"``).
        name:
            Optional human-friendly engine name; defaults to ``openai_{model}``
            with non-identifier characters replaced by underscores.
        namespace:
            Grouping label; inherited by the base engine as ``"llm"`` by default.
        description:
            Human-readable description for this engine instance.
        client:
            Pre-built ``OpenAI`` or ``AsyncOpenAI`` client. When provided, it is
            used directly and no client is constructed from ``client_kwargs``.
            When ``None``, a sync ``OpenAI(**client_kwargs)`` is built. An injected
            ``AsyncOpenAI`` routes sync calls through ``run_coro_sync``; an injected
            ``OpenAI`` routes async calls through ``asyncio.to_thread``.
        temperature:
            Responses API sampling temperature. Pass ``None`` to omit the parameter
            entirely (required for reasoning models).
        max_output_tokens:
            Optional maximum number of output tokens to generate.
        reasoning:
            Optional reasoning configuration dict (e.g. ``{"effort": "high"}``).
            When set, ``temperature`` is suppressed regardless of its value.
        truncation:
            Optional truncation strategy string (e.g. ``"auto"``).
        strict:
            Whether structured-output requests (``output_structure``) use
            OpenAI's strict schema-conformance mode. ``True`` (default) is
            OpenAI's own recommended setting and matches this pass's original
            behavior; also governs ``structure_omitted_keys`` (``default`` is
            only stripped from schemas while ``strict`` is ``True`` — OpenAI's
            own docs tie that restriction specifically to strict mode).
        inline_cutoff_chars:
            Maximum characters to inline from text/code attachments.
        timeout_seconds:
            Per-call timeout inserted into the client's ``timeout`` option via
            ``setdefault`` (user-supplied ``timeout`` in ``client_kwargs`` wins).
        max_retries, retry_backoff_base, retry_backoff_max:
            Shared ``LLMEngine`` retry/backoff configuration.
        **client_kwargs:
            Additional keyword arguments forwarded verbatim to ``OpenAI(...)``
            during client construction. Common uses: ``api_key``, ``base_url``,
            ``organization``. Not forwarded when ``client`` is injected directly.
        """
        # Step 1 — Name sanitization and base init.
        sanitized_name = (
            (name or f"openai_{model}")
            .replace(":", "_").replace("-", "_")
            .replace(" ", "_").replace(".", "_")
        )
        super().__init__(
            name=sanitized_name,
            namespace=namespace,
            description=description or "OpenAI LLM Engine",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        # Step 2 — SDK presence check.
        if OpenAI is None:
            raise LLMEngineError(
                "OpenAIEngine requires the `openai` package; install `openai` to use it."
            )

        # Step 3 — Build shared client kwargs; seed timeout from the base-engine knob.
        _ckw = dict(client_kwargs)
        _ckw.setdefault("timeout", self._timeout_seconds)

        # Step 4 — Single client: injected as-is or built sync from kwargs.
        self._client: OpenAI | AsyncOpenAI = (
            client if client is not None else OpenAI(**_ckw)
        )

        # Step 5 — Store model and Responses API config. The five knobs below
        # go through their own property setters (validation lives in one
        # place, shared by construction and later mutation).
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning = reasoning
        self.truncation = truncation
        self.strict = strict
        self._inline_cutoff_chars = int(inline_cutoff_chars)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def inline_cutoff_chars(self) -> int:
        """Max characters inlined from text attachments; fixed at construction."""
        return self._inline_cutoff_chars

    @property
    def temperature(self) -> float | None:
        """Responses API sampling temperature; ``None`` omits the parameter."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float | None) -> None:
        if value is not None and not isinstance(value, (int, float)):
            raise LLMEngineError(
                f"OpenAIEngine.temperature must be a number or None, got {type(value).__name__}."
            )
        self._temperature = value

    @property
    def max_output_tokens(self) -> int | None:
        """Maximum output tokens to generate; ``None`` omits the parameter."""
        return self._max_output_tokens

    @max_output_tokens.setter
    def max_output_tokens(self, value: int | None) -> None:
        if value is not None and not isinstance(value, int):
            raise LLMEngineError(
                f"OpenAIEngine.max_output_tokens must be an int or None, got {type(value).__name__}."
            )
        self._max_output_tokens = value

    @property
    def reasoning(self) -> dict[str, str] | None:
        """Reasoning configuration dict; suppresses ``temperature`` when set."""
        return self._reasoning

    @reasoning.setter
    def reasoning(self, value: dict[str, str] | None) -> None:
        if value is not None and not isinstance(value, dict):
            raise LLMEngineError(
                f"OpenAIEngine.reasoning must be a dict or None, got {type(value).__name__}."
            )
        self._reasoning = value

    @property
    def truncation(self) -> str | None:
        """Truncation strategy string; ``None`` omits the parameter."""
        return self._truncation

    @truncation.setter
    def truncation(self, value: str | None) -> None:
        if value is not None and not isinstance(value, str):
            raise LLMEngineError(
                f"OpenAIEngine.truncation must be a str or None, got {type(value).__name__}."
            )
        self._truncation = value

    @property
    def strict(self) -> bool:
        """Whether structured-output requests use strict schema conformance."""
        return self._strict

    @strict.setter
    def strict(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise LLMEngineError(
                f"OpenAIEngine.strict must be a bool, got {type(value).__name__}."
            )
        self._strict = value

    # ------------------------------------------------------------------ #
    # Overrides / template hooks
    # ------------------------------------------------------------------ #

    def _validate_attachment_path(self, path: str) -> None:
        """
        Validate `path` against the OpenAI illegal-ext set and MIME-prefix rules.

        Delegates to the shared ``validate_attachment_path`` utility, which
        checks existence, extension, allow-list membership, and MIME prefixes.
        Converts ``ValueError`` from the helper into ``LLMEngineError``.
        """
        try:
            validate_attachment_path(
                path,
                illegal_exts=ILLEGAL_ATTACHMENT_EXTS,
                allowed_exts=OPENAI_ALLOWED_EXTS,
                illegal_mime_prefixes=ENGINE_ILLEGAL_MIME_PREFIXES,
            )
        except ValueError as exc:
            raise LLMEngineError(str(exc)) from exc

    def _build_provider_payload(
            self,
            messages: List[Dict[str, str]],
            attachments: Mapping[str, Mapping[str, Any]],
            output_structure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the payload for the OpenAI Responses API from normalized messages,
        the current attachments snapshot, and (optionally) an already-cleaned
        structured-output template.
        """
        instructions = self._collect_instructions(messages)
        blocks = self._build_role_blocks(messages)

        payload: Dict[str, Any] = {"blocks": blocks}
        if instructions:
            payload["instructions"] = instructions
        if output_structure is not None:
            # Already pruned by clean_structure_template upstream (base
            # LLMEngine._call_model/_call_model_async); stash as-is.
            payload["output_structure"] = output_structure

        # No attachments: avoid creating an artificial empty user turn.
        if not attachments:
            return payload

        # Find or create a `user` block to hold attachments.
        user_idx = self._ensure_user_block(blocks)
        user_parts: List[Dict[str, Any]] = blocks[user_idx]["content"]

        for path, meta in attachments.items():
            kind = str(meta.get("kind", "text"))

            # Inlined text (text/code attachments prepared by _prepare_attachment).
            if meta.get("inlined"):
                text = str(meta.get("inlined_text") or "")
                if not text:
                    continue
                header = f"\n[Inlined file: {os.path.basename(path)}]\n"
                user_parts.append(
                    {"type": "input_text", "text": header + text}
                )
                continue

            # Uploaded files/images with a file_id
            if meta.get("uploaded") and meta.get("file_id"):
                file_id = str(meta["file_id"])
                if kind == "pdf":
                    user_parts.append({"type": "input_file", "file_id": file_id})
                else:
                    user_parts.append({"type": "input_image", "file_id": file_id})
                continue

            # Fallback: try to handle the local path directly (older metadata, etc.)
            try:
                self._attach_local_path(user_parts, path)
            except Exception:
                # Best-effort; silently skip failing attachments.
                continue

        return payload

    def _build_call_kwargs(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Build the shared Responses API kwargs from a pre-built payload dict.

        Used by both sync and async call methods to avoid duplication.

        Temperature gate: ``temperature`` is only forwarded when it is not
        ``None`` AND ``reasoning`` is not set — reasoning-mode models reject
        the ``temperature`` parameter.
        """
        blocks = payload["blocks"]
        instructions = payload.get("instructions")

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": blocks,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if self.temperature is not None and self.reasoning is None:
            kwargs["temperature"] = self.temperature
        if self.max_output_tokens is not None:
            kwargs["max_output_tokens"] = self.max_output_tokens
        if self.reasoning is not None:
            kwargs["reasoning"] = self.reasoning
        if self.truncation is not None:
            kwargs["truncation"] = self.truncation
        output_structure = payload.get("output_structure")
        if output_structure is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "output_structure",
                    "schema": output_structure,
                    "strict": self.strict,
                }
            }
        return kwargs

    def _call_provider(self, payload: Dict[str, Any]) -> Any:
        """
        Perform a single Responses API call using the pre-built payload.

        Routes via ``run_coro_sync`` when the injected client is ``AsyncOpenAI``.
        """
        if isinstance(self._client, AsyncOpenAI):
            return run_coro_sync(
                self._client.responses.create(**self._build_call_kwargs(payload))
            )
        return self._client.responses.create(**self._build_call_kwargs(payload))

    async def _call_provider_async(self, payload: Dict[str, Any]) -> Any:
        """
        Native async Responses API call.

        Falls back to ``asyncio.to_thread`` when only a sync client is available.
        """
        if isinstance(self._client, AsyncOpenAI):
            return await self._client.responses.create(**self._build_call_kwargs(payload))
        return await asyncio.to_thread(self._call_provider, payload)

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        """
        Retry on OpenAI connection/timeout errors and on retryable HTTP
        status codes (429, 500, 502, 503, 504) from ``APIStatusError``.
        """
        if attempt > self._max_retries:
            return False
        if isinstance(exc, APIConnectionError):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code in {429, 500, 502, 503, 504}
        return False

    def _extract_result(
        self, response: Any, requested_structured: bool
    ) -> str | list[Any] | dict[str, Any]:
        """
        Extract the assistant's textual or structured reply from a Responses
        API response object.

        ``requested_structured`` is the base-supplied signal (was
        ``output_structure`` given for this call) — authoritative, since it's
        the same condition that decided whether ``_build_call_kwargs`` set
        ``text.format`` in the first place; no response introspection needed.
        When not requested, returns ``response.output_text`` unchanged (empty
        string when no text is present; does not raise). When requested,
        attempts ``json.loads(response.output_text)``; a parse failure
        (shouldn't happen under ``strict: True``, but possible under e.g.
        output-token truncation) falls back to returning the raw text instead
        of raising.
        """
        if not requested_structured:
            return response.output_text
        try:
            return json.loads(response.output_text)
        except json.JSONDecodeError:
            return response.output_text

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        """
        Extract OpenAI Responses API token usage into an OpenAITokenUsage record.

        OpenAI Responses usage reports:

        - input_tokens: prompt/input-side tokens
        - output_tokens: generated-side tokens
        - total_tokens: total input + generated tokens
        - input_tokens_details.cached_tokens: cached input-token subset (optional)
        - output_tokens_details.reasoning_tokens: hidden reasoning-token subset
          counted inside output_tokens (optional)

        Both ``output_tokens_details`` and ``input_tokens_details`` are optional and
        may be absent (``None``) from the response.  When absent, ``reasoning_tokens``
        defaults to ``0`` (no reasoning occurred) and ``cached_tokens`` defaults to
        ``None`` (not reported).

        ``response_tokens`` is derived as ``output_tokens - reasoning_tokens``.
        """
        usage = response.usage
        if usage is None:
            raise LLMEngineError("OpenAI response did not include usage.")

        output_details = usage.output_tokens_details
        reasoning_tokens = output_details.reasoning_tokens if output_details is not None else 0

        response_tokens = usage.output_tokens - reasoning_tokens

        input_details = usage.input_tokens_details
        cached_tokens = input_details.cached_tokens if input_details is not None else None

        return OpenAITokenUsage(
            input_tokens=usage.input_tokens,
            generated_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            response_tokens=response_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def _get_model_data(self) -> LLMModelData:
        """
        Return configured OpenAI model identity data for this engine.

        Model data is derived from engine configuration, not from the provider
        response object.
        """
        return RemoteLLMModelData(
            provider="openai",
            model_name=self.model,
        )

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        """
        Prepare a local path for reuse with this engine.

        - PDFs/images → upload once to Files API; metadata contains `file_id`.
        - Text/code → inline as text (UTF-8, with a length cutoff).
        """
        try:
            kind = self._classify_path(path)
            mime, _ = mimetypes.guess_type(path)
            mime = mime or ""
            ext = os.path.splitext(path)[1].lower()

            # PDF/image: upload and keep a handle
            if kind in ("pdf", "image"):
                file_id = self._upload_file(path)
                return {
                    "kind": kind,
                    "mime": mime,
                    "ext": ext,
                    "uploaded": True,
                    "file_id": file_id,
                }

            # Text/code → inline as text
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
            # Normalize unexpected errors to LLMEngineError so callers see a consistent type.
            raise LLMEngineError(
                f"OpenAIEngine._prepare_attachment failed for {path!r}"
            ) from exc

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        """
        Attempt to delete any remote file created via the Files API.

        This is best-effort; errors are ignored.
        """
        file_id = meta.get("file_id")
        if not file_id:
            return
        try:
            if isinstance(self._client, AsyncOpenAI):
                run_coro_sync(self._client.files.delete(file_id))
            else:
                self._client.files.delete(file_id)
        except Exception:
            return

    # ------------------------------------------------------------------ #
    # OpenAI-specific helpers (not part of the template surface)
    # ------------------------------------------------------------------ #

    def _classify_path(self, path: str) -> str:
        """
        Classify `path` into 'pdf' | 'image' | 'text'.

        Base validation has already run; this is a semantic bucketization.
        """
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf" or mime == "application/pdf":
            return "pdf"
        if ext in OPENAI_IMAGE_EXTS or mime.startswith("image/"):
            return "image"
        return "text"

    def _collect_instructions(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Concatenate all system message contents into an `instructions` string.
        """
        parts = [
            m["content"]
            for m in messages
            if (m.get("role") or "").lower() == "system" and m.get("content")
        ]
        joined = "\n\n".join(parts).strip()
        return joined or None

    def _build_role_blocks(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convert chat messages into Responses API blocks.

        - `assistant` turns → `output_text`
        - Other non-system turns → `input_text`
        - `system` content is carried only via `instructions`.
        """
        blocks: List[Dict[str, Any]] = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            if role == "system":
                continue  # handled by _collect_instructions
            text = m.get("content") or ""
            if not text and role != "user":
                # Keep empty user (we attach into it), skip other empty roles.
                continue
            part_type = "output_text" if role == "assistant" else "input_text"
            parts: List[Dict[str, Any]] = []
            if text:
                parts.append({"type": part_type, "text": text})
            blocks.append({"role": role, "content": parts})
        return blocks

    def _ensure_user_block(self, blocks: List[Dict[str, Any]]) -> int:
        """
        Return the index of a `user` block; create an empty one at the end if none exist.
        """
        for i in range(len(blocks) - 1, -1, -1):
            if blocks[i].get("role") == "user":
                return i
        blocks.append({"role": "user", "content": []})
        return len(blocks) - 1

    def _read_text_file(self, path: str) -> str:
        """
        Read a local file as UTF-8 (with replacement). Returns a best-effort string.
        """
        try:
            with open(path, "rb") as f:
                raw = f.read()
            return raw.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - unlikely IO failures
            return f"[Error reading file '{os.path.basename(path)}': {exc}]"

    def _upload_file(self, path: str) -> str:
        """
        Upload a local file to the OpenAI Files API and return its `file_id`.

        We use purpose="assistants" which is appropriate for model context files.
        """
        with open(path, "rb") as fp:
            if isinstance(self._client, AsyncOpenAI):
                f = run_coro_sync(
                    self._client.files.create(file=fp, purpose="assistants")
                )
            else:
                f = self._client.files.create(file=fp, purpose="assistants")
        return str(f.id)

    def _attach_local_path(
            self,
            user_parts: List[Dict[str, Any]],
            path: str,
    ) -> None:
        """
        Fallback helper for attaching a path directly when metadata is incomplete.

        - PDFs   → upload + `input_file` reference
        - Images → upload + `input_image` reference
        - Other  → inline as text (UTF-8, truncated by `inline_cutoff_chars`).
        """
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        lower = path.lower()
        is_pdf = mime == "application/pdf" or lower.endswith(".pdf")
        is_image = mime.startswith("image/") or lower.endswith(OPENAI_IMAGE_EXTS)

        if is_pdf:
            file_id = self._upload_file(path)
            user_parts.append({"type": "input_file", "file_id": file_id})
            return

        if is_image:
            file_id = self._upload_file(path)
            user_parts.append({"type": "input_image", "file_id": file_id})
            return

        # Inline as text
        text = self._read_text_file(path)
        if len(text) > self._inline_cutoff_chars:
            text = text[: self._inline_cutoff_chars] + "\n…[truncated]\n"

        header = f"\n[Inlined file: {os.path.basename(path)}]\n"
        user_parts.append({"type": "input_text", "text": header + text})

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """
        Diagnostic snapshot for OpenAIEngine, without secrets.

        Includes model, Responses API config, and inline cutoff in addition to
        base engine info. Does not expose ``client_kwargs`` contents (which may
        include API keys).
        """
        base = super().to_dict()
        base.update({
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "reasoning": self.reasoning,
            "truncation": self.truncation,
            "strict": self.strict,
            "inline_cutoff_chars": self._inline_cutoff_chars,
        })
        return base
