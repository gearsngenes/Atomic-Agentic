from __future__ import annotations

import asyncio
import json
import mimetypes
try:
    from openai import AsyncOpenAI, OpenAI
    _SDK_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:
    AsyncOpenAI = None
    OpenAI = None
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
    OPENAI_IMAGE_EXTS,
    OPENAI_ALLOWED_EXTS,
)
from ..exceptions import LLMEngineError
from ..models.results.llm import (
    TokenUsage,
    OpenAITokenUsage,
    LLMModelData,
    RemoteLLMModelData
)
from ..utils.llm import (
    classify_path,
    collect_system_text,
    inlined_file_block,
    read_text_file,
    validate_attachment_path,
)
from ..utils.core import format_missing_dependency, run_coro_sync

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

    ``_upload_file`` and ``detach`` also use ``isinstance`` to route
    file-API calls, wrapping async file operations in ``run_coro_sync``
    when needed.

    File policy
    -----------
    Attachments are persistent engine state, managed by this engine's own
    ``attach``/``detach``/``clear_attachments``:

    - PDFs    → uploaded once via Files API; attached as
      ``{"type": "input_file", "file_id": ...}``
    - Images  → uploaded once via Files API; attached as
      ``{"type": "input_image", "file_id": ...}``
    - Text/Code → read and inlined as
      ``{"type": "input_text", "text": ...}`` (whole file).

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
    ``_extract_result`` returns ``response.output_text``, parsed with
    ``json.loads`` on a structured call (raw text on a parse failure).
    ``_extract_token_usage`` maps Responses API usage into ``OpenAITokenUsage``.

    Structured-output schema policy
    -------------------------------
    ``output_structure`` is forwarded unmodified to ``text.format``; OpenAI
    rejects what it does not accept natively (strict mode, for example,
    requires ``additionalProperties: false`` on every object).

    Runtime parameters and configuration
    ------------------------------------
    Generation settings (``temperature``, ``max_output_tokens``, ``reasoning``,
    ``truncation``, ``text`` options such as ``verbosity``, ...) are Responses
    API call parameters under OpenAI's own names, held in the base engine's
    runtime-parameter store and forwarded verbatim:
    ``engine.update_runtime_parameters(temperature=0.2, reasoning={"effort": "high"})``.
    AA applies no gate between them (the provider decides which combinations are
    valid). ``strict`` is the one AA behavior attribute: a validated, settable
    bool that only shapes the structured-output envelope. Timeout and retry are
    the SDK client's own settings (``timeout``, ``max_retries``), passed through
    ``client_kwargs``. ``require_usage`` (base) controls a response with no
    usage object.
    """

    _RESERVED_RUNTIME_KEYS = frozenset({"model", "input", "instructions", "stream"})

    def __init__(
            self,
            model: str,
            name: str | None = None,
            description: str | None = None,
            client: OpenAI | AsyncOpenAI | None = None,
            **client_kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        model:
            OpenAI model identifier (e.g. ``"gpt-4.1"``, ``"gpt-4o-mini"``).
        name:
            Optional human-friendly engine name; defaults to the class name.
        description:
            Human-readable description for this engine instance.
        client:
            Pre-built ``OpenAI`` or ``AsyncOpenAI`` client. When provided, it is
            used directly; combining it with ``client_kwargs`` raises. When
            ``None``, a sync ``OpenAI(**client_kwargs)`` is built. An injected
            ``AsyncOpenAI`` routes sync calls through ``run_coro_sync``; an injected
            ``OpenAI`` routes async calls through ``asyncio.to_thread``.
        **client_kwargs:
            Keyword arguments forwarded unchanged to ``OpenAI(...)``. Common uses:
            ``api_key``, ``base_url``, ``organization``, and the SDK's own
            ``timeout`` and ``max_retries``.

        Lifecycle: SDK presence check; reject ``client`` combined with
        ``client_kwargs``; base init; build or adopt the client; ``strict``
        starts ``True``. Generation parameters start empty
        (``update_runtime_parameters``).
        """
        # 1. SDK presence check.
        if OpenAI is None:
            raise LLMEngineError(
                "OpenAIEngine "
                + format_missing_dependency("openai", "openai", _SDK_IMPORT_ERROR)
            ) from _SDK_IMPORT_ERROR

        # 2. Exactly one client source.
        if client is not None and client_kwargs:
            raise LLMEngineError(
                "OpenAIEngine: pass either 'client' or client kwargs, not both."
            )

        # 3. Base init, then engine state.
        super().__init__(name=name, description=description or "OpenAI LLM Engine")
        self._attachments: Dict[str, Mapping[str, Any]] = {}
        self._client: OpenAI | AsyncOpenAI = (
            client if client is not None else OpenAI(**client_kwargs)
        )
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
                f"OpenAIEngine.strict must be a bool, got {type(value).__name__}."
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
            raise LLMEngineError("OpenAIEngine.attach: path must be a non-empty string")
        if path in self._attachments:
            return self._attachments[path]

        self._validate_attachment_path(path)
        meta = self._prepare_attachment(path)
        self._attachments[path] = meta
        return meta

    def detach(self, path: str) -> bool:
        """
        Detach ``path``; ``True`` when something was removed. An uploaded file
        is deleted best-effort (a failed delete is ignored).
        """
        meta = self._attachments.pop(path, None)
        if meta is None:
            return False
        file_id = meta.get("file_id")
        if file_id:
            try:
                if isinstance(self._client, AsyncOpenAI):
                    run_coro_sync(self._client.files.delete(file_id))
                else:
                    self._client.files.delete(file_id)
            except Exception:
                pass
        return True

    def clear_attachments(self) -> None:
        """Detach every attached path."""
        for path in list(self._attachments):
            self.detach(path)

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
            output_structure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the exact keyword arguments of ``responses.create``.

        1. Start from a snapshot of the runtime parameters.
        2. Convert messages into ``input`` blocks and system turns into
           ``instructions``.
        3. Append the engine's attachments (a snapshot) to the last ``user``
           block.
        4. ``update`` the engine-owned fields (``model``, ``input``,
           ``instructions``) over the runtime snapshot, so they always win.
        5. Structured call: set ``text["format"]`` on a copy of the runtime
           ``text`` (the caller's raw schema, unmodified; other ``text``
           options such as ``verbosity`` survive).
        """
        blocks = self._build_role_blocks(messages)
        attachments = dict(self._attachments)
        if attachments:
            last_user = next(b for b in reversed(blocks) if b["role"] == "user")
            user_parts: List[Dict[str, Any]] = last_user["content"]
            for path, meta in attachments.items():
                kind = str(meta.get("kind", "text"))
                if meta.get("inlined"):
                    text = str(meta.get("inlined_text") or "")
                    if text:
                        user_parts.append(
                            {"type": "input_text", "text": inlined_file_block(path, text)}
                        )
                elif meta.get("uploaded") and meta.get("file_id"):
                    part_type = "input_file" if kind == "pdf" else "input_image"
                    user_parts.append({"type": part_type, "file_id": str(meta["file_id"])})

        payload: Dict[str, Any] = dict(self._runtime_parameters)
        payload.update(model=self.model, input=blocks)
        instructions = collect_system_text(messages)
        if instructions:
            payload.update(instructions=instructions)
        if output_structure is not None:
            text = dict(payload.get("text") or {})
            text["format"] = {
                "type": "json_schema",
                "name": "output_structure",
                "schema": output_structure,
                "strict": self.strict,
            }
            payload.update(text=text)
        return payload

    def _call_provider(self, payload: Dict[str, Any]) -> Any:
        """
        Perform a single Responses API call with the payload as its kwargs.

        Routes via ``run_coro_sync`` when the injected client is ``AsyncOpenAI``.
        """
        if isinstance(self._client, AsyncOpenAI):
            return run_coro_sync(self._client.responses.create(**payload))
        return self._client.responses.create(**payload)

    async def _call_provider_async(self, payload: Dict[str, Any]) -> Any:
        """
        Native async Responses API call.

        Falls back to ``asyncio.to_thread`` when only a sync client is available.
        """
        if isinstance(self._client, AsyncOpenAI):
            return await self._client.responses.create(**payload)
        return await asyncio.to_thread(self._call_provider, payload)

    def _extract_result(
        self, response: Any, requested_structured: bool
    ) -> str | int | float | bool | list[Any] | dict[str, Any] | None:
        """
        Return ``response.output_text``; on a structured call, its
        ``json.loads`` value (raw text when it does not parse, e.g. after
        output-token truncation).
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
        - input_tokens_details.cached_tokens: input-token subset served from an
          existing prompt-cache entry, i.e. a reuse (optional)
        - input_tokens_details.cache_write_tokens: input-token subset newly
          stored into the prompt cache by this call, i.e. seeding a future
          reuse (optional; requires openai>=2.45)
        - output_tokens_details.reasoning_tokens: hidden reasoning-token subset
          counted inside output_tokens (optional)

        Both ``output_tokens_details`` and ``input_tokens_details`` are optional and
        may be absent (``None``) from the response.  When absent, ``reasoning_tokens``
        defaults to ``0`` (no reasoning occurred) and ``cached_tokens``/
        ``cache_write_tokens`` both default to ``None`` (not reported).

        ``response_tokens`` is derived as ``output_tokens - reasoning_tokens``.
        A response with no usage object raises unless ``require_usage`` is
        ``False``, which yields an all-zero record.
        """
        usage = response.usage
        if usage is None:
            if not self.require_usage:
                return OpenAITokenUsage(
                    input_tokens=0, generated_tokens=0, total_tokens=0, response_tokens=0
                )
            raise LLMEngineError("OpenAI response did not include usage.")

        output_details = usage.output_tokens_details
        reasoning_tokens = output_details.reasoning_tokens if output_details is not None else 0

        response_tokens = usage.output_tokens - reasoning_tokens

        input_details = usage.input_tokens_details
        cached_tokens = input_details.cached_tokens if input_details is not None else None
        cache_write_tokens = input_details.cache_write_tokens if input_details is not None else None

        return OpenAITokenUsage(
            input_tokens=usage.input_tokens,
            generated_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            response_tokens=response_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
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
        - Text/code → inline whole as text (UTF-8).
        """
        try:
            kind = classify_path(path, OPENAI_IMAGE_EXTS)
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
            # Normalize unexpected errors to LLMEngineError so callers see a consistent type.
            raise LLMEngineError(
                f"OpenAIEngine._prepare_attachment failed for {path!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # OpenAI-specific helpers (not part of the template surface)
    # ------------------------------------------------------------------ #

    def _build_role_blocks(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convert chat messages into Responses API blocks.

        - `assistant` turns → `output_text`
        - Other non-system turns → `input_text`
        - `system` content is carried only via `instructions`.
        - An empty user turn is kept (attachments attach to the last user
          block); other empty turns are skipped.
        """
        blocks: List[Dict[str, Any]] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                continue  # carried via `instructions`
            text = m["content"]
            if not text and role != "user":
                # Keep empty user (we attach into it), skip other empty roles.
                continue
            part_type = "output_text" if role == "assistant" else "input_text"
            parts: List[Dict[str, Any]] = []
            if text:
                parts.append({"type": part_type, "text": text})
            blocks.append({"role": role, "content": parts})
        return blocks

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

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """
        Diagnostic snapshot for OpenAIEngine, without secrets.

        Includes model and the strict flag in addition to
        base engine info (which carries the runtime parameters). Does not expose ``client_kwargs`` contents (which may
        include API keys).
        """
        base = super().to_dict()
        base.update({
            "model": self.model,
            "strict": self.strict,
        })
        return base
