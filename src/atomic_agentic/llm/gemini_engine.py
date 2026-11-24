from __future__ import annotations

import json
import mimetypes
try:
    from google import genai
    _SDK_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:
    genai = None
    _SDK_IMPORT_ERROR = _exc
import os
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from .base import LLMEngine
from ..constants.llm import (
    ILLEGAL_ATTACHMENT_EXTS,
    ENGINE_ILLEGAL_MIME_PREFIXES,
)
from ..exceptions import LLMEngineError
from ..models.results.llm import (
    TokenUsage,
    GeminiTokenUsage,
    LLMModelData,
    RemoteLLMModelData
)
from ..utils.core import format_missing_dependency
from ..utils.llm import collect_system_text, validate_attachment_path

__all__ = ["GeminiEngine"]

# ── GEMINI (flat contents: file objects + strings) ─────────────────────────────
class GeminiEngine(LLMEngine):
    """
    Google Gemini adapter using the Google Gen AI SDK.

    Client model
    ------------
    A single ``_client`` attribute holds a ``genai.Client`` instance.
    ``genai.Client`` exposes both sync (``client.models``) and async
    (``client.aio.models``) paths — no isinstance routing is needed.

    ``client=None`` builds ``genai.Client(**client_kwargs)`` with the kwargs
    unchanged; timeout is the SDK's own ``http_options.timeout`` (milliseconds).
    An injected client is used as-is, so set its options on the client itself;
    combining ``client`` with ``client_kwargs`` raises.

    Runtime parameters
    ------------------
    Generation settings (``temperature``, ``max_output_tokens``,
    ``thinking_config``, ...) are ``GenerateContentConfig`` fields held in the
    base engine's runtime-parameter store and forwarded verbatim, e.g.
    ``engine.update_runtime_parameters(thinking_config={"thinking_budget": 0})``
    (the SDK coerces a dict to ``ThinkingConfig``). Nothing is set by default,
    so Gemini's own defaults apply. Keys are checked against
    ``GenerateContentConfig`` when set (``_validate_runtime_keys``), so a typo
    fails at ``update_runtime_parameters``.

    API version
    -----------
    The engine never sets the Gemini API version; the SDK default applies
    (``v1beta`` on google-genai 2.x, ``v1`` on 3.x). The Gemini 2.5 family needs
    ``v1beta`` on 3.x (``v1`` rejects system instructions and JSON mode for it);
    newer models work on ``v1``. Pass it explicitly through ``http_options``, which
    behaves the same on 2.x and 3.x (the ``GOOGLE_GENAI_API_VERSION`` variable
    only works on 3.x)::

        GeminiEngine(model="gemini-2.5-flash", api_key=key,
                     http_options={"api_version": "v1beta"})

    Attachments
    -----------
    ``attach(path)`` uploads the file once through ``client.files.upload`` and
    stores its metadata and ``File`` object; the files are appended to the
    last user turn on every call. ``detach(path)`` deletes the upload
    best-effort.

    Result extraction and usage
    ---------------------------
    ``_extract_result`` returns ``response.text`` (``""`` when absent); on a
    structured call it parses it with ``json.loads`` and falls back to the raw
    text on ``JSONDecodeError``. ``_extract_token_usage``
    maps ``usage_metadata`` into ``GeminiTokenUsage``;
    ``generated_tokens = total_token_count - prompt_token_count`` so it covers
    every non-prompt token Gemini counts (candidates, thoughts, tool use).
    With no ``usage_metadata`` it raises unless ``require_usage`` is ``False``.

    Structured-output schema policy
    -------------------------------
    ``output_structure`` is forwarded unmodified to ``response_json_schema``
    (together with ``response_mime_type="application/json"``); Gemini decides
    natively whether to accept, ignore or reject a keyword.
    """

    _RESERVED_RUNTIME_KEYS = frozenset({
        "model", "contents", "system_instruction",
        "response_mime_type", "response_json_schema", "stream",
    })

    def __init__(
            self,
            model: str,
            name: str | None = None,
            description: str | None = None,
            client: genai.Client | None = None,
            **client_kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Gemini model identifier (e.g. ``"gemini-2.5-flash"``, ``"gemini-2.0-pro"``).
        name:
            Optional human-friendly engine name; defaults to the class name.
        description:
            Human-readable description for this engine instance.
        client:
            Pre-built ``genai.Client``. When provided, used directly (combining
            it with ``client_kwargs`` raises). The same client handles both sync
            (``client.models``) and async (``client.aio.models``) paths.
        **client_kwargs:
            Keyword arguments forwarded unchanged to ``genai.Client(...)``.
            Common uses: ``api_key``, ``credentials``, ``project``, ``location``,
            ``vertexai``, ``http_options`` (a dict or ``types.HttpOptions``,
            including its ``timeout`` in milliseconds).

        Lifecycle: SDK presence check; reject ``client`` combined with
        ``client_kwargs``; base init; build or adopt the client.
        """
        # 1. SDK presence check.
        if genai is None:
            raise LLMEngineError(
                "GeminiEngine "
                + format_missing_dependency("google-genai", "gemini", _SDK_IMPORT_ERROR)
            ) from _SDK_IMPORT_ERROR

        # 2. Exactly one client source.
        if client is not None and client_kwargs:
            raise LLMEngineError(
                "GeminiEngine: pass either 'client' or client kwargs, not both."
            )

        # 3. Base init, then engine state.
        super().__init__(name=name, description=description or "Gemini LLM Engine")
        self._attachments: Dict[str, Mapping[str, Any]] = {}
        self._client: genai.Client = (
            client if client is not None else genai.Client(**client_kwargs)
        )
        self.model = model

    def _validate_runtime_keys(self, keys: Iterable[str]) -> None:
        """
        1. Reserved keys are rejected (base check).
        2. Every key must be a ``GenerateContentConfig`` field.
        """
        keys = list(keys)
        super()._validate_runtime_keys(keys)
        valid = set(genai.types.GenerateContentConfig.model_fields)
        unknown = set(keys) - valid
        if unknown:
            raise LLMEngineError(
                f"GeminiEngine.update_runtime_parameters: unknown keys {sorted(unknown)}; "
                f"valid keys: {sorted(valid - self._RESERVED_RUNTIME_KEYS)}"
            )

    # ------------------------------------------------------------------ #
    # Attachments
    # ------------------------------------------------------------------ #
    @property
    def attachments(self) -> Mapping[str, Mapping[str, Any]]:
        """Copy of attached local path -> provider metadata (shallow per entry)."""
        return {path: dict(meta) for path, meta in self._attachments.items()}

    def attach(self, path: str) -> Mapping[str, Any]:
        """
        Attach a local file for every later call; returns a shallow copy of
        its metadata.

        1. Reject a blank or non-string path.
        2. Return the stored metadata when ``path`` is already attached.
        3. Validate the path, prepare the provider metadata, store it.
        """
        if not isinstance(path, str) or not path:
            raise LLMEngineError("GeminiEngine.attach: path must be a non-empty string")
        if path in self._attachments:
            return dict(self._attachments[path])

        self._validate_attachment_path(path)
        meta = self._prepare_attachment(path)
        self._attachments[path] = meta
        return dict(meta)

    def detach(self, path: str) -> bool:
        """
        Detach ``path``; ``True`` when something was removed. The uploaded file
        is deleted best-effort (a failed delete is ignored).
        """
        meta = self._attachments.pop(path, None)
        if meta is None:
            return False
        name = meta.get("resource_name")
        if name:
            try:
                self._client.files.delete(name=name)
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

    def _allows_blank_user_content(self) -> bool:
        """A blank user turn is allowed while attachments exist (they attach to it)."""
        return bool(self._attachments)

    def _validate_attachment_path(self, path: str) -> None:
        """
        Validate ``path`` against the shared illegal-ext set and MIME-prefix rules.

        Gemini uses a blacklist-only policy (no positive allow-list). Delegates
        to ``validate_attachment_path``; converts ``ValueError`` to
        ``LLMEngineError``.
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
        Prepare a local path for Gemini: upload once and store the File object.

        ``attach`` has already validated the path and extension and applied
        MIME-based checks via `_validate_attachment_path`. This method assumes
        the path is supported and focuses on:

        - uploading the file via the Gemini Files API, and
        - returning metadata of the form:

            {
                "kind": "file",
                "mime": <str>,
                "ext": <str>,
                "uploaded": True,
                "file_obj": <File>,
                "resource_name": <str | None>,
            }

        The metadata shape is only interpreted by this class.
        """
        try:
            mime, _ = mimetypes.guess_type(path)
            mime = mime or ""
            ext = os.path.splitext(path)[1].lower()

            file_obj = self._upload_path(path)
            resource_name = getattr(file_obj, "name", None)

            return {
                "kind": "file",
                "mime": mime,
                "ext": ext,
                "uploaded": True,
                "file_obj": file_obj,
                "resource_name": resource_name,
            }
        except LLMEngineError:
            raise
        except Exception as exc:
            # Normalize unexpected errors into LLMEngineError so callers see a
            # consistent engine-level error type.
            raise LLMEngineError(
                f"GeminiEngine._prepare_attachment failed for {path!r}: {exc}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Template hooks for invocation
    # ------------------------------------------------------------------ #

    def _build_content_list(
            self,
            messages: List[Dict[str, str]],
            attachments: Mapping[str, Mapping[str, Any]],
    ) -> List[genai.types.Content]:
        """
        Build a ``list[Content]`` for ``generate_content`` from normalized messages
        and the current attachments snapshot.

        Non-system messages are wrapped in ``Content(role=..., parts=[...])`` with
        ``role`` mapped as ``'user'`` for user turns and ``'model'`` for assistant
        turns. Attachment parts (uploaded files via ``Part.from_uri``) are
        appended to the last user ``Content``'s ``parts`` list. A blank user
        turn (allowed only with attachments) has no text part, so its parts
        are the attachment parts alone.
        """
        # 1. Build attachment_parts list.
        attachment_parts: List[Any] = []
        for _path, meta in attachments.items():
            if meta.get("uploaded") and meta.get("file_obj") is not None:
                file_obj = meta["file_obj"]
                attachment_parts.append(
                    genai.types.Part.from_uri(
                        file_uri=file_obj.uri,
                        mime_type=meta.get("mime") or file_obj.mime_type,
                    )
                )

        # 2. Build contents list with role mapping (no attachment injection yet).
        contents: List[Any] = []
        for m in messages:
            role = m["role"]
            if role == "system":
                continue
            gemini_role = "model" if role == "assistant" else "user"
            text = m["content"]
            contents.append(genai.types.Content(
                role=gemini_role,
                parts=[genai.types.Part.from_text(text=text)] if text else [],
            ))

        # 3. Append attachment_parts to the last user Content.
        if attachment_parts:
            last_user = next(c for c in reversed(contents) if c.role == "user")
            last_user.parts = list(last_user.parts) + attachment_parts

        return contents

    def _build_provider_payload(
            self,
            messages: List[Dict[str, str]],
            output_structure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the exact keyword arguments of ``generate_content``.

        1. ``contents``: non-system turns as ``Content`` objects, with a
           snapshot of the engine's attachments appended to the last user turn.
        2. ``config``: a ``GenerateContentConfig`` from a snapshot of the
           runtime parameters, with the engine-owned fields (``system_instruction``
           and, for a structured call, ``response_mime_type`` plus
           ``response_json_schema``) updated over it so they always win. Mime
           type alone is only a soft hint, so the two are always sent together;
           the schema is the caller's raw one, unmodified. An unknown runtime key
           fails here as the SDK's own validation error.
        3. Return ``{"model", "contents", "config"}``.
        """
        contents = self._build_content_list(messages, dict(self._attachments))

        cfg: Dict[str, Any] = dict(self._runtime_parameters)
        system_instruction = collect_system_text(messages)
        if system_instruction:
            cfg.update(system_instruction=system_instruction)
        if output_structure is not None:
            cfg.update(
                response_mime_type="application/json",
                response_json_schema=output_structure,
            )

        return {
            "model": self.model,
            "contents": contents,
            "config": genai.types.GenerateContentConfig(**cfg),
        }

    def _call_provider(self, payload: Dict[str, Any]) -> Any:
        """Perform a single synchronous ``models.generate_content`` call."""
        return self._client.models.generate_content(**payload)

    async def _call_provider_async(self, payload: Dict[str, Any]) -> Any:
        """
        Native async ``generate_content`` call via ``client.aio.models``.

        No thread offload needed — ``genai.Client`` exposes both sync and async
        paths on the same object.
        """
        return await self._client.aio.models.generate_content(**payload)

    def _extract_result(
        self, response: Any, requested_structured: bool
    ) -> str | int | float | bool | list[Any] | dict[str, Any] | None:
        """
        Return ``response.text`` (``""`` when absent); on a structured call,
        its ``json.loads`` value, falling back to the text on a parse failure.
        """
        text = response.text or ""
        if not requested_structured:
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        """
        Extract Gemini GenerateContent token usage into a GeminiTokenUsage record.

        Gemini usage metadata reports:

        - prompt_token_count: prompt/input-side tokens
        - candidates_token_count: candidate response tokens, when present
        - total_token_count: total request usage
        - thoughts_token_count: generated thinking/reasoning tokens, when present
        - tool_use_prompt_token_count: tool-use prompt tokens, when present
        - cached_content_token_count: cached input-token subset, when present

        ``generated_tokens`` is derived as
        ``total_token_count - prompt_token_count``. Missing ``usage_metadata``
        raises unless ``require_usage`` is ``False``, which yields an all-zero
        record.
        """
        usage = response.usage_metadata
        if usage is None:
            if not self.require_usage:
                return GeminiTokenUsage(
                    input_tokens=0, generated_tokens=0, total_tokens=0, response_tokens=0
                )
            raise LLMEngineError("Gemini response did not include usage_metadata.")

        input_tokens = usage.prompt_token_count
        total_tokens = usage.total_token_count

        if input_tokens is None:
            raise LLMEngineError("Gemini usage_metadata missing prompt_token_count.")
        if total_tokens is None:
            raise LLMEngineError("Gemini usage_metadata missing total_token_count.")

        generated_tokens = total_tokens - input_tokens
        if generated_tokens < 0:
            raise LLMEngineError(
                "Gemini usage_metadata total_token_count is less than "
                "prompt_token_count."
            )

        # Gemini natively reports candidates_token_count as the visible-reply
        # token count; fall back to generated_tokens if it's ever absent,
        # since base response_tokens is a required field.
        response_tokens = (
            usage.candidates_token_count
            if usage.candidates_token_count is not None
            else generated_tokens
        )

        return GeminiTokenUsage(
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            total_tokens=total_tokens,
            response_tokens=response_tokens,
            thoughts_token_count=usage.thoughts_token_count,
            tool_use_prompt_token_count=usage.tool_use_prompt_token_count,
            cached_content_token_count=usage.cached_content_token_count,
        )

    def _get_model_data(self) -> LLMModelData:
        """Return configured Gemini model identity (from ``self.model``)."""
        return RemoteLLMModelData(
            provider="gemini",
            model_name=self.model,
        )

    # ------------------------------------------------------------------ #
    # Gemini-specific helpers (not part of the template surface)
    # ------------------------------------------------------------------ #

    def _upload_path(self, path: str) -> Any:
        """
        Upload a local path via the Gemini Files API and return the File object.

        The Gen AI SDK supports passing File objects directly in `contents`.
        """
        abs_path = os.path.abspath(path)
        return self._client.files.upload(file=abs_path)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """
        Diagnostic snapshot for GeminiEngine: provider + model (the base adds the
        runtime parameters).

        Keeps output minimal to avoid leaking client or API keys.
        """
        base = super().to_dict()
        base.update({
            "model": self.model,
        })
        return base
