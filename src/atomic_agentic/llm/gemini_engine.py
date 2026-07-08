from __future__ import annotations

import mimetypes
try:
    from google import genai
except ImportError:
    genai = None
import os
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)

from .base import LLMEngine
from ..constants.engines import (
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
from ..utils.llm import validate_attachment_path

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

    ``client=None`` builds ``genai.Client(**client_kwargs)`` with
    ``http_options.timeout`` seeded from ``timeout_seconds`` (in milliseconds,
    as required by the SDK). An injected client is used as-is.

    Flow per call
    -------------
    1) Engine-level attachments are prepared via ``attach(path)``:
       - ``_prepare_attachment`` uploads supported files via
         ``client.files.upload``.
       - Attachment metadata stores the returned File object and its resource
         name.

    2) ``invoke({"messages": messages})`` runs the shared ``LLMResult``
       lifecycle:
       - normalize chat messages;
       - snapshot current attachments;
       - build the Gemini provider payload;
       - call ``client.models.generate_content(...)``;
       - extract assistant text, token usage, and configured model data;
       - return ``LLMResult``.

    3) ``detach(path)`` calls ``_on_detach`` for best-effort file deletion via
       ``client.files.delete``.

    Token usage
    -----------
    ``_extract_token_usage`` maps ``response.usage_metadata`` into
    ``GeminiTokenUsage``. The base ``generated_tokens`` value is derived as
    ``total_token_count - prompt_token_count`` so it captures all non-prompt
    usage counted by Gemini, including candidate, thoughts, and tool-use prompt
    tokens when present.

    Model data
    ----------
    ``_get_model_data`` returns configured model identity from ``self.model``.
    """

    def __init__(
            self,
            model: str,
            name: str | None = None,
            namespace: str = "llm",
            description: str = "Gemini LLM Engine",
            client: genai.Client | None = None,
            temperature: float | None = 0.1,
            max_output_tokens: int | None = None,
            thinking_config: dict[str, Any] | None = None,
            *,
            filter_extraneous_inputs: bool = True,
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
            Gemini model identifier (e.g. ``"gemini-2.5-flash"``, ``"gemini-2.0-pro"``).
        name:
            Optional human-friendly engine name; defaults to ``gemini_{model}`` with
            non-identifier characters replaced by underscores.
        namespace:
            Grouping label; inherited by the base engine as ``"llm"`` by default.
        description:
            Human-readable description for this engine instance.
        client:
            Pre-built ``genai.Client``. When provided, used directly; no client is
            constructed from ``client_kwargs``. The same client handles both sync
            (``client.models``) and async (``client.aio.models``) paths.
        temperature:
            Sampling temperature passed to ``GenerateContentConfig``. ``None`` omits
            the parameter entirely.
        max_output_tokens:
            Optional maximum number of output tokens. Maps to
            ``GenerateContentConfig.max_output_tokens``.
        thinking_config:
            Optional thinking configuration dict (e.g.
            ``{"thinking_budget": 2000}`` or ``{"thinking_level": "high"}``).
            Expanded into ``genai.types.ThinkingConfig(**thinking_config)`` at call
            time. ``None`` omits the field.
        filter_extraneous_inputs, timeout_seconds, max_retries, retry_backoff_base, retry_backoff_max:
            Shared ``LLMEngine`` configuration (see base class).
        **client_kwargs:
            Additional keyword arguments forwarded verbatim to ``genai.Client(...)``
            during client construction. Common uses: ``api_key``, ``credentials``,
            ``project``, ``location``, ``vertexai``, ``http_options``. A default
            ``http_options={"timeout": <ms>}`` is seeded from ``timeout_seconds``
            unless ``http_options`` is supplied explicitly. Not forwarded when
            ``client`` is injected.
        """
        # Step 1 — Name sanitization and base init.
        sanitized_name = (name or f"gemini_{model}").replace(":", "_").replace("-", "_").replace(" ", "_").replace(".", "_")
        super().__init__(
            name=sanitized_name,
            namespace=namespace,
            description=description or "Gemini LLM Engine",
            filter_extraneous_inputs=filter_extraneous_inputs,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        # Step 2 — SDK presence check.
        if genai is None:
            raise LLMEngineError(
                "GeminiEngine requires the `google-genai` package; "
                "install `google-genai` to use it."
            )

        # Step 3 — Build client kwargs; seed http_options.timeout (milliseconds).
        _ckw = dict(client_kwargs)
        _ckw.setdefault("http_options", {"timeout": int(self._timeout_seconds * 1000)})

        # Step 4 — Single client: injected as-is or built from kwargs.
        self._client: genai.Client = client if client is not None else genai.Client(**_ckw)

        # Step 5 — Store model and generation config.
        self.model = model
        self.temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._thinking_config = thinking_config

    # ------------------------------------------------------------------ #
    # Attachment validation & preparation
    # ------------------------------------------------------------------ #

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

        The base engine has already validated the path and extension and applied
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

        The metadata shape is opaque to the base engine and only interpreted
        by this class.
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
                f"GeminiEngine._prepare_attachment failed for {path!r}"
            ) from exc

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        """
        Delete Gemini file resource if present.

        This is best-effort; errors are ignored.
        """
        name = meta.get("resource_name")
        if not name:
            return
        try:
            self._client.files.delete(name=name)
        except Exception:
            return

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
        turns. Attachment parts (uploaded files via ``Part.from_uri``; inlined text
        via ``Part.from_text``) are prepended to the first user ``Content``'s
        ``parts`` list. If no user turn is present, a standalone leading
        ``Content(role='user', parts=attachment_parts)`` is inserted.
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
            elif meta.get("inlined") and meta.get("inlined_text"):
                attachment_parts.append(
                    genai.types.Part.from_text(text=str(meta["inlined_text"]))
                )

        # 2. Build contents list with role mapping; inject attachments into first user turn.
        contents: List[Any] = []
        attachment_injected = False
        for m in messages:
            role = m.get("role", "").lower()
            if role == "system":
                continue
            gemini_role = "model" if role == "assistant" else "user"
            text = m.get("content") or ""
            if not text:
                continue
            parts: List[Any] = []
            if gemini_role == "user" and attachment_parts and not attachment_injected:
                parts.extend(attachment_parts)
                attachment_injected = True
            parts.append(genai.types.Part.from_text(text=text))
            contents.append(genai.types.Content(role=gemini_role, parts=parts))

        # 3. If no user turn was present, insert a standalone leading Content for attachments.
        if attachment_parts and not attachment_injected:
            contents.insert(0, genai.types.Content(role="user", parts=attachment_parts))

        return contents

    def _build_provider_payload(
            self,
            messages: List[Dict[str, str]],
            attachments: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build the payload for ``generate_content`` from normalized messages and the
        current attachments snapshot.

        System messages are extracted into ``system_instruction`` for
        ``GenerateContentConfig``; all other turns are returned as
        ``list[Content]`` via ``_build_content_list``.
        """
        system_instruction = self._collect_system(messages)
        contents = self._build_content_list(messages, attachments)
        return {
            "system_instruction": system_instruction,
            "contents": contents,
        }

    def _build_generate_config(
            self,
            system_instruction: str | None,
    ) -> genai.types.GenerateContentConfig:
        """
        Build a ``GenerateContentConfig`` from stored engine params and the per-call
        ``system_instruction``. Fields are omitted when their stored value is ``None``.
        """
        cfg: Dict[str, Any] = {}
        if system_instruction:
            cfg["system_instruction"] = system_instruction
        if self.temperature is not None:
            cfg["temperature"] = self.temperature
        if self._max_output_tokens is not None:
            cfg["max_output_tokens"] = self._max_output_tokens
        if self._thinking_config is not None:
            cfg["thinking_config"] = genai.types.ThinkingConfig(**self._thinking_config)
        return genai.types.GenerateContentConfig(**cfg)

    def _call_provider(self, payload: Dict[str, Any]) -> Any:
        """Perform a single synchronous ``models.generate_content`` call."""
        cfg = self._build_generate_config(payload.get("system_instruction"))
        return self._client.models.generate_content(
            model=self.model,
            contents=payload["contents"],
            config=cfg,
        )

    async def _call_provider_async(self, payload: Dict[str, Any]) -> Any:
        """
        Native async ``generate_content`` call via ``client.aio.models``.

        No thread offload needed — ``genai.Client`` exposes both sync and async
        paths on the same object.
        """
        cfg = self._build_generate_config(payload.get("system_instruction"))
        return await self._client.aio.models.generate_content(
            model=self.model,
            contents=payload["contents"],
            config=cfg,
        )

    def _extract_text(self, response: Any) -> str:
        """
        Extract the assistant's textual reply from a Gen AI SDK response object.

        The SDK exposes a `.text` convenience property for text responses.
        """
        return response.text

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
        ``total_token_count - prompt_token_count``.
        """
        usage = response.usage_metadata
        if usage is None:
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

        return GeminiTokenUsage(
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            total_tokens=total_tokens,
            candidates_token_count=usage.candidates_token_count,
            thoughts_token_count=usage.thoughts_token_count,
            tool_use_prompt_token_count=usage.tool_use_prompt_token_count,
            cached_content_token_count=usage.cached_content_token_count,
        )

    def _get_model_data(self) -> LLMModelData:
        """
        Return configured Gemini model identity data for this engine.

        Model data is derived from engine configuration, not from the provider
        response object.
        """
        return RemoteLLMModelData(
            provider="gemini",
            model_name=self.model,
        )

    # ------------------------------------------------------------------ #
    # Gemini-specific helpers (not part of the template surface)
    # ------------------------------------------------------------------ #

    def _collect_system(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Join system message contents into a single `system_instruction` string.
        """
        parts = [
            m["content"]
            for m in messages
            if (m.get("role") or "").lower() == "system" and m.get("content")
        ]
        joined = "\n\n".join(parts).strip()
        return joined or None

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
        Diagnostic snapshot for GeminiEngine: provider + model + generation config.

        Keeps output minimal to avoid leaking client or API keys.
        """
        base = super().to_dict()
        base.update({
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self._max_output_tokens,
            "thinking_config": self._thinking_config,
        })
        return base
