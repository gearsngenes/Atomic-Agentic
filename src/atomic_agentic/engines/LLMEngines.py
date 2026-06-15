from __future__ import annotations

# ~~~Standard Library Imports~~~
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import logging
import mimetypes
import os
import random
import time
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Union,
)
import warnings

# ~~~Provider SDK Imports~~~
# OpenAI
try: from openai import OpenAI
except: OpenAI = None
# Google
try: from google import genai
except: genai = None
# Mistral
try: from mistralai import Mistral
except: Mistral = None
# Llama-CPP-Python
try: from llama_cpp import Llama
except: Llama = None
# Hugging Face Hub
try: from huggingface_hub import hf_hub_download
except: hf_hub_download = None

# ~~~Local Imports~~~
from ..core.Invokable import AtomicInvokable
from ..core.constants import NO_VAL
from ..core.Parameters import ParamSpec
from ..core.Exceptions import LLMEngineError
from ..models.results import (
    GeminiTokenUsage,
    LlamaCppModelData,
    LlamaCppTokenUsage,
    LLMModelData,
    LLMResult,
    MistralTokenUsage,
    OpenAITokenUsage,
    RemoteLLMModelData,
    TokenUsage,
)

__all__ = [
    "GeminiEngine",
    "LlamaCppEngine",
    "LLMEngine",
    "MistralEngine",
    "OpenAIEngine",
    ]

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────
# LLMEngine primitive
# ───────────────────────────────────────────────────────────────────────────────
class LLMEngine(AtomicInvokable, ABC):
    """
    Base template-method primitive for LLM provider adapters.

    Engines are stateless with respect to conversation history: the Agent owns
    message history. An engine instance represents a particular provider +
    model configuration plus a persistent set of attachments.

    Public contract
    ---------------
    ``LLMEngine`` is an ``AtomicInvokable``, so its canonical public invocation
    entrypoint is dict-first:

        invoke({"messages": list[{"role": str, "content": str}]}) -> LLMResult

    ``LLMResult.result`` is the generated assistant text string. The declared
    invokable ``return_type`` remains ``"str"`` because ``return_type`` describes
    the caller-facing payload stored inside ``AtomicResult.result``, not the
    result envelope class.

    The declared invokable schema exposes one input parameter named
    ``messages``. The value must be a non-empty list of chat-message mappings
    containing string ``role`` and ``content`` fields.

    Deprecated compatibility
    ------------------------
    ``invoke_messages(messages) -> str`` is retained as a deprecated text-only
    compatibility wrapper during the v2 migration. Prefer:

        invoke({"messages": messages}).result

    Engine lifecycle
    ----------------
    The canonical result-envelope lifecycle lives in ``invoke(inputs)``:

    1. Filter dict-first inputs.
    2. Validate and normalize ``messages``.
    3. Snapshot current attachments.
    4. Ask the subclass to build a provider-specific payload.
    5. Call the provider with retries/timeouts.
    6. Extract assistant text, token usage, and configured model data.
    7. Construct and return an ``LLMResult``.

    Provider-specific behavior should normally be implemented through the
    protected template hooks:

    - ``_build_provider_payload``
    - ``_call_provider``
    - ``_extract_text``
    - ``_extract_token_usage``
    - ``_get_model_data``
    - ``_prepare_attachment``
    - ``_on_detach``

    Attachments are managed separately via ``attach`` / ``detach`` /
    ``clear_attachments`` and are snapshotted for each call.
    """

    # Attachment policy defaults
    # --------------------------
    # Subclasses are expected to override `allowed_attachment_exts` with the set
    # of extensions their provider can meaningfully consume (e.g. {".pdf", ".png"}).
    # `illegal_attachment_exts` is a coarse security/robustness guard applied
    # before provider-specific checks.
    illegal_attachment_exts: set[str] = {
        ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",
        ".exe", ".dll", ".so", ".bin", ".o",
        ".db", ".sqlite",
        ".h5", ".pt", ".pth", ".onnx",
    }
    allowed_attachment_exts: Optional[set[str]] = None

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        description: str = "",
        filter_extraneous_inputs: bool = True,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 8.0,
    ) -> None:
        """
        Parameters
        ----------
        name:
            Optional human-friendly identifier for logging/introspection.
        timeout_seconds:
            Suggested per-call timeout; subclasses should honor this where
            their provider SDKs allow it.
        max_retries:
            Maximum number of *retries* after the initial call (so total
            attempts is `max_retries + 1`).
        retry_backoff_base:
            Base seconds for exponential backoff (approx base * 2^(attempt-1)).
        retry_backoff_max:
            Upper bound in seconds for backoff delay.
        """
        # Pass the provided name (or the class name) directly to AtomicInvokable.
        AtomicInvokable.__init__(
            self,
            name=name or type(self).__name__,
            description=description or "LLM Engine",
            parameters=[ParamSpec(name ="messages",
                                  index = 0,
                                  kind = "POSITIONAL_OR_KEYWORD",
                                  type="List[Dict[str, str]]",
                                  default = NO_VAL)],
            return_type="str",
            filter_extraneous_inputs=filter_extraneous_inputs,
        )

        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._retry_backoff_base = float(retry_backoff_base)
        self._retry_backoff_max = float(retry_backoff_max)
        # Persistent mapping: local path -> provider-specific metadata
        self._attachments: Dict[str, Mapping[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def attachments(self) -> Mapping[str, Mapping[str, Any]]:
        """
        Read-only view of currently attached paths.

        Keys are local paths; values are provider-specific metadata dicts as
        returned by `_prepare_attachment`.
        """
        # Return a shallow copy to discourage mutation of internal state.
        return dict(self._attachments)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def attach(self, path: str) -> Mapping[str, Any]:
        """
        Attach a local file path to this engine.

        The base implementation:
        - validates the path and extension using `_validate_attachment_path`;
        - delegates to `_prepare_attachment` to build provider-specific metadata;
        - stores and returns that metadata.

        The *shape* of the metadata mapping is entirely determined by the subclass.
        """
        if not isinstance(path, str) or not path:
            raise LLMEngineError("LLMEngine.attach: path must be a non-empty string")
        if path in self._attachments:
            return self._attachments[path]

        self._validate_attachment_path(path)
        meta = self._prepare_attachment(path)
        if not isinstance(meta, Mapping):
            raise LLMEngineError(
                f"{type(self).__name__}._prepare_attachment must return a mapping; "
                f"got {type(meta)!r}"
            )

        self._attachments[path] = meta
        logger.debug("LLMEngine %s attached %s", self.name, path)
        return meta

    def detach(self, path: str) -> bool:
        """
        Detach a previously attached path.

        Calls `_on_detach` with the stored metadata for provider-specific cleanup.
        Returns True if an attachment was removed, False if the path was not
        attached.
        """
        meta = self._attachments.pop(path, None)
        if meta is None:
            return False
        try:
            self._on_detach(meta)
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.debug(
                "LLMEngine %s._on_detach raised %r for %s; ignoring",
                self.name,
                exc,
                path,
            )
        logger.debug("LLMEngine %s detached %s", self.name, path)
        return True

    def clear_attachments(self) -> None:
        """Detach all currently attached paths."""
        for path in list(self._attachments.keys()):
            self.detach(path)

    def invoke_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Deprecated text-only compatibility wrapper.

        Prefer ``invoke({"messages": messages}).result`` so callers use the
        canonical ``LLMResult`` envelope path.
        """
        warnings.warn(
            (
                "LLMEngine.invoke_messages(...) is deprecated and will be removed "
                "in a future v2 release. Use "
                "LLMEngine.invoke({'messages': messages}).result instead."
            ),
            DeprecationWarning,
            stacklevel=2,
        )

        try:
            response = self._call_model(messages)
            text = self._extract_text(response)

            if not isinstance(text, str):
                raise LLMEngineError(
                    f"{type(self).__name__}._extract_text must return str; "
                    f"got {type(text)!r}"
                )

            return text.strip()
        except LLMEngineError:
            raise
        except Exception as exc:
            raise LLMEngineError(f"{self.name}.invoke_messages failed") from exc

    def invoke(self, inputs: Mapping[str, Any]) -> LLMResult:
        """
        Invoke this engine through the canonical v2 result-envelope path.

        Returns
        -------
        LLMResult
            Result envelope whose ``.result`` field contains the generated
            assistant text string.
        """
        with self._invoke_lock:
            logger.info("[%s started]", self.full_name)
            started_at = datetime.now(timezone.utc)

            try:
                filtered_inputs = self.filter_inputs(inputs)
                messages = filtered_inputs.get("messages")
                if not isinstance(messages, list):
                    raise LLMEngineError(
                        "LLMEngine.invoke: 'messages' input must be a list"
                    )

                response = self._call_model(messages)
                text, token_usage, model_data = self.extract(response)
                ended_at = datetime.now(timezone.utc)

                result = self.make_result(
                    result=text,
                    started_at=started_at,
                    ended_at=ended_at,
                    token_usage=token_usage,
                    model_data=model_data,
                )

                logger.info("[%s finished]", self.full_name)
                return result

            except LLMEngineError:
                raise
            except Exception as exc:
                raise LLMEngineError(f"{self.name}.invoke failed") from exc

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _call_model(self, messages: List[Dict[str, Any]]) -> Any:
        """
        Normalize messages, snapshot current attachments, build the provider
        payload, and return the raw provider response.

        This helper owns only the shared provider-call sequence. It does not
        extract text, extract token usage, construct results, capture timing, or
        emit deprecation warnings.
        """
        normalized = self._normalize_messages(messages)
        attachments = dict(self._attachments)
        payload = self._build_provider_payload(normalized, attachments)
        return self._call_with_retries(payload)

    def extract(self, response: Any) -> tuple[str, TokenUsage, LLMModelData]:
        """
        Extract normalized generated text, token usage, and configured model
        data from one provider response.

        This method coordinates result-path extraction only. It does not call
        the provider, capture timestamps, or construct ``LLMResult``.
        """
        text = self._extract_text(response)
        if not isinstance(text, str):
            raise LLMEngineError(
                f"{type(self).__name__}._extract_text must return str; "
                f"got {type(text)!r}"
            )

        token_usage = self._extract_token_usage(response)
        if not isinstance(token_usage, TokenUsage):
            raise LLMEngineError(
                f"{type(self).__name__}._extract_token_usage must return "
                f"TokenUsage, got {type(token_usage)!r}."
            )

        model_data = self._get_model_data()
        if not isinstance(model_data, LLMModelData):
            raise LLMEngineError(
                f"{type(self).__name__}._get_model_data must return "
                f"LLMModelData, got {type(model_data)!r}."
            )

        return text.strip(), token_usage, model_data

    def make_result(
        self,
        result: str,
        started_at: datetime,
        ended_at: datetime,
        **result_kwargs: Any,
    ) -> LLMResult:
        """
        Construct this engine's ``LLMResult`` envelope.

        ``result`` is the caller-facing generated text payload stored in
        ``LLMResult.result``. Token usage and configured model data are stored as
        explicit LLM-specific result fields.
        """
        unexpected = set(result_kwargs) - {"token_usage", "model_data"}
        if unexpected:
            raise LLMEngineError(
                f"make_result: unexpected result kwarg(s): {sorted(unexpected)!r}."
            )

        token_usage = result_kwargs.get("token_usage")
        model_data = result_kwargs.get("model_data")

        if not isinstance(token_usage, TokenUsage):
            raise LLMEngineError(
                "make_result: token_usage must be a TokenUsage instance."
            )

        if not isinstance(model_data, LLMModelData):
            raise LLMEngineError(
                "make_result: model_data must be an LLMModelData instance."
            )

        return self._make_result(
            result=result,
            started_at=started_at,
            ended_at=ended_at,
            result_cls=LLMResult,
            token_usage=token_usage,
            model_data=model_data,
        )
    def _normalize_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Validate and normalize a sequence of chat messages.

        - Ensures `messages` is a list of mappings.
        - Ensures each entry has string `role` and `content` keys.
        - Normalizes `role` to lowercase.
        """
        if not isinstance(messages, list):
            raise LLMEngineError("LLMEngine.invoke: messages must be a list")
        if not messages:
            raise LLMEngineError("LLMEngine.invoke: messages must not be empty")

        normalized: List[Dict[str, str]] = []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, Mapping):
                raise LLMEngineError(
                    f"LLMEngine.invoke: message {idx} is not a mapping (got {type(msg)!r})"
                )
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise LLMEngineError(
                    "LLMEngine.invoke: each message must have 'role' and 'content' as strings"
                )
            normalized.append({"role": role.lower(), "content": content})
        return normalized

    def _validate_attachment_path(self, path: str) -> None:
        """
        Generic path/extension validation used by `attach`.

        Subclasses are expected to further restrict supported types by setting
        `allowed_attachment_exts` and/or overriding this method.
        """
        if not os.path.isfile(path):
            raise LLMEngineError(
                f"LLMEngine.attach: path does not exist or is not a file: {path!r}"
            )

        _, ext = os.path.splitext(path)
        ext = ext.lower()

        if ext and ext in self.illegal_attachment_exts:
            raise LLMEngineError(
                f"LLMEngine.attach: extension {ext!r} is not allowed for safety/robustness"
            )

        allowed_exts = self.allowed_attachment_exts
        if allowed_exts is not None and ext not in allowed_exts:
            raise LLMEngineError(
                f"LLMEngine.attach: extension {ext!r} is not supported by {self.name}"
            )

    def _call_with_retries(self, payload: Any) -> Any:
        """
        Call `_call_provider` with a basic retry/backoff loop.

        Subclasses can customize retry behavior by overriding `_should_retry`
        or by setting `max_retries`/backoff parameters in the constructor.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._call_provider(payload)
            except LLMEngineError:
                # Already normalized; do not re-wrap or retry.
                raise
            except Exception as exc:
                if not self._should_retry(exc, attempt):
                    raise
                sleep = min(
                    self._retry_backoff_base * (2 ** (attempt - 1)),
                    self._retry_backoff_max,
                )
                # Add a little jitter to avoid thundering herds.
                sleep *= random.uniform(0.8, 1.2)
                logger.debug(
                    "LLMEngine %s attempt %d failed with %r; retrying in %.2fs",
                    self.name,
                    attempt,
                    exc,
                    sleep,
                )
                time.sleep(sleep)

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        """
        Decide whether a failed `_call_provider` should be retried.

        Default policy:
        - Do not exceed `self._max_retries`.
        - Retry on basic timeout/connection-style errors.
        Subclasses may override this to recognize provider-specific error types.
        """
        if attempt > self._max_retries:
            return False

        # Simple baseline: retry on common transient conditions. We avoid importing
        # provider SDK exceptions here; subclasses can override for finer control.
        return isinstance(exc, (TimeoutError, ConnectionError))

    # --------------------------------------------------------------------- #
    # Abstract Helpers
    # --------------------------------------------------------------------- #
    @abstractmethod
    def _build_provider_payload(self, messages: List[Dict[str, str]], attachments: Mapping[str, Mapping[str, Any]]) -> Any:
        """
        Convert normalized messages and attachments into the provider-specific
        request payload.
        """
        raise NotImplementedError

    @abstractmethod
    def _call_provider(self, payload: Any) -> Any:
        """
        Perform a single call to the underlying provider using the given payload.

        This method should honor `self._timeout_seconds` where possible.
        """
        raise NotImplementedError

    @abstractmethod
    def _extract_text(self, response: Any) -> str:
        """
        Extract the assistant's textual reply from a provider response object.
        """
        raise NotImplementedError

    @abstractmethod
    def _extract_token_usage(self, response: Any) -> TokenUsage:
        """
        Extract normalized token usage from a provider response object.

        Implementations must return a ``TokenUsage``-family record, not a raw
        provider usage dictionary or SDK object.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_model_data(self) -> LLMModelData:
        """
        Return configured model identity data for this engine.

        Model data is derived from engine configuration, not from a provider
        response object.
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        """
        Prepare a local path for reuse with this engine.

        Implementations typically:
        - validate the path and extension vs provider capabilities,
        - perform any remote upload or inlining,
        - return an opaque metadata mapping used later by `_build_provider_payload`.
        """
        raise NotImplementedError
    
    @abstractmethod
    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        """
        Optional hook called when an attachment is detached.

        Subclasses implement provider-specific cleanup (e.g. remote delete).
        """
        # Intentionally a no-op by default.
        raise NotImplementedError


    # --------------------------------------------------------------------- #
    # Serialization
    # --------------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """
        Shallow, non-secret configuration snapshot for debugging / logging.

        Invocation result data such as token usage and model data belongs to
        ``LLMResult``, not to engine metadata serialization.
        """
        d = super().to_dict()
        d.update({
            "type": type(self).__name__,
            "timeout_seconds": self._timeout_seconds,
            "max_retries": self._max_retries,
            "retry_backoff_base": self._retry_backoff_base,
            "retry_backoff_max": self._retry_backoff_max,
            "attachments": self._attachments,
        })
        return d



# ── OPENAI (Responses API) ─────────────────────────────────────────────────────
class OpenAIEngine(LLMEngine):
    """
    OpenAI adapter using the Responses API.

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
    ``_extract_text`` reads the generated assistant text from the Responses API
    response. ``_extract_token_usage`` maps Responses API usage fields into an
    ``OpenAITokenUsage`` record. ``_get_model_data`` returns configured model
    identity from ``self.model``.
    """

    # Image extensions that map to `input_image`
    _IMAGE_EXTS: tuple[str, ...] = (
        ".png", ".jpg", ".jpeg",
        ".webp", ".gif", ".bmp",
        ".tif", ".tiff", ".heic",
    )

    # Text/code-ish extensions we are willing to inline as text.
    _TEXT_EXTS: tuple[str, ...] = (
        ".txt", ".md", ".rst", ".log",
        ".json", ".jsonl", ".yaml", ".yml",
        ".csv", ".tsv", ".py", ".ipynb",
        ".js", ".ts", ".jsx", ".tsx",
        ".java", ".c", ".cpp", ".h",
        ".hpp", ".rs", ".go", ".rb",
        ".php", ".cs", ".html", ".htm",
        ".xml",
    )

    # Extra illegal extensions for this provider (merged with base `illegal_attachment_exts`)
    _ILLEGAL_EXTS: set[str] = {
        ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z", # archives
        ".exe", ".dll", ".so", ".bin", ".o",          # executables/binaries
        ".db", ".sqlite",                             # databases
        ".h5", ".pt", ".pth", ".onnx",                # model weights
    }

    # MIME prefixes we never accept even if extension would otherwise pass.
    _ILLEGAL_MIME_PREFIXES: tuple[str, ...] = ("audio/", "video/")

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        inline_cutoff_chars: int = 200_000,
        extra_illegal_exts: Optional[set[str]] = None,
        *,
        name: Optional[str] = None,
        description: str = "OpenAI LLM Engine",
        filter_extraneous_inputs: bool = True,
        timeout_seconds: float = 600.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 8.0,
    ) -> None:
        """
        Parameters
        ----------
        model:
            OpenAI model identifier (e.g. "gpt-4.1", "gpt-4o-mini").
        api_key:
            Optional API key; if omitted, `OPENAI_API_KEY` from the environment is used.
        temperature:
            Sampling temperature (ignored for certain reasoning models if not applicable).
        inline_cutoff_chars:
            Maximum number of characters to inline from text/code attachments.
        extra_illegal_exts:
            Optional set of additional extensions to reject at `attach` time.
        name, description, filter_extraneous_inputs, timeout_seconds, max_retries, retry_backoff_base, retry_backoff_max:
            Template-method engine configuration (see `_primitives.LLMEngine`).
        """
        sanitized_name = (name or f"openai_{model}").replace(":", "_").replace("-", "_").replace(" ", "_").replace(".", "_").replace(".", "_")
        super().__init__(
            name=sanitized_name,
            description=description or "OpenAI LLM Engine",
            filter_extraneous_inputs=filter_extraneous_inputs,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        if OpenAI is None:
            raise RuntimeError(
                "OpenAIEngine requires the `openai` package; install `openai` to use it."
            )

        # Honor the base engine's timeout knob when constructing the OpenAI client.
        # The official SDK exposes a `timeout` option for this.
        self.llm = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            timeout=self._timeout_seconds,
        )

        self.model = model
        self.temperature = float(temperature)
        self.inline_cutoff_chars = int(inline_cutoff_chars)

        # Merge illegal extension policy with base defaults + any user-supplied extras.
        merged_illegal = set(self.illegal_attachment_exts) | set(self._ILLEGAL_EXTS)
        if extra_illegal_exts:
            merged_illegal |= set(extra_illegal_exts)
        self.illegal_attachment_exts = merged_illegal

        # Positive allow-list: PDFs + known image + text/code extensions.
        allowed = set(self._TEXT_EXTS) | set(self._IMAGE_EXTS) | {".pdf"}
        self.allowed_attachment_exts = allowed

    # ------------------------------------------------------------------ #
    # Overrides / template hooks
    # ------------------------------------------------------------------ #

    def _validate_attachment_path(self, path: str) -> None:
        """
        Extend the base validation with MIME-type checks (reject audio/video).
        """
        super()._validate_attachment_path(path)

        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        if any(mime.startswith(pref) for pref in self._ILLEGAL_MIME_PREFIXES):
            raise LLMEngineError(
                f"OpenAIEngine.attach: MIME type {mime!r} is not supported"
            )

    def _build_provider_payload(
        self,
        messages: List[Dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build the payload for the OpenAI Responses API from normalized messages
        and the current attachments snapshot.
        """
        instructions = self._collect_instructions(messages)
        blocks = self._build_role_blocks(messages)

        payload: Dict[str, Any] = {"blocks": blocks}
        if instructions:
            payload["instructions"] = instructions

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

    def _call_provider(self, payload: Dict[str, Any]) -> Any:
        """
        Perform a single Responses API call using the pre-built payload.

        Retries + backoff are handled by the base `_call_with_retries` wrapper.
        """
        blocks = payload["blocks"]
        instructions = payload.get("instructions")

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": blocks,
        }
        if instructions:
            kwargs["instructions"] = instructions

        # For non-reasoning models, respect temperature; for some `gpt-5`-class
        # models this may be ignored or overridden by the provider.
        if "gpt-5" not in self.model.lower():
            kwargs["temperature"] = self.temperature

        return self.llm.responses.create(**kwargs)

    def _extract_text(self, response: Any) -> str:
        """
        Extract the assistant's textual reply from a Responses API response object.

        Returns the empty string when no text is present (does not raise).
        """
        return response.output_text

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        """
        Extract OpenAI Responses API token usage into an OpenAITokenUsage record.

        OpenAI Responses usage reports:

        - input_tokens: prompt/input-side tokens
        - output_tokens: generated-side tokens
        - total_tokens: total input + generated tokens
        - input_tokens_details.cached_tokens: cached input-token subset
        - output_tokens_details.reasoning_tokens: hidden reasoning-token subset
          counted inside output_tokens

        ``response_tokens`` is derived as ``output_tokens - reasoning_tokens``.
        """
        usage = response.usage
        if usage is None:
            raise LLMEngineError("OpenAI response did not include usage.")

        reasoning_tokens = usage.output_tokens_details.reasoning_tokens
        response_tokens = usage.output_tokens - reasoning_tokens
        if response_tokens < 0:
            raise LLMEngineError(
                "OpenAI response usage produced a negative response token count "
                "after subtracting reasoning_tokens from output_tokens."
            )

        return OpenAITokenUsage(
            input_tokens=usage.input_tokens,
            generated_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            response_tokens=response_tokens,
            cached_tokens=usage.input_tokens_details.cached_tokens,
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
            if len(text) > self.inline_cutoff_chars:
                text = text[: self.inline_cutoff_chars] + "\n…[truncated]\n"

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
            self.llm.files.delete(file_id)
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
        if ext in self._IMAGE_EXTS or mime.startswith("image/"):
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
            f = self.llm.files.create(file=fp, purpose="assistants")
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
        is_image = mime.startswith("image/") or lower.endswith(self._IMAGE_EXTS)

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
        if len(text) > self.inline_cutoff_chars:
            text = text[: self.inline_cutoff_chars] + "\n…[truncated]\n"

        header = f"\n[Inlined file: {os.path.basename(path)}]\n"
        user_parts.append({"type": "input_text", "text": header + text})

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """
        Diagnostic snapshot for OpenAIEngine, without secrets.

        Includes model, temperature, and inline cutoff in addition to base engine info.
        """
        base = super().to_dict()
        base.update({
            "model" : self.model,
            "temperature" : self.temperature,
            "inline_cutoff_chars": self.inline_cutoff_chars
        })
        return base

# ── GEMINI (flat contents: file objects + strings) ─────────────────────────────
class GeminiEngine(LLMEngine):
    """
    Google Gemini adapter using the Google Gen AI SDK.

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

    # Extra illegal extensions for this provider (merged with base `illegal_attachment_exts`)
    _ILLEGAL_EXTS: set[str] = {
        ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",  # archives
        ".exe", ".dll", ".so", ".bin", ".o",  # executables/binaries
        ".db", ".sqlite",  # databases
        ".h5", ".pt", ".pth", ".onnx",  # model weights
    }

    # MIME prefixes we never accept even if extension would otherwise pass.
    _ILLEGAL_MIME_PREFIXES: tuple[str, ...] = ("audio/", "video/")

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        extra_illegal_exts: Optional[set[str]] = None,
        *,
        name: Optional[str] = None,
        description: str = "Gemini LLM Engine",
        filter_extraneous_inputs: bool = True,
        timeout_seconds: float = 600.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 8.0,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Gemini model identifier (e.g. "gemini-2.5-flash", "gemini-2.0-pro").
        api_key:
            Optional API key. If omitted, the client uses the GOOGLE_API_KEY
            environment variable.
        temperature:
            Sampling temperature for text generation.
        extra_illegal_exts:
            Optional set of additional extensions to reject at `attach` time.
        name, description, filter_extraneous_inputs, timeout_seconds, max_retries, retry_backoff_base, retry_backoff_max:
            Template-method engine configuration (see `_primitives.LLMEngine`).
        """
        sanitized_name = (name or f"gemini_{model}").replace(":", "_").replace("-", "_").replace(" ", "_").replace(".", "_")
        super().__init__(
            name=sanitized_name,
            description=description or "Gemini LLM Engine",
            filter_extraneous_inputs=filter_extraneous_inputs,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        if genai is None:
            raise RuntimeError(
                "GeminiEngine requires the `google-genai` package; "
                "install `google-genai` to use it."
            )

        # Honor the base engine's timeout knob via http_options.
        # The SDK accepts a plain dict for HTTP options.
        http_options: Dict[str, Any] = {
            "timeout": int(self._timeout_seconds * 1000),  # ms
        }

        client_kwargs: Dict[str, Any] = {"http_options": http_options}
        if api_key is not None:
            client_kwargs["api_key"] = api_key

        self.client = genai.Client(**client_kwargs)
        self.model = model
        self.temperature = float(temperature)

        # Merge coarse illegal extension policy with base defaults + any user-supplied extras.
        merged_illegal = set(self.illegal_attachment_exts) | set(self._ILLEGAL_EXTS)
        if extra_illegal_exts:
            merged_illegal |= set(extra_illegal_exts)
        self.illegal_attachment_exts = merged_illegal

        # Gemini supports a wide range of file types; we stick with a blacklist +
        # MIME filter instead of a strict allow-list, so `allowed_attachment_exts`
        # stays as None.

    # ------------------------------------------------------------------ #
    # Attachment validation & preparation
    # ------------------------------------------------------------------ #

    def _validate_attachment_path(self, path: str) -> None:
        """
        Extend the base validation with Gemini-specific MIME-type checks.

        Base validation has already ensured that `path` exists, is a file,
        and passes the coarse extension policy. Here we reject audio/video
        upfront; other illegal types are controlled by `illegal_attachment_exts`.
        """
        super()._validate_attachment_path(path)

        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        if any(mime.startswith(pref) for pref in self._ILLEGAL_MIME_PREFIXES):
            raise LLMEngineError(
                f"GeminiEngine.attach: MIME type {mime!r} is not supported"
            )

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
            self.client.files.delete(name=name)
        except Exception:
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
        Build the payload for `client.models.generate_content`.

        - `messages` are normalized chat turns (role/content strings).
        - `attachments` is a snapshot of the engine's attachment metadata.

        Attachments are added first so they are available to all subsequent
        text turns, followed by the plain text messages in order.
        """
        system_instruction = self._collect_system(messages)
        flat_texts = self._collect_non_system_texts(messages)

        contents: List[Any] = []

        # Attach uploaded files first so they are available for all text turns.
        for path, meta in attachments.items():
            if meta.get("uploaded") and meta.get("file_obj") is not None:
                contents.append(meta["file_obj"])
            elif meta.get("inlined") and meta.get("inlined_text"):
                contents.append(str(meta["inlined_text"]))

        # Then append plain text turns in order.
        contents.extend([t for t in flat_texts if t])

        return {
            "system_instruction": system_instruction,
            "contents": contents,
        }

    def _call_provider(self, payload: Dict[str, Any]) -> Any:
        """
        Perform a single `models.generate_content` call.

        Retries and backoff are handled by the base `_call_with_retries` wrapper.
        """
        # Use GenerateContentConfig to carry temperature and system instruction.
        cfg = genai.types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=payload.get("system_instruction") or None,
        )

        return self.client.models.generate_content(
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

    def _collect_non_system_texts(
        self, messages: List[Dict[str, str]]
    ) -> List[str]:
        """
        Return a list of non-system message contents, preserving order.

        For Gemini's flat `contents` call style we just send plain strings
        rather than structured chat roles.
        """
        out: List[str] = []
        for m in messages:
            role = (m.get("role") or "").lower()
            if role == "system":
                continue
            txt = m.get("content") or ""
            if txt:
                out.append(txt)
        return out

    def _upload_path(self, path: str) -> Any:
        """
        Upload a local path via the Gemini Files API and return the File object.

        The Gen AI SDK supports passing File objects directly in `contents`.
        """
        abs_path = os.path.abspath(path)
        return self.client.files.upload(file=abs_path)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """
        Diagnostic snapshot for GeminiEngine: provider + model + temperature.

        Keeps output minimal to avoid leaking client or API keys.
        """
        base = super().to_dict()
        base.update({
            "model": self.model,
            "temperature": self.temperature,
        })
        return base

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

    _IMAGE_EXTS: tuple[str, ...] = (
        ".png", ".jpg", ".jpeg",
        ".webp", ".gif", ".bmp",
        ".tif", ".tiff", ".heic",
    )
    _ILLEGAL_EXTS: set[str] = {
        ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",  # archives
        ".exe", ".dll", ".so", ".bin", ".o",  # executables/binaries
        ".db", ".sqlite",  # databases
        ".h5", ".pt", ".pth", ".onnx",  # model weights
    }
    _ILLEGAL_MIME_PREFIXES: tuple[str, ...] = ("audio/", "video/")

    def __init__(
        self,
        model: str = "mistral-medium-latest",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        inline_cutoff_chars: int = 200_000,
        extra_illegal_exts: Optional[set[str]] = None,
        *,
        name: Optional[str] = None,
        description: str = "Mistral LLM Engine",
        filter_extraneous_inputs: bool = True,
        timeout_seconds: float = 600.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 8.0,
    ) -> None:
        """
        Mistral adapter using the chat completion API.

        Flow per call
        -------------
        1) Attachments are prepared via ``attach(path)``:
        - PDFs/images upload through the Mistral Files API, are signed, and are
            attached to the last user message as URL parts.
        - Text/code files are read and inlined into the last user message.

        2) ``invoke({"messages": messages})`` runs the shared ``LLMResult``
        lifecycle:
        - normalize chat messages;
        - snapshot current attachments;
        - build the Mistral provider payload;
        - call ``client.chat.complete(...)``;
        - extract assistant text, token usage, and configured model data;
        - return ``LLMResult``.

        3) ``detach(path)`` triggers best-effort deletion of uploaded files via
        ``_on_detach``, which calls ``client.files.delete(file_id=...)``.

        Token usage
        -----------
        ``_extract_token_usage`` maps ``response.usage`` into
        ``MistralTokenUsage`` using prompt, completion, total, and optional cached
        prompt-token details.

        Model data
        ----------
        ``_get_model_data`` returns configured model identity from ``self.model``.
        """
        if Mistral is None:
            raise RuntimeError(
                "MistralEngine requires the `mistralai` package to be installed."
            )

        sanitized_name = (name or f"mistral_{model}").replace(":", "_").replace("-", "_").replace(" ", "_").replace(".", "_")
        super().__init__(
            name=sanitized_name,
            description=description or "Mistral LLM Engine",
            filter_extraneous_inputs=filter_extraneous_inputs,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        import httpx

        self.client = Mistral(
            api_key=api_key or os.getenv("MISTRAL_API_KEY", ""),
            client=httpx.Client(timeout=self._timeout_seconds),
        )
        self.model = model
        self.temperature = float(temperature)
        self.inline_cutoff_chars = int(inline_cutoff_chars)

        # Merge subclass-specific illegal extensions into the base policy.
        merged_illegal = set(self.illegal_attachment_exts) | set(self._ILLEGAL_EXTS)
        if extra_illegal_exts:
            merged_illegal |= set(extra_illegal_exts)
        self.illegal_attachment_exts = merged_illegal
        # We leave allowed_attachment_exts as None (blacklist-based policy).

    # ------------------------------------------------------------------ #
    # Attachment validation & preparation
    # ------------------------------------------------------------------ #

    def _validate_attachment_path(self, path: str) -> None:
        """
        Extend the base validation with Mistral-specific MIME-type checks.

        Base `_validate_attachment_path` has already ensured that `path` exists,
        is a file, and passes the coarse extension policy. Here we reject
        audio/video upfront; other illegal types are controlled by
        `illegal_attachment_exts`.
        """
        super()._validate_attachment_path(path)

        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        if any(mime.startswith(pref) for pref in self._ILLEGAL_MIME_PREFIXES):
            raise LLMEngineError(
                f"MistralEngine.attach: MIME type {mime!r} is not supported"
            )

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
            if len(text) > self.inline_cutoff_chars:
                text = text[: self.inline_cutoff_chars] + "\n…[truncated]\n"
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
            self.client.files.delete(file_id=file_id)
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
                budget = self.inline_cutoff_chars - total_inlined
                if budget <= 0:
                    if total_inlined == self.inline_cutoff_chars:
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
        Single Mistral chat completion call.

        Retries and error wrapping are handled by the shared `LLMEngine` template.
        """
        return self.client.chat.complete(
            model=self.model,
            messages=payload["messages"],
            temperature=self.temperature,
        )

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
        prompt_tokens_details = usage.prompt_tokens_details
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
        if ext in self._IMAGE_EXTS or mime.startswith("image/"):
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
            up = self.client.files.upload(
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
            signed = self.client.files.get_signed_url(file_id=file_id)
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
            "inline_cutoff_chars": self.inline_cutoff_chars
        })
        return base

# ── LLAMA.CPP (local; no remote file store) ────────────────────────────────────
class LlamaCppEngine(LLMEngine):
    """
    Local llama.cpp adapter using ``llama-cpp-python``.

    This engine wraps a local GGUF/GGML model via ``llama_cpp.Llama`` and plugs
    into the shared ``LLMEngine`` template.

    Model source
    ------------
    A model may be loaded from either:

    - ``model_path``: a direct path to an existing local GGUF/GGML file.
    - ``repo_id`` + ``filename``: a Hugging Face Hub file reference resolved
      through ``hf_hub_download(...)`` into a concrete local file path.

    In both cases, ``self.model_path`` stores the concrete local path loaded by
    ``Llama(model_path=...)``.

    Flow per call
    -------------
    1) Conversation turns are passed through as an OpenAI-compatible
       ``messages`` list.

    2) ``invoke({"messages": messages})`` runs the shared ``LLMResult``
       lifecycle:
       - normalize chat messages;
       - build the llama.cpp provider payload;
       - call ``llm.create_chat_completion(...)``;
       - extract assistant text, token usage, and configured model data;
       - return ``LLMResult``.

    3) Attachments are not supported. Any attempt to call ``attach(path)`` fails
       with ``LLMEngineError``.

    Token usage
    -----------
    ``_extract_token_usage`` maps the OpenAI-compatible response
    ``response["usage"]`` dictionary into ``LlamaCppTokenUsage`` using prompt,
    completion, and total token counts.

    Model data
    ----------
    ``_get_model_data`` returns the concrete local model path loaded by
    llama.cpp.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_dir: Optional[str] = None,
        local_files_only: bool = False,
        hf_token: Optional[Union[str, bool]] = None,
        force_download: bool = False,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        chat_format: Optional[str] = None,
        verbose: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        *,
        name: Optional[str] = None,
        description: str = "Llama.cpp LLM Engine",
        filter_extraneous_inputs: bool = True,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 8.0,
    ) -> None:
        """
        Parameters
        ----------
        model_path:
            Direct path to a local GGUF/GGML model file. If provided, no
            Hugging Face download is performed.
        repo_id:
            Optional Hugging Face repo ID used with ``filename`` when
            ``model_path`` is omitted.
        filename:
            Model filename within ``repo_id`` when resolving from Hugging Face.
        revision:
            Optional Hugging Face branch, tag, or commit hash to resolve.
        cache_dir:
            Optional Hugging Face cache root. Keeps Hugging Face cache layout.
        local_dir:
            Optional human-readable local directory where the model file should
            be materialized.
        local_files_only:
            If True, resolve only from local cache/files and do not download.
        hf_token:
            Optional Hugging Face auth token. ``True`` means use the locally
            configured Hugging Face token.
        force_download:
            If True, re-download the file even if a cached copy exists.

        n_ctx:
            Context window size to configure on the llama.cpp model.
        n_threads:
            Optional number of CPU threads to use for token generation.
        n_threads_batch:
            Optional number of threads to use for batching/prompt processing.
        n_gpu_layers:
            Optional number of model layers to offload to GPU.
        chat_format:
            Optional llama-cpp-python chat format name used to serialize
            role/content messages for local inference.
        verbose:
            If True, enable verbose logging from the underlying llama.cpp runtime.

        temperature:
            Optional default sampling temperature for chat completions.
        max_tokens:
            Optional default maximum generated-token count.
        top_p:
            Optional default nucleus sampling value.
        stop:
            Optional default stop sequence or stop-sequence list.

        name:
            Optional human-friendly engine name; defaults to ``"llama_cpp"``.
        description:
            Human-friendly description for this engine instance.
        filter_extraneous_inputs:
            Whether to filter extraneous dict-first inputs.
        timeout_seconds, max_retries, retry_backoff_base, retry_backoff_max:
            Shared ``LLMEngine`` retry/introspection configuration.
        """
        sanitized_name = (
            name or "llama_cpp"
        ).replace(":", "_").replace("-", "_").replace(" ", "_").replace(".", "_")

        super().__init__(
            name=sanitized_name,
            description=description or "Llama.cpp LLM Engine",
            filter_extraneous_inputs=filter_extraneous_inputs,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        if Llama is None:
            raise RuntimeError("LlamaCppEngine requires the `llama-cpp-python` package.")

        llama_kwargs: Dict[str, Any] = {
            "n_ctx": int(n_ctx),
            "verbose": bool(verbose),
        }
        if n_threads is not None:
            llama_kwargs["n_threads"] = int(n_threads)
        if n_threads_batch is not None:
            llama_kwargs["n_threads_batch"] = int(n_threads_batch)
        if n_gpu_layers is not None:
            llama_kwargs["n_gpu_layers"] = int(n_gpu_layers)
        if chat_format is not None:
            llama_kwargs["chat_format"] = chat_format

        if model_path:
            resolved_model_path = model_path
        elif repo_id and filename:
            if hf_hub_download is None:
                raise RuntimeError(
                    "LlamaCppEngine requires the `huggingface_hub` package when "
                    "`repo_id` and `filename` are used."
                )

            resolved_model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
                local_dir=local_dir,
                local_files_only=bool(local_files_only),
                token=hf_token,
                force_download=bool(force_download),
            )
        else:
            raise LLMEngineError(
                "LlamaCppEngine requires either `model_path` or both "
                "`repo_id` and `filename`."
            )

        self.model_path = str(resolved_model_path)
        self.repo_id = repo_id
        self.filename = filename
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_dir = local_dir
        self.local_files_only = bool(local_files_only)
        self.force_download = bool(force_download)
        self._has_hf_token = hf_token is not None

        self.n_ctx = int(n_ctx)
        self.n_threads = int(n_threads) if n_threads is not None else None
        self.n_threads_batch = (
            int(n_threads_batch) if n_threads_batch is not None else None
        )
        self.n_gpu_layers = int(n_gpu_layers) if n_gpu_layers is not None else None
        self.chat_format = chat_format
        self.verbose = bool(verbose)

        self.temperature = float(temperature) if temperature is not None else None
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.top_p = float(top_p) if top_p is not None else None
        self.stop = stop

        self.llm = Llama(model_path=self.model_path, **llama_kwargs)

    # ------------------------------------------------------------------ #
    # LLMEngine template hooks
    # ------------------------------------------------------------------ #

    def _build_provider_payload(
        self,
        messages: List[Dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
        Map normalized messages to llama.cpp's chat completion schema.

        - `messages` are already validated (role/content strings).
        - `attachments` are ignored; this engine does not support attachments.
        """
        # llama-cpp-python exposes an OpenAI-compatible chat API, so we
        # simply pass the messages through unchanged.
        return {"messages": messages}

    def _call_provider(self, payload: Any) -> Any:
        """
        Perform a single llama.cpp chat completion call.

        Retries and error-wrapping are handled by `LLMEngine._call_with_retries`.
        """
        if getattr(self, "llm", None) is None:
            raise LLMEngineError("LlamaCppEngine: model is not loaded.")

        chat_kwargs: Dict[str, Any] = {}
        if self.temperature is not None:
            chat_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            chat_kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            chat_kwargs["top_p"] = self.top_p
        if self.stop is not None:
            chat_kwargs["stop"] = self.stop

        return self.llm.create_chat_completion(
            messages=payload["messages"],
            **chat_kwargs,
        )

    def _extract_text(self, response: Any) -> str:
        """
        Extract assistant text from a llama.cpp chat completion response.

        Expected structure (OpenAI-compatible):

            response["choices"][0]["message"]["content"] -> str
        """
        try:
            choices = response["choices"]
            if not choices:
                raise KeyError("choices is empty")
            message = choices[0].get("message", {})
            content = message.get("content", "")
        except Exception as exc:
            raise LLMEngineError(
                "LlamaCppEngine._extract_text: unexpected response shape"
            ) from exc
        return str(content).strip()

    def _extract_token_usage(self, response: Any) -> TokenUsage:
        """
        Extract llama-cpp-python chat-completion usage into a LlamaCppTokenUsage
        record.

        The default ``create_chat_completion(...)`` path returns an
        OpenAI-compatible dictionary response whose ``usage`` mapping contains
        prompt, completion, and total token counts.
        """
        try:
            usage = response["usage"]
        except Exception as exc:
            raise LLMEngineError(
                "LlamaCppEngine._extract_token_usage: response missing usage."
            ) from exc

        if not isinstance(usage, Mapping):
            raise LLMEngineError(
                "LlamaCppEngine._extract_token_usage: usage must be a mapping."
            )

        try:
            input_tokens = usage["prompt_tokens"]
            generated_tokens = usage["completion_tokens"]
            total_tokens = usage["total_tokens"]
        except Exception as exc:
            raise LLMEngineError(
                "LlamaCppEngine._extract_token_usage: unexpected usage shape."
            ) from exc

        return LlamaCppTokenUsage(
            input_tokens=input_tokens,
            generated_tokens=generated_tokens,
            total_tokens=total_tokens,
            response_tokens=generated_tokens,
        )

    def _get_model_data(self) -> LLMModelData:
        """
        Return configured llama.cpp model identity data for this engine.

        Model data is derived from the concrete local path loaded by
        llama-cpp-python.
        """
        return LlamaCppModelData(
            provider="llama_cpp",
            model_path=self.model_path,
        )

    # ------------------------------------------------------------------ #
    # Attachments: explicitly unsupported
    # ------------------------------------------------------------------ #

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        """
        Attachments are not supported for local llama.cpp models.

        Any call to `attach(path)` will fail via this method.
        """
        raise LLMEngineError(
            f"{type(self).__name__} does not support attachments; "
            "use plain text in messages instead."
        )

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        """No-op detach hook for llama.cpp (attachments unsupported)."""
        return None

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """
        Diagnostic snapshot for LlamaCppEngine.

        Includes non-secret model source, model-load, compute, and generation
        defaults. The Hugging Face token value is never serialized.
        """
        base = super().to_dict()
        base.update({
            "model_path": self.model_path,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "revision": self.revision,
            "cache_dir": self.cache_dir,
            "local_dir": self.local_dir,
            "local_files_only": self.local_files_only,
            "force_download": self.force_download,
            "has_hf_token": self._has_hf_token,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "n_threads_batch": self.n_threads_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "chat_format": self.chat_format,
            "verbose": self.verbose,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": self.stop,
        })
        return base

# ── PLACEHOLDERS (keep the same abstract contract) ─────────────────────────────
class AzureOpenAIEngine(LLMEngine):
    """
    Placeholder for an Azure OpenAI adapter.

    This class documents the intended constructor/contract but provides **no**
    implementation: `upload`, `delete`, and `invoke` are intentionally unimplemented.
    """
    pass


class BedrockEngine(LLMEngine):
    """
    Placeholder for an AWS Bedrock adapter.

    This class documents the intended constructor/contract but provides **no**
    implementation: `upload`, `delete`, and `invoke` are intentionally unimplemented.
    """
    pass
