from __future__ import annotations

# ~~~Standard Library Imports~~~
import asyncio
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

# ~~~Provider SDK Imports~~~
# OpenAI
try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    OpenAI = None
    AsyncOpenAI = None
# Google
try: from google import genai
except ImportError: genai = None
# Mistral
try: from mistralai.client import Mistral
except ImportError: Mistral = None
# Llama-CPP-Python
try: from llama_cpp import Llama
except ImportError: Llama = None
# Hugging Face Hub
try: from huggingface_hub import hf_hub_download
except ImportError: hf_hub_download = None

# ~~~Local Imports~~~
from ..core.Invokable import AtomicInvokable
from ..constants.core import NO_VAL
from ..models.parameters import ParamSpec
from ..exceptions import LLMEngineError
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
from ..utils.core import run_coro_sync
from ..constants.engines import (
    ILLEGAL_ATTACHMENT_EXTS,
    ENGINE_ILLEGAL_MIME_PREFIXES,
    MISTRAL_IMAGE_EXTS,
    OPENAI_IMAGE_EXTS,
    OPENAI_ALLOWED_EXTS,
)
from ..utils.engines import validate_attachment_path

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

    # Attachment policy — coarse safety/robustness guard applied before
    # provider-specific checks. Sourced from constants.engines.
    illegal_attachment_exts: set[str] = ILLEGAL_ATTACHMENT_EXTS
    allowed_attachment_exts: Optional[set[str]] = None

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        namespace: str = "llm",
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
        namespace:
            Grouping label for this engine instance.
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
            namespace=namespace,
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

    @property
    def timeout_seconds(self) -> float:
        """Suggested per-call timeout; subclasses honor this where their
        provider SDK allows it to be configured."""
        return self._timeout_seconds

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

    async def async_invoke(self, inputs: Mapping[str, Any]) -> LLMResult:
        """
        Async analog of ``invoke``. Does not acquire ``_invoke_lock``
        (``threading.RLock`` would block the event loop).
        """
        logger.info("[Async %s started]", self.full_name)
        started_at = datetime.now(timezone.utc)

        try:
            filtered_inputs = self.filter_inputs(inputs)
            messages = filtered_inputs.get("messages")
            if not isinstance(messages, list):
                raise LLMEngineError(
                    "LLMEngine.async_invoke: 'messages' input must be a list"
                )

            response = await self._call_model_async(messages)
            text, token_usage, model_data = self.extract(response)
            ended_at = datetime.now(timezone.utc)

            result = self.make_result(
                result=text,
                started_at=started_at,
                ended_at=ended_at,
                token_usage=token_usage,
                model_data=model_data,
            )

            logger.info("[Async %s finished]", self.full_name)
            return result

        except LLMEngineError:
            raise
        except Exception as exc:
            raise LLMEngineError(f"{self.name}.async_invoke failed") from exc

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

    async def _call_model_async(self, messages: List[Dict[str, Any]]) -> Any:
        """Async analog of ``_call_model``; calls ``_call_with_retries_async``."""
        normalized = self._normalize_messages(messages)
        attachments = dict(self._attachments)
        payload = self._build_provider_payload(normalized, attachments)
        return await self._call_with_retries_async(payload)

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

    async def _call_with_retries_async(self, payload: Any) -> Any:
        """
        Async analog of ``_call_with_retries``; uses ``await asyncio.sleep``
        so the event loop is not blocked during backoff waits.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._call_provider_async(payload)
            except LLMEngineError:
                raise
            except Exception as exc:
                if not self._should_retry(exc, attempt):
                    raise
                sleep = min(
                    self._retry_backoff_base * (2 ** (attempt - 1)),
                    self._retry_backoff_max,
                )
                sleep *= random.uniform(0.8, 1.2)
                logger.debug(
                    "LLMEngine %s attempt %d failed with %r; retrying in %.2fs",
                    self.name,
                    attempt,
                    exc,
                    sleep,
                )
                await asyncio.sleep(sleep)

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

    async def _call_provider_async(self, payload: Any) -> Any:
        """
        Async provider call. Default wraps the sync ``_call_provider`` in a
        worker thread. Remote engine subclasses override with a native async
        client in Passes 2–4.
        """
        return await asyncio.to_thread(self._call_provider, payload)

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

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        """
        Prepare a local path for reuse with this engine.

        The base implementation rejects all attachments. Remote engine subclasses
        override this to perform provider-specific upload or inlining.
        """
        raise LLMEngineError(
            f"{type(self).__name__} does not support attachments; "
            "use plain text in messages instead."
        )

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        """Hook called when an attachment is detached. No-op by default."""
        return None


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
    ``_extract_text`` reads the generated assistant text from the Responses API
    response. ``_extract_token_usage`` maps Responses API usage fields into an
    ``OpenAITokenUsage`` record. ``_get_model_data`` returns configured model
    identity from ``self.model``.
    """

    def __init__(
        self,
        model: str,
        name: str | None = None,
        namespace: str = "llm",
        description: str = "OpenAI LLM Engine",
        client: OpenAI | AsyncOpenAI | None = None,
        temperature: float | None = 0.1,
        max_output_tokens: int | None = None,
        reasoning: dict[str, str] | None = None,
        truncation: str | None = None,
        inline_cutoff_chars: int = 200_000,
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
        inline_cutoff_chars:
            Maximum characters to inline from text/code attachments.
        filter_extraneous_inputs:
            Whether to silently drop inputs not declared in the engine's schema.
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
            filter_extraneous_inputs=filter_extraneous_inputs,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        # Step 2 — SDK presence check.
        if OpenAI is None:
            raise RuntimeError(
                "OpenAIEngine requires the `openai` package; install `openai` to use it."
            )

        # Step 3 — Build shared client kwargs; seed timeout from the base-engine knob.
        _ckw = dict(client_kwargs)
        _ckw.setdefault("timeout", self._timeout_seconds)

        # Step 4 — Single client: injected as-is or built sync from kwargs.
        self._client: OpenAI | AsyncOpenAI = (
            client if client is not None else OpenAI(**_ckw)
        )

        # Step 5 — Store model and Responses API config.
        self.model = model
        self.temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._reasoning = reasoning
        self._truncation = truncation
        self._inline_cutoff_chars = int(inline_cutoff_chars)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def inline_cutoff_chars(self) -> int:
        """Max characters inlined from text attachments; fixed at construction."""
        return self._inline_cutoff_chars

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
        if self.temperature is not None and self._reasoning is None:
            kwargs["temperature"] = self.temperature
        if self._max_output_tokens is not None:
            kwargs["max_output_tokens"] = self._max_output_tokens
        if self._reasoning is not None:
            kwargs["reasoning"] = self._reasoning
        if self._truncation is not None:
            kwargs["truncation"] = self._truncation
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
        if response_tokens < 0:
            raise LLMEngineError(
                "OpenAI response usage produced a negative response token count "
                "after subtracting reasoning_tokens from output_tokens."
            )

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
            "type": "OpenAIEngine",
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self._max_output_tokens,
            "reasoning": self._reasoning,
            "truncation": self._truncation,
            "inline_cutoff_chars": self._inline_cutoff_chars,
        })
        return base

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
            raise RuntimeError(
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
        filter_extraneous_inputs, timeout_seconds, max_retries,
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
            filter_extraneous_inputs=filter_extraneous_inputs,
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
        namespace: str = "llm",
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
            namespace=namespace,
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

        self._model_path = str(resolved_model_path)
        self._repo_id = repo_id
        self._filename = filename
        self._revision = revision
        self._cache_dir = cache_dir
        self._local_dir = local_dir
        self._local_files_only = bool(local_files_only)
        self._force_download = bool(force_download)
        self._has_hf_token = hf_token is not None

        self._n_ctx = int(n_ctx)
        self._n_threads = int(n_threads) if n_threads is not None else None
        self._n_threads_batch = (
            int(n_threads_batch) if n_threads_batch is not None else None
        )
        self._n_gpu_layers = int(n_gpu_layers) if n_gpu_layers is not None else None
        self._chat_format = chat_format
        self._verbose = bool(verbose)

        self.temperature = float(temperature) if temperature is not None else None
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.top_p = float(top_p) if top_p is not None else None
        self.stop = stop

        self._llm = Llama(model_path=self._model_path, **llama_kwargs)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def model_path(self) -> str:
        """Concrete local path of the loaded model; fixed at construction."""
        return self._model_path

    @property
    def repo_id(self) -> Optional[str]:
        """Hugging Face repo ID used to download the model; fixed at construction."""
        return self._repo_id

    @property
    def filename(self) -> Optional[str]:
        """Model filename within the repo; fixed at construction."""
        return self._filename

    @property
    def revision(self) -> Optional[str]:
        """Hugging Face revision used at download; fixed at construction."""
        return self._revision

    @property
    def cache_dir(self) -> Optional[str]:
        """Hugging Face cache root used at download; fixed at construction."""
        return self._cache_dir

    @property
    def local_dir(self) -> Optional[str]:
        """Local directory where the model was materialized; fixed at construction."""
        return self._local_dir

    @property
    def local_files_only(self) -> bool:
        """Whether only local files were used at download; fixed at construction."""
        return self._local_files_only

    @property
    def force_download(self) -> bool:
        """Whether a forced re-download was requested; fixed at construction."""
        return self._force_download

    @property
    def n_ctx(self) -> int:
        """Context window size passed to Llama at construction."""
        return self._n_ctx

    @property
    def n_threads(self) -> Optional[int]:
        """CPU thread count passed to Llama at construction."""
        return self._n_threads

    @property
    def n_threads_batch(self) -> Optional[int]:
        """Batch CPU thread count passed to Llama at construction."""
        return self._n_threads_batch

    @property
    def n_gpu_layers(self) -> Optional[int]:
        """GPU layer count passed to Llama at construction."""
        return self._n_gpu_layers

    @property
    def chat_format(self) -> Optional[str]:
        """Chat format name passed to Llama at construction."""
        return self._chat_format

    @property
    def verbose(self) -> bool:
        """Verbose flag passed to Llama at construction."""
        return self._verbose

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
        if getattr(self, "_llm", None) is None:
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

        return self._llm.create_chat_completion(
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
            model_path=self._model_path,
        )

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
            "model_path": self._model_path,
            "repo_id": self._repo_id,
            "filename": self._filename,
            "revision": self._revision,
            "cache_dir": self._cache_dir,
            "local_dir": self._local_dir,
            "local_files_only": self._local_files_only,
            "force_download": self._force_download,
            "has_hf_token": self._has_hf_token,
            "n_ctx": self._n_ctx,
            "n_threads": self._n_threads,
            "n_threads_batch": self._n_threads_batch,
            "n_gpu_layers": self._n_gpu_layers,
            "chat_format": self._chat_format,
            "verbose": self._verbose,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": self.stop,
        })
        return base
