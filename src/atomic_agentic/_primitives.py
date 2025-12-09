from __future__ import annotations

import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional
from ._exceptions import LLMEngineError

logger = logging.getLogger(__name__)

__all__ = [
    "LLMEngine",
]


class LLMEngine(ABC):
    """
    Base template-method primitive for LLM provider adapters.

    Engines are stateless with respect to conversation history: the Agent owns
    the message history. An engine instance represents a particular provider +
    model configuration plus a persistent set of attachments.

    Public contract
    ---------------
    - `invoke(messages: list[{"role": str, "content": str}]) -> str`
      is the *only* required public entrypoint for making a call.
    - Attachments are managed via `attach` / `detach` / `clear_attachments`.
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
        self._name = name or type(self).__name__
        self._timeout_seconds = float(timeout_seconds)
        self._max_retries = int(max_retries)
        self._retry_backoff_base = float(retry_backoff_base)
        self._retry_backoff_max = float(retry_backoff_max)
        # Persistent mapping: local path -> provider-specific metadata
        self._attachments: Dict[str, Mapping[str, Any]] = {}

    # --------------------------------------------------------------------- #
    # Public surface
    # --------------------------------------------------------------------- #

    @property
    def name(self) -> str:
        """Human-friendly identifier for this engine instance."""
        return self._name

    @property
    def attachments(self) -> Mapping[str, Mapping[str, Any]]:
        """
        Read-only view of currently attached paths.

        Keys are local paths; values are provider-specific metadata dicts as
        returned by `_prepare_attachment`.
        """
        # Return a shallow copy to discourage mutation of internal state.
        return dict(self._attachments)

    # Attach / detach ----------------------------------------------------- #

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
        logger.debug("LLMEngine %s attached %s", self._name, path)
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
                self._name,
                exc,
                path,
            )
        logger.debug("LLMEngine %s detached %s", self._name, path)
        return True

    def clear_attachments(self) -> None:
        """Detach all currently attached paths."""
        for path in list(self._attachments.keys()):
            self.detach(path)

    # Template `invoke` --------------------------------------------------- #

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """
        Template method that defines the engine invocation lifecycle.

        Steps:
        1. Normalize and validate the input `messages`.
        2. Snapshot current attachments.
        3. Ask the subclass to build a provider-specific payload.
        4. Call the provider with retries/timeouts.
        5. Extract and normalize the assistant text.

        Subclasses **must not** override this method; they customize behavior
        via the protected hooks documented below.
        """
        start = time.time()
        try:
            normalized = self._normalize_messages(messages)
            attachments = dict(self._attachments)
            payload = self._build_provider_payload(normalized, attachments)
            response = self._call_with_retries(payload)
            text = self._extract_text(response)

            if not isinstance(text, str):
                raise LLMEngineError(
                    f"{type(self).__name__}._extract_text must return str; "
                    f"got {type(text)!r}"
                )
            return text.strip()
        except LLMEngineError:
            # Already normalized; bubble up unchanged.
            raise
        except Exception as exc:
            raise LLMEngineError(f"{self._name}.invoke failed") from exc
        finally:
            duration = time.time() - start
            logger.debug(
                "LLMEngine %s.invoke completed in %.3fs", self._name, duration
            )

    # --------------------------------------------------------------------- #
    # Shared helpers used by the template
    # --------------------------------------------------------------------- #

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
                f"LLMEngine.attach: extension {ext!r} is not supported by {self._name}"
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
                    self._name,
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
    # Abstract hooks for subclasses
    # --------------------------------------------------------------------- #

    @abstractmethod
    def _build_provider_payload(
        self,
        messages: List[Dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> Any:
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
    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        """
        Prepare a local path for reuse with this engine.

        Implementations typically:
        - validate the path and extension vs provider capabilities,
        - perform any remote upload or inlining,
        - return an opaque metadata mapping used later by `_build_provider_payload`.
        """
        raise NotImplementedError

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        """
        Optional hook called when an attachment is detached.

        Subclasses may implement provider-specific cleanup (e.g. remote delete).
        Default implementation is a no-op.
        """
        # Intentionally a no-op by default.
        return None

    # --------------------------------------------------------------------- #
    # Introspection
    # --------------------------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        """
        Shallow, non-secret configuration snapshot for debugging / logging.
        """
        return {
            "name": self._name,
            "timeout_seconds": self._timeout_seconds,
            "max_retries": self._max_retries,
            "attachments": list(self._attachments.keys()),
            "provider": type(self).__name__,
        }
