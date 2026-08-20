from __future__ import annotations

# ~~~Standard Library Imports~~~
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import logging
import os
import random
import time
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    ClassVar,
)

# ~~~Local Imports~~~
from ..core.Invokable import AtomicInvokable
from ..constants.core import NO_VAL
from ..models.parameters import ParamSpec
from ..exceptions import LLMEngineError
from ..models.results import (
    LLMModelData,
    LLMResult,
    TokenUsage,
)
from ..constants.llm import (
    ILLEGAL_ATTACHMENT_EXTS
)
from ..utils.llm import clean_structure_template

__all__ = ["LLMEngine"]

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

        invoke({"messages": list[{"role": str, "content": str}],
                "output_structure": dict[str, Any] | None}) -> LLMResult

    ``LLMResult.result`` is the generated assistant reply: a plain string for
    ordinary calls, or a ``list``/``dict`` when ``output_structure`` requested
    provider-native structured output. The declared invokable ``return_type``
    is ``"str | list[Any] | dict[str, Any]"`` to reflect this.

    Structured-output result contract
    ----------------------------------
    ``extract()`` computes, once, whether a call requested structured output
    (``output_structure is not None`` at ``invoke``/``async_invoke`` time) and
    passes that as ``requested_structured`` into ``_extract_result``. Every
    engine's ``_extract_result`` override must honor the same contract: when
    ``requested_structured`` is ``False``, return plain text unchanged; when
    ``True``, best-effort coerce the provider's raw output into structured
    data and, on a parse failure, fall back to returning the raw text rather
    than raising uncaught. This keeps detection response-shape-agnostic — no
    engine needs to introspect its own response object to figure out what it
    was asked for; the caller's own request already answers that.

    Structured-output cleaning contract
    -------------------------------------
    Before ``output_structure`` reaches ``_build_provider_payload``, it is
    pruned by ``self._clean_structure_template(output_structure)``. The
    default implementation delegates to ``clean_structure_template`` using
    this engine's ``structure_permitted_keys``/``structure_omitted_keys``
    class attributes — correct for any engine whose omission policy is
    fixed per class. Override ``_clean_structure_template`` itself (not the
    two class attributes) when an engine's policy depends on instance state
    instead — e.g. ``MistralEngine``'s omission set is conditional on its
    mutable ``strict`` property (structured-generation Pass 5); a fixed
    ``ClassVar`` cannot express that, but this hook can.

    The declared invokable schema exposes two input parameters:

    - ``messages`` — a non-empty list of chat-message mappings containing
      string ``role`` and ``content`` fields.
    - ``output_structure`` — optional JSON-Schema-shaped mapping requesting
      schema-constrained structured output for this call; ``None`` (default)
      requests plain text and leaves every existing call path unchanged.

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

    - ``_clean_structure_template``
    - ``_build_provider_payload``
    - ``_call_provider``
    - ``_extract_result``
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

    # Structured-output schema policy — mirrors the attachment allow/deny-list
    # pattern above. Base defaults are a full no-op (unrestricted, nothing
    # dropped); each provider's own pass overrides these directly as class
    # attributes on its own LLMEngine subclass.
    structure_permitted_keys:ClassVar[Optional[frozenset[str]]] = None
    structure_omitted_keys:ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        namespace: str = "llm",
        description: str = "",
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
            parameters=[
                ParamSpec(name="messages",
                          index=0,
                          kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                          type="list[dict[str, str]]",
                          default=NO_VAL),
                ParamSpec(name="output_structure",
                          index=1,
                          kind=ParamSpec.KEYWORD_ONLY,
                          type=("None", "dict[str, Any]"),
                          default=None,
                          description=(
                              "Optional JSON-Schema-shaped mapping requesting "
                              "provider-native, schema-constrained structured "
                              "output for this call. None (default) requests "
                              "plain text."
                          )),
            ],
            return_type="str | list[Any] | dict[str, Any]",
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
                output_structure = filtered_inputs.get("output_structure")
                requested_structured = output_structure is not None

                response = self._call_model(messages, output_structure)
                text, token_usage, model_data = self.extract(response, requested_structured)
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
                raise LLMEngineError(f"{self.name}.invoke failed: {exc}") from exc

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
            output_structure = filtered_inputs.get("output_structure")
            requested_structured = output_structure is not None

            response = await self._call_model_async(messages, output_structure)
            text, token_usage, model_data = self.extract(response, requested_structured)
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
            raise LLMEngineError(f"{self.name}.async_invoke failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _call_model(
        self,
        messages: List[Dict[str, Any]],
        output_structure: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """
        Normalize messages, snapshot current attachments, build the provider
        payload, and return the raw provider response.

        When ``output_structure`` is ``None`` this calls ``_build_provider_payload``
        exactly as before (2-arg call) — plain-text behavior is untouched. When
        given, it is pruned via ``self._clean_structure_template`` (default:
        prunes using this engine's ``structure_permitted_keys``/
        ``structure_omitted_keys``; overridable per-engine — see that method's
        own docstring) and passed as a third positional arg.

        This helper owns only the shared provider-call sequence. It does not
        extract text, extract token usage, construct results, capture timing, or
        emit deprecation warnings.
        """
        normalized = self._normalize_messages(messages)
        attachments = dict(self._attachments)
        if output_structure is None:
            payload = self._build_provider_payload(normalized, attachments)
        else:
            cleaned = self._clean_structure_template(output_structure)
            payload = self._build_provider_payload(normalized, attachments, cleaned)
        return self._call_with_retries(payload)

    async def _call_model_async(
        self,
        messages: List[Dict[str, Any]],
        output_structure: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Async analog of ``_call_model``; calls ``_call_with_retries_async``."""
        normalized = self._normalize_messages(messages)
        attachments = dict(self._attachments)
        if output_structure is None:
            payload = self._build_provider_payload(normalized, attachments)
        else:
            cleaned = self._clean_structure_template(output_structure)
            payload = self._build_provider_payload(normalized, attachments, cleaned)
        return await self._call_with_retries_async(payload)

    def _clean_structure_template(
        self, output_structure: Mapping[str, Any]
    ) -> dict[str, Any]:
        """
        Prune a caller-supplied ``output_structure`` template down to the
        JSON-Schema keywords this engine actually supports, before it
        reaches ``_build_provider_payload``.

        Default hook body — delegates to ``clean_structure_template`` using
        this engine's ``structure_permitted_keys``/``structure_omitted_keys``
        class attributes. Every engine that doesn't override this method
        gets behavior byte-identical to the pre-Pass-5 inlined call. See the
        class docstring's "Structured-output cleaning contract" for the
        override contract.
        """
        return clean_structure_template(
            output_structure, self.structure_permitted_keys, self.structure_omitted_keys
        )

    def extract(
        self, response: Any, requested_structured: bool
    ) -> tuple[str | list[Any] | dict[str, Any], TokenUsage, LLMModelData]:
        """
        Extract the normalized generated result (text, or structured list/dict),
        token usage, and configured model data from one provider response.

        ``requested_structured`` reflects whether the originating call passed
        a real ``output_structure`` (computed once by ``invoke``/
        ``async_invoke``); forwarded to ``_extract_result`` so each engine can
        honor the structured-output result contract without re-deriving it.

        This method coordinates result-path extraction only. It does not call
        the provider, capture timestamps, or construct ``LLMResult``.
        """
        result = self._extract_result(response, requested_structured)
        if not isinstance(result, (str, list, dict)):
            raise LLMEngineError(
                f"{type(self).__name__}._extract_result must return str, list, "
                f"or dict; got {type(result)!r}"
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

        return (result.strip() if isinstance(result, str) else result), token_usage, model_data

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

    # --------------------------------------------------------------------- #
    # Abstract Helpers
    # --------------------------------------------------------------------- #
    @abstractmethod
    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        """
        Decide whether a failed `_call_provider` should be retried.

        Every engine must recognize its own provider SDK's real transient-error
        shape (connection/timeout errors, retryable HTTP status codes) — there
        is no cross-provider default, since no two SDKs in this family share
        one exception hierarchy. Implementations should still respect
        `attempt <= self._max_retries` as the outer bound.
        """
        raise NotImplementedError
    @abstractmethod
    def _build_provider_payload(
        self,
        messages: List[Dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
        output_structure: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Convert normalized messages, attachments, and (optionally) a cleaned
        structured-output template into the provider-specific request payload.

        ``output_structure`` is ``None`` for plain-text calls; a cleaned dict
        (already pruned by ``clean_structure_template``) when the caller
        requested structured output. Engines that only implement the 2-arg
        form will ``TypeError`` on the structured path until their own pass
        adds support — expected mid-migration, not a defensive concern here.
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
    def _extract_result(
        self, response: Any, requested_structured: bool
    ) -> str | list[Any] | dict[str, Any]:
        """
        Extract the assistant's textual OR structured reply from a provider
        response object.

        ``requested_structured`` reflects the caller's own request
        (``output_structure is not None``), computed once by ``invoke``/
        ``async_invoke`` — never re-derived from the response object itself.
        Contract:

        - ``requested_structured=False``: return plain text unchanged, same
          as pre-structured-generation behavior.
        - ``requested_structured=True``: attempt to coerce the provider's raw
          text output into structured (list/dict) data; on any parse
          failure, fall back to returning the raw text string rather than
          raising uncaught. A caller that requested structure and receives a
          ``str`` back is expected to detect and handle that itself.

        Renamed from ``_extract_text`` (Pass 1, ``structured-generation``);
        gained ``requested_structured`` (Pass 3). Every engine's override
        must match this 2-arg signature, or that engine has an unimplemented
        abstract method and is non-instantiable until its own pass catches
        up.
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
