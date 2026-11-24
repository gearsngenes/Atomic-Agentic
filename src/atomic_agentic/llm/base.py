from __future__ import annotations

# ~~~Standard Library Imports~~~
import asyncio
import copy
from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

# ~~~Local Imports~~~
from ..invokables.atomic_invokable import AtomicInvokable
from ..constants.core import NO_VAL
from ..models.parameters import ParamSpec
from ..exceptions import LLMEngineError
from ..models.results import (
    LLMModelData,
    LLMResult,
    TokenUsage,
)

__all__ = ["LLMEngine"]

_SECRET_KEY_FRAGMENTS = ("key", "token", "secret", "auth", "password", "credential", "header")


def _redact_secrets(value: Any) -> Any:
    """Return ``value`` with every secret-named dict entry replaced by ``"***"``.

    A key is secret when its lowercased name contains one of
    ``_SECRET_KEY_FRAGMENTS``; the plural ``tokens`` is ignored so token-count
    settings (``max_tokens``, ``max_output_tokens``) stay visible. Recurses into nested dicts and lists in place,
    so pass a deep copy.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            name = k.lower().replace("tokens", "") if isinstance(k, str) else ""
            if any(s in name for s in _SECRET_KEY_FRAGMENTS):
                value[k] = "***"
            else:
                _redact_secrets(v)
    elif isinstance(value, list):
        for item in value:
            _redact_secrets(item)
    return value

# ───────────────────────────────────────────────────────────────────────────────
# LLMEngine primitive
# ───────────────────────────────────────────────────────────────────────────────
class LLMEngine(AtomicInvokable, ABC):
    """
    Base template-method primitive for LLM provider adapters.

    Engines are stateless with respect to conversation history: the Agent owns
    message history. An engine instance represents a particular provider +
    model configuration.

    Public contract
    ---------------
    ``LLMEngine`` is an ``AtomicInvokable``, so its canonical public invocation
    entrypoint is dict-first:

        invoke({"messages": list[{"role": str, "content": str}],
                "output_structure": dict[str, Any] | None}) -> LLMResult

    ``LLMResult.result`` is the generated assistant reply: a plain string for
    ordinary calls, or a ``str``/``int``/``float``/``bool``/``list``/``dict``/
    ``None`` when ``output_structure`` requested provider-native structured
    output -- whatever a conformant JSON value decodes to via ``json.loads``.
    The declared invokable ``return_type`` is the tuple
    ``("str", "int", "float", "bool", "list[Any]", "dict[str, Any]", "None")``
    to reflect this.

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

    Structured-output request contract
    -------------------------------------
    ``output_structure`` is forwarded to ``_build_provider_payload``
    completely unmodified from what the caller supplied — this base class
    performs no JSON-Schema key pruning, allow-listing, or validation of any
    kind (removed, structured-generation Pass 8: the prior
    ``clean_structure_template``/``structure_permitted_keys``/
    ``structure_omitted_keys`` mechanism silently traded away
    caller-declared schema constraints even in cases where every omitted
    keyword would otherwise have produced an immediate, diagnosable
    provider error — in tension with this codebase's explicit-or-raise
    discipline). A schema keyword a given provider rejects fails as that
    provider's own native error, wrapped in ``LLMEngineError`` by
    the engine's invoke hooks — the same failure shape already used for
    any other illegal per-provider configuration (e.g. an unsupported
    ``temperature`` value). No engine gates on capability: a model/provider
    combination with no structured-output path fails natively and is wrapped
    in ``LLMEngineError`` like any other provider error.

    Note: ``_normalize_messages`` still requires every message's ``content``
    to be a plain ``str`` — a structured ``LLMResult.result`` (``list``/
    ``dict``) cannot currently be fed back into a later call's ``messages``
    unmodified. No ``src/`` code does this today; whichever future pass
    wires ``output_structure`` through the ``Agent`` layer will need to make
    an explicit choice here (``str(...)``, ``json.dumps(...)``, or a richer
    message-content shape).

    The declared invokable schema exposes two input parameters:

    - ``messages`` — a non-empty list of chat-message mappings containing
      string ``role`` and ``content`` fields.
    - ``output_structure`` — optional JSON-Schema-shaped mapping requesting
      schema-constrained structured output for this call; ``None`` (default)
      requests plain text and leaves every existing call path unchanged.

    Engine lifecycle
    ----------------
    The inherited ``invoke`` template filters the inputs and builds the
    ``LLMResult``; ``_invoke_impl`` does:

    1. Validate and normalize ``messages``.
    2. Build the provider-specific payload.
    3. Call the provider.
    4. Extract assistant text, token usage, and configured model data.

    and returns ``{"result", "token_usage", "model_data"}``.

    Provider-specific behavior should normally be implemented through the
    protected template hooks:

    - ``_build_provider_payload``
    - ``_call_provider``
    - ``_extract_result``
    - ``_extract_token_usage``
    - ``_get_model_data``

    Retry and timeout are each provider SDK's own client settings, and
    attachments are an API of the individual engines that support them; none
    of the three lives here.

    Runtime parameters
    ------------------
    The base owns one store of provider-call parameters (``temperature``,
    token caps, reasoning settings, ... under the provider's own names).
    ``runtime_parameters`` returns a deep copy; ``update_runtime_parameters``
    is the only mutator (merge, ``None`` removes a key, every key is validated
    before any is applied). Each engine's ``_build_provider_payload`` takes a
    shallow snapshot of the store once per call (live objects such as a Llama
    grammar are never deep-copied per call) and forwards it to the provider
    verbatim, so a later ``update_runtime_parameters`` never changes an
    in-flight call. Keys the engine owns on the call (model, the message
    field, ``stream``, the structured-output field) are listed in
    ``_RESERVED_RUNTIME_KEYS`` and rejected; anything else unknown is rejected
    by the provider at call time and surfaces as ``LLMEngineError``, except on
    ``LlamaCppEngine``, which validates keys against
    ``create_chat_completion`` in ``_validate_runtime_keys``.

    Streaming is unsupported: ``stream`` is reserved on every engine.
    Credentials belong in the client / client kwargs, not runtime parameters;
    ``to_dict`` redacts runtime parameters by key name (any key containing
    key, token, secret, auth, password, credential or header becomes ``"***"``,
    recursively; the plural ``tokens`` is ignored so ``max_tokens`` stays
    visible), as a safety net.
    """

    result_cls: ClassVar[type[LLMResult]] = LLMResult

    # Keys the engine owns on the provider call; update_runtime_parameters
    # rejects them. Subclasses override.
    _RESERVED_RUNTIME_KEYS: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        description: str = "",
    ) -> None:
        """
        Parameters
        ----------
        name:
            Optional human-friendly identifier for logging/introspection;
            defaults to the class name.
        description:
            Human-readable description; defaults to ``"LLM Engine"``.
        """
        # Pass the provided name (or the class name) directly to AtomicInvokable.
        AtomicInvokable.__init__(
            self,
            name=name or type(self).__name__,
            description=description or "LLM Engine",
            parameters=[
                ParamSpec(name="messages",
                          index=0,
                          kind=ParamSpec.POSITIONAL_OR_KEYWORD,
                          type="list[dict[str, str]]",
                          default=NO_VAL,
                          description=(
                              "A non-empty list of chat-message mappings. "
                              "Each mapping must have a string 'role' key "
                              "(one of 'system', 'user', 'assistant') and a "
                              "string 'content' key (free-form text)."
                          )),
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
            return_type=("str", "int", "float", "bool", "list[Any]", "dict[str, Any]", "None"),
        )

        self._runtime_parameters: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def runtime_parameters(self) -> Dict[str, Any]:
        """Deep copy of the stored provider-call parameters; mutating the copy
        (nested values included) changes nothing. Use
        ``update_runtime_parameters`` to change them."""
        return copy.deepcopy(self._runtime_parameters)

    @property
    def model_data(self) -> LLMModelData:
        """Configured model identity for this engine.

        Derived from engine configuration, so it is available without making
        an LLM call. Delegates to ``_get_model_data``.
        """
        return self._get_model_data()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update_runtime_parameters(self, **kwargs: Any) -> None:
        """
        Merge provider-call parameters into the store.

        1. Validate every key (``_validate_runtime_keys``); raise before
           anything is applied.
        2. For each key: a ``None`` value removes it (absent keys are a
           no-op), any other value sets it.
        """
        self._validate_runtime_keys(kwargs.keys())
        for key, value in kwargs.items():
            if value is None:
                self._runtime_parameters.pop(key, None)
            else:
                self._runtime_parameters[key] = value

    def _validate_runtime_keys(self, keys: Iterable[str]) -> None:
        """Reject keys the engine owns on the provider call. Engines extend
        this to add their own key checks."""
        reserved = sorted(set(keys) & self._RESERVED_RUNTIME_KEYS)
        if reserved:
            raise LLMEngineError(
                f"{type(self).__name__}.update_runtime_parameters: {reserved} "
                "are managed by the engine and cannot be set"
            )

    def _invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        1. messages = inputs.get("messages"); not a list -> LLMEngineError.
        2. output_structure = inputs.get("output_structure");
           requested_structured = output_structure is not None.
        3. response = self._call_model(messages, output_structure).
        4. text, token_usage, model_data = self.extract(response, requested_structured).
        5. Return {"result", "token_usage", "model_data"}.

        LLMEngineError propagates; any other exception is raised as
        LLMEngineError(f"{self.name}.invoke failed: {exc}") from exc.
        """
        try:
            messages = inputs.get("messages")
            if not isinstance(messages, list):
                raise LLMEngineError(
                    "LLMEngine.invoke: 'messages' input must be a list"
                )
            output_structure = inputs.get("output_structure")
            requested_structured = output_structure is not None

            response = self._call_model(messages, output_structure)
            text, token_usage, model_data = self.extract(response, requested_structured)
            return {"result": text, "token_usage": token_usage, "model_data": model_data}
        except LLMEngineError:
            raise
        except Exception as exc:
            raise LLMEngineError(f"{self.name}.invoke failed: {exc}") from exc

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async twin of ``_invoke_impl`` (awaits ``_call_model_async``)."""
        try:
            messages = inputs.get("messages")
            if not isinstance(messages, list):
                raise LLMEngineError(
                    "LLMEngine.async_invoke: 'messages' input must be a list"
                )
            output_structure = inputs.get("output_structure")
            requested_structured = output_structure is not None

            response = await self._call_model_async(messages, output_structure)
            text, token_usage, model_data = self.extract(response, requested_structured)
            return {"result": text, "token_usage": token_usage, "model_data": model_data}
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
        Normalize messages, build the provider payload, and return the raw
        provider response.

        ``_build_provider_payload`` is called unconditionally as a 2-arg call
        — every concrete engine implements that signature with
        ``output_structure`` defaulting to ``None``, so plain-text calls are
        unaffected by always passing it through.

        This helper owns only the shared provider-call sequence. It does not
        extract text, extract token usage, or construct results.
        """
        normalized = self._normalize_messages(messages)
        payload = self._build_provider_payload(normalized, output_structure)
        return self._call_provider(payload)

    async def _call_model_async(
        self,
        messages: List[Dict[str, Any]],
        output_structure: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Async analog of ``_call_model``; awaits ``_call_provider_async``."""
        normalized = self._normalize_messages(messages)
        payload = self._build_provider_payload(normalized, output_structure)
        return await self._call_provider_async(payload)

    def extract(
        self, response: Any, requested_structured: bool
    ) -> tuple[str | int | float | bool | list[Any] | dict[str, Any] | None, TokenUsage, LLMModelData]:
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
        if not isinstance(result, (str, int, float, bool, list, dict, type(None))):
            raise LLMEngineError(
                f"{type(self).__name__}._extract_result must return str, int, "
                f"float, bool, list, dict, or None; got {type(result)!r}"
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

    # --------------------------------------------------------------------- #
    # Abstract Helpers
    # --------------------------------------------------------------------- #
    @abstractmethod
    def _build_provider_payload(
        self,
        messages: List[Dict[str, str]],
        output_structure: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Convert normalized messages and (optionally) a caller-supplied
        structured-output template into the provider-specific request
        payload: the exact keyword arguments of the SDK call, so
        ``_call_provider`` only picks the client and calls it with
        ``**payload``. Build it from a snapshot of the runtime parameters and
        ``update`` the engine-owned fields (model, messages, the
        structured-output field) over it, so they always win.

        ``output_structure`` is ``None`` for plain-text calls; the caller's
        raw JSON-Schema-shaped mapping, completely unmodified, when the
        caller requested structured output (structured-generation Pass 8:
        this base class no longer prunes or validates it — see the class
        docstring's "Structured-output request contract"). Called
        unconditionally as a 2-arg call by ``_call_model``/``_call_model_async``
        — every concrete engine implements this signature.
        """
        raise NotImplementedError

    @abstractmethod
    def _call_provider(self, payload: Any) -> Any:
        """
        Perform a single call to the underlying provider using the given payload.
        """
        raise NotImplementedError

    async def _call_provider_async(self, payload: Any) -> Any:
        """
        Async provider call. Default wraps the sync ``_call_provider`` in a
        worker thread; engines with a native async client override it.
        """
        return await asyncio.to_thread(self._call_provider, payload)

    @abstractmethod
    def _extract_result(
        self, response: Any, requested_structured: bool
    ) -> str | int | float | bool | list[Any] | dict[str, Any] | None:
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
          text output into any JSON-decodable value (object, array, string,
          number, boolean, or null — not just object/array); on any parse
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

    # --------------------------------------------------------------------- #
    # Serialization
    # --------------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """
        Non-secret configuration snapshot for debugging / logging.

        ``runtime_parameters`` is deep-copied and redacted by key name
        (see ``_redact_secrets``).

        Invocation result data such as token usage and model data belongs to
        ``LLMResult``, not to engine metadata serialization.
        """
        d = super().to_dict()
        d.update({
            "type": type(self).__name__,
            "runtime_parameters": _redact_secrets(copy.deepcopy(self._runtime_parameters)),
        })
        return d
