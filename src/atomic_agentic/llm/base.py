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

_VALID_ROLES = frozenset({"system", "user", "assistant"})
_SECRET_KEY_FRAGMENTS = ("key", "token", "secret", "auth", "password", "credential", "header")


def _redact_secrets(value: Any) -> Any:
    """Return ``value`` with every secret-named dict entry replaced by ``"***"``.

    A key is secret when its lowercased name contains one of
    ``_SECRET_KEY_FRAGMENTS``; the plural ``tokens`` is ignored so token-count
    settings (``max_tokens``, ``max_output_tokens``) stay visible. Matching is
    by substring, so some benign keys (e.g. ``prompt_cache_key``) are redacted
    too. Recurses into nested dicts and lists in place, so pass a deep copy.
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
    message history. An engine instance represents one provider + model
    configuration. ``LLMEngine`` is an ``AtomicInvokable``; its entrypoint is

        invoke({"messages": list[{"role": str, "content": str}],
                "output_structure": dict[str, Any] | None}) -> LLMResult

    ``LLMResult.result`` is the assistant reply: a ``str`` for a plain call, or
    any ``json.loads`` value (``str``/``int``/``float``/``bool``/``list``/
    ``dict``/``None``) when ``output_structure`` requested structured output.

    Layers of one call, in order
    ----------------------------
    1. Public ``invoke`` / ``async_invoke`` (inherited template): filters the
       inputs, times the call and builds the ``LLMResult`` from the dict
       ``_invoke_impl`` returns.
    2. ``_invoke_impl`` / ``_async_invoke_impl``: receive the input dict and
       return ``{"result", "token_usage", "model_data"}``. They guard
       ``messages`` (must be a list), run layers 3 to 6, and wrap any
       non-``LLMEngineError`` exception as ``LLMEngineError``
       (``"<Class>.invoke: ..."``).
    3. ``_normalize_messages``: receives the raw list and returns
       ``[{"role", "content"}]`` with lowercased roles. The messages contract:
       a non-empty list of mappings with string ``role`` and ``content``; the
       role must be ``system``, ``user`` or ``assistant``; at least one
       ``user`` message is required (system content is optional). Anything
       else raises ``LLMEngineError``. Content must be a plain ``str``, so a
       structured ``LLMResult.result`` cannot be fed back unmodified.
    4. ``_build_provider_payload(messages, output_structure)``: receives the
       normalized messages and the caller's raw schema (or ``None``) and
       returns the exact keyword arguments of the SDK call. Build it with
       ``dict(self._runtime_parameters)`` and ``update`` the engine-owned
       fields over it so they always win. The schema is forwarded
       unmodified: the base prunes and validates nothing, and a keyword the
       provider rejects fails as that provider's own error, wrapped in
       ``LLMEngineError``. No engine gates on capability.
    5. ``_call_provider(payload)`` / ``_call_provider_async(payload)``:
       receive the payload and return the raw SDK response. The async default
       runs the sync hook in a worker thread; engines with a native async
       client override it.
    6. ``extract(response, requested_structured)``: calls the three hooks
       below, checks their return types, strips a string result and returns
       ``(result, token_usage, model_data)``. ``requested_structured`` is
       ``output_structure is not None``, computed once by the caller.

       - ``_extract_result(response, requested_structured)``: returns the
         reply. Plain call: the text. Structured call: ``json.loads`` of the
         text, falling back to the raw text on a parse failure.
       - ``_extract_token_usage(response)``: returns the engine's
         ``TokenUsage`` subclass record.
       - ``_get_model_data()``: returns ``LLMModelData`` from configuration
         (also exposed as the ``model_data`` property).
    7. Result: ``LLMResult`` carrying the result, token usage and model data.

    What an engine overrides
    ------------------------
    ``_build_provider_payload``, ``_call_provider`` (plus
    ``_call_provider_async`` for a native async client), ``_extract_result``,
    ``_extract_token_usage`` and ``_get_model_data``; optionally
    ``_RESERVED_RUNTIME_KEYS``, ``_validate_runtime_keys`` and ``to_dict``.
    Retry and timeout are each SDK client's own settings, and attachments are
    an API of the individual engines that support them.

    Runtime parameters
    ------------------
    The base owns one store of provider-call parameters (``temperature``,
    token caps, reasoning settings, ... under the provider's own names).
    ``runtime_parameters`` returns a deep copy; ``update_runtime_parameters``
    is the only mutator (merge, ``None`` removes a key, every key is validated
    before any is applied, each value is deep-copied on the way in). Payload
    builders take a shallow snapshot of the store once per call, so a later
    update never changes an in-flight call.

    Keys the engine owns on the call are listed in ``_RESERVED_RUNTIME_KEYS``
    and rejected: the model, the message field (``messages``, ``input``,
    ``contents``), the system field (``system``, ``instructions``,
    ``system_instruction``), the response-format fields an engine sets itself
    (``response_format``; Gemini's ``response_mime_type`` and
    ``response_json_schema``) and ``stream`` (streaming is unsupported on
    every engine). OpenAI's ``text`` and Anthropic's ``output_config`` are not
    reserved: a structured call merges its format key into a copy of them.
    Other unknown keys are rejected by the provider at call time (surfacing as
    ``LLMEngineError``), except on ``LlamaCppEngine`` and ``GeminiEngine``,
    which check keys against the call signature or config model when they are
    set.

    Credentials belong in the client or client kwargs, not runtime
    parameters; ``to_dict`` redacts runtime parameters by key name as a
    safety net (see ``_redact_secrets``).

    ``require_usage``
    -----------------
    A validated, settable ``bool`` (default ``True``). When a provider
    response carries no usage object at all, the engine raises
    ``LLMEngineError``; with ``require_usage = False`` it returns an all-zero
    token-usage record instead. A partially malformed usage object still
    raises. ``AnthropicEngine`` always reports usage and ignores the flag.
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
        self.require_usage = True

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
    def require_usage(self) -> bool:
        """Whether a response with no usage object raises (``True``) or yields zero counts."""
        return self._require_usage

    @require_usage.setter
    def require_usage(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise LLMEngineError(
                f"{type(self).__name__}.require_usage must be a bool, "
                f"got {type(value).__name__}."
            )
        self._require_usage = value

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
                self._runtime_parameters[key] = copy.deepcopy(value)

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
        1. ``messages`` must be a list; ``output_structure`` is optional.
        2. Normalize the messages, build the payload, call the provider.
        3. Extract result, token usage and model data from the response.
        4. Return ``{"result", "token_usage", "model_data"}``.

        ``LLMEngineError`` propagates; any other exception is raised as
        ``LLMEngineError`` chained from it.
        """
        try:
            messages = inputs.get("messages")
            if not isinstance(messages, list):
                raise LLMEngineError(
                    f"{type(self).__name__}.invoke: 'messages' input must be a list"
                )
            output_structure = inputs.get("output_structure")
            normalized = self._normalize_messages(messages)
            payload = self._build_provider_payload(normalized, output_structure)
            response = self._call_provider(payload)
            text, token_usage, model_data = self.extract(
                response, output_structure is not None
            )
            return {"result": text, "token_usage": token_usage, "model_data": model_data}
        except LLMEngineError:
            raise
        except Exception as exc:
            raise LLMEngineError(f"{type(self).__name__}.invoke: {exc}") from exc

    async def _async_invoke_impl(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async twin of ``_invoke_impl`` (awaits ``_call_provider_async``)."""
        try:
            messages = inputs.get("messages")
            if not isinstance(messages, list):
                raise LLMEngineError(
                    f"{type(self).__name__}.invoke: 'messages' input must be a list"
                )
            output_structure = inputs.get("output_structure")
            normalized = self._normalize_messages(messages)
            payload = self._build_provider_payload(normalized, output_structure)
            response = await self._call_provider_async(payload)
            text, token_usage, model_data = self.extract(
                response, output_structure is not None
            )
            return {"result": text, "token_usage": token_usage, "model_data": model_data}
        except LLMEngineError:
            raise
        except Exception as exc:
            raise LLMEngineError(f"{type(self).__name__}.invoke: {exc}") from exc

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def extract(
        self, response: Any, requested_structured: bool
    ) -> tuple[str | int | float | bool | list[Any] | dict[str, Any] | None, TokenUsage, LLMModelData]:
        """
        Extract the result, token usage and model data from one provider
        response.

        1. ``_extract_result(response, requested_structured)``; the value must
           be a JSON-shaped type.
        2. ``_extract_token_usage(response)``; must be a ``TokenUsage``.
        3. ``_get_model_data()``; must be ``LLMModelData``.
        4. Return the three, with a string result stripped.
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
        Validate ``messages`` and return ``[{"role", "content"}]`` with
        lowercased roles.

        1. The list must not be empty.
        2. Each entry must be a mapping with string ``role`` and ``content``.
        3. Each role (lowercased) must be ``system``, ``user`` or ``assistant``.
        4. At least one ``user`` message must be present.
        """
        prefix = f"{type(self).__name__}.invoke"
        if not messages:
            raise LLMEngineError(f"{prefix}: messages must not be empty")

        normalized: List[Dict[str, str]] = []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, Mapping):
                raise LLMEngineError(
                    f"{prefix}: message {idx} is not a mapping (got {type(msg)!r})"
                )
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise LLMEngineError(
                    f"{prefix}: each message must have 'role' and 'content' as strings"
                )
            role = role.lower()
            if role not in _VALID_ROLES:
                raise LLMEngineError(
                    f"{prefix}: message {idx} has unsupported role {role!r}; "
                    f"expected one of {sorted(_VALID_ROLES)}"
                )
            normalized.append({"role": role, "content": content})
        if not any(m["role"] == "user" for m in normalized):
            raise LLMEngineError(f"{prefix}: messages must include at least one user message")
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
        Receive normalized messages and the caller's raw ``output_structure``
        (``None`` for plain text); return the exact keyword arguments of the
        SDK call, so ``_call_provider`` only picks the client and calls it with
        ``**payload``. Build it from a snapshot of the runtime parameters and
        ``update`` the engine-owned fields (model, messages, the
        structured-output field) over it, so they always win.
        """
        raise NotImplementedError

    @abstractmethod
    def _call_provider(self, payload: Any) -> Any:
        """
        Receive the payload; perform one provider call and return the raw response.
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
        Receive a provider response; return the assistant's reply.

        ``requested_structured`` is the caller's own request
        (``output_structure is not None``), never re-derived from the response.

        - ``False``: return the plain text.
        - ``True``: ``json.loads`` the text (any JSON value); on a parse
          failure return the raw text rather than raising. A caller that
          requested structure and gets a ``str`` back must handle that itself.
        """
        raise NotImplementedError

    @abstractmethod
    def _extract_token_usage(self, response: Any) -> TokenUsage:
        """
        Receive a provider response; return a ``TokenUsage``-family record
        (never a raw provider usage object). When the response has no usage
        object at all, raise ``LLMEngineError`` unless ``require_usage`` is
        ``False``, in which case return an all-zero record.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_model_data(self) -> LLMModelData:
        """
        Return model identity data derived from engine configuration, not
        from a response.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------- #
    # Serialization
    # --------------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """
        Non-secret configuration snapshot for debugging / logging.

        ``runtime_parameters`` is deep-copied and redacted by key name
        (see ``_redact_secrets``); ``require_usage`` is included.
        """
        d = super().to_dict()
        d.update({
            "require_usage": self.require_usage,
            "runtime_parameters": _redact_secrets(copy.deepcopy(self._runtime_parameters)),
        })
        return d
