from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional

try:
    import litellm
    _SDK_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:
    litellm = None  # type: ignore[assignment]
    _SDK_IMPORT_ERROR = _exc

from .base import LLMEngine
from ..exceptions import LLMEngineError
from ..models.results.llm import LiteLLMTokenUsage, RemoteLLMModelData
from ..utils.core import format_missing_dependency

__all__ = ["LiteLLMEngine"]

# Serializes the process-wide ``litellm.suppress_debug_info`` toggle around provider lookups.
_PROVIDER_LOOKUP_LOCK = threading.Lock()


class LiteLLMEngine(LLMEngine):
    """
    Provider-agnostic adapter over litellm's ``completion``/``acompletion``.

    No SDK client
    -------------
    litellm has no client object: ``litellm.completion``/``acompletion`` are
    module functions. The optional ``client`` is a ``litellm.Router`` (a load
    balancer over a ``model_list``); with one, ``router.completion`` /
    ``router.acompletion`` are called, without one the module functions are.
    A Router owns its connection settings, so combining it with ``kwargs``
    raises. ``kwargs`` are per-call litellm settings (``api_key``,
    ``base_url``, ``api_version``, ``timeout``, ``num_retries``, ...) merged
    into every call; they are kept private and never dumped (credentials).

    Model identity
    --------------
    ``model`` is litellm's own model string (``"openai/gpt-4o-mini"``,
    ``"ollama/llama3"``, ``"litellm_proxy/alias"``) or, with a Router, a
    group name from its ``model_list``. It is a plain mutable attribute, sent
    as ``model=`` on every call, so reassigning it takes effect on the next
    call. The provider reported in model data is derived from it through
    ``litellm.get_llm_provider`` (``"unknown"`` when litellm cannot resolve
    it, as for a Router group name).

    Runtime parameters
    ------------------
    Keys are litellm's own ``completion`` keyword arguments (``temperature``,
    ``max_tokens``, ``drop_params``, ``reasoning_effort``, ...), forwarded
    verbatim; nothing is set by default. ``drop_params`` is an ordinary
    runtime key: litellm raises ``UnsupportedParamsError`` for a knob the
    target does not support unless ``drop_params=True`` is set. Streaming is
    unsupported.

    Structured-output schema policy
    -------------------------------
    ``output_structure`` is wrapped in a strict ``json_schema``
    ``response_format`` envelope (``strict`` is fixed ``True``) and forwarded
    unmodified; litellm does any per-provider schema transformation and the
    destination provider rejects what it does not accept natively. There is
    no capability gate.

    ``require_usage`` (base) controls a response with no ``usage``.
    """

    _RESERVED_RUNTIME_KEYS = frozenset({"model", "messages", "response_format", "stream"})

    def __init__(
        self,
        model: str,
        name: str | None = None,
        description: str | None = None,
        client: "litellm.Router | None" = None,
        **kwargs: Any,
    ) -> None:
        """
        model:
            litellm's own model string (``"openai/gpt-4o-mini"``,
            ``"ollama/llama3"``, ``"litellm_proxy/alias"``) or, with a Router,
            a group name from its ``model_list``.
        name:
            Defaults to the class name.
        description:
            Defaults to ``"LiteLLM provider-agnostic language model engine."``.
        client:
            ``None`` (call ``litellm.completion``) or a ``litellm.Router``
            (call ``router.completion``). Anything else raises; a Router
            together with ``kwargs`` raises (the Router owns the connection).
        **kwargs:
            Per-call litellm settings merged into every call: ``api_key``,
            ``base_url``, ``api_version``, ``timeout``, ``num_retries``, ...
            Kept private and never dumped (credentials). The
            engine-owned keys (``model``, ``messages``, ``response_format``,
            ``stream``) are rejected. Credentials belong here (or
            on a Router), not in runtime parameters, which ``to_dict``
            redacts by key name (key, token, secret, auth, password,
            credential, header) as a safety net.

        1. SDK presence check.
        2. ``client`` must be ``None`` or a ``litellm.Router``.
        3. A Router together with ``kwargs`` raises.
        4. Base init, then store the client, kwargs and model.
        """
        # 1. SDK presence check
        if litellm is None:
            raise LLMEngineError(
                "LiteLLMEngine "
                + format_missing_dependency("litellm", "litellm", _SDK_IMPORT_ERROR)
            ) from _SDK_IMPORT_ERROR

        # 2. Client type
        if client is not None and not isinstance(client, litellm.Router):
            raise LLMEngineError(
                f"LiteLLMEngine: 'client' must be a litellm.Router, got {type(client).__name__}."
            )

        # 3. One connection source
        if client is not None and kwargs:
            raise LLMEngineError(
                "LiteLLMEngine: pass either 'client' or call kwargs, not both."
            )

        # Engine-owned keys (and unsupported streaming) cannot be call kwargs
        reserved = sorted(set(kwargs) & self._RESERVED_RUNTIME_KEYS)
        if reserved:
            raise LLMEngineError(
                f"LiteLLMEngine: {reserved} are managed by the engine and cannot be "
                "passed as call kwargs (streaming is unsupported)."
            )

        # 4. Base init and state
        super().__init__(
            name=name,
            description=description or "LiteLLM provider-agnostic language model engine.",
        )
        self._client = client
        self._kwargs = dict(kwargs)
        self._provider_cache: tuple[str, str] | None = None
        self.model = model

    # ------------------------------------------------------------------ #
    # Provider call
    # ------------------------------------------------------------------ #

    def _call_provider(self, payload: Any) -> Any:
        """Synchronous ``completion`` on the Router when given, else on litellm."""
        target = self._client if self._client is not None else litellm
        return target.completion(**payload)

    async def _call_provider_async(self, payload: Any) -> Any:
        """Native async ``acompletion`` on the Router when given, else on litellm."""
        target = self._client if self._client is not None else litellm
        return await target.acompletion(**payload)

    # ------------------------------------------------------------------ #
    # Payload construction
    # ------------------------------------------------------------------ #

    def _build_provider_payload(
        self,
        messages: List[Dict[str, Any]],
        output_structure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        The exact ``completion``/``acompletion`` kwargs, layered so the later
        layer wins.

        1. Per-call ``kwargs``, then the runtime-parameter snapshot over them.
        2. Engine-owned ``model`` and ``messages`` over both.
        3. Structured call: the strict ``json_schema`` ``response_format``
           envelope around the caller's schema, forwarded unmodified (no
           capability gate).
        """
        payload: dict[str, Any] = dict(self._kwargs)
        payload.update(self._runtime_parameters)
        payload.update(
            model=self.model,
            messages=list(messages),
        )
        if output_structure is not None:
            payload.update(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "schema": output_structure,
                        "name": "output_structure",
                        "strict": True,
                    },
                }
            )
        return payload

    # ------------------------------------------------------------------ #
    # Response extraction
    # ------------------------------------------------------------------ #

    def _extract_result(
        self, response: Any, requested_structured: bool
    ) -> str | int | float | bool | list[Any] | dict[str, Any] | None:
        """
        Return the first choice's message content (``""`` if empty); on a
        structured call, its ``json.loads`` value (raw text on a parse failure).
        """
        text = response.choices[0].message.content or ""
        if not requested_structured:
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _extract_token_usage(self, response: Any) -> LiteLLMTokenUsage:
        """
        Map ``response.usage`` (litellm's normalized ``Usage``) to
        ``LiteLLMTokenUsage``. ``total_tokens`` is derived as
        ``input_tokens + generated_tokens``; the prompt and completion detail
        objects are optional. A response with no ``usage`` raises unless
        ``require_usage`` is ``False``, which yields an all-zero record.
        """
        u = getattr(response, "usage", None)
        if u is None:
            if not self.require_usage:
                return LiteLLMTokenUsage(
                    input_tokens=0, generated_tokens=0, total_tokens=0, response_tokens=0
                )
            raise LLMEngineError(
                "LiteLLMEngine._extract_token_usage: provider response did "
                "not include usage data"
            )
        inp = u.prompt_tokens
        gen = u.completion_tokens

        # litellm's `PromptTokensDetailsWrapper` deletes `cache_creation_tokens`
        # when it is `None`, so read it with `getattr`; `cached_tokens` is
        # always present.
        prompt_details = u.prompt_tokens_details
        cached_tokens = (
            prompt_details.cached_tokens if prompt_details is not None else None
        )
        cache_creation_tokens = (
            getattr(prompt_details, "cache_creation_tokens", None)
            if prompt_details is not None
            else None
        )

        completion_details = u.completion_tokens_details
        reasoning_tokens = (
            completion_details.reasoning_tokens
            if completion_details is not None
            else None
        )

        # Plain subtraction: litellm's `completion_tokens_details.text_tokens`
        # is rarely populated.
        response_tokens = gen - (reasoning_tokens or 0)

        return LiteLLMTokenUsage(
            input_tokens=inp,
            generated_tokens=gen,
            total_tokens=inp + gen,
            response_tokens=response_tokens,
            cached_tokens=cached_tokens,
            cache_creation_tokens=cache_creation_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def _get_model_data(self) -> RemoteLLMModelData:
        """Provider derived from ``model`` by litellm (``"unknown"`` when unresolvable).

        The lookup is cached per ``model`` value; a changed ``model`` recomputes.
        """
        cache = self._provider_cache
        if cache is not None and cache[0] == self.model:
            return RemoteLLMModelData(provider=cache[1], model_name=self.model)
        # litellm prints a "Provider List" banner on failure unless this
        # process-wide flag is on; the lock keeps concurrent lookups from
        # restoring a stale value.
        with _PROVIDER_LOOKUP_LOCK:
            previous = litellm.suppress_debug_info
            litellm.suppress_debug_info = True
            try:
                provider = litellm.get_llm_provider(self.model)[1]
            except Exception:
                provider = "unknown"
            finally:
                litellm.suppress_debug_info = previous
        self._provider_cache = (self.model, provider)
        return RemoteLLMModelData(provider=provider, model_name=self.model)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Base fields plus ``model``. No kwargs (credentials), no client."""
        d = super().to_dict()
        d.update({"model": self.model})
        return d
