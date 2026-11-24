from __future__ import annotations

import json
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
    --------------------------------
    Structured output is requested via ``response_format = {"type":
    "json_schema", "json_schema": {"schema": ..., "name":
    "output_structure", "strict": True}}`` -- litellm's own OpenAI
    Chat-Completions dialect. A plain dict passed this way is used by
    litellm near-verbatim internally, so this engine supplies that envelope
    itself rather than relying on litellm to build it from a bare schema.

    Unlike every other engine in this package, this engine does not clean
    or prune the caller's schema at all. litellm itself performs the real
    per-destination-provider schema cleaning/transformation, confirmed by
    reading its installed source (not just its docs). Two examples,
    verified directly:

    - Anthropic: model-capability-gated between real native structured
      output (the same mechanism ``AnthropicEngine`` uses directly) and a
      tool-call-emulation fallback for older models without native
      support. It resolves ``$defs``/``$ref``, strips a specific set of
      numeric/length/array keywords (appending a human-readable constraint
      note to each affected field's ``description`` rather than silently
      dropping it), and auto-injects ``additionalProperties: false`` on
      object schemas that omit it.
    - Gemini/Vertex: model-capability-gated between the modern
      JSON-Schema-native dialect (the same one ``GeminiEngine`` targets)
      and a legacy OpenAPI-style fallback for older model generations.

    Notably, litellm's own Anthropic strip-list does **not** match the
    unsupported-keyword set AA's own dedicated ``AnthropicEngine`` found
    live (structured-generation Pass 4: ``minimum``, ``maximum``,
    ``multipleOf``, ``maxItems``): litellm additionally strips
    ``minLength``/``maxLength`` -- which AA's own live testing found the
    real Anthropic API actually supports -- and does not strip
    ``multipleOf`` at all -- which AA's own testing found needs stripping
    (as of Pass 8, neither engine prunes anything AA-side any more, but
    litellm's own downstream transformation still diverges from what the
    real Anthropic API needs). Routing an Anthropic model through this
    engine therefore behaves measurably differently than AA's own dedicated
    ``AnthropicEngine`` for the identical schema. This is litellm's own
    implementation detail, not a bug in either engine, and this pass makes
    no attempt to reconcile the two.

    Because litellm already owns this work, this engine adds no
    schema-level cleaning of its own -- there is nothing left for AA to
    add.

    ``"strict": True`` is a fixed literal in the ``response_format``
    envelope this engine builds -- there is no per-instance toggle (unlike
    ``OpenAIEngine``'s/``MistralEngine``'s mutable ``strict`` property).
    Confirmed live: both OpenAI and Anthropic reject an object schema whose
    ``additionalProperties`` is explicitly ``true`` (not merely omitted)
    under strict mode -- OpenAI requires it be exactly ``false``; litellm's
    own Anthropic auto-injection (see above) only fills the key in when it
    is *absent*, and deliberately does not override an explicit caller
    value, so an explicit ``true`` reaches Anthropic's API unmodified and
    is rejected there too. A schema that merely *omits*
    ``additionalProperties`` fares differently per destination (Anthropic
    gets it auto-set to ``false``; OpenAI still requires the caller to set
    it themselves). Gemini has no such requirement at all -- no
    strict-mode split exists for that dialect, matching ``GeminiEngine``'s
    own Pass 3 finding.

    There is no capability check: this engine does not gate on
    ``litellm.supports_response_schema``. A model/provider combination with no
    structured-output path fails natively in litellm and surfaces as
    ``LLMEngineError`` like any other provider error. ``output_structure`` is
    forwarded completely unmodified; litellm's own per-destination-provider
    transformation does all real cleaning downstream.
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
            Defaults to the class name; stored as given.
        description:
            Defaults to ``"LiteLLM provider-agnostic language model engine."``.
        client:
            ``None`` (call ``litellm.completion``) or a ``litellm.Router``
            (call ``router.completion``). Anything else raises; a Router
            together with ``kwargs`` raises (the Router owns the connection).
        **kwargs:
            Per-call litellm settings merged into every call: ``api_key``,
            ``base_url``, ``api_version``, ``timeout``, ``num_retries``, ...
            Kept private and never dumped (credentials). ``stream`` is
            rejected (streaming is unsupported). Credentials belong here (or
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

        # Streaming is unsupported
        if "stream" in kwargs:
            raise LLMEngineError("LiteLLMEngine: 'stream' is not supported (streaming is unsupported).")

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
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
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
        Return the first choice's message content, or ``""`` if empty. When
        `requested_structured` is True, the text is parsed via `json.loads`;
        a parse failure falls back to returning the raw text rather than
        raising, matching the base `LLMEngine` fallback contract
        (structured-generation Pass 3).

        Renamed from `_extract_text` (structured-generation Pass 1
        convention); gains `requested_structured` per the Pass 3 base
        contract.
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
        ``LiteLLMTokenUsage``. ``total_tokens`` is explicitly derived as
        ``input_tokens + generated_tokens`` rather than trusted from the raw
        ``usage.total_tokens`` field — same defensive posture as
        ``AnthropicEngine``. ``prompt_tokens_details``/
        ``completion_tokens_details`` are both genuinely ``Optional`` on
        litellm's ``Usage`` object (e.g. absent on OpenAI/Mistral's generic
        passthrough path when the raw provider JSON lacks these keys).

        ``response.usage`` itself is only conditionally set by litellm's own
        dict-to-response conversion (populated only when the raw provider
        JSON included a non-``None`` ``"usage"`` key) — unlike
        ``AnthropicEngine``, which reads a raw SDK object where ``usage`` is
        a required field. Raises ``LLMEngineError`` rather than crashing with
        a bare ``AttributeError`` if a provider response omits it.
        """
        u = getattr(response, "usage", None)
        if u is None:
            raise LLMEngineError(
                "LiteLLMEngine._extract_token_usage: provider response did "
                "not include usage data"
            )
        inp = u.prompt_tokens
        gen = u.completion_tokens

        # `PromptTokensDetailsWrapper.__init__` (litellm/types/utils.py)
        # `del`s `cache_creation_tokens` from the instance entirely whenever
        # it's `None` post-construction — plain attribute access raises
        # `AttributeError` for any response with no cache-creation activity
        # (e.g. OpenAI, or Anthropic without prompt caching). `cached_tokens`
        # is inherited from the base `PromptTokensDetails` and is not subject
        # to this deletion, so it stays safe as plain attribute access.
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

        # Plain subtraction rather than litellm's own
        # completion_tokens_details.text_tokens — verified via source that
        # text_tokens is essentially never populated on the dominant
        # response-conversion path (raw provider usage dicts nest
        # reasoning_tokens under completion_tokens_details rather than
        # passing it as a top-level Usage.__init__ kwarg, which is what
        # actually triggers litellm's own text_tokens auto-calculation).
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
        # litellm prints a "Provider List" banner on failure unless this is on.
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
