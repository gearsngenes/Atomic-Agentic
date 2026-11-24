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

    No client object
    -----------------
    Unlike every other engine in this family, ``LiteLLMEngine`` holds no
    injectable SDK client — ``litellm.completion``/``acompletion`` are bare
    module functions, not methods on a constructed object. ``api_key``,
    ``base_url``, and ``api_version`` are resent as request-level payload
    fields on every call rather than baked into a client at construction.

    Model identity
    --------------
    ``provider`` (default ``"litellm_proxy"``) and ``model`` are combined at
    *call time* into ``f"{self.provider}/{self.model}"`` and sent as
    litellm's ``model=`` kwarg. Both attributes are plain mutable fields —
    same "model is intentionally mutable" posture as every other engine
    (v2.0.0a10) — and the composed string is recomputed on every call rather
    than cached at construction, so reassigning either attribute after
    construction takes effect on the next call.

    ``drop_params``
    ----------------
    Forwarded as ``drop_params=`` on every call. Defaults to ``False``: by
    default litellm raises ``UnsupportedParamsError`` for any non-``None``
    generation knob unsupported by the target provider/model, matching this
    codebase's fail-loud discipline (natural exceptions surface rather than
    being silently absorbed). Set ``True`` for deliberate cross-provider use
    with one shared knob configuration.

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

    The capability check (``litellm.supports_response_schema(...)``) lives
    inline at the top of ``_build_provider_payload``'s
    ``output_structure is not None`` branch, not a separate hook override
    (structured-generation Pass 8 removed the base
    ``_clean_structure_template`` hook entirely -- ``MistralEngine``'s own
    pruning override is gone too, since it was the only other consumer).
    It raises ``LLMEngineError`` naming ``self.model``/``self.provider``
    when litellm reports no structured-output support for this
    combination -- unconditionally, regardless of ``self.drop_params``
    (``drop_params`` governs individual generation knobs being dropped for
    cross-provider convenience, not an entire feature the caller explicitly
    requested being silently unavailable). Otherwise ``output_structure``
    is forwarded completely unmodified; litellm's own
    per-destination-provider transformation does all real cleaning
    downstream.
    """

    def __init__(
        self,
        model: str,
        provider: str = "litellm_proxy",
        name: str | None = None,
        description: str = "",
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        max_completion_tokens: int | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
        reasoning_effort: str | None = None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool = False,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        *,
        timeout_seconds: float = 600.0,
        **litellm_kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Model identifier, combined with ``provider`` at call time into
            ``f"{provider}/{model}"``.
        provider:
            Provider prefix sent to litellm; defaults to ``"litellm_proxy"``.
        name:
            Optional human-friendly engine name; defaults to
            ``{provider}_{model}`` with non-identifier characters replaced by
            underscores.
        description:
            Human-readable description for this engine instance.
        temperature, top_p, max_tokens, max_completion_tokens, stop, seed,
        reasoning_effort, thinking:
            Optional generation knobs; omitted from the payload when ``None``.
        drop_params:
            Forwarded as ``drop_params=`` on every call. See class docstring.
        api_key, base_url, api_version:
            Per-call request-level fields (no client object exists to hold
            these); omitted when ``None``.
        timeout_seconds:
            Per-call timeout, forwarded as ``timeout=`` on every call.
        **litellm_kwargs:
            Escape hatch forwarded verbatim to ``completion``/``acompletion``,
            applied last so it can override any explicit field above. litellm's
            own retry settings (``num_retries``, ``max_retries``) go here.
        """
        # 1. Name sanitization + super init
        sanitized = (name or f"{provider}_{model}").replace(":", "_").replace(
            "-", "_").replace(" ", "_").replace(".", "_")
        super().__init__(
            name=sanitized,
            description=description or "LiteLLM provider-agnostic language model engine.",
        )
        # Interim: the base no longer owns a timeout; deleted in Pass 9.
        self._timeout_seconds = float(timeout_seconds)

        # 2. SDK presence check
        if litellm is None:
            raise LLMEngineError(
                "LiteLLMEngine "
                + format_missing_dependency("litellm", "litellm", _SDK_IMPORT_ERROR)
            ) from _SDK_IMPORT_ERROR

        # 3. Store request-level params (no client construction step)
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.stop = stop
        self.seed = seed
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.drop_params = drop_params
        self.api_key = api_key
        self.base_url = base_url
        self.api_version = api_version
        self._litellm_kwargs = dict(litellm_kwargs)

    # ------------------------------------------------------------------ #
    # Provider call
    # ------------------------------------------------------------------ #

    def _call_provider(self, payload: Any) -> Any:
        """Synchronous ``litellm.completion`` call — no client, no isinstance routing."""
        return litellm.completion(**payload)

    async def _call_provider_async(self, payload: Any) -> Any:
        """Native async ``litellm.acompletion`` call."""
        return await litellm.acompletion(**payload)

    # ------------------------------------------------------------------ #
    # Payload construction
    # ------------------------------------------------------------------ #

    def _build_provider_payload(
        self,
        messages: List[Dict[str, Any]],
        output_structure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the ``completion``/``acompletion`` payload from normalized
        messages and (optionally) a structured-output schema.

        Lifecycle: (1) convert messages to a litellm/OpenAI-shaped list,
        string content passed through unchanged; (2) assemble the final
        payload dict with the composed ``model=`` string and all conditional
        fields; (3) when ``output_structure`` is given, check
        ``litellm.supports_response_schema(...)`` inline (raises
        ``LLMEngineError`` if unsupported -- see class docstring's
        "Structured-output schema policy") and set ``response_format`` from
        the caller's schema, forwarded unmodified; (4) apply the
        ``litellm_kwargs`` escape hatch last.
        """
        # 1. Convert messages — content stays a plain string.
        litellm_msgs: list[dict[str, Any]] = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        # 2. Assemble payload — model/messages/drop_params/timeout always
        #    present; api_key/base_url/api_version and generation knobs
        #    conditional on non-None.
        payload: dict[str, Any] = {
            "model": f"{self.provider}/{self.model}",
            "messages": litellm_msgs,
            "drop_params": self.drop_params,
            "timeout": self._timeout_seconds,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.max_completion_tokens is not None:
            payload["max_completion_tokens"] = self.max_completion_tokens
        if self.stop is not None:
            payload["stop"] = self.stop
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        if self.api_key is not None:
            payload["api_key"] = self.api_key
        if self.base_url is not None:
            payload["base_url"] = self.base_url
        if self.api_version is not None:
            payload["api_version"] = self.api_version
        if output_structure is not None:
            if not litellm.supports_response_schema(
                model=self.model, custom_llm_provider=self.provider
            ):
                raise LLMEngineError(
                    f"LiteLLMEngine: structured output ('output_structure') is "
                    f"not supported for model={self.model!r}, "
                    f"provider={self.provider!r}."
                )
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": output_structure,
                    "name": "output_structure",
                    "strict": True,
                },
            }

        # 3. Escape hatch — applied last; a key colliding with an explicit
        #    field above overrides it (caller error, not guarded against).
        payload.update(self._litellm_kwargs)
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
        """Return configured provider/model identity for this engine."""
        return RemoteLLMModelData(provider=self.provider, model_name=self.model)

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "stop": self.stop,
            "seed": self.seed,
            "reasoning_effort": self.reasoning_effort,
            "thinking": self.thinking,
            "drop_params": self.drop_params,
            "base_url": self.base_url,
            "api_version": self.api_version,
        })
        return d
