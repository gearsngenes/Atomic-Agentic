from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Mapping, Optional

try:
    import litellm
    from litellm.exceptions import (
        RateLimitError,
        Timeout,
        APIConnectionError,
        ServiceUnavailableError,
        InternalServerError,
        BadGatewayError,
    )
except ImportError:
    litellm = None  # type: ignore[assignment]
    RateLimitError = None
    Timeout = None
    APIConnectionError = None
    ServiceUnavailableError = None
    InternalServerError = None
    BadGatewayError = None

from .base import LLMEngine
from ..constants.llm import (
    ILLEGAL_ATTACHMENT_EXTS,
    ENGINE_ILLEGAL_MIME_PREFIXES,
    LITELLM_IMAGE_EXTS,
    LITELLM_DOCUMENT_EXTS,
    LITELLM_ALLOWED_EXTS,
)
from ..exceptions import LLMEngineError
from ..models.results.llm import LiteLLMTokenUsage, RemoteLLMModelData
from ..utils.llm import validate_attachment_path

__all__ = ["LiteLLMEngine"]

_MIME_MAP: dict[str, str] = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".pdf":  "application/pdf",
}


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

    Retry ownership
    ----------------
    litellm's own ``max_retries`` kwarg is never forwarded — ``LLMEngine``'s
    base ``_call_with_retries``/``_call_with_retries_async`` loop is the sole
    retry authority.

    File policy
    ------------
    Inline-only attachments via litellm's generic content-block shapes:

    - Images (PNG, JPG, JPEG, GIF, WEBP) → base64 ``image_url`` blocks.
    - PDFs → base64 ``file`` blocks. **Raises when ``self.provider ==
      "mistral"``** — litellm's Mistral transformation requires an
      already-uploaded ``file_id`` and strips inline ``file_data``, and
      ``litellm.create_file()``'s own provider allowlist excludes ``mistral``
      entirely. This check is based on the ``provider`` constructor value —
      a proxy-routed Mistral model reached via an alias
      (``provider="litellm_proxy"``) is not detected by this guard; the
      underlying litellm proxy call fails on its own instead.
    - Text / code → plain ``text`` blocks (no base64; not gated by provider —
      the Mistral gap is specific to inline binary/PDF ``file`` blocks).

    Attachment blocks are appended to the last user-turn ``content`` in
    ``_build_provider_payload``. ``_on_detach`` is a no-op. Provider
    translation of these generic blocks into each provider's native request
    shape is litellm's own responsibility, not this engine's.

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

    Notably, litellm's own Anthropic strip-list does **not** match this
    package's own ``AnthropicEngine.structure_omitted_keys``
    (``ANTHROPIC_STRUCTURE_OMITTED_KEYS``, structured-generation Pass 4):
    litellm additionally strips ``minLength``/``maxLength`` -- which AA's
    own live testing found the real Anthropic API actually supports -- and
    does not strip ``multipleOf`` at all -- which AA's own testing found
    needs stripping. Routing an Anthropic model through this engine
    therefore behaves measurably differently than AA's own dedicated
    ``AnthropicEngine`` for the identical schema. This is litellm's own
    implementation detail, not a bug in either engine, and this pass makes
    no attempt to reconcile the two.

    Because litellm already owns this work, ``structure_permitted_keys``/
    ``structure_omitted_keys`` stay the inherited base ``LLMEngine`` no-op
    defaults -- there is nothing left for AA to add.

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

    ``_clean_structure_template`` **is** overridden on this class, but for
    a different reason than ``MistralEngine`` overrides the same hook.
    Mistral's override prunes specific keywords because its omission
    policy depends on instance state (its mutable ``strict`` property).
    This engine's override prunes nothing at all -- it calls
    ``litellm.supports_response_schema(model=self.model,
    custom_llm_provider=self.provider)`` and raises outright when the
    target model/provider combination has no structured-output path,
    rather than silently degrading to plain text. That raise is
    unconditional and does not consult ``drop_params``: ``drop_params``
    governs individual generation knobs being dropped for cross-provider
    convenience, not an entire feature the caller explicitly requested
    being silently unavailable.
    """

    def __init__(
        self,
        model: str,
        provider: str = "litellm_proxy",
        name: str | None = None,
        namespace: str = "llm",
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
        inline_cutoff_chars: int = 200_000,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        *,
        timeout_seconds: float = 600.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 8.0,
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
        namespace:
            Grouping label; defaults to ``"llm"``.
        description:
            Human-readable description for this engine instance.
        temperature, top_p, max_tokens, max_completion_tokens, stop, seed,
        reasoning_effort, thinking:
            Optional generation knobs; omitted from the payload when ``None``.
        drop_params:
            Forwarded as ``drop_params=`` on every call. See class docstring.
        inline_cutoff_chars:
            Maximum characters to inline from text/code attachments.
        api_key, base_url, api_version:
            Per-call request-level fields (no client object exists to hold
            these); omitted when ``None``.
        timeout_seconds:
            Per-call timeout, forwarded as ``timeout=`` on every call.
        max_retries, retry_backoff_base, retry_backoff_max:
            Shared ``LLMEngine`` retry/backoff configuration. litellm's own
            ``max_retries`` kwarg is never forwarded — see class docstring.
        **litellm_kwargs:
            Escape hatch forwarded verbatim to ``completion``/``acompletion``,
            applied last so it can override any explicit field above.
        """
        # 1. Name sanitization + super init
        sanitized = (name or f"{provider}_{model}").replace(":", "_").replace(
            "-", "_").replace(" ", "_").replace(".", "_")
        super().__init__(
            name=sanitized,
            namespace=namespace,
            description=description or "LiteLLM provider-agnostic language model engine.",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        # 2. SDK presence check
        if litellm is None:
            raise LLMEngineError(
                "LiteLLMEngine requires the `litellm` package: pip install litellm"
            )

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
        self._inline_cutoff_chars = int(inline_cutoff_chars)
        self._litellm_kwargs = dict(litellm_kwargs)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def inline_cutoff_chars(self) -> int:
        """Max characters inlined from text attachments; fixed at construction."""
        return self._inline_cutoff_chars

    # ------------------------------------------------------------------ #
    # Provider call
    # ------------------------------------------------------------------ #

    def _call_provider(self, payload: Any) -> Any:
        """Synchronous ``litellm.completion`` call — no client, no isinstance routing."""
        return litellm.completion(**payload)

    async def _call_provider_async(self, payload: Any) -> Any:
        """Native async ``litellm.acompletion`` call."""
        return await litellm.acompletion(**payload)

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        """
        Retry on litellm's own named transient-error classes, referenced
        directly (not via the fact that they happen to subclass openai's
        exceptions today — that's an implementation detail, not a contract).
        """
        if attempt > self._max_retries:
            return False
        return isinstance(exc, (
            RateLimitError,
            Timeout,
            APIConnectionError,
            ServiceUnavailableError,
            InternalServerError,
            BadGatewayError,
        ))

    # ------------------------------------------------------------------ #
    # Structured-output capability gating
    # ------------------------------------------------------------------ #

    def _clean_structure_template(
        self, output_structure: Mapping[str, Any]
    ) -> dict[str, Any]:
        """
        Override the base cleaning hook for capability-gating, not
        keyword-level pruning (contrast `MistralEngine`'s own override of
        this same hook -- see class docstring's "Structured-output schema
        policy").

        Raises `LLMEngineError` naming `self.model`/`self.provider` when
        litellm reports no structured-output support for this combination
        -- unconditionally, regardless of `self.drop_params`. Otherwise
        returns `output_structure` unchanged; litellm's own
        per-destination-provider transformation does all real cleaning
        downstream.
        """
        if not litellm.supports_response_schema(
            model=self.model, custom_llm_provider=self.provider
        ):
            raise LLMEngineError(
                f"LiteLLMEngine: structured output ('output_structure') is "
                f"not supported for model={self.model!r}, "
                f"provider={self.provider!r}."
            )
        return dict(output_structure)

    # ------------------------------------------------------------------ #
    # Payload construction
    # ------------------------------------------------------------------ #

    def _build_provider_payload(
        self,
        messages: List[Dict[str, Any]],
        attachments: Dict[str, Any],
        output_structure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build the ``completion``/``acompletion`` payload from normalized
        messages, the current attachments snapshot, and (optionally) a
        structured-output schema.

        Lifecycle: (1) convert messages to litellm/OpenAI-shaped list, string
        content passed through unchanged where no attachments apply; (2)
        append attachment blocks to the last user turn's content (converting
        its content to a list first); (3) assemble the final payload dict
        with the composed ``model=`` string and all conditional fields,
        including ``response_format`` when ``output_structure`` is given
        (already capability-checked and passed through unchanged by
        ``_clean_structure_template`` -- see class docstring's
        "Structured-output schema policy").
        """
        # 1. Convert messages — content stays a plain string unless it is the
        #    last user turn and attachments exist below.
        litellm_msgs: list[dict[str, Any]] = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        # 2. Append attachment blocks to the last user turn
        blocks = list(attachments.values())
        if blocks:
            idx = next(
                (i for i in range(len(litellm_msgs) - 1, -1, -1)
                 if litellm_msgs[i]["role"] == "user"),
                None,
            )
            if idx is not None:
                content = litellm_msgs[idx]["content"]
                if not isinstance(content, list):
                    content = [{"type": "text", "text": content}]
                litellm_msgs[idx]["content"] = content + blocks
            else:
                litellm_msgs.append({"role": "user", "content": blocks})

        # 3. Assemble payload — model/messages/drop_params/timeout always
        #    present; api_key/base_url/api_version and generation knobs
        #    conditional on non-None.
        payload: dict[str, Any] = {
            "model": f"{self.provider}/{self.model}",
            "messages": litellm_msgs,
            "drop_params": self.drop_params,
            "timeout": self.timeout_seconds,
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
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "schema": output_structure,
                    "name": "output_structure",
                    "strict": True,
                },
            }

        # 4. Escape hatch — applied last; a key colliding with an explicit
        #    field above overrides it (caller error, not guarded against).
        payload.update(self._litellm_kwargs)
        return payload

    # ------------------------------------------------------------------ #
    # Response extraction
    # ------------------------------------------------------------------ #

    def _extract_result(
        self, response: Any, requested_structured: bool
    ) -> str | list[Any] | dict[str, Any]:
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
    # Attachment pipeline (inline only)
    # ------------------------------------------------------------------ #

    def _validate_attachment_path(self, path: str) -> None:
        """Validate ``path`` against the LiteLLM allow-list and shared blacklists; raises ``LLMEngineError`` on rejection."""
        try:
            validate_attachment_path(
                path,
                illegal_exts=ILLEGAL_ATTACHMENT_EXTS,
                allowed_exts=LITELLM_ALLOWED_EXTS,
                illegal_mime_prefixes=ENGINE_ILLEGAL_MIME_PREFIXES,
            )
        except ValueError as exc:
            raise LLMEngineError(str(exc)) from exc

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        """
        Build an inline litellm-generic content block for ``path``.

        Images → base64 ``image_url`` block; PDFs → base64 ``file`` block
        (raises ``LLMEngineError`` when ``self.provider == "mistral"`` — a
        documented gap, not a workaround); text/code → plain UTF-8 text
        block, truncated at ``inline_cutoff_chars``.
        """
        try:
            ext = os.path.splitext(path)[1].lower()

            if ext in LITELLM_IMAGE_EXTS:
                with open(path, "rb") as fh:
                    data = base64.b64encode(fh.read()).decode()
                return {
                    "type": "image_url",
                    "image_url": {"url": f"data:{_MIME_MAP[ext]};base64,{data}"},
                }

            if ext in LITELLM_DOCUMENT_EXTS:
                if self.provider == "mistral":
                    raise LLMEngineError(
                        "LiteLLMEngine: inline PDF/document attachments are not "
                        "supported for provider='mistral' — litellm's Mistral "
                        "transformation requires an already-uploaded file_id and "
                        "strips inline file_data; litellm.create_file() does not "
                        "support the mistral provider either."
                    )
                with open(path, "rb") as fh:
                    data = base64.b64encode(fh.read()).decode()
                return {
                    "type": "file",
                    "file": {"file_data": f"data:{_MIME_MAP[ext]};base64,{data}"},
                }

            # Text / code — plain text block, not gated by provider
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            if len(text) > self._inline_cutoff_chars:
                text = text[: self._inline_cutoff_chars] + "\n…[truncated]\n"
            return {"type": "text", "text": text}
        except LLMEngineError:
            raise
        except Exception as exc:
            raise LLMEngineError(
                f"LiteLLMEngine._prepare_attachment failed for {path!r}"
            ) from exc

    def _on_detach(self, meta: Any) -> None:
        """No-op — inline attachments leave no server-side state to clean up."""

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
            "inline_cutoff_chars": self._inline_cutoff_chars,
        })
        return d
