from __future__ import annotations

import inspect
import json

try:
    from llama_cpp import Llama
    _LLAMA_IMPORT_ERROR: ImportError | None = None
except ImportError as _exc:
    Llama = None
    _LLAMA_IMPORT_ERROR = _exc
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from .base import LLMEngine
from ..exceptions import LLMEngineError
from ..utils.core import format_missing_dependency
from ..models.results.llm import (
    TokenUsage,
    LlamaCppTokenUsage,
    LLMModelData,
    LlamaCppModelData
)

__all__ = ["LlamaCppEngine"]

# ── LLAMA.CPP (local; no remote file store) ────────────────────────────────────
class LlamaCppEngine(LLMEngine):
    """
    Local llama.cpp adapter using ``llama-cpp-python``.

    This engine wraps a local GGUF/GGML model via ``llama_cpp.Llama`` and plugs
    into the shared ``LLMEngine`` template.

    Model source
    ------------
    Exactly one of three sources:

    - ``model``: a path to a local GGUF/GGML file, loaded through
      ``Llama(model_path=...)`` with ``client_kwargs`` forwarded to ``Llama``.
    - ``llm``: a preloaded ``Llama`` instance, used as is (combining it with
      ``client_kwargs`` raises).
    - ``LlamaCppEngine.from_pretrained(repo_id, filename, ...)``: a shallow
      wrapper over ``Llama.from_pretrained`` that downloads (or reuses from
      the cache) a GGUF from the Hugging Face Hub and wraps the result.

    ``self.model_path`` is the concrete local path of the loaded model, read
    from ``llm.model_path`` after construction.

    Runtime parameters and configuration
    ------------------------------------
    Runtime-parameter keys are ``create_chat_completion`` parameters under
    llama.cpp's own names, validated against its signature when set
    (``_validate_runtime_keys``). Nothing is set by default, so Llama's own
    defaults apply. Load-time settings (``n_ctx``, ``n_gpu_layers``,
    ``chat_format``, ...) go through ``client_kwargs``; call-time ones
    (``temperature``, ``max_tokens``, ...) go through
    ``update_runtime_parameters``. ``seed`` is one of each: the load-time
    seed is the model's sampler seed, the call-time one overrides it for a
    call. Streaming is unsupported.

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

    3) Attachments are not supported; this engine has no ``attach`` API.

    Token usage
    -----------
    ``_extract_token_usage`` maps the OpenAI-compatible response
    ``response["usage"]`` dictionary into ``LlamaCppTokenUsage`` using prompt,
    completion, and total token counts.

    Model data
    ----------
    ``_get_model_data`` returns the concrete local model path loaded by
    llama.cpp.

    Structured-output schema policy
    --------------------------------
    Structured output is requested via ``response_format = {"type":
    "json_object", "schema": ...}`` on ``create_chat_completion``. Unlike
    every remote engine in this package, enforcement happens entirely
    locally: llama-cpp-python's own bundled grammar converter
    (``llama_cpp.llama_grammar`` -- a separate Python port, not the same
    codebase as upstream ``ggml-org/llama.cpp``'s own converter or server)
    compiles the schema into a GBNF grammar and passes it to the underlying
    completion call as a distinct ``grammar`` sampling parameter. That
    grammar masks every token the model is allowed to emit at each
    generation step, so structural conformance is a property of the
    installed ``llama-cpp-python`` version, not of the loaded model's own
    training -- confirmed by direct testing across unrelated GGUF models.

    Enforced keywords (verified against the installed converter's source,
    not just its docs): ``type``, ``enum``, ``const``, ``properties``,
    ``required``, ``additionalProperties``, ``items``, ``prefixItems``,
    ``minItems``, ``maxItems``, ``minLength``, ``maxLength``, ``pattern``,
    ``oneOf``, ``anyOf``, ``allOf``, ``$ref`` (and ``$defs`` via ref
    resolution), and ``format`` restricted to ``uuid``/``date``/``time``/
    ``date-time``. Decorative-only keywords -- accepted without error but
    never enforced -- are ``title``, ``description``, ``examples``,
    ``default``, and, notably, ``minimum``/``maximum``/
    ``exclusiveMinimum``/``exclusiveMaximum``. That last group is a real,
    live-verified limitation of the installed dependency, not a
    documentation gap: the converter's numeric-type handling carries a
    literal ``# TODO: support minimum, maximum, exclusiveMinimum,
    exclusiveMaximum`` and silently ignores those keywords outright. A
    caller who relies on a numeric bound being enforced gets no error
    telling them it wasn't.

    This engine needs no override of any kind -- every engine now forwards
    ``output_structure`` unmodified by default (structured-generation
    Pass 8 removed AA's own schema-pruning mechanism entirely).

    The compiled grammar constrains sampling only; it is never woven into
    the prompt or message content the model actually reads. A caller who
    wants the model to understand what a field is *for* -- not just its
    name and type -- must still describe that intent in the prompt or
    system message themselves. Confirmed directly: a prompt with no mention
    of the requested schema at all still produces structurally valid output
    against that schema, but with semantically arbitrary field values. This
    is a known, documented limitation for this pass, not mitigated here.
    """

    _RESERVED_RUNTIME_KEYS = frozenset({"messages", "response_format", "stream"})

    def __init__(
            self,
            model: str | None = None,
            name: str | None = None,
            description: str | None = None,
            llm: Llama | None = None,
            **client_kwargs: Any,
    ) -> None:
        """
        model:
            Path to a local GGUF/GGML file, loaded through
            ``Llama(model_path=...)``.
        llm:
            A preloaded ``Llama``, used as is. Exactly one of ``model`` and
            ``llm`` is required.
        name:
            Defaults to the class name; stored as given.
        description:
            Defaults to ``"Llama.cpp LLM Engine"``.
        **client_kwargs:
            Keyword arguments forwarded unchanged to ``Llama(...)``
            (``n_gpu_layers``, ``n_threads``, ``flash_attn``, ``chat_format``,
            ...). ``n_ctx=2048`` and ``verbose=False`` are applied under them.
            Combining ``llm`` with ``client_kwargs`` raises.

        1. SDK presence check.
        2. Exactly one of ``model`` / ``llm``.
        3. ``llm`` together with ``client_kwargs`` raises.
        4. Base init.
        5. ``model`` given: load ``Llama`` with the defaults under the caller's kwargs.
        6. Store the ``Llama``, its model path and the client kwargs.
        """
        # 1. SDK presence check
        self._require_sdk()

        # 2. Source check
        if (model is None) == (llm is None):
            raise LLMEngineError(
                "LlamaCppEngine requires exactly one of 'model' or 'llm'."
            )

        # 3. A preloaded Llama takes no load kwargs
        if llm is not None and client_kwargs:
            raise LLMEngineError(
                "LlamaCppEngine: pass either 'llm' or client kwargs, not both."
            )

        # 4. Base init
        super().__init__(name=name, description=description or "Llama.cpp LLM Engine")

        # 5. Load the model when given a path
        if model is not None:
            load_kwargs: Dict[str, Any] = {"n_ctx": 2048, "verbose": False}
            load_kwargs.update(client_kwargs)
            llm = Llama(model_path=str(model), **load_kwargs)

        # 6. Store state
        self._llm = llm
        self._model_path = str(llm.model_path)
        self._client_kwargs = dict(client_kwargs)

    @staticmethod
    def _require_sdk() -> None:
        """Raise ``LLMEngineError`` when llama-cpp-python is not installed."""
        if Llama is None:
            raise LLMEngineError(
                "LlamaCppEngine "
                + format_missing_dependency("llama-cpp-python", "llama", _LLAMA_IMPORT_ERROR)
            ) from _LLAMA_IMPORT_ERROR

    @classmethod
    def from_pretrained(
            cls,
            repo_id: str,
            filename: str,
            *,
            additional_files: list[str] | None = None,
            local_dir: str | None = None,
            cache_dir: str | None = None,
            name: str | None = None,
            description: str | None = None,
            **client_kwargs: Any,
    ) -> LlamaCppEngine:
        """
        Download (or reuse from cache) a GGUF from the Hugging Face Hub and
        wrap it. A shallow wrapper over ``Llama.from_pretrained``: ``filename``
        is a glob, sharded GGUF files come through ``additional_files``.
        ``revision``, ``subfolder``, ``hf_token`` and ``local_files_only`` are
        not exposed (use ``HF_TOKEN`` and a pre-downloaded ``model`` path).

        1. SDK presence check (before any download).
        2. Load defaults (``n_ctx=2048``, ``verbose=False``) under the caller's kwargs.
        3. ``Llama.from_pretrained``; a missing ``huggingface-hub`` becomes ``LLMEngineError``.
        4. Wrap the result through ``llm=``.
        5. Record the client kwargs and return.
        """
        # 1. SDK presence check
        cls._require_sdk()

        # 2. Load kwargs
        load_kwargs: Dict[str, Any] = {"n_ctx": 2048, "verbose": False}
        load_kwargs.update(client_kwargs)

        # 3. Download and load
        try:
            llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                additional_files=additional_files,
                local_dir=local_dir,
                cache_dir=cache_dir,
                **load_kwargs,
            )
        except ImportError as exc:
            raise LLMEngineError(
                "LlamaCppEngine.from_pretrained "
                + format_missing_dependency("huggingface-hub", "llama", exc)
            ) from exc

        # 4-5. Wrap and record
        engine = cls(name=name, description=description, llm=llm)
        engine._client_kwargs = dict(client_kwargs)
        return engine

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def model_path(self) -> str:
        """Concrete local path of the loaded model (``llm.model_path``); fixed at construction."""
        return self._model_path

    def _validate_runtime_keys(self, keys: Iterable[str]) -> None:
        """
        1. Reserved keys are rejected (base check).
        2. Every key must be a named parameter of ``create_chat_completion``.
        """
        keys = list(keys)
        super()._validate_runtime_keys(keys)
        valid = {
            p.name
            for p in inspect.signature(self._llm.create_chat_completion).parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        unknown = set(keys) - valid
        if unknown:
            raise LLMEngineError(
                f"LlamaCppEngine.update_runtime_parameters: unknown keys {sorted(unknown)}; "
                f"valid keys: {sorted(valid - self._RESERVED_RUNTIME_KEYS)}"
            )

    # ------------------------------------------------------------------ #
    # LLMEngine template hooks
    # ------------------------------------------------------------------ #

    def _build_provider_payload(
            self,
            messages: List[Dict[str, str]],
            output_structure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        The exact ``create_chat_completion`` kwargs.

        1. Snapshot the runtime parameters.
        2. Engine-owned ``messages``.
        3. Structured call: ``response_format={"type": "json_object", "schema": output_structure}``
           (the schema is forwarded unmodified).
        """
        payload: Dict[str, Any] = dict(self._runtime_parameters)
        payload.update(messages=messages)
        if output_structure is not None:
            payload.update(
                response_format={"type": "json_object", "schema": output_structure}
            )
        return payload

    def _call_provider(self, payload: Any) -> Any:
        """One ``create_chat_completion`` call; error wrapping is the template's."""
        return self._llm.create_chat_completion(**payload)

    def _extract_result(
        self, response: Any, requested_structured: bool
    ) -> str | int | float | bool | list[Any] | dict[str, Any] | None:
        """
        Extract the assistant's textual OR structured reply from a llama.cpp
        chat completion response.

        Expected structure (OpenAI-compatible):

            response["choices"][0]["message"]["content"] -> str

        When `requested_structured` is True, the joined text is parsed via
        `json.loads`; a parse failure falls back to returning the raw text
        rather than raising, matching the base `LLMEngine` fallback contract
        (structured-generation Pass 3). Grammar-constrained decoding makes a
        genuinely invalid-JSON response effectively impossible here except
        via `max_tokens` truncation mid-structure -- the fallback exists for
        that case, not for provider unreliability the way it does for the
        remote engines.

        Renamed from `_extract_text` (structured-generation Pass 1
        convention); gains `requested_structured` per the Pass 3 base
        contract.
        """
        try:
            choices = response["choices"]
            if not choices:
                raise KeyError("choices is empty")
            message = choices[0].get("message", {})
            content = message.get("content", "")
        except Exception as exc:
            raise LLMEngineError(
                "LlamaCppEngine._extract_result: unexpected response shape"
            ) from exc
        text = str(content or "").strip()
        if not requested_structured:
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

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
        Diagnostic snapshot: the base fields plus ``model_path``. No client
        kwargs (live objects, credentials).
        """
        base = super().to_dict()
        base.update({"model_path": self._model_path})
        return base
