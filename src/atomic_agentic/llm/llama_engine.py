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

# Load defaults applied under the caller's ``client_kwargs``.
_LOAD_DEFAULTS: Dict[str, Any] = {"n_ctx": 2048, "verbose": False}

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

    ``engine.model`` is the concrete local path of the loaded model (read from
    ``llm.model_path`` after construction); it is read-only.

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

    Attachments are not supported; this engine has no ``attach`` API.

    Result extraction and usage
    ---------------------------
    ``_extract_result`` reads ``choices[0].message.content`` (``None`` counts
    as ``""``) and, on a structured call, parses it with ``json.loads`` (raw
    text on a parse failure, e.g. ``max_tokens`` truncation mid-structure).
    ``_extract_token_usage`` maps ``response["usage"]`` into
    ``LlamaCppTokenUsage``; with no ``usage`` key it raises unless
    ``require_usage`` is ``False``, which yields an all-zero record.
    ``_get_model_data`` returns the loaded model path.

    Structured-output schema policy
    -------------------------------
    ``output_structure`` is forwarded unmodified as ``response_format =
    {"type": "json_object", "schema": ...}``. llama-cpp-python compiles the
    schema into a GBNF grammar that constrains sampling locally, so
    conformance is a property of the installed ``llama-cpp-python`` version,
    not of the model. Structural keywords (``type``, ``enum``, ``properties``,
    ``required``, ``items``, ``pattern``, ``oneOf``/``anyOf``/``allOf``,
    ``$ref``, ...) are enforced; ``minimum``/``maximum`` and their exclusive
    forms are silently ignored. The grammar never reaches the prompt, so
    describe each field's intent in the prompt or system message.
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
            load_kwargs: Dict[str, Any] = dict(_LOAD_DEFAULTS)
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
        load_kwargs: Dict[str, Any] = dict(_LOAD_DEFAULTS)
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
    def model(self) -> str:
        """Concrete local path of the loaded model (``llm.model_path``); read-only."""
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
        Read ``response["choices"][0]["message"]["content"]`` (``None`` counts
        as ``""``); on a structured call, return its ``json.loads`` value
        (raw text on a parse failure). An unexpected shape raises.
        """
        try:
            choices = response["choices"]
        except Exception as exc:
            raise LLMEngineError(
                "LlamaCppEngine._extract_result: unexpected response shape"
            ) from exc
        if not choices:
            raise LLMEngineError(
                "LlamaCppEngine._extract_result: unexpected response shape (empty choices)"
            )
        try:
            content = choices[0].get("message", {}).get("content", "")
        except Exception as exc:
            raise LLMEngineError(
                "LlamaCppEngine._extract_result: unexpected response shape"
            ) from exc
        text = str(content or "")
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

        The response is an OpenAI-compatible dict whose ``usage`` mapping holds
        prompt, completion and total counts. A response with no ``usage`` key
        raises unless ``require_usage`` is ``False``, which yields an all-zero
        record; a present but malformed ``usage`` always raises.
        """
        try:
            usage = response["usage"]
        except KeyError as exc:
            if not self.require_usage:
                return LlamaCppTokenUsage(
                    input_tokens=0, generated_tokens=0, total_tokens=0, response_tokens=0
                )
            raise LLMEngineError(
                "LlamaCppEngine._extract_token_usage: response missing usage."
            ) from exc
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
        """Return the llama.cpp model identity (the loaded local path)."""
        return LlamaCppModelData(
            provider="llama_cpp",
            model_path=self._model_path,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """
        Diagnostic snapshot: the base fields plus ``model`` (the loaded path).
        No client kwargs (live objects, credentials).
        """
        base = super().to_dict()
        base.update({"model": self._model_path})
        return base
