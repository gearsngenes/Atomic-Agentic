from __future__ import annotations

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)

from .base import LLMEngine
from ..exceptions import LLMEngineError
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
    A model may be loaded from either:

    - ``model_path``: a direct path to an existing local GGUF/GGML file.
    - ``repo_id`` + ``filename``: a Hugging Face Hub file reference resolved
      through ``hf_hub_download(...)`` into a concrete local file path.

    In both cases, ``self.model_path`` stores the concrete local path loaded by
    ``Llama(model_path=...)``.

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

    3) Attachments are not supported. Any attempt to call ``attach(path)`` fails
       with ``LLMEngineError``.

    Token usage
    -----------
    ``_extract_token_usage`` maps the OpenAI-compatible response
    ``response["usage"]`` dictionary into ``LlamaCppTokenUsage`` using prompt,
    completion, and total token counts.

    Model data
    ----------
    ``_get_model_data`` returns the concrete local model path loaded by
    llama.cpp.
    """

    def __init__(
            self,
            # ── AtomicInvokable identity ──────────────────────────────────────────
            name: str | None = None,
            namespace: str = "llm",
            description: str = "Llama.cpp LLM Engine",
            # ── Model source ──────────────────────────────────────────────────────
            model_path: str | None = None,
            repo_id: str | None = None,
            filename: str | None = None,
            # ── Llama constructor config (explicit surface) ───────────────────────
            n_ctx: int = 2048,
            verbose: bool = False,
            # ── Generation defaults — applied at every create_chat_completion call ─
            temperature: float | None = None,
            top_k: int | None = None,
            top_p: float | None = None,
            min_p: float | None = None,
            max_tokens: int | None = None,
            repeat_penalty: float | None = None,
            seed: int | None = None,
            stop: str | list[str] | None = None,
            *,
            # ── HF download options — only used when loading via Hub ─────────────
            subfolder: str | None = None,
            revision: str | None = None,
            hf_token: str | bool | None = None,
            cache_dir: str | None = None,
            local_dir: str | None = None,
            local_files_only: bool = False,
            # ── Base engine config ────────────────────────────────────────────────
            timeout_seconds: float = 30.0,
            max_retries: int = 2,
            retry_backoff_base: float = 0.5,
            retry_backoff_max: float = 8.0,
            # ── Remaining Llama constructor params forwarded verbatim ─────────────
            **llama_kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        name:
            Optional human-friendly engine name; defaults to ``"llama_cpp"``.
        namespace, description:
            Shared ``AtomicInvokable`` identity fields.
        model_path:
            Direct path to a local GGUF/GGML model file.
        repo_id:
            Hugging Face repo ID used with ``filename`` when ``model_path`` is
            omitted.
        filename:
            Model filename within ``repo_id`` when resolving from Hugging Face.
        n_ctx:
            Context window size; overrides Llama's default of 512.
        verbose:
            Verbose logging flag; overrides Llama's default of True.
        temperature, top_k, top_p, min_p, max_tokens, repeat_penalty, seed, stop:
            Generation defaults applied at every ``create_chat_completion`` call.
            ``None`` omits the parameter from the call.
        subfolder:
            Optional subfolder within the HF repo (for repos with nested files).
        revision:
            Optional HF branch, tag, or commit hash.
        hf_token:
            Optional HF auth token; ``True`` uses the locally configured token.
        cache_dir:
            Optional HF cache root directory.
        local_dir:
            Optional local directory where the model file is materialized.
        local_files_only:
            If True, resolve only from local cache; do not download.
        timeout_seconds, max_retries,
        retry_backoff_base, retry_backoff_max:
            Shared ``LLMEngine`` configuration.
        **llama_kwargs:
            Additional keyword arguments forwarded verbatim to ``Llama(...)``.
            Covers the full llama-cpp-python constructor surface: ``n_gpu_layers``,
            ``n_threads``, ``flash_attn``, ``chat_format``, ``use_mmap``, etc.
        """
        # 1. Name sanitization + super init
        sanitized_name = (
                name or "llama_cpp"
        ).replace(":", "_").replace("-", "_").replace(" ", "_").replace(".", "_")
        super().__init__(
            name=sanitized_name,
            namespace=namespace,
            description=description or "Llama.cpp LLM Engine",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        # 2. SDK presence check
        if Llama is None:
            raise LLMEngineError(
                "LlamaCppEngine requires the `llama-cpp-python` package."
            )

        # 3. Resolve model path
        if model_path:
            resolved_model_path = model_path
        elif repo_id and filename:
            if hf_hub_download is None:
                raise LLMEngineError(
                    "LlamaCppEngine requires the `huggingface_hub` package when "
                    "`repo_id` and `filename` are used."
                )
            resolved_model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder,
                revision=revision,
                cache_dir=cache_dir,
                local_dir=local_dir,
                local_files_only=bool(local_files_only),
                token=hf_token,
            )
        else:
            raise LLMEngineError(
                "LlamaCppEngine requires either `model_path` or both "
                "`repo_id` and `filename`."
            )

        # 4. Build Llama instance
        self._llm = Llama(
            model_path=str(resolved_model_path),
            n_ctx=int(n_ctx),
            verbose=bool(verbose),
            **llama_kwargs,
        )

        # 5. Store config
        self._model_path = str(resolved_model_path)
        self._repo_id = repo_id
        self._filename = filename
        self._has_hf_token = hf_token is not None
        self._n_ctx = int(n_ctx)
        self._verbose = bool(verbose)
        self._llama_kwargs = dict(llama_kwargs)
        self.temperature = float(temperature) if temperature is not None else None
        self.top_k = int(top_k) if top_k is not None else None
        self.top_p = float(top_p) if top_p is not None else None
        self.min_p = float(min_p) if min_p is not None else None
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.repeat_penalty = float(repeat_penalty) if repeat_penalty is not None else None
        self.seed = int(seed) if seed is not None else None
        self.stop = stop

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def model_path(self) -> str:
        """Concrete local path of the loaded model; fixed at construction."""
        return self._model_path

    @property
    def repo_id(self) -> Optional[str]:
        """Hugging Face repo ID used to download the model; fixed at construction."""
        return self._repo_id

    @property
    def filename(self) -> Optional[str]:
        """Model filename within the repo; fixed at construction."""
        return self._filename

    @property
    def n_ctx(self) -> int:
        """Context window size passed to Llama at construction."""
        return self._n_ctx

    @property
    def verbose(self) -> bool:
        """Verbose flag passed to Llama at construction."""
        return self._verbose

    # ------------------------------------------------------------------ #
    # LLMEngine template hooks
    # ------------------------------------------------------------------ #

    def _build_provider_payload(
            self,
            messages: List[Dict[str, str]],
            attachments: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
        Map normalized messages to llama.cpp's chat completion schema.

        - `messages` are already validated (role/content strings).
        - `attachments` are ignored; this engine does not support attachments.
        """
        # llama-cpp-python exposes an OpenAI-compatible chat API, so we
        # simply pass the messages through unchanged.
        return {"messages": messages}

    def _call_provider(self, payload: Any) -> Any:
        """
        Perform a single llama.cpp chat completion call.

        Retries and error-wrapping are handled by `LLMEngine._call_with_retries`.
        """
        chat_kwargs: Dict[str, Any] = {}
        if self.temperature is not None:
            chat_kwargs["temperature"] = self.temperature
        if self.top_k is not None:
            chat_kwargs["top_k"] = self.top_k
        if self.top_p is not None:
            chat_kwargs["top_p"] = self.top_p
        if self.min_p is not None:
            chat_kwargs["min_p"] = self.min_p
        if self.max_tokens is not None:
            chat_kwargs["max_tokens"] = self.max_tokens
        if self.repeat_penalty is not None:
            chat_kwargs["repeat_penalty"] = self.repeat_penalty
        if self.seed is not None:
            chat_kwargs["seed"] = self.seed
        if self.stop is not None:
            chat_kwargs["stop"] = self.stop

        return self._llm.create_chat_completion(
            messages=payload["messages"],
            **chat_kwargs,
        )

    def _should_retry(self, exc: Exception, attempt: int) -> bool:
        """
        Local llama.cpp inference has no network layer, so there is no
        status-code or connection-error signal to discriminate transient
        from permanent failures. Deliberate policy: retry blindly, bounded
        only by the shared attempt budget.
        """
        return attempt <= self._max_retries

    def _extract_text(self, response: Any) -> str:
        """
        Extract assistant text from a llama.cpp chat completion response.

        Expected structure (OpenAI-compatible):

            response["choices"][0]["message"]["content"] -> str
        """
        try:
            choices = response["choices"]
            if not choices:
                raise KeyError("choices is empty")
            message = choices[0].get("message", {})
            content = message.get("content", "")
        except Exception as exc:
            raise LLMEngineError(
                "LlamaCppEngine._extract_text: unexpected response shape"
            ) from exc
        return str(content).strip()

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
        Diagnostic snapshot for LlamaCppEngine.

        Includes non-secret model source, model-load config, and generation
        defaults. The Hugging Face token value is never serialized.
        """
        base = super().to_dict()
        base.update({
            "model_path": self._model_path,
            "repo_id": self._repo_id,
            "filename": self._filename,
            "has_hf_token": self._has_hf_token,
            "n_ctx": self._n_ctx,
            "verbose": self._verbose,
            "llama_kwargs": dict(self._llama_kwargs),
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "max_tokens": self.max_tokens,
            "repeat_penalty": self.repeat_penalty,
            "seed": self.seed,
            "stop": self.stop,
        })
        return base
