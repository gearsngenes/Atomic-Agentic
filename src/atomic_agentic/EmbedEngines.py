from __future__ import annotations
import math, time, os, random
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Any

try: from openai import OpenAI
except: pass
try: from google import genai
except:pass
try: from google.genai import types as genai_types
except: pass
try: from mistralai import Mistral, SDKError
except: pass
try: from llama_cpp import Llama  # local inference
except: pass

class EmbedEngine(ABC):
    """
    Abstract base class for all embedding engines.

    Contract
    --------
    vectorize(text: str) -> list[float]
        Takes a single string and returns a list of floats representing
        its embedding vector. The output must be compatible with any
        standard vector database (e.g., Pinecone, Qdrant, Redis, etc.).

    Attributes
    ----------
    dimension : Optional[int]
        Length of the embedding vectors. If None, inferred on first call.
    normalize : bool
        If True, apply L2 normalization before returning the vector.

    Design Notes
    ------------
    - Engines must remain stateless across calls (no hidden caches).
    - All subclasses must implement `vectorize()`.
    - Returned values must always be plain `list[float]` — never NumPy arrays.
    """

    def __init__(self, *, dimension: Optional[int] = None, normalize: bool = False):
        self._dimension: Optional[int] = dimension
        self.normalize: bool = bool(normalize)

    @property
    def dimension(self) -> Optional[int]:
        """Return the dimensionality of the embedding vectors."""
        return self._dimension

    @dimension.setter
    def dimension(self, value: int) -> None:
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError("dimension must be a positive integer or None")
        self._dimension = value

    @abstractmethod
    def vectorize(self, text: str) -> List[float]:
        """Return a list[float] embedding for `text`."""
        raise NotImplementedError

    # ── helper methods ───────────────────────────────────────────
    def _to_list_float(self, vec: Sequence[float]) -> List[float]:
        """Convert iterable to list[float] and normalize if requested."""
        out = [float(x) for x in vec]
        if self.normalize:
            out = self._l2_normalize(out)
        if self._dimension is None:
            self._dimension = len(out)
        return out

    @staticmethod
    def _l2_normalize(vec: Sequence[float]) -> List[float]:
        """Return L2-normalized version of a vector."""
        norm = math.sqrt(sum((x * x) for x in vec)) or 1.0
        return [x / norm for x in vec]

class OpenAIEmbedEngine(EmbedEngine):
    """
    OpenAI implementation of the EmbedEngine interface.

    Uses the OpenAI Embeddings API to generate vector embeddings for text.

    Parameters
    ----------
    model : str
        Name of the OpenAI embedding model. Common choices:
          • "text-embedding-3-small" → 1536 dimensions
          • "text-embedding-3-large" → 3072 dimensions
    api_key : Optional[str]
        OpenAI API key. Defaults to environment variable `OPENAI_API_KEY`.
    dimension : int
        Required. Embedding vector dimension that matches the chosen model.
        Example: 1536 for `text-embedding-3-small`, 3072 for `text-embedding-3-large`.
    normalize : bool
        Whether to apply L2 normalization to the returned vectors.

    Behavior
    --------
    - Returns a plain `list[float]`, compatible with any vector DB.
    - Enforces consistent dimensionality on every call.
    - Stateless across invocations.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = os.getenv("OPENAI_API_KEY"),
        dimension: int = 1536,
        normalize: bool = False,
    ):
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("`dimension` must be a positive integer and is now mandatory.")
        super().__init__(dimension=dimension, normalize=normalize)
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def vectorize(self, text: str) -> List[float]:
        """Generate an OpenAI embedding for the given text."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        response = self.client.embeddings.create(
            model=self.model,
            input=text.strip(),
            dimensions=self.dimension  # Explicitly request dimension match
        )
        vec = response.data[0].embedding
        out = self._to_list_float(vec)

        if len(out) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {len(out)}"
            )
        return out

class GeminiEmbedEngine(EmbedEngine):
    """
    Google Gemini text-embedding engine.

    Parameters
    ----------
    model : str
        Gemini embedding model id (e.g., "text-embedding-004").
    api_key : Optional[str]
        Google API key. Defaults to env var `GOOGLE_API_KEY`.
    task_type : Optional[str]
        Optional embedding task hint ("SEMANTIC_SIMILARITY", etc.).
    dimension : int
        Required. The exact dimensionality of embeddings.
        Passed to the Gemini API as `output_dimensionality`
        and enforced locally for DB consistency.
    normalize : bool
        Whether to apply L2 normalization to the returned vector.
    """

    def __init__(
        self,
        model: str = "text-embedding-004",
        api_key: str = os.getenv("GOOGLE_API_KEY"),
        dimension: int = 1536,
        *,
        task_type: Optional[str] = None,
        normalize: bool = False,
    ):
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("`dimension` must be a positive integer and is mandatory.")
        super().__init__(dimension=dimension, normalize=normalize)
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.task_type = task_type

    def vectorize(self, text: str) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        text = text.strip()

        # Build typed config
        if genai_types is not None and hasattr(genai_types, "EmbedContentConfig"):
            cfg = genai_types.EmbedContentConfig(
                task_type=self.task_type,
                output_dimensionality=self.dimension,  # unified!
            )
        else:
            cfg = {
                "task_type": self.task_type,
                "output_dimensionality": self.dimension,
            }

        # Call API (contents= required)
        resp = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=cfg,
        )

        # Extract the vector
        try:
            vec = list(resp.embeddings[0].values)
        except Exception:
            if isinstance(resp, dict):
                vec = list(resp["embeddings"][0]["values"])
            else:
                raise RuntimeError("Unexpected Gemini embedding response format.")

        out = self._to_list_float(vec)

        if len(out) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {len(out)}"
            )
        return out

class MistralEmbedEngine(EmbedEngine):
    """
    Mistral implementation of EmbedEngine with retry/backoff.

    - Fixed-size embeddings (use dimension=1024 for 'mistral-embed').
    - Handles 429 'service_tier_capacity_exceeded' by exponential backoff.
    - Returns list[float].
    """

    def __init__(
        self,
        model: str = "mistral-embed",
        api_key: str = os.getenv("MISTRAL_API_KEY"),
        dimension: int = 1024,
        normalize: bool = False,
        *,
        # throttle controls
        max_retries: int = 6,
        backoff_base: float = 0.6,   # starting sleep (seconds)
        backoff_factor: float = 2.0, # exponential growth
        jitter: float = 0.15,        # +/- jitter fraction
    ):
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("`dimension` must be a positive integer and is mandatory.")
        super().__init__(dimension=dimension, normalize=normalize)
        self.client = Mistral(api_key=api_key)
        self.model = model

        self.max_retries = int(max_retries)
        self.backoff_base = float(backoff_base)
        self.backoff_factor = float(backoff_factor)
        self.jitter = float(jitter)

    def vectorize(self, text: str) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        text = text.strip()

        attempt = 0
        while True:
            try:
                # NOTE: current SDK requires 'inputs' (plural)
                resp = self.client.embeddings.create(model=self.model, inputs=[text])

                # Expected shape: resp.data[0].embedding
                data = getattr(resp, "data", None)
                if not data or not hasattr(data[0], "embedding"):
                    raise RuntimeError("Unexpected Mistral embedding response structure.")
                vec = list(data[0].embedding)

                out = self._to_list_float(vec)

                if len(out) != self.dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {self.dimension}, got {len(out)}"
                    )
                return out

            except SDKError as e:
                # Retry on throttling/capacity or transient server issues
                msg = (e.message or "").lower()
                code = getattr(e, "status_code", None)

                retryable = (
                    code in (429, 500, 502, 503, 504)
                    or "service_tier_capacity_exceeded" in msg
                    or "rate limit" in msg
                    or "temporarily unavailable" in msg
                )

                if not retryable or attempt >= self.max_retries:
                    # give a clearer error if we’re done retrying
                    hint = (
                        "Hit Mistral free-tier capacity. Reduce RPS (≈1 req/s), "
                        "add backoff, batch multiple texts in one call, or upgrade your tier."
                    )
                    raise RuntimeError(f"Mistral embeddings failed: {e}. {hint}") from e

                # exponential backoff with jitter
                sleep_s = self.backoff_base * (self.backoff_factor ** attempt)
                jitter = sleep_s * (self.jitter * (2 * random.random() - 1.0))
                time.sleep(max(0.0, sleep_s + jitter))
                attempt += 1

class LlamaCppEmbedEngine(EmbedEngine):
    """
    Local embedding engine using llama-cpp-python (GGUF models),
    with two loading paths:
      1) Direct GGUF path via `model_path`
      2) Hugging Face repo via `Llama.from_pretrained(repo_id, filename, ...)`

    Notes
    -----
    - Works offline with a local GGUF embedding model (e.g.,
      nomic-embed-text-v1.5.f16.gguf, bge-small/base/large gguf, gte-* gguf).
    - `dimension` is **mandatory** and must match the model’s embedding size
      (e.g., 768 for nomic v1.5, 384 for bge-small, 1024 for bge-large).
    - Returns plain `list[float]`.

    Parameters
    ----------
    model_path : Optional[str]
        Filesystem path to the GGUF model. Mutually exclusive with (repo_id, filename).
    repo_id : Optional[str]
        Hugging Face repository id (e.g., "nomic-ai/nomic-embed-text-v1.5-GGUF").
    filename : Optional[str]
        GGUF filename within the repo (e.g., "nomic-embed-text-v1.5.f16.gguf").
    dimension : int
        Required. Expected embedding size; enforced on every call.
    normalize : bool
        If True, L2-normalize the returned vector.
    n_ctx : int
        Context window (tokens). For embeddings, this mainly bounds internal chunking.
    n_threads : Optional[int]
        Thread count for llama-cpp; 0/None lets the lib choose.
    **llama_kwargs :
        Passed through to llama_cpp.Llama / Llama.from_pretrained
        (e.g., n_gpu_layers=..., tensor_split=..., seed=..., mmap=...).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        dimension: int = 1024,
        normalize: bool = False,
        *,
        verbose: bool = False,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        **llama_kwargs: Any,
    ):
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("`dimension` must be a positive integer and is mandatory.")
        super().__init__(dimension=dimension, normalize=normalize)

        # ensure embedding fast-path
        llama_kwargs = dict(llama_kwargs)  # copy so we can modify
        llama_kwargs.setdefault("embedding", True)

        # Two loading paths, mirroring your LlamaCppEngine
        if model_path:
            if not isinstance(model_path, str) or not os.path.exists(model_path):
                raise ValueError("`model_path` must point to an existing GGUF file.")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=(n_threads or 0),
                verbose = verbose,
                **llama_kwargs,
            )
        elif repo_id and filename:
            # auto-downloads to cache, then loads; kwargs propagate
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=n_ctx,
                n_threads=(n_threads or 0),
                verbose = verbose,
                **llama_kwargs,
            )
        else:
            raise ValueError("Provide either `model_path` OR both `repo_id` and `filename`.")

    def vectorize(self, text: str) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        text = text.strip()

        # llama-cpp exposes both OpenAI-like and helper shapes; support both.
        # Preferred: OpenAI-like create_embedding
        try:
            out = self.llm.create_embedding(input=text)  # {"data":[{"embedding":[...]}], ...}
            vec = list(out["data"][0]["embedding"])
        except Exception:
            # Fallback: helper returning list-of-lists or dict
            out2 = self.llm.embed(texts=[text])
            if isinstance(out2, dict) and "data" in out2:
                vec = list(out2["data"][0]["embedding"])
            elif isinstance(out2, list) and out2 and isinstance(out2[0], (list, tuple)):
                vec = [float(x) for x in out2[0]]
            else:
                raise RuntimeError("llama-cpp returned an unexpected embedding response.")

        out_vec = self._to_list_float(vec)

        if len(out_vec) != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {len(out_vec)}"
            )
        return out_vec
