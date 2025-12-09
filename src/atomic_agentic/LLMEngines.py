from __future__ import annotations

# LLMEngines.py
# Engines are stateless adapters around provider SDKs.
# The Agent owns conversation history; engines map messages + attachments
# to provider-specific requests.

import mimetypes
import os
import random
import time
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional

# ~~~Provider SDKs~~~
# OpenAI
try: from openai import OpenAI
except: OpenAI = None
# Google
try: from google import genai
except: genai = None
# Mistral
try: from mistralai import Mistral
except: Mistral = None
# Lamma-CPP-Python
try: from llama_cpp import Llama
except: Llama = None

from ._primitives import LLMEngine
from ._exceptions import LLMEngineError

__all__ = ["GeminiEngine", "LlamaCppEngine", "LLMEngine", "MistralEngine", "OpenAIEngine"]

# ── OPENAI (Responses API + Chat fallback) ─────────────────────────────────────
class OpenAIEngine(LLMEngine):
    """
    OpenAI adapter using the Responses API.

    File policy
    -----------
    Attachments are persistent engine state:

    - PDFs    → uploaded once via Files API; attached as `{ "type": "input_file", "file_id": ... }`
    - Images  → uploaded once via Files API; attached as `{ "type": "input_image", "file_id": ... }`
    - Text/Code → read and inlined as `{ "type": "input_text", "text": ... }`
      (with a configurable character cutoff).

    Unsupported file classes (audio/video, obviously binary types, etc.) are
    rejected at `attach` time.

    System messages are carried via the `instructions` field; non-system messages
    are encoded as `input_text` or `output_text` blocks.
    """

    # Image extensions that map to `input_image`
    _IMAGE_EXTS: tuple[str, ...] = (
        ".png", ".jpg", ".jpeg",
        ".webp", ".gif", ".bmp",
        ".tif", ".tiff", ".heic",
    )

    # Text/code-ish extensions we are willing to inline as text.
    _TEXT_EXTS: tuple[str, ...] = (
        ".txt", ".md", ".rst", ".log",
        ".json", ".jsonl", ".yaml", ".yml",
        ".csv", ".tsv", ".py", ".ipynb",
        ".js", ".ts", ".jsx", ".tsx",
        ".java", ".c", ".cpp", ".h",
        ".hpp", ".rs", ".go", ".rb",
        ".php", ".cs", ".html", ".htm",
        ".xml",
    )

    # Extra illegal extensions for this provider (merged with base `illegal_attachment_exts`)
    _ILLEGAL_EXTS: set[str] = {
        ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",  # archives
        ".exe", ".dll", ".so", ".bin", ".o",  # executables/binaries
        ".db", ".sqlite",  # databases
        ".h5", ".pt", ".pth", ".onnx",  # model weights
    }

    # MIME prefixes we never accept even if extension would otherwise pass.
    _ILLEGAL_MIME_PREFIXES: tuple[str, ...] = ("audio/", "video/")

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        inline_cutoff_chars: int = 200_000,
        extra_illegal_exts: Optional[set[str]] = None,
        *,
        name: Optional[str] = None,
        timeout_seconds: float = 600.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 8.0,
    ) -> None:
        """
        Parameters
        ----------
        model:
            OpenAI model identifier (e.g. "gpt-4.1", "gpt-4o-mini").
        api_key:
            Optional API key; if omitted, `OPENAI_API_KEY` from the environment is used.
        temperature:
            Sampling temperature (ignored for certain reasoning models if not applicable).
        inline_cutoff_chars:
            Maximum number of characters to inline from text/code attachments.
        extra_illegal_exts:
            Optional set of additional extensions to reject at `attach` time.
        name, timeout_seconds, max_retries, retry_backoff_base, retry_backoff_max:
            Template-method engine configuration (see `_primitives.LLMEngine`).
        """
        super().__init__(
            name=name or f"openai:{model}",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        if OpenAI is None:
            raise RuntimeError(
                "OpenAIEngine requires the `openai` package; install `openai` to use it."
            )

        # Honor the base engine's timeout knob when constructing the OpenAI client.
        # The official SDK exposes a `timeout` option for this. :contentReference[oaicite:2]{index=2}
        self.llm = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            timeout=self._timeout_seconds,
        )

        self.model = model
        self.temperature = float(temperature)
        self.inline_cutoff_chars = int(inline_cutoff_chars)

        # Merge illegal extension policy with base defaults + any user-supplied extras.
        merged_illegal = set(self.illegal_attachment_exts) | set(self._ILLEGAL_EXTS)
        if extra_illegal_exts:
            merged_illegal |= set(extra_illegal_exts)
        self.illegal_attachment_exts = merged_illegal

        # Positive allow-list: PDFs + known image + text/code extensions.
        allowed = set(self._TEXT_EXTS) | set(self._IMAGE_EXTS) | {".pdf"}
        self.allowed_attachment_exts = allowed

    # ------------------------------------------------------------------ #
    # Overrides / template hooks
    # ------------------------------------------------------------------ #

    def _validate_attachment_path(self, path: str) -> None:
        """
        Extend the base validation with MIME-type checks (reject audio/video).
        """
        super()._validate_attachment_path(path)

        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        if any(mime.startswith(pref) for pref in self._ILLEGAL_MIME_PREFIXES):
            raise LLMEngineError(
                f"OpenAIEngine.attach: MIME type {mime!r} is not supported"
            )

    def _build_provider_payload(
        self,
        messages: List[Dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build the payload for the OpenAI Responses API from normalized messages
        and the current attachments snapshot.
        """
        instructions = self._collect_instructions(messages)
        blocks = self._build_role_blocks(messages)

        # Find or create a `user` block to hold attachments.
        user_idx = self._ensure_user_block(blocks)
        user_parts: List[Dict[str, Any]] = blocks[user_idx]["content"]

        for path, meta in attachments.items():
            kind = str(meta.get("kind", "text"))

            # Inlined text
            if meta.get("inlined") or kind == "text":
                text = str(meta.get("inlined_text") or "")
                if not text:
                    continue
                header = f"\n[Inlined file: {os.path.basename(path)}]\n"
                user_parts.append(
                    {"type": "input_text", "text": header + text}
                )
                continue

            # Uploaded files/images with a file_id
            if meta.get("uploaded") and meta.get("file_id"):
                file_id = str(meta["file_id"])
                if kind == "pdf":
                    user_parts.append({"type": "input_file", "file_id": file_id})
                else:
                    user_parts.append({"type": "input_image", "file_id": file_id})
                continue

            # Fallback: try to handle the local path directly
            try:
                self._attach_local_path(user_parts, path)
            except Exception:
                # Best-effort; silently skip failing attachments.
                continue

        payload: Dict[str, Any] = {"blocks": blocks}
        if instructions:
            payload["instructions"] = instructions
        return payload

    def _call_provider(self, payload: Dict[str, Any]) -> Any:
        """
        Perform a single Responses API call using the pre-built payload.

        Retries + backoff are handled by the base `_call_with_retries` wrapper.
        """
        blocks = payload["blocks"]
        instructions = payload.get("instructions")

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": blocks,
        }
        if instructions:
            kwargs["instructions"] = instructions

        # For non-reasoning models, respect temperature; for some `gpt-5`-class
        # models this may be ignored or overridden by the provider.
        if "gpt-5" not in self.model.lower():
            kwargs["temperature"] = self.temperature

        return self.llm.responses.create(**kwargs)

    def _extract_text(self, response: Any) -> str:
        """
        Extract the assistant's textual reply from a Responses API response object.
        """
        return (getattr(response, "output_text", None) or "").strip()

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        """
        Prepare a local path for reuse with this engine.

        - PDFs/images → upload once to Files API; metadata contains `file_id`.
        - Text/code → inline as text (UTF-8, with a length cutoff).
        """
        kind = self._classify_path(path)
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        ext = os.path.splitext(path)[1].lower()

        # PDF/image: upload and keep a handle
        if kind in ("pdf", "image"):
            file_id = self._upload_file(path)
            return {
                "kind": kind,
                "mime": mime,
                "ext": ext,
                "uploaded": True,
                "file_id": file_id,
            }

        # Text/code → inline as text
        text = self._read_text_file(path)
        if len(text) > self.inline_cutoff_chars:
            text = text[: self.inline_cutoff_chars] + "\n…[truncated]\n"

        return {
            "kind": "text",
            "mime": mime,
            "ext": ext,
            "inlined": True,
            "inlined_text": text,
        }

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        """
        Attempt to delete any remote file created via the Files API.

        This is best-effort; errors are ignored.
        """
        file_id = meta.get("file_id")
        if not file_id:
            return
        try:
            self.llm.files.delete(file_id)
        except Exception:
            return

    # ------------------------------------------------------------------ #
    # OpenAI-specific helpers (not part of the template surface)
    # ------------------------------------------------------------------ #

    def _classify_path(self, path: str) -> str:
        """
        Classify `path` into 'pdf' | 'image' | 'text'.

        Base validation has already run; this is a semantic bucketization.
        """
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf" or mime == "application/pdf":
            return "pdf"
        if ext in self._IMAGE_EXTS or mime.startswith("image/"):
            return "image"
        return "text"

    def _collect_instructions(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Concatenate all system message contents into an `instructions` string.
        """
        parts = [
            m["content"]
            for m in messages
            if (m.get("role") or "").lower() == "system" and m.get("content")
        ]
        joined = "\n\n".join(parts).strip()
        return joined or None

    def _build_role_blocks(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Convert chat messages into Responses API blocks.

        - `assistant` turns → `output_text`
        - Other non-system turns → `input_text`
        - `system` content is carried only via `instructions`.
        """
        blocks: List[Dict[str, Any]] = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            if role == "system":
                continue  # handled by _collect_instructions
            text = m.get("content") or ""
            if not text and role != "user":
                # Keep empty user (we attach into it), skip other empty roles.
                continue
            part_type = "output_text" if role == "assistant" else "input_text"
            parts: List[Dict[str, Any]] = []
            if text:
                parts.append({"type": part_type, "text": text})
            blocks.append({"role": role, "content": parts})
        return blocks

    def _ensure_user_block(self, blocks: List[Dict[str, Any]]) -> int:
        """
        Return the index of a `user` block; create an empty one at the end if none exist.
        """
        for i in range(len(blocks) - 1, -1, -1):
            if blocks[i].get("role") == "user":
                return i
        blocks.append({"role": "user", "content": []})
        return len(blocks) - 1

    def _read_text_file(self, path: str) -> str:
        """
        Read a local file as UTF-8 (with replacement). Returns a best-effort string.
        """
        try:
            with open(path, "rb") as f:
                raw = f.read()
            return raw.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - unlikely IO failures
            return f"[Error reading file '{os.path.basename(path)}': {exc}]"

    def _upload_file(self, path: str) -> str:
        """
        Upload a local file to the OpenAI Files API and return its `file_id`.

        We use purpose="assistants" which is appropriate for model context files. :contentReference[oaicite:3]{index=3}
        """
        with open(path, "rb") as fp:
            f = self.llm.files.create(file=fp, purpose="assistants")
        return str(f.id)

    def _attach_local_path(
        self,
        user_parts: List[Dict[str, Any]],
        path: str,
    ) -> None:
        """
        Fallback helper for attaching a path directly when metadata is incomplete.

        - PDFs   → upload + `input_file` reference
        - Images → upload + `input_image` reference
        - Other  → inline as text (UTF-8, truncated by `inline_cutoff_chars`).
        """
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        lower = path.lower()
        is_pdf = mime == "application/pdf" or lower.endswith(".pdf")
        is_image = mime.startswith("image/") or lower.endswith(self._IMAGE_EXTS)

        if is_pdf:
            file_id = self._upload_file(path)
            user_parts.append({"type": "input_file", "file_id": file_id})
            return

        if is_image:
            file_id = self._upload_file(path)
            user_parts.append({"type": "input_image", "file_id": file_id})
            return

        # Inline as text
        text = self._read_text_file(path)
        if len(text) > self.inline_cutoff_chars:
            text = text[: self.inline_cutoff_chars] + "\n…[truncated]\n"

        header = f"\n[Inlined file: {os.path.basename(path)}]\n"
        user_parts.append({"type": "input_text", "text": header + text})

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def to_dict(self) -> OrderedDict[str, Any]:
        """
        Diagnostic snapshot for OpenAIEngine, without secrets.

        Includes model, temperature, and inline cutoff in addition to base engine info.
        """
        base = super().to_dict()
        base.update(
            OrderedDict(
                model=self.model,
                temperature=self.temperature,
                inline_cutoff_chars=self.inline_cutoff_chars,
            )
        )
        return base


# ── GEMINI (flat contents: file objects + strings) ─────────────────────────────
class GeminiEngine(LLMEngine):
    """
    Google Gemini adapter using the Google Gen AI SDK.

    Strategy per call
    -----------------
    1) Engine-level attachments are prepared via `attach(path)`:
       - `_prepare_attachment` uploads supported files via `client.files.upload`.
       - Attachment metadata stores the returned File object and its resource name.
    2) `invoke(messages)` (from the base `LLMEngine`) will:
       - normalize chat messages (role/content pairs),
       - snapshot current attachments,
       - call `_build_provider_payload` to construct:
           * `system_instruction` from system messages
           * a flat `contents` list:
             - File objects for uploaded attachments
             - plain strings for non-system turns
       - call `_call_with_retries` → `_call_provider`
       - call `_extract_text` to normalize the response.
    3) `detach(path)` calls `_on_detach` for best-effort file deletion via
       `client.files.delete`.
    """

    # Extra illegal extensions for this provider (merged with base `illegal_attachment_exts`)
    _ILLEGAL_EXTS: set[str] = {
        ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",  # archives
        ".exe", ".dll", ".so", ".bin", ".o",  # executables/binaries
        ".db", ".sqlite",  # databases
        ".h5", ".pt", ".pth", ".onnx",  # model weights
    }

    # MIME prefixes we never accept even if extension would otherwise pass.
    _ILLEGAL_MIME_PREFIXES: tuple[str, ...] = ("audio/", "video/")

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        extra_illegal_exts: Optional[set[str]] = None,
        *,
        name: Optional[str] = None,
        timeout_seconds: float = 600.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 8.0,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Gemini model identifier (e.g. "gemini-2.5-flash", "gemini-2.0-pro").
        api_key:
            Optional API key. If omitted, the SDK uses GOOGLE_API_KEY
            env vars. :contentReference[oaicite:3]{index=3}
        temperature:
            Sampling temperature for text generation.
        extra_illegal_exts:
            Optional set of additional extensions to reject at `attach` time.
        name, timeout_seconds, max_retries, retry_backoff_base, retry_backoff_max:
            Template-method engine configuration (see `_primitives.LLMEngine`).
        """
        super().__init__(
            name=name or f"gemini:{model}",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        if genai is None:
            raise RuntimeError(
                "GeminiEngine requires the `google-genai` package; "
                "install `google-genai` to use it."
            )

        # Honor the base engine's timeout knob via http_options.
        # The SDK accepts a plain dict here. :contentReference[oaicite:4]{index=4}
        http_options: Dict[str, Any] = {
            "timeout": int(self._timeout_seconds * 1000),  # ms
        }

        client_kwargs: Dict[str, Any] = {"http_options": http_options}
        if api_key is not None:
            client_kwargs["api_key"] = api_key

        self.client = genai.Client(**client_kwargs)
        self.model = model
        self.temperature = float(temperature)

        # Merge coarse illegal extension policy with base defaults + any user-supplied extras.
        merged_illegal = set(self.illegal_attachment_exts) | set(self._ILLEGAL_EXTS)
        if extra_illegal_exts:
            merged_illegal |= set(extra_illegal_exts)
        self.illegal_attachment_exts = merged_illegal

        # Gemini supports a wide range of file types; we stick with a blacklist +
        # MIME filter instead of a strict allow-list, so `allowed_attachment_exts`
        # stays as None.

    # ------------------------------------------------------------------ #
    # Attachment validation & preparation
    # ------------------------------------------------------------------ #

    def _validate_attachment_path(self, path: str) -> None:
        """
        Extend the base validation with Gemini-specific MIME-type checks.

        We reject audio/video upfront; other illegal types are controlled by
        `illegal_attachment_exts`.
        """
        super()._validate_attachment_path(path)

        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        if any(mime.startswith(pref) for pref in self._ILLEGAL_MIME_PREFIXES):
            raise LLMEngineError(
                f"GeminiEngine.attach: MIME type {mime!r} is not supported"
            )

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        """
        Prepare a local path for Gemini: upload once and store the File object.

        The metadata returned here is opaque to the base; only this class needs
        to understand its shape.
        """
        if not os.path.exists(path):
            raise LLMEngineError(f"GeminiEngine.attach: file not found: {path!r}")

        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        ext = os.path.splitext(path)[1].lower()

        # Base + subclass `_validate_attachment_path` already rejected illegal
        # extensions and audio/video MIME types.
        file_obj = self._upload_path(path)
        resource_name = getattr(file_obj, "name", None)

        return {
            "kind": "file",
            "mime": mime,
            "ext": ext,
            "uploaded": True,
            "file_obj": file_obj,
            "resource_name": resource_name,
        }

    def _on_detach(self, meta: Mapping[str, Any]) -> None:
        """
        Delete Gemini file resource if present.

        This is best-effort; errors are ignored.
        """
        name = meta.get("resource_name")
        if not name:
            return
        try:
            self.client.files.delete(name=name)
        except Exception:
            return

    # ------------------------------------------------------------------ #
    # Template hooks for invocation
    # ------------------------------------------------------------------ #

    def _build_provider_payload(
        self,
        messages: List[Dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build the payload for `client.models.generate_content`.

        - `messages` are normalized chat turns (role/content strings).
        - `attachments` is a snapshot of the engine's attachment metadata.
        """
        system_instruction = self._collect_system(messages)
        flat_texts = self._collect_non_system_texts(messages)

        contents: List[Any] = []

        # Attach uploaded files first so they are available for all text turns.
        for _path, meta in attachments.items():
            if meta.get("uploaded") and meta.get("file_obj") is not None:
                contents.append(meta["file_obj"])
            elif meta.get("inlined") and meta.get("inlined_text"):
                contents.append(str(meta["inlined_text"]))

        # Then append plain text turns in order.
        contents.extend([t for t in flat_texts if t])

        return {
            "system_instruction": system_instruction,
            "contents": contents,
        }

    def _call_provider(self, payload: Dict[str, Any]) -> Any:
        """
        Perform a single `models.generate_content` call.

        Retries and backoff are handled by the base `_call_with_retries` wrapper.
        """
        # Use the canonical GenerateContentConfig type from google.genai.types. :contentReference[oaicite:5]{index=5}
        cfg = genai.types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=payload.get("system_instruction") or None,
        )

        return self.client.models.generate_content(
            model=self.model,
            contents=payload["contents"],
            config=cfg,
        )

    def _extract_text(self, response: Any) -> str:
        """
        Extract the assistant's textual reply from a Gen AI SDK response object.

        The SDK exposes a `.text` convenience property for text responses. :contentReference[oaicite:6]{index=6}
        """
        return response.text

    # ------------------------------------------------------------------ #
    # Gemini-specific helpers (not part of the template surface)
    # ------------------------------------------------------------------ #

    def _collect_system(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Join system message contents into a single `system_instruction` string.
        """
        parts = [
            m["content"]
            for m in messages
            if (m.get("role") or "").lower() == "system" and m.get("content")
        ]
        joined = "\n\n".join(parts).strip()
        return joined or None

    def _collect_non_system_texts(
        self, messages: List[Dict[str, str]]
    ) -> List[str]:
        """
        Return a list of non-system message contents, preserving order.

        For Gemini's flat `contents` call style we just send plain strings
        rather than structured chat roles.
        """
        out: List[str] = []
        for m in messages:
            role = (m.get("role") or "").lower()
            if role == "system":
                continue
            txt = m.get("content") or ""
            if txt:
                out.append(txt)
        return out

    def _upload_path(self, path: str) -> Any:
        """
        Upload a local path via the Gemini Files API and return the File object.

        The Gen AI SDK supports passing File objects directly in `contents`. :contentReference[oaicite:7]{index=7}
        """
        abs_path = os.path.abspath(path)
        return self.client.files.upload(file=abs_path)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def to_dict(self) -> OrderedDict[str, Any]:
        """
        Diagnostic snapshot for GeminiEngine: provider + model + temperature.

        Keeps output minimal to avoid leaking client or API keys.
        """
        base = super().to_dict()
        base.update(
            OrderedDict(
                model=self.model,
                temperature=self.temperature,
            )
        )
        return base

# ── MISTRAL ─────────────────────────────
class MistralEngine(LLMEngine):
    """
    Mistral adapter using: **upload → sign → attach URL parts**.

    Flow per call
    -------------
    1) Attachments are prepared via `attach(path)`:
       - PDFs  → upload + sign → `{ "kind": "pdf", "signed_url": ... }`
       - Images → upload + sign → `{ "kind": "image", "signed_url": ... }`
       - Text/code → read + inline → `{ "kind": "text", "inlined_text": ... }`
    2) `invoke(messages)` (from the base `LLMEngine`) will:
       - normalize chat messages (role/content strings),
       - snapshot current attachments,
       - call `_build_provider_payload` to:
         * convert messages into Mistral's chat schema
         * ensure the last user message has a parts array
         * append inline text and signed URL parts
       - call `_call_with_retries` → `_call_provider`
       - call `_extract_text` to normalize the response.
    3) `detach(path)` triggers best-effort deletion of uploaded files via
       `_on_detach`, which calls `client.files.delete(file_id=...)`.
    """

    _IMAGE_EXTS: tuple[str, ...] = (
        ".png", ".jpg", ".jpeg",
        ".webp", ".gif", ".bmp",
        ".tif", ".tiff", ".heic",
    )
    _ILLEGAL_EXTS: set[str] = {
        ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",  # archives
        ".exe", ".dll", ".so", ".bin", ".o",  # executables/binaries
        ".db", ".sqlite",  # databases
        ".h5", ".pt", ".pth", ".onnx",  # model weights
    }
    _ILLEGAL_MIME_PREFIXES: tuple[str, ...] = ("audio/", "video/")

    def __init__(
        self,
        model: str = "mistral-medium-latest",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        inline_cutoff_chars: int = 200_000,
        extra_illegal_exts: Optional[set[str]] = None,
        *,
        name: Optional[str] = None,
        timeout_seconds: float = 600.0,
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
        retry_backoff_max: float = 8.0,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Mistral chat model identifier (e.g. "mistral-small-latest").
        api_key:
            API key for the Mistral service. If omitted, `MISTRAL_API_KEY` is used.
        temperature:
            Sampling temperature for generation.
        inline_cutoff_chars:
            Soft cap for total inlined text across all attachments. Once exceeded,
            additional text attachments are truncated with a marker.
        extra_illegal_exts:
            Optional extra file extensions to treat as unsupported/illegal on top of
            the base `illegal_attachment_exts`.
        name:
            Optional human-friendly name for this engine instance. Defaults to
            `"mistral:{model}"`.
        timeout_seconds:
            Suggested timeout per completion call (honored by the base engine's
            retry/timeout config where applicable).
        max_retries, retry_backoff_base, retry_backoff_max:
            Passed through to the shared `LLMEngine` retry handler for the chat call.
        """
        if Mistral is None:
            raise RuntimeError(
                "MistralEngine requires the `mistralai` package to be installed."
            )

        super().__init__(
            name=name or f"mistral:{model}",
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_base=retry_backoff_base,
            retry_backoff_max=retry_backoff_max,
        )

        import httpx
        self.client = Mistral(
            api_key=api_key or os.getenv("MISTRAL_API_KEY", ""),
            client=httpx.Client(timeout=self._timeout_seconds),
        )
        self.model = model
        self.temperature = float(temperature)
        self.inline_cutoff_chars = int(inline_cutoff_chars)

        # Merge subclass-specific illegal extensions into the base policy.
        merged_illegal = set(self.illegal_attachment_exts) | set(self._ILLEGAL_EXTS)
        if extra_illegal_exts:
            merged_illegal |= set(extra_illegal_exts)
        self.illegal_attachment_exts = merged_illegal
        # We leave allowed_attachment_exts as None (blacklist-based policy).

    # ------------------------------------------------------------------ #
    # Attachment validation & preparation
    # ------------------------------------------------------------------ #

    def _validate_attachment_path(self, path: str) -> None:
        """
        Extend the base validation with Mistral-specific MIME-type checks.

        We reject audio/video upfront; other illegal types are controlled by
        `illegal_attachment_exts`.
        """
        super()._validate_attachment_path(path)

        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        if any(mime.startswith(pref) for pref in self._ILLEGAL_MIME_PREFIXES):
            raise LLMEngineError(
                f"MistralEngine.attach: MIME type {mime!r} is not supported"
            )

    def _prepare_attachment(self, path: str) -> Dict[str, Any]:
        """
        Prepare a local path for Mistral: upload/sign or inline as text/code.

        - PDFs/images → upload + sign; store `file_id` + `signed_url`.
        - Text/code → read & inline, respecting `inline_cutoff_chars`.
        """
        try:
            kind = self._classify_kind(path)
            mime, _ = mimetypes.guess_type(path)
            mime = mime or ""
            ext = os.path.splitext(path)[1].lower()

            if kind in ("pdf", "image"):
                file_id = self._upload_file(path)
                signed_url = self._sign_file(file_id)
                return {
                    "kind": kind,
                    "mime": mime,
                    "ext": ext,
                    "uploaded": True,
                    "file_id": file_id,
                    "signed_url": signed_url,
                }

            # text/code → inline
            text = self._read_text_file(path)
            if len(text) > self.inline_cutoff_chars:
                text = text[: self.inline_cutoff_chars] + "\n…[truncated]\n"
            return {
                "kind": "text",
                "mime": mime,
                "ext": ext,
                "inlined": True,
                "inlined_text": text,
            }
        except LLMEngineError:
            raise
        except Exception as exc:
            raise LLMEngineError(
                f"MistralEngine._prepare_attachment failed for {path!r}"
            ) from exc

    def _on_detach(self, meta: Dict[str, Any]) -> None:
        """
        Delete Mistral file resource if present (best-effort).

        Errors are swallowed; the base engine logs detach errors at debug level.
        """
        file_id = meta.get("file_id")
        if not file_id:
            return
        try:
            self.client.files.delete(file_id=file_id)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Template hooks for invocation
    # ------------------------------------------------------------------ #

    def _build_provider_payload(
        self,
        messages: List[Dict[str, str]],
        attachments: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """
        Map normalized messages + prepared attachments to Mistral chat schema.

        - `messages` are already validated and have lowercase `role` and string `content`.
        - `attachments` is a snapshot of `self._attachments` at invoke time.
        """
        # Start from normalized messages.
        native: List[Dict[str, Any]] = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        # Ensure the last user message is a parts array we can extend.
        user_idx = self._ensure_user_parts(native)
        parts = native[user_idx]["content"]

        # Inline text/code with cutoff from persistent attachments.
        total_inlined = 0
        cutoff_marker = "\n[Inline cutoff reached; additional text files omitted]\n"
        for path, meta in attachments.items():
            kind = meta.get("kind")
            inlined_text = meta.get("inlined_text")
            if kind == "text" and inlined_text:
                budget = self.inline_cutoff_chars - total_inlined
                if budget <= 0:
                    if total_inlined == self.inline_cutoff_chars:
                        parts.append({"type": "text", "text": cutoff_marker})
                        total_inlined += len(cutoff_marker)
                    continue

                text = inlined_text
                if len(text) > budget:
                    text = text[:budget] + "\n…[truncated]\n"
                header = f"\n[Inlined file: {os.path.basename(path)}]\n"
                parts.append({"type": "text", "text": header + text})
                total_inlined += len(text)

        # Attach signed URLs for PDFs & images from persistent attachments.
        for _, meta in attachments.items():
            kind = meta.get("kind")
            signed_url = meta.get("signed_url")
            if kind == "pdf" and signed_url:
                parts.append({"type": "document_url", "document_url": signed_url})
            elif kind == "image" and signed_url:
                parts.append({"type": "image_url", "image_url": signed_url})

        return {"messages": native}

    def _call_provider(self, payload: Any) -> Any:
        """
        Single Mistral chat completion call.

        Retries and error wrapping are handled by the shared `LLMEngine` template.
        """
        return self.client.chat.complete(
            model=self.model,
            messages=payload["messages"],
            temperature=self.temperature,
            
        )

    def _extract_text(self, response: Any) -> str:
        """
        Extract assistant text from a Mistral chat completion response.

        `response.choices[0].message.content` may be a string or a list of chunks.
        """
        msg = getattr(response.choices[0].message, "content", "")
        if isinstance(msg, list):
            msg = "".join(
                c.get("text", "") if isinstance(c, dict) else str(c) for c in msg
            )
        return (msg or "").strip()

    # ------------------------------------------------------------------ #
    # Helpers: classification / IO / uploads
    # ------------------------------------------------------------------ #

    def _classify_kind(self, path: str) -> str:
        """
        Classify `path` into 'pdf' | 'image' | 'text'.

        Base `_validate_attachment_path` has already checked existence and coarse
        extension policy; here we only bucket by type.
        """
        if not os.path.exists(path):
            raise LLMEngineError(f"MistralEngine.attach: file not found: {path!r}")

        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf" or mime == "application/pdf":
            return "pdf"
        if ext in self._IMAGE_EXTS or mime.startswith("image/"):
            return "image"
        # Fallback: treat as text/code and attempt to inline.
        return "text"

    def _ensure_user_parts(self, native: List[Dict[str, Any]]) -> int:
        """
        Ensure there is a user message with `content` as a parts list.

        - If none exists: append an empty user turn.
        - If it's a string: convert to `[{\"type\": \"text\", \"text\": ...}]`.
        """
        idx = next(
            (
                i
                for i in range(len(native) - 1, -1, -1)
                if (native[i].get("role") or "").lower() == "user"
            ),
            None,
        )
        if idx is None:
            native.append({"role": "user", "content": []})
            idx = len(native) - 1

        content = native[idx].get("content", "")
        if isinstance(content, list):
            # already parts
            return idx

        parts: List[Dict[str, Any]] = []
        if isinstance(content, str) and content:
            parts.append({"type": "text", "text": content})
        native[idx]["content"] = parts
        return idx

    def _read_text_file(self, path: str) -> str:
        """Read a local file as UTF-8 (with replacement)."""
        try:
            with open(path, "rb") as f:
                raw = f.read()
            return raw.decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - defensive
            return f"[Error reading file '{os.path.basename(path)}': {exc}]"

    def _upload_file(self, path: str) -> str:
        """Upload to Mistral Files; return file handle ID."""
        with open(path, "rb") as f:
            up = self.client.files.upload(
                file={"file_name": os.path.basename(path), "content": f},
                purpose="ocr",  # suitable for PDFs/images; used for doc/image understanding
            )
        return up.id  # string handle

    def _sign_file(self, file_id: str) -> str:
        """
        Obtain a signed URL for an uploaded file.

        If this call fails, the attachment fails; no additional retries are
        performed here (chat-level retries are handled by the base engine).
        """
        try:
            signed = self.client.files.get_signed_url(file_id=file_id)
            return signed.url
        except Exception as exc:
            raise LLMEngineError(
                f"MistralEngine._sign_file failed for file_id {file_id!r}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def to_dict(self) -> OrderedDict[str, Any]:
        """
        Diagnostic snapshot for MistralEngine: provider + requested knobs.

        Includes non-secret configuration only.
        """
        base = super().to_dict()
        base.update(
            OrderedDict(
                model=self.model,
                temperature=self.temperature,
                inline_cutoff_chars=self.inline_cutoff_chars,
            )
        )
        return base

# ── LLAMA.CPP (local; no remote file store) ────────────────────────────────────
class LlamaCppEngine(LLMEngine):
    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,         # new
        n_threads_batch: Optional[int] = None,   # optional new
        verbose: bool = False,
        *,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name or "llama_cpp")

        if Llama is None:
            raise RuntimeError("llama-cpp-python is required for LlamaCppEngine")

        llama_kwargs = dict(
            n_ctx=n_ctx,
            verbose=verbose,
        )
        if n_threads is not None:
            llama_kwargs["n_threads"] = n_threads
        if n_threads_batch is not None:
            llama_kwargs["n_threads_batch"] = n_threads_batch

        if model_path:
            self.llm = Llama(model_path=model_path, **llama_kwargs)
        elif repo_id and filename:
            self.llm = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                **llama_kwargs,
            )
        else:
            raise LLMEngineError(
                "LlamaCppEngine requires either `model_path` or both `repo_id` and `filename`."
            )


        self.n_ctx = int(n_ctx)
        self.verbose = bool(verbose)
        self.model_path = model_path
        self.repo_id = repo_id
        self.filename = filename

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
        # We simply pass the messages through unchanged.
        return {"messages": messages}

    def _call_provider(self, payload: Any) -> Any:
        """
        Perform a single llama.cpp chat completion call.

        Retries and error-wrapping are handled by `LLMEngine`'s `_call_with_retries`.
        """
        if self.llm is None:
            raise LLMEngineError("LlamaCppEngine: model is not loaded.")
        return self.llm.create_chat_completion(messages=payload["messages"])

    def _extract_text(self, response: Any) -> str:
        """
        Extract assistant text from a llama.cpp chat completion response.

        Expected structure (matching llama-cpp-python's OpenAI-compatible API):
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

    # ------------------------------------------------------------------ #
    # Attachments: explicitly unsupported
    # ------------------------------------------------------------------ #

    def _prepare_attachment(self, path: str) -> Mapping[str, Any]:
        """
        LlamaCppEngine does not support attachments.

        Any call to `attach(path)` will fail via this method.
        """
        raise LLMEngineError(
            f"{type(self).__name__} does not support attachments; "
            "use plain text in messages instead."
        )

    # No `_on_detach` override is required; the base implementation is a no-op.

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def to_dict(self) -> OrderedDict[str, Any]:
        """
        Diagnostic snapshot for LlamaCppEngine.

        Includes n_ctx, verbose, and any of model_path/repo_id/filename that are set.
        """
        base = super().to_dict()
        base.update(
            OrderedDict(
                model_path=self.model_path,
                repo_id=self.repo_id,
                filename=self.filename,
                n_ctx=self.n_ctx,
                verbose=self.verbose,
            )
        )
        return base


# ── PLACEHOLDERS (keep the same abstract contract) ─────────────────────────────
class AzureOpenAIEngine(LLMEngine):
    """
    Placeholder for an Azure OpenAI adapter.

    This class documents the intended constructor/contract but provides **no**
    implementation: `upload`, `delete`, and `invoke` are intentionally unimplemented.
    """
    pass


class BedrockEngine(LLMEngine):
    """
    Placeholder for an AWS Bedrock adapter.

    This class documents the intended constructor/contract but provides **no**
    implementation: `upload`, `delete`, and `invoke` are intentionally unimplemented.
    """
    pass
