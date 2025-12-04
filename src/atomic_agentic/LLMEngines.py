from __future__ import annotations
# LLMEngines.py
# Engines are stateless adapters around provider SDKs.
# The Agent owns conversation history and a map of local path -> provider handle.


import os, time, random
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional, Mapping
import os, mimetypes
from collections import OrderedDict

# Provider SDKs
try: from openai import OpenAI
except: pass
try: from google import genai
except: pass
try: from mistralai import Mistral
except: pass
try: from llama_cpp import Llama
except: pass

__all__ = ["GeminiEngine", "LlamaCppEngine", "LLMEngine", "MistralEngine", "OpenAIEngine"]

# ── ABSTRACT ENGINE ───────────────────────────────────────────────────────────
class LLMEngine(ABC):
    """
    Abstract, **stateless** interface for LLM provider adapters.

    Contract
    --------
    - `invoke(messages: list[str, str]) -> str` must be implemented by subclasses.
      * `messages`: a list of `{"role": str, "content": str}` dicts in chat order
        (roles typically: "system", "user", "assistant").

    Expectations for concrete engines
    ---------------------------------
    1) Validate each file path and raise `TypeError` for unsupported/illegal types.
    2) Decide how to incorporate files per provider:
       - PDFs/images: upload and attach in the provider's format; delete after the call.
       - Text/code: either upload when supported or inline with a sensible size cutoff.
    3) Remain stateless: temporary uploads created inside `invoke` should be cleaned up
       best-effort in a `finally:` block.
    4) Return a plain `str` containing the assistant's text response (trimmed).

    Hints for implementations
    -------------------------
    Engines often keep small knobs on `__init__`, e.g.:
      - `inline_cutoff_chars`: max characters of text to inline.
      - `reject_exts`: extra extensions to reject early.
      - `vision_required`: whether images/PDFs imply a vision model.
    Helper methods like `_classify_path`, `_upload`, `_cleanup` are recommended but
    not enforced here.

    Baseline “illegal” examples (guidance, not binding)
    ---------------------------------------------------
    Consider rejecting:
      - MIME prefixes: `"audio/"`, `"video/"`
      - Common opaque/exec/archive/db/weights:
        `.zip .tar .gz .tgz .rar .7z .exe .dll .so .bin .o .db .sqlite .h5 .pt .pth .onnx`
    """

    # Optional shared guidelines (concrete engines may override in __init__)
    inline_cutoff_chars: int = 200_000
    illegal_exts: set = {
        ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",
        ".exe", ".dll", ".so", ".bin", ".o",
        ".db", ".sqlite",
        ".h5", ".pt", ".pth", ".onnx",
    }
    illegal_mime_prefixes: tuple = ("audio/", "video/")

    def __init__(self) -> None:
        # persistent attachments mapping: path -> metadata
        self._attachments: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def invoke(self, messages: list[dict]) -> str:
        """
        Run a single request against the backing provider.

        Subclasses **must**:
        - Upload/inline/attach as required by the provider.
        - Best-effort delete temporary resources.
        - Return the assistant's message text.
        """
        raise NotImplementedError("LLMEngine.invoke must be implemented by subclasses")

    def to_dict(self) -> OrderedDict[str, Any]:
        """Base diagnostic: only expose the provider name to avoid leaking internals.

        Subclasses should extend or merge this dict with provider-specific fields.
        """
        return OrderedDict(provider = type(self).__name__)

    def attach(self, path: str) -> Dict[str, Any]:
        """Attach a local path to this engine and prepare provider metadata.

        Returns the metadata dict stored for this path. Idempotent: re-attaching
        an already-attached path returns the existing metadata.
        """
        if not isinstance(path, str) or not path:
            raise TypeError("LLMEngine.attach: path must be a non-empty string")
        if path in self._attachments:
            return self._attachments[path]

        meta: Dict[str, Any] = {"path": path, "provider": type(self).__name__, "created_at": time.time()}
        # Delegate engine-specific preparation (upload/inlining/classification)
        try:
            prepared = self._prepare_attachment(path)
            if not isinstance(prepared, dict):
                raise TypeError("_prepare_attachment must return a dict of metadata")
            meta.update(prepared)
        except Exception as e:  # propagate to caller (they may want to handle)
            meta["error"] = str(e)
            # Do not store failing metadata; raise so caller knows attach failed.
            raise

        self._attachments[path] = meta
        return meta

    def detach(self, path: str) -> bool:
        """Remove an attachment from this engine. Best-effort remote cleanup via
        `_on_detach` is attempted; subclasses may override `_on_detach` to implement
        provider-specific deletion.
        Returns True if removed, False if path was not attached.
        """
        meta = self._attachments.pop(path, None)
        if not meta:
            return False
        try:
            self._on_detach(meta)
        except Exception:
            # best-effort: ignore remote deletion errors
            pass
        return True

    @property
    def attachments(self) -> Dict[str, Dict[str, Any]]:
        """Return a shallow copy of the attachments mapping."""
        return dict(self._attachments)

    def get_attachment(self, path: str) -> Optional[Dict[str, Any]]:
        return self._attachments.get(path)

    # Hooks for subclasses to implement provider-specific behavior
    @abstractmethod
    def _prepare_attachment(self, path: str) -> Dict[str, Any]:
        """Prepare a local path for reuse with this engine.

        Subclasses should override this to return metadata describing how the
        engine will refer to the file (e.g., `{'kind':'pdf','file_id':...}` or
        `{'kind':'text','inlined_text':...}`). The base implementation raises
        `NotImplementedError` to encourage engines to provide an implementation.
        """
    @abstractmethod
    def _on_detach(self, meta: Dict[str, Any]) -> None:
        """Optional hook called when an attachment is detached. Subclasses may
        implement remote deletion here. Default is a no-op."""


# ── OPENAI (Responses API + Chat fallback) ─────────────────────────────────────
class OpenAIEngine(LLMEngine):
    """
    OpenAI adapter using the **Responses API**.

    File policy (per call)
    ----------------------
    - **PDFs** → upload to Files API; attach `{ "type": "input_file", "file_id": ... }`
    - **Images** → upload to Files API; attach `{ "type": "input_image", "file_id": ... }`
    - **Text/Code** (e.g., `.txt/.md/.py/.json/.html/.js/.java/...`) → read and inline as
      `{ "type": "input_text", "text": ... }` subject to a global inline cutoff.
    - Unsupported (audio/video/archives/executables/db/weights) → `TypeError`.

    Other notes
    -----------
    - System prompts are provided via top-level `instructions`.
    - Assistant history is encoded as `output_text`; user/other turns as `input_text`.
    - Any files uploaded inside `invoke` are best-effort deleted in a `finally:` block.
    """

    # Default config/knobs (override in __init__ if desired)
    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic")
    _ILLEGAL_EXTS = {
        ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",          # archives
        ".exe", ".dll", ".so", ".bin", ".o",                    # executables/binaries
        ".db", ".sqlite",                                       # databases
        ".h5", ".pt", ".pth", ".onnx",                          # model weights
    }
    _ILLEGAL_MIME_PREFIXES = ("audio/", "video/")

    def __init__(self, model: str, api_key: str | None = None,
                 temperature: float = 0.1, inline_cutoff_chars: int = 200_000,
                 extra_illegal_exts: set[str] | None = None):
        super().__init__()
        self.llm = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = float(temperature)
        self.inline_cutoff_chars = int(inline_cutoff_chars)
        # allow caller to expand the illegal set
        self.illegal_exts = set(self._ILLEGAL_EXTS) | (set(extra_illegal_exts or ()))

    # ── Public API ───────────────────────────────────────────────
    def invoke(self, messages: list[dict]) -> str:
        # Attachments are expected to be prepared via `engine.attach(path)` ahead
        # of calling `invoke`.
        
        # 1) Build Responses blocks: instructions + role-aware turns
        instructions = self._collect_instructions(messages)
        blocks = self._build_role_blocks(messages)

        # Ensure there is a user block to carry files/inline text
        user_idx = self._ensure_user_block(blocks)

        # 2) Append prepared attachments (persistent) to the user block
        attachments = self._attachments
        for path, meta in attachments.items():
            kind = meta.get("kind", "text")
            # Inlined text
            if meta.get("inlined") or kind == "text":
                text = meta.get("inlined_text", "")
                if not text:
                    continue
                header = f"\n[Inlined file: {os.path.basename(path)}]\n"
                blocks[user_idx]["content"].append({"type": "input_text", "text": header + text})
                continue

            # Uploaded handle (file_id)
            if meta.get("uploaded") and meta.get("file_id"):
                fid = meta.get("file_id")
                part_type = "input_file" if kind == "pdf" else "input_image"
                blocks[user_idx]["content"].append({"type": part_type, "file_id": fid})
                continue

            # Fallback: try to attach local content directly (best-effort)
            try:
                self._attach_local_path(blocks[user_idx]["content"], path)
            except Exception:
                # ignore failing attachment for this path
                continue

        # 3) Call Responses API
        if "gpt-5" not in self.model.lower():
            resp = self.llm.responses.create(
                model=self.model,
                instructions=instructions,
                input=blocks,
                temperature=self.temperature,
            )
        else:
            resp = self.llm.responses.create(
                model=self.model,
                instructions=instructions,
                input=blocks,
            )
        return (getattr(resp, "output_text", "") or "").strip()

    # ── Helpers: classification / validation / IO / uploads ─────
    def _classify_or_raise(self, path: str) -> tuple[str, str]:
        """Classify `path` into ('pdf'|'image'|'text'); raise on illegal/unknown file types."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"OpenAIEngine: file not found: {path}")

        # Infer by MIME and extension
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        ext = os.path.splitext(path)[1].lower()

        # Illegal quick checks
        if ext in self.illegal_exts or any(mime.startswith(pref) for pref in self._ILLEGAL_MIME_PREFIXES):
            raise TypeError(f"OpenAIEngine: illegal/unsupported file type for path: {path}")

        # Supported buckets
        if ext == ".pdf" or mime == "application/pdf":
            return (path, "pdf")

        if ext in self._IMAGE_EXTS or mime.startswith("image/"):
            return (path, "image")

        # default → treat as text/code (we'll inline)
        return (path, "text")

    # ---------------- OpenAI attachment preparation / cleanup ----------------
    def _prepare_attachment(self, path: str) -> Dict[str, Any]:
        """Engine-specific preparation for a local path.

        Returns a metadata dict describing how the engine will reference the file.
        """
        # Validate & classify
        p, kind = self._classify_or_raise(path)
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        ext = os.path.splitext(path)[1].lower()

        if kind in ("pdf", "image"):
            # upload once and keep the file_id
            fid = self._upload_file(path)
            return {
                "kind": kind,
                "mime": mime,
                "ext": ext,
                "uploaded": True,
                "file_id": fid,
            }

        # text/code → inline
        text = self._read_text_file(path)
        if len(text) > self.inline_cutoff_chars:
            text = text[: self.inline_cutoff_chars] + "\n…[truncated]\n"
        return {"kind": "text", "inlined": True, "inlined_text": text}

    def _on_detach(self, meta: Dict[str, Any]) -> None:
        """Attempt to delete any remote file created by OpenAI Files API."""
        fid = meta.get("file_id")
        if not fid:
            return
        try:
            self.llm.files.delete(fid)
        except Exception:
            # best-effort
            pass

    def _collect_instructions(self, messages: list[dict]) -> str | None:
        """Concatenate all `system` message contents into a single instructions string (or None)."""
        parts = [m["content"] for m in messages
                 if (m.get("role") or "").lower() == "system" and m.get("content")]
        return "\n\n".join(parts) or None

    def _build_role_blocks(self, messages: list[dict]) -> list[dict]:
        """
        Convert chat messages to Responses API blocks:
        - `assistant` turns become `output_text`.
        - All other non-system turns become `input_text`.
        - `system` content is not a block; it is carried via `instructions`.
        """
        blocks: list[dict] = []
        for m in messages:
            role = (m.get("role") or "user").lower()
            if role == "system":
                continue  # carried in instructions
            text = m.get("content") or ""
            if not text and role != "user":
                # keep empty user (we attach into it), skip other empty roles
                continue
            ptype = "output_text" if role == "assistant" else "input_text"
            blocks.append({"role": role, "content": ([{"type": ptype, "text": text}] if text else [])})
        return blocks

    def _ensure_user_block(self, blocks: list[dict]) -> int:
        """Return index of a `user` block; create an empty one at the end if none exists."""
        idx = next((i for i in range(len(blocks) - 1, -1, -1) if blocks[i]["role"] == "user"), None)
        if idx is None:
            blocks.append({"role": "user", "content": []})
            idx = len(blocks) - 1
        return idx

    def _read_text_file(self, path: str) -> str:
        """Read a local file as UTF-8 (with replacement). Return error text if reading fails."""
        try:
            with open(path, "rb") as f:
                raw = f.read()
            return raw.decode("utf-8", errors="replace")
        except Exception as e:
            return f"[Error reading file '{os.path.basename(path)}': {e}]"

    def _upload_file(self, path: str) -> str:
        """Upload a local file to OpenAI Files; return `file_id`."""
        with open(path, "rb") as fp:
            f = self.llm.files.create(file=fp, purpose="assistants")
        return f.id

    # ── Helpers ─────────────────────────────────────────────────
    def _attach_local_path(self, user_parts: List[Dict[str, Any]], path: str) -> None:
        """
        Attach a local path to a User parts list:
        - PDFs → `input_file`
        - Images → `input_image`
        - Everything else → inline text (UTF-8 with replacement), truncated if needed.
        """
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        is_pdf   = mime == "application/pdf" or path.lower().endswith(".pdf")
        is_image = mime.startswith("image/") or path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic"))

        if is_pdf:
            fid = self.upload(path)
            user_parts.append({"type": "input_file", "file_id": fid})
            return

        if is_image:
            fid = self.upload(path)
            user_parts.append({"type": "input_image", "file_id": fid})
            return

        # Fallback: inline as text (read file bytes → UTF-8 w/ replacement)
        try:
            with open(path, "rb") as f:
                raw = f.read()
            text = raw.decode("utf-8", errors="replace")
        except Exception as e:
            text = f"[Error reading file '{os.path.basename(path)}': {e}]"

        if len(text) > self._MAX_INLINE_CHARS:
            text = text[: self._MAX_INLINE_CHARS] + "\n…[truncated]\n"

        # You can optionally prefix the filename for clarity
        header = f"\n[Inlined file: {os.path.basename(path)}]\n"
        user_parts.append({"type": "input_text", "text": header + text})

    def to_dict(self) -> OrderedDict[str, Any]:
        """Diagnostic snapshot for OpenAIEngine: provider + model + temperature.

        Intentionally minimal to avoid leaking client objects or API keys.
        """
        base = super().to_dict()
        base.update(OrderedDict(
            model = self.model,
            temperature = self.temperature,
            inline_cutoff_chars = self.inline_cutoff_chars,
        ))
        return base


# ── LLAMA.CPP (local; no remote file store) ────────────────────────────────────
class LlamaCppEngine(LLMEngine):
    """
    Local llama.cpp adapter.

    Behavior
    --------
    - Loads a local GGUF (via `model_path`) or a pre-trained artifact (`repo_id` + `filename`).
    - Ignores `files` (no remote file store).
    - Uses llama.cpp `create_chat_completion` with the given messages as is.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_id: Optional[str] = None,
        filename: Optional[str] = None,
        n_ctx: int = 2048,
        verbose: bool = False,
    ):
        super().__init__()
        self.llm = None
        if model_path:
            self.llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=verbose)
        elif repo_id and filename:
            self.llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, n_ctx=n_ctx, verbose=verbose)
        else:
            raise ValueError("Must provide either model_path or both repo_id and filename.")
        self.n_ctx = n_ctx
        self.verbose = verbose
        self.model_path = model_path
        self.repo_id = repo_id
        self.filename = filename

    def attach(self, path: str) -> Any:
        """Not supported for local inference."""
        raise NotImplementedError("LlamaCppEngine has no remote file storage.")

    def detach(self, path: Any) -> bool:
        """Not supported for local inference."""
        raise NotImplementedError("LlamaCppEngine has no remote file storage.")
    
    def _prepare_attachment(self, path):
        raise NotImplementedError("LlamaCppEngine has no remote file storage.")
    
    def _on_detach(self, meta: Dict[str, Any]) -> None:
        raise NotImplementedError("LlamaCppEngine has no remote file storage.")

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """Run a local chat completion; `files` are ignored."""
        if not self.llm:
            raise RuntimeError("Llama model not loaded.")
        # local models ignore 'files'; messages are standard chat format
        response = self.llm.create_chat_completion(messages=messages)
        return response["choices"][0]["message"]["content"].strip()

    def to_dict(self) -> OrderedDict[str, Any]:
        """Diagnostic snapshot for LlamaCppEngine.

        Includes provider, n_ctx, verbose, and any of model_path/repo_id/filename that are set.
        Also reports whether a model is loaded.
        """
        base = super().to_dict()
        base.update(OrderedDict(
            model_path = self.model_path,
            repo_id = self.repo_id,
            filename = self.filename,
            n_ctx = self.n_ctx,
            verbose = self.verbose,
        ))
        return base


# ── GEMINI (flat contents: file handle objects + strings) ──────────────────────
class GeminiEngine(LLMEngine):
    """
    Google Gemini adapter (Files API + **flat** `contents` call style).

    Strategy per call
    -----------------
    1) Validate each local path; reject audio/video/archives/executables/etc.
    2) Upload **all** supported files (PDF/image/text/code/Office/csv/xlsx/…)
       using `client.files.upload(...)`; pass returned File objects directly in
       `contents` (no manual inlining required).
    3) Flatten all non-system chat turns to plain strings and append to `contents`.
       Provide system prompts via `system_instruction`.
    4) Best-effort delete uploaded files created during the call in a `finally:` block.
    """

    _ILLEGAL_EXTS = {
        ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",        # archives
        ".exe", ".dll", ".so", ".bin", ".o",                 # executables/binaries
        ".db", ".sqlite",                                    # databases
        ".h5", ".pt", ".pth", ".onnx",                       # model weights
    }
    _ILLEGAL_MIME_PREFIXES = ("audio/", "video/")

    def __init__(self, model: str, api_key: str | None = None,
                 temperature: float = 0.1, extra_illegal_exts: set[str] | None = None):
        super().__init__()
        self.client = genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.model = model
        self.temperature = float(temperature)
        # allow the caller to expand the illegal set
        self.illegal_exts = set(self._ILLEGAL_EXTS) | (set(extra_illegal_exts or ()))

    # ── Public API ───────────────────────────────────────────────
    def invoke(self, messages: list[dict]) -> str:
        """
        Generate content with Gemini using persistent attachments.
        Attachments must be prepared via `engine.attach(path)` before calling invoke.
        """
        
        # Prepare system + flat text turns (order-preserving)
        system_instruction = self._collect_system(messages)
        flat_texts = self._collect_non_system_texts(messages)

        # Build contents from persistent attachments
        contents = []
        for path, meta in self._attachments.items():
            if meta.get("uploaded") and meta.get("file_obj") is not None:
                contents.append(meta["file_obj"])
            elif meta.get("inlined") and meta.get("inlined_text"):
                contents.append(meta["inlined_text"])
            # else: skip
        contents.extend([t for t in flat_texts if t])  # then plain strings

        cfg = genai.types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_instruction or None,
        )
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=cfg,
        )
        return (getattr(resp, "text", None) or "").strip()

    # ── Helpers: validation / preparation / uploads ─────────────
    # ---------------- Gemini attachment preparation / cleanup ----------------
    def _prepare_attachment(self, path: str) -> Dict[str, Any]:
        """Prepare a local path for Gemini: upload once and store File object."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"GeminiEngine: file not found: {path}")
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        ext = os.path.splitext(path)[1].lower()
        if ext in self.illegal_exts or any(mime.startswith(pref) for pref in self._ILLEGAL_MIME_PREFIXES):
            raise TypeError(f"GeminiEngine: illegal/unsupported file type: {path}")

        # Upload all supported files (Gemini prefers upload for all types)
        file_obj = self._upload_path(path)
        resource_name = getattr(file_obj, "name", None)
        return {
            "kind": "file",  # Gemini treats all as file
            "mime": mime,
            "ext": ext,
            "uploaded": True,
            "file_obj": file_obj,
            "resource_name": resource_name,
        }

    def _on_detach(self, meta: Dict[str, Any]) -> None:
        """Delete Gemini file resource if present."""
        name = meta.get("resource_name")
        if not name:
            return
        try:
            self.client.files.delete(name=name)
        except Exception:
            pass

    def _collect_system(self, messages: list[dict]) -> str | None:
        """Join system message contents into a single string (or None)."""
        parts = [m["content"] for m in messages
                 if (m.get("role") or "").lower() == "system" and m.get("content")]
        return "\n\n".join(parts) or None

    def _collect_non_system_texts(self, messages: list[dict]) -> list[str]:
        """Return a list of non-system message contents, preserving order."""
        out = []
        for m in messages:
            role = (m.get("role") or "").lower()
            if role == "system":
                continue
            txt = m.get("content") or ""
            if txt:
                out.append(txt)
        return out

    def _upload_path(self, path: str):
        """Upload a local path via Gemini Files API and return the resulting File object."""
        # Upload local path and return the File object.
        # genai SDK will infer MIME and handle supported types.
        abs_path = os.path.abspath(path)
        return self.client.files.upload(file=abs_path)

    def to_dict(self) -> OrderedDict[str, Any]:
        """Diagnostic snapshot for GeminiEngine: provider + model + temperature.

        Keeps output minimal to avoid leaking client or API keys.
        """
        base = super().to_dict()
        base.update(OrderedDict(
            model = self.model,
            temperature = self.temperature,
        ))
        return base


# ── MISTRAL (Document QnA: upload -> sign -> document_url) ─────────────────────
class MistralEngine(LLMEngine):
    """
    Mistral adapter using: **upload → (eventual) sign → attach URL parts**.

    Flow per call
    -------------
    1) Validate & classify each local path.
       - PDFs → upload + sign → add `{ "type": "document_url", "document_url": ... }`
       - Images → upload + sign → add `{ "type": "image_url", "image_url": ... }`
       - Text/code → read + inline `{ "type": "text", "text": ... }` (respect cutoff)
    2) Convert the last user message to a parts array if needed, then append file parts.
    3) Call `chat.complete`.
    4) Best-effort delete uploaded files created by this call.

    Note: signing can be briefly eventual-consistent after upload; a small retry loop
    with backoff is used when signing returns 404.
    """

    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic")
    _ILLEGAL_EXTS = {
        ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",          # archives
        ".exe", ".dll", ".so", ".bin", ".o",                    # executables/binaries
        ".db", ".sqlite",                                       # databases
        ".h5", ".pt", ".pth", ".onnx",                          # model weights
    }
    _ILLEGAL_MIME_PREFIXES = ("audio/", "video/")

    def __init__(self,
                 model: str = "mistral-medium-latest",
                 api_key: str | None = None,
                 temperature: float = 0.1, inline_cutoff_chars: int = 200_000,
                 extra_illegal_exts: set[str] | None = None,
                 retry_sign_attempts: int = 5,
                 retry_base_delay: float = 0.3):
        super().__init__()
        self.client = Mistral(api_key=api_key or os.getenv("MISTRAL_API_KEY"))
        self.model = model
        self.temperature = float(temperature)
        self.inline_cutoff_chars = int(inline_cutoff_chars)
        self.retry_sign_attempts = int(retry_sign_attempts)
        self.retry_base_delay = float(retry_base_delay)
        # allow caller to expand the illegal set
        self.illegal_exts = set(self._ILLEGAL_EXTS) | (set(extra_illegal_exts or ()))

    # ── Public API ───────────────────────────────────────────────
    def invoke(self, messages: list[dict]) -> str:
        """
        Complete a Mistral chat turn using persistent attachments.
        Attachments must be prepared via `engine.attach(path)` before calling invoke.
        """
        
        # Start from the incoming messages
        native: list[dict] = [
            {"role": (m.get("role") or "user"), "content": (m.get("content") or "")}
            for m in messages
        ]

        # Ensure the last user message is a parts array we can extend
        user_idx = self._ensure_user_parts(native)
        parts = native[user_idx]["content"]

        # Inline text/code with cutoff from persistent attachments
        total_inlined = 0
        for path, meta in self._attachments.items():
            kind = meta.get("kind")
            if kind == "text" and meta.get("inlined_text"):
                text = meta["inlined_text"]
                budget = self.inline_cutoff_chars - total_inlined
                if budget <= 0:
                    if total_inlined == self.inline_cutoff_chars:
                        parts.append({"type": "text", "text": "\n[Inline cutoff reached; additional text files omitted]\n"})
                        total_inlined += len("[Inline cutoff reached; additional text files omitted]\n")
                    continue
                if len(text) > budget:
                    text = text[:budget] + "\n…[truncated]\n"
                header = f"\n[Inlined file: {os.path.basename(path)}]\n"
                parts.append({"type": "text", "text": header + text})
                total_inlined += len(text)

        # Attach signed URLs for PDFs & images from persistent attachments
        for path, meta in self._attachments.items():
            kind = meta.get("kind")
            if kind == "pdf" and meta.get("signed_url"):
                parts.append({"type": "document_url", "document_url": meta["signed_url"]})
            elif kind == "image" and meta.get("signed_url"):
                parts.append({"type": "image_url", "image_url": meta["signed_url"]})

        # Make the chat call
        res = self.client.chat.complete(
            model=self.model,
            messages=native,
            temperature=self.temperature,
        )
        msg = res.choices[0].message.content
        if isinstance(msg, list):
            msg = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in msg])
        return (msg or "").strip()
    # ---------------- Mistral attachment preparation / cleanup ----------------
    def _prepare_attachment(self, path: str) -> Dict[str, Any]:
        """Prepare a local path for Mistral: upload and sign for PDFs/images, inline for text/code."""
        p, kind = self._classify_or_raise(path)
        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        ext = os.path.splitext(path)[1].lower()

        if kind in ("pdf", "image"):
            file_id = self._upload_file(path)
            signed_url = self._sign_with_retry(file_id, self.retry_sign_attempts, self.retry_base_delay)
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
        return {"kind": "text", "inlined": True, "inlined_text": text}

    def _on_detach(self, meta: Dict[str, Any]) -> None:
        """Delete Mistral file resource if present."""
        file_id = meta.get("file_id")
        if not file_id:
            return
        try:
            self.client.files.delete(file_id=file_id)
        except Exception:
            pass

    # ── Helpers: validation / classification / IO / uploads ─────
    def _classify_or_raise(self, path: str) -> tuple[str, str]:
        """Classify `path` into ('pdf'|'image'|'text'); raise on illegal types or missing files."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"MistralEngine: file not found: {path}")

        mime, _ = mimetypes.guess_type(path)
        mime = mime or ""
        ext = os.path.splitext(path)[1].lower()

        # Illegal types
        if ext in self.illegal_exts or any(mime.startswith(pref) for pref in self._ILLEGAL_MIME_PREFIXES):
            raise TypeError(f"MistralEngine: illegal/unsupported file type: {path}")

        # Supported buckets
        if ext == ".pdf" or mime == "application/pdf":
            return (path, "pdf")
        if ext in self._IMAGE_EXTS or mime.startswith("image/"):
            return (path, "image")
        return (path, "text")

    def _ensure_user_parts(self, native: list[dict]) -> int:
        """
        Ensure there is a user message with `content` as a parts list.
        - If none exists: append an empty user turn.
        - If it's a string: convert to `[{"type": "text", "text": ...}]` (preserving content).
        """
        idx = next((i for i in range(len(native) - 1, -1, -1) if (native[i].get("role") or "").lower() == "user"), None)
        if idx is None:
            native.append({"role": "user", "content": []})
            idx = len(native) - 1

        content = native[idx].get("content", "")
        if isinstance(content, list):
            # already parts
            return idx
        # convert string → single text part (if any)
        parts = []
        if isinstance(content, str) and content:
            parts.append({"type": "text", "text": content})
        native[idx]["content"] = parts
        return idx

    def _read_text_file(self, path: str) -> str:
        """Read a local file as UTF-8 (with replacement). Return error text on failure."""
        try:
            with open(path, "rb") as f:
                raw = f.read()
            return raw.decode("utf-8", errors="replace")
        except Exception as e:
            return f"[Error reading file '{os.path.basename(path)}': {e}]"

    def _upload_file(self, path: str) -> str:
        """Upload to Mistral Files for OCR/doc understanding; return file handle ID."""
        with open(path, "rb") as f:
            up = self.client.files.upload(
                file={"file_name": os.path.basename(path), "content": f},
                purpose="ocr",  # suitable for PDFs/images; used for doc/image understanding
            )
        return up.id  # string handle

    def _sign_with_retry(self, file_id: str, max_retries: int, base_delay: float) -> str:
        """
        Obtain a signed URL for a previously uploaded file, retrying on transient 404s.

        Strategy: exponential backoff with small jitter until success or `max_retries` exhausted.
        """
        for attempt in range(max_retries):
            try:
                signed = self.client.files.get_signed_url(file_id=file_id)
                return signed.url
            except Exception as e:
                msg = str(e)
                is_404 = ("Status 404" in msg) or ("No file matches the given query" in msg)
                if not is_404 or attempt == max_retries - 1:
                    raise
                # exponential backoff with tiny jitter
                delay = base_delay * (2 ** attempt)
                try:
                    time.sleep(delay + (random.random() * 0.1))
                except Exception:
                    # if time/random not available for some reason, sleep best-effort minimal
                    pass

    def to_dict(self) -> OrderedDict[str, Any]:
        """Diagnostic snapshot for MistralEngine: provider + requested knobs.

        Minimal and safe: excludes client and keys.
        """
        base = super().to_dict()
        base.update(OrderedDict(
            model = self.model,
            temperature = self.temperature,
            inline_cutoff_chars = self.inline_cutoff_chars,
            retry_sign_attempts = self.retry_sign_attempts,
            retry_base_delay = self.retry_base_delay,
        ))
        return base

# ── PLACEHOLDERS (keep the same abstract contract) ─────────────────────────────
class AzureOpenAIEngine(LLMEngine):
    """
    Placeholder for an Azure OpenAI adapter.

    This class documents the intended constructor/contract but provides **no**
    implementation: `upload`, `delete`, and `invoke` are intentionally unimplemented.
    """
    def __init__(self, api_key: str, endpoint: str, api_version: str, model: str):
        super().__init__()
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.model = model

    def to_dict(self) -> Dict[str, Any]:
        """Return a small diagnostic snapshot for AzureOpenAIEngine without secrets."""
        return {"provider": type(self).__name__, "endpoint": getattr(self, "endpoint", None), "api_version": getattr(self, "api_version", None), "model": getattr(self, "model", None), "has_api_key": bool(getattr(self, "api_key", None))}

    def upload(self, path: str) -> Any:
        raise NotImplementedError("AzureOpenAIEngine.upload not implemented")

    def delete(self, handle: Any) -> bool:
        raise NotImplementedError("AzureOpenAIEngine.delete not implemented")

    def invoke(self, messages: List[Dict[str, str]], files: Optional[List[Any]] = None) -> str:
        raise NotImplementedError("AzureOpenAIEngine.invoke not implemented")


class BedrockEngine(LLMEngine):
    """
    Placeholder for an AWS Bedrock adapter.

    This class documents the intended constructor/contract but provides **no**
    implementation: `upload`, `delete`, and `invoke` are intentionally unimplemented.
    """
    def __init__(self, access_key: str, secret_key: str, region: str, model: str):
        super().__init__()
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.model = model

    def to_dict(self) -> Dict[str, Any]:
        """Return a small diagnostic snapshot for BedrockEngine without secrets."""
        return {"provider": type(self).__name__, "region": getattr(self, "region", None), "model": getattr(self, "model", None), "has_access_key": bool(getattr(self, "access_key", None))}

    def upload(self, path: str) -> Any:
        raise NotImplementedError("BedrockEngine.upload not implemented")

    def delete(self, handle: Any) -> bool:
        raise NotImplementedError("BedrockEngine.delete not implemented")

    def invoke(self, messages: List[Dict[str, str]], files: Optional[List[Any]] = None) -> str:
        raise NotImplementedError("BedrockEngine.invoke not implemented")
