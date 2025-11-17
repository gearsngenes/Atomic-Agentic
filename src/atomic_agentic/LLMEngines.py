from __future__ import annotations
# LLMEngines.py
# Engines are stateless adapters around provider SDKs.
# The Agent owns conversation history and a map of local path -> provider handle.


import os, time, random
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional

from dotenv import load_dotenv

# Provider SDKs
try: from openai import OpenAI
except: pass
try: from google import genai
except: pass
try: from mistralai import Mistral
except: pass
try: from llama_cpp import Llama
except: pass
# ── ENV / CONSTANTS ───────────────────────────────────────────────────────────
load_dotenv()
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_GEMINI_KEY     = os.getenv("GOOGLE_API_KEY")
DEFAULT_MISTRAL_KEY    = os.getenv("MISTRAL_API_KEY")


# ── ABSTRACT ENGINE ───────────────────────────────────────────────────────────
class LLMEngine:
    """
    Abstract, **stateless** interface for LLM provider adapters.

    Contract
    --------
    - `invoke(messages, file_paths=None) -> str` must be implemented by subclasses.
      * `messages`: a list of `{"role": str, "content": str}` dicts in chat order
        (roles typically: "system", "user", "assistant").
      * `file_paths`: optional list of local file paths the *Agent* wants to include.
        The *Agent* is the only stateful owner of these paths; engines decide how to
        upload/inline/ignore per provider.

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

    @abstractmethod
    def invoke(self, messages: list[dict], file_paths: list[str] | None = None) -> str:
        """
        Run a single request against the backing provider.

        Subclasses **must**:
        - Validate `file_paths`.
        - Upload/inline/attach as required by the provider.
        - Best-effort delete temporary resources.
        - Return the assistant's message text.
        """
        raise NotImplementedError("LLMEngine.invoke must be implemented by subclasses")

    def to_dict(self) -> Dict[str, Any]:
        """Base diagnostic: only expose the provider name to avoid leaking internals.

        Subclasses should extend or merge this dict with provider-specific fields.
        """
        return {"provider": type(self).__name__}


# ── OPENAI (Responses API + Chat fallback) ─────────────────────────────────────
import os, mimetypes
from typing import Any, List, Dict, Optional
from openai import OpenAI

DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        self.llm = OpenAI(api_key=api_key or DEFAULT_OPENAI_API_KEY)
        self.model = model
        self.temperature = float(temperature)
        self.inline_cutoff_chars = int(inline_cutoff_chars)
        # allow caller to expand the illegal set
        self.illegal_exts = set(self._ILLEGAL_EXTS) | (set(extra_illegal_exts or ()))

    # ── Public API ───────────────────────────────────────────────
    def invoke(self, messages: list[dict], file_paths: list[str] | None = None) -> str:
        file_paths = list(dict.fromkeys(file_paths or []))  # de-dupe, preserve order

        # 1) Validate and classify paths (raise early for illegal/unsupported)
        classifications = [self._classify_or_raise(p) for p in file_paths]
        # kinds: "pdf" | "image" | "text"

        # 2) Build Responses blocks: instructions + role-aware turns
        instructions = self._collect_instructions(messages)
        blocks = self._build_role_blocks(messages)

        # Ensure there is a user block to carry files/inline text
        user_idx = self._ensure_user_block(blocks)

        # 3) Prepare uploads/inline content
        uploaded_ids: list[str] = []      # track for cleanup
        total_inlined = 0
        try:
            # Inline text/code first (deterministic order)
            for path, kind in classifications:
                if kind != "text":
                    continue
                text = self._read_text_file(path)
                if not text:
                    continue
                budget = self.inline_cutoff_chars - total_inlined
                if budget <= 0:
                    # no more room; add a small marker once
                    if total_inlined == self.inline_cutoff_chars:
                        blocks[user_idx]["content"].append({
                            "type": "input_text",
                            "text": "\n[Inline cutoff reached; additional text files omitted]\n"
                        })
                        total_inlined += len("[Inline cutoff reached; additional text files omitted]\n")
                    continue
                if len(text) > budget:
                    text = text[:budget] + "\n…[truncated]\n"
                header = f"\n[Inlined file: {os.path.basename(path)}]\n"
                blocks[user_idx]["content"].append({"type": "input_text", "text": header + text})
                total_inlined += len(text)

            # Attach PDFs and images via temporary uploads
            for path, kind in classifications:
                if kind not in ("pdf", "image"):
                    continue
                file_id = self._upload_file(path)
                uploaded_ids.append(file_id)
                part_type = "input_file" if kind == "pdf" else "input_image"
                blocks[user_idx]["content"].append({"type": part_type, "file_id": file_id})

            # 4) Call Responses API
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

        finally:
            # 5) Best-effort cleanup for any uploads this call created
            for fid in uploaded_ids:
                try:
                    self.llm.files.delete(fid)
                except Exception:
                    pass

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

    def _attach_uploaded_handle(self, user_parts: List[Dict[str, Any]], file_id: str) -> None:
        """
        Attach an **existing** uploaded file by inspecting its metadata:
        - If PDF → `input_file`
        - If image → `input_image`
        - Otherwise, try to fetch bytes and inline as text (if `.files.content` exists).
          If content retrieval isn't supported by the SDK, raise a clear error.
        """
        # Try to detect from metadata
        filename, mime = "", ""
        try:
            meta = self.llm.files.retrieve(file_id)
            filename = getattr(meta, "filename", None) or getattr(meta, "name", "") or ""
            mime = getattr(meta, "mime_type", None) or ""
        except Exception:
            # Best-effort: infer by id alone (we'll try image/pdf by extension fallback)
            pass

        name_lc = (filename or "").lower()
        is_pdf   = (mime == "application/pdf") or name_lc.endswith(".pdf")
        is_image = (mime.startswith("image/") if mime else False) or name_lc.endswith(
            (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic")
        )

        if is_pdf:
            user_parts.append({"type": "input_file", "file_id": file_id})
            return

        if is_image:
            user_parts.append({"type": "input_image", "file_id": file_id})
            return

        # Non-PDF/non-image: try to download and inline
        text = None
        try:
            # Newer SDKs provide a streaming reader for file content
            stream = self.llm.files.content(file_id)  # may raise if not supported
            data = stream.read() if hasattr(stream, "read") else getattr(stream, "content", None)
            if isinstance(data, bytes):
                text = data.decode("utf-8", errors="replace")
        except Exception:
            # If your SDK/version doesn't support .content, we can't inline a handle-only file.
            raise ValueError(
                f"OpenAIEngine: file_id '{file_id}' is not a PDF/image and this SDK "
                f"cannot fetch contents to inline. Pass a local path instead so I can read() it."
            )

        if text is None:
            raise ValueError(
                f"OpenAIEngine: unable to inline contents for file_id '{file_id}'."
            )

        if len(text) > self._MAX_INLINE_CHARS:
            text = text[: self._MAX_INLINE_CHARS] + "\n…[truncated]\n"

        label = filename or file_id
        header = f"\n[Inlined file (uploaded): {label}]\n"
        user_parts.append({"type": "input_text", "text": header + text})

    @staticmethod
    def _as_openai_file_id(handle: Any) -> str:
        """Normalize a Files handle (string ID / dict with `id` or `file_id` / SDK object) to a string ID."""
        if isinstance(handle, str):
            return handle
        if isinstance(handle, dict):
            fid = handle.get("file_id") or handle.get("id")
            if isinstance(fid, str) and fid:
                return fid
        if hasattr(handle, "id"):
            fid = getattr(handle, "id")
            if isinstance(fid, str) and fid:
                return fid
        raise TypeError(f"Unsupported OpenAI file handle type: {type(handle)}")

    def to_dict(self) -> Dict[str, Any]:
        """Diagnostic snapshot for OpenAIEngine: provider + model + temperature.

        Intentionally minimal to avoid leaking client objects or API keys.
        """
        return {"provider": type(self).__name__, "model": getattr(self, "model", None), "temperature": getattr(self, "temperature", None)}


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

    def upload(self, path: str) -> Any:
        """Not supported for local inference."""
        raise NotImplementedError("LlamaCppEngine has no remote file storage.")

    def delete(self, handle: Any) -> bool:
        """Not supported for local inference."""
        raise NotImplementedError("LlamaCppEngine has no remote file storage.")

    def invoke(self, messages: List[Dict[str, str]], files: Optional[List[Any]] = None) -> str:
        """Run a local chat completion; `files` are ignored."""
        if not self.llm:
            raise RuntimeError("Llama model not loaded.")
        # local models ignore 'files'; messages are standard chat format
        response = self.llm.create_chat_completion(messages=messages)
        return response["choices"][0]["message"]["content"].strip()

    def to_dict(self) -> Dict[str, Any]:
        """Diagnostic snapshot for LlamaCppEngine.

        Includes provider, n_ctx, verbose, and any of model_path/repo_id/filename that are set.
        Also reports whether a model is loaded.
        """
        llm = getattr(self, "llm", None)
        out: Dict[str, Any] = {
            "provider": type(self).__name__,
            "n_ctx": getattr(self, "n_ctx", None),
            "verbose": getattr(self, "verbose", None),
            "model_loaded": bool(llm),
        }
        # Include whichever model identifiers are present (non-None)
        if getattr(self, "model_path", None):
            out["model_path"] = self.model_path
        if getattr(self, "repo_id", None):
            out["repo_id"] = self.repo_id
        if getattr(self, "filename", None):
            out["filename"] = self.filename
        return out


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
        self.client = genai.Client(api_key=api_key or DEFAULT_GEMINI_KEY)
        self.model = model
        self.temperature = float(temperature)
        # allow the caller to expand the illegal set
        self.illegal_exts = set(self._ILLEGAL_EXTS) | (set(extra_illegal_exts or ()))

    # ── Public API ───────────────────────────────────────────────
    def invoke(self, messages: list[dict], file_paths: list[str] | None = None) -> str:
        """
        Generate content with Gemini:
        - Uploads all validated files and includes the resulting File objects in `contents`.
        - Appends user/assistant strings (non-system) after the files.
        - Supplies system text via `system_instruction`.
        - Deletes uploaded files created by this call (best-effort).
        """
        file_paths = list(dict.fromkeys(file_paths or []))  # de-dupe, preserve order

        # 1) Validate paths (raise early on illegal types)
        self._validate_paths_or_raise(file_paths)

        # 2) Prepare system + flat text turns (order-preserving)
        system_instruction = self._collect_system(messages)
        flat_texts = self._collect_non_system_texts(messages)

        # 3) Upload files and assemble contents
        uploaded_files = []      # File objects (returned by upload) to include in contents
        uploaded_names = []      # resource names (e.g., "files/abc123") for cleanup
        try:
            for path in file_paths:
                fh = self._upload_path(path)           # genai File object
                uploaded_files.append(fh)
                uploaded_names.append(getattr(fh, "name", None))

            contents = []
            contents.extend(uploaded_files)            # pass File objects directly
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

        finally:
            # 4) Best-effort cleanup: delete any uploaded files from this call
            for name in uploaded_names:
                if not name:
                    continue
                try:
                    self.client.files.delete(name=name)
                except Exception:
                    pass

    # ── Helpers: validation / preparation / uploads ─────────────
    def _validate_paths_or_raise(self, paths: list[str]) -> None:
        """Raise if any path does not exist or matches disallowed MIME/ext patterns."""
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"GeminiEngine: file not found: {p}")
            mime, _ = mimetypes.guess_type(p)
            mime = mime or ""
            ext = os.path.splitext(p)[1].lower()

            if ext in self.illegal_exts or any(mime.startswith(pref) for pref in self._ILLEGAL_MIME_PREFIXES):
                raise TypeError(f"GeminiEngine: illegal/unsupported file type: {p}")

            # Everything else is acceptable for Gemini upload (PDF, image, text/code, Office, csv/xlsx, etc.)

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

    def to_dict(self) -> Dict[str, Any]:
        """Diagnostic snapshot for GeminiEngine: provider + model + temperature.

        Keeps output minimal to avoid leaking client or API keys.
        """
        return {"provider": type(self).__name__, "model": getattr(self, "model", None), "temperature": getattr(self, "temperature", None)}


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
        self.client = Mistral(api_key=api_key or DEFAULT_MISTRAL_KEY)
        self.model = model
        self.temperature = float(temperature)
        self.inline_cutoff_chars = int(inline_cutoff_chars)
        self.retry_sign_attempts = int(retry_sign_attempts)
        self.retry_base_delay = float(retry_base_delay)
        # allow caller to expand the illegal set
        self.illegal_exts = set(self._ILLEGAL_EXTS) | (set(extra_illegal_exts or ()))

    # ── Public API ───────────────────────────────────────────────
    def invoke(self, messages: list[dict], file_paths: list[str] | None = None) -> str:
        """
        Complete a Mistral chat turn, attaching inlined text and signed file URLs as parts.
        Any temporary uploads created in this call are deleted best-effort.
        """
        file_paths = list(dict.fromkeys(file_paths or []))  # de-dupe, preserve order

        # 1) Validate & classify
        classifications = [self._classify_or_raise(p) for p in file_paths]  # (path, kind) where kind ∈ {'pdf','image','text'}

        # 2) Start from the incoming messages
        native: list[dict] = [
            {"role": (m.get("role") or "user"), "content": (m.get("content") or "")}
            for m in messages
        ]

        # 3) Ensure the last user message is a parts array we can extend
        user_idx = self._ensure_user_parts(native)  # converts content → [{"type":"text","text":...}, ...]
        parts = native[user_idx]["content"]

        # 4) Inline text/code with cutoff
        total_inlined = 0
        for path, kind in classifications:
            if kind != "text":
                continue
            text = self._read_text_file(path)
            if not text:
                continue
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

        # 5) Upload/sign/attach PDFs & images; clean up after call
        uploaded_ids: list[str] = []
        try:
            for path, kind in classifications:
                if kind not in ("pdf", "image"):
                    continue
                file_id = self._upload_file(path)
                uploaded_ids.append(file_id)
                url = self._sign_with_retry(file_id, self.retry_sign_attempts, self.retry_base_delay)
                if kind == "pdf":
                    parts.append({"type": "document_url", "document_url": url})
                else:
                    parts.append({"type": "image_url", "image_url": url})

            # 6) Make the chat call
            res = self.client.chat.complete(
                model=self.model,
                messages=native,
                temperature=self.temperature,
            )
            msg = res.choices[0].message.content
            if isinstance(msg, list):
                # Some SDKs return a list of parts for the assistant
                msg = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in msg])
            return (msg or "").strip()

        finally:
            # 7) Best-effort delete uploads we created this call
            for fid in uploaded_ids:
                try:
                    self.client.files.delete(file_id=fid)
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

    def to_dict(self) -> Dict[str, Any]:
        """Diagnostic snapshot for MistralEngine: provider + requested knobs.

        Minimal and safe: excludes client and keys.
        """
        return {
            "provider": type(self).__name__,
            "model": getattr(self, "model", None),
            "temperature": getattr(self, "temperature", None),
            "inline_cutoff_chars": getattr(self, "inline_cutoff_chars", None),
            "retry_sign_attempts": getattr(self, "retry_sign_attempts", None),
            "retry_base_delay": getattr(self, "retry_base_delay", None),
        }


# ── PLACEHOLDERS (keep the same abstract contract) ─────────────────────────────
class AzureOpenAIEngine(LLMEngine):
    """
    Placeholder for an Azure OpenAI adapter.

    This class documents the intended constructor/contract but provides **no**
    implementation: `upload`, `delete`, and `invoke` are intentionally unimplemented.
    """
    def __init__(self, api_key: str, endpoint: str, api_version: str, model: str):
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
