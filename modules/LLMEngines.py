from __future__ import annotations
# LLMEngines.py
# Engines are stateless adapters around provider SDKs.
# The Agent owns conversation history and a map of local path -> provider handle.


import os, time, random
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional

from dotenv import load_dotenv
from llama_cpp import Llama

# Provider SDKs
from openai import OpenAI
from google import genai
from mistralai import Mistral

# ── ENV / CONSTANTS ───────────────────────────────────────────────────────────
load_dotenv()
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_GEMINI_KEY     = os.getenv("GOOGLE_API_KEY")
DEFAULT_MISTRAL_KEY    = os.getenv("MISTRAL_API_KEY")


# ── ABSTRACT ENGINE ───────────────────────────────────────────────────────────
class LLMEngine:
    """
    Stateless provider interface.

    Contract
    --------
    invoke(messages, file_paths=None) -> str
      • `messages`: list of {"role": str, "content": str} (system/user/assistant).
      • `file_paths`: list[str] of local file paths (the Agent is the only stateful owner).
      • Engines MUST:
          1) Validate each path and raise TypeError on illegal/unsupported types.
          2) Decide, per provider, how to handle supported files:
             - PDFs/images: upload/attach as the provider expects; delete after the call.
             - Text/code: either upload if natively supported or inline text (subject to cutoff).
          3) Stay stateless: any uploads created inside `invoke` must be cleaned up (best-effort).
          4) Keep the model call surface consistent for the provider (e.g., always Responses for OpenAI).
      • Return the final assistant text.

    Design Notes
    ------------
    - Engines may expose instance knobs (set in their own __init__), e.g.:
        inline_cutoff_chars: int  # max total chars to inline from text/code files
        reject_exts: set[str]     # extra extensions to reject up-front
        vision_required: bool     # enforce a vision-capable model when images/PDFs present
    - Engines should implement small internal helpers for:
        _classify_path(path)  -> {"kind": "pdf"|"image"|"text"|"illegal", "ext": ".pdf", ...}
        _upload(path)         -> provider handle (if needed)
        _attach(...), _inline(...), _cleanup(...)
      but these helpers are implementation details and NOT part of this abstract interface.

    Illegal Types (guideline)
    -------------------------
    Treat these as illegal by default; raise TypeError on sight:
      • MIME prefixes: "audio/", "video/"
      • Common opaque/binary/exec/archive/db/model extensions:
        .zip, .tar, .gz, .tgz, .rar, .7z,
        .exe, .dll, .so, .bin, .o,
        .db, .sqlite,
        .h5, .pt, .pth, .onnx
    Concrete engines may widen/narrow this set.
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

    def invoke(self, messages: list[dict], file_paths: list[str] | None = None) -> str:
        """
        Execute a single model call with the given messages and optional local file paths.

        Responsibilities of concrete implementations:
          - Validate `file_paths` against provider support; raise TypeError for illegal types.
          - Upload + attach PDFs/images as required by the provider; inline text/code if applicable.
          - Ensure any temporary uploads are best-effort deleted after the call (statelessness).
          - Return the assistant's text response (stripped).

        MUST be overridden by concrete engines.
        """
        raise NotImplementedError("LLMEngine.invoke must be implemented by subclasses")


# ── OPENAI (Responses API + Chat fallback) ─────────────────────────────────────
import os, mimetypes
from typing import Any, List, Dict, Optional
from openai import OpenAI

DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class OpenAIEngine(LLMEngine):
    """
    OpenAI engine (stateless) using the Responses API for ALL calls.

    Supported file handling (per-call):
      • PDFs   → upload to Files API; attach as {"type":"input_file","file_id":...}
      • Images → upload to Files API; attach as {"type":"input_image","file_id":...}
      • Text/code (e.g., .txt/.md/.py/.json/.html/.js/.java, etc.)
               → read() and inline as {"type":"input_text","text":...} with a global cutoff.
      • Illegal/unsupported types (audio/video/archives/executables/db/models) → TypeError

    Statelessness:
      • Any uploads created inside `invoke` are best-effort deleted in a finally block.

    Notes:
      • Use a vision-capable model (e.g., gpt-4o / gpt-4o-mini) when providing PDFs or images.
      • System prompts are set via top-level `instructions`.
      • Assistant history is encoded as 'output_text'; other turns as 'input_text'.
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
            resp = self.llm.responses.create(
                model=self.model,
                instructions=instructions,
                input=blocks,
                temperature=self.temperature,
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
        """Return (path, kind) where kind ∈ {'pdf','image','text'}; raise on illegal/unknown."""
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
        parts = [m["content"] for m in messages
                 if (m.get("role") or "").lower() == "system" and m.get("content")]
        return "\n\n".join(parts) or None

    def _build_role_blocks(self, messages: list[dict]) -> list[dict]:
        """assistant→output_text; others (non-system)→input_text."""
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
        idx = next((i for i in range(len(blocks) - 1, -1, -1) if blocks[i]["role"] == "user"), None)
        if idx is None:
            blocks.append({"role": "user", "content": []})
            idx = len(blocks) - 1
        return idx

    def _read_text_file(self, path: str) -> str:
        try:
            with open(path, "rb") as f:
                raw = f.read()
            return raw.decode("utf-8", errors="replace")
        except Exception as e:
            return f"[Error reading file '{os.path.basename(path)}': {e}]"

    def _upload_file(self, path: str) -> str:
        with open(path, "rb") as fp:
            f = self.llm.files.create(file=fp, purpose="assistants")
        return f.id

    # ── Helpers ─────────────────────────────────────────────────
    def _attach_local_path(self, user_parts: List[Dict[str, Any]], path: str) -> None:
        """Decide PDF/image/other from local path; upload as needed or inline as text."""
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
        Decide PDF/image/other from uploaded file metadata.
        If not PDF/image, try to download content and inline; if the SDK
        doesn't support .files.content, raise a clear error.
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
        """Accept string id, dict with 'id'/'file_id', or SDK File object; return id string."""
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

# ── LLAMA.CPP (local; no remote file store) ────────────────────────────────────
class LlamaCppEngine(LLMEngine):
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

    def upload(self, path: str) -> Any:
        raise NotImplementedError("LlamaCppEngine has no remote file storage.")

    def delete(self, handle: Any) -> bool:
        raise NotImplementedError("LlamaCppEngine has no remote file storage.")

    def invoke(self, messages: List[Dict[str, str]], files: Optional[List[Any]] = None) -> str:
        if not self.llm:
            raise RuntimeError("Llama model not loaded.")
        # local models ignore 'files'; messages are standard chat format
        response = self.llm.create_chat_completion(messages=messages)
        return response["choices"][0]["message"]["content"].strip()


# ── GEMINI (flat contents: file handle objects + strings) ──────────────────────
class GeminiEngine(LLMEngine):
    """
    Google Gemini engine (stateless) using the Files API + flat `contents`.

    Strategy per call:
      • Validate each provided path; reject audio/video/archives/executables/etc.
      • Upload ALL supported files (PDFs, images, text/code, Office docs, CSV/XLSX, etc.)
        via `client.files.upload(...)` and pass the returned File objects directly
        in `contents` (no manual inlining needed).
      • Flatten non-system chat turns into plain strings (order preserved).
      • Put system prompts into `system_instruction`.
      • Best-effort delete every uploaded file in a `finally:` block to stay stateless.

    Notes:
      • This mirrors Google's recommended “flat contents” pattern:
          contents = [ <File>, <File>, ..., "user text", "assistant text", ... ]
      • If you later want to cache uploads across turns, do that in the Agent (stateful)
        and skip deletion; this engine intentionally cleans up to remain stateless.
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
        parts = [m["content"] for m in messages
                 if (m.get("role") or "").lower() == "system" and m.get("content")]
        return "\n\n".join(parts) or None

    def _collect_non_system_texts(self, messages: list[dict]) -> list[str]:
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
        # Upload local path and return the File object.
        # genai SDK will infer MIME and handle supported types.
        abs_path = os.path.abspath(path)
        return self.client.files.upload(file=abs_path)

# ── MISTRAL (Document QnA: upload -> sign -> document_url) ─────────────────────
class MistralEngine(LLMEngine):
    """
    Mistral engine (stateless) using file upload → signed URL → chat parts.

    Per-call strategy:
      • Validate each provided path; raise TypeError on illegal/unsupported types.
      • PDFs  → upload, sign, attach as {"type":"document_url","document_url": ...}
      • Images→ upload, sign, attach as {"type":"image_url","image_url": ...}
      • Text/code → read() and inline as {"type":"text","text": ...} (with cutoff)
      • Best-effort delete every uploaded file in a finally block (statelessness)

    Notes:
      • Signing can be briefly eventual-consistent after upload; we retry on 404.
      • We preserve incoming messages; only convert the last user turn to a parts array
        so we can append text/doc/image parts cleanly.
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
        """Return (path, kind) where kind ∈ {'pdf','image','text'}; raise on illegal."""
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
        Ensure there is a user message with content as a parts list.
        - If none exists, append an empty user turn.
        - If content is a string, convert to [{"type":"text","text":...}] (preserving original).
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
        try:
            with open(path, "rb") as f:
                raw = f.read()
            return raw.decode("utf-8", errors="replace")
        except Exception as e:
            return f"[Error reading file '{os.path.basename(path)}': {e}]"

    def _upload_file(self, path: str) -> str:
        with open(path, "rb") as f:
            up = self.client.files.upload(
                file={"file_name": os.path.basename(path), "content": f},
                purpose="ocr",  # suitable for PDFs/images; used for doc/image understanding
            )
        return up.id  # string handle

    def _sign_with_retry(self, file_id: str, max_retries: int, base_delay: float) -> str:
        """
        Mint a signed URL; retry on brief 404 propagation windows after upload.
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


# ── PLACEHOLDERS (keep the same abstract contract) ─────────────────────────────
class AzureOpenAIEngine(LLMEngine):
    def __init__(self, api_key: str, endpoint: str, api_version: str, model: str):
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.model = model

    def upload(self, path: str) -> Any:
        raise NotImplementedError("AzureOpenAIEngine.upload not implemented")

    def delete(self, handle: Any) -> bool:
        raise NotImplementedError("AzureOpenAIEngine.delete not implemented")

    def invoke(self, messages: List[Dict[str, str]], files: Optional[List[Any]] = None) -> str:
        raise NotImplementedError("AzureOpenAIEngine.invoke not implemented")


class BedrockEngine(LLMEngine):
    def __init__(self, access_key: str, secret_key: str, region: str, model: str):
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.model = model

    def upload(self, path: str) -> Any:
        raise NotImplementedError("BedrockEngine.upload not implemented")

    def delete(self, handle: Any) -> bool:
        raise NotImplementedError("BedrockEngine.delete not implemented")

    def invoke(self, messages: List[Dict[str, str]], files: Optional[List[Any]] = None) -> str:
        raise NotImplementedError("BedrockEngine.invoke not implemented")
