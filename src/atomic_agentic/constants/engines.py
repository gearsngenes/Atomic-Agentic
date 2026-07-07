"""Attachment policy constants for LLM engine adapters."""

# ── Shared blacklist (all current providers use this identical set) ──────────

ILLEGAL_ATTACHMENT_EXTS: set[str] = {
    ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".bin", ".o",
    ".db", ".sqlite",
    ".h5", ".pt", ".pth", ".onnx",
}

ENGINE_ILLEGAL_MIME_PREFIXES: tuple[str, ...] = ("audio/", "video/")

# ── OpenAI ───────────────────────────────────────────────────────────────────

OPENAI_IMAGE_EXTS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg",
    ".webp", ".gif", ".bmp",
    ".tif", ".tiff", ".heic",
)

OPENAI_TEXT_EXTS: tuple[str, ...] = (
    ".txt", ".md", ".rst", ".log",
    ".json", ".jsonl", ".yaml", ".yml",
    ".csv", ".tsv", ".py", ".ipynb",
    ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".h",
    ".hpp", ".rs", ".go", ".rb",
    ".php", ".cs", ".html", ".htm",
    ".xml",
)

OPENAI_ALLOWED_EXTS: frozenset[str] = (
    frozenset(OPENAI_IMAGE_EXTS) | frozenset(OPENAI_TEXT_EXTS) | {".pdf"}
)

# ── Mistral ───────────────────────────────────────────────────────────────────
# Mistral vision supports PNG, JPG/JPEG, GIF, WEBP — not BMP, TIFF, or HEIC.

MISTRAL_IMAGE_EXTS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg",
    ".webp", ".gif",
)
