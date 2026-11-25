"""Constants for the LLM engines: attachment policy and ``to_llm`` client routing."""

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

# ── Anthropic ─────────────────────────────────────────────────────────────────
# Inline base64 only (Files API deferred — still in beta).
# Images: PNG, JPG/JPEG, GIF, WEBP.
# Documents: PDF (base64), plain-text files (document/text source type).

ANTHROPIC_IMAGE_EXTS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg",
    ".gif", ".webp",
)

ANTHROPIC_DOCUMENT_EXTS: tuple[str, ...] = (".pdf",)

ANTHROPIC_TEXT_EXTS: tuple[str, ...] = (
    ".txt", ".md", ".rst", ".log",
    ".csv", ".tsv",
    ".json", ".yaml", ".yml",
    ".py", ".js", ".ts",
    ".html", ".xml",
)

ANTHROPIC_ALLOWED_EXTS: frozenset[str] = (
    frozenset(ANTHROPIC_IMAGE_EXTS)
    | frozenset(ANTHROPIC_DOCUMENT_EXTS)
    | frozenset(ANTHROPIC_TEXT_EXTS)
)

# ── Client routing (`to_llm`) ────────────────────────────────────────────────
# (SDK module to import, client class names there, engine class name in
# ``atomic_agentic.llm``, PyPI distribution, atomic-agentic extra). Names are
# strings so this module imports nothing. Order matters only for cost: the
# lightest SDKs come first, because the resolver imports them in order and
# stops at the first match.

LLM_CLIENT_ROUTES: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    ("openai", ("OpenAI", "AsyncOpenAI"), "OpenAIEngine", "openai", "openai"),
    ("anthropic", ("Anthropic", "AsyncAnthropic"), "AnthropicEngine", "anthropic", "anthropic"),
    ("mistralai.client", ("Mistral",), "MistralEngine", "mistralai", "mistral"),
    ("google.genai", ("Client",), "GeminiEngine", "google-genai", "gemini"),
    ("llama_cpp", ("Llama",), "LlamaCppEngine", "llama-cpp-python", "llama"),
    ("litellm", ("Router",), "LiteLLMEngine", "litellm", "litellm"),
)
