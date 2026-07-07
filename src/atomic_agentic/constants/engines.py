"""Attachment policy constants for LLM engine adapters."""

# Base engine attachment blacklist — applied to all LLMEngine subclasses.
ILLEGAL_ATTACHMENT_EXTS: set[str] = {
    ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".bin", ".o",
    ".db", ".sqlite",
    ".h5", ".pt", ".pth", ".onnx",
}

# Per-provider illegal extension sets.
# Currently identical to the base set; separated for future per-provider divergence.
OPENAI_ILLEGAL_EXTS: set[str] = {
    ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".bin", ".o",
    ".db", ".sqlite",
    ".h5", ".pt", ".pth", ".onnx",
}
GEMINI_ILLEGAL_EXTS: set[str] = {
    ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".bin", ".o",
    ".db", ".sqlite",
    ".h5", ".pt", ".pth", ".onnx",
}
MISTRAL_ILLEGAL_EXTS: set[str] = {
    ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z",
    ".exe", ".dll", ".so", ".bin", ".o",
    ".db", ".sqlite",
    ".h5", ".pt", ".pth", ".onnx",
}

# Per-provider MIME prefix tuples.
# Currently all identical; separated for future per-provider divergence.
OPENAI_ILLEGAL_MIME_PREFIXES: tuple[str, ...] = ("audio/", "video/")
GEMINI_ILLEGAL_MIME_PREFIXES: tuple[str, ...] = ("audio/", "video/")
MISTRAL_ILLEGAL_MIME_PREFIXES: tuple[str, ...] = ("audio/", "video/")
