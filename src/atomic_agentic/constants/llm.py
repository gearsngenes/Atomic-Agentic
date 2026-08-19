"""Attachment policy and structured-output-schema constants for LLM engine
adapters."""

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

# ── LiteLLM ───────────────────────────────────────────────────────────────────
# Inline base64 only, via litellm's generic image_url/file/text content
# blocks. Images: PNG, JPG/JPEG, GIF, WEBP. Documents: PDF (base64; not
# supported when provider == "mistral" — see LiteLLMEngine._prepare_attachment).
# Text/code: same set as ANTHROPIC_TEXT_EXTS.

LITELLM_IMAGE_EXTS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg",
    ".gif", ".webp",
)

LITELLM_DOCUMENT_EXTS: tuple[str, ...] = (".pdf",)

LITELLM_TEXT_EXTS: tuple[str, ...] = (
    ".txt", ".md", ".rst", ".log",
    ".csv", ".tsv",
    ".json", ".yaml", ".yml",
    ".py", ".js", ".ts",
    ".html", ".xml",
)

LITELLM_ALLOWED_EXTS: frozenset[str] = (
    frozenset(LITELLM_IMAGE_EXTS)
    | frozenset(LITELLM_DOCUMENT_EXTS)
    | frozenset(LITELLM_TEXT_EXTS)
)

# ── JSON-Schema recursion vocabulary (structured-output cleaning) ─────────────
# Consumed by utils/llm.py's clean_structure_template. A key belongs to at
# most one of these three sets. Any key in none of them is treated as a plain
# keyword: checked against a caller's permitted/omitted sets, its value
# deep-copied but not walked further.

# This key's own value-dict's keys are user-defined names, never keywords --
# never checked against permitted/omitted. Recurse into each entry's value
# only (itself a nested schema object).
NAME_KEYED_CONTAINER_KEYS: frozenset[str] = frozenset({
    "properties", "$defs", "definitions", "patternProperties",
})

# This key's value is a list of schema objects -- recurse into each element.
LIST_OF_SCHEMAS_KEYS: frozenset[str] = frozenset({
    "anyOf", "oneOf", "allOf", "prefixItems",
})

# This key's value is one schema object (when it's a Mapping) -- recurse
# directly into it. `items`/`additionalProperties` can also be a list/bool
# respectively (older draft / boolean form); see utils/llm.py's
# _schema_value_kind.
SINGLE_SCHEMA_KEYS: frozenset[str] = frozenset({
    "items", "additionalProperties", "not", "contains", "propertyNames",
})

# Sentinel return values for utils/llm.py's _schema_value_kind -- name the
# four possible answers to "what kind of layer does this keyword's value
# route into?" instead of leaving that decision spread across inline
# isinstance branches.
NAME_KEYED_LAYER: str = "name_keyed_layer"      # value's keys are user-defined names
SCHEMA_LIST_LAYER: str = "schema_list_layer"    # value is a list of schema objects
SINGLE_SCHEMA_LAYER: str = "single_schema_layer"  # value is one schema object
OPAQUE_VALUE: str = "opaque_value"              # not schema-recursable; deep-copy only

# ── OpenAI structured-output schema policy ─────────────────────────────────
# Consumed by OpenAIEngine.structure_omitted_keys. OpenAI's strict JSON-Schema
# mode rejects `default` outright; every other JSON-Schema keyword this
# codebase's clean_structure_template understands is accepted by the API
# (even when not actually enforced at generation time), so nothing else needs
# stripping.
OPENAI_STRUCTURE_OMITTED_KEYS: frozenset[str] = frozenset({"default"})

# ── Gemini structured-output schema policy ─────────────────────────────────
# Consumed by GeminiEngine.structure_permitted_keys. Confirmed directly
# against the installed google-genai SDK's GenerateContentConfig.response_json_schema
# field docstring (v1.75.0) -- Gemini's supported JSON-Schema keyword set is
# narrower than AA's canonical (OpenAI-shaped) dialect, so this is a positive
# allow-list rather than an omit-list.
GEMINI_STRUCTURE_PERMITTED_KEYS: frozenset[str] = frozenset({
    "$id", "$defs", "$ref", "$anchor",
    "type", "format", "title", "description", "enum",
    "items", "prefixItems", "minItems", "maxItems",
    "minimum", "maximum", "anyOf", "oneOf",
    "properties", "additionalProperties", "required",
})
