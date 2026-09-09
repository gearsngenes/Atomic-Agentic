"""shared_engine.py

One shared ``llm_engine`` for every example in this folder, so switching
providers doesn't mean editing each example file by hand. Reuses the exact
same provider/model choices as
``test_code/script_agent_live_smoke_test.py`` -- one cheap/small default
model per provider.

Provider selection: set the ``SCRIPT_AGENT_PROVIDER`` env var to one of
``o``/``g``/``m``/``a``/``l``/``t`` (openai/gemini/mistral/anthropic/
llamacpp/litellm). If unset (or not a recognized value), falls back to an
interactive prompt -- the same one the live smoke test uses -- so this also
works untouched when just running an example by hand.

Usage in an example file (same directory, no path shim needed -- Python
puts a script's own directory on ``sys.path[0]``)::

    from shared_engine import llm_engine
"""
import os

from dotenv import load_dotenv

from atomic_agentic.llm import (
    OpenAIEngine,
    GeminiEngine,
    MistralEngine,
    AnthropicEngine,
    LlamaCppEngine,
    LiteLLMEngine,
)

load_dotenv()

PROVIDER_MODELS: dict[str, tuple[type, dict]] = {
    "o": (OpenAIEngine, {"api_key": os.getenv("OPENAI_API_KEY"), "model": "gpt-4o-mini"}),
    "g": (GeminiEngine, {"api_key": os.getenv("GOOGLE_API_KEY"), "model": "gemini-2.5-flash"}),
    "m": (MistralEngine, {"api_key": os.getenv("MISTRAL_API_KEY"), "model": "mistral-medium-latest"}),
    "a": (AnthropicEngine, {"api_key": os.getenv("ANTHROPIC_API_KEY"), "model": "claude-haiku-4-5"}),
    "l": (
        LlamaCppEngine,
        {
            "model_path": os.getenv("LLAMA_MODEL_PATH"),
            "repo_id": "unsloth/phi-4-GGUF",
            "filename": "phi-4-Q4_K_M.gguf",
            "n_ctx": 8182,
            "verbose": False,
            "n_threads": 4,
        },
    ),
    "t": (LiteLLMEngine, {"model": "gpt-4o-mini"}),
}


def _pick_provider() -> str:
    env_choice = (os.getenv("SCRIPT_AGENT_PROVIDER") or "").strip().lower()
    if env_choice in PROVIDER_MODELS:
        return env_choice
    return input(
        "Pick provider: (o)penai, (g)emini, (m)istral, (a)nthropic, (l)lamacpp, (t)lite: "
    ).strip().lower()


_engine_cls, _engine_kwargs = PROVIDER_MODELS[_pick_provider()]
llm_engine = _engine_cls(**_engine_kwargs)
