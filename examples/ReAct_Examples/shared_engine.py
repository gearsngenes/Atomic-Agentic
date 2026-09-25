"""shared_engine.py

One shared ``llm_engine`` for every example in this folder, so switching
providers doesn't mean editing each example file by hand. Mirrors this
repo's established pattern (``examples/PlanAct_Examples/shared_engine.py``,
``examples/ScriptAgent_Examples/shared_engine.py``) -- one cheap/small
default model per provider.

Provider selection: set the ``REACT_AGENT_PROVIDER`` env var to one of
``o``/``g``/``m``/``a``/``p``/``3``/``4``/``t`` (openai/gemini/mistral/
anthropic/llamacpp-phi4/llamacpp-gemma3/llamacpp-granite4.1/litellm). If
unset (or not a recognized value), falls back to an interactive prompt --
so this also works untouched when just running an example by hand.

Only the example's own core ReActAgent under test should import this --
any BasicAgent sub-agents an example builds as delegation targets (e.g.
02_orchestrating_agents.py's builder/reviewer) get their own separate,
fixed engine instead, so switching the provider under test never also
silently switches what the sub-agents run on.

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
    "p": (
        LlamaCppEngine,
        {
            "model_path": os.getenv("PHI_4_PATH"),
            "repo_id": "unsloth/phi-4-GGUF",
            "filename": "phi-4-Q4_K_M.gguf",
            "n_ctx": 16384,
            "verbose": False,
            "n_threads": 4,
        },
    ),
    "3": (
        LlamaCppEngine,
        {
            "model_path": os.getenv("GEMMA_3_PATH"),
            "repo_id": "ggml-org/gemma-3-4b-it-GGUF",
            "filename": "gemma-3-4b-it-Q4_K_M.gguf",
            "n_ctx": 32768,
            "verbose": False,
            "n_threads": 4,
        },
    ),
    "4": (
        LlamaCppEngine,
        {
            "model_path": os.getenv("GRANITE_4_1_PATH"),
            "repo_id": "ibm-granite/granite-4.1-8b-GGUF",
            "filename": "granite-4.1-8b-Q4_K_M.gguf",
            "n_ctx": 32768,
            "verbose": False,
            "n_threads": 4,
        },
    ),
    "t": (LiteLLMEngine, {"model": "gpt-4o-mini"}),
}


def _pick_provider() -> str:
    env_choice = (os.getenv("REACT_AGENT_PROVIDER") or "").strip().lower()
    if env_choice in PROVIDER_MODELS:
        return env_choice
    return input(
        "Pick provider: (o)penai, (g)emini, (m)istral, (a)nthropic, "
        "gemma(3), (p)hi4, granite(4).1, (t)lite: "
    ).strip().lower()


_engine_cls, _engine_kwargs = PROVIDER_MODELS[_pick_provider()]
llm_engine = _engine_cls(**_engine_kwargs)
