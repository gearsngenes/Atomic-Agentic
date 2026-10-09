"""shared_engine.py

One shared ``llm_engine`` for every example under ``examples/`` (PlanAct,
ReAct, and ScriptAct alike), so switching providers doesn't mean editing
each example file by hand. Single shared module at `examples/` root,
mirroring `prebuilt_tools.py`'s own move here -- one source of truth
instead of a per-folder copy that drifts.

Provider selection: set the ``AA_AGENT_PROVIDER`` env var to one of
``o``/``g``/``m``/``a``/``p``/``3``/``4``/``t`` (openai/gemini/mistral/
anthropic/llamacpp-phi4/llamacpp-gemma3/llamacpp-granite4.1/litellm). If
unset (or not a recognized value), falls back to an interactive prompt --
so this also works untouched when just running an example by hand.

Per-provider model override: ``OPENAI_MODEL``/``GEMINI_MODEL``/
``MISTRAL_MODEL``/``ANTHROPIC_MODEL`` swap that one provider's model
without touching this file. Unset falls back to this repo's own confirmed
floor model for that provider -- the weakest model live-tested (via this
repo's own multi-round ReAct/PlanAct/ScriptAct examples, not a generic
benchmark) to reliably complete them, not simply "whatever is cheapest."
As of this writing that floor is OpenAI gpt-4o (gpt-4o-mini and gpt-5-nano
were both tested and rejected -- each failed the bulk of ReAct trial runs
in ways untied to raw reasoning, e.g. repeating a side effect already
performed, or folding a value meant to be printed into the return value
instead); Gemini/Mistral/Anthropic's existing small-tier defaults
(gemini-2.5-flash/mistral-medium-latest/claude-haiku-4-5) passed that same
live check cleanly and are unchanged.

Only an example's own core ToolAgent under test should import this -- any
BasicAgent/PlanActAgent/etc. sub-agents an example builds as delegation
targets (e.g. ReAct_Examples/02_orchestrating_agents.py's builder/
reviewer) get their own separate, fixed engine instead, so switching the
provider under test never also silently switches what the sub-agents run
on.

Import convention (same shim as `prebuilt_tools.py` -- this module lives
one directory higher, at `examples/` itself, not beside the importing
script)::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    "o": (OpenAIEngine, {"api_key": os.getenv("OPENAI_API_KEY"), "model": os.getenv("OPENAI_MODEL", "gpt-4o")}),
    "g": (GeminiEngine, {"api_key": os.getenv("GOOGLE_API_KEY"), "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")}),
    "m": (MistralEngine, {"api_key": os.getenv("MISTRAL_API_KEY"), "model": os.getenv("MISTRAL_MODEL", "mistral-medium-latest")}),
    "a": (AnthropicEngine, {"api_key": os.getenv("ANTHROPIC_API_KEY"), "model": os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")}),
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
    "t": (LiteLLMEngine, {"model": os.getenv("OPENAI_MODEL", "gpt-4o")}),
}


def _pick_provider() -> str:
    env_choice = (os.getenv("AA_AGENT_PROVIDER") or "").strip().lower()
    if env_choice in PROVIDER_MODELS:
        return env_choice
    return input(
        "Pick provider: (o)penai, (g)emini, (m)istral, (a)nthropic, "
        "gemma(3), (p)hi4, granite(4).1, (t)lite: "
    ).strip().lower()


_engine_cls, _engine_kwargs = PROVIDER_MODELS[_pick_provider()]
llm_engine = _engine_cls(**_engine_kwargs)
