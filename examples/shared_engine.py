"""shared_engine.py

One shared ``llm_engine`` for every example under ``examples/`` (PlanAct,
ReAct, and ScriptAct alike), so switching providers doesn't mean editing
each example file by hand. It lives at the `examples/` root as one source
of truth instead of a per-folder copy.

Provider selection: set the ``AA_AGENT_PROVIDER`` env var to one of
``o``/``g``/``m``/``a``/``p``/``3``/``4``/``t`` (openai/gemini/mistral/
anthropic/llamacpp-phi4/llamacpp-gemma3/llamacpp-granite4.1/litellm). If
unset (or not a recognized value), falls back to an interactive prompt --
so this also works untouched when just running an example by hand.

Per-provider model override: ``OPENAI_MODEL``/``GEMINI_MODEL``/
``MISTRAL_MODEL``/``ANTHROPIC_MODEL`` swap that one provider's model
without touching this file. Unset falls back to the defaults below: OpenAI
gpt-4o, gemini-2.5-flash, mistral-medium-latest and claude-haiku-4-5, the
models the multi-round ReAct/PlanAct/ScriptAct examples are run against.
The Gemini entry pins the API version to ``v1beta``: google-genai 3.x defaults to ``v1``, which rejects system instructions
and JSON mode for the 2.5 family (Gemini 3.x models work on either).

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
from typing import Any, Callable

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

def _llama(path_env: str, repo_id: str, filename: str, n_ctx: int) -> Callable[[], Any]:
    """Local GGUF from ``path_env`` when set, else downloaded from the Hub."""
    def build() -> Any:
        path = os.getenv(path_env)
        if path:
            return LlamaCppEngine(model=path, n_ctx=n_ctx, verbose=False, n_threads=4)
        return LlamaCppEngine.from_pretrained(
            repo_id, filename, n_ctx=n_ctx, verbose=False, n_threads=4
        )
    return build


PROVIDER_MODELS: dict[str, Callable[[], Any]] = {
    "o": lambda: OpenAIEngine(model=os.getenv("OPENAI_MODEL", "gpt-4o"), api_key=os.getenv("OPENAI_API_KEY")),
    "g": lambda: GeminiEngine(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), api_key=os.getenv("GOOGLE_API_KEY"),
                              http_options={"api_version": "v1beta"}),
    "m": lambda: MistralEngine(model=os.getenv("MISTRAL_MODEL", "mistral-medium-latest"), api_key=os.getenv("MISTRAL_API_KEY")),
    "a": lambda: AnthropicEngine(model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"), api_key=os.getenv("ANTHROPIC_API_KEY")),
    "p": _llama("PHI_4_PATH", "unsloth/phi-4-GGUF", "phi-4-Q4_K_M.gguf", 16384),
    "3": _llama("GEMMA_3_PATH", "ggml-org/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf", 32768),
    "4": _llama("GRANITE_4_1_PATH", "ibm-granite/granite-4.1-8b-GGUF", "granite-4.1-8b-Q4_K_M.gguf", 32768),
    "t": lambda: LiteLLMEngine(model="litellm_proxy/" + os.getenv("OPENAI_MODEL", "gpt-4o")),
}


def _pick_provider() -> str:
    env_choice = (os.getenv("AA_AGENT_PROVIDER") or "").strip().lower()
    if env_choice in PROVIDER_MODELS:
        return env_choice
    while True:
        choice = input(
            "Pick provider: (o)penai, (g)emini, (m)istral, (a)nthropic, "
            "gemma(3), (p)hi4, granite(4).1, (t)lite: "
        ).strip().lower()
        if choice in PROVIDER_MODELS:
            return choice
        print(f"Unrecognized choice {choice!r}; expected one of {sorted(PROVIDER_MODELS)}.")


llm_engine = PROVIDER_MODELS[_pick_provider()]()
