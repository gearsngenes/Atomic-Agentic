"""
LiteLLMEngine demo — provider-agnostic proxy/alias routing.

Prerequisite: start the litellm proxy in a separate terminal first, from the
repo root, using the config alongside this script:

    litellm --config examples/LLM_Examples/litellm_proxy_config.yaml --port 4000

The proxy process needs LITELLM_MASTER_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY,
GOOGLE_API_KEY (or GEMINI_API_KEY), and MISTRAL_API_KEY in its environment.
This script only needs LITELLM_MASTER_KEY — it authenticates to the proxy
directly with the master key, never sees the upstream provider keys, and asks
for each alias declared in the config rather than a raw provider/model string.
"""

import asyncio
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from atomic_agentic.llm import LiteLLMEngine
import logging
from pprint import pprint

load_dotenv()
logging.basicConfig(level=logging.INFO)

PROXY_API_BASE = "http://localhost:4000"
CONFIG_PATH = Path(__file__).parent / "litellm_proxy_config.yaml"

# --- Load aliases from the proxy config — the engine is built entirely from
# what the config declares, not from hardcoded provider/model strings. ---
with open(CONFIG_PATH, encoding="utf-8") as fh:
    proxy_config = yaml.safe_load(fh)

aliases = [entry["model_name"] for entry in proxy_config["model_list"]]

engines = {
    alias: LiteLLMEngine(
        model=alias,
        provider="litellm_proxy",
        base_url=PROXY_API_BASE,
        api_key=os.getenv("LITELLM_MASTER_KEY"),
    )
    for alias in aliases
}

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]


async def main() -> None:
    # Sync invoke, then async invoke, one after the other — per alias, not
    # concurrent gather — so each engine's two call paths are easy to compare
    # side by side.
    for alias, engine in engines.items():
        print(f"=== {alias} (sync) ===")
        sync_result = engine.invoke({"messages": messages})
        print(sync_result.result)
        print(f"\n{alias} SYNC RESULT OBJECT:")
        pprint(sync_result)

        print(f"\n=== {alias} (async) ===")
        async_result = await engine.async_invoke({"messages": messages})
        print(async_result.result)
        print(f"\n{alias} ASYNC RESULT OBJECT:")
        pprint(async_result)
        print()


asyncio.run(main())
