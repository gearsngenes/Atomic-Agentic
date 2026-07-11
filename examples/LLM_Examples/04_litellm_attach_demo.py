"""
LiteLLMEngine demo — attachment support across providers via proxy routing.

Prerequisite: start the litellm proxy in a separate terminal first (same
config as 03_litellm_engine_demo.py), from the repo root:

    litellm --config examples/LLM_Examples/litellm_proxy_config.yaml --port 4000

Attaches examples/Agent_Examples/Dromaeosaurs.pdf and asks each provider
alias to summarize it. attach()+invoke() are wrapped in one try/except per
alias so a provider that can't handle the attachment doesn't stop the rest
of the run — this is how we observe which providers actually support inline
PDF attachments through litellm's proxy path.

Note: LiteLLMEngine's documented Mistral-PDF gap only raises cleanly when
provider="mistral" is passed directly. Under proxy routing (provider=
"litellm_proxy"), that guard never triggers — a Mistral failure here would
come from litellm's own Mistral transformation rejecting the file
server-side, not from our own attach-time check.
"""

import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from atomic_agentic.llm import LiteLLMEngine, LLMEngineError

load_dotenv()
logging.basicConfig(level=logging.INFO)

PROXY_API_BASE = "http://localhost:4000"
CONFIG_PATH = Path(__file__).parent / "litellm_proxy_config.yaml"
FILE_PATH = str(Path(__file__).parent.parent / "Agent_Examples" / "Dromaeosaurs.pdf")

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File not found: {FILE_PATH}")

# --- Load aliases from the proxy config, same as 03. ---
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
    {"role": "system", "content": "You are a concise technical summarizer."},
    {
        "role": "user",
        "content": (
            "Analyze and summarize the content of the attached file thoroughly. "
            "Call out any key features, images, description, information, etc."
        ),
    },
]

for alias, engine in engines.items():
    print(f"=== {alias} ===")
    try:
        engine.attach(FILE_PATH)
        result = engine.invoke({"messages": messages})
        print("SUCCESS:")
        print(result.result)
    except LLMEngineError as exc:
        print(f"FAILED: {exc}")
    finally:
        if FILE_PATH in engine.attachments:
            engine.detach(FILE_PATH)
    print()
