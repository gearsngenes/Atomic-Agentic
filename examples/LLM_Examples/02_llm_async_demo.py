import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from atomic_agentic.engines.LLMEngines import OpenAIEngine
import logging
from pprint import pprint

load_dotenv()
logging.basicConfig(level=logging.INFO)

# --- Initialize engine with an async OpenAI client ---
# Passing AsyncOpenAI routes all calls through the async path natively.
llm = OpenAIEngine(
    model="gpt-4.1",
    client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
)

# --- Example list of messages (OpenAI-style chat format) ---
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

async def main() -> None:
    # --- Invoke the engine asynchronously ---
    result = await llm.async_invoke({"messages": messages})
    print("LLM RESPONSE STRING:")
    print(result.result)
    print("\nLLM RESULT OBJECT:")
    pprint(result)

asyncio.run(main())
