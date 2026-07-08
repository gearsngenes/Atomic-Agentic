import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from google import genai
from mistralai.client import Mistral
from atomic_agentic.engines.LLMEngines import GeminiEngine, MistralEngine, OpenAIEngine
import logging
from pprint import pprint

load_dotenv()
logging.basicConfig(level=logging.INFO)

# --- OpenAI: inject AsyncOpenAI for native async ---
# Passing AsyncOpenAI routes all calls through the async path natively.
openai_engine = OpenAIEngine(
    model="gpt-4.1",
    client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")),
)

# --- Gemini: single genai.Client handles both sync and async paths ---
# No separate async client needed — client.aio.models is the async surface.
gemini_engine = GeminiEngine(
    model="gemini-2.5-flash",
    client=genai.Client(api_key=os.getenv("GOOGLE_API_KEY")),
)

# --- Mistral: single Mistral client handles both sync and async paths ---
# chat.complete (sync) and chat.complete_async (async) live on the same object.
mistral_engine = MistralEngine(
    model="mistral-small-latest",
    client=Mistral(api_key=os.getenv("MISTRAL_API_KEY")),
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]


async def main() -> None:
    # Run all three engines concurrently — same async_invoke interface for all.
    openai_result, gemini_result, mistral_result = await asyncio.gather(
        openai_engine.async_invoke({"messages": messages}),
        gemini_engine.async_invoke({"messages": messages}),
        mistral_engine.async_invoke({"messages": messages}),
    )

    print("=== OpenAI ===")
    print(openai_result.result)
    print("\nOpenAI RESULT OBJECT:")
    pprint(openai_result)

    print("\n=== Gemini ===")
    print(gemini_result.result)
    print("\nGemini RESULT OBJECT:")
    pprint(gemini_result)

    print("\n=== Mistral ===")
    print(mistral_result.result)
    print("\nMistral RESULT OBJECT:")
    pprint(mistral_result)


asyncio.run(main())
