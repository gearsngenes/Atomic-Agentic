import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from google import genai
from atomic_agentic.engines.LLMEngines import GeminiEngine, OpenAIEngine
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

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]


async def main() -> None:
    # Run both engines concurrently — same async_invoke interface for both.
    openai_result, gemini_result = await asyncio.gather(
        openai_engine.async_invoke({"messages": messages}),
        gemini_engine.async_invoke({"messages": messages}),
    )

    print("=== OpenAI ===")
    print(openai_result.result)
    print("\nOpenAI RESULT OBJECT:")
    pprint(openai_result)

    print("\n=== Gemini ===")
    print(gemini_result.result)
    print("\nGemini RESULT OBJECT:")
    pprint(gemini_result)


asyncio.run(main())
