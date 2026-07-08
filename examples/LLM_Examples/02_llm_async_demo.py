import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from google import genai
from mistralai.client import Mistral
from atomic_agentic.llm import GeminiEngine, LlamaCppEngine, MistralEngine, OpenAIEngine
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

# --- LlamaCpp: no native async — offloads to asyncio.to_thread ---
# llama-cpp-python is a synchronous C++ binding with no async surface.
# async_invoke dispatches to a thread pool so it participates in gather
# without blocking the event loop, but gains no I/O concurrency benefit.
llama_engine = LlamaCppEngine(
    repo_id="unsloth/phi-4-GGUF",
    filename="phi-4-Q4_K_M.gguf",
    n_ctx=512,
    verbose=False,
    n_threads=4,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]


async def main() -> None:
    # Run all four engines concurrently — same async_invoke interface for all.
    openai_result, gemini_result, mistral_result, llama_result = await asyncio.gather(
        openai_engine.async_invoke({"messages": messages}),
        gemini_engine.async_invoke({"messages": messages}),
        mistral_engine.async_invoke({"messages": messages}),
        llama_engine.async_invoke({"messages": messages}),
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

    print("\n=== LlamaCpp ===")
    print(llama_result.result)
    print("\nLlamaCpp RESULT OBJECT:")
    pprint(llama_result)


asyncio.run(main())
