import os
from dotenv import load_dotenv
from atomic_agentic.engines.LLMEngines import OpenAIEngine, GeminiEngine, MistralEngine, LlamaCppEngine
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
# --- Initialize one LLM engine (uncomment the one you want to use) ---
llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1")
# llm = GeminiEngine(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.5-flash")
# llm = MistralEngine(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")
# llm = LlamaCppEngine(repo_id="unsloth/phi-4-GGUF", filename="phi-4-Q4_K_M.gguf", n_ctx=512, verbose=False, n_threads=4)

# --- Example list of messages (OpenAI-style chat format) ---
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

# --- Invoke the engine as an AtomicInvokable (dict-first contract) ---
result = llm.invoke({"messages": messages})
print("LLM Response:", result)
