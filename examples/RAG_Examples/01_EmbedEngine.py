import sys, os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Set root to repo root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.EmbedEngines import OpenAIEmbedEngine, GeminiEmbedEngine, MistralEmbedEngine, LlamaCppEmbedEngine

# --- define our embedding engine (openai, cohere, etc.) ---
engine = LlamaCppEmbedEngine(repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF", filename="nomic-embed-text-v1.5.f16.gguf", dimension=768, normalize=True, n_threads=8)
    # MistralEmbedEngine(model="mistral-embed", dimension=1024, normalize=True)
    # GeminiEmbedEngine(model="gemini-embedding-001", dimension = 1536, normalize=True)
    # OpenAIEmbedEngine(model="text-embedding-3-small", dimension = 1536, normalize=True)

# --- embed some text ---
vec = engine.vectorize("Atomic-Agentic is an agentic AI framework.")
# --- print some of the results ---
head, tail = [str(x) for x in vec[:5]], [str(x) for x in vec[-5:]]
print("Length of vector:", len(vec), "\t\tDimension of engine:", engine.dimension)
print("[" + ",".join(head) + ",...," + ",".join(tail) + "]")
