import sys, os
from pathlib import Path
# Set root to repo root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMNuclei import LlamaCppNucleus

nucleus = LlamaCppNucleus(
    repo_id = "unsloth/phi-4-GGUF",
    filename= "phi-4-Q4_K_M.gguf",
    n_ctx   = 2048,
    verbose = False
)

Agent_Atom = Agent(
    name        = "SLM-Agent Atom",
    nucleus     = nucleus,
    role_prompt = "You are a helpful assistant named SLM-Agent Atom!",
    context_enabled=True
)

print(f"Chat with {Agent_Atom.name}! To exit the conversation, type 'q' or 'exit'!")
query = input("YOU: ")
while query.strip().lower() not in ['q', 'exit']:
    print(f"{Agent_Atom.name.upper()}: {Agent_Atom.invoke(query)}")
    query = input("YOU: ")