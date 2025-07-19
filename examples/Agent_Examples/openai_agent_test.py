import sys, os
from pathlib import Path
from dotenv import load_dotenv
# Set root to repo root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# --- local imports ---
from modules.Agents import Agent
from modules.LLMNuclei import OpenAINucleus

# --- define our agent's nucleus (openai, bedrock, azure, etc.) ---
load_dotenv()
nucleus = OpenAINucleus(
    api_key=os.getenv("OPENAI_API_KEY"), # in LLMNuclei.py this is the default api key environment variable name
    model = "gpt-4o-mini")

# --- define our agent ---
Agent_Atom = Agent(
    name = "Agent Atom",
    nucleus = nucleus,
    role_prompt = "You are a helpful and enthusiastic assistant named Agent Atom.",
    context_enabled=True)

# --- begin a conversation with the agent ---
print(f"Chat with {Agent_Atom.name}! To exit the conversation, type 'q' or 'exit'!")
query = input("YOU: ")
while query.strip().lower() not in ['q', 'exit']:
    print(f"{Agent_Atom.name.upper()}: {Agent_Atom.invoke(query)}")
    query = input("YOU: ")