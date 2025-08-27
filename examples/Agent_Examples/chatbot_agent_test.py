import sys, os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Set root to repo root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# --- local imports ---
from modules.Agents import Agent
from modules.LLMEngines import GeminiEngine, OpenAIEngine, MistralEngine, LlamaCppEngine

# --- define our agent's llm (openai, bedrock, azure, etc.) ---
llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model = "gpt-4o-mini")#GeminiEngine(api_key = os.getenv("GOOGLE_API_KEY"), model = "gemini-2.5-flash")#MistralEngine(api_key= os.getenv("MISTRAL_API_KEY"), model = "mistral-small-latest")#LlamaCppEngine(repo_id = "unsloth/phi-4-GGUF", filename= "phi-4-Q4_K_M.gguf", n_ctx   = 512, verbose = False)#

# --- define our agent ---
Agent_Atom = Agent(
    name = "Agent Atom",
    llm_engine = llm,
    role_prompt = "You are a helpful and enthusiastic assistant named Agent Atom.",
    description= "A generically helpful conversational AI assistant for basic Q&A",
    context_enabled=True)

# --- begin a conversation with the agent ---
print(f"Chat with {Agent_Atom.name}! To exit the conversation, type 'q' or 'exit'!")
query = input("YOU: ")
while query.strip().lower() not in ['q', 'exit']:
    print(f"{Agent_Atom.name.upper()}: {Agent_Atom.invoke(query)}")
    query = input("YOU: ")