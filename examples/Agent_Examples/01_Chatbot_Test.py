import os
from dotenv import load_dotenv
from atomic_agentic.agents.toolagents import Agent
from atomic_agentic.engines.LLMEngines import OpenAIEngine, GeminiEngine, MistralEngine, LlamaCppEngine

load_dotenv()

# --- define our agent's llm (openai, bedrock, azure, etc.) ---
llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model = "gpt-5-mini")
# llm = GeminiEngine(api_key = os.getenv("GOOGLE_API_KEY"), model = "gemini-2.5-flash")
# llm = MistralEngine(api_key= os.getenv("MISTRAL_API_KEY"), model = "mistral-small-latest")
# llm = LlamaCppEngine(repo_id = "unsloth/phi-4-GGUF", filename= "phi-4-Q4_K_M.gguf", n_ctx = 512, verbose = False, n_threads=16)

# --- define our agent ---
Agent_Atom = Agent(
    name = "Agent Atom",
    llm_engine = llm,
    role_prompt = """
    You are a helpful and enthusiastic assistant named Agent Atom.
    You always end your responses with excitement and emojis.""",
    description= "A generically helpful conversational AI assistant for basic Q&A",
    context_enabled=True)

# --- begin a conversation with the agent ---
print(f"Chat with {Agent_Atom.name}! To exit the conversation, type 'q' or 'exit'!")
query = input("YOU: ")
while query.strip().lower() not in ['q', 'exit']:
    response = Agent_Atom.invoke({"prompt": query})  # <-- UPDATED: dict-first invoke
    print(f"{Agent_Atom.name.upper()}: {response}")
    query = input("YOU: ")
