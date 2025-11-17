# example_doc_summarizer.py
# Requires:
#   - OPENAI_API_KEY in your environment (or .env; your modules already call load_dotenv)
#   - Your updated Agents.py and LLMEngines.py on PYTHONPATH (e.g., in a "modules" package)
import os, time
from dotenv import load_dotenv
from atomic_agentic.LLMEngines import OpenAIEngine,MistralEngine,GeminiEngine
from atomic_agentic.Agents import Agent
from atomic_agentic.LLMEngines import GeminiEngine, OpenAIEngine, MistralEngine, LlamaCppEngine
load_dotenv()
# --- define our agent's llm (openai, bedrock, azure, etc.) ---
llm = OpenAIEngine(model = "gpt-4o-mini")#MistralEngine(model = "mistral-small-latest")#GeminiEngine(model = "gemini-2.5-flash")#LlamaCppEngine(repo_id = "unsloth/phi-4-GGUF", filename= "phi-4-Q4_K_M.gguf", n_ctx   = 512, verbose = False)#

# --- define our agent ---
agent = Agent(
    name="DocSummarizer",
    description="Summarizes uploaded documents.",
    llm_engine=llm,
    role_prompt="You are a concise technical summarizer.",
    context_enabled=True,   # enables files/history tracking
)

def main():
    # 1) Upload a local document once; agent stores provider handle in agent.files
    file_path = "./examples/Agent_Examples/Dromaeosaurs.pdf"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    agent.attach(file_path)
    print(f"Uploaded and tracked file: {file_path}")
    time.sleep(2)  # wait a moment for upload to complete if needed
    
    # 2) Ask for a summary (the Agent will pass files -> engine.invoke(...))
    prompt = (
        "Analyze and summarize the content of the file thoroughly. "
        "Call out any key features, images, description, information, etc."
    )
    summary = agent.invoke({"prompt": prompt})  # <-- UPDATED: dict-first invoke

    # 3) Print the summary
    print("\n=== Summary ===\n")
    print(summary)

    # 4) detach the file if no longer needed
    agent.detach(file_path)

if __name__ == "__main__":
    main()
