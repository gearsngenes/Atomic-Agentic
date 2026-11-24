# example_doc_summarizer.py
# Requires:
#   - OPENAI_API_KEY in your environment (or .env)
import os
from dotenv import load_dotenv
from atomic_agentic.agents import BasicAgent
from atomic_agentic.llm import GeminiEngine

load_dotenv()

# --- define our agent's llm (openai, bedrock, azure, etc.) ---
# (import the engine you switch to from atomic_agentic.llm)
# llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model = "gpt-5-mini")
llm = GeminiEngine(api_key = os.getenv("GOOGLE_API_KEY"), model = "gemini-2.5-flash-lite", http_options={"api_version": "v1beta"})
# llm = MistralEngine(api_key= os.getenv("MISTRAL_API_KEY"), model = "mistral-small-latest")
# llm = LlamaCppEngine(model=os.getenv("LLAMA_MODEL_PATH"), n_ctx=512, verbose=False)
# Note: attachments are supported only by the OpenAI, Gemini, Mistral and Anthropic
# engines. LlamaCppEngine and LiteLLMEngine have no attach API.

# --- define our agent ---
agent = BasicAgent(
    name="DocSummarizer",
    description="Summarizes uploaded documents.",
    llm_engine=llm,
    instructions="You are a concise technical summarizer.",
    record_enabled=True,   # enables history tracking
)

def main():
    # 1) Attach a local document once. Attachments live on the engine (the agent
    #    has no file store); `llm` is the concrete engine, so `attach` type-checks.
    file_path = "./examples/Agent_Examples/Dromaeosaurs.pdf"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    llm.attach(file_path)
    print(f"Attached file: {file_path}")

    # 2) Ask for a summary (the engine adds its attachments to every call)
    prompt = (
        "Analyze and summarize the content of the file thoroughly. "
        "Call out any key features, images, description, information, etc."
    )
    summary = agent.invoke({"prompt": prompt})  # <-- UPDATED: dict-first invoke

    # 3) Print the summary
    print("\n=== Summary ===\n")
    print(summary.result)

    # 4) detach the file if no longer needed
    llm.detach(file_path)

if __name__ == "__main__":
    main()
