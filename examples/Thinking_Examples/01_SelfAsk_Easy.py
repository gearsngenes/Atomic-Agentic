import os
from dotenv import load_dotenv
from atomic_agentic.agents import SelfAskAgent
from atomic_agentic.llm import OpenAIEngine, GeminiEngine, MistralEngine, LlamaCppEngine

load_dotenv()

# --- define our agent's llm (openai, bedrock, azure, etc.) ---
llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model = "gpt-4o-mini")
# llm = GeminiEngine(api_key = os.getenv("GOOGLE_API_KEY"), model = "gemini-2.5-flash")
# llm = MistralEngine(api_key= os.getenv("MISTRAL_API_KEY"), model = "mistral-small-latest")
# llm = LlamaCppEngine(repo_id = "unsloth/phi-4-GGUF", filename= "phi-4-Q4_K_M.gguf", n_ctx = 512, verbose = False, n_threads=16)

# --- define our SelfAskAgent: thinks through multi-hop questions one
#     self-asked follow-up at a time before answering ---
trivia_agent = SelfAskAgent(
    name="Trivia_Thinker",
    namespace="examples",
    llm_engine=llm,
    role_prompt="You are a careful trivia assistant. Answer concisely, in one or two sentences.",
    role_description="Answering multi-hop trivia questions that require intermediate facts.",
    description="A trivia assistant that reasons through multi-hop questions via self-questioning.",
    max_thinking_rounds=5,
    generation_retries=2,
)

# --- ask a multi-hop question: answering it well requires an intermediate
#     fact (who directed the movie) before the final fact (their birth year) ---
question = "What year was the director of the movie 'Inception' born?"
result = trivia_agent.invoke({"prompt": question})

print(f"QUESTION: {question}\n")
print("SELF-ASK THOUGHTS:")
for i, thought in enumerate(trivia_agent.get_thoughts(result.run_id)):
    print(f"  [{i}] Q: {thought.question}")
    print(f"      A: {thought.answer}")

print(f"\nFINAL ANSWER: {result.result}")
