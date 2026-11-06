import os
from dotenv import load_dotenv
from atomic_agentic.agents import ScratchpadAgent
from atomic_agentic.llm import OpenAIEngine, GeminiEngine, MistralEngine, LlamaCppEngine

load_dotenv()

# --- define our agent's llm (openai, bedrock, azure, etc.) ---
llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
# llm = GeminiEngine(api_key = os.getenv("GOOGLE_API_KEY"), model = "gemini-2.5-flash")
# llm = MistralEngine(api_key= os.getenv("MISTRAL_API_KEY"), model = "mistral-small-latest")
# llm = LlamaCppEngine(model_path=os.getenv("LLAMA_MODEL_PATH"), repo_id = "unsloth/phi-4-GGUF", filename= "phi-4-Q4_K_M.gguf", n_ctx = 512, verbose = False, n_threads=16)

# --- stop_early lets the model end thinking before the cap. Every thinking
#     round returns a thought plus a decision flag (more_thinking_required);
#     the first round always runs, and thinking ends at the first "false" or
#     at thinking_rounds, whichever comes first -- so thinking_rounds is a cap
#     here, not an exact count. The thinking instructions below only describe
#     HOW to think; the framework adds the thought/flag format itself ---
estimator_agent = ScratchpadAgent(
    name="Quick_Estimator",
    namespace="examples",
    llm_engine=llm,
    thinking_rounds=8,
    stop_early=True,
    role_prompt="You are a practical estimator. Give a single number with a one-sentence justification.",
    thinking_instructions=(
        "You are estimating a quantity. Each round, add one focused thought: "
        "break the quantity into factors you can reason about, or sanity-check "
        "the estimate so far. Stop once the estimate is well supported -- do "
        "not keep going for its own sake."
    ),
    description="An estimator that thinks only as long as the question needs.",
)

# --- an easy question needs very few rounds; a hard one uses more. The
#     result reports how many thoughts were actually produced against the cap ---
question = "Roughly how many piano tuners are there in Chicago?"
result = estimator_agent.invoke({"prompt": question})
record = estimator_agent.get_conversation().records[-1:][0]

print(f"QUESTION: {question}\n")
print("THINKING THOUGHTS:")
for round_index, thought in enumerate(record.thoughts):
    print(f"  Round {round_index}: {thought}")

print(f"\nTHOUGHTS PRODUCED: {result.thoughts_produced} (cap: {estimator_agent.thinking_rounds})")
print(f"FINAL ANSWER: {result.result}")
