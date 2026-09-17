import os
from dotenv import load_dotenv
from atomic_agentic.agents import ThinkingAgent
from atomic_agentic.llm import OpenAIEngine, GeminiEngine, MistralEngine, LlamaCppEngine

load_dotenv()

# --- define our agent's llm (openai, bedrock, azure, etc.) ---
llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
# llm = GeminiEngine(api_key = os.getenv("GOOGLE_API_KEY"), model = "gemini-2.5-flash")
# llm = MistralEngine(api_key= os.getenv("MISTRAL_API_KEY"), model = "mistral-small-latest")
# llm = LlamaCppEngine(model_path=os.getenv("LLAMA_MODEL_PATH"), repo_id = "unsloth/phi-4-GGUF", filename= "phi-4-Q4_K_M.gguf", n_ctx = 512, verbose = False, n_threads=16)

# --- define our ThinkingAgent: runs a fixed number of thinking rounds before
#     replying, no early exit -- each round produces exactly one free-form
#     thought, building on every prior round ---
trivia_agent = ThinkingAgent(
    name="Trivia_Thinker",
    namespace="examples",
    llm_engine=llm,
    role_prompt="You are a careful trivia assistant. Answer concisely, in one or two sentences.",
    thinking_instructions=(
        "You answer multi-hop questions -- they depend on a chain of "
        "intermediate facts you don't yet have. Each round, produce exactly "
        "one focused thought that builds on the rounds before it: first "
        "identify what the question is really asking, then work through the "
        "chain of intermediate facts one at a time, then combine them into "
        "the final answer, then double-check your reasoning."
    ),
    description="A trivia assistant that reasons through multi-hop questions via self-questioning.",
)

# --- ask a multi-hop question with a genuine chain of intermediate facts,
#     not just a single hop: director of Inception -> Inception's release
#     year -> that year's Best Picture winner -> that film's director ->
#     both directors' birth years -> the age difference. Each arrow is a
#     fact the agent doesn't start with, which is what actually warrants
#     several rounds of self-questioning rather than one lucky guess.
#     thinking_rounds is a per-invocation runtime parameter, not a
#     construction-time knob -- the agent always runs exactly this many
#     rounds, with no early exit ---
question = (
    "What is the age difference between the director of the movie "
    "'Inception' and the director of the movie that won the Academy Award "
    "for Best Picture in the same year 'Inception' was released?"
)
result = trivia_agent.invoke({"prompt": question, "thinking_rounds": 6})
record = trivia_agent.get_conversation(turns=1)[0]

print(f"QUESTION: {question}\n")
print("THINKING THOUGHTS:")
for round_index, thought in enumerate(record.thoughts):
    print(f"  Round {round_index}: {thought}")

print(f"\nTHINKING ROUNDS USED: {result.thinking_rounds_used}")
print(f"FINAL ANSWER: {result.result}")
