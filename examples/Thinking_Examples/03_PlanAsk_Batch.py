import os
from dotenv import load_dotenv
from atomic_agentic.agents import PlanAskAgent
from atomic_agentic.llm import OpenAIEngine, GeminiEngine, MistralEngine, LlamaCppEngine

load_dotenv()

# --- define our agent's llm (openai, bedrock, azure, etc.) ---
llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model = "gpt-4o-mini")
# llm = GeminiEngine(api_key = os.getenv("GOOGLE_API_KEY"), model = "gemini-2.5-flash")
# llm = MistralEngine(api_key= os.getenv("MISTRAL_API_KEY"), model = "mistral-small-latest")
# llm = LlamaCppEngine(repo_id = "unsloth/phi-4-GGUF", filename= "phi-4-Q4_K_M.gguf", n_ctx = 512, verbose = False, n_threads=16)

# --- define our PlanAskAgent: scopes ALL of its background questions in one
#     upfront batch call (already answered, no back-and-forth), then writes
#     the brief. Contrast with SelfAskAgent (01_SelfAsk_Easy.py,
#     02_SelfAsk_Hard.py): that strategy fits multi-hop questions where each
#     follow-up genuinely depends on the answer to the previous one. This
#     task is the opposite shape -- three independent background facts, none
#     depending on the others -- so batching them into one pass is strictly
#     better than paying for adaptive, one-at-a-time rounds ---
briefing_agent = PlanAskAgent(
    name="Company_Briefing_Writer",
    namespace="examples",
    llm_engine=llm,
    role_prompt="You are a concise research analyst. Write a short, factual briefing.",
    role_description="Writing a brief on the founding of several independent companies.",
    description="A research-brief writer that scopes independent background facts in one batch before writing.",
    max_thinking_rounds=None,
    generation_retries=2,
)

# --- the three sub-facts here are independent of each other -- knowing when
#     Apple was founded doesn't change what question you'd ask about Amazon.
#     PlanAskAgent can lay out and answer the full set in a single call ---
task = (
    "Write a short briefing comparing the founding of Apple, Google, and "
    "Amazon: for each, note the founding year and founder(s)."
)
result = briefing_agent.invoke({"prompt": task})

print(f"TASK: {task}\n")
print("PLAN-ASK THOUGHTS (scoped in one batch, before writing):")
for i, thought in enumerate(briefing_agent.get_thoughts(result.run_id)):
    print(f"  [{i}] Q: {thought.question}")
    print(f"      A: {thought.answer}")

print(f"\nFINAL BRIEFING:\n{result.result}")
