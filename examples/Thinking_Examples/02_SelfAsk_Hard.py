import os
from dotenv import load_dotenv
from atomic_agentic.agents import SelfAskAgent
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.llm import OpenAIEngine, GeminiEngine, MistralEngine, LlamaCppEngine

load_dotenv()

# --- define our agent's llm (openai, bedrock, azure, etc.) ---
llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model = "gpt-5-mini")
# llm = GeminiEngine(api_key = os.getenv("GOOGLE_API_KEY"), model = "gemini-2.5-flash")
# llm = MistralEngine(api_key= os.getenv("MISTRAL_API_KEY"), model = "mistral-small-latest")
# llm = LlamaCppEngine(repo_id = "unsloth/phi-4-GGUF", filename= "phi-4-Q4_K_M.gguf", n_ctx = 512, verbose = False, n_threads=16)

# --- the role_prompt is a PromptConfig with its OWN two caller-facing
#     placeholders, {max_word_count}/{writing_rules} -- these are real,
#     required extra_parameters on the agent's schema (role_prompt's own
#     discovered placeholders are the sole legitimate extra_parameters
#     source on ThinkingAgent), supplied at invoke() time like any other
#     input, distinct from role_description below. writing_rules is
#     deliberately genre-agnostic craft guidance, not genre-specific --
#     genre lives in the task prompt itself, alongside a starting premise ---
STORY_WRITER_ROLE_PROMPT = PromptConfig(
    template="""\
You are an accomplished short-fiction author known for tightly-plotted stories with
genuinely surprising, recontextualizing twist endings -- not jump-scares or last-line
gotchas, but endings that force the reader to reinterpret everything that came before.

Constraints you must always follow:
- Maximum length: {max_word_count} words.
- Writing rules: {writing_rules}

Write in clean, evocative prose. Every detail you include should earn its place --
nothing decorative, nothing wasted.
""",
    description="Twist-ending short-fiction writer persona, parameterized by word budget and writing rules.",
    field_specs={
        "max_word_count": {"type": "int", "description": "Hard word-count ceiling for the finished story."},
        "writing_rules": {"type": "str", "description": "Genre-agnostic craft rules the story must satisfy."},
    },
)

# --- role_description is separate from role_prompt: a narrow, non-parameter
#     pointer visible only to the self-ask thinking phase's internal render
#     context. It deliberately does NOT repeat the literal word count or
#     writing_rules text -- the self-ask rounds reason about the twist in
#     the abstract; only the reply phase (which renders the fully-resolved
#     role_prompt) ever sees the literal 1000/writing_rules values ---
story_agent = SelfAskAgent(
    name="Twist_Story_Writer",
    namespace="examples",
    llm_engine=llm,
    role_prompt=STORY_WRITER_ROLE_PROMPT,
    role_description=(
        "Planning and writing a tightly word-budgeted short story built around "
        "a twist ending that recontextualizes the whole narrative."
    ),
    description="A short-fiction writer that scopes a twist ending via self-questioning before drafting.",
    max_thinking_rounds=8,
    generation_retries=2,
)

# --- the task itself declares genre + a small seed premise -- an open-ended
#     starting point, not a full plot. Figuring out an actual satisfying
#     twist (and what it requires setting up beforehand) from just this seed
#     takes several rounds of self-asked scoping questions before writing is
#     even possible ---
task = (
    "Write a horror story about a lighthouse keeper who begins receiving radio "
    "transmissions from a ship that sank decades ago. The story's ending should "
    "recontextualize everything the reader believed up to that point -- a "
    "genuine twist, not a jump-scare."
)
result = story_agent.invoke({
    "prompt": task,
    "max_word_count": 1000,
    "writing_rules": (
        "Show, don't tell. Every sentence must either advance the plot or "
        "deepen a character. The twist must be reread-proof: on a second "
        "reading, every earlier detail should make sense in hindsight. No "
        "unearned deus ex machina resolutions."
    ),
})

print(f"TASK: {task}\n")
print("SELF-ASK THOUGHTS (scoping the twist before writing):")
for i, thought in enumerate(story_agent.get_thoughts(result.run_id)):
    print(f"  [{i}] Q: {thought.question}")
    print(f"      A: {thought.answer}")

print(f"\nFINAL STORY:\n{result.result}")
