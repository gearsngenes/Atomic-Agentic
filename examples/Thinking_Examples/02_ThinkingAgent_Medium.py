import os
from dotenv import load_dotenv
from atomic_agentic.agents import ThinkingAgent
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.llm import OpenAIEngine, GeminiEngine, MistralEngine, LlamaCppEngine

load_dotenv()

# --- the main engine drafts the final story -- reserved for the reply phase
#     only, never used for thinking rounds once thinking_llm_engine is set ---
llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-5-mini")
# llm = GeminiEngine(api_key = os.getenv("GOOGLE_API_KEY"), model = "gemini-2.5-flash")
# llm = MistralEngine(api_key= os.getenv("MISTRAL_API_KEY"), model = "mistral-small-latest")
# llm = LlamaCppEngine(model_path=os.getenv("LLAMA_MODEL_PATH"), repo_id = "unsloth/phi-4-GGUF", filename= "phi-4-Q4_K_M.gguf", n_ctx = 512, verbose = False, n_threads=16)

# --- a second, cheaper/faster engine handles every thinking round instead --
#     brainstorming plot-twist ideas doesn't need the same model quality as
#     actually drafting the finished prose. thinking_llm_engine is optional;
#     omitting it just makes every round use the main engine above ---
thinking_llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# --- the role_prompt is a PromptConfig with its OWN two caller-facing
#     placeholders, {max_word_count}/{writing_rules} -- these are real,
#     required extra_parameters on the agent's schema (role_prompt's and
#     thinking_instructions' own discovered placeholders are ThinkingAgent's
#     two extra_parameters sources, reconciled against each other at
#     construction), supplied at invoke() time like any other input,
#     distinct from thinking_instructions below. writing_rules is
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

# --- thinking_instructions is separate from role_prompt: extra guidance
#     visible only to the thinking phase's own prompt, spliced into its
#     reserved slot. It deliberately does NOT repeat the literal word count
#     or writing_rules text -- the thinking rounds reason about the twist in
#     the abstract; only the reply phase (rendering the fully-resolved
#     role_prompt, on the main engine) ever sees the literal 1000/writing_rules
#     values ---
story_agent = ThinkingAgent(
    name="Twist_Story_Writer",
    namespace="examples",
    llm_engine=llm,
    role_prompt=STORY_WRITER_ROLE_PROMPT,
    thinking_llm_engine=thinking_llm,
    thinking_instructions=(
        "You are planning a tightly word-budgeted short story built around a "
        "twist ending that recontextualizes the whole narrative. Each round, "
        "produce exactly one focused thought that builds on the rounds "
        "before it -- an observation about what the premise implies, a "
        "question about what the twist could be, a commitment to a specific "
        "twist, or what needs to be planted earlier for it to land. Focus "
        "your thoughts on what the twist should be and what needs to be set "
        "up beforehand for it to land. Respond with a single brief "
        "paragraph containing exactly one idea -- no headers, no numbered "
        "lists, no labeling your own response with a round number or any "
        "other heading; earlier rounds shown to you above are for context "
        "only, not a format to imitate."
    ),
    description="A short-fiction writer that scopes a twist ending via self-questioning before drafting.",
)

# --- the task itself declares genre + a small seed premise -- an open-ended
#     starting point, not a full plot. Figuring out an actual satisfying
#     twist (and what it requires setting up beforehand) from just this seed
#     takes several rounds of thinking-phase scoping questions before writing
#     is even possible. thinking_rounds is a per-invocation runtime
#     parameter -- passed here alongside the task's own inputs ---
task = (
    "Write a horror story about a lighthouse keeper who begins receiving radio "
    "transmissions from a ship that sank decades ago. The story's ending should "
    "recontextualize everything the reader believed up to that point -- a "
    "genuine twist, not a jump-scare."
)
result = story_agent.invoke({
    "prompt": task,
    "thinking_rounds": 6,
    "max_word_count": 1000,
    "writing_rules": (
        "Show, don't tell. Every sentence must either advance the plot or "
        "deepen a character. The twist must be reread-proof: on a second "
        "reading, every earlier detail should make sense in hindsight. No "
        "unearned deus ex machina resolutions."
    ),
})

print(f"TASK: {task}\n")
print("THINKING THOUGHTS (scoping the twist before writing, via the cheaper thinking engine):")
for round_index, thought in enumerate(story_agent.get_thoughts(result.run_id)):
    print(f"  Round {round_index}: {thought}")

print(f"\n~~~ FINAL STORY (drafted by the main engine) ~~~\n{result.result}")
