import sys
import os
from dotenv import load_dotenv
from atomic_agentic.agents import SelfAskAgent
from atomic_agentic.models.agents.prompts import PromptConfig
from atomic_agentic.llm import OpenAIEngine, GeminiEngine, MistralEngine, LlamaCppEngine

# --- some models produce non-ASCII punctuation (e.g. an arrow) in their
#     final answer; Windows' console defaults to a cp1252 codepage that
#     can't encode it, crashing a plain print() -- reconfigure stdout to
#     UTF-8 so this example runs regardless of the host console's codepage ---
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# --- define our agent's llm (openai, bedrock, azure, etc.) ---
llm = OpenAIEngine(api_key=os.getenv("OPENAI_API_KEY"), model = "gpt-5-mini")
# llm = GeminiEngine(api_key = os.getenv("GOOGLE_API_KEY"), model = "gemini-2.5-flash")
# llm = MistralEngine(api_key= os.getenv("MISTRAL_API_KEY"), model = "mistral-small-latest")
# llm = LlamaCppEngine(repo_id = "unsloth/phi-4-GGUF", filename= "phi-4-Q4_K_M.gguf", n_ctx = 512, verbose = False, n_threads=16)

# --- thinking_instructions is the parameterized side of this example
#     (Examples 1/2 parameterized role_prompt instead -- this one shows the
#     other combination: a plain string role_prompt below, paired with a
#     thinking_instructions PromptConfig that contributes its OWN
#     {verification_style} placeholder to the agent's schema, reconciled
#     against role_prompt's own params the same way at construction time).
#
#     The multi-round guarantee here is deliberately NOT just "ask nicely" --
#     a first pass that only told the model "one stage per round" still let
#     it resolve two stages in a single round as two separate thoughts, live
#     tested. thoughts_per_round=1 below is the real backstop: parse_thoughts
#     hard-truncates every round to its first thought, so even if the model
#     tries to pack multiple stages into one round as separate [CATEGORY]
#     lines, only the first survives -- the rest is silently discarded and
#     must be regenerated in a later round. This instructions text tells the
#     model about that mechanic explicitly, so it plans one deliberate
#     thought per round instead of wasting output that will be dropped ---
PUZZLE_THINKING_INSTRUCTIONS = PromptConfig(
    template="""\
You are working through a strictly sequential, multi-stage puzzle -- each
stage's clue only makes sense once the previous stage is fully resolved, and
you cannot know a later stage's answer without first solving every stage
before it.

Only your FIRST thought each round is kept -- anything after it is silently
discarded, and you get a fresh round to continue. So: produce exactly ONE
thought per round, covering exactly ONE stage. Do not try to resolve two
stages in the same round even as separate thoughts -- the second one will
simply be thrown away. Trust that the next round is coming; there is no
advantage to rushing ahead.

Verification style: {verification_style}
""",
    description="Sequential-puzzle-solving thinking instructions, parameterized by verification rigor.",
    field_specs={
        "verification_style": {
            "type": "str",
            "description": (
                "How rigorously to double-check each stage's answer before "
                "moving to the next (e.g. 'briefly sanity-check each stage's "
                "answer before proceeding' or 'move quickly, trusting your "
                "first answer')."
            ),
        },
    },
)

puzzle_agent = SelfAskAgent(
    name="Sequential_Puzzle_Solver",
    namespace="examples",
    llm_engine=llm,
    role_prompt="You are a meticulous puzzle solver who shows its work clearly.",
    thinking_instructions=PUZZLE_THINKING_INSTRUCTIONS,
    description="A puzzle solver whose task is structurally sequential, guaranteeing several rounds of self-questioning before an answer is possible.",
    max_thinking_rounds=8,
    thoughts_per_round=1,
)

# --- the task itself is a three-stage chain, each stage unsolvable without
#     the previous stage's answer -- there is no way to "skip ahead" to the
#     final answer, which is what makes multiple thinking rounds a
#     near-certainty rather than a matter of model preference ---
task = (
    "Solve this three-stage lock puzzle. Solve the stages strictly in order "
    "-- each stage's clue depends on the previous stage's answer and cannot "
    "be solved without it.\n\n"
    "Stage 1 (the number lock): The code is the number of letters in the "
    "longest word of this sentence: 'A resourceful fox often outsmarts a "
    "lazy hound.'\n\n"
    "Stage 2 (the color lock): Using the code from Stage 1, count that many "
    "letters into the alphabet (A=1, B=2, ...) to find a letter, then name "
    "a common color that starts with that letter.\n\n"
    "Stage 3 (the final door): The unlock phrase is the color from Stage 2, "
    "combined with the first letter of the animal named in Stage 1's "
    "sentence that is NOT the fox. State the final unlock phrase and "
    "briefly explain how each stage led to it."
)
result = puzzle_agent.invoke({
    "prompt": task,
    "verification_style": (
        "Briefly sanity-check each stage's answer against the clue before "
        "moving on to the next stage."
    ),
})

print(f"TASK: {task}\n")
print("SELF-ASK THOUGHTS (one stage resolved per round):")
for round_index, round_thoughts in enumerate(puzzle_agent.get_thoughts(result.run_id)):
    print(f"  Round {round_index}:")
    for thought in round_thoughts:
        print(f"    [{thought.category}] {thought.content}")

print(f"\nFINAL ANSWER:\n{result.result}")
