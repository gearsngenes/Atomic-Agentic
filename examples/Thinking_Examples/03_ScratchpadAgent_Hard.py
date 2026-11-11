import sys
import os
from dotenv import load_dotenv
from atomic_agentic.agents import ScratchpadAgent
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
# llm = LlamaCppEngine(model_path=os.getenv("LLAMA_MODEL_PATH"), repo_id = "unsloth/phi-4-GGUF", filename= "phi-4-Q4_K_M.gguf", n_ctx = 512, verbose = False, n_threads=16)

# --- thinking_instructions is the parameterized side of this example
#     (Example 1 used plain prompts and example 2 a parameterized
#     role_prompt -- this one shows a plain string role_prompt paired with a
#     thinking_instructions PromptConfig contributing its OWN
#     {verification_style} placeholder, reconciled against role_prompt's own
#     params the same way at construction time).
#
#     Thoughts are free text, so the instructions below carry the whole
#     structure (focus, leads, working hypothesis, ruled-out leads) and are the
#     guardrail: they explicitly forbid stating the final unlock phrase during
#     thinking, so the solution only ever appears once, in the reply phase ---
PUZZLE_THINKING_INSTRUCTIONS = PromptConfig(
    template="""\
You are working through a strictly sequential, multi-stage puzzle -- each
stage's clue only makes sense once the previous stage is fully resolved, and
you cannot know a later stage's answer without first solving every stage
before it.

Each round, focus on exactly one stage. Propose a few candidate leads for
that stage with a plausibility score each (0 to 1), note anything you're
ruling out and why, and commit to a working hypothesis for THIS STAGE ONLY. Write
each round as labelled lines: Focus, Leads, Working hypothesis, Ruled out. Never
state the puzzle's overall final answer or unlock phrase while thinking, even
if you're confident you already know it. The final answer belongs solely in
your reply, after thinking concludes.

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

puzzle_agent = ScratchpadAgent(
    name="Branching_Puzzle_Solver",
    namespace="examples",
    llm_engine=llm,
    thinking_rounds=3,
    role_prompt="You are a meticulous puzzle solver who shows its work clearly.",
    thinking_instructions=PUZZLE_THINKING_INSTRUCTIONS,
    description="A puzzle solver that explores multiple candidate leads per stage before committing, via multi-round labelled thinking.",
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

record = puzzle_agent.get_conversation().records[-1:][0]

print(f"TASK: {task}\n")
print("THINKING THOUGHTS (leads explored per stage):")
for round_index, thought in enumerate(record.thoughts):
    print(f"  Round {round_index}:")
    print(thought)

print(f"\nTHOUGHTS PRODUCED: {result.thoughts_produced}")
print(f"\nFINAL ANSWER:\n{result.result}")
