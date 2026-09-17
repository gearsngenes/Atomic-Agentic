import sys
import os
import json
from dotenv import load_dotenv
from atomic_agentic.agents import ThinkingAgent
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

# --- thinking_schema constrains every thinking round to this exact shape.
#     Deliberately vocabulary that never claims to BE the answer: "focus" is
#     this round's own sub-problem, "leads" are candidate lines of reasoning
#     (each self-scored, giving a tree-of-thought-style breadth-then-narrow
#     move), "working_hypothesis" is scoped to the round's own focus and
#     explicitly revisable, and "ruled_out" records eliminated leads so a
#     later round can see what was already tried and rejected. Nothing here
#     is "the final answer" -- that's the reply phase's job alone ---
PUZZLE_THINKING_SCHEMA = {
    "type": "object",
    "properties": {
        "focus": {
            "type": "string",
            "description": "The sub-problem this round is working on.",
        },
        "leads": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "plausibility": {"type": "number"},
                },
                "required": ["reasoning", "plausibility"],
                "additionalProperties": False,
            },
            "description": "Candidate lines of reasoning worth exploring this round, each self-scored.",
        },
        "working_hypothesis": {
            "type": "string",
            "description": (
                "The most-supported lead to carry forward -- scoped to this "
                "round's own focus only, explicitly tentative and revisable. "
                "Never the puzzle's final solution."
            ),
        },
        "ruled_out": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Leads eliminated this round, and why.",
        },
    },
    "required": ["focus", "leads", "working_hypothesis", "ruled_out"],
    "additionalProperties": False,
}

# --- thinking_instructions is the parameterized side of this example
#     (Examples 1/2 parameterized role_prompt/thinking engine instead -- this
#     one shows a plain string role_prompt paired with a thinking_instructions
#     PromptConfig contributing its OWN {verification_style} placeholder,
#     reconciled against role_prompt's own params the same way at
#     construction time).
#
#     The schema alone only constrains SHAPE, not content -- nothing stops a
#     model from writing the puzzle's actual solution into working_hypothesis
#     if not told otherwise. The instruction below is the real guardrail: it
#     explicitly forbids stating the final unlock phrase during thinking, so
#     the solution only ever appears once, in the reply phase ---
PUZZLE_THINKING_INSTRUCTIONS = PromptConfig(
    template="""\
You are working through a strictly sequential, multi-stage puzzle -- each
stage's clue only makes sense once the previous stage is fully resolved, and
you cannot know a later stage's answer without first solving every stage
before it.

Each round, focus on exactly one stage. Propose a few candidate leads for
that stage with a plausibility score each, note anything you're ruling out
and why, and commit to a working hypothesis for THIS STAGE ONLY -- never
state the puzzle's overall final answer or unlock phrase while thinking, even
if you're confident you already know it. The final answer belongs solely in
your reply, after thinking concludes.

Keep every field's text plain prose with no headers, numbering, or round
labels of your own -- earlier rounds shown to you above are for context
only, not a format to imitate.

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

puzzle_agent = ThinkingAgent(
    name="Branching_Puzzle_Solver",
    namespace="examples",
    llm_engine=llm,
    role_prompt="You are a meticulous puzzle solver who shows its work clearly.",
    thinking_instructions=PUZZLE_THINKING_INSTRUCTIONS,
    thinking_schema=PUZZLE_THINKING_SCHEMA,
    description="A puzzle solver that explores multiple candidate leads per stage before committing, via schema-structured thinking rounds.",
)

# --- the task itself is a three-stage chain, each stage unsolvable without
#     the previous stage's answer -- there is no way to "skip ahead" to the
#     final answer, which is what makes multiple thinking rounds a
#     near-certainty rather than a matter of model preference. thinking_rounds
#     is a per-invocation runtime parameter, passed alongside the task's own
#     inputs ---
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
    "thinking_rounds": 3,
    "verification_style": (
        "Briefly sanity-check each stage's answer against the clue before "
        "moving on to the next stage."
    ),
})

print(f"TASK: {task}\n")
print("THINKING THOUGHTS (structured leads explored per stage):")
for round_index, thought in enumerate(puzzle_agent.get_thoughts(result.run_id)):
    print(f"  Round {round_index}:")
    print(json.dumps(thought, indent=4))

print(f"\nFINAL ANSWER:\n{result.result}")
