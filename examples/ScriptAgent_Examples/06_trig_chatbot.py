"""06_trig_chatbot.py

A fixed, scripted sequence of trig/geometry questions run as separate turns
against the same context_enabled=True ScriptAgent -- testing whether it
correctly reuses an EARLIER turn's result via ScriptAgent's own
task_result_i cross-invocation addressing (render_turn/_initialize_task),
rather than v1 ToolAgent's <<__cN__>> placeholder scheme.

ONESHOT_PLANNER_PROMPT's STRICT RULES #3 already instructs the model to
prefer a pre-existing declared name (a constant, an earlier result,
task_result_i) over hand-writing an equivalent value -- this example is a
live test of whether that instruction actually gets followed, not just
stated.

Turn sequence, each deliberately probing a different angle:
1. Establish task_result_0 via a real tool call (sin(PI/6) = 0.5).
2. Immediate reuse: square task_result_0 without recomputing sin() again.
3. Non-adjacent reuse: task_result_0 + task_result_1 together (0.75) --
   deliberately not a "nice" number either input alone would produce, so a
   correct answer here is only reachable by genuinely reusing BOTH prior
   results, not a lucky guess.
4. Bare-arithmetic use of the PI constant (degree -> radian conversion) --
   STRICT RULES #1 already permits call-free arithmetic, so this needs no
   dedicated conversion tool, only PI itself.
5. Full-history recall: identify and restate the very first turn's question
   AND answer together -- the hardest reuse case, since it needs both a
   piece of `task_result_i` (the answer) and something never stored as a
   value at all (the ORIGINAL PROMPT TEXT of turn 0), which only lives in
   the rendered conversation history, not the cache.

After each turn, prints the answer plus record.render_as_code() -- that's
what makes reuse vs. hallucination vs. wasted recompute actually visible,
not just the final numeric answer (which a lucky guess could also match).
"""
import logging
import math

from atomic_agentic.agents import ScriptAgent
from atomic_agentic.tools.prebuilt import TRIG_TOOLS

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)

# Curated subset -- the 6 primary trig/inverse-trig functions. TRIG_TOOLS'
# cotangent/hyperbolic variants add prompt surface without adding anything
# this example is testing.
_trig_by_name = {tool.name: tool for tool in TRIG_TOOLS}
CORE_TRIG_TOOLS = [_trig_by_name[name] for name in ("sin", "cos", "tan", "asin", "acos", "atan")]

chatbot = ScriptAgent(
    name="TrigChatbot",
    namespace="examples",
    description="Answers trig/geometry questions conversationally, reusing prior answers when relevant.",
    llm_engine=llm_engine,
    context_enabled=True,
    tool_calls_limit=5,
)
chatbot.register_tools(CORE_TRIG_TOOLS)
chatbot.register_constant(math.pi, alias="PI", description="The mathematical constant pi.")

TURNS = [
    "What is the sine of PI/6?",
    "Now square that result.",
    "Add my first answer to my second answer.",
    "Convert 45 degrees to radians using PI, then give me its cosine.",
    "What was the first question I asked you? And what was its answer? format it as a '<q>: <a>' string",
]

if __name__ == "__main__":
    for i, turn in enumerate(TURNS):
        print(f"\n{'=' * 60}\nTurn {i}: {turn}")
        result = chatbot.invoke({"prompt": turn})
        print(f"Bot: {result.result}")

        record = chatbot.get_conversation()[-1]
        print("--- Executed script this turn ---")
        print(record.render_as_code())
