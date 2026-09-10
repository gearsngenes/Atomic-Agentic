"""04_planner_delegator.py

Task-decomposition/delegation, deliberately redesigned from
PlanAct_Examples/04_planner_delegator.py rather than ported verbatim.

PlanAct's version needs two tool calls per item (generate, then a separate
print_* call fed the generated value) -- meaning every print call has a real
data dependency on its own generate call, and batching quality then depends
on whether the model interleaves item-by-item or front-loads all generates
before all prints.

Here, each leaf specialist bakes its own printing into `post_invoke`
(pre(question) -> prompt string; post(answer, question) -> prints fancily,
returns answer) -- `premise`/`question` appear in both hook signatures and
get reconciled into one exposed parameter automatically (Agent's own
pre/post N-way reconciliation; no context_keys needed). So from the
Delegator's point of view, EVERY item is one complete, self-contained,
fire-and-forget call -- there is no second call depending on a first one.
With zero data dependencies among any of the ~10 delegator-level calls
(5 haiku + 5 math, two different tool types mixed together),
`compile_batches` puts all of them in one concurrent batch regardless of
what order the model writes them in -- no interleaving pitfall to even
worry about, unlike the two-call-per-item design. It also means each call
is a short literal arg (`premise="..."`/`question="..."`), never a large
verbatim string block to reproduce -- a materially lower-risk delegation
payload than handing sub-planners a whole precomputed task string.

MathSpecialist is itself a nested ScriptAgent with its own math toolbox and
its own tool_calls_limit/planning_rounds_limit -- that internal budget is
completely invisible to the Delegator, which only ever sees one call per
question no matter how many math tools it took inside.

`context_enabled=False` on both leaves is deliberate, not just default
statelessness: the Delegator dispatches several concurrent calls into the
SAME leaf instance (e.g. 5 concurrent HaikuWriter calls), and each call is
already fully self-contained -- cross-call memory would buy nothing and
would only produce nondeterministically-interleaved history under real
concurrency. The Delegator itself stays `context_enabled=True` so its own
render_as_code() can be inspected afterward.
"""
import time
import logging
from typing import Any

from atomic_agentic.agents import BasicAgent, ScriptAgent
from atomic_agentic.tools.prebuilt import EXPONENT_TOOLS, STAT_TOOLS

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)

# ──────────────────────────  HAIKU WRITER  ──────────────────────────


def haiku_pre(premise: str) -> str:
    return premise


def haiku_post(poem: str, premise: str) -> str:
    print(f"---\n**{premise}**\n{poem}\n---")
    return poem


haiku_writer = BasicAgent(
    name="HaikuWriter",
    namespace="examples",
    description="Writes and prints a 3-line 5-7-5 haiku for the given premise, returning the final poem.",
    llm_engine=llm_engine,
    role_prompt=(
        "You are a master of writing haiku. Given a topic, write a "
        "3-line haiku about it, following a 5-7-5 syllable structure. "
        "Be creative; do not just restate the topic."
    ),
    context_enabled=False,
    pre_invoke=haiku_pre,
    post_invoke=haiku_post,
)

# ──────────────────────────  MATH SPECIALIST  ───────────────────────


def math_pre(question: str) -> str:
    return f"Compute the result of: {question}"


def math_post(answer: Any, question: str) -> Any:
    print(f"\U0001f9ee {question} = {answer}")
    return answer


math_specialist = ScriptAgent(
    name="MathSpecialist",
    namespace="examples",
    description="Solves one math question and prints the question/answer pair, while returning the final result.",
    llm_engine=llm_engine,
    context_enabled=False,
    tool_calls_limit=3,
    planning_rounds_limit=1,
    pre_invoke=math_pre,
    post_invoke=math_post,
)
math_specialist.register_tools(STAT_TOOLS)
math_specialist.register_tools(EXPONENT_TOOLS)

# ──────────────────────────  DELEGATOR  ─────────────────────────────

delegator = ScriptAgent(
    name="Delegator",
    namespace="examples",
    description="Delegates each item in a batch to the appropriate specialist agent.",
    llm_engine=llm_engine,
    context_enabled=True,
    planning_rounds_limit=1,
)
delegator.register_tool(haiku_writer)
delegator.register_tool(math_specialist)

# ──────────────────────────  RUN  ───────────────────────────────────

haiku_prompts = [
    "A frog jumps in pond",
    "Autumn leaves falling",
    "Snow on mountain peak",
    "A roaring fire",
    "balsam flowers",
]

math_problems = [
    "12 * 8 + 5",
    "9 plus 7",
    "(3 + 4) * 2",
    "2^5-3",
    "maximum of [3, 7, 2, 9, 4]",
]

if __name__ == "__main__":
    # One HaikuWriter call per topic + one MathSpecialist call per problem;
    # small slack, no separate budget slot for printing (baked into each
    # leaf's own post_invoke) or for `return` (a free language terminal).
    delegator.tool_calls_limit = len(haiku_prompts) + len(math_problems) + 1

    task_prompt = (
        f"Write a haiku for each topic here:\n{haiku_prompts}\n\n"
        f"Solve each math problem here:\n{math_problems}\n\n"
        "Each call already prints its own result -- you don't need to print, bind, or return anything yourself."
    )

    print("\n⇢ Planning + execution …")
    start = time.time()
    delegator.invoke({"prompt": task_prompt})
    elapsed = time.time() - start
    print(f"\nTotal elapsed: {elapsed:.2f}s")

    record = delegator.get_conversation()[-1]
    print("\n=== EXECUTED SCRIPT ===")
    print(record.render_as_code())
