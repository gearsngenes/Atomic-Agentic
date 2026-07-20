from dotenv import load_dotenv
import logging

from atomic_agentic.agents import ReActKAgent
from atomic_agentic.tools.prebuilt import MATH_TOOLS, CONSOLE_TOOLS
from atomic_agentic.llm import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

print("Testing adaptive, cross-round branch selection (ReActK)")


def is_prime(n: int) -> bool:
    n = int(n)
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def is_even(n: int) -> bool:
    return int(n) % 2 == 0


# ──────────────────────────  SET-UP  ───────────────────────────
llm_engine = OpenAIEngine(model="gpt-4o-mini")
agent = ReActKAgent(
    name="Adaptive_Branch_Tester",
    namespace="examples",
    description="Classifies a number then dispatches to a branch-specific operation.",
    llm_engine=llm_engine,
    context_enabled=False,
    steps_per_round=2,   # too small to fit compute+classify+branch in one round
    tool_calls_limit=10,
)

agent.batch_register(MATH_TOOLS)
agent.batch_register(CONSOLE_TOOLS)
agent.register(is_prime, name="IsPrime", description="Return True if n is a prime number, else False.")
agent.register(is_even, name="IsEven", description="Return True if n is even, else False.")

# ──────────────────────────  TASK  ─────────────────────────────
task_template = """
Given x = {x}:
1. Compute y = 3 * x - 1.
2. Determine whether y is prime, and independently whether y is even.
3. Apply EXACTLY ONE of the following branches. Prime status takes priority
   over even/odd:
   - If y is prime: compute y ** 2.
   - Else if y is even: compute y / 2.
   - Else (y is odd): compute y + 7.
   You cannot know which branch applies until you see the actual results of
   the primality/evenness checks — do not guess ahead of that.
4. Print the final computed value.
5. Return the final computed value.
"""

# Seeds chosen so 3*x - 1 lands in each branch:
#   x=1  -> y=2  (prime AND even -> tests prime-priority tie-break)
#   x=3  -> y=8  (composite, even -> branch B)
#   x=12 -> y=35 (composite, odd  -> branch C)
runs = [
    (1, "prime (priority over even)"),
    (3, "composite, even"),
    (12, "composite, odd"),
]

from pprint import pprint

for x, expected_branch in runs:
    print(f"\n⇢ Running x={x} (expected branch: {expected_branch}) …")
    result = agent.invoke({"prompt": task_template.format(x=x)})
    print(f"=== RESULT for x={x} ===")
    pprint(result)
    agent.clear_memory()
