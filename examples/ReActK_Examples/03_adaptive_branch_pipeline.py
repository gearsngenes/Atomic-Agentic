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
    tool_calls_limit=20,
)

agent.batch_register(MATH_TOOLS)
agent.batch_register(CONSOLE_TOOLS)
agent.register(is_prime, name="IsPrime", description="Return True if n is a prime number, else False.")
agent.register(is_even, name="IsEven", description="Return True if n is even, else False.")

# ──────────────────────────  TASK  ─────────────────────────────
task_template = """
Given x = {x}:
1. Compute y = three times x minus one
2. Check if y is prime or even.
3. Based on what those checks show, do one of the following:
    a. If it is prime, compute and return y**2
    b. Else if y is even, compute and return y/2
    c. Else if y is odd, compute and return y + 7
4. Return the final computed result.

DO NOT SKIP STEPS
"""

# Seeds chosen so 3*x - 1 lands in each branch:
#   x=1  -> y=2  (prime AND even -> tests prime-priority tie-break)
#   x=3  -> y=8  (composite, even -> branch B)
#   x=12 -> y=35 (composite, odd  -> branch C)
runs = [
    (1, "prime (priority over even)"),
    # (3, "composite, even"),
    # (12, "composite, odd"),
]

from pprint import pprint

for x, expected_branch in runs:
    print(f"\n⇢ Running x={x} (expected branch: {expected_branch}) …")
    result = agent.invoke({"prompt": task_template.format(x=x)})
    print(f"=== RESULT for x={x} ===")
    pprint(f"FINAL RESULT: {result.result}")
    print(f"=== BLACKBOARD for x={x} ===")
    pprint(agent.blackboard)
    agent.clear_memory()
