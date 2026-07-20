from dotenv import load_dotenv
import logging
import math

from atomic_agentic.agents import ReActKAgent
from atomic_agentic.tools.prebuilt import MATH_TOOLS, CONSOLE_TOOLS
from atomic_agentic.llm import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

print("Testing Task Decomposition and Printing capabilities (ReActK)")

# ──────────────────────────  SET-UP  ───────────────────────────
llm_engine = OpenAIEngine(model="gpt-4o-mini")
agent = ReActKAgent(
    name="Test_ReActK",
    namespace="examples",
    description="Testing the prebuilt plugins with reactive, per-round batched planning.",
    llm_engine=llm_engine,
    context_enabled=True,
    steps_per_round=2,   # deliberately below the task's natural batch size of 3
    tool_calls_limit=10,
)

# Register tool lists
agent.batch_register(MATH_TOOLS)
agent.batch_register(CONSOLE_TOOLS)

# Register the pi constant
agent.register_constant("PI", math.pi, "Mathematical constant `pi`")

# ──────────────────────────  TASK  ─────────────────────────────
task_prompt = """
1) Compute the area of a circle with a radius of 5 [A(r) = pi * r^2].
2) Compute the length of the hypotenuse of a triangle with legs a=3, b=4
3) Compute the volume of a cylinder with radius of 2 and height of 10 [V(r, h) = pi * r^2 * h].

Print each result as #) <question>: <answer> and print them IN THE ORDER GIVEN ORDER ABOVE.
Then return None.
"""

print("\n⇢ Executing math demo (steps_per_round=2, forcing multiple rounds) …")
result = agent.invoke({"prompt": task_prompt})
from pprint import pprint
print("\n=== FINAL AGENT RESULT ===")
pprint(result)
print("BLACKBOARD AFTER MATH DEMO:")
pprint(agent.blackboard)
agent.clear_memory()
