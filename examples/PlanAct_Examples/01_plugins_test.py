from dotenv import load_dotenv
import logging

from atomic_agentic.agents import PlanActAgent
from atomic_agentic.tools.Plugins import MATH_TOOLS, CONSOLE_TOOLS
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

print("\n───────────────────────────────\n")
print("Testing Task Decomposition and Printing capabilities")

# ──────────────────────────  SET-UP  ───────────────────────────
llm_engine = OpenAIEngine(model="gpt-4o-mini")
agent = PlanActAgent(
    name="Test_PlanAct",
    description="Testing the prebuilt plugins with one-shot planning + execution.",
    llm_engine=llm_engine,
    context_enabled=True,
)

# Register tool lists
agent.batch_register(MATH_TOOLS)
agent.batch_register(CONSOLE_TOOLS)

# ──────────────────────────  TASK  ─────────────────────────────
task_prompt = """
Assuming pi = 3.14159265358979323846 where needed

1) Compute the area of a circle with a radius of 5 (pi * r^2).
2) Compute the length of the hypotenuse of a triangle with legs a=3, b=4
3) Compute the volume of a cylinder with radius of 2 and height of 10 (pi * r^2 * h).

Print each result as #) <question>: <answer> and print them IN THE ORDER GIVEN ORDER ABOVE.
"""

print("\n⇢ Executing math demo …")
agent.invoke({"prompt": task_prompt})
print("BLACKBOARD AFTER MATH DEMO:", agent.blackboard_dumps())
agent.clear_memory()
