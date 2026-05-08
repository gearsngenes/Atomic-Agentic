from dotenv import load_dotenv
import logging
import math

from atomic_agentic.agents import PlanActAgent
from atomic_agentic.tools.Plugins import MATH_TOOLS, CONSOLE_TOOLS
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

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

# Register the pi constant
agent.register_constant("PI", math.pi, "Mathematical constant `pi`")

# ──────────────────────────  TASK  ─────────────────────────────
task_prompt = """
1) Compute the area of a circle with a radius of 5.
2) Compute the length of the hypotenuse of a triangle with legs a=3, b=4
3) Compute the volume of a cylinder with radius of 2 and height of 10.

Print each result as #) <question>: <answer> and print them IN THE ORDER GIVEN ORDER ABOVE.
"""

print("\n⇢ Executing math demo …")
agent.invoke({"prompt": task_prompt})
from pprint import pprint
print("BLACKBOARD AFTER MATH DEMO:")
pprint(agent.blackboard)
agent.clear_memory()
