from dotenv import load_dotenv
import logging

from atomic_agentic.agents import PlanActAgent
from atomic_agentic.tools.Plugins import MATH_TOOLS, CONSOLE_TOOLS, PARSER_TOOLS
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

print("\n───────────────────────────────\n")
print("Testing Task Decomposition and Printing capabilities")

# ──────────────────────────  SET-UP  ───────────────────────────
llm_engine = OpenAIEngine(model="gpt-5-mini")
agent = PlanActAgent(
    name="Test_PlanAct",
    description="Testing the prebuilt plugins with one-shot planning + execution.",
    llm_engine=llm_engine,
    context_enabled=True,
)

# Register tool lists
agent.batch_register(MATH_TOOLS)
agent.batch_register(CONSOLE_TOOLS)
agent.batch_register(PARSER_TOOLS)

# ──────────────────────────  TASK  ─────────────────────────────
task_prompt = """
Assume pi = 3.14159265358979323846 when needed.

Perform the following two tasks:
• Compute the area of a circle with a radius of 5.

• You have two legs of a right triangle: a=3, b=4.
  Compute the length of the hypotenuse c.

AFTERWARDS:
• Print your answers for each task, preceded by a label for what question they are answering.
• Return None as the end result.
"""

print("\n⇢ Executing math demo …")
agent.invoke({"prompt": task_prompt})
print("BLACKBOARD AFTER MATH DEMO:", agent.blackboard_dumps())
agent.clear_memory()

print("\n───────────────────────────────\n")
print("Now testing math and parsing capabilities…")
task_prompt = """
TASK
Given the string "<TAG>[23.4, 25.1, 22.8]<TAG>"
1. Extract the json string from the above input.
2. Load the resulting string into a list of numbers.
3. Print the list.
4. Calculate the average temperature of the list.
5. Print BOTH the extracted list's max value and its mean (each labeled).
"""

print("\n⇢ Executing parser+math demo …")
agent.invoke({"prompt": task_prompt})
print("BLACKBOARD AFTER PARSER DEMO:", agent.blackboard_dumps())
agent.clear_memory()

