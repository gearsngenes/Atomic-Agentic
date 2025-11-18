from dotenv import load_dotenv
import logging
from atomic_agentic.ToolAgents import PlannerAgent
from atomic_agentic.Plugins import MATH_TOOLS, CONSOLE_TOOLS, PARSER_TOOLS
from atomic_agentic.LLMEngines import OpenAIEngine

load_dotenv()

logging.getLogger().setLevel(level=logging.INFO)

print("\n───────────────────────────────\n")
print("Testing Task Decomposition and Printing capabilities")

# ──────────────────────────  SET-UP  ───────────────────────────
llm_engine = OpenAIEngine(model="gpt-4o-mini")
planner = PlannerAgent(name="Test-Planner", description="Testing the prebuilt plugins", llm_engine=llm_engine)

# Register tool lists
planner.batch_register(MATH_TOOLS)
planner.batch_register(CONSOLE_TOOLS)
planner.batch_register(PARSER_TOOLS)

# ──────────────────────────  TASK  ─────────────────────────────
task_prompt = """
Assume pi = 3.14159265358979323846 when needed.

Perform the following two tasks:
TASK 1:
• Compute the area of a circle with a radius of 5.

TASK 2:
• You have two legs of a right triangle: a=3, b=4.
  Compute the length of the hypotenuse c.

AFTERWARDS:
• Print your answers for each task, preceded by a label for what
  question they are answering.
• Return None as the end result.
"""

print("\n⇢ Executing math demo …")
planner.invoke({"prompt": task_prompt})

print("\n───────────────────────────────\n")
print("Now testing math and parsing capabilities…")
task_prompt = """
TASK
Given the string "[23.4, 25.1, 22.8]"
1. Extract the JSON string, then parse it into a list of numbers.
2. Print the list.
3. Calculate the average temperature of the list.
4. Print BOTH the extracted list's max value and its mean (each labeled).
"""
print("\n⇢ Executing parser+math demo …")
planner.invoke({"prompt": task_prompt})
