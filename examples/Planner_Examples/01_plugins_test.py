import sys,os
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import logging
# math_console_demo.py
from modules.ToolAgents import PlannerAgent
from modules.Plugins import ParserPlugin, MathPlugin, ConsolePlugin, PythonPlugin
from modules.LLMEngines import *

logging.getLogger().setLevel(level=logging.INFO)

print("\n───────────────────────────────\n")
print("Testing Task Decomposition and Printing capabilities")
# ──────────────────────────  SET-UP  ───────────────────────────
# define a global llm_engine to give to each of our agents
llm_engine = OpenAIEngine(model = "gpt-4o-mini")

planner = PlannerAgent(name="Test-Planner", description="Testing the prebuilt plugins", llm_engine=llm_engine)
planner.register(MathPlugin())
planner.register(ConsolePlugin())
planner.register(ParserPlugin())
planner.register(PythonPlugin())

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
planner.invoke(task_prompt)

# ────────────────────────  PARSER + MATH DEMO  ─────────────────────
print("\n───────────────────────────────\n")
print("Now testing math and parsing capabilities…")
task_prompt = """
TASK
Given the string "[23.4, 25.1, 22.8]"
1. Parse the JSON list of numbers from the string.
2. Print the list
3. Then calculate the average temperature of the list.
4. Print BOTH the extracted list's max value and its mean (each labeled as such).
"""
print("\n⇢ Executing parser+math demo …")
planner.invoke(task_prompt)

# ────────────────────────  PYTHON + CONSOLE DEMO  ─────────────────────
print("\n───────────────────────────────\n")
print("Now testing type-determining and return…")

task_prompt = """
TASK
• Return the type value of the following object: {'a': [1, 2, 3], 'b': 4.5}
  using PythonPlugin.get_type, and format the returned result as:
  "TYPE RESULT -- <result here>".
"""
print("\n⇢ Executing python-type demo …")
tname = planner.invoke(task_prompt)
print("RETURNED VALUE: ", tname)
