import sys,os
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import logging
# math_console_demo.py
from modules.Agents  import PlannerAgent
from modules.Plugins import ParserPlugin, MathPlugin, ConsolePlugin, PythonPlugin
from modules.LLMNuclei import *

logging.basicConfig(level=logging.INFO)

print("\n───────────────────────────────\n")
print("Testing MathPlugin + ConsolePlugin …")
# ──────────────────────────  SET-UP  ───────────────────────────
# define a global nucleus to give to each of our agents
nucleus = OpenAINucleus(model = "gpt-4o-mini")

planner = PlannerAgent(name="math-console-tester", nucleus=nucleus)
planner.register_plugin(MathPlugin())
planner.register_plugin(ConsolePlugin())      # for `print`

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
result = planner.invoke(task_prompt)
print(f"\nReturned value → {result}\n")

# ────────────────────────  PARSER + MATH DEMO  ─────────────────────
print("\n───────────────────────────────\n")
print("Now testing ParserPlugin + MathPlugin …")
planner = PlannerAgent(name="parser-math-tester", nucleus=nucleus)
planner.register_plugin(ParserPlugin())
planner.register_plugin(MathPlugin())
planner.register_plugin(ConsolePlugin())

task_prompt = """
TASK
Given the string ae9w8r98[23.4, 25.1, 22.8]afdadfew
1. Extract the JSON list of numbers from the string.
2. Print the extracted json string
3. Then parse it as a list.
4. Then calculate the average temperature of the list.
5. Print BOTH the extracted list's max value and its mean (each labeled as such).
"""
print("\n⇢ Executing parser+math demo …")
mean_val = planner.invoke(task_prompt)
print(f"\nMean returned → {mean_val}\n")

# ────────────────────────  PYTHON + CONSOLE DEMO  ─────────────────────
print("\n───────────────────────────────\n")
print("Now testing PythonPlugin + ConsolePlugin …")
planner = PlannerAgent(name="python-type-tester", nucleus=nucleus)
planner.register_plugin(PythonPlugin())
planner.register_plugin(ConsolePlugin())

task_prompt = """
TASK
• Determine the type of the following object: {'a': [1, 2, 3], 'b': 4.5}
  using PythonPlugin.get_type.
• Print the resulting type name via ConsolePlugin.print.
• Return the type name.
"""
print("\n⇢ Executing python-type demo …")
tname = planner.invoke(task_prompt)
print(f"\nType returned → {tname}\n")
