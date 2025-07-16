import logging
# math_console_demo.py
from atomic_agents.Agents  import PlannerAgent
from atomic_agents.Plugins import ParserPlugin, MathPlugin, ConsolePlugin, PythonPlugin

logging.basicConfig(level=logging.INFO)
print("\n───────────────────────────────\n")
print("Testing MathPlugin + ConsolePlugin …")
# ──────────────────────────  SET-UP  ───────────────────────────
planner = PlannerAgent(name="math-console-tester", debug=True)
planner.register_plugin(MathPlugin())
planner.register_plugin(ConsolePlugin())      # for `print`

# ──────────────────────────  TASK  ─────────────────────────────
task_prompt = """
Assume pi = 3.14159265358979323846 when needed.

Perform the following two tasks:
TASK 1:
• Compute the area of a circle with a radius of 5.

TASK 2:
• A triangle with an angle of 30 degrees has two sides of
  lengths 3 and 4 making up the angle. Compute the length
  of the third side using the law of cosines.

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
planner = PlannerAgent(name="parser-math-tester", debug=True)
planner.register_plugin(ParserPlugin())
planner.register_plugin(MathPlugin())
planner.register_plugin(ConsolePlugin())

task_prompt = """
TASK
Given the text: "Temperatures (C): [23.4, 25.1, 22.8]"
1. Extract a json string from the text
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
planner = PlannerAgent(name="python-type-tester", debug=True)
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
