"""01_tool_orchestrator_test.py

ReActAgent iteratively orchestrating math + console tools to solve a
multi-step arithmetic task: one tool call generated, resolved, and
dispatched per round, reacting to each result before deciding the next.

Analogous to PlanAct_Examples/00_plan_test.py, but no upfront plan is ever
written -- each round's single best next call is decided fresh against
everything completed so far this run.
"""
import logging
import math
from pprint import pprint

from atomic_agentic.agents import ReActAgent
from atomic_agentic.tools.prebuilt import BASIC_MATH_TOOLS, CONSOLE_TOOLS, EXPONENT_TOOLS

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)

orchestrator = ReActAgent(
    name="MathOrchestrator",
    namespace="examples",
    description="Orchestrates math + console tools to solve multi-step arithmetic tasks.",
    llm_engine=llm_engine,
    records_window=20,    # send-window (turns) to the model
    tool_calls_limit=15,  # max *non-return* tool calls per run
    context_enabled=True,
)

orchestrator.register_tools(BASIC_MATH_TOOLS)
orchestrator.register_tools(CONSOLE_TOOLS)
orchestrator.register_tools(EXPONENT_TOOLS)  # power/sqrt -- the task below needs both

orchestrator.register_constant(
    math.pi, "PI",
    "Use ONLY THIS constant in place of a literal or float for any calculations that involve it.",
)

task = """
1) Compute the area of a circle with a radius of 5 [A(r) = pi * r^2].
2) Compute the length of the hypotenuse of a triangle with legs a=3, b=4
3) Compute the volume of a cylinder with radius of 2 and height of 10 [V(r, h) = pi * r^2 * h].

Do NOT skip any steps, and do NOT attempt to combine them into a single calculation.

Print each result as #) <question>: <answer> and print them IN THE ORDER GIVEN ORDER ABOVE.
"""

final_result = orchestrator.invoke({"prompt": task})

print(f"\nFinal Result: {final_result.result}")

print("\nExecuted calls:\n")
pprint(orchestrator.get_conversation()[-1].statements)
