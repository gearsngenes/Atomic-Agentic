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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prebuilt_tools import BASIC_MATH_TOOLS, CONSOLE_TOOLS, EXPONENT_TOOLS

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
    tool_instructions="""
    You are a mathematical assistant who solves lists of math problems and displays their results ONE AT A TIME.
    Before printing your final answer, make sure you do ALL the necessary calculations and partial steps needed,
    AND use the mathematical PI CONSTANT wherever applicable instead of a hard-coded number.
    Each question and its answer should be printed in the format below:
    "<question>: {{calculated_answer}}"
    NEVER print the same question-answer pair twice.
    """
    
)

orchestrator.register_tools(BASIC_MATH_TOOLS)
orchestrator.register_tools(CONSOLE_TOOLS)
orchestrator.register_tools(EXPONENT_TOOLS)  # power/sqrt -- the task below needs both

orchestrator.register_constant(
    math.pi, "PI",
    "Mathematical constant `pi` placeholder",
)

task = """
Solve EACH question and print its result:
1) The area of a circle with a radius of 5 (pi * r^2).
2) The hypotenuse length of a triangle with legs a=3, b=4
3) The volume of a cylinder with radius of 2 and height of 10 (pi * r^2 * h).
"""

final_result = orchestrator.invoke({"prompt": task})

print(f"\nFinal Result: {final_result.result}")

print("\nExecuted calls:\n")
print(orchestrator.get_conversation()[-1].render_as_code())
