import logging
import math

from dotenv import load_dotenv

from atomic_agentic.agents import ReActAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine
from atomic_agentic.tools.Plugins import CONSOLE_TOOLS, MATH_TOOLS  # lists of Tool objects

load_dotenv()  # take environment variables from .env file (if exists)

logging.basicConfig(level=logging.INFO)

# 1) LLM engine (model/key via env)
llm = OpenAIEngine(model="gpt-4o-mini")

# 2) ReAct-style Orchestrator (iterative tool-use)
orchestrator = ReActAgent(
    name="MathOrchestrator",
    description="Orchestrates math + console tools to solve multi-step arithmetic tasks.",
    llm_engine=llm,
    history_window=20,    # send-window (turns) to the model
    tool_calls_limit=15,  # max *non-return* tool calls per run
    context_enabled=True,
)

# 3) Register tool lists
orchestrator.batch_register(MATH_TOOLS)
orchestrator.batch_register(CONSOLE_TOOLS)

# 4) Register the pi constant
orchestrator.register_constant("PI", math.pi, "Mathematical constant `PI`")

# 4) Task (schema-first: mapping with 'prompt')
task = """
Complete the following problem:
1. Compute the volume of a cylinder with a radius of 2 and a height of 10 [V(r, h) = pi * r^2 * h].
2. Then print the result in the format "The volume of the cylinder is: <result>".
Return None.
"""

final_result = orchestrator.invoke({"prompt": task})

print(f"\nFinal Result: {final_result}")
from pprint import pprint
pprint(orchestrator.blackboard)
