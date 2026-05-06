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
    context_enabled=False,
)

# 3) Register tool lists
orchestrator.batch_register(MATH_TOOLS)
orchestrator.batch_register(CONSOLE_TOOLS)

# 4) Register the pi constant
orchestrator.register_constant("PI", math.pi, "Mathematical constant `pi`")

# 4) Task (schema-first: mapping with 'prompt')
task = """
1) Compute the area of a circle with a radius of 5.
2) Compute the length of the hypotenuse of a triangle with legs a=3, b=4
3) Compute the volume of a cylinder with radius of 2 and height of 10.

Print each result as #) <question>: <answer> and print them IN THE ORDER GIVEN ORDER ABOVE.
"""

final_result = orchestrator.invoke({"prompt": task})

print(f"\nFinal Result: {final_result}")
