import logging

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

# 4) Task (schema-first: mapping with 'prompt')
task = (
    "Calculate the sum of 42 and 58, multiply it by three, take the square root, "
    "then divide by four. Print the intermediate result to console as: "
    "PARTIAL RESULT: <value>. "
    "Finally, return three times that partial result."
)

final_result = orchestrator.invoke({"prompt": task})

print(f"\nFinal Result: {final_result}")
