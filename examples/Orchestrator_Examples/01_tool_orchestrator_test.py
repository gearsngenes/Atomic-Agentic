import os, sys, logging
from pathlib import Path

# Setting the repo root on sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.ToolAgents import OrchestratorAgent
from modules.LLMEngines import OpenAIEngine
from modules.Plugins import MATH_TOOLS, CONSOLE_TOOLS  # lists of Tool objects

logging.getLogger().setLevel(logging.INFO)

# 1) LLM engine (model/key via env)
llm = OpenAIEngine("gpt-4o-mini")

# 2) Orchestrator (schema-driven, with explicit guards)
orchestrator = OrchestratorAgent(
    name="MathOrchestrator",
    description="Orchestrates math + console tools to solve multi-step arithmetic tasks.",
    llm_engine=llm,
    history_window=20,   # send-window (turns) to the model
    max_steps=15,        # safety: stop after N executed steps
    max_failures=4,      # safety: stop after M failed iterations
)

# 3) Register tool lists (no legacy Plugin classes)
orchestrator.batch_register(MATH_TOOLS)
orchestrator.batch_register(CONSOLE_TOOLS)

# 4) Task (schema-first: mapping with 'prompt')
task = (
    "Calculate the sum of 42 and 58, multiply it by three, then divide by four. "
    "Print the intermediate result to console as: PARTIAL RESULT: <value>. "
    "Finally, return three times that result."
)

final_result = orchestrator.invoke({"prompt": task})

print(f"\nFinal Result: {final_result}")
