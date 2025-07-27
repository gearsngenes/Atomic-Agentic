import sys
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.OrchestratorAgents import ToolOrchestratorAgent
from modules.LLMEngines import OpenAIEngine
from modules.Plugins import MathPlugin, ConsolePlugin
import modules.Prompts as Prompts

# Step 1: Set up the LLM engine
llm = OpenAIEngine(model="gpt-4o")  # or any OpenAI-compatible model

# Step 2: Instantiate the orchestrator agent
orchestrator = ToolOrchestratorAgent(
    name="MathOrchestrator",
    description="Orchestrates calls to math methods to solve problems",
    llm_engine=llm,
)

# Step 3: Register plugins
orchestrator.register(MathPlugin())
orchestrator.register(ConsolePlugin())

# Step 4: Give the orchestrator a task
task =  ("Calculate the sum of 42 and 58, multiply it by three, and divide it by four, "
        "and return the final result.")

# Step 5: Let it orchestrate step-by-step
final_result = orchestrator.invoke(task)

print(f"\nFinal Result: {final_result}")
