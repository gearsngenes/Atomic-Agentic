import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import logging
from modules.ToolAgents import PlannerAgent
from modules.Plugins import MATH_TOOLS
from modules.LLMEngines import OpenAIEngine

logging.getLogger().setLevel(level=logging.INFO)

my_planner = PlannerAgent(
    "Context Enabled Planner",
    description="Creates plans utilizing context memory",
    llm_engine=OpenAIEngine(model="gpt-4o-mini"),
    context_enabled=True
)

my_planner.batch_register(MATH_TOOLS)

while True:
    query = input("Enter a planning task (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    result = my_planner.invoke({"prompt": query})
    print(f"Result: {result}\n")
