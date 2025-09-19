import sys
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import *
from modules.ToolAgents import PlannerAgent
from modules.Plugins import *

logging.getLogger().setLevel(level=logging.INFO) # Uncomment to see the logging info
my_planner = PlannerAgent("Context Enabled Planner",
                            description="Creates plans utilizing context memory",
                            llm_engine=OpenAIEngine(model="gpt-4o-mini"),
                            context_enabled=True)
my_planner.register(MathPlugin)
while True:
    query = input("Enter a planning task (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    result = my_planner.invoke(query)
    print(f"Result: {result}\n")