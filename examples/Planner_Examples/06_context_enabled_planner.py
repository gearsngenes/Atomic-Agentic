from dotenv import load_dotenv
import logging
from atomic_agentic.ToolAgents import PlannerAgent
from atomic_agentic.Plugins import MATH_TOOLS
from atomic_agentic.LLMEngines import OpenAIEngine

load_dotenv()

logging.basicConfig(level=logging.INFO)

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
