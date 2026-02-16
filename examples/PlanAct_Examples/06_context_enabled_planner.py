"""06_context_enabled_planner.py

Interactive PlanActAgent with context memory enabled.

Type a task, get an answer, then ask follow-ups that can refer to prior steps.

Updated to use PlanActAgent (formerly PlannerAgent).
"""
import logging

from dotenv import load_dotenv

from atomic_agentic.agents import PlanActAgent
from atomic_agentic.tools.Plugins import MATH_TOOLS
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

my_planner = PlanActAgent(
    name="Context_Enabled_Planner",
    description="Creates plans utilizing context memory",
    llm_engine=OpenAIEngine(model="gpt-4o-mini"),
    context_enabled=True,
    peek_at_cache=True,
)

my_planner.batch_register(MATH_TOOLS)

while True:
    query = input("Enter a planning task (or 'q' or 'exit' to quit): ")
    if query.lower() in ("exit", 'q'):
        break

    query = f"Use all available context to answer the query/task by the user: {query}"
    result = my_planner.invoke({"prompt": query})
    print(f"Result: {result}\n")
    print(f"Blackboard:\n{my_planner.blackboard_dumps(peek=False)}")
    print("-" * 40 + "\n")
