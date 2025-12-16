"""06_context_enabled_planner.py

Interactive PlanActAgent with context memory enabled.

Type a task, get an answer, then ask follow-ups that can refer to prior steps.

Updated to use PlanActAgent (formerly PlannerAgent).
"""
import logging

from dotenv import load_dotenv

from atomic_agentic.Agents import PlanActAgent
from atomic_agentic.Plugins import MATH_TOOLS
from atomic_agentic.LLMEngines import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

my_planner = PlanActAgent(
    name="Context Enabled Planner",
    description="Creates plans utilizing context memory",
    llm_engine=OpenAIEngine(model="gpt-4o-mini"),
    context_enabled=True,
    run_concurrent=False,
)

my_planner.batch_register(MATH_TOOLS)

while True:
    query = input("Enter a planning task (or 'q' or 'exit' to quit): ")
    if query.lower() in ("exit", 'q'):
        break

    result = my_planner.invoke({"prompt": query})
    print(f"Result: {result}\n")
