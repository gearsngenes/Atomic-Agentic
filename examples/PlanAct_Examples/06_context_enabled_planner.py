"""06_context_enabled_planner.py

Interactive PlanActAgent with context memory enabled.

Type a task, get an answer, then ask follow-ups that can refer to prior steps.

Updated to use PlanActAgent (formerly PlannerAgent).
"""
import logging

from atomic_agentic.agents import PlanActAgent
from atomic_agentic.tools.prebuilt import BASIC_MATH_TOOLS

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)

my_planner = PlanActAgent(
    name="Context_Enabled_Planner",
    namespace="examples",
    description="Creates plans utilizing context memory",
    llm_engine=llm_engine,
    context_enabled=True,
    regeneration_limit=3,
)

my_planner.register_tools(BASIC_MATH_TOOLS)

while True:
    query = input("Enter a planning task (or 'q' or 'exit' to quit): ")
    if query.lower() in ("exit", 'q'):
        break

    query = f"Use all available context to answer the query/task by the user: {query}"
    result = my_planner.invoke({"prompt": query})
    print(f"Result: {result.result}\n")
    from pprint import pprint
    print("Executed calls:")
    record = my_planner.get_conversation()[-1]
    pprint(record.statements)
    print("-" * 40 + "\n")
