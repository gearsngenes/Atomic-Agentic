"""03_context_enabled_reactor.py

Interactive ReActAgent with context memory enabled.

Type a task, get an answer, then ask follow-ups that can refer to prior
turns' results.

Analogous to PlanAct_Examples/06_context_enabled_planner.py but using a
ReActAgent: one tool call is generated, resolved, and dispatched per
round -- reacting to each result before deciding the next -- rather than
a whole plan written upfront, so the number of LLM calls varies per run
instead of being fixed at one.
"""
import logging
from pprint import pprint

from atomic_agentic.agents import ReActAgent
from atomic_agentic.tools.prebuilt import BASIC_MATH_TOOLS

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)

reactor = ReActAgent(
    name="Context_Enabled_Reactor",
    namespace="examples",
    description="Solves math tasks step-by-step using context memory across turns.",
    llm_engine=llm_engine,
    context_enabled=True,
    tool_calls_limit=10,
    regeneration_limit=3,
)

reactor.register_tools(BASIC_MATH_TOOLS)

while True:
    query = input("Enter a task (or 'q' / 'exit' to quit): ")
    if query.lower() in ("q", "exit"):
        break

    query = f"Use all available context to answer the query/task by the user: {query}"
    result = reactor.invoke({"prompt": query})
    print(f"Result: {result.result}\n")

    record = reactor.get_conversation()[-1]
    print(f"LLM calls this turn: {len(record.llm_records)}")
    print("Executed calls:")
    pprint(record.statements)
    if record.failed_statements:
        print("Failed calls:")
        pprint(record.failed_statements)
    print("-" * 40 + "\n")
