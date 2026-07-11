"""03_context_enabled_reactor.py

Interactive ReActAgent with context memory enabled.

Type a task, get an answer, then ask follow-ups that can refer to prior steps.

Analogous to PlanAct_Examples/06_context_enabled_planner.py but using a
ReActAgent: steps are generated one-at-a-time reactively rather than planned
upfront, so the number of LLM calls varies per run.
"""
import logging

from dotenv import load_dotenv

from atomic_agentic.agents import ReActAgent
from atomic_agentic.tools.prebuilt import MATH_TOOLS
from atomic_agentic.llm import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

reactor = ReActAgent(
    name="Context_Enabled_Reactor",
    namespace="examples",
    description="Solves math tasks step-by-step using context memory across turns.",
    llm_engine=OpenAIEngine(model="gpt-4o-mini"),
    context_enabled=True,
    peek_at_cache=True,
    tool_calls_limit=10,
    generation_retries=3,
)

reactor.batch_register(MATH_TOOLS)

while True:
    query = input("Enter a task (or 'q' / 'exit' to quit): ")
    if query.lower() in ("q", "exit"):
        break

    query = f"Use all available context to answer the query/task by the user: {query}"
    result = reactor.invoke({"prompt": query})
    print(f"Result: {result.result}\n")

    from pprint import pprint
    record = reactor.records[-1]
    llm_calls = len(record.llm_records)
    print(f"LLM calls this turn: {llm_calls}")
    print("Blackboard:")
    pprint(reactor.blackboard[record.blackboard_start:record.blackboard_end + 1])
    print("-" * 40 + "\n")
