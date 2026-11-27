"""03_record_enabled_reactor.py

Interactive ReActAgent with context memory enabled.

Type a task, get an answer, then ask follow-ups that can refer to prior
turns' results.

Analogous to PlanAct_Examples/06_record_enabled_planner.py but using a
ReActAgent: one Python statement is generated, checked and run per
round -- reacting to each result before deciding the next -- rather than
a whole plan written upfront, so the number of LLM calls varies per run
instead of being fixed at one.
"""
import logging

from atomic_agentic.agents import ReActAgent

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prebuilt_tools import BASIC_MATH_TOOLS, TRIG_TOOLS, EXPONENT_TOOLS

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)

reactor = ReActAgent(
    name="Record_Enabled_Reactor",
    description="Solves math tasks step-by-step using context memory across turns.",
    llm_engine=llm_engine,
    record_enabled=True,
    tool_calls_limit=10,
    regeneration_limit=3,
)

reactor.register_tools(BASIC_MATH_TOOLS)
reactor.register_tools(TRIG_TOOLS)
reactor.register_tools(EXPONENT_TOOLS)

while True:
    query = input("Enter a task (or 'q' / 'exit' to quit): ")
    if query.lower() in ("q", "exit"):
        break

    query = f"Use all available context to answer the query/task by the user: {query}"
    result = reactor.invoke({"prompt": query})
    print(f"Result: {result.result}\n")

    record = reactor.get_conversation().records[-1]
    print(f"LLM calls this turn: {len(record.llm_records)}")
    print("Executed statements:")
    print(record.render_as_code())
    if record.failed_statements:
        print("Failed statements:")
        for statement in record.failed_statements:
            failed = next(step for step in statement.steps if step.status == "failed")
            print(f"  {statement.source}  # {failed.error!r}")
    print("-" * 40 + "\n")
