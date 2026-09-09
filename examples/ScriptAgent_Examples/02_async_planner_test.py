"""02_async_planner_test.py

Mirrors PlanAct_Examples/02_async_planner_test.py, rebuilt on ScriptAgent.

PlanAct's JSON plan marks ordering with a numeric "await" step index;
ScriptAgent's native grammar spells the same idea as a Python `await`
keyword directly in front of a call -- a pure ordering barrier that closes
the current dependency batch, forcing every later statement into a new one.
Five *bare* calls with no `await` and no data dependencies between them
land in a single batch and get dispatched concurrently (each sync callable
runs via asyncio.to_thread under the hood); five `await`-prefixed calls each
close their own batch, forcing five sequential round trips.

After the run, prints the reconstructed source via
ScriptAgentRecord.render_as_code() -- so you can see directly whether the
model actually used `await` or not, not just infer it from the timing.
This is the one deliberate deviation from the PlanAct original, which uses
`context_enabled=False` and never inspects history (a JSON plan isn't as
directly inspectable); ScriptAgent's render_as_code() is worth the extra
`context_enabled=True` here.
"""
import logging
import time

from atomic_agentic.agents import ScriptAgent

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)


def testDelayPrint() -> None:
    print("Called Print, now waiting 2 seconds...")
    time.sleep(2)


async_tester = ScriptAgent(
    name="Async_Delay_Tester",
    namespace="examples",
    description="Tests the ability to run independent statements concurrently.",
    llm_engine=llm_engine,
    context_enabled=True,
    planning_rounds_limit=1,
    generation_retries=2,
)

async_tester.register_tool(
    testDelayPrint,
    alias="DelayPrint",
    description="Delay for 5 seconds.",
)

if __name__ == "__main__":
    choice = input("Run steps sequentially? (y/n): ").strip().lower()
    pattern = "SEQUENTIALLY - await each call" if choice == "y" else "CONCURRENTLY - don't await any calls"
    prompt = f"Call 'DelayPrint' EXACTLY FIVE TIMES, but call them {pattern}."

    start = time.time()
    async_tester.invoke({"prompt": prompt})
    end = time.time()

    print("Time taken:", end - start, "seconds")

    record = async_tester.get_conversation()[-1]
    print("\n=== EXECUTED SCRIPT ===")
    print(record.render_as_code())
