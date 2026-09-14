"""02_async_planner_test.py

Mirrors PlanAct_Examples/02_async_planner_test.py, rebuilt on ScriptAgent.

PlanAct's JSON plan marks ordering with a numeric "await" step index;
ScriptAgent has no model-facing concurrency signal at all anymore -- five
*bare* calls with no data dependencies between them always batch together
and dispatch concurrently (each sync callable runs via asyncio.to_thread
under the hood) unless the *developer* caps it. `tool_concurrency_limit`
is that cap: `None` (the default) leaves the dependency-inferred batching
alone; `1` forces every dispatched call into its own batch, i.e. strictly
sequential, matching CodeAct/`CodeAgent`'s own fully-sequential posture.

After the run, prints the reconstructed source via
ScriptAgentRecord.render_as_code() -- its `# Batch N:` grouping is the
visible proof of what actually happened, replacing the old "did the model
write await" inspection (there's nothing left for the model to write or
omit here). This is the one deliberate deviation from the PlanAct original,
which uses `context_enabled=False` and never inspects history (a JSON plan
isn't as directly inspectable); ScriptAgent's render_as_code() is worth the
extra `context_enabled=True` here.
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
)

async_tester.register_tool(
    testDelayPrint,
    alias="DelayPrint",
    description="Delay for 5 seconds.",
)

if __name__ == "__main__":
    choice = input("Run steps sequentially? (y/n): ").strip().lower()
    async_tester.tool_concurrency_limit = 1 if choice == "y" else None
    prompt = "Call 'DelayPrint' EXACTLY FIVE TIMES."

    start = time.time()
    async_tester.invoke({"prompt": prompt})
    end = time.time()

    print("Time taken:", end - start, "seconds")

    record = async_tester.get_conversation()[-1]
    print("\n=== EXECUTED SCRIPT (grouped by concurrent batch) ===")
    print(record.render_as_code())
