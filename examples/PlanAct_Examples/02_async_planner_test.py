import logging
import time

from atomic_agentic.agents import PlanActAgent
from atomic_agentic.adapters import to_tool

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared_engine import llm_engine

logging.basicConfig(level=logging.INFO)


def testDelayPrint() -> None:
    print("Called Print, now waiting 3 seconds...")
    time.sleep(3)


async_tester = PlanActAgent(
    name="Async_Delay_Tester",
    description="Tests the ability to run independent steps concurrently.",
    llm_engine=llm_engine,
    record_enabled=False,
)

# Build the tool under an explicit name, then register it.
async_tester.register_tool(
    to_tool(testDelayPrint, name="DelayPrint", description="delay for 3 seconds.")
)

if __name__ == "__main__":
    choice = input("Run steps sequentially? (y/n): ").strip().lower()
    async_tester.tool_concurrency_limit = 1 if choice == "y" else None
    prompt = "Call 'DelayPrint' exactly five times"

    start = time.time()
    async_tester.invoke({"prompt": prompt})
    end = time.time()

    print("Time taken:", end - start, "seconds")
