import logging
import time
from dotenv import load_dotenv

from atomic_agentic.agents import PlanActAgent
from atomic_agentic.llm import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

llm_engine = OpenAIEngine(model="gpt-4o-mini")


def testDelayPrint() -> None:
    print("Called Print, now waiting 3 seconds...")
    time.sleep(3)


async_tester = PlanActAgent(
    name="Async_Delay_Tester",
    namespace="examples",
    description="Tests the ability to run independent steps concurrently.",
    llm_engine=llm_engine,
    context_enabled=False,
)

# Register the callable under an explicit alias.
async_tester.register_tool(
    testDelayPrint,
    alias="DelayPrint",
    description="delay for 3 seconds.",
)

if __name__ == "__main__":
    choice = input("Run steps sequentially? (y/n): ").strip().lower()
    async_tester.tool_concurrency_limit = 1 if choice == "y" else None
    prompt = "Call 'DelayPrint' exactly five times"

    start = time.time()
    async_tester.invoke({"prompt": prompt})
    end = time.time()

    print("Time taken:", end - start, "seconds")
