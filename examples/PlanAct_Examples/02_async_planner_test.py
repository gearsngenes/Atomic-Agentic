import logging
import time
from dotenv import load_dotenv

from atomic_agentic.agents import PlanActAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

llm_engine = OpenAIEngine(model="gpt-4o-mini")


def testDelayPrint() -> None:
    print("Called Print, now waiting 10 seconds...")
    time.sleep(10)


async_tester = PlanActAgent(
    name="Async_Delay_Tester",
    description="Tests the ability to run independent steps concurrently.",
    llm_engine=llm_engine,
    context_enabled=False,
)

# Register the callable (capture the fully-qualified tool id)
tool_id = async_tester.register(
    testDelayPrint,
    name="DelayPrint",
    description="delay for 10 seconds.",
    namespace="local",
)[0]

if __name__ == "__main__":
    sequentially = True
    prompt = (
        f"Call 'DelayPrint' EXACTLY FIVE TIMES, but call them {"SEQUENTIALLY" if sequentially else "CONCURRENTLY"}.\n"
    )

    start = time.time()
    async_tester.invoke({"prompt": prompt})
    end = time.time()

    print("Time taken:", end - start, "seconds")
