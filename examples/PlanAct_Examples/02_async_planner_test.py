import logging
import time
from dotenv import load_dotenv

from atomic_agentic.agents.toolagents import PlanActAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()
logging.basicConfig(level=logging.INFO)

llm_engine = OpenAIEngine(model="gpt-4o-mini")


def testDelayPrint() -> None:
    print("Called Print, now waiting 10 seconds...")
    time.sleep(10)


async_tester = PlanActAgent(
    name="Async-Delay-Tester",
    description="Tests the ability to run independent steps concurrently.",
    llm_engine=llm_engine,
    run_concurrent=True,     # when True: executes all ready/independent steps simultaneously
    context_enabled=False,
    tool_calls_limit=5,      # doesn't count the final return step
)

# Register the callable (capture the fully-qualified tool id)
tool_id = async_tester.register(
    testDelayPrint,
    name="testDelayPrint",
    description="Print, then delay for 10 seconds.",
    namespace="local",
)[0]

if __name__ == "__main__":
    prompt = (
        f"Call {tool_id} five times.\n"
        "These calls are independent (no dependencies).\n"
        "Return None."
    )

    start = time.time()
    async_tester.invoke({"prompt": prompt})
    end = time.time()

    print("Time taken:", end - start, "seconds")
