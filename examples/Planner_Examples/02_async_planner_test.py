import logging, time
from dotenv import load_dotenv
from atomic_agentic.ToolAgents import PlannerAgent
from atomic_agentic.LLMEngines import OpenAIEngine

load_dotenv()

logging.basicConfig(level=logging.INFO)

llm_engine = OpenAIEngine(model="gpt-4o-mini")

def testDelayPrint() -> None:
    print("Called Print, now waiting 10 seconds...")
    time.sleep(10)

async_tester = PlannerAgent(
    name="Async-Delay-Tester",
    description="Tests the ability to run methods asynchronously",
    llm_engine=llm_engine,
    run_concurrent=True,   # Toggle between async and sync planning
)

# Register the callable (provide name + description for functions)
async_tester.register(
    testDelayPrint,
    name="testDelayPrint",
    description="Print, then delay for 10 seconds."
)

if __name__ == "__main__":
    prompt = "Run the 'testDelayPrint' method five times."
    start = time.time()
    result = async_tester.invoke({"prompt": prompt})
    end = time.time()
    print("Time taken:", end - start, "seconds")
