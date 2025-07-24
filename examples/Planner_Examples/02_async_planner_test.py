import sys, os, logging, time, json
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# ----------------- Atomic Agents ----------------
from modules.Agents import *
from modules.PlannerAgents import PlannerAgent

# ----------------- Setup Logging ----------------
logging.basicConfig(level=logging.INFO)

# define a global llm engine to give to each of our agents
llm_engine = OpenAIEngine(model = "gpt-4o-mini")

def testDelayPrint():
    print("Called Print, now waiting 10 seconds...")
    time.sleep(10)

# Create a PlannerAgent for testing our delay print tool
async_tester = PlannerAgent(
    name        = "Async-Delay-Tester",
    llm_engine  = llm_engine,
    is_async    = True,   # Toggle between async and sync planning
)


# Register the test delay print method as a tool
async_tester.register(tool=testDelayPrint,
                      description = "This method gives a test print call, then delays for 10s.")

if __name__ == "__main__":
    
    # Compose the prompt for the planner agent: calling the delayed print statement five times
    prompt = "Run the 'testDelayPrint' method five times."

    # Invoke the planner agent and time it. In theory, the methods should all start around the same time.
    start = time.time()
    result = async_tester.invoke(prompt)
    end = time.time()
    # Should be around 10-15 seconds to complete if async, closer to 50s to a minute if sync
    print("Time taken:", end - start, "seconds")