import sys, logging
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import *
from modules.PlannerAgents import McpoPlannerAgent
from modules.Plugins import *

logging.basicConfig(level=logging.INFO)

# define a global llm_engine to give to each of our agents
llm_engine = OpenAIEngine(model = "gpt-4o-mini")

planner = McpoPlannerAgent("MCPO Agent", llm_engine)
planner.register("http://localhost:8000") # our MCP Math server
planner.register(ConsolePlugin())

result = planner.invoke(
    "Use the capabilities of our MCP's math server to perform the following task:\n"
    "Give me the derivative of the function: '2.718281**x - 3*x' at the point x = 2."
    "Once you've calculated the derivative, print it out (formatted as MCPO RESULT -- <result here>), and return it"
)
print("FINAL RESULT: ", result)