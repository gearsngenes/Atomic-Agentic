import sys, logging
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import *
from modules.PlannerAgents import McpoPlannerAgent
from modules.Plugins import *

# logging.basicConfig(level=logging.INFO) # Uncomment to see the logging details.

# define a global llm_engine to give to each of our agents
llm_engine = OpenAIEngine(model = "gpt-4o-mini")

planner = McpoPlannerAgent("MCPO Agent", llm_engine)
planner.register("http://localhost:8000") # our MCP Calculus server
planner.register(ConsolePlugin())

result = planner.invoke(
    "Give me the derivative of the function: 'x**5 + 1' at the point x = 2."
    "Once you've calculated the derivative, print it out (formatted as MCPO RESULT -- <result here>).\n"
    "Once that is finished, return the message 'STATUS: COMPLETE'"
)
print(result)