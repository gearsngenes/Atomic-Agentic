import sys
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import *
from modules.ToolAgents import PlannerAgent
from modules.Plugins import *

# logging.getLogger().setLevel(level=logging.INFO) # Uncomment to see the logging info

# define a global llm_engine to give to each of our agents
llm_engine = OpenAIEngine(model = "gpt-4o-mini")

planner = PlannerAgent("MCPO Agent",
                       description="Creates plans utilizing our sample MCPO server",
                       llm_engine=llm_engine,
                       allow_mcp=True)
planner.register("http://localhost:8000/mcp") # remove /mcp if using mcpo and stdio transport

result = planner.invoke(
    "Give me the derivative of the function: 'x**5 + 1' at the point x = 2."
    "Once you've calculated the derivative, return the formatted string: MCPO RESULT -- <result here>"
)
print(result)