import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.ToolAgents import PlannerAgent
from modules.LLMEngines import OpenAIEngine

# logging.getLogger().setLevel(level=logging.INFO)

llm_engine = OpenAIEngine(model="gpt-4o-mini")

planner = PlannerAgent(
    "MCP Agent",
    description="Creates plans utilizing our sample MCP server",
    llm_engine=llm_engine
)

# Register MCP endpoint (use server_name=...)
planner.register("http://localhost:8000/mcp", server_name="CalculusServer")

result = planner.invoke({
    "prompt": (
        "Give me the derivative of the function: 'x**5 + 1' at the point x = 2. "
        "Then, multiply by 10, and return the output."
    )
})
print(result, type(result))
