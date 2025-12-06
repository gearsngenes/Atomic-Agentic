from dotenv import load_dotenv
from atomic_agentic.ToolAgents import PlannerAgent
from atomic_agentic.LLMEngines import OpenAIEngine

load_dotenv()

# logging.basicConfig(level=logging.INFO)

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
