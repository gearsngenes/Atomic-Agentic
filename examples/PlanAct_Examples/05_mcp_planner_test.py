"""05_mcp_planner_test.py

PlanActAgent planning against a local MCP server.

Expected local MCP endpoint:
  http://localhost:8000/mcp

Updated to use PlanActAgent (formerly PlannerAgent).
"""
from dotenv import load_dotenv

from atomic_agentic.agents import PlanActAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine

load_dotenv()

llm_engine = OpenAIEngine(model="gpt-4o-mini")

planner = PlanActAgent(
    name="MCP_Agent",
    description="Creates plans utilizing our sample MCP server",
    llm_engine=llm_engine,
    run_concurrent=False,
)

# Register MCP endpoint (bulk discover) under a clear namespace.
planner.register("http://localhost:8000/mcp", namespace="CalculusServer")

result = planner.invoke(
    {
        "prompt": (
            "Give me the derivative of the function: 'x**5 + 1' at the point x = 2. "
            "Then, multiply by 10, and return the output."
        )
    }
)

print(result, type(result))
