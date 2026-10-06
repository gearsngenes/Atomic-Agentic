"""05_mcp_planner_test.py

PlanActAgent planning against a local MCP server.

Expected local MCP endpoint:
  http://localhost:8000/mcp

Updated to use PlanActAgent (formerly PlannerAgent).
"""
from atomic_agentic.agents import PlanActAgent
from atomic_agentic.mcp import MCPClientHub

from shared_engine import llm_engine

planner = PlanActAgent(
    name="MCP_Agent",
    namespace="examples",
    description="Creates plans utilizing our sample MCP server",
    llm_engine=llm_engine,
)

# Register all tools from MCP server (bulk discover via a hub entry in the list).
planner.register_tools(
    [MCPClientHub("streamable_http", persistent=False, endpoint="http://localhost:8000/mcp")]
)

result = planner.invoke(
    {
        "prompt": (
            "Give me the derivative of the function: 'x**5 + 1' at the point x = 2. "
            "Then, multiply by 10, and return the output."
        )
    }
)

print(result.result, type(result.result))
