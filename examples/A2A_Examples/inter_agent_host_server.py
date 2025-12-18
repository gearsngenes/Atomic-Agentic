# inter_agent_planner_host_server.py
"""
Inter-Agent Communicator A2A Host Server
---------------------------------------
A tool-using planning agent that communicates with OTHER agents via A2AgentTool
proxies (remote A2A tools).

This registers:
  - TriviaAgent (http://localhost:6000) as a tool
  - MathPlannerAgent (http://localhost:7000) as a tool

Run (after starting the two servers above):
  python trivia_host_server.py
  python math_planner_host_server.py
  python inter_agent_planner_host_server.py

Then test:
  python a2a_proxy_client.py inter
"""

from dotenv import load_dotenv

from atomic_agentic.Agents import PlanActAgent, A2AgentHost
from atomic_agentic.LLMEngines import OpenAIEngine
from atomic_agentic.Tools import A2AgentTool

load_dotenv()


def main() -> None:
    llm = OpenAIEngine(model="gpt-4o-mini")

    seed = PlanActAgent(
        name="InterAgentPlanner",
        description="Planner that orchestrates other agents via A2A proxy tools.",
        llm_engine=llm,
        context_enabled=False,
        run_concurrent=True,
        tool_calls_limit=16,
    )

    # Register remote agents as tools (A2AgentTool is already a Tool instance).
    trivia_tool = A2AgentTool(url="http://localhost:6000")
    math_tool = A2AgentTool(url="http://localhost:7000")

    # Preserve each tool’s remote namespace (agent-card name) and description.
    seed.register(
        trivia_tool,
        namespace=trivia_tool.namespace,
        name_collision_mode="replace",
    )
    seed.register(
        math_tool,
        namespace=math_tool.namespace,
        name_collision_mode="replace",
    )

    host = A2AgentHost(seed_agent=seed, host="localhost", port=8000, version="1.0.0")
    host.run(debug=True)


if __name__ == "__main__":
    main()
