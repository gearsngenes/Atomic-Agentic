# math_planner_host_server.py
"""
Math Tool-Agent A2A Host Server
-------------------------------
A tool-using planning agent (PlanActAgent) that uses the Math plugin tools.

Run:
  python math_planner_host_server.py
Then test:
  python a2a_proxy_client.py math
"""

from __future__ import annotations

from dotenv import load_dotenv

from atomic_agentic.agents.toolagents import PlanActAgent, A2AgentHost
from atomic_agentic.engines.LLMEngines import OpenAIEngine
from atomic_agentic.tools.Plugins import MATH_TOOLS

load_dotenv()


def main() -> None:
    llm = OpenAIEngine(model="gpt-4o-mini")

    seed = PlanActAgent(
        name="MathPlannerAgent",
        description="Planner that solves problems by calling math tools.",
        llm_engine=llm,
        context_enabled=False,
        run_concurrent=False,
        tool_calls_limit=12,
    )

    # Register only plugin tools (math)
    seed.batch_register(MATH_TOOLS, name_collision_mode="raise")

    host = A2AgentHost(seed_agent=seed, host="localhost", port=7000, version="1.0.0")
    host.run(debug=True)


if __name__ == "__main__":
    main()
