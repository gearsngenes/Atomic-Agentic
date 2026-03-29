from __future__ import annotations

import logging

from dotenv import load_dotenv

from atomic_agentic.a2a.PyA2AtomicClient import PyA2AtomicClient
from atomic_agentic.a2a.PyA2AtomicHost import PyA2AtomicHost
from atomic_agentic.agents.tool_agents import PlanActAgent
from atomic_agentic.engines.LLMEngines import OpenAIEngine
from atomic_agentic.tools.Toolify import toolify

load_dotenv()

TRIVIA_URL = "http://localhost:6000"
TRIVIA_REMOTE_NAME = "TriviaAgent"

MATH_URL = "http://localhost:7000"
MATH_REMOTE_NAME = "MathPlannerAgent"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    llm = OpenAIEngine(model="gpt-4o-mini")

    planner = PlanActAgent(
        name="InterAgentPlanner",
        description="A planner that orchestrates remote trivia and math agents through PyA2AtomicTool proxies.",
        llm_engine=llm,
        context_enabled=False,
        tool_calls_limit=16,
    )

    trivia_client = PyA2AtomicClient(url=TRIVIA_URL)
    math_client = PyA2AtomicClient(url=MATH_URL)

    planner.register(trivia_client,
                      remote_name=TRIVIA_REMOTE_NAME,
                      name_collision_mode="raise")
    planner.register(math_client,
                     remote_name=MATH_REMOTE_NAME,
                     name_collision_mode="raise")

    host = PyA2AtomicHost(
        invokables=[planner],
        name="inter_agent_host",
        description="PyA2AtomicHost exposing the InterAgentPlanner invokable.",
        version="1.0.0",
        host="localhost",
        port=8000,
    )
    host.run_server(debug=True)


if __name__ == "__main__":
    main()
