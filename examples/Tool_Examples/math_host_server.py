from __future__ import annotations

import logging

from dotenv import load_dotenv

from atomic_agentic.a2a import PyA2AtomicHost
from atomic_agentic.agents import PlanActAgent
from atomic_agentic.llm import OpenAIEngine
from atomic_agentic.tools.prebuilt import MATH_TOOLS

load_dotenv()


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    llm = OpenAIEngine(model="gpt-4o-mini")

    math_agent = PlanActAgent(
        name="MathPlannerAgent",
        namespace="math_host",
        description="A tool-using planner that solves problems with the local math tools.",
        llm_engine=llm,
        context_enabled=False,
        tool_calls_limit=12,
    )
    math_agent.batch_register(MATH_TOOLS, name_collision_mode="raise")

    host = PyA2AtomicHost(
        invokables=[math_agent],
        name="math_host",
        description="PyA2AtomicHost exposing the MathPlannerAgent invokable.",
        version="1.0.0",
        host="localhost",
        port=7000,
    )
    host.run_server(debug=True)


if __name__ == "__main__":
    main()
