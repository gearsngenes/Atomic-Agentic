import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.A2Agents import A2AProxyAgent, A2AServerAgent
from modules.PlannerAgents import PlannerAgent
from modules.LLMEngines import OpenAIEngine

echo = A2AProxyAgent("http://localhost:6000")
shakespeare = A2AProxyAgent("http://localhost:5000")

seed = PlannerAgent(name = "A2APlanner",
                    description="Creates plans on calling A2A proxy agents",
                    llm_engine=OpenAIEngine("gpt-4o-mini"),
                    is_async=True,
                    allow_agentic=True)

seed.register(echo)
seed.register(shakespeare)

if __name__ == "__main__":
    A2AServerAgent(seed, host = "localhost", port = 7000).run()