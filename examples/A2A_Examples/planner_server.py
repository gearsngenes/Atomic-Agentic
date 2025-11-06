import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from modules.A2Agents import A2AProxyAgent, A2AServerAgent
from modules.ToolAgents import PlannerAgent
from modules.LLMEngines import OpenAIEngine

def main():
    """
    Planner A2A server
    - The seed is a PlannerAgent that orchestrates calls to registered tools.
    - We register two remote A2A tools (echo/trivia at :6000, shakespeare at :5000),
      both contacted through A2AProxyAgent using the message-level function-call path.
    """
    # Proxies to remote A2A servers
    # (Make sure shakespearean_server.py is on :5000 and trivia_server.py is on :6000)
    echo = A2AProxyAgent(url="http://127.0.0.1:6000", name="TriviaProxy", description="Generates trivia facts")
    shakespeare = A2AProxyAgent(url="http://127.0.0.1:5000", name="ShakespeareProxy", description="Talks in Shakespearean English")

    seed = PlannerAgent(
        name="A2APlanner",
        description="Creates plans that call remote A2A agents via proxy tools.",
        llm_engine=OpenAIEngine(model="gpt-4o-mini"),
        run_concurrent=True,
        context_enabled=False,
    )

    # Register remote A2A proxies as tools (toolify wraps Agent into an AgentTool)
    seed.register(echo)
    seed.register(shakespeare)

    server = A2AServerAgent(
        seed=seed,
        name="Planner Agent",
        description="Planner that orchestrates A2A tools",
        host="127.0.0.1",
        port=7000,
        version="1.0.0",
    )
    server.run(debug=True)

if __name__ == "__main__":
    main()
