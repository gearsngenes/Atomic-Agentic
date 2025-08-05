import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.Agents import A2AProxyAgent
from modules.PlannerAgents import PlannerAgent
from modules.LLMEngines import OpenAIEngine

echo = A2AProxyAgent("http://localhost:6000")
shakespeare = A2AProxyAgent("http://localhost:5000")

planner = PlannerAgent("A2APlanner",
                       "Creates plans on calling A2A proxy agents",
                       llm_engine=OpenAIEngine("gpt-4o-mini"),
                       is_async=True,
                       allow_agentic=True)

planner.register(echo)
planner.register(shakespeare)

task = """
Perform the following tasks:
1)  Ask the echo agent to echo back a capital version of "this sentence is capitalized".
2)  Ask the shakespeare agent to return a sonnet about bluebirds
3)  Return the results of both agents' outputs in the format:
    Echo Agent Result: <echo result here>
    Shakespeare Agent Result: <shakespeare result here>
"""
result = planner.invoke(task)
print(result)