import sys, logging
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Workflows import SingleAgent

my_agent = Agent(
    name="MyAgent",
    description="An example agent that echoes the input prompt.",
    role_prompt="You are an agent that echoes the input prompt in all capital letters.",
    llm_engine=OpenAIEngine(model="gpt-4o"),
    context_enabled=False
)
workflow = SingleAgent(agent=my_agent)

print(workflow.invoke("Hello, Agent!"))