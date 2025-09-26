import sys, logging
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.ToolAgents import PlannerAgent
from modules.Workflows import Delegator
from modules.Plugins import MathPlugin

logging.getLogger().setLevel(logging.INFO)

LLM = OpenAIEngine(model="gpt-4o")
agent1 = Agent(
    name = "Agent1",
    description = "An expert in paleontology",
    role_prompt = "You are a paleontology expert. Answer the user's questions about dinosaurs with detailed and accurate information.",
    llm_engine = LLM,
    context_enabled=True
)
agent2 = PlannerAgent(
    name = "Agent2",
    description = "A mathematician who excels in calculations",
    llm_engine = LLM,
    context_enabled=True,
    is_async=True
)
agent2.register(MathPlugin)
agent3 = Agent(
    name = "Agent3",
    description = "Tells jokes and acts as a general assistant with a quasi-rebellious streak",
    role_prompt="You are a jokester who talks in a comedic, non-serious manner. You never take what the user says seriously and always respond with a witty comeback, dodging questions and the like.",
    llm_engine=LLM,
)


workflow = Delegator(
    name = "ParallelWorkflowExample",
    description = "A workflow that runs two agents in parallel to answer user questions",
    delegator_component=LLM,
    branches=[agent1, agent2, agent3]
)

print(workflow.invoke("""Do the following:
                      * Tell me about the Tyrannosaurus rex
                      * Calculate the square root of 256
                      * What is a good vacation spot for the winter?""")) # add or edit these three lines to test behavior