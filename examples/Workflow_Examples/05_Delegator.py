import sys, logging
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.ToolAgents import PlannerAgent
from modules.Workflows import Delegator
from modules.Tools import Tool
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


def manual_delegate(task1, task2, task3):
    return {
        "Agent1":task1,
        "Agent2":task2,
        "Agent3":task3,
    }
manual_delegator = Tool("ManualDelegate", manual_delegate)

delegator_component = LLM#manual_delegator#


workflow = Delegator(
    name = "ParallelWorkflowExample",
    description = "A workflow that runs two agents in parallel to answer user questions",
    delegator_component=delegator_component,
    branches=[agent1, agent2, agent3]
)

agentic_input = """Can you tell me about T-Rex, calculate 256, and tell me a good vacation spot"""

manual_input = "Tell me about the Tyrannosaurus rex","Calculate the square root of 256","What is a good vacation spot for the winter?"

print(workflow.invoke(agentic_input)) # add or edit these three lines to test behavior