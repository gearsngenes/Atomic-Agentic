import sys, logging, json
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
    name = "Paleontologist",
    description = "An expert in paleontology",
    role_prompt = "You are a paleontology expert. Answer the user's questions about dinosaurs with detailed and accurate information.",
    llm_engine = LLM,
    context_enabled=True
)
agent2 = PlannerAgent(
    name = "Mathematician",
    description = "A mathematician who excels in calculations",
    llm_engine = LLM,
    context_enabled=True,
    is_async=True
)
agent2.register(MathPlugin)
agent3 = Agent(
    name = "Jokester",
    description = "Tells jokes and acts as a general assistant with a quasi-rebellious streak",
    role_prompt="You are a jokester who talks in a comedic, non-serious manner. You never take what the user says seriously and always respond with a witty comeback, dodging questions and the like.",
    llm_engine=LLM,
)


def manual_delegate(task1, task2, task3):
    return {
        "Paleontologist":task1,
        "Mathematician":task2,
        "Jokester":task3,
    }
manual_delegator = Tool("ManualDelegate", manual_delegate)

delegator_component = manual_delegator#LLM#


workflow = Delegator(
    name = "ParallelWorkflowExample",
    description = "A workflow that runs two agents in parallel to answer user questions",
    task_master=delegator_component,
    branches=[agent1, agent2, agent3]
)

agentic_input = """
Can you tell me about T-Rex,
calculate 256 to the power of one half,
and tell me a good vacation spot"""

manual_input = ("Tell me about the Tyrannosaurus rex",
                "Calculate the square root of 256",
                "What is a dog's favorite snack?")

task = manual_input

if isinstance(task, tuple):
    print(json.dumps(workflow.invoke(*task), indent = 2))
else:
    print(json.dumps(workflow.invoke(task), indent = 2))