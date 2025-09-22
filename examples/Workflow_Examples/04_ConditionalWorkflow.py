import sys, logging
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.ToolAgents import PlannerAgent
from modules.Workflows import ConditionalWorkflow
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

decider = Agent(
    name = "Decider",
    description = "Decides which agent is best suited to answer the user's question",
    role_prompt = "You are a selector agent. Based on the user's questions, determine which workflow is best suited for handling the task.",
    llm_engine = LLM,
    context_enabled=True
)

workflow = ConditionalWorkflow(
    name = "ConditionalWorkflowExample",
    description = "A workflow that routes user questions to the appropriate agent based on the topic",
    decider=decider,
    branches=[agent1, agent2]
)

print(workflow.invoke("what is twenty minus three, all to the power of 3.14159?"))