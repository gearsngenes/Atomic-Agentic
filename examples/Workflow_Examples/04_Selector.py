import sys, logging
from pathlib import Path
from typing import Any
import random
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.Tools import Tool
from modules.LLMEngines import OpenAIEngine
from modules.ToolAgents import PlannerAgent
from modules.Workflows import Selector
from modules.Plugins import MathPlugin

logging.getLogger().setLevel(logging.INFO)

LLM = OpenAIEngine(model="gpt-4o")

#~~~Define our branches~~~
agent1 = Agent(
    name = "Agent1",
    description = "An paleontologist who excels in answering paleontological questions",
    role_prompt = "You are a paleontology expert. Answer the user's questions about dinosaurs with detailed and accurate information.",
    llm_engine = LLM,
    context_enabled=True
)
agent2 = PlannerAgent(
    name = "Agent2",
    description = "A mathematician who excels in solving math problems and performing calculations",
    llm_engine = LLM,
    context_enabled=True,
    is_async=True
)
agent2.register(MathPlugin)

def format_addr(bld, str, twn, st, zip):
    return f"Address: {bld} {str}, {twn}, {st} {zip}"
branch_tool = Tool("AddressFormatter",
                   format_addr,
                   description="formats the input into a valid address")

#~~~Define our potential deciders~~~
def check_for_numbers(task):
    select_two = False
    for i in range(10):
        select_two = select_two or (str(i) in task)
    select_one = False
    if "dinosaur" in task or "fossil" in task or "tyranosaur" in task or "paleo" in task:
        select_one = True
    return agent1.name if select_one else (agent2.name if select_two else branch_tool.name)
filter_tool = Tool("number-filter", check_for_numbers)

decider = filter_tool# LLM #

#~~~~Build our selector workflow~~~
workflow = Selector(
    name = "ConditionalWorkflowExample",
    description = "A workflow that routes user questions to the appropriate agent based on the topic",
    decider=decider,
    branches=[agent1, agent2, branch_tool]
)

#~~~Define our tasks that we'd want specific agents to handle~~~
agent1_input = "Difference between tyranosaurs and dromaeosaurs?"
agent2_input = "what is twenty minus three, all to the power of 3.14159?"
agent3_input = "123", "Fort Hamilton", "New York", "NY", 10364
#~~~Select the task~~~
task = agent3_input

#~~~print result~~~
print(workflow.invoke(task))