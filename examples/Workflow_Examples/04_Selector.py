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
from modules.Workflows import Selector, ToolFlow
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
format_tool = Tool("AddressFormatter",
                   format_addr,
                   description="formats the building number,  into a valid address")

#~~~Define our potential deciders~~~
def fill_args(a,b = None,c = None, d = None, e = None):
    if b:
        return format_tool.name
    for i in range(10):
        if str(i) in a:
            return agent2.name
    return agent1.name
    
filter_tool = Tool("arg-filler", fill_args)

decider = filter_tool# LLM #

#~~~~Build our selector workflow~~~
workflow = Selector(
    name = "ConditionalWorkflowExample",
    description = "A workflow that routes user questions to the appropriate agent based on the topic",
    decider=decider,
    branches=[agent1, agent2, format_tool],
    result_schema=["selected_output"]
)

#~~~Define our tasks that we'd want specific agents to handle~~~
input1 = "Difference between tyranosaurs and dromaeosaurs?"
input2 = "what is twenty minus three, all to the power of 3.14159?"
input3 = "123", "Fort Hamilton", "New York", "NY", 10364
#~~~Select the task~~~
task = input2

#~~~print result~~~
if isinstance(task, tuple):
    print(workflow.invoke(*task))
else:
    print(workflow.invoke(task))