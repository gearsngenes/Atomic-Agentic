import sys, logging,json
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Workflows import ChainOfThought, ToolFlow, AgentFlow
from modules.Tools import Tool

logging.getLogger().setLevel(logging.INFO)

agent1 = Agent(
    name="Agent1",
    description="First agent in the chain",
    role_prompt="You are Agent 1. Your task is to take the input and add 10 to any number you find.",
    llm_engine=OpenAIEngine(model="gpt-4o"),
    context_enabled=False
)
agent2 = Agent(
    name="Agent2",
    description="Second agent in the chain",
    role_prompt="You are Agent 2. Your task is to take the input and multiply any number you find by 2.",
    llm_engine=OpenAIEngine(model="gpt-4o"),
    context_enabled=False
)
agent3 = Agent(
    name="Agent3",
    description="Third agent in the chain",
    role_prompt="You are Agent 3. Your task is to take the input and subtract 5 from any number you find.",
    llm_engine=OpenAIEngine(model="gpt-4o"),
    context_enabled=False
)

def json_parse(string: str):
    return json.loads(string)
tool1 = Tool("parser", json_parse)

def separate(output: dict):
    return output.keys(), sum(list(output.values()))
tool2 = Tool("separator", separate)

def format_out(a, b):
    return f"The value of 'A' is {a}, and the value of 'B' is {b}"
tool3 = Tool("formatter", format_out)

agentic_chain = False # change to switch from the agent chain to the tool chain
steps = [
    AgentFlow(agent1, ["prompt"]),
    AgentFlow(agent2, ["prompt"]),
    AgentFlow(agent3, [])
] if agentic_chain else [
    ToolFlow(tool1, ["output"]),
    ToolFlow(tool2, ["a", "b"]),
    ToolFlow(tool3, [])
]

workflow = ChainOfThought(
    name="ChainOfThoughtExample",
    description="A chain of thought workflow with three agents.",
    steps=steps,
    result_schema= ["final_string"] # change/remove string to alter output format (empty list = raw string)
)

task = "There are 5 sheep, and twenty-three ox and zero point five chicken eggs." if agentic_chain else '{"a":-1.74,"b":7,"c":4}'
print(workflow.invoke(task))
