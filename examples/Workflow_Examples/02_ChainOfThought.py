import sys, logging,json
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Workflows import ChainOfThought
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
    keys, total = list(output.keys()), sum(list(output.values()))
    return (keys, total) #{"a":keys, "b": total} #
tool2 = Tool("separator", separate)
def formatter(a:list, b:float|int):
    return f"The value of 'A' is {a}, and the value of 'B' is {b}"
tool3 = Tool("formatter", formatter)

agentic_chain = True
steps = [agent1, agent2, agent3] if agentic_chain else [tool1, tool2, tool3]
workflow = ChainOfThought(
    name="ChainOfThoughtExample",
    description="A chain of thought workflow with three agents.",
    steps=steps
)

task = "There are five sheep, and twenty-three ox and zero point five chicken eggs." if agentic_chain else '{"a":-1.74,"b":7,"c":4}'
print(workflow.invoke(task))
