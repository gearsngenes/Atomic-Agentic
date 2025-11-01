# 02_ChainFlow.py — minimal, direct ChainFlow examples (dict-in → dict-out)
import sys, logging, json
from pathlib import Path

# Repo root on path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Workflows import ChainFlow
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

def format_out(_keys, _sum):
    return f"The keys of our dicitonary are {_keys}, and the sum of their values is is {_sum}"
tool3 = Tool("formatter", format_out)

agentic_chain = False # change to switch between the agent and the tool steps

workflow = ChainFlow(
    name = "ChainFlow_Example",
    description ="A chain of thought workflow with three {obj_type}.".format(
        obj_type = "agents" if agentic_chain else "tools"),
    steps = [agent1, agent2, agent3] if agentic_chain else [tool1,tool2,tool3],
    output_schema=["chain_result"]
)

task = {
    "prompt":"There are 5 sheep, and twenty-three ox and zero point five chicken eggs."
    } if agentic_chain else {"string":'{"a":-1.74,"b":7,"c":4}'}
print(workflow.invoke(task))
