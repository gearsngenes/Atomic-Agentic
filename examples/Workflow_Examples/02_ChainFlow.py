# 02_ChainFlow.py — minimal, direct ChainFlow examples (dict-in → dict-out)
import logging, json
from atomic_agentic.Agents import Agent
from atomic_agentic.LLMEngines import OpenAIEngine
from atomic_agentic.Workflows import ChainFlow
from atomic_agentic.Tools import Tool
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level = logging.INFO)#logging.DEBUG)

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
tool1 = Tool(func=json_parse, name = "parser")

def separate(output: dict):
    return output.keys(), sum(list(output.values()))
tool2 = Tool(func=separate, name="separator")

def format_out(_keys, _sum):
    return f"The keys of our dicitonary are {_keys}, and the sum of their values is is {_sum}"
tool3 = Tool(func=format_out, name="formatter")

agentic_chain = True # change to switch between the agent and the tool steps

workflow = ChainFlow(
    name = "ChainFlow_Example",
    description ="A chain of thought workflow with three {obj_type}.".format(
        obj_type = "agents" if agentic_chain else "tools"),
    steps = [agent1, agent2, agent3] if agentic_chain else [tool1,tool2,tool3],
    output_schema=["chain_result"]
)

agentic_task = {"prompt":"There are 5 sheep, and twenty-three ox and zero point five chicken eggs."}
tool_task = {"string":'{"a":-1.74,"b":7,"c":4}'}

task = agentic_task if agentic_chain else tool_task
print(workflow.invoke(task))
