import sys, logging, json
from pathlib import Path
from typing import Any
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.ToolAgents import PlannerAgent
from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Plugins import MathPlugin
from modules.Workflows import ToolWorkflow, ChainOfThought, Delegator, Workflow
from modules.Tools import Tool

logging.getLogger().setLevel(logging.INFO)

# ~~~Define Agentic Workflow~~~
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

def formatOutput(Agent1: str, Agent2: str, Agent3: str):
    output = f"""
Work completed by the Paleontologist:
{Agent1}

Work completed by the Mathematician:
{Agent2}

Work completed by the Jokester:
{Agent3}
"""
    print(output)

task_splitter = Delegator(
    name = "ParallelWorkflowExample",
    description = "A workflow that runs two agents in parallel to answer user questions",
    delegator_engine=LLM,
    branches=[agent1, agent2, agent3]
)
formatter = Tool("Format_Output", formatOutput)

agentic_workflow = ChainOfThought(
    name = "Agentic-Chain",
    description = "Combines Tool Workflow and Delegator Workflow inside a chain workflow",
    steps = [task_splitter, formatter]
)

# ~~~Define Functional Workflow~~~
def json_parse(string: str):
    return json.loads(string)
tool1 = Tool("parser", json_parse)

def separate(output: dict):
    return {"keys":list(output.keys()), "sum": sum(list(output.values()))}
tool2 = Tool("separator", separate)

def printer(keys, sum):
    print(f"The sum of the values are {sum}, and the keys are {keys}")
tool3 = Tool("Printer", printer)

functional_workflow = ChainOfThought(
    name = "Functional-Chain",
    description="Combines Tool workflow with chain of thought workflow",
    steps = [tool1, tool2, tool3]
)

# ~~~Define and Pair Workflow to Tasks~~~
agentic_input = """
                Do the following:
                * Tell me about the Tyrannosaurus rex
                * Calculate the square root of 256
                * What is a good vacation spot for the winter?"""

functional_input = '{"a": 2, "b": 6, "c":-1, "d": 8}'

tasks_and_workflows: dict[str, tuple[str, Workflow]] = {
    "agentic": (agentic_input, agentic_workflow),
    "functional": (functional_input, functional_workflow)
}

# ~~~Run Workflow~~~
pick = "agentic"
task, workflow = tasks_and_workflows[pick]

workflow.invoke(task)