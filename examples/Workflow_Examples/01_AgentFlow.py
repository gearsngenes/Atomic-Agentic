import sys
from pathlib import Path
# Setting the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine
from modules.Workflows import AgentFlow

my_agent = Agent(
    name="MyAgent",
    description="An example agent that echoes the input prompt.",
    role_prompt="You are a helpful assistant, who ends their responses with random animal fun-facts.",
    llm_engine=OpenAIEngine(model="gpt-4o"),
    context_enabled=True
)
workflow = AgentFlow(agent=my_agent)

# using scalar inputs
print("1)", workflow.invoke("Hello, Agent! What is pi used for?"))
print("===")
# using keyword inputs
print("2)", workflow.invoke(prompt = "What percent of the earth's mass is carbon?"))
print("===")
# using dicts as keywords
print("3)", workflow.invoke(**{"prompt" : "Can you tell me how far the moon is?"}))
print("===")
# using lists as positionals
print("4)", workflow.invoke(*["What is a the meaning of life as a number?"]))
print("===")
# with result schema
print("Now we have a result schema of { 'answer' : ...}")
workflow = AgentFlow(agent=my_agent, result_schema=["answer"])
print("5)", workflow.invoke("How many parameters does model gpt-4o have?"))
