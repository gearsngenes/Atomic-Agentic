from atomic_agentic.Agents import Agent
from atomic_agentic.Tools import Tool
from atomic_agentic.LLMEngines import OpenAIEngine
from atomic_agentic.Factory import load_agent, load_tool
import json
from dotenv import load_dotenv
load_dotenv()

def setup(topic: str) -> str:
    return f"""
    Create an outline for the following topic: {topic}.
    """

seed_agent = Agent(
    llm_engine=OpenAIEngine(model="gpt-4", temperature=0.2),
    name="Outliner",
    description="An agent that creates detailed outlines for given topics.",
    context_enabled=False,
    pre_invoke=setup,
    role_prompt="""
    You are an expert outliner. Given an input topic, you will create a detailed outline with sections and subsections."""
)

print("=== Seed Agent Outline ===")
topic = "A story about a boy with the ability to control time."
print("Story outline:", seed_agent.invoke({"topic": topic}))

# Dehydrate the agent to a dict
agent_dict = seed_agent.to_dict()
print("\n=== Dehydrated Agent Dict ===")
print(json.dumps(agent_dict, indent=2))

# Rehydrate the agent from the dict
rehydrated_agent = load_agent(agent_dict)
topic = "A story about a cat who vomits CD's in a world where nobody remembers what CD's are."
print("\n=== Rehydrated Agent Outliner ===")
print("Story outline:", rehydrated_agent.invoke({"topic": topic}))