import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from modules.A2Agents import A2AServerAgent
from modules.Agents import Agent
from modules.LLMEngines import OpenAIEngine

llm = OpenAIEngine(model = "gpt-4o-mini")
seed = Agent(
    name = "TriviaAgent",
    llm_engine = llm,
    role_prompt = "An agent that accepts an animal name and returns a single fun fact about that animal. If no animal is provided, then default to giving a fun fact about dogs.",
    description= "Gives fun facts about animals.",
    context_enabled=False)

if __name__ == "__main__":
    A2AServerAgent(seed, host = "localhost", port = 6000).run()
